import cupy as cp
import numpy as np
import logging

import config
from unlearning.unlearning_methods.GraphRevoker.lib_partition.partition import Partition
from unlearning.unlearning_methods.GraphRevoker.lib_partition.node_embedding import NodeEmbedding
# from lib_utils import utils

from torch import nn, optim
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from tqdm import tqdm

class DistMatchLoss(nn.Module):
    def __init__(self, labels, n_classes, n_shards):
        super(DistMatchLoss, self).__init__()
        
        self.one_hot = F.embedding(input=labels, weight=torch.eye(n_classes).to(labels.device))
        self.target_dist = torch.sum(self.one_hot, dim=0)
        self.target_dist = self.target_dist / torch.sum(self.target_dist)
        self.target_dist = torch.repeat_interleave(self.target_dist.unsqueeze(0), n_shards, 0)

        self.one_hot = nn.Parameter(self.one_hot, requires_grad=False)
        self.target_dist = nn.Parameter(self.target_dist, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, partition):
        shard_samples = partition.T @ self.one_hot
        shard_dists = shard_samples / torch.sum(shard_samples, dim=1, keepdim=True)

        return self.kl_div(torch.log(shard_dists), self.target_dist)

class LabelEntropyLoss(nn.Module):
    def __init__(self, labels, n_classes):
        super(LabelEntropyLoss, self).__init__()
        self.one_hot = F.embedding(input=labels, weight=torch.eye(n_classes).to(labels.device))
        self.one_hot = nn.Parameter(self.one_hot, requires_grad=False)

    def forward(self, partition):
        shard_samples = partition.T @ self.one_hot + 1e-5 # Avoid division by zero
        shard_dists = (shard_samples) / torch.sum(shard_samples, dim=1, keepdim=True)
        shard_entropy = torch.sum(torch.mean(-shard_dists * torch.log(shard_dists), dim=1))

        return shard_entropy

class Partitioner(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_shards):
        super(Partitioner, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.0),
            nn.GELU(),
            nn.Linear(hidden_dim, num_shards),
        )
    
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            if i + 1 < len(self.convs):
                x = F.gelu(x)
        x = F.gelu(x)
        x = self.cls(x)
        x = F.softmax(x / 1, dim=1) # Cora & Citeseer
        return x

class PartitionGPA(Partition):
    def __init__(self, args, graph, dataset,logger,model_zoo):
        super(PartitionGPA, self).__init__(args, graph, dataset)
        self.logger = logger
        self.model_zoo = model_zoo
        cp.cuda.Device(self.args['cuda']).use()
        self.load_embeddings()

    def load_embeddings(self):
        node_embedding = NodeEmbedding(self.args, self.logger,self.graph, self.dataset,self.model_zoo)

        self.node_to_embedding = node_embedding.encoder(256, 2)
    
    def partition(self):
        self.logger.info("Training the partition network")

        embedding = np.array(list(self.node_to_embedding.values()))

        device = torch.device(self.args['cuda'])

        train_indices = torch.nonzero(self.dataset.train_mask.cpu()).squeeze(1).detach().cpu().numpy()
        val_indices = torch.nonzero(self.dataset.val_mask).squeeze(1).detach().cpu().numpy()
        # edge_index = self.dataset.edge_index_train.numpy()
        edge_index = self.dataset.edge_index.detach().cpu().numpy()
        val_edge_mask = np.logical_or(np.isin(edge_index[0], val_indices),
                                      np.isin(edge_index[1], val_indices))
        train_only_edge_mask = np.logical_not(val_edge_mask)
        edge_index = edge_index[:, train_only_edge_mask]
        edge_index = np.searchsorted(train_indices, edge_index)
        edge_index = torch.LongTensor(edge_index).to(device)
        
        x = torch.from_numpy(embedding).to(device)
        y = self.dataset.y[self.dataset.train_mask.cpu()].to(device)
        n_classes = len(np.unique(self.dataset.y.cpu().numpy()))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.n_id = torch.arange(data.num_nodes).to(device)

        #loader = NeighborLoader(data, num_neighbors=[10, 7], input_nodes=None, batch_size=512, shuffle=True, drop_last=True) # Flickr
        loader = NeighborLoader(data, num_neighbors=[-1, -1], input_nodes=None, batch_size=512, shuffle=True) # Cora & Citeseer

        model = Partitioner(x.shape[1], 256, self.num_shards).to(device)
        model.train()
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        #for epoch in range(30): # Flickr
        for epoch in range(10): # Cora & Citeseer
            self.logger.info('epoch %s' % (epoch,))
            for it, batch in enumerate(loader):
                optimizer.zero_grad()
                adj = to_dense_adj(batch['edge_index'])[0]
                output = model(batch['x'], batch['edge_index'])
                #output = output * 0.9 + 0.1 / self.num_shards # Flickr 
                #balance_loss = balance_loss(output, output.shape[0])
                balance_loss = eff_norm(output, adj, batch['edge_index'].shape[1])
                semloss_criterion = LabelEntropyLoss(y[batch['n_id']], n_classes)
                cut_loss = ncut_loss(output, adj)
                sem_loss = semloss_criterion(output)
                loss = cut_loss + sem_loss * 1e-3 + balance_loss * 0.001 #+ sem_loss * 1e-3 #+ sem_loss * 1e-3 # Cora & Citeseer & Flickr
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                if it % 100 == 0:
                    self.logger.info('loss: %.4f, balance loss: %.4f, cut_loss: %.4f, sem_loss: %.4f' % \
                        (loss.item(), balance_loss.item(), cut_loss.item(), sem_loss.item()))
        
        self.logger.info("Inferencing")
        model.eval()
        loader2 = NeighborLoader(data, num_neighbors=[-1], input_nodes=None, batch_size=data.num_nodes, shuffle=False) # Cora & Citeseer & Flickr
        results = []
        # Top-k available
        k = 1
        with torch.no_grad():
            for batch in tqdm(loader2):
                x, edge_index, batch_size = batch['x'], batch['edge_index'], batch['batch_size']
                out = torch.argsort(model(x, edge_index)[:batch_size, :], dim=1, descending=True)[:, :k]
                results.append(out.cpu())
        #input(results[0].flatten()[:200])
        self.logger.info("Postprocessing")
        results = torch.concat(results, dim=0).numpy()
        node_cnt = [0 for i in range(self.num_shards)]
        node_idx2com = []
        
        # Aggregate nodes that are assigned to a same shard together
        for node_idx in sorted([i for i in range(results.shape[0])], key=lambda i:results[i][0]):
            res = results[node_idx]
            final_label = -1
            for label in res:
                if node_cnt[label] < x.shape[0] / self.num_shards + x.shape[0] * self.args['shard_size_delta']:
                    final_label = label
                    break
            if final_label == -1:
                final_label = np.argmin(node_cnt)
            node_idx2com.append((node_idx, final_label))
            node_cnt[final_label] += 1
        node_to_community = {}
        nodes = list(self.node_to_embedding.keys())
        for node_idx, final_label in node_idx2com:
            node_to_community[nodes[node_idx]] = final_label
        '''
        for it, batch in enumerate(loader2):
            # Top-k available
            k = 1
            results = torch.argsort(model(batch['x'], batch['edge_index'])[:128, :], dim=1, descending=True)[:, :k]
            results = results.detach().cpu().numpy()
            for res in results:
                final_label = -1
                for label in res:
                    if node_cnt[label] < x.shape[0] / self.num_shards + x.shape[0] * self.args['shard_size_delta']:
                        final_label = label
                        break
                if final_label == -1:
                    final_label = np.argmin(node_cnt)
                cluster_labels.append(final_label)
                node_cnt[final_label] += 1

        node_to_community = {}
        for com, node in zip(cluster_labels, self.node_to_embedding.keys()):
            node_to_community[node] = com
        '''

        community_to_node = {}
        for com in range(len(set(node_to_community.values()))):
            community_to_node[com] = [train_indices[idx] for idx in np.where(np.array(list(node_to_community.values())) == com)[0]]
        community_to_node = dict(sorted(community_to_node.items()))

        return community_to_node
        

def balance_loss(y, n):
    g = y.shape[1]
    return torch.sum((torch.sum(y, dim=0) - n / g) ** 2) / g

def ncut_loss(Y, A):
    D = torch.sum(A, dim=1)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    loss = torch.sum(torch.mm(torch.div(Y.float(), Gamma.t()), (1 - Y).t().float()) * A.float())
    #loss = torch.sum(torch.mm(Y.float(), (1 - Y).t().float()) * A.float())

    return loss

def eff_norm(Y, A, edge_cnt):
    shard_num_nodes = torch.sum(Y, dim=0)
    y = Y.unsqueeze(2) # (N, S, 1)
    shard_edges = torch.einsum('nsc,msc->snm', y, y) # For each shard, (N, 1) matmul (1, N)
    # filter invalid edges, then sum up
    shard_num_edges = torch.sum((A.unsqueeze(0) * shard_edges).view(y.shape[1], -1),  
                                 dim=1)

    return torch.sum(((shard_num_nodes / Y.shape[0]) * (shard_num_nodes / Y.shape[0]) * (shard_num_edges / edge_cnt)) ** (1/3))
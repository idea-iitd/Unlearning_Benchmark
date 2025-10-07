import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import SIGN
from unlearning.unlearning_methods.GraphEraser.aggregation.opt_dataset import OptDataset
from dataset.original_dataset import original_dataset
from utils import utils
from tqdm import tqdm
from utils.dataset_utils import *
from utils import dataset_utils,utils

class OptimalEdgeAggregator:
    def __init__(self, run, model_zoo, data, args,logger):
        self.logger = logger
        self.args = args

        self.run = run
        self.model_zoo = model_zoo
        self.target_model = model_zoo.model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_shards = args['num_shards']

    def generate_train_data(self,data):

        temp = Data(x=data.x, y=data.y, edge_index=data.edge_index)

        train_edges = np.array(self.data.train_edge_index.detach().cpu().numpy())
        # sample a set of nodes from train_indices
        if self.args["num_opt_samples"] == 1000:
            num_unlearned_edges = 1000
        elif self.args["num_opt_samples"] == 10000:
            num_unlearned_edges = int(train_edges.shape[1]*0.1)
        elif self.args["num_opt_samples"] == 1:
            num_unlearned_edges = int(train_edges.shape[1])
        
        shuffle_num = torch.randperm(train_edges.shape[1])
        train_edge_index = torch.tensor(train_edges[:, shuffle_num][:, :num_unlearned_edges])

        x = self.data.x
        y = self.data.y

        train_data = Data(x=x, edge_index=data.edge_index, y=y)
        train_data.train_edge_index = train_edge_index
        train_data.test_edge_index = train_edge_index
        pos_edge_labels = torch.ones((1,num_unlearned_edges),dtype=torch.float32)
        neg_edge_labels = torch.zeros((1,num_unlearned_edges),dtype=torch.float32)
        self.true_labels = torch.cat((pos_edge_labels,neg_edge_labels),dim=-1).squeeze(0)
        # print(self.true_labels)
        if self.args["base_model"] == "SIGN":
            train_data = SIGN(self.args["GNN_layer"])(train_data)
            train_data.xs = [train_data.x] + [train_data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
            train_data.xs = torch.tensor([x.detach().numpy() for x in train_data.xs]).cuda()
            train_data.xs = train_data.xs.transpose(0, 1)
        
        
        self.posteriors = {}
        for shard in range(self.num_shards):
            self.model_zoo.data = train_data
            load_target_model(self.logger,self.args,self.run, self.model_zoo, shard)

            self.posteriors[shard] = self.model_zoo.posterior_edge()
            
    def optimization(self):
        weight_para = nn.Parameter(torch.full((self.num_shards,), fill_value=1.0 / self.num_shards), requires_grad=True)
        optimizer = optim.Adam([weight_para], lr=self.args['opt_lr'])
        # scheduler = MultiStepLR(optimizer, milestones=[500, 1000], gamma=self.args['opt_lr'])

        # train_dset = OptDataset(self.posteriors, self.true_labels)
        # train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=0)

        min_loss = 1000.0
        time_sum = 0
        for epoch in tqdm(range(self.args['opt_num_epochs']), desc="Training", unit="epoch"):
            start_time = time.time()
            loss_all = 0.0

            # for posteriors, labels in train_loader:
            # labels = self.true_labels.to(self.device)
            # posteriors = posteriors.to(self.device)
            optimizer.zero_grad()
            loss = self._loss_fn(self.posteriors, self.true_labels, weight_para)
            
            loss.backward()
            loss_all += loss

            optimizer.step()
            with torch.no_grad():
                weight_para[:] = torch.clamp(weight_para, min=0.0,max=1.0)

            # scheduler.step()

            if loss_all < min_loss:
                ret_weight_para = copy.deepcopy(weight_para)
                min_loss = loss_all
            time_sum += time.time()- start_time
            self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss_all))
        self.logger.info('Aggregating Time: {:.4f} seconds per epoch'.format(time_sum / self.args['opt_num_epochs']))

        return ret_weight_para / torch.sum(ret_weight_para)

    def _loss_fn(self, posteriors, labels, weight_para):
        aggregate_posteriors = torch.zeros_like(posteriors[0])
        for shard in range(self.num_shards):
            aggregate_posteriors += weight_para[shard] * posteriors[shard]
        # print(aggregate_posteriors, labels)
        # aggregate_posteriors = F.softmax(aggregate_posteriors, dim=1).to(self.device)
        loss_1 = F.binary_cross_entropy_with_logits(aggregate_posteriors.cpu(), labels.cpu())
        # print(loss_1)
        loss_2 = torch.sqrt(torch.sum(weight_para ** 2))

        return loss_1 + loss_2

    def _generate_train_data_revoker(self):
        data = dataset_utils.load_train_test_split(self.logger)
        # train_indices,_= load_train_test_split(self.logger)
        
        train_indices = data.train_indices
        train_indices = np.array(train_indices)

        com2node = dataset_utils.load_community_data(self.logger)
        node2com = {}
        
        for com, indices in com2node.items():
            for node in indices:
                node2com[node] = com
        node2com = np.array(list(node2com.values()), dtype=np.int64)
        
        # sample a set of nodes from train_indices
        if self.args["num_opt_samples"] == 1000:
            selected_indices = np.random.choice(np.arange(len(train_indices)), size=1000, replace=False)
        elif self.args["num_opt_samples"] == 10000:
            selected_indices = np.random.choice(np.arange(len(train_indices)), size=int(len(train_indices) * 0.1), replace=False)
        elif self.args["num_opt_samples"] == 1:
            selected_indices = np.random.choice(np.arange(len(train_indices)), size=int(len(train_indices)), replace=False)

        train_indices = train_indices[selected_indices]
        self.node2com = node2com[selected_indices]

        train_indices = np.sort(train_indices)
        self.logger.info("Using %s samples for optimization" % (int(train_indices.shape[0])))

        x = self.data.x[train_indices]
        y = self.data.y[train_indices]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices)

        train_data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
        train_data.train_mask = torch.zeros(train_indices.shape[0], dtype=torch.bool)
        train_data.test_mask = torch.ones(train_indices.shape[0], dtype=torch.bool)
        self.true_labels = y

        self.posteriors = {}
        
        return train_data
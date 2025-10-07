import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import embedding, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from utils import dataset_utils
from unlearning.unlearning_methods.GraphRevoker.lib_aggregator.opt_dataset import NegSamplingDataset, OptDataset
from unlearning.unlearning_methods.GraphRevoker.lib_aggregator.optimal_aggregator import OptimalAggregator

from tqdm import tqdm

class ContraHead(nn.Module):
    def __init__(self, proj_dim, hidden_dim, tau=2):
        super(ContraHead, self).__init__()

        self.tau = tau
        #self.w1 = nn.Linear(proj_dim, hidden_dim)
        #self.w2 = nn.Linear(proj_dim, hidden_dim)

    @staticmethod
    def cosine_similarity(p, z):
        # [N, E]

        p = p / p.norm(dim=1, keepdim=True)
        z = z / z.norm(dim=1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1).mean()  # dot product & normalization

    def forward(self, global_pos, global_neg, local_pos, local_neg, beta=1):
        #gp, gn, lp, ln = self.w1(global_pos), self.w1(global_neg), self.w2(local_pos), self.w2(local_neg)
        gp, gn, lp, ln = global_pos, global_neg, local_pos, local_neg

        f = lambda x: torch.exp(x / self.tau)
        x1 = f(self.cosine_similarity(gp, lp))
        x2 = f(self.cosine_similarity(gp, gn))
        x3 = f(self.cosine_similarity(gp, ln))

        return -torch.log(beta * x1 / (beta * x1 + x2 + x3))

class ContraAggr(nn.Module):
    def __init__(self, num_shards, embed_dim, proj_dim, attn_dim, num_classes):
        super(ContraAggr, self).__init__()

        self.w_trans = nn.Parameter(
            torch.normal(mean=0, 
                         std=0.01, 
                         size=(num_shards, embed_dim, proj_dim))
        )
        self.b_trans = nn.Parameter(
            torch.normal(mean=0, 
                         std=0.01, 
                         size=(num_shards, proj_dim))
        )
        self.w = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(proj_dim, attn_dim))
        )
        self.b = nn.Parameter(torch.zeros(attn_dim))
        #self.h = nn.Parameter(torch.zeros(attn_dim, 1) + 1 / attn_dim)
        self.h = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(attn_dim, 1))
        )
        self.cls = nn.Sequential(
                    nn.Linear(proj_dim, num_classes),
                    ##nn.Dropout(0.2), 
                    #nn.GELU(), nn.Linear(128, num_classes)
                )
        nn.init.normal_(self.cls[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.cls[0].bias, mean=0, std=0.01)
        #nn.init.uniform_(self.cls[0].bias, -0.05, 0.05)

        self.contra_head = ContraHead(proj_dim, proj_dim)
        #self.t_loss = nn.TripletMarginLoss()
        
    def compute_recon_loss(self, anchor, pos, neg):
        dist_func = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        res = F.triplet_margin_with_distance_loss(anchor, pos, neg, distance_function=dist_func)

        return res

    def softmax(self, x):
        #x = torch.exp(x)
        #x = x / torch.sum(x, dim=1, keepdim=True)
        
        return torch.softmax(x / 2, 1)
        #return torch.softmax(x, 1)
    
    def single_pass(self, projected, mask=None):
        attn_w = torch.einsum('abc,ck->abk', 
                              F.relu(torch.einsum('abc,ck->abk', projected, self.w) + self.b),
                              self.h)
        if not mask is None:
            attn_w = attn_w.masked_fill_(mask, -float('inf'))
        attn_ws = self.softmax(attn_w)
        #input(attn_ws[0])
        return torch.sum(projected * attn_ws, 1)
    
    def forward(self, x, c_neg=None, t_pos=None, t_neg=None, mask=None, is_eval=False):
        assert (not mask is None) ^ is_eval # Train with softmax masks, evaluate without softmax masks

        proj = lambda t: torch.relu(torch.einsum('nsc,scd->nsd', t, self.w_trans) + self.b_trans)
        x = proj(x)

        x_attn_w = torch.einsum('abc,ck->abk', 
                                F.relu(torch.einsum('abc,ck->abk', x, self.w) + self.b),
                                self.h)
        x_attn_ws = self.softmax(x_attn_w)
        #input(torch.topk( x_attn_ws[:10].squeeze(-1),k=3))
        #input())
        anchor = torch.sum(x * x_attn_ws, 1)
        logits = self.cls(anchor)

        if not is_eval:
            c_neg, t_pos, t_neg = proj(c_neg), proj(t_pos), proj(t_neg)
            # Compute the Contrastive Loss
            global_pos = anchor

            local_pos_ws = self.softmax(x_attn_w.masked_fill_(mask, -float('inf')))
            local_pos = torch.sum(x * local_pos_ws, 1) # Give up using self.single_pass to avoid repeated calculation of x_attn_w

            global_neg = self.single_pass(c_neg)
            local_neg = self.single_pass(c_neg, mask)

            con_loss = self.contra_head(global_pos, global_neg, local_pos, local_neg)

            # Compute the Triplet Loss
            t_pos = self.single_pass(t_pos)
            t_neg = self.single_pass(t_neg)

            #tri_loss = self.t_loss(anchor, t_pos, t_neg)
            tri_loss = self.compute_recon_loss(anchor, t_pos, t_neg)

            return logits, con_loss, tri_loss

        return logits

class ContrastiveAggregator(OptimalAggregator):
    def __init__(self, run, target_model, data, args,logger):
        super(ContrastiveAggregator, self).__init__(run, target_model, data, args,logger)
        self.logger = logger
        self.embeddings = {}
        
    def generate_train_data(self):
        train_data= self._generate_train_data()
        
        for shard in range(self.num_shards):
            # Inference on the selected nodes
            edge_index_tmp = train_data.edge_index
            train_data = T.ToSparseTensor()(train_data)
            if shard == 0:
                self.adj = train_data.adj_t
            train_data.edge_index = edge_index_tmp
            self.target_model.data_full = train_data

            if self.args['is_use_test_batch']:
                self.target_model.gen_test_loader()
            dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard)

            z, f = self.target_model._inference()
            #input((z.shape, f.shape))
            self.posteriors[shard] = z.to(self.device)
            #self.embeddings[shard] = torch.cat([f.to(self.device), self.posteriors[shard]], 1)
            self.embeddings[shard] = f.to(self.device)
        
        self.train_dset = NegSamplingDataset(self.embeddings, self.true_labels, self.adj, self.node2com)
    
    def optimization(self):
        train_dset = self.train_dset
        train_loader = DataLoader(train_dset, batch_size=512, shuffle=True, num_workers=0)
        masks = self._generate_mask(19, train_loader.batch_size).to(self.device)

        model = ContraAggr(self.args['num_shards'], train_dset.posteriors[0].shape[1], 128, 16, len(self.true_labels.unique()))
        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.args['opt_lr'], weight_decay=self.args['opt_decay'])
        #optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
        #optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=0)#self.args['opt_decay'])
        
        scheduler = MultiStepLR(optimizer, milestones=[999], gamma=0.7)

        for epoch in range(self.args['opt_num_epochs']):
            loss_all = 0.0

            for e, c_neg_e, t_pos_e, t_neg_e, labels, s in train_loader:#tqdm(train_loader):
                #masks = self._generate_mask(3, e.shape[0]).to(self.device)
                #input(e)
                e, c_neg_e, t_pos_e = e.to(self.device), c_neg_e.to(self.device), t_pos_e.to(self.device)
                t_neg_e, labels = t_neg_e.to(self.device), labels.to(self.device)
                if e.shape[0] < train_loader.batch_size:
                    logits, c_loss, t_loss = model(e, c_neg_e, t_pos_e, t_neg_e, masks[:e.shape[0]])
                else:
                    logits, c_loss, t_loss = model(e, c_neg_e, t_pos_e, t_neg_e, masks)
                cls_loss = F.cross_entropy(logits, labels)
                #print(s[:10])
                loss = cls_loss + 1e-4 * c_loss + 1e-4 * t_loss # Others
                #loss = cls_loss + 1e-3 * c_loss + 1e-3 * t_loss # JKNet
                optimizer.zero_grad()
                #input(logits.shape)
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()
                loss_all += loss.item()

            scheduler.step()
            
            ret_model = copy.deepcopy(model)#.cpu()

            self.logger.info('epoch: %s, loss: %s' % (epoch, loss_all))

        return ret_model

    def _generate_mask(self, num_local=3, batch_size=512):
        masks = []
        for _ in range(batch_size):
            chosen = np.random.choice(self.args['num_shards'], num_local)
            mask = np.zeros(self.args['num_shards'])
            mask[chosen] = 1
            masks.append(1 - mask)
        
        mask = torch.tensor(np.stack(masks), device=self.device, dtype=torch.bool).unsqueeze(2)
        return mask
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from model.base_gnn.abstract_model import abstract_model,RandomizedClassifier
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch import Tensor
import yaml
from typing import Union
from config import root_path
from parameter_parser import parameter_parser
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from utils.utils import sparse_mx_to_torch_sparse_tensor,normalize_adj
import torch.nn as nn
class GCNNet(abstract_model):
    def __init__(self,args,in_channels, out_channels, num_layers=2):
        super(GCNNet,self).__init__()
        self.args = args
        self.config = self.load_config()
        self.num_layers = num_layers
        hidden_channels = 64
        self.convs = torch.nn.ModuleList()
        self.adj = None
        if self.args['dataset_name'] == "ogbn-products":
            self.convs.append(torch.nn.Linear(in_channels,hidden_channels))
            self.convs.append(torch.nn.Linear(hidden_channels,out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            if self.args["downstream_task"]=="graph":
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.linear = torch.nn.Linear(hidden_channels,out_channels)
            else:
                self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index,return_all_emb=False,return_feature=False,batch=None): 
        if self.args['dataset_name'] != "ogbn-products":
            x_list = []
            x = self.convs[0](x, edge_index)
            x_list.append(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            if self.args["downstream_task"]=="graph":
                x = global_mean_pool(x,batch)
                x = F.relu(x)
                x = F.dropout(x, p=0.5)
                x = self.linear(x)
            x_list.append(x)
            
            if return_all_emb:
                return x_list
            if return_feature:
                return x,x_list[0]
            return x
        else:
            x_list = []
            x = torch.mm(self.adj, x)
            x = self.convs[0](x)
            x_list.append(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = torch.mm(self.adj, x)
            x = self.convs[-1](x)
            if self.args["downstream_task"]=="graph":
                x = global_mean_pool(x,batch)
                x = F.relu(x)
                x = F.dropout(x, p=0.5)
                x = self.linear(x)
            x_list.append(x)
            
            if return_all_emb:
                return x_list
            if return_feature:
                return x,x_list[0]
            return x
    
    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()



    
    def reason_once(self,data):
        x, edge_index = data.x, data.edge_index
        if not x.is_cuda:
            x = x.to('cuda') 
        if not edge_index.is_cuda:
            edge_index = edge_index.to('cuda')
        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)

        return  x
    
    def reason_once_unlearn(self,data):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)

        return  x


    def forward_once(self, data, edge_weight):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.convs[0](x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, data, edge_weight):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = F.relu(self.convs[0](x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)
    
    
        
    
    
        
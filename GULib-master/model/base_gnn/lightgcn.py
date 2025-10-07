import torch
import torch.nn.functional as F
from torch_geometric.nn import LGConv,global_mean_pool
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
class LightGCNNet(abstract_model):
    def __init__(self,args,in_channels, out_channels, num_layers=2):
        super(LightGCNNet,self).__init__()
        self.args = args
        self.config = self.load_config()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.adj = None
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))
        self.linear = torch.nn.Linear(in_channels,out_channels)

    def forward(self, x, edge_index,return_all_emb=False,return_feature=False,batch=None): 
        x_list = []
        x = self.convs[0](x, edge_index)
        x_list.append(x)    
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        x= self.linear(x)
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

    
        
    
    
        
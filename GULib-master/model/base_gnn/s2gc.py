import torch
import yaml
from torch_geometric.nn import SGConv,global_mean_pool
from typing import Union
import torch.nn.functional as F

from model.base_gnn.Convs import S2GConv
from model.base_gnn.abstract_model import abstract_model,RandomizedClassifier
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch import Tensor
from config import root_path
from parameter_parser import parameter_parser

class S2GCNet(abstract_model):
    def __init__(self, args,in_channels, out_channels, num_layers=2):
        super(S2GCNet, self).__init__()
        self.args = args
        self.config = self.load_config()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        hidden_channels = 64
        if self.args["unlearning_methods"] == "SGU" or self.args["unlearning_methods"] == "GNNDelete":
            self.convs.append(torch.nn.Linear(in_channels, 64, bias=False))
            self.convs.append(torch.nn.Linear(64,out_channels,bias=False))
            self.num_layers = 2
        else:
            if self.args["downstream_task"]=="graph":
                self.convs.append(S2GConv(hidden_channels, hidden_channels))
                self.linear = torch.nn.Linear(hidden_channels,out_channels)
            else:
                self.convs.append(S2GConv(in_channels, out_channels, K=self.num_layers, bias=False))
            self.num_layers = 1

    def forward(self, x, edge_index=None,return_feature=False,return_all_emb=False,batch=None):
        x_list = []
        if self.args["unlearning_methods"] == "SGU" or self.args["unlearning_methods"] == "GNNDelete":
            x = self.convs[0](x)
            x_list.append(x)
            x = self.convs[1](x)
            x_list.append(x)
        else:
            x = self.convs[0](x, edge_index)
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
        x = self.convs[0](x,edge_index)
        x = self.convs[1](x)

        return  x
    
    def reason_once_unlearn(self,data):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x,edge_index)
        x = self.convs[1](x)

        return  x

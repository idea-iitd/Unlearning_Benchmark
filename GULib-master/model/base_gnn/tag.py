import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_mean_pool
from model.base_gnn.abstract_model import abstract_model,RandomizedClassifier
from parameter_parser import parameter_parser

class TAGNet(abstract_model):
    def __init__(self,args,in_channels, out_channels, num_layers=2):
        super(TAGNet,self).__init__()
        self.args = args
        self.config = self.load_config()
        self.num_layers = 1
        self.convs = torch.nn.ModuleList()
        self.convs.append(TAGConv(in_channels, out_channels))

    def forward(self, x, edge_index,return_all_emb=False,return_feature=False,batch=None):
        x_list = []
        x = self.convs[0](x, edge_index)
        x_list.append(x)
        if self.args["downstream_task"]=="graph":
            x = global_mean_pool(x,batch)
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
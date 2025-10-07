import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
from model.base_gnn.abstract_model import abstract_model,RandomizedClassifier
from parameter_parser import parameter_parser

class SAGENet(abstract_model):
    def __init__(self,args,in_channels, out_channels, num_layers=2):
        super(SAGENet,self).__init__()
        self.args = args
        self.config = self.load_config()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        hidden_channels = 64
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.args["downstream_task"]=="graph":
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.linear = torch.nn.Linear(hidden_channels,out_channels)
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index,return_all_emb=False,return_feature=False,batch=None):
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
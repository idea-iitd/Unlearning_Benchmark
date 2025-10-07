import torch
import yaml
from torch_geometric.nn import SGConv,global_mean_pool
from typing import Union
import torch.nn.functional as F
from model.base_gnn.abstract_model import abstract_model,RandomizedClassifier
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch import Tensor
from config import root_path
from parameter_parser import parameter_parser
class SGCNet(abstract_model):
    def __init__(self,args, in_channels, out_channels, num_layers=2):
        super(SGCNet, self).__init__()
        self.args = args
        self.config = self.load_config()
        hidden_channels = 64
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if self.args["unlearning_methods"] == "SGU"  or self.args["unlearning_methods"] == "GNNDelete":
            self.convs.append(torch.nn.Linear(in_channels, 64, bias=False))
            self.convs.append(torch.nn.Linear(64,out_channels,bias=False))
            self.num_layers = 2
        else:
            self.num_layers = 1
            if self.args["downstream_task"]=="graph":
                self.convs.append(SGConv(hidden_channels, hidden_channels))
                self.linear = torch.nn.Linear(hidden_channels,out_channels)
            else:
                self.convs.append(SGConv(in_channels, out_channels, K=3, bias=False))
            

    def forward(self, x, edge_index=None,return_feature=False,return_all_emb=False,batch = None):
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


    # for link prediction
    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            # Z1 = z[edge_index[0]]
            # Z2 = z[edge_index[1]]
            # SUM = Z1*Z2
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()


    
    def reason_once(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x,edge_index)
        # x = self.convs[1](x)

        return  x
    
    def reason_once_unlearn(self,data):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x,edge_index)
        # x = self.convs[1](x)

        return  x



    # def GIF_inference(self, x_all, subgraph_loader, edge_weight, device):
    #     for i in range(self.num_layers):
    #         xs = []

    #         for batch_size, n_id, adj in subgraph_loader:
    #             edge_index, e_id, size = adj.to(device)
    #             x = x_all[n_id].to(device)
    #             x_target = x[:size[1]]
    #             x = self.convs_batch[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

    #             xs.append(x.cpu())

    #         x_all = torch.cat(xs, dim=0)

    #     return x_all

    def forward_once(self, data, edge_weight):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index, edge_weight)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, data, edge_weight):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x, edge_index, edge_weight)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)


    # def load_config(self):
    #     f = open(root_path + '/model/properties/sgc' +'.yaml','r')
    #     config_str = f.read()
    #     config = yaml.load(config_str,Loader=yaml.FullLoader)
    #     self.lr = config['lr']
    #     self.decay = config['decay']
    #     return config

# class SGConvBatch(SGConv):
#     def __init__(self, in_channels: int, out_channels: int, K: int = 1,
#                  cached: bool = False, add_self_loops: bool = True,
#                  bias: bool = True, **kwargs):
#         super(SGConvBatch, self).__init__(in_channels, out_channels,
#                                           cached=cached, add_self_loops=add_self_loops,
#                                           bias=bias)

#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_weight: OptTensor = None) -> Tensor:

#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
#         out = self.lin(out)

#         return out

#     def decode(self, z, pos_edge_index, neg_edge_index=None):
#         if neg_edge_index is not None:
#             edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
#             # Z1 = z[edge_index[0]]
#             # Z2 = z[edge_index[1]]
#             # SUM = Z1*Z2
#             logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

#         else:
#             edge_index = pos_edge_index
#             logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

#         return logits

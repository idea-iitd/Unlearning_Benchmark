import torch
import torch.nn.functional as F
from model.base_gnn.gcn import GCNNet
class ClusterNet(GCNNet):
    def __init(self,args,in_channels, out_channels, num_layers=2):
        super(ClusterNet,self).__init__(self,args,in_channels, out_channels, num_layers=2)

import torch
import os
import numpy as np
import yaml
import torch.nn.functional as F
import torch_geometric.transforms as T
from utils import utils
from model.base_gnn.graphsage import SAGENet
from model.base_gnn.gat import GATNet
from model.base_gnn.gcn import GCNNet
from model.base_gnn.gin import GINNet
from model.base_gnn.sgc import SGCNet
from model.base_gnn.s2gc import S2GCNet
from model.base_gnn.sign import SIGNNet
from model.base_gnn.cheb import ChebNet
from model.base_gnn.gcn2 import GCN2Net
from model.base_gnn.lightgcn import LightGCNNet
from model.base_gnn.gatv2 import GATv2Net
from model.base_gnn.tag import TAGNet
from model.base_gnn.appnp import APPNPNet
from model.base_gnn.graphsaint import SAINTNet
from model.base_gnn.cluster_gcn import ClusterNet
from model.base_gnn.gst import *
from model.base_gnn.gukd_teacher import GCNII
# from unlearning.unlearning_methods.GST.gst_main import *
from .base_gnn.deletion import GCNDelete, GATDelete, GINDelete,SAGEDelete,SGCDelete,S2GCDelete,SIGNDelete,SAINTDelete
from model.base_gnn.ceu_model import CEU_GNN
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model.base_gnn.abstract_model import abstract_model
from unlearning.unlearning_methods.Projector.utils.graph_projector_model_utils import Pro_GNN
from torch_geometric.transforms import SIGN
from config import root_path


class model_zoo(abstract_model):
    def __init__(self,args,data):
        super(model_zoo, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.data = data
        self.determine_model()


    def determine_model(self):
        if self.args['unlearning_methods']=="CEU":
            self.model = self.CEU_get_model().to(self.device)
        elif self.args['unlearning_methods']=="Projector":
            self.model = Pro_GNN(self.data.num_features+1, self.data.num_classes+1,self.device,self.args)
        else:
            if self.args["dataset_name"] == "ppi":
                self.model = self.init_model(50,121).to(self.device)
            else:
                self.model = self.init_model(self.data.num_features, self.data.num_classes).to(self.device)
    def init_model(self, num_feats, num_classes):
        if self.args['base_model'] == 'SAGE':
            return SAGENet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'GAT':
            return GATNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'GCN':
            return GCNNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'GIN':
            return GINNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'SGC':
            return SGCNet(self.args, num_feats, num_classes,num_layers=3)
        elif self.args['base_model'] == 'S2GC':
            return S2GCNet( self.args,num_feats, num_classes,num_layers=3)
        elif self.args['base_model'] == 'SAINT':
            self.args["use_batch"] = True
            return SAINTNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'Cluster_GCN':
            self.args["use_batch"] = True
            return ClusterNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'APPNP':
            return APPNPNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'TAG':
            return TAGNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'GCN2':
            return GCN2Net(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'GAT2':
            return GATv2Net(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'LightGCN':
            return LightGCNNet(self.args,num_feats, num_classes)
        elif self.args['base_model'] == 'SIGN':
            self.data = SIGN(self.args["GNN_layer"])(self.data)
            self.data.xs = [self.data.x] + [self.data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
            self.data.xs = torch.stack(self.data.xs).to('cuda')
            self.data.xs = self.data.xs.transpose(0,1)
            return SIGNNet(self.args,num_feats, num_classes, num_layers=3)
        elif self.args['base_model'] == 'GST':
            return GeometricScattering(self.args["J"], self.args["Q"], self.args["L"]).to(self.device)
        else:
            raise Exception('unsupported target models')





    def get_model(self,mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):

        if 'gnndelete' in self.args["unlearning_model"]:
            model_mapping = {'GCN': GCNDelete, 'GAT': GATDelete, 'GIN': GINDelete,'SAGE':SAGEDelete,"SGC":SGCDelete,"S2GC":S2GCDelete,"SIGN":SIGNDelete,"SAINT":SAINTDelete}
            return model_mapping[self.args["base_model"]](self.args,self.data.num_features, self.data.num_classes,
                                                   mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes)
        else:
            model_mapping = {'GCN': GCNNet, 'GAT': GATNet, 'GIN': GINNet,'SAGE':SAGENet,"SGC":SGCNet,"S2GC":S2GCNet,"SIGN":SIGNNet,"SAINT":SAINTNet}
            return model_mapping[self.args["base_model"]](self.args,self.data.num_features, self.data.num_classes)

    def CEU_get_model(self):
        embedding_size = self.args['emb_dim'] if self.data['features'] is None else self.data['features'].shape[1]
        model = CEU_GNN(self.data['num_nodes'], embedding_size,
                        self.args["hidden"], self.data['num_classes'], self.data['features'],
                        self.args["feature_update"], self.args["base_model"])
        return model


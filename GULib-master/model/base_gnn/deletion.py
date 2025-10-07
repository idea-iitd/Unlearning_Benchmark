import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.base_gnn.gcn import GCNNet
from model.base_gnn.gat import GATNet
from model.base_gnn.gin import GINNet
from model.base_gnn.sgc import SGCNet
from model.base_gnn.sign import SIGNNet
from model.base_gnn.graphsage import SAGENet
from model.base_gnn.s2gc import S2GCNet
from model.base_gnn.graphsaint import SAINTNet
from model.base_gnn.abstract_model import abstract_model



class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask.clone()
        
        if mask is not None:
            new_rep = x
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight.clone())

            return new_rep

        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class GCNDelete(GCNNet):
    def __init__(self, args,in_channels, out_channels, num_layers=2, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=2)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        self.convs[0].requires_grad = False
        self.convs[1].requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.convs[0](x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)
    
class SAINTDelete(SAINTNet):
    def __init__(self, args,in_channels, out_channels, num_layers=2, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=2)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        self.convs[0].requires_grad = False
        self.convs[1].requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.convs[0](x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GATDelete(GATNet):
    def __init__(self, args,in_channels, out_channels, num_layers=2, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=2)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        self.convs[0].requires_grad = False
        self.convs[1].requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        
        x1 = self.convs[0](x, edge_index)
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GINDelete(GINNet):
    def __init__(self, args, in_channels, out_channels, num_layers=2, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=2)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        self.convs[0].requires_grad = False
        self.convs[1].requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        x1 = self.convs[0](x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)
    
    
class SAGEDelete(SAGENet):
    def __init__(self, args, in_channels, out_channels, num_layers=2, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=2)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        self.convs[0].requires_grad = False
        self.convs[1].requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        self.convs[0].requires_grad = False
        self.convs[1].requires_grad = False
        x1 = self.convs[0](x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)
    
    
class SGCDelete(SGCNet):
    def __init__(self, args, in_channels, out_channels, num_layers=3, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=3)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        for i in range(self.num_layers):
            self.convs[i].requires_grad = False

    def forward(self, x, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        x1 = self.convs[0](x)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self,x,return_all_emb=False):
        return super().forward(x,return_all_emb=True)

class S2GCDelete(S2GCNet):
    def __init__(self, args, in_channels, out_channels, num_layers=3, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=3)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        for i in range(self.num_layers):
            self.convs[i].requires_grad = False

    def forward(self, x, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        x1 = self.convs[0](x)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.convs[1](x)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self,x,return_all_emb=False):
        return super().forward(x,return_all_emb=True)
    
class SIGNDelete(SIGNNet):
    def __init__(self, args, in_channels, out_channels, num_layers=3, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args,in_channels, out_channels, num_layers=3)
        self.args = args
        self.deletion1 = DeletionLayer(args['hidden_dim'], mask_1hop)
        self.deletion2 = DeletionLayer(args['out_dim'], mask_2hop)

        for lin in self.lins:
            lin.requires_grad = False
        self.lin1.requires_grad = False
        self.lin2.requires_grad = False

    def forward(self, xs, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        xs = xs.transpose(0, 1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x1 = self.lin1(x)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.lin2(x)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self,x,return_all_emb=False):
        return super().forward(x,return_all_emb=True)
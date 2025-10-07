from audioop import bias
import torch
import torch.nn as nn
# from torch_geometric.nn.models import MLP   present in new version not compataible with mine
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

# Custom MLPo implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use PyTorch's Linear instead since torch_geometric.nn.dense.linear doesn't exist in this version
Linear = nn.Linear  # Alias for compatibility

class MLP(nn.Module):
    def __init__(self, channel_list, dropout=0.0, batch_norm=True, act="relu"):
        super(MLP, self).__init__()
        
        self.channel_list = channel_list
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.lins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(len(channel_list) - 1):
            self.lins.append(nn.Linear(channel_list[i], channel_list[i+1]))
            
        if batch_norm:
            for i in range(len(channel_list) - 2):
                self.batch_norms.append(nn.BatchNorm1d(channel_list[i+1]))
        
        if act == "relu":
            self.act = F.relu
        elif act == "elu":
            self.act = F.elu
        elif act == "tanh":
            self.act = torch.tanh
        else:
            self.act = lambda x: x
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
            
    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class CEU_GNN(nn.Module):

    def __init__(self, num_nodes, embedding_size, hidden_sizes, num_classes, weights, feature_update, model, dropout=0.5):

        super(CEU_GNN, self).__init__()
        self.feature_update = feature_update

        self.embedding = nn.Embedding(num_nodes, embedding_dim=embedding_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=feature_update)

        def gnn_layer(model, input_size, out_size, dropout):
            if model == 'GCN':
                gnn = GCNConv(input_size, out_size, bias=True)
            elif model == 'GAT':
                gnn = GATConv(input_size, out_size, bias=True, dropout=dropout)
            elif model == 'SAGE':
                gnn = SAGEConv(input_size, out_size, bias=True)
            elif model == 'GIN':
                mlp = MLP([input_size, out_size], dropout=0.5, batch_norm=True)

                gnn = GINConv(mlp)
            else:
                raise NotImplementedError('Unsupposed GNN', model)
            return gnn

        self.gnns = nn.ModuleList()
        output_size = embedding_size
        for hidden_size in hidden_sizes:
            self.gnns.append(gnn_layer(model, output_size, hidden_size, dropout))
            output_size = hidden_size
        self.gnns.append(gnn_layer(model, output_size, num_classes, dropout))

        self.ce = nn.CrossEntropyLoss()
        self.ce2 = nn.CrossEntropyLoss(reduction='none')
        self.ce3 = nn.CrossEntropyLoss(reduction='sum')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nodes, edge_index, v=None, delta=None):
        x = self.embedding.weight
        if v is not None and delta is not None:
            x += delta

        for i, gnn in enumerate(self.gnns):
            if i == len(self.gnns) - 1:
                x = gnn(x, edge_index)
            else:
                x = self.relu(gnn(x, edge_index))
                x = self.dropout(x)
        return x[nodes]

    def loss(self, y_hat, y):
        return self.ce(y_hat, y)

    def losses(self, y_hat, y):
        return self.ce2(y_hat, y)

    def loss_sum(self, y_hat, y):
        return self.ce3(y_hat, y)

    def embeddings(self, nodes=None):
        return self.embedding.weight if nodes is None else self.embedding(nodes)

    def reset_parameters(self, weights):
        for gnn in self.gnns:
            gnn.reset_parameters()
        if self.feature_update:
            self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=self.feature_update)


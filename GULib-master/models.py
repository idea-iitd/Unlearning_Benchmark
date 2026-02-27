import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool


class abstract_model(nn.Module):
    """Abstract base class for models with save/load functionality."""
    def __init__(self):
        super(abstract_model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_model(self, save_path):
        """Save model state dictionary to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)


class RandomizedClassifier(nn.Module):
    """Randomized classifier for model outputs."""
    def __init__(self, hidden_channels, out_channels, requires_grad=False):
        super(RandomizedClassifier, self).__init__()
        self.cls = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        # Set requires_grad for all parameters
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.cls(x)
        return x




class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index)
        if return_all_emb:
            return x1, x2, x3
        return x3

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index)
        if return_all_emb:
            return x1, x2, x3
        return x3

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(in_dim, hidden_dim))
        self.conv2= GINConv(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x2 = self.conv2(x, edge_index)
        if return_all_emb:
            return x1, x2
        return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits


class GCNNet3(abstract_model):
    """GCN model class for cognac unlearning framework."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, 
                 dataset_name=None, downstream_task="node", adj=None):
        super(GCNNet3, self).__init__()
        self.num_layers = num_layers
        self.dataset_name = dataset_name
        self.downstream_task = downstream_task
        self.adj = adj
        
        if hidden_dim is None:
            hidden_dim = 64
        
        self.convs = nn.ModuleList()
        
        if self.dataset_name == "ogbn-products":
            self.convs.append(nn.Linear(in_dim, hidden_dim))
            self.convs.append(nn.Linear(hidden_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            if self.downstream_task == "graph":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.linear = nn.Linear(hidden_dim, out_dim)
            else:
                self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index, return_all_emb=False, return_feature=False, batch=None):
        if self.dataset_name == "ogbn-products":
            if self.adj is None:
                raise ValueError("adj matrix must be provided for ogbn-products dataset")
            
            x_list = []
            x = torch.mm(self.adj, x)
            x = self.convs[0](x)
            x_list.append(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = torch.mm(self.adj, x)
            x = self.convs[-1](x)
            
            if self.downstream_task == "graph":
                x = global_mean_pool(x, batch)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.linear(x)
            x_list.append(x)
            
            if return_all_emb:
                return x_list
            if return_feature:
                return x, x_list[0]
            return x
        else:
            x_list = []
            x = self.convs[0](x, edge_index)
            x_list.append(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            if self.downstream_task == "graph":
                x = global_mean_pool(x, batch)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.linear(x)
            x_list.append(x)
            
            if return_all_emb:
                return x_list
            if return_feature:
                return x, x_list[0]
            return x

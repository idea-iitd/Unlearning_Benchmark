import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse

##########################################################################
# graph adj utils
def row_norm(adj):
    if isinstance(adj, torch_sparse.SparseTensor):
        # Add self loop
        adj_t = torch_sparse.fill_diag(adj, 1)
        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        return adj_t

def sym_norm(adj):
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    if isinstance(adj, torch_sparse.SparseTensor):
        adj_t = gcn_norm(adj, add_self_loops=True) 
        return adj_t
    
def remove_self_loop(adj):
    # this also support adjs of any shapes
    row   = adj.storage.row()
    col   = adj.storage.col()    
    value = adj.storage.value()
    size  = adj.sparse_sizes()
    
    keep_inds = [row != col]
    
    return torch_sparse.SparseTensor(row   = row[keep_inds], 
                                     col   = col[keep_inds],
                                     value = adj.storage.value()[keep_inds], 
                                     sparse_sizes=size)

##########################################################################
# cross and smoothing utils
def pred_test(out, data, evaluator):
    pred = out.argmax(dim=-1, keepdim=True)
    data.y = data.y.view(-1,1)
    val_acc = 0
    train_acc = evaluator.eval({
        'y_true': data.y[data.train_indices],
        'y_pred': pred[data.train_indices]
    })['acc']
    if len(data.val_indices) != 0:
        val_acc = evaluator.eval({
            'y_true': data.y[data.val_indices],
            'y_pred': pred[data.val_indices]
        })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_indices],
        'y_pred': pred[data.test_indices]
    })['acc']
    return train_acc, val_acc, test_acc

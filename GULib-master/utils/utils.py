import copy
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import networkx as nx
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, recall_score
from model.base_gnn.ceu_model import CEU_GNN
from tqdm import tqdm
from scipy.stats import entropy
from sklearn import preprocessing
from numpy.linalg import norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
from typing import Optional
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics import roc_curve, auc
from torch_geometric.utils import k_hop_subgraph, is_undirected, to_undirected, negative_sampling, subgraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inductive_edge(data):
    train_edge_index = data.edge_index.clone()
    test_edge_index = data.edge_index.clone()
    data.train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    train_new_index = 0
    test_new_index = 0
    train_dict = {}
    test_dict = {}
    for node in range(data.x.size(0)):
        if data.train_mask[node]:
            train_dict[node] = train_new_index
            train_new_index += 1
        elif data.test_mask[node]:
            test_dict[node] = test_new_index
            test_new_index += 1
    train_edge_mask = data.train_mask[train_edge_index[0]] & data.train_mask[train_edge_index[1]]
    data.train_edge_index = train_edge_index[:, train_edge_mask]
    for edge in range(data.train_edge_index.size(1)):
        data.train_edge_index[0][edge] = train_dict[data.train_edge_index[0][edge].item()]
        data.train_edge_index[1][edge] = train_dict[data.train_edge_index[1][edge].item()]

    test_edge_mask = data.test_mask[test_edge_index[0]] & data.test_mask[test_edge_index[1]]
    data.test_edge_index = test_edge_index[:, test_edge_mask]
    for edge in range(data.test_edge_index.size(1)):
        data.test_edge_index[0][edge] = test_dict[data.test_edge_index[0][edge].item()]
        data.test_edge_index[1][edge] = test_dict[data.test_edge_index[1][edge].item()]


def connected_component_subgraphs(graph):
    """
    Find all connected subgraphs in a networkx Graph

    Args:
        graph (Graph): A networkx Graph

    Yields:
        generator: A subgraph generator
    """
    for c in nx.connected_components(graph):
        yield graph.subgraph(c)


def filter_edge_index_1(data, node_indices):
    """
    Remove unnecessary edges from a torch geometric Data, only keep the edges between node_indices.

    Args:
        data (Data): A torch geometric Data.
        node_indices (list): A list of nodes to be deleted from data.

    Returns:
        data.edge_index: The new edge_index after removing the node_indices.
    """
    if isinstance(data.edge_index, torch.Tensor):
        data.edge_index = data.edge_index.cpu()

    edge_index = data.edge_index
    node_index = np.isin(edge_index, node_indices)

    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = data.edge_index[:, col_index]

    return np.searchsorted(node_indices, edge_index)


def filter_edge_index(edge_index, node_indices, reindex=True):
    """
    Filters the edge index based on specified node indices and optionally reindexes the node indices.
    
    Args:
        edge_index (torch.Tensor or numpy.ndarray):
            
            The edge index of the graph in COO format.
        
        node_indices (list or array-like):
            
            Sorted list of node indices to retain in the edge index.
        
        reindex (bool, optional):
            
            If `True`, reindexes the node indices to the positions in `node_indices`. Defaults to `True`.
    
    Returns:
        Union[numpy.ndarray, torch.Tensor]:
            The filtered edge index, reindexed if `reindex=True`, else the original edge index type.
    """
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index


@torch.no_grad()
def negative_sampling_kg(edge_index, edge_type):
    '''Generate negative samples but keep the node type the same'''

    edge_index_copy = edge_index.clone()
    for et in edge_type.unique():
        mask = (edge_type == et)
        old_source = edge_index_copy[0, mask]
        new_index = torch.randperm(old_source.shape[0])
        new_source = old_source[new_index]
        edge_index_copy[0, mask] = new_source

    return edge_index_copy


# for GNNDelete
def to_directed(edge_index):
    row, col = edge_index
    mask = row < col

    return torch.stack((row[mask], col[mask]), dim=0)


def get_loss_fct(name):
    # if name == 'mse':
    #     loss_fct = nn.MSELoss(reduction='mean')
    # elif name == 'kld':
    #     loss_fct = BoundedKLDMean
    # elif name == 'cosine':
    #     loss_fct = CosineDistanceMean

    if name == 'kld_mean':
        loss_fct = BoundedKLDMean
    elif name == 'kld_sum':
        loss_fct = BoundedKLDSum
    elif name == 'mse_mean':
        loss_fct = nn.MSELoss(reduction='mean')
    elif name == 'mse_sum':
        loss_fct = nn.MSELoss(reduction='sum')
    elif name == 'cosine_mean':
        loss_fct = CosineDistanceMean
    elif name == 'cosine_sum':
        loss_fct = CosineDistanceSum
    elif name == 'linear_cka':
        loss_fct = LinearCKA
    elif name == 'rbf_cka':
        loss_fct = RBFCKA
    else:
        raise NotImplementedError

    return loss_fct


def BoundedKLDMean(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))


def BoundedKLDSum(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'sum'))


def CosineDistanceMean(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).mean()


def CosineDistanceSum(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).sum()


def LinearCKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def RBFCKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)


def kernel_HSIC(X, Y, sigma=None):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)


def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)


# CEU
def remove_undirected_edges(edges, edges_to_remove):
    _edges = set(copy.deepcopy(edges))
    for e in edges_to_remove:
        if not isinstance(e, tuple):
            e = tuple(e)
        if e in _edges:
            _edges.remove(e)
        if (e[1], e[0]) in _edges:
            _edges.remove((e[1], e[0]))
    return list(_edges)


def CEU_load_model(args, data, type='original', edges=None, edge=None, node=None):
    assert type in ['original', 'edge', 'node', 'retrain', 'unlearn'], f'Invalid type of model, {type}'
    if type == 'edge':
        model = CEU_create_model(args, data)
        model.load_state_dict(torch.load(os.path.join('./checkpoint', args.data, 'edges',
                                                      f'{args.model}_{args.data}_{edge[0]}_{edge[1]}_best.pt')))
        return model
    elif type == 'node':
        model = CEU_create_model(args, data)
        model.load_state_dict(torch.load(os.path.join('./checkpoint', args.data, 'nodes',
                                                      f'{args.model}_{args.data}_{node}_best.pt')))
        return model
    else:
        model = CEU_create_model(args, data)
        model.load_state_dict(torch.load(CEU_model_path(args, type, edges)))
        return model


def CEU_create_model(args, data):
    embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
    model = CEU_GNN(data['num_nodes'], embedding_size,
                    args.hidden, data['num_classes'], data['features'], args.feature_update, args.model)
    return model


def CEU_model_path(args, type, edges=None):
    if args["hidden"]:
        layers = '-'.join([str(h) for h in args.hidden])
        prefix = f'{args.model}_{args.data}_{layers}'
    else:
        prefix = f'{args.model}_{args.data}'

    if type == 'original':
        return os.path.join('./checkpoint/CEU', args.data, f'{prefix}_best.pt')
    elif type == 'retrain':
        if args["max_degree"]:
            filename = f'{prefix}_{type}_max_{args.method}{edges}_best.pt'
        else:
            filename = f'{prefix}_{type}_{args.method}{edges}_best.pt'
        return os.path.join('./checkpoint/CEU', args.data, filename)
    elif type == 'unlearn':
        assert edges is not None
        if args.batch_unlearn:
            prefix += '_batch'
        if args.unlearn_batch_size is not None:
            prefix += f'args.unlearn_batch_size'
        if args.approx == 'lissa':
            filename = f'{prefix}_{type}_{args["unlearning_methods"]}{edges}_{args["approx"]}d{args["depth"]}r{args.r}_best.pt'
        else:
            filename = f'{prefix}_{type}_{args["unlearning_methods"]}{edges}_{args["approx"]}_d{args["damping"]}_best.pt'
        return os.path.join('./checkpoint/CEU', args["dataset_name"], filename)
    else:
        raise ValueError('Invalid type of model,', type)


def JSD(P, Q):
    _M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, _M, axis=1) + entropy(Q, _M, axis=1))


##################for CGU###############

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, test_lb=1000, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    if Flag == 0:
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        all_index = torch.randperm(data.y.shape[0])
        data.val_mask = index_to_mask(all_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(all_index[val_lb: (val_lb + test_lb)], size=data.num_nodes)
        data.train_mask = index_to_mask(all_index[(val_lb + test_lb):], size=data.num_nodes)
    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def preprocess_data(X):
    '''
    input:
        X: (n,d), torch.Tensor

    '''
    X_np = X.cpu().numpy()
    scaler = preprocessing.StandardScaler().fit(X_np)
    X_scaled = scaler.transform(X_np)
    row_norm = norm(X_scaled, axis=1)
    X_scaled = X_scaled / row_norm.max()
    return torch.from_numpy(X_scaled)


class MyGraphConv(MessagePassing):
    """
    Customized graph convolution layer using a customized propagation matrix.

    This layer performs graph convolution by propagating messages through the graph's edges.
    It supports options like degree normalization and Generalized PageRank (GPR) for message passing.
    """
    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1,
                 add_self_loops: bool = True,
                 alpha=0.5, XdegNorm=False, GPR=False, **kwargs):
        """
        Initializes the MyGraphConv layer.

        Args:
            K (int, optional):
                Number of propagation steps. Defaults to `1`.
            
            add_self_loops (bool, optional):
                Whether to add self-loops to the graph. Defaults to `True`.
            
            alpha (float, optional):
                Scaling factor for the propagation. Defaults to `0.5`.
            
            XdegNorm (bool, optional):
                Whether to apply degree normalization to the node features. Defaults to `False`.
            
            GPR (bool, optional):
                Whether to use Generalized PageRank during propagation. Defaults to `False`.
            
            **kwargs:
                Additional keyword arguments passed to the base class `MessagePassing`.
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.K = K
        self.add_self_loops = add_self_loops
        self.alpha = alpha
        self.XdegNorm = XdegNorm
        self.GPR = GPR
        self._cached_x = None  # Not used
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the cached node features.
        """
        self._cached_x = None  # Not used

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """
        Performs a forward pass of the graph convolution.

        Args:
            x (Tensor):
                Node feature matrix.
            
            edge_index (Adj):
                Graph connectivity in COO format or as a SparseTensor.
            
            edge_weight (OptTensor, optional):
                Edge weights. Defaults to `None`.

        Returns:
            Tensor:
                The updated node feature matrix after propagation.
        """
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        elif isinstance(edge_index, SparseTensor):
            edge_index = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)

        if self.XdegNorm:
            # X <-- D^{-1}X, our degree normalization trick
            num_nodes = maybe_num_nodes(edge_index, None)
            row, col = edge_index[0], edge_index[1]
            deg = degree(row).unsqueeze(-1)

            deg_inv = deg.pow(-1)
            deg_inv = deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        if self.GPR:
            xs = []
            xs.append(x)
            if self.XdegNorm:
                x = deg_inv * x  # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                xs.append(x)
            return torch.cat(xs, dim=1) / (self.K + 1)
        else:
            if self.XdegNorm:
                x = deg_inv * x  # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Constructs messages from neighboring nodes.

        Args:
            x_j (Tensor):
                Features of neighboring nodes.
            
            edge_weight (Tensor):
                Weights of the edges.

        Returns:
            Tensor:
                Weighted messages to be aggregated.
        """
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        Aggregates messages from neighbors and performs the message passing.

        Args:
            adj_t (SparseTensor):
                Sparse adjacency matrix in COO format.
            
            x (Tensor):
                Node feature matrix.

        Returns:
            Tensor:
                Aggregated messages.
        """
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')


# prepare P matrix in PyG format
def get_propagation(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None,
                    alpha=0.5):
    """
    return:
        P = D^{-\alpha}AD^{-(1-alpha)}.

    """
    fill_value = 2. if improved else 1.
    assert (0 <= alpha) and (alpha <= 1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_left = deg.pow(-alpha)
    deg_inv_right = deg.pow(alpha - 1)
    deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
    deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)

    return edge_index, deg_inv_left[row] * edge_weight * deg_inv_right[col]


# training iteration for binary classification
def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0,
                X_val=None, y_val=None):
    '''
        b is the noise here. It is either pre-computed for worst-case, or pre-defined.

    '''
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)

    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()

        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))

        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


# loss for binary classification
def lr_loss(w, X, y, lam):
    '''
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar

    return:
        averaged training loss with L2 regularization

    '''
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


# evaluate function for binary classification
def lr_eval(w, X, y):
    '''
    input:
        w: (d,)
        X: (n,d)
        y: (n,)

    return:
        prediction accuracy

    '''
    return X.mv(w).sign().eq(y).float().mean()


# gradient of loss wrt w for binary classification
def lr_grad(w, X, y, lam):
    '''
    The gradient here is computed wrt sum.

    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar

    return:
        gradient: (d,)

    '''
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w


# hessian of loss wrt w for binary classification
def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    '''
    The hessian here is computed wrt sum.

    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
        batch_size: int

    return:
        hessian: (d,d)
    '''
    z = torch.sigmoid(y * X.mv(w))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()


# training iteration for binary classification
def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.05, wd=0,
                X_val=None, y_val=None):
    '''

        b is the noise here. It is either pre-computed for worst-case, or pre-defined.

    '''
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)

    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()

        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))

        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


# aggregated loss for multiclass classification
def ovr_lr_loss(w, X, y, lam, weight=None):
    '''

    input:

        w: (d,c)
        X: (n,d)
        y: (n,c), one-hot
        lambda: scalar
        weight: (c,) / None

    return:
        loss: scalar
    '''
    z = batch_multiply(X, w) * y
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


def ovr_lr_eval(w, X, y):
    '''
    input:

        w: (d,c)
        X: (n,d)
        y: (n,), NOT one-hot

    return:

        loss: scalar

    '''
    pred = X.mm(w).max(1)[1]
    # softlabel = F.softmax(X.mm(w))
    # y_true = torch.zeros(y.size(0),7).cpu()
    # y_index = y.view(y.size(0),-1).cpu()
    # y_true = y_true.scatter_(1, y_index, 1)
    F1_score = f1_score(y.cpu(), pred.cpu(), average="micro",zero_division=np.nan)
    Recall_score = recall_score(y.cpu(), pred.cpu(), average="micro",zero_division=np.nan)
    return pred.eq(y).float().mean(), F1_score, Recall_score


def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS',
                    lr=0.01, wd=0, X_val=None, y_val=None):
    '''

    y: (n_train, c). one-hot

    y_val: (n_val,) NOT one-hot

    '''
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)

    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()

        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))

        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        if X_val is not None:
            val_acc = ovr_lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i + 1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)


def get_worst_Gbound_feature(lam, m, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2 * c * lam + (c * gamma1 + lam * c1) * deg_m) ** 2) / (lam ** 4) / (m - 1)


def get_worst_Gbound_edge(lam, m, K, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return 16 * gamma2 * (K ** 2) * ((c * gamma1 + lam * c1) ** 2) / (lam ** 4) / m


def get_worst_Gbound_node(lam, m, K, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2 * c * lam + K * (c * gamma1 + lam * c1) * (2 * deg_m - 1)) ** 2) / (lam ** 4) / (m - 1)


def get_c(delta):
    return np.sqrt(2 * np.log(1.5 / delta))


def get_budget(std, eps, c):
    return std * eps / c


# K = X^T * X for fast computation of spectral norm
def get_K_matrix(X):
    K = X.t().mm(X)
    return K


# using power iteration to find the maximum eigenvalue
def sqrt_spectral_norm(A, num_iters=100):
    '''
    return:

        sqrt of maximum eigenvalue/spectral norm

    '''
    x = torch.randn(A.size(0)).float().to(device)
    for i in range(num_iters):
        x = A.mv(x)
        x_norm = x.norm()
        x /= x_norm
    max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
    return math.sqrt(max_lam)


################SGU###################
import scipy.sparse as sp


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def cal_distance(vector1, vector2):
    # vector1 = vector1.reshape(1,-1)
    # vector2 = vector2.reshape(1, -1)
    # cos similarity
    # similarity = F.cosine_similarity(vector1,vector2,dim=0)
    similarity = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity


def Reverse_CE(out, y):
    return 0


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def plot_auc(y_true, y_score):
    y_true = y_true
    y_score = y_score

    # 计算ROC曲线上的点
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 计算AUC
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def remove_node_from_graph(data, node_id=None, removal_queue=None):
    """
    Node removal for graph classification. datalist is a list of Data objects. Each Data object corresponds to one training graph.

    If graph/node id are provided, remove accordingly.

    Otherwise remove a random node from a random graph.

    Can optionally record the removal queue.

    """

    # if graph_id is None:
    #     graph_id = np.random.choice(len(datalist))
    #     while datalist[graph_id].empty:
    #         # Ensure we do not further remove from an empty graph!
    #         graph_id = np.random.choice(len(datalist))

    # data = datalist[graph_id]

    # if node_id is None:
    #     # data.mask records the remaining valid nodes in the graph
    #     node_id = np.random.choice(torch.arange(data.x.shape[0])[data.mask])

    # Editing graph
    data.mask = torch.ones((data.x.shape[0],)).bool()
    data.empty = False
    data.x[node_id] = torch.zeros_like(data.x[node_id])
    data.mask[node_id] = False
    edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
    edge_mask[data.edge_index[0] == node_id] = False
    edge_mask[data.edge_index[1] == node_id] = False
    data.edge_index = data.edge_index[:, edge_mask]
    if data.edge_weight is not None:
        data.edge_weight = data.edge_weight[:, edge_mask]

    # Put edited graph back
    # if torch.any(data.mask):
    #     # Still have nodes for this graph, just put it back
    #     datalist[graph_id] = data
    # else:
    #     # Becomes an empty graph
    #     datalist[graph_id].empty = True
    #     # This ensure the masked feature matrix does not becomes empty! Still, the feature is already removed (i.e. zeros)
    #     data.mask[0] = True
    #     # Now, put the graph back.
    #     datalist[graph_id] = data

    # if removal_queue is not None:
    #     removal_queue.append([graph_id, node_id])
    #     return datalist, removal_queue

    return data


##MEGU##

def normalize_adj(adj, r=0.5):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list[-1]


def criterionKD(p, q, T=1.5):
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    soft_p = F.log_softmax(p / T, dim=1)
    soft_q = F.softmax(q / T, dim=1).detach()
    return loss_kl(soft_p, soft_q)


def calc_f1(y_true, y_pred, mask, multilabel=False):
    if multilabel:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    mask = mask.cpu()
    return f1_score(y_true[mask], y_pred[mask], average="micro")


###GraphRevoker
def filter_test_edges(edge_index, test_indices):
    """
    Remove edges which connect test set nodes.
    Args:
        edge_index (torch.LongTensor): A torch geometric edge index.
        test_indices (np.array): A list of nodes in the test set.

    Returns:
        edge_index2: The new edge_index after removing the edges.
    """
    assert isinstance(test_indices, np.ndarray)

    test_edge_mask = np.logical_or(np.isin(edge_index[0], test_indices),
                                   np.isin(edge_index[1], test_indices))
    other_edge_mask = np.logical_not(test_edge_mask)
    edge_index2 = edge_index[:, other_edge_mask]
    if not isinstance(edge_index2, torch.Tensor):
        edge_index2 = torch.from_numpy(edge_index2)

    return edge_index2


def filter_edge_index_2(data, node_indices):
    """
    Remove unnecessary edges from a torch geometric Data, only keep the edges between node_indices.
    An extended version of filter_edge_index_1 which also processes edge_index_train.

    Args:

        data (Data): A torch geometric Data.
        node_indices (list): A list of nodes to be deleted from data.

    Returns:

        data.edge_index: The new edge_index after removing the node_indices.
        
    """
    if isinstance(data.edge_index, torch.Tensor):
        data.edge_index = data.edge_index.cpu()

    edge_index = data.edge_index
    node_index = np.isin(edge_index, node_indices)

    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = data.edge_index[:, col_index]

    edge_index = np.searchsorted(node_indices, edge_index)

    if isinstance(data.edge_index_train, torch.Tensor):
        data.edge_index_train = data.edge_index_train.cpu()

    edge_index_train = data.edge_index_train
    node_index_train = np.isin(edge_index_train, node_indices)

    col_index_train = np.nonzero(np.logical_and(node_index_train[0], node_index_train[1]))[0]
    edge_index_train = data.edge_index_train[:, col_index_train]

    edge_index_train = np.searchsorted(node_indices, edge_index_train)

    return edge_index, edge_index_train


def evaluate_attack_with_AUC(data, label):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(label, data.reshape(-1, 1))


def _calculate_distance(data0, data1, distance='l2_norm'):
    if distance == 'l2_norm':
        return np.array([np.linalg.norm(data0[i] - data1[i]) for i in range(len(data0))])
    elif distance == 'direct_diff':
        return data0 - data1
    else:
        raise Exception("Unsupported distance")


####UTU####

@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def split_forget_retain(data, df_size, subset='in'):
    if df_size >= 100:  # df_size is number of nodes/edges to be deleted
        df_size = int(df_size)
    else:  # df_size is the ratio
        df_size = int(df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')
    df_mask_all = gen_inout_mask(data)[subset]
    df_nonzero = df_mask_all.nonzero().squeeze()  # subgraph子图内/外的edge idx序号

    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # Collect enclosing subgraph of Df for loss computation
    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        1,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    # assert not is_undirected(data.train_pos_edge_index)

    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index,
                                                                  [df_mask.cuda().int(), two_hop_mask.int()])
    # to_undirected return full undirected edges and corresponding mask for given edge_attrs
    two_hop_mask = two_hop_mask.bool()
    df_mask = df_mask.bool()
    dr_mask = ~df_mask

    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask
    return data


def gen_inout_mask(data):
    _, local_edges, _, mask = k_hop_subgraph(
        data.val_pos_edge_index.flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    distant_edges = data.train_pos_edge_index[:, ~mask]
    print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

    in_mask = mask
    out_mask = ~mask

    return {'in': in_mask, 'out': out_mask}

@torch.no_grad()
def member_infer_attack(target_model, attack_model, data, logits=None):
    '''Membership inference attack'''

    edge = data.train_pos_edge_index[:, data.df_mask]  # Deleted edges in the training set
    z = target_model(data.x, data.train_pos_edge_index[:, data.dr_mask])
    if attack_model.fc1.in_features == 2:
        feature1 = target_model.decode(z, edge).sigmoid()
        feature0 = 1 - feature1
        feature = torch.stack([feature0, feature1], dim=1) # Posterior MI
    else:
        feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1)  # Embedding/Repr. MI
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()  # label should be zero, aka if pred is 1(member) then attack success

    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()


def filter_edge_index_3(train_data, node_indices, all_edges_to_remove=None,edge_set=None):
    """
    Filters the edge index of training data based on specified node indices and removes specified edges.

    Args:
        train_data:
            
            The training data object containing the edge index.
        
        node_indices (list or array-like):
            
            List of node indices to retain in the edge index.
        
        all_edges_to_remove (list or array-like, optional):
            
            List of edges to remove from the edge index. Each edge should be a tuple of (source, destination).
            Defaults to `None`.
    
    Returns:
        numpy.ndarray:
            The filtered edge index with node indices mapped back to the `node_indices` array.
    """
    # print(all_edges_to_remove)
    if edge_set==None:
        if isinstance(train_data.train_edge_index, torch.Tensor):
            train_data.train_edge_index = train_data.train_edge_index.cpu()

        edge_index = train_data.train_edge_index
        node_index = np.isin(edge_index, node_indices)

        col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
        edge_index = train_data.train_edge_index[:, col_index]
    else:
        if isinstance(train_data.test_edge_index, torch.Tensor):
            train_data.test_edge_index = train_data.test_edge_index.cpu()

        edge_index = train_data.test_edge_index
        node_index = np.isin(edge_index, node_indices)

        col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
        edge_index = train_data.test_edge_index[:, col_index]

    if all_edges_to_remove is not None:
        all_edges_to_remove_set = set(map(tuple, all_edges_to_remove.tolist()))
    else:
        all_edges_to_remove_set = set()
    
    # Convert edge_index to a set of tuples for easy removal
    edge_set = set(map(tuple, edge_index.T.tolist()))
    # all_edges_to_remove_set = set(map(tuple, all_edges_to_remove))
    #判断all_edges_to_remove是否在edge_set中
    print(all_edges_to_remove_set.issubset(edge_set))
    # Remove edges specified in all_edges_to_remove
    edge_set.difference_update(all_edges_to_remove_set)

    # Convert back to numpy array
    edge_index = np.array(list(edge_set)).T

    return np.searchsorted(node_indices, edge_index)


# def remove_node_from_graph(data, remove_indices=None):

#     for node_id in remove_indices:
#         data.x[node_id] = torch.zeros_like(data.x[node_id])
#         node_mask = torch.ones(data.x.size(0), dtype=torch.bool)
#         node_mask[remove_indices] = False 
#         edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
#         data.edge_index = data.edge_index[:, edge_mask]
#         # edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
#         # edge_mask[data.edge_index[0] == node_id] = False
#         # edge_mask[data.edge_index[1] == node_id] = False
#         # data.edge_index = data.edge_index[:, edge_mask]
#         if data.edge_weight is not None:
#             data.edge_weight = data.edge_weight[:, edge_mask]
        
    
#     return data


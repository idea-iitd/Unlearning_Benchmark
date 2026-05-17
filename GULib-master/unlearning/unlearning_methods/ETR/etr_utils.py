"""
unlearning/unlearning_methods/ETR/etr_utils.py
==============================================

ETR-specific utility functions, ported from ETR's open-source
implementation (lib_utils/utils.py) with two adaptations:

  1. Hyperparameter loading removed — values come from CLI args
     (parameter_parser.py), not from a YAML config file.

  2. update_edge_index_unlearn() — node-task branch: the original ETR
     code returns positions WITHIN the unique-edges subset and then uses
     them as positions into the FULL edge_index, which is incorrect for
     any non-trivial graph. Fixed by mapping back through `unique_indices`.

These helpers are used by etr.py (pipeline) and ETRTrainer.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def filter_edge_index(edge_index, node_indices, reindex=True):
    """Keep only edges where both endpoints are in `node_indices` (must be sorted)."""
    assert np.all(np.diff(node_indices) >= 0), "node_indices must be sorted"
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    node_mask = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_mask[0], node_mask[1]))[0]
    edge_index = edge_index[:, col_index]
    if reindex:
        return np.searchsorted(node_indices, edge_index)
    return edge_index


def get_dataset_train(data):
    """
    Return a Data object with edges restricted to train-to-train pairs.
    Node features / labels / masks are preserved without reindexing.
    """
    train_indices = np.nonzero(data.train_mask.cpu().numpy())[0]
    edge_index = filter_edge_index(data.edge_index, train_indices, reindex=False)
    if edge_index.shape[1] == 0:
        edge_index = np.array([[1, 2], [2, 1]])
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index).long()
    return Data(
        x=data.x,
        edge_index=edge_index,
        y=data.y,
        train_mask=data.train_mask,
        test_mask=data.test_mask,
        train_indices=data.train_indices,
        test_indices=data.test_indices,
    )


def get_influence_nodes(unlearn_nodes, edge_index, hops=2, unlearn_task="node"):
    """
    Set of nodes within `hops` of any `unlearn_nodes` in `edge_index`.

    For node-unlearning the unlearn_nodes themselves are excluded (only
    neighbours are returned). For edge / feature-unlearning they are
    kept, matching ETR's original semantics.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    influenced = np.asarray(unlearn_nodes)
    for _ in range(hops):
        mask = np.isin(edge_index[0], influenced)
        neighbors = edge_index[1, mask]
        influenced = np.unique(np.append(influenced, neighbors))
    if unlearn_task == "node":
        influenced = np.setdiff1d(influenced, np.asarray(unlearn_nodes))
    return influenced


def get_subgraph(node_id, data, hops=2):
    """
    Build a k-hop subgraph Data object around `node_id` with a relabeled
    edge_index.

    The returned object carries:
        .x            node features for the subset
        .edge_index   relabeled (local) edges
        .y            labels for the subset
        .batch_size   number of seed nodes (len(node_id))
        .mapping      positions of seed nodes within the relabeled subset
                      (as returned by k_hop_subgraph)

    *** WHY mapping IS CRITICAL ***
    k_hop_subgraph(relabel_nodes=True) returns `subset` as a SORTED
    tensor of global node IDs.  The seed nodes appear wherever their
    global IDs sort to — they are NOT necessarily the first elements.
    `mapping` gives their exact positions.  ETRTrainer._grad_and_info
    uses `data.mapping` to index predictions and labels for the seed
    nodes; do NOT change this to `[:data.batch_size]`.
    """
    if isinstance(node_id, list):
        node_id = np.asarray(node_id)
    if isinstance(node_id, np.ndarray):
        ei_np = data.edge_index.cpu().numpy()
        node_id = node_id[np.isin(node_id, ei_np)]
        if len(node_id) == 0:
            node_id = np.array([int(ei_np[0, 0])])
    node_id_t = torch.tensor(node_id, dtype=torch.long)
    subset, edge_index, mapping, _ = k_hop_subgraph(
        node_id_t, hops, data.edge_index, relabel_nodes=True
    )
    return Data(
        x=data.x[subset],
        edge_index=edge_index,
        y=data.y[subset],
        batch_size=len(node_id),
        mapping=mapping,          # positions of seed nodes in the sorted subset
    )


def update_edge_index_unlearn(args, edge_index, delete_nodes, delete_edge_index=None):
    """
    Build a new edge_index after removing `delete_nodes` (node task) or
    specific edges identified by their positions `delete_edge_index` (edge task).

    Ported from ETR's lib_utils/utils.py with one correctness fix to the
    node branch (see inline comment).
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    unique_indices = np.where(edge_index[0] < edge_index[1])[0]
    unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

    if args["unlearn_task"] == "edge":
        remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
    else:
        unique_edge_index = edge_index[:, unique_indices]
        delete_mask = np.logical_or(
            np.isin(unique_edge_index[0], delete_nodes),
            np.isin(unique_edge_index[1], delete_nodes),
        )
        keep_mask = np.logical_not(delete_mask)
        # BUG-FIX vs original ETR: np.where(keep_mask) gives positions
        # WITHIN the unique-edges subset, not within the full edge_index.
        # Map back through unique_indices so remain_indices is valid for
        # full edge_index.
        remain_indices = unique_indices[np.where(keep_mask)[0]]

    remain_encode = (
        edge_index[0, remain_indices] * edge_index.shape[1] * 2
        + edge_index[1, remain_indices]
    )
    unique_encode_not = (
        edge_index[1, unique_indices_not] * edge_index.shape[1] * 2
        + edge_index[0, unique_indices_not]
    )
    sort_indices = np.argsort(unique_encode_not)
    remain_indices_not = unique_indices_not[
        sort_indices[
            np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)
        ]
    ]
    remain_indices = np.union1d(remain_indices, remain_indices_not)
    return torch.from_numpy(edge_index[:, remain_indices]).long()


def get_dataset_unlearn(args, data, unlearning_id, delete_edge_index=None):
    """Construct the post-unlearning Data object."""
    if args["unlearn_task"] == "feature":
        x = data.x.clone()
        x[unlearning_id] = 0
        return Data(
            x=x,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            test_mask=data.test_mask,
            train_indices=data.train_indices,
            test_indices=data.test_indices,
        )
    if args["unlearn_task"] == "node":
        edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index, unlearning_id)
    else:  # 'edge'
        edge_index_unlearn = update_edge_index_unlearn(
            args, data.edge_index, unlearning_id, delete_edge_index
        )
    return Data(
        x=data.x,
        edge_index=edge_index_unlearn,
        y=data.y,
        train_mask=data.train_mask,
        test_mask=data.test_mask,
        train_indices=data.train_indices,
        test_indices=data.test_indices,
    )
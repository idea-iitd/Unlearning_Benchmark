import torch
import torch.nn.functional as F
import pickle
import numpy as np
import sys
from model.base_gnn.gcn import GCNNet
from model.base_gnn.deletion import GCNDelete
from model.base_gnn.gat import GATNet
from model.base_gnn.deletion import GATDelete
from model.base_gnn.gin import GINNet
from model.base_gnn.deletion import GINDelete
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
import copy
import torch_sparse
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models import GCNNet3
import models
# add near top with other imports
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import joblib  


# --- place near top of file with other imports ---
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict

# --- helpers ---
def compute_normalized_adj(edge_index, num_nodes):
    """
    Returns (u, v, vals) arrays for symmetric normalized adjacency Atilde:
    vals[k] = deg^{-1/2}[u[k]] * deg^{-1/2}[v[k]]
    """
    u, v = edge_index.cpu().numpy()
    # undirected edges: count both directions when computing degree
    deg = np.bincount(u, minlength=num_nodes).astype(float) + np.bincount(v, minlength=num_nodes).astype(float)
    deg = deg.clip(min=1.0)
    deg_sqrt_inv = 1.0 / np.sqrt(deg)
    vals = deg_sqrt_inv[u] * deg_sqrt_inv[v]
    return u, v, vals

def compute_trend_features_from_probs(probs, edge_index, num_nodes, order=2):
    """
    probs: torch tensor (N, C) probabilities (softmax)
    returns conf (N,), tau_tilde (N,4), taus list
    """
    probs_np = probs.detach().cpu().numpy() if hasattr(probs, "detach") else probs.cpu().numpy()
    conf = probs_np.max(axis=1)           # tau^(0)
    u, v, vals = compute_normalized_adj(edge_index, num_nodes)
    # build symmetric neighbor lists
    neigh_map = [[] for _ in range(num_nodes)]
    neigh_w   = [[] for _ in range(num_nodes)]
    for a,b,w in zip(u, v, vals):
        neigh_map[a].append(b); neigh_w[a].append(w)
        neigh_map[b].append(a); neigh_w[b].append(w)
    # compute taus
    taus = [None] * (order+1)
    taus[0] = conf.copy()
    for k in range(1, order+1):
        tnext = np.zeros(num_nodes, dtype=float)
        for i in range(num_nodes):
            if not neigh_map[i]:
                tnext[i] = taus[k-1][i]
            else:
                s = 0.0
                for nb, w in zip(neigh_map[i], neigh_w[i]):
                    s += w * taus[k-1][nb]
                tnext[i] = s
        taus[k] = tnext
    delta1 = taus[1] - taus[0] if order >= 1 else np.zeros_like(conf)
    delta2 = taus[2] - taus[1] if order >= 2 else np.zeros_like(conf)
    tau_tilde = np.stack([
        (delta1 < 0).astype(int),
        (delta1 > 0).astype(int),
        (delta2 < 0).astype(int),
        (delta2 > 0).astype(int)
    ], axis=1)
    return conf, tau_tilde, taus

# --- TrendAttack function ---
def TrendAttack(original_probs,
                probs,
                data,
                train_mask,
                test_mask,
                unlearned_indices=None,
                run_number=0,
                u_ratio=None,
                dataset=None,
                order=2,
                train_attack=True,
                verbose=True):
    """
    Build node-level TrendAttack features from probabilities and optionally train a simple logistic attack.

    Args:
      probs: (N, C) torch tensor or numpy array of model probabilities (post-unlearning model outputs).
      data: your `data` object (contains edge_index, x, etc).
      train_mask/test_mask: boolean numpy arrays (same as in your script).
      unlearned_indices: optional list of ints marking nodes to treat as positive (endpoints of unlearned edges).
                         If None and train_attack=True, function will attempt to load the same file your script uses.
      run_number, u_ratio, dataset, copy_str: used only to construct the default unlearn_idx_path if unlearned_indices is None.
      order: number of aggregation steps for taus (default 2).
      train_attack: if True, will try to train LogisticRegression if labels are available; otherwise only returns features.
    Returns:
      (clf, X_node, y_node, attack_probs, attack_preds)
      clf is the sklearn model (or None if not trained).
      attack_probs/preds are for test_mask nodes (or None if no clf).
    """
    # ensure probs is torch tensor
    import torch
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)

    num_nodes = data.x.size(0)
    # compute trend features
    conf, tau_tilde, _ = compute_trend_features_from_probs(probs, data.edge_index, num_nodes, order=order)

    # degree
    deg = np.zeros(num_nodes, dtype=float)
    u,v = data.edge_index.cpu().numpy()
    for a,b in zip(u,v):
        deg[a] += 1; deg[b] += 1

    # mean neighbor conf
    s_conf = defaultdict(float)
    count_nb = defaultdict(int)
    for a,b in zip(u,v):
        s_conf[a] += conf[b]; count_nb[a] += 1
        s_conf[b] += conf[a]; count_nb[b] += 1
    mean_nb_conf = np.zeros(num_nodes, dtype=float)
    for i in range(num_nodes):
        mean_nb_conf[i] = s_conf[i] / count_nb[i] if count_nb[i] > 0 else conf[i]

    # assemble X_node
    X_node = np.concatenate([
        conf.reshape(-1,1),
        deg.reshape(-1,1),
        mean_nb_conf.reshape(-1,1),
        tau_tilde
    ], axis=1)

    # Prepare labels y_node if requested
    y_node = np.zeros(num_nodes, dtype=int)
    if unlearned_indices is None and train_attack:
        # try to load same unlearned index file as in your script
        try:
            if dataset is None or u_ratio is None:
                raise ValueError("dataset and u_ratio required to auto-load unlearned indices when unlearned_indices is None.")
            unlearn_idx_path = f"/data/unlearning_task/transductive/imbalanced/unlearning_nodes_{u_ratio}_{dataset}_{run_number}_nodes_{int(u_ratio * num_nodes)}.txt"
            with open(unlearn_idx_path, "r") as f:
                unlearned_indices = list(map(int, f.readlines()))
            # if verbose:
            #     print(f"[TrendAttack] Loaded unlearned indices from {unlearn_idx_path}; positives={len(unlearned_indices)}")
        except Exception as e:
            if verbose:
                print("[TrendAttack] Could not auto-load unlearned indices:", e)
            unlearned_indices = None

    if unlearned_indices is not None:
        for idx in unlearned_indices:
            if 0 <= idx < num_nodes:
                y_node[idx] = 1

    clf = None
    attack_probs = None
    attack_preds = None
    # breakpoint()
    if train_attack:
        if y_node.sum() == 0:
            if verbose:
                print("[TrendAttack] No positive labels available to train attack")
            return None
        # train on train_mask nodes
        tr_mask = np.array(train_mask).astype(bool)
        Xtr = X_node[tr_mask]
        ytr = y_node[tr_mask]
        # If there are too few positives in train, this may fail — we keep default logistic with class_weight='balanced'
        try:
            clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
            # if verbose:
            #     print("[TrendAttack] Trained LogisticRegression attack.")
            # evaluate on test_mask
            eval_mask = train_mask   # evaluate on training nodes
            Xte = X_node[np.array(eval_mask).astype(bool)]
            yte = y_node[np.array(eval_mask).astype(bool)]
            # Xte = X_node[np.array(test_mask).astype(bool)]
            # yte = y_node[np.array(test_mask).astype(bool)]
            if Xte.shape[0] > 0:
                attack_probs = clf.predict_proba(Xte)[:,1]
                attack_preds = (attack_probs > 0.5).astype(int)
                
                if yte.sum() > 0:
                    auc = roc_auc_score(yte, attack_probs)
                    acc = accuracy_score(yte, attack_preds)
                    # if verbose:
                    #     print(f"[TrendAttack] attack AUC: {auc:.4f}, accuracy: {acc:.4f}")
                else:
                    if verbose:
                        print("[TrendAttack] No positives in attack eval set; printed raw metrics suppressed.")
        except Exception as e:
            if verbose:
                print("[TrendAttack] Training failed:", e)
            clf = None
    return auc


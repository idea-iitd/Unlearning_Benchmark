"""
evaluate_unlearning.py
======================
Evaluates GNN unlearning methods (e.g., MEGU, GOLD, GNNDelete) against a
retrained-from-scratch "gold standard" model.

Metrics reported:
  - Node-classification accuracy and micro-F1 on the test set.
  - Pairwise prediction fidelity (exact-match rate) and logit L2 distance.
  - Optional attack AUROC scores.

Usage
-----
python evaluate_unlearning.py \
    --unlearn_ratio 0.1 \
    --dataset_name cora \
    --unlearning_methods MEGU \
    --unlearn_task node \
    --num_runs 3 \
    --base_model GCN \
    --attack_type MIattack
"""

import argparse
import copy
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor

import models
from MIA_attack import MI_attack
from Membership_Recall_Attack import MRattack
from model.base_gnn.deletion import GATDelete, GCNDelete, GINDelete
from model.base_gnn.gat import GATNet
from model.base_gnn.gcn import GCNNet
from model.base_gnn.gin import GINNet
from models import GCNNet3  # noqa: F401  (needed for checkpoint compatibility)
from Trend_attack import TrendAttack
from unlearning.unlearning_methods.Projector.utils.graph_projector_model_utils import (
    Pro_GNN,
)

# ---------------------------------------------------------------------------
# Compatibility shim: checkpoints saved as "models.models.GCNNet3"
# ---------------------------------------------------------------------------
if "models.models" not in sys.modules:
    sys.modules["models.models"] = models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path()  #Add your root path for Unlearning_Benchmark folder
PROCESSED_DATA_DIR = DATA_ROOT / "data/processed/transductive"
UNLEARN_TASK_DIR = DATA_ROOT / "data/unlearning_task/transductive/imbalanced"
MODEL_DIR = DATA_ROOT / "data/model/node_level"
UNLEARNED_MODEL_DIR = DATA_ROOT / "unlearned_models"
COGNAC_DIR = DATA_ROOT / "data/model"   #Add path to Cognac model directory 
ETR_DIR = DATA_ROOT / "data/model"    #Add path to ETR model directory 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTACK_MAP = {
    "MIattack": MI_attack,
    "TrendAttack": TrendAttack,
    "MRattack": MRattack,
}

STANDARD_GNN = {"GCN": GCNNet, "GAT": GATNet, "GIN": GINNet}
DELETE_GNN = {"GCN": GCNDelete, "GAT": GATDelete, "GIN": GINDelete}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GNN unlearning methods against a gold-standard model."
    )
    parser.add_argument(
        "--unlearn_ratio", type=float, default=0.1,
        help="Fraction of training nodes/edges to unlearn (e.g. 0.1).",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cora",
        help="Name of the graph dataset.",
    )
    parser.add_argument(
        "--unlearning_methods", type=str, default="MEGU",
        help="Unlearning method to evaluate.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1,
        help="Number of independent runs to average over.",
    )
    parser.add_argument(
        "--unlearn_task", type=str, default="node",
        choices=["node", "edge", "feature"],
        help="Granularity of the unlearning request.",
    )
    parser.add_argument(
        "--base_model", type=str, default="GCN",
        choices=["GCN", "GAT", "GIN"],
        help="Backbone GNN architecture.",
    )
    parser.add_argument(
        "--attack_type", type=str, default=None,
        choices=[None, *ATTACK_MAP.keys()],
        help="Membership-inference attack variant for forgetting evaluation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Return a dict of all file-system paths derived from CLI arguments."""
    dataset = args.dataset_name
    u_ratio = args.unlearn_ratio
    unlearn_ratio_tag = f"ratio_{u_ratio:.2f}"
    unlearn_task = args.unlearn_task
    base_suffix = f"_{args.base_model}" if args.base_model != "GCN" else ""

    # Map GNNDelete variant to its internal name
    method = (
        "gnndelete_nodeemb" if args.unlearning_methods == "GNNDelete"
        else args.unlearning_methods
    )

    data_path = (
        PROCESSED_DATA_DIR / f"{dataset}0.8_0_0.2.pkl"
    )
    original_model_path = (
        MODEL_DIR / f"{dataset}/{unlearn_task}/{args.base_model}"
        if method != "Projector"
        else MODEL_DIR / f"{dataset}/Projector.pt"
    )
    gold_model_path = (
        MODEL_DIR / f"{dataset}/Projector.pt"
        if method == "Projector"
        else UNLEARNED_MODEL_DIR
        / f"GOLD/{dataset}/{unlearn_task}/{unlearn_ratio_tag}"
        / f"GOLD_{dataset}_node_{unlearn_ratio_tag}{base_suffix}.pt"
    )

    if method == "Cognac":
        dataset_label = "Cora" if dataset == "cora" else dataset
        unlearn_model_path = (
            COGNAC_DIR
            / f"GCNNet3_{dataset_label}_edge_0.1_0_run0_cacdc_unlearnt_model.pt"
        )
    elif method == "ETR":
        unlearn_model_path = (
            ETR_DIR / f"{dataset}/unlearned_model_GCNNet3_{dataset}_seed0.pt"
        )
    else:
        unlearn_model_path = (
            UNLEARNED_MODEL_DIR
            / f"{method}/{dataset}/{unlearn_task}/{unlearn_ratio_tag}"
            / f"{method}_{dataset}_node_{unlearn_ratio_tag}{base_suffix}.pt"
        )

    return {
        "data": data_path,
        "original_model": original_model_path,
        "gold_model": gold_model_path,
        "unlearn_model": unlearn_model_path,
        "method": method,
        "base_suffix": base_suffix,
        "unlearn_ratio_tag": unlearn_ratio_tag,
    }


def gold_model_path_for_run(
    paths: Dict, dataset: str, unlearn_task: str, run: int
) -> Path:
    """Return the per-run gold model path when ``num_runs > 1``."""
    ratio_tag = paths["unlearn_ratio_tag"]
    base_suffix = paths["base_suffix"]
    method = paths["method"]

    if method == "Projector":
        return MODEL_DIR / f"{dataset}/Projector_GOLD_{run}.pt"
    return (
        UNLEARNED_MODEL_DIR
        / f"GOLD/{dataset}/{unlearn_task}/{ratio_tag}"
        / f"GOLD_{dataset}_node_{ratio_tag}_{run}{base_suffix}.pt"
    )


# ---------------------------------------------------------------------------
# Unlearn-index helpers
# ---------------------------------------------------------------------------

def load_node_unlearn_indices(
    dataset: str, u_ratio: float, run: int, num_nodes: int
) -> List[int]:
    path = (
        UNLEARN_TASK_DIR
        / f"unlearning_nodes_{u_ratio}_{dataset}_{run}"
          f"_nodes_{int(u_ratio * num_nodes)}.txt"
    )
    return list(map(int, path.read_text().splitlines()))


def compute_hop_masks(
    unlearned_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boolean node masks for 1-hop and 2-hop neighborhoods."""
    def _hop_nodes(hops: int) -> torch.Tensor:
        _, _, _, edge_mask = k_hop_subgraph(
            unlearned_nodes, num_hops=hops,
            edge_index=edge_index, num_nodes=num_nodes,
            relabel_nodes=False,
        )
        return edge_index[:, edge_mask].flatten().unique()

    one_hop_nodes = _hop_nodes(1)
    two_hop_nodes = _hop_nodes(2)

    mask_1hop = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    mask_2hop = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    mask_1hop[one_hop_nodes] = True
    mask_2hop[two_hop_nodes] = True
    mask_2hop[unlearned_nodes] = False  # exclude the unlearned nodes themselves

    return mask_1hop, mask_2hop


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_model_args(dataset: str, data, base_model: str) -> dict:
    return {
        "dataset_name": dataset,
        "downstream_task": "node",
        "base_model": base_model,
        "hidden_dim": 64,
        "out_dim": int(data.y.max().item()) + 1,
    }


def _load_state_dict(checkpoint_path: Path) -> dict:
    """Load a checkpoint and return its state-dict regardless of format."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    return ckpt  # assume it is already a state-dict


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def get_logits(
    model_path: Path,
    *,
    model_type: str,
    data,
    args_cli: argparse.Namespace,
    model_args: dict,
    run_number: int = 0,
    unlearned_param_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Load a GNN model from *model_path* and return raw logits for every node.

    Parameters
    ----------
    model_path:
        Path to the saved checkpoint.
    model_type:
        Identifier string (``"GOLD"``, ``"MEGU"``, ``"GNNDelete"``, …).
    data:
        PyG ``Data`` object (already on ``DEVICE``).
    args_cli:
        Parsed CLI arguments.
    model_args:
        Dict passed to GNN constructors.
    run_number:
        Index of the current evaluation run.
    unlearned_param_path:
        For GIF/IDEA: path to the serialised parameter list.
    """
    copy_data = copy.deepcopy(data)
    num_nodes = copy_data.x.size(0)
    edge_index = copy_data.edge_index
    u_ratio = args_cli.unlearn_ratio
    unlearn_task = args_cli.unlearn_task
    base_suffix = f"_{args_cli.base_model}" if args_cli.base_model != "GCN" else ""
    num_runs = args_cli.num_runs

    # ------------------------------------------------------------------
    # Resolve hop masks (needed by some methods)
    # ------------------------------------------------------------------
    mask_1hop = mask_2hop = None

    if unlearn_task in ("node", "feature"):
        unlearned_indices = load_node_unlearn_indices(
            args_cli.dataset_name, u_ratio, run_number, num_nodes
        )
        unlearned_nodes = torch.tensor(unlearned_indices, device=DEVICE)
        mask_1hop, mask_2hop = compute_hop_masks(unlearned_nodes, edge_index, num_nodes)

    elif unlearn_task == "edge":
        edge_index_np = edge_index.cpu().numpy()
        unlearn_edge_path = (
            UNLEARN_TASK_DIR
            / f"unlearning_edges_{u_ratio}_{args_cli.dataset_name}_{run_number}"
              f"_edges_{edge_index_np[0].size}.txt"
        )
        unlearning_edges = torch.tensor(
            np.loadtxt(unlearn_edge_path, dtype=int),
            dtype=torch.long, device=DEVICE,
        )
        # Use `unlearned_nodes` consistently so the Projector branch below can reference it
        unlearned_nodes = unlearning_edges.flatten().unique()
        mask_1hop, mask_2hop = compute_hop_masks(unlearned_nodes, edge_index, num_nodes)

    # ------------------------------------------------------------------
    # Projector (handled separately due to its unique data-loading flow)
    # ------------------------------------------------------------------
    if model_type == "Projector":
        return _get_projector_logits(model_path, copy_data, data, unlearned_nodes, args_cli)

    # ------------------------------------------------------------------
    # Build model skeleton
    # ------------------------------------------------------------------
    is_delete = model_type in ("GNNDelete", "gnndelete_nodeemb")
    cls_map = DELETE_GNN if is_delete else STANDARD_GNN
    model = cls_map[args_cli.base_model](model_args, data.num_node_features, model_args["out_dim"])
    model = model.to(DEVICE)

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    if model_type in ("GIF", "IDEA"):
        assert unlearned_param_path is not None, "GIF/IDEA require unlearned_param_path"
        params = torch.load(unlearned_param_path, map_location=DEVICE)
        for p, saved in zip(model.parameters(), params):
            p.data = saved.to(p.device)

        save_dir = Path(unlearned_param_path).parent
        run_tag = f"_{run_number}" if num_runs > 1 else ""
        copy_data.x_unlearn = torch.load(
            save_dir / f"x_unlearn{run_tag}{base_suffix}.pt"
        ).to(DEVICE)
        copy_data.edge_index_unlearn = torch.load(
            save_dir / f"edge_index_unlearn{run_tag}{base_suffix}.pt"
        ).to(DEVICE)
    else:
        model.load_state_dict(_load_state_dict(model_path), strict=True)

    model.eval()
    with torch.no_grad():
        if is_delete:
            logits = model(data.x, data.edge_index, mask_1hop=mask_1hop, mask_2hop=mask_2hop)

        elif model_type == "GIF":
            logits = model.reason_once_unlearn(copy_data)

        elif model_type == "IDEA":
            if args_cli.base_model == "GCN":
                _, edge_weight = gcn_norm(
                    copy_data.edge_index_unlearn,
                    edge_weight=None,
                    num_nodes=copy_data.x.size(0),
                    add_self_loops=False,
                )
                logits = model.forward_once_unlearn(copy_data, edge_weight)
            else:
                logits = model.reason_once_unlearn(copy_data)

        else:  # GOLD, MEGU, and other standard methods
            logits = model(copy_data.x, copy_data.edge_index)

    return logits


def _get_projector_logits(
    model_path: Path,
    copy_data,
    data,
    unlearned_nodes: torch.Tensor,
    args_cli: argparse.Namespace,
) -> torch.Tensor:
    """Inference path for the Projector-based unlearning method."""
    # Augment node features with an 'is-unlearned' flag
    extra = torch.zeros(copy_data.x.size(0), device=DEVICE)
    extra[unlearned_nodes] = 1.0
    copy_data.x = torch.cat([copy_data.x, extra.unsqueeze(-1)], dim=1)

    copy_data.node_inds = torch.arange(copy_data.x.size(0))
    copy_data.y[unlearned_nodes] = copy_data.num_classes
    copy_data.adj_t = torch_sparse.fill_diag(
        SparseTensor(
            row=copy_data.edge_index[1], col=copy_data.edge_index[0]
        ).to_symmetric(),
        1,
    )
    # Bug fix: use copy_data.y (not bare `data.y`) — copy_data is our working copy
    copy_data.y_one_hot_train = F.one_hot(
        data.y.squeeze(), copy_data.num_classes + 1
    ).float()
    copy_data.y_one_hot_train[copy_data.test_indices, :] = 0

    state_dict = _load_state_dict(model_path)
    y_dims = state_dict["W"].shape[1]
    x_dims = copy_data.x.size(1)

    projector_args = {
        "dataset_name": args_cli.dataset_name,
        "downstream_task": "node",
        "base_model": args_cli.base_model,
        "hidden_dim": 64,
        "out_dim": y_dims,
        "x_iters": 3,
        "y_iters": 3,
        "use_adapt_gcs": False,
        "use_cross_entropy": True,
    }
    model = Pro_GNN(x_dims, y_dims, device=DEVICE, args=projector_args).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    copy_data.y_one_hot_train = F.one_hot(copy_data.y, y_dims).float().to(DEVICE)

    loader = ShaDowKHopSampler(
        copy_data, depth=2, num_neighbors=20, batch_size=1024, shuffle=False
    )
    all_preds: List[torch.Tensor] = []
    with torch.no_grad():
        for subgraph in loader:
            subgraph.adj_t = SparseTensor(
                row=subgraph.edge_index[1], col=subgraph.edge_index[0]
            )
            subgraph.y_one_hot_train = copy_data.y_one_hot_train[subgraph.node_inds]
            all_preds.append(model(subgraph.to(DEVICE)).detach().cpu())

    return torch.cat(all_preds, dim=0)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def node_accuracy(y_true: np.ndarray, preds: np.ndarray, mask: np.ndarray) -> float:
    return float(accuracy_score(y_true[mask], preds[mask]))


def node_f1(y_true: np.ndarray, preds: np.ndarray, mask: np.ndarray) -> float:
    return float(f1_score(y_true[mask], preds[mask], average="micro"))


def mean_std(values: List[float]) -> str:
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def pairwise_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    """
    Returns
    -------
    exact_match : float
        Fraction of nodes where both models predict the same class.
    l2_dist : float
        Mean per-node L2 distance between raw logits.
    """
    # Align output dimensions (e.g. Projector may have extra classes)
    if logits_a.size(1) != logits_b.size(1):
        min_dim = min(logits_a.size(1), logits_b.size(1))
        logits_a, logits_b = logits_a[:, :min_dim], logits_b[:, :min_dim]

    logits_a, logits_b = logits_a.to(DEVICE), logits_b.to(DEVICE)

    if mask is not None:
        exact = float(np.mean(preds_a[mask] == preds_b[mask]))
        l2 = torch.norm(logits_a[mask] - logits_b[mask], p=2, dim=1).mean().item()
    else:
        exact = float(np.mean(preds_a == preds_b))
        l2 = torch.norm(logits_a - logits_b, p=2, dim=1).mean().item()

    return exact, l2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(
    method_name: str,
    acc_runs: Dict[str, List[float]],
    f1_runs: Dict[str, List[float]],
    pairwise: Dict[str, Dict[str, Dict[str, List[float]]]],
    attack_auc_runs: List[float],
    attack_type: Optional[str],
) -> None:
    separator = "=" * 60

    print(f"\n{separator}")
    print("Aggregate Results over all runs")
    print(separator)
    for model_label, accs in acc_runs.items():
        print(f"  [{model_label}]  Accuracy: {mean_std(accs)}")
    for model_label, f1s in f1_runs.items():
        print(f"  [{model_label}]  Micro-F1: {mean_std(f1s)}")

    print(f"\n{separator}")
    print("Pairwise Fidelity (exact-match rate, mean ± std)")
    print(separator)
    for pair, regions in pairwise["exact"].items():
        print(f"  {pair}:")
        for region, vals in regions.items():
            print(f"    {region}: {mean_std(vals)}")

    print(f"\n{separator}")
    print("Pairwise Logit L2 Distance (mean ± std)")
    print(separator)
    for pair, regions in pairwise["l2"].items():
        print(f"  {pair}:")
        for region, vals in regions.items():
            print(f"    {region}: {mean_std(vals)}")

    if attack_type and attack_auc_runs:
        print(f"\n{separator}")
        print(f"{attack_type} Attack AUROC (mean ± std)")
        print(separator)
        print(f"  {mean_std(attack_auc_runs)}")

    print(separator)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    paths = build_paths(args)

    dataset = args.dataset_name
    u_ratio = args.unlearn_ratio
    unlearn_task = args.unlearn_task
    method = paths["method"]

    # ------------------------------------------------------------------
    # Load graph data
    # ------------------------------------------------------------------
    with open(paths["data"], "rb") as fh:
        data = pickle.load(fh)
    data = data.to(DEVICE)

    y_true = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    num_nodes = data.x.size(0)

    model_args = _build_model_args(dataset, data, args.base_model)

    # ------------------------------------------------------------------
    # Original (pre-unlearning) model — loaded once
    # ------------------------------------------------------------------
    logits_original = get_logits(
        paths["original_model"],
        model_type="Original",
        data=data,
        args_cli=args,
        model_args=model_args,
    )
    preds_original = logits_original.argmax(dim=1).cpu().numpy()
    probs_original = F.softmax(logits_original, dim=-1)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    acc_runs: Dict[str, List[float]] = {"Original": [], "GOLD": [], method: []}
    f1_runs: Dict[str, List[float]] = {"Original": [], "GOLD": [], method: []}
    attack_auc_runs: List[float] = []

    # Pairwise metric storage: metric_type → pair_label → region_label → [values]
    pairwise: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "exact": {}, "l2": {}
    }

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    for run in range(args.num_runs):
        # Resolve per-run gold path when doing multiple runs
        gold_path = (
            gold_model_path_for_run(paths, dataset, unlearn_task, run)
            if args.num_runs > 1
            else paths["gold_model"]
        )

        logits_gold = get_logits(
            gold_path, model_type="GOLD",
            data=data, args_cli=args, model_args=model_args, run_number=run,
        )
        logits_unlearn = get_logits(
            paths["unlearn_model"], model_type=method,
            data=data, args_cli=args, model_args=model_args, run_number=run,
            unlearned_param_path=paths["unlearn_model"],
        )

        preds_gold = logits_gold.argmax(dim=1).cpu().numpy()
        preds_unlearn = logits_unlearn.argmax(dim=1).cpu().numpy()
        probs_unlearn = F.softmax(logits_unlearn, dim=-1)

        # ---- Accuracy & F1 -------------------------------------------
        for label, preds in [
            ("Original", preds_original),
            ("GOLD", preds_gold),
            (method, preds_unlearn),
        ]:
            acc_runs[label].append(node_accuracy(y_true, preds, test_mask))
            f1_runs[label].append(node_f1(y_true, preds, test_mask))

        # ---- Membership-inference attack (nodes only) -----------------
        if args.attack_type and unlearn_task == "node":
            attack_fn = ATTACK_MAP[args.attack_type]
            auc = attack_fn(
                probs_original,
                probs_unlearn,
                data,
                train_mask,
                test_mask,
                unlearned_indices=None,
                run_number=run,
                u_ratio=u_ratio,
                dataset=dataset,
                order=2,
                train_attack=True,
                verbose=True,
            )
            attack_auc_runs.append(auc)

        # ---- Pairwise metrics ----------------------------------------
        if unlearn_task == "node":
            # Load indices only to stay consistent with the run; mask is available
            # for future use if additional regions (e.g. retained train) are re-enabled.
            unlearned_indices = load_node_unlearn_indices(
                dataset, u_ratio, run, num_nodes
            )
        regions = [("Test Nodes", test_mask)]

        comparisons = {
            "Gold vs Original":          (logits_gold, logits_original, preds_gold, preds_original),
            f"Gold vs {method}":         (logits_gold, logits_unlearn, preds_gold, preds_unlearn),
            f"Original vs {method}":     (logits_original, logits_unlearn, preds_original, preds_unlearn),
        }

        for pair, (la, lb, pa, pb) in comparisons.items():
            pairwise["exact"].setdefault(pair, {r: [] for r, _ in regions})
            pairwise["l2"].setdefault(pair, {r: [] for r, _ in regions})
            for region, mask in regions:
                exact, l2 = pairwise_metrics(la, lb, pa, pb, mask)
                pairwise["exact"][pair][region].append(exact)
                pairwise["l2"][pair][region].append(l2)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print_results(method, acc_runs, f1_runs, pairwise, attack_auc_runs, args.attack_type)


if __name__ == "__main__":
    main()
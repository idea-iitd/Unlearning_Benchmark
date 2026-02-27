import numpy as np
from sklearn.metrics import roc_auc_score
import os

def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(x)

def MI_attack(
    original_probs,
    unlearn_probs,
    data=None,
    train_mask=None,
    test_mask=None,
    unlearned_indices=None,
    run_number: int = 0,
    u_ratio: float = None,
    dataset: str = None,
    order: int = 2,
    train_attack: bool = True,
    mia_num: int = None,
    verbose: bool = False,
):
    """
    Compute MIA AUROC (balanced members vs non-members) using Lp distance between softlabels.

    Returns:
      auroc (float) if verbose==False; otherwise (auroc, scores, labels, meta)
    """
    orig = _to_numpy(original_probs)
    unlk = _to_numpy(unlearn_probs)
    if orig.shape != unlk.shape:
        raise ValueError("original_probs and unlearn_probs must have same shape")

    N = orig.shape[0]
    def mask_to_idx(m):
        if m is None: return np.array([], dtype=int)
        m = np.array(m)
        return np.flatnonzero(m) if m.dtype == bool else np.array(m, dtype=int)

    train_idx = mask_to_idx(train_mask)
    test_idx = mask_to_idx(test_mask)

    # Try load unlearned_indices if not provided (use same path pattern from your top-level script)
    if unlearned_indices is None:
        try:
            # mirror path you used earlier in script
            num_nodes = data.x.size(0)
            if u_ratio is None or dataset is None:
                raise FileNotFoundError()
            unlearn_idx_path = f"/data/unlearning_task/transductive/imbalanced/unlearning_nodes_{u_ratio}_{dataset}_{run_number}_nodes_{int(u_ratio * num_nodes)}.txt"
            if os.path.exists(unlearn_idx_path):
                with open(unlearn_idx_path, "r") as f:
                    unlearned_indices = list(map(int, f.readlines()))
                # if verbose:
                #     print(f"[MRA] Loaded unlearn list from {unlearn_idx_path} ({len(unlearned_indices)} nodes)")
            else:
                unlearned_indices = None
        except Exception:
            unlearned_indices = None

    mia_num = len(unlearned_indices)

    members = np.array(unlearned_indices, dtype=int)
    nonmembers = np.array(test_idx[:mia_num], dtype=int)
    # compute Lp distance scores
    def score(idx):
        return np.linalg.norm(orig[idx] - unlk[idx])

    scores_members = np.array([score(i) for i in members])
    scores_nonmembers = np.array([score(i) for i in nonmembers])

    scores = np.concatenate([scores_members, scores_nonmembers])
    labels = np.concatenate([np.ones_like(scores_members), np.zeros_like(scores_nonmembers)])

    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")

    meta = {"run_number": run_number, "u_ratio": u_ratio, "dataset": dataset, "order": order, "mia_num": mia_num}

    return auc

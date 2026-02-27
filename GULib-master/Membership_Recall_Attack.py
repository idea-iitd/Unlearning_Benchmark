# MR_attack.py
# Membership Recall Attack (MRA) adapted from the uploaded paper for node-level data.
# Usage: from MR_attack import MRattack
# Then call with the same signature you used previously for TrendAttack.

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Defaults taken from the paper (adapted for node features)
DEFAULTS = {
    "KE": 3,        # outer epochs
    "KD": 3,        # denoising KD iterations per KE
    "KR": 2,        # recall iterations per KE
    "alpha_x": 0.20,
    "alpha_y": 0.75,
    "gamma_l": 0.01,   # Laplace smoothing
    "gamma_s": 0.05,   # label smoothing in top-k
    "tau": 0.6,        # fraction for top-K selection (paper uses 0.6 for CIFAR; adjust per dataset)
    "batch_size": 256,
    "student_lr": 1e-4,
    "warmup_epochs": 1,  # for beta_y=1 in first warmup epoch
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def _sample_beta(alpha, size):
    if alpha <= 0:
        return torch.ones(size)
    return torch.from_numpy(np.random.beta(alpha, alpha, size=size)).float()


class StudentMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def _mixup_batch(x1, x2, y1, y2, alpha_x, alpha_y, warmup=False):
    # x1,x2: [B, D], y1,y2: [B, C] (soft labels)
    B = x1.size(0)
    betas_x = _sample_beta(alpha_x, B).to(x1.device).unsqueeze(1)
    if warmup:
        # paper uses beta_y=1 for warmup: accept teacher fully
        betas_y = torch.ones(B, device=x1.device).unsqueeze(1)
    else:
        betas_y = _sample_beta(alpha_y, B).to(x1.device).unsqueeze(1)
    x_tilde = betas_x * x1 + (1.0 - betas_x) * x2
    y_u_tilde = betas_y * y1 + (1.0 - betas_y) * y2
    return x_tilde, y_u_tilde, betas_y.squeeze(1)


def _kl_loss_with_soft_targets(logits_pred, soft_targets):
    # logits_pred: raw logits, soft_targets: probabilites (sum to 1)
    logp = F.log_softmax(logits_pred, dim=-1)
    # KLDivLoss expects input=log_probs, target=probs
    return F.kl_div(logp, soft_targets, reduction="batchmean")


def _ce_with_soft_targets(logits_pred, soft_targets):
    # Equivalent to cross-entropy with soft targets via KLDiv with teacher probs
    return _kl_loss_with_soft_targets(logits_pred, soft_targets)


def MRattack(
    original_probs,
    probs,            # torch.Tensor [N, C] : logits from unlearned model (teacher)
    data,                  # PyG data object (has x, y etc.)
    train_mask,            # numpy or tensor boolean mask
    test_mask,             # numpy or tensor boolean mask
    unlearned_indices=None,
    run_number=0,
    u_ratio=None,
    dataset=None,
    order=2,
    train_attack=True,
    verbose=True,
    # optional overrides
    KE=None, KD=None, KR=None, alpha_x=None, alpha_y=None, gamma_l=None, gamma_s=None, tau=None,
):
    """
    Adapted MRA attack. Returns (attack_auc, attack_acc).
    attack_acc: accuracy of recalled labels on unlearned nodes (if unlearned list available; else np.nan)
    attack_auc: ROC AUC using joint-confidence scores to separate unlearned vs retained nodes (if unlearn list available; else np.nan)
    """

    device = DEFAULTS["device"]
    KE = DEFAULTS["KE"] if KE is None else KE
    KD = DEFAULTS["KD"] if KD is None else KD
    KR = DEFAULTS["KR"] if KR is None else KR
    alpha_x = DEFAULTS["alpha_x"] if alpha_x is None else alpha_x
    alpha_y = DEFAULTS["alpha_y"] if alpha_y is None else alpha_y
    gamma_l = DEFAULTS["gamma_l"] if gamma_l is None else gamma_l
    gamma_s = DEFAULTS["gamma_s"] if gamma_s is None else gamma_s
    tau = DEFAULTS["tau"] if tau is None else tau
    batch_size = DEFAULTS["batch_size"]

    # Prepare tensors
    x = data.x.to(device)
    y_true = data.y.cpu().numpy()
    N = x.size(0)
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)
    probs = probs.to(device)
    C = probs.size(1)

    # Build prediction set Xp: as in paper Xp = test ∪ (forgotten subset). Here use test_mask ∪ train_mask (or whole nodes)
    # We'll use nodes that are in train_mask OR test_mask (same as many of your flows)
    if isinstance(train_mask, torch.Tensor):
        train_mask_np = train_mask.cpu().numpy()
    else:
        train_mask_np = np.array(train_mask, dtype=bool)
    if isinstance(test_mask, torch.Tensor):
        test_mask_np = test_mask.cpu().numpy()
    else:
        test_mask_np = np.array(test_mask, dtype=bool)

    xp_mask = np.logical_or(train_mask_np, test_mask_np)
    xp_idx = np.where(xp_mask)[0]
    if len(xp_idx) == 0:
        xp_idx = np.arange(N)  # fallback: all nodes

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

    # Teacher probabilities (already softmaxed by caller)
    probs_teacher = probs.detach()  # [N, C]

    # Student model
    student = StudentMLP(in_dim=x.size(1), out_dim=C, hidden=128).to(device)
    optim = torch.optim.AdamW(student.parameters(), lr=DEFAULTS["student_lr"], weight_decay=0)

    # Create dataset for Xp
    xp_tensor = x[xp_idx].to(device)
    teacher_probs_xp = probs_teacher[xp_idx].to(device)

    # DataLoader indices for mixup batches (we will sample pairs randomly)
    dataset_tensor = TensorDataset(xp_tensor, teacher_probs_xp)
    dl = DataLoader(dataset_tensor, batch_size=min(batch_size, len(xp_idx)), shuffle=True)

    # warmup flag for beta_y = 1 first epoch
    for epoch in range(KE):
        warmup = (epoch < DEFAULTS["warmup_epochs"])

        # (1) Denoising Knowledge Distillation (KD iterations)
        for kd in range(KD):
            for (bx, by) in dl:
                B = bx.size(0)
                # sample partner indices
                idx2 = torch.randint(0, len(xp_idx), (B,), device=device)
                bx2 = xp_tensor[idx2]
                by2 = teacher_probs_xp[idx2]

                x_tilde, y_u_tilde, betas_y = _mixup_batch(bx, bx2, by, by2, alpha_x, alpha_y, warmup=warmup)
                # student prediction on mixed features
                logits_s = student(x_tilde)
                probs_s = F.softmax(logits_s, dim=-1).detach()
                # y_tilde_US = beta_y * y_u_tilde + (1-beta_y) * y_s (per paper)
                betas_y = betas_y.to(device).unsqueeze(1)
                y_tilde_us = betas_y * y_u_tilde + (1.0 - betas_y) * probs_s
                # Loss: CE between student(logits_s) and y_tilde_us (soft targets)
                loss = _ce_with_soft_targets(logits_s, y_tilde_us)
                optim.zero_grad()
                loss.backward()
                optim.step()

        # (2) Confident Membership Recall (KR iterations)
        # compute latest student predictions on Xp
        with torch.no_grad():
            logits_student_xp = student(xp_tensor)
            probs_student_xp = F.softmax(logits_student_xp, dim=-1)

        # Laplace smoothing
        # y_tilde = (y + gamma_l) / (1 + C*gamma_l)
        probs_teacher_sm = (teacher_probs_xp + gamma_l) / (1.0 + C * gamma_l)
        probs_student_sm = (probs_student_xp + gamma_l) / (1.0 + C * gamma_l)

        # joint matrix
        joint = probs_teacher_sm * probs_student_sm  # [N_xp, C]

        N_xp = joint.size(0)
        K = math.ceil(tau * N_xp / C)
        if K < 1:
            K = 1

        # select top-K per class
        chosen_indices = []
        chosen_soft_targets = []
        for c in range(C):
            col = joint[:, c].cpu().numpy()
            topk_idx = np.argsort(-col)[:K]  # indices within xp_tensor
            for idx in topk_idx:
                # build label ycf per paper: y_tilde_c = (1-gamma_s) * one_hot(c) + gamma_s / K * 1
                one_hot = np.zeros(C, dtype=float)
                one_hot[c] = 1.0
                ycf = (1.0 - gamma_s) * one_hot + (gamma_s / C) * np.ones(C, dtype=float)
                chosen_indices.append(int(xp_idx[idx]))  # convert to absolute node index
                chosen_soft_targets.append(ycf)

        if len(chosen_indices) == 0:
            # nothing selected; skip recall updates
            if verbose:
                print("[MRA] No confident samples selected in recall step; skipping.")
        else:
            # unique them to avoid duplicates; aggregate by averaging soft targets if duplicates
            idx_to_targets = {}
            for idx_abs, soft in zip(chosen_indices, chosen_soft_targets):
                if idx_abs not in idx_to_targets:
                    idx_to_targets[idx_abs] = []
                idx_to_targets[idx_abs].append(soft)
            final_idxs = []
            final_targets = []
            for k_idx, list_targets in idx_to_targets.items():
                final_idxs.append(k_idx)
                avg_target = np.mean(np.stack(list_targets, axis=0), axis=0)
                final_targets.append(avg_target)
            final_idxs = np.array(final_idxs, dtype=int)
            final_targets = np.stack(final_targets, axis=0).astype(np.float32)

            # form tensors
            X_cf = x[final_idxs].to(device)
            Y_cf = torch.from_numpy(final_targets).to(device)

            # KR iterations of updating student on (X_cf, Y_cf)
            for kr in range(KR):
                # simple full-batch update (X_cf small usually)
                logits_cf = student(X_cf)
                loss_cf = _ce_with_soft_targets(logits_cf, Y_cf)
                optim.zero_grad()
                loss_cf.backward()
                optim.step()

            # (Optional) if teacher model is accessible/trainable (open-source case), update teacher here.
            # In your pipeline we only receive probs, so teacher update is not possible.
            # If you have teacher model object, you can fine tune it similarly.

    # After training, compute recalled labels for whole node set (or xp set)
    with torch.no_grad():
        logits_student_all = student(x.to(device))
        probs_student_all = F.softmax(logits_student_all, dim=-1)
        # compute joint for all nodes
        probs_teacher_all = probs_teacher
        probs_teacher_sm_all = (probs_teacher_all + gamma_l) / (1.0 + C * gamma_l)
        probs_student_sm_all = (probs_student_all + gamma_l) / (1.0 + C * gamma_l)
        joint_all = probs_teacher_sm_all * probs_student_sm_all  # [N, C]
        # for each node, joint confidence = max_c joint_all[node,c]
        joint_confidence = joint_all.max(dim=1)[0].cpu().numpy()
        pred_student = probs_student_all.argmax(dim=1).cpu().numpy()

    # Compute attack metrics if unlearned_indices known
    attack_acc = np.nan
    attack_auc = np.nan
    if unlearned_indices is not None and len(unlearned_indices) > 0:
        unlearn_mask = np.zeros(N, dtype=bool)
        unlearn_mask[unlearned_indices] = True
        # Attack accuracy: how well student predicted true labels on unlearned nodes
        try:
            attack_acc = accuracy_score(y_true[unlearn_mask], pred_student[unlearn_mask])
        except Exception:
            attack_acc = np.nan

        # Build binary labels (1 if unlearned, 0 otherwise) and scores = joint_confidence
        try:
            y_bin = unlearn_mask.astype(int)
            if len(np.unique(y_bin)) == 1:
                attack_auc = float('nan')  # AUROC undefined if only one class present
            else:
                attack_auc = roc_auc_score(y_bin, joint_confidence)
        except Exception:
            attack_auc = np.nan

    # if verbose:
    #     print(f"[MRA] Finished. attack_acc={attack_acc}, attack_auc={attack_auc}")

    return float(attack_auc) if not (attack_auc is np.nan) else np.nan
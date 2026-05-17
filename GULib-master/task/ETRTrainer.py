"""
task/ETRTrainer.py
==================
ETR (Erase Then Rectify) trainer for GULib benchmark.

Architecture mirrors MEGUTrainer / CognacTrainer / IDEATrainer:
  - Inherits GULib's BaseTrainer.
  - All args accessed as dict: self.args["key"].
  - Entry-point: etr_unlearning(temp_node, run) — called by the etr
    pipeline class in unlearning/unlearning_methods/ETR/etr.py.
  - Model is saved to the standard GULib path at the end of unlearning,
    exactly as MEGUTrainer.megu_unlearning() / CognacTrainer do.

The pipeline class (etr.py) is responsible for building the subgraphs
ETR needs and attaching them via attach_subgraphs() BEFORE calling
etr_unlearning().

Reference:
  Zhang et al., "Erase then Rectify: A Training-Free Parameter Editing
  Approach for Cost-Effective Graph Unlearning", AAAI 2025.
  https://arxiv.org/pdf/2409.16684
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from task.BaseTrainer import BaseTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ETRTrainer(BaseTrainer):
    """
    ETR unlearning trainer, conforming to the GULib BaseTrainer interface.

    Constructor matches all other *Trainer classes in task/:
        ETRTrainer(args, logger, model, data)
    where args is a dict (GULib convention).
    """

    def __init__(self, args, logger, model, data):
        super().__init__(args, logger, model, data)
        # Placeholders — pipeline attaches these before etr_unlearning()
        self.dataset_train = None
        self.dataset_train_unlearned = None
        self.dataset_unlearned = None
        self.unlearn_subgraph = None
        self.influenced_subgraph = None
        self.influenced_subgraph_unlearned = None
        self.influenced_nodes = None
        self.unlearning_id = None

    # ------------------------------------------------------------------
    # Pipeline hook — attach subgraphs built by etr.py
    # ------------------------------------------------------------------
    def attach_subgraphs(
        self,
        dataset_train,
        dataset_train_unlearned,
        dataset_unlearned,
        influenced_subgraph,
        influenced_subgraph_unlearned,
        influenced_nodes,
        unlearn_subgraph=None,
        unlearning_id=None,
    ):
        self.dataset_train = dataset_train
        self.dataset_train_unlearned = dataset_train_unlearned
        self.dataset_unlearned = dataset_unlearned
        self.influenced_subgraph = influenced_subgraph
        self.influenced_subgraph_unlearned = influenced_subgraph_unlearned
        self.influenced_nodes = influenced_nodes
        self.unlearn_subgraph = unlearn_subgraph
        self.unlearning_id = unlearning_id

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------
    def etr_unlearning(self, temp_node, run=0):
        """
        Run ETR's erase-then-rectify unlearning.

        Parameters
        ----------
        temp_node : np.ndarray
            Global node indices whose influence should be removed.
        run : int
            Run index used for the save-path filename.

        Returns
        -------
        unlearn_time : float
        test_f1      : float
        """
        assert self.dataset_train is not None, (
            "[ETR] Pipeline must call attach_subgraphs() before etr_unlearning()."
        )

        self.model = self.model.to(device)

        # Deep-copy: pipeline already captured pre-unlearning softlabels
        # from self.model in etr.unlearn() before calling us.
        model_unlearn = copy.deepcopy(self.model)

        # Adam is used only for zero_grad() — weights are edited in-place
        # by the modify_weight* helpers, not by optimizer.step().
        opt = torch.optim.Adam(
            model_unlearn.parameters(),
            lr=self.args.get("etr_lr", 1e-3),
            weight_decay=self.args.get("etr_wd", 1e-4),
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

        erase_ratio = float(self.args.get("etr_erase_ratio", 0.01))
        l_coef = float(self.args.get("etr_l", 0.3))
        task = self.args["unlearn_task"]

        # ---- Step 1: gradients on the full training graph ----
        grad_all, imp_all = self._grad_and_info(
            model_unlearn, self.dataset_train, opt, criterion, batch=False
        )

        start = time.time()

        # ---- Step 2: unlearn-subgraph gradients (node task only) ----
        grad_f = imp_unlearn = None
        if task == "node":
            grad_f, imp_unlearn = self._grad_and_info(
                model_unlearn, self.unlearn_subgraph, opt, criterion, batch=True
            )

        # ---- Step 3: influenced-subgraph gradients ----
        grad_k, imp_inf = self._grad_and_info(
            model_unlearn, self.influenced_subgraph, opt, criterion, batch=True
        )

        # ---- Erase step ----
        if task == "node":
            self._modify_weight(model_unlearn, imp_all, imp_unlearn, imp_inf, erase_ratio)
        else:
            self._modify_weight_ef(model_unlearn, imp_all, imp_inf, erase_ratio)

        # ---- Step 4: re-compute influenced-subgraph gradients after edit ----
        grad_u, _ = self._grad_and_info(
            model_unlearn,
            self.influenced_subgraph_unlearned,
            opt,
            criterion,
            batch=True,
        )

        # ---- Rectify step ----
        num_train = int(self.dataset_train.train_mask.sum().item())
        i_num = len(self.influenced_nodes)
        if task == "node":
            u_num = len(self.unlearning_id)
            self._modify_weight2(
                model_unlearn, grad_all, grad_f, grad_k, grad_u,
                l_coef, u_num, i_num, num_train,
            )
        else:
            self._modify_weight2_ef(
                model_unlearn, grad_all, grad_k, grad_u,
                l_coef, i_num, num_train,
            )

        unlearn_time = time.time() - start

        # Push edited weights into self.model so the pipeline's
        # mia_attack() and evaluate_unlearning.py see the unlearned model.
        self.model.load_state_dict(model_unlearn.state_dict())

        test_f1 = self._eval_on(self.dataset_unlearned)
        self.logger.info(
            "ETR | run=%d | time=%.4fs | TestF1=%.4f" % (run, unlearn_time, test_f1)
        )

        self._save_unlearned_model(run)
        return unlearn_time, test_f1

    # ------------------------------------------------------------------
    # Gradient / importance helpers
    # ------------------------------------------------------------------
    def _zerolike(self, model):
        return {k: torch.zeros_like(p, device=p.device) for k, p in model.named_parameters()}

    def _grad_and_info(self, model, data, optimizer, criterion, batch=False):
        """
        One forward+backward pass; returns (grad_dict, importance_dict).

        batch=False  →  full train graph; loss over data.train_mask.
        batch=True   →  subgraph; loss over the seed nodes only.

        *** CRITICAL INDEXING NOTE ***
        PyG's k_hop_subgraph(relabel_nodes=True) returns `subset` as a
        *sorted* tensor of global node IDs.  Seed nodes appear wherever
        their global IDs sort to — they are NOT guaranteed to occupy the
        first `batch_size` positions.  `data.mapping` stores the exact
        positions of seed nodes within the relabeled subset (as set by
        etr_utils.get_subgraph).

        We therefore index with `data.mapping`, NOT `[:data.batch_size]`.
        Using `[:data.batch_size]` would compute gradients for the
        lowest-indexed neighborhood nodes, not the actual seed nodes,
        silently corrupting both the Erase and Rectify steps.
        """
        model.train()
        parameter_grad = self._zerolike(model)
        parameter_importance = self._zerolike(model)
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data.x, data.edge_index)

        if batch:
            # data.mapping: positions of seed nodes in the relabeled subset.
            seed_idx = data.mapping.to(device)
            y = data.y[seed_idx].squeeze().long()
            loss = criterion(output[seed_idx], y)
        else:
            y = data.y[data.train_mask].squeeze().long()
            loss = criterion(output[data.train_mask], y)

        loss.backward()

        for (_, p), (_, g), (_, h) in zip(
            model.named_parameters(),
            parameter_grad.items(),
            parameter_importance.items(),
        ):
            if p.grad is not None:
                g.data += p.grad.data.clone()
                h.data += p.grad.data.clone().pow(2)

        return parameter_grad, parameter_importance

    # ------------------------------------------------------------------
    # Weight-editing helpers (Erase / Rectify)
    # ------------------------------------------------------------------
    def _modify_weight(self, model, imp_all, imp_u, imp_n, k):
        """Erase step — node task (uses unlearn + influenced importances)."""
        with torch.no_grad():
            for (_, p), (_, oimp), (_, uimp), (_, nimp) in zip(
                model.named_parameters(),
                imp_all.items(),
                imp_u.items(),
                imp_n.items(),
            ):
                b1 = torch.min(torch.nan_to_num(oimp / uimp, nan=1.0), torch.ones_like(oimp))
                b2 = torch.min(torch.nan_to_num(oimp.pow(2) / (uimp * nimp), nan=1.0), torch.ones_like(oimp))
                q1 = torch.quantile(b1, k)
                q2 = torch.quantile(b2, k)
                loc1 = b1 <= q1
                loc2 = (b2 <= q2)
                loc2[loc1] = False
                p[loc1] = p[loc1].mul(b1[loc1] / q1)
                p[loc2] = p[loc2].mul(b2[loc2] / q2)

    def _modify_weight_ef(self, model, imp_all, imp_inf, k):
        """Erase step — edge / feature task (uses influenced importances only)."""
        with torch.no_grad():
            for (_, p), (_, aimp), (_, bimp) in zip(
                model.named_parameters(),
                imp_all.items(),
                imp_inf.items(),
            ):
                b = torch.min(torch.nan_to_num(aimp / bimp, nan=1.0), torch.ones_like(bimp))
                q = torch.quantile(b, k)
                loc = b <= q
                p[loc] = p[loc].mul(b[loc] / q)

    def _modify_weight2(self, model, g_all, g_f, g_k, g_u, l, u_num, i_num, num):
        """Rectify step — node task."""
        m = num - u_num
        with torch.no_grad():
            for (_, p), (_, ap), (_, fp), (_, kp), (_, up) in zip(
                model.named_parameters(),
                g_all.items(),
                g_f.items(),
                g_k.items(),
                g_u.items(),
            ):
                p.sub_(l * (ap * num / m - fp * u_num / m - kp * i_num / m + up * i_num / m))

    def _modify_weight2_ef(self, model, g_all, g_k, g_u, l, i_num, num):
        """Rectify step — edge / feature task."""
        with torch.no_grad():
            for (_, p), (_, ap), (_, kp), (_, up) in zip(
                model.named_parameters(),
                g_all.items(),
                g_k.items(),
                g_u.items(),
            ):
                p.sub_(l * (ap - kp * i_num / num + up * i_num / num))

    # ------------------------------------------------------------------
    # Evaluation and model saving
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_on(self, data):
        """Micro-F1 on test_mask nodes of `data`."""
        data = data.to(device)
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        mask = data.test_mask.to(device)
        pred = torch.argmax(out[mask], dim=1).cpu().numpy()
        true = data.y[mask].squeeze().long().cpu().numpy()
        return f1_score(true, pred, average="micro")

    def _save_unlearned_model(self, run):
        """
        Save to standard GULib path (mirrors MEGUTrainer / CognacTrainer):
            unlearned_models/ETR/{dataset}/{unlearn_task}/ratio_{r:.2f}/
                ETR_{dataset}_{downstream_task}_ratio_{r:.2f}{run_str}{base_str}.pt
        """
        copy_str = "_copy" if self.args.get("use_copy", False) else ""
        run_str = f"_{run}" if self.args["num_runs"] > 1 else ""
        base_str = "" if self.args["base_model"] == "GCN" else f"_{self.args['base_model']}"
        unlearn_ratio = self.args["unlearn_ratio"]

        save_dir = os.path.join(
            "unlearned_models", "ETR",
            self.args["dataset_name"],
            self.args["unlearn_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}",
        )
        os.makedirs(save_dir, exist_ok=True)

        model_name = (
            f"ETR_{self.args['dataset_name']}_{self.args['downstream_task']}_"
            f"ratio_{unlearn_ratio:.2f}{run_str}{base_str}.pt"
        )
        save_path = os.path.join(save_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"[ETR] Unlearned model saved: {save_path}")
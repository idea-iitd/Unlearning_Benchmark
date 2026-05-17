"""
unlearning/unlearning_methods/ETR/etr.py
========================================
ETR (Erase Then Rectify) unlearning pipeline for GULib.

Structure mirrors megu.py / cognac.py:
  - Inherits Learning_based_pipeline.
  - determine_target_model() sets args["unlearn_trainer"] = "ETRTrainer"
    and calls get_trainer().
  - train_original_model() uses _train_model() helper — same as MEGU.
  - unlearning_request() reads the standard GULib text-file unlearning
    indices (same path convention as MEGU / Cognac / IDEA / GIF) and
    builds the subgraphs ETR needs.
  - unlearn() calls self.target_model.etr_unlearning(temp_node, run).

No gold-standard retraining is produced (matches Cognac's policy).

Reference:
  Zhang et al., "Erase then Rectify: A Training-Free Parameter Editing
  Approach for Cost-Effective Graph Unlearning", AAAI 2025.
  https://arxiv.org/pdf/2409.16684
"""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from pipeline.Learning_based_pipeline import Learning_based_pipeline
from task import get_trainer
from config import BLUE_COLOR, RESET_COLOR
from config import unlearning_path, unlearning_edge_path

from .etr_utils import (
    get_dataset_train,
    get_dataset_unlearn,
    get_influence_nodes,
    get_subgraph,
)


class etr(Learning_based_pipeline):
    """
    ETR pipeline.

    Registered in unlearning_manager.py as:
        method_map["ETR"] = etr

    CLI usage:
        python GULib-master/main.py --unlearning_methods ETR ...
    """

    def __init__(self, args, logger, model_zoo):
        super().__init__(args, logger, model_zoo)
        self.args = args
        self.logger = logger
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self._data = copy.deepcopy(self.data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_feats = self.data.num_features

        num_runs = self.args["num_runs"]
        self.run = 0
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_unlearning_time = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)

        # for MIA (same pattern as MEGU / Cognac)
        self.train_indices = self.data.train_indices
        self.test_indices = self.data.test_indices

    # ----------------------------------------------------------------
    # Hook 1 — mirrors megu.determine_target_model()
    # ----------------------------------------------------------------
    def determine_target_model(self):
        """Construct an ETRTrainer wrapping the base GNN model."""
        self.logger.info("target model: %s" % self.args["base_model"])
        self.args["unlearn_trainer"] = "ETRTrainer"
        self.target_model = get_trainer(
            self.args, self.logger, self.model_zoo.model, self._data
        )

    # ----------------------------------------------------------------
    # Hook 2 — mirrors megu.train_original_model()
    # ----------------------------------------------------------------
    def train_original_model(self):
        """Train (or load from disk) the base GNN on the clean graph."""
        self.logger.info("training target models, run %s" % self.run)
        run_training_time, _ = self._train_model(self.run)
        self.avg_training_time[self.run] = run_training_time

    # ----------------------------------------------------------------
    # Hook 3 — mirrors megu.unlearning_request() / cognac.unlearning_request()
    # ----------------------------------------------------------------
    def unlearning_request(self):
        """
        Read the standard GULib unlearning text file and build the
        four subgraphs ETR needs:
          - dataset_train            (train-only edges of full graph)
          - dataset_train_unlearned  (train-only edges after removing forget set)
          - dataset_unlearned        (full graph after removing forget set)
          - unlearn_subgraph         (k-hop subgraph around unlearn nodes, node task only)
          - influenced_subgraph      (k-hop subgraph around influenced nodes)
          - influenced_subgraph_unlearned (same, using post-unlearn edges)

        Edge file convention: the benchmark saves edge files as [num_edges, 2]
        rows via np.savetxt(..., edges.T).  All other methods (GIF, IDEA,
        GraphEraser, ScaleGUN) load with np.loadtxt(...).T to get back
        [2, num_edges].  We follow the same convention here.
        """
        unlearning_id = None
        delete_edge_index = None

        if self.args["unlearn_task"] == "node":
            path_un = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path_un, dtype=int)
            if unique_nodes.ndim == 0:
                unique_nodes = np.array([unique_nodes.item()])
            unlearning_id = unique_nodes
            self.unlearing_nodes = unique_nodes   # MEGU spelling kept for MIA
            self.temp_node = unique_nodes

        elif self.args["unlearn_task"] == "edge":
            path_un = (
                unlearning_edge_path
                + "_" + str(self.run)
                + "_edges_" + str(self.args["num_unlearned_edges"])
                + ".txt"
            )
            # --- BUG-FIX vs original code ---
            # The file stores rows of (src, dst) — i.e. shape [num_edges, 2]
            # after np.savetxt. All other GULib methods load with .T to get
            # [2, num_edges] (standard edge_index format). We do the same.
            raw = np.loadtxt(path_un, dtype=int)
            if raw.ndim == 1:
                # Single edge: raw is shape [2] → make it [1, 2] before .T
                raw = raw.reshape(1, 2)
            # Now raw is [num_edges, 2]; transpose to [2, num_edges].
            remove_edges = raw.T            # shape [2, num_edges]

            unique_nodes = np.unique(remove_edges)

            # Build (src, dst) pair set for fast membership testing.
            edges_set = set(zip(remove_edges[0].tolist(), remove_edges[1].tolist()))

            # Find positions in the full edge_index that match any removed edge
            # (either direction, since the graph is undirected).
            ei_np = self.data.edge_index.cpu().numpy()
            delete_edge_index = np.array(
                [
                    i for i in range(ei_np.shape[1])
                    if (int(ei_np[0, i]), int(ei_np[1, i])) in edges_set
                    or (int(ei_np[1, i]), int(ei_np[0, i])) in edges_set
                ],
                dtype=int,
            )
            self.unlearing_nodes = unique_nodes
            self.temp_node = unique_nodes

        elif self.args["unlearn_task"] == "feature":
            path_un = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path_un, dtype=int)
            if unique_nodes.ndim == 0:
                unique_nodes = np.array([unique_nodes.item()])
            unlearning_id = unique_nodes
            self.unlearing_nodes = unique_nodes
            self.temp_node = unique_nodes

        else:
            raise ValueError(
                f"[ETR] unsupported unlearn_task: {self.args['unlearn_task']}"
            )

        # ---- Build the subgraphs ----
        dataset_train = get_dataset_train(self.data)

        if self.args["unlearn_task"] == "edge":
            influenced_nodes = get_influence_nodes(
                unique_nodes,
                dataset_train.edge_index,
                hops=1,
                unlearn_task="edge",
            )
            dataset_unlearned = get_dataset_unlearn(
                self.args, self.data, unique_nodes, delete_edge_index
            )
            dataset_train_unlearned = get_dataset_train(dataset_unlearned)
            unlearn_subgraph = None  # not used by the _ef code path
        else:
            # node or feature
            influenced_nodes = get_influence_nodes(
                unique_nodes,
                dataset_train.edge_index,
                hops=2,
                unlearn_task=self.args["unlearn_task"],
            )
            dataset_train_unlearned = get_dataset_unlearn(
                self.args, dataset_train, unique_nodes
            )
            dataset_unlearned = get_dataset_unlearn(
                self.args, self.data, unique_nodes
            )
            if self.args["unlearn_task"] == "node":
                # Mark unlearn nodes as non-training in the modified set
                dataset_train_unlearned.train_mask = dataset_train_unlearned.train_mask.clone()
                dataset_train_unlearned.train_mask[unique_nodes] = False
                unlearn_subgraph = get_subgraph(unique_nodes, dataset_train)
            else:
                unlearn_subgraph = None  # feature task uses the _ef path

        influenced_subgraph = get_subgraph(influenced_nodes, dataset_train)
        influenced_subgraph_unlearned = get_subgraph(
            influenced_nodes, dataset_train_unlearned
        )

        # ---- Hand all subgraphs to the trainer ----
        self.target_model.data = self.data
        self.target_model.attach_subgraphs(
            dataset_train=dataset_train,
            dataset_train_unlearned=dataset_train_unlearned,
            dataset_unlearned=dataset_unlearned,
            influenced_subgraph=influenced_subgraph,
            influenced_subgraph_unlearned=influenced_subgraph_unlearned,
            influenced_nodes=influenced_nodes,
            unlearn_subgraph=unlearn_subgraph,
            unlearning_id=unlearning_id,
        )

    # ----------------------------------------------------------------
    # Hook 4 — mirrors cognac.unlearn()
    # ----------------------------------------------------------------
    def unlearn(self):
        """
        Capture the original model's softlabels BEFORE the trainer edits
        weights (so mia_attack() can use them after), then run ETR.
        """
        self.data = self.data.to(self.device)
        self.target_model.model.eval()
        with torch.no_grad():
            self.original_softlabels = F.softmax(
                self.target_model.model(self.data.x, self.data.edge_index), dim=1
            ).clone().detach().float()

        unlearn_time, test_f1 = self.target_model.etr_unlearning(
            self.temp_node, run=self.run
        )
        self.avg_unlearning_time[self.run] = unlearn_time
        self.average_f1[self.run] = test_f1

        self.logger.info(
            "%sETR Performance | run=%d | TestF1=%.4f | UnlearnTime=%.4f s%s"
            % (BLUE_COLOR, self.run, test_f1, unlearn_time, RESET_COLOR)
        )
        # NOTE: mia_attack() is NOT called here.
        # Learning_based_pipeline.run_exp() calls it after unlearn() when
        # attack=True (same pattern as Cognac).

    # ----------------------------------------------------------------
    # Helpers — mirror cognac._train_model() / mia_attack()
    # ----------------------------------------------------------------
    def _train_model(self, run):
        """Thin wrapper matching cognac._train_model()."""
        start_time = time.time()
        res = self.target_model.train()
        train_time = time.time() - start_time
        return train_time, res

    def mia_attack(self):
        """
        Membership-inference attack mirroring cognac.mia_attack().
        Uses self.original_softlabels (captured in unlearn() BEFORE weights
        were changed) vs the post-unlearning model outputs.
        """
        try:
            mia_num = self.unlearing_nodes.shape[0]
            if mia_num > len(self.data.test_indices):
                mia_num = len(self.data.test_indices)

            original_softlabels = self.original_softlabels

            self.target_model.model.eval()
            with torch.no_grad():
                unlearn_softlabels = F.softmax(
                    self.target_model.model(self.data.x, self.data.edge_index), dim=1
                ).clone().detach().float()

            orig_member = original_softlabels[self.unlearing_nodes[:mia_num]]
            orig_non    = original_softlabels[self.data.test_indices[:mia_num]]
            unl_member  = unlearn_softlabels[self.unlearing_nodes[:mia_num]]
            unl_non     = unlearn_softlabels[self.data.test_indices[:mia_num]]

            mia_test_y = torch.cat((torch.ones(mia_num), torch.zeros(mia_num)))
            posterior1 = torch.cat((orig_member, orig_non), 0).cpu().detach()
            posterior2 = torch.cat((unl_member, unl_non), 0).cpu().detach()
            posterior = np.array(
                [np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))]
            )
            auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            self.average_auc[self.run] = auc
            return auc
        except Exception as e:
            self.logger.warning(f"[ETR] MIA attack skipped: {e}")
            return 0.0
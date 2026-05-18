# Contributing to the GNN Unlearning Benchmark

Thank you for your interest in extending this benchmark!  
Below is everything you need to add new unlearning methods, datasets, or attacks into our benchmark, along with code style expectations and a PR checklist.

---

## Table of Contents

- [Project Layout](#project-layout)
- [Adding a New Unlearning Method](#adding-a-new-unlearning-method)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New Forgetting Attack](#adding-a-new-forgetting-attack)
- [Code Style](#code-style)
- [Pull Request Checklist](#pull-request-checklist)

---

## Project Layout

```
Unlearning_Benchmark/
├── GULib-master/
│   ├── config.py                  # Path constants (root_path, unlearning_path, …)
│   ├── evaluate_unlearning.py     # Standalone utility/forgetting evaluator
│   ├── generate_workload_sets.py  # generates node sets for robustness analysis
│   ├── main.py                    # Entry point: train → unlearn → log
│   ├── parameter_parser.py        # All CLI arguments          ← edit when adding a method
│   ├── unlearning_manager.py      # method_map: name → class  ← edit when adding a method
│   │
│   ├── dataset/
│   │   └── original_dataset.py    # Dataset loaders            ← edit when adding a dataset
│   │
│   ├── model/
│   │   ├── base_gnn/              # gcn.py, gat.py, gin.py, deletion.py, …
│   │   └── model_zoo.py
│   │
│   ├── task/                      # Trainer classes
│   │   ├── BaseTrainer.py         # Default trainer; covers most methods
│   │   ├── MEGUTrainer.py         # Example of a method-specific trainer
│   │   ├── __init__.py            # trainer_mapping dict  ← edit when adding a trainer
│   │   └── …
│   │
│   ├── pipeline/                  # Abstract base pipelines — inherit from these
│   │   ├── Learning_based_pipeline.py   # gradient / fine-tuning methods
│   │   ├── IF_based_pipeline.py         # influence-function methods
│   │   └── Shard_based_pipeline.py      # partition / SISA methods
│   │
│   ├── unlearning/
│   │   └── unlearning_methods/    # One subfolder per method
│   │       ├── MEGU/              # Good reference for learning-based methods
│   │       │   ├── __init__.py
│   │       │   └── megu.py
│   │       ├── GIF/               # Good reference for IF-based methods
│   │       │   ├── __init__.py
│   │       │   └── gif.py
│   │       └── …
│   │
│   └── attack/
│       ├── MIA_attack.py          # MI_attack() AUROC scorer — used at evaluation time
│       ├── Trend_attack.py        # Inversion attack (TrendAttack)
│       ├── Membership_Recall_Attack.py  # Noisy-labeler attack (MRattack)
│       └── Attack_methods/        # Method-specific MIA variants (GraphEraser, GUIDE, …)
│
├── unlearn_model.sh               # Run training + unlearning
├── utility_stats.sh               # Run evaluation metrics
├── README.md
└── CONTRIBUTIONS.md
```

---

## Adding a New Unlearning Method

Follow these **6 steps** in order. Each step is small and isolated.  
When in doubt, look at `unlearning/unlearning_methods/MEGU/megu.py` (learning-based)
or `unlearning/unlearning_methods/GIF/gif.py` (influence-function) as concrete references for example.

---

### Step 1 — Create the method module

Create `GULib-master/unlearning/unlearning_methods/YourMethod/` with two files:

**`__init__.py`**
```python
from .yourmethod import yourmethod
```

**`yourmethod.py`** — choose the right base pipeline:

| Your method type | Extend this class |
|---|---|
| Gradient / fine-tuning based | `pipeline.Learning_based_pipeline.Learning_based_pipeline` |
| Influence-function based | `pipeline.IF_based_pipeline.IF_based_pipeline` |
| Shard / SISA based | `pipeline.Shard_based_pipeline.Shard_based_pipeline` |

Minimal skeleton (learning-based shown):

```python
# GULib-master/unlearning/unlearning_methods/YourMethod/yourmethod.py
import os
import time
import numpy as np
import torch
from pipeline.Learning_based_pipeline import Learning_based_pipeline
from config import root_path, unlearning_path, unlearning_edge_path
from task import get_trainer


class yourmethod(Learning_based_pipeline):
    """
    One-sentence description of what your method does.

    Reference: Author et al., "Paper Title", Venue Year.
    """

    def __init__(self, args, logger, model_zoo):
        super().__init__(args, logger, model_zoo)
        # any additional initialisation

    # -----------------------------------------------------------------------
    # The four methods below are called in order by run_exp() / run_exp_mem().
    # You must implement all four.
    # -----------------------------------------------------------------------

    def determine_target_model(self):
        """Initialise self.target_model (the trainer wrapping the GNN)."""
        self.args["unlearn_trainer"] = "BaseTrainer"   # or "YourTrainer"
        self.target_model = get_trainer(
            self.args, self.logger, self.model_zoo.model, self._data
        )

    def train_original_model(self):
        """Train the GNN on the full dataset before any forgetting."""
        self.logger.info("Training original model, run %s" % self.run)
        run_time, _ = self._train_model(self.run)
        self.avg_training_time[self.run] = run_time

    def unlearning_request(self):
        """Load the forget-set and update self.data for the residual graph."""
        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()

        if self.args["unlearn_task"] == "node":
            path = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path, dtype=int)
            self.unlearing_nodes = unique_nodes
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        elif self.args["unlearn_task"] == "edge":
            path = (
                unlearning_edge_path
                + "_" + str(self.run)
                + "_edges_" + str(self.args["num_unlearned_edges"])
                + ".txt"
            )
            remove_edges = np.loadtxt(path, dtype=int)
            unique_nodes = np.unique(remove_edges)
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(
                unique_nodes, remove_edges
            )

        elif self.args["unlearn_task"] == "feature":
            path = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path, dtype=int)
            self.unlearing_nodes = unique_nodes
            self.data.x_unlearn[unique_nodes] = 0.0

        self.temp_node = unique_nodes
        self.target_model.data = self.data

    def unlearn(self):
        """Apply your forgetting update to self.target_model."""
        t0 = time.time()

        # --- your method's core logic here ---
        # model = self.target_model.model
        # ...

        self.avg_unlearning_time[self.run] = time.time() - t0

        # Evaluate on the residual graph and store results
        self.average_f1[self.run] = self.target_model.evaluate()

    # run_exp() and run_exp_mem() are fully implemented in the base pipeline.
    # Override only if you need a fundamentally different evaluation loop.
```

---

### Step 2 — (Optional) Add a custom Trainer

`BaseTrainer` handles standard GCN/GAT/GIN training and covers the majority of methods.
Create a custom trainer only if your method genuinely needs a different training loop
(e.g. a different loss function, multi-step optimisation, or custom checkpoint naming).

```python
# GULib-master/task/YourTrainer.py
from task.BaseTrainer import BaseTrainer

class YourTrainer(BaseTrainer):
    def train(self, save=False, model_path=None, needs_retrain=False):
        # override only what differs from BaseTrainer.train()
        ...
```

Register it in `GULib-master/task/__init__.py`:

```python
from task.YourTrainer import YourTrainer    # add this import

trainer_mapping = {
    ...
    'YourTrainer': YourTrainer,             # add this entry
}
```

Then in your method's `determine_target_model`, set:
```python
self.args["unlearn_trainer"] = "YourTrainer"
```

---

### Step 3 — Register in `unlearning_manager.py`

Open `GULib-master/unlearning_manager.py` and add two lines:

```python
# 1. Add import near the other method imports at the top
from unlearning.unlearning_methods.YourMethod.yourmethod import yourmethod

# 2. Add entry to method_map
method_map = {
    ...
    "YourMethod": yourmethod,    # add this line
}
```

---

### Step 4 — Add the CLI option in `parameter_parser.py`

Add your method name to the `--unlearning_methods` choices list:

```python
parser.add_argument('--unlearning_methods', type=str, default='GOLD',
    choices=[
        'GOLD', 'GraphEraser', 'GUIDE', 'GNNDelete', 'GIF',
        'CGU', 'GST', 'Projector', 'MEGU', 'IDEA', 'ScaleGUN',
        'COGNAC', 'ETR',
        'YourMethod',   # ← add here
    ])
```

Add any method-specific hyperparameters in a clearly labelled block at the bottom
of `parameter_parser.py`:

```python
### YourMethod ###
parser.add_argument('--your_lr', type=float, default=1e-3,
                    help='YourMethod: learning rate for the unlearn step.')
parser.add_argument('--your_steps', type=int, default=10,
                    help='YourMethod: number of gradient steps in the forgetting update.')
```

---

### Step 5 — Add model-save path in `evaluate_unlearning.py` (if needed)

Most methods follow the standard checkpoint naming convention already handled by
`evaluate_unlearning.py`. If your method saves checkpoints under a different name
or directory, add a branch in `build_paths()`:

```python
# GULib-master/evaluate_unlearning.py  →  build_paths()
if method == "YourMethod":
    unlearn_model_path = (
        UNLEARNED_MODEL_DIR
        / f"YourMethod/{dataset}/{unlearn_task}/{unlearn_ratio_tag}"
        / f"YourMethod_{dataset}_node_{unlearn_ratio_tag}{base_suffix}.pt"
    )
```

The standard convention (used by most methods and the default fallback) is:
```
unlearned_models/{METHOD}/{dataset}/{unlearn_task}/ratio_{ratio:.2f}/
    {METHOD}_{dataset}_node_ratio_{ratio:.2f}_{run}{base_suffix}.pt
```

**Weight-space comparison:** if your method saves standard model checkpoints (i.e. a
`state_dict` or a `{"model_state": ...}` dict) and comparing its weights against GOLD
is meaningful, add it to `WEIGHT_COMPARISON_METHODS` near the top of
`evaluate_unlearning.py`:

```python
WEIGHT_COMPARISON_METHODS = {"MEGU", "GIF", "IDEA", "COGNAC", "ETR", "YourMethod"}
```

If your method saves a raw parameter list (like GIF/IDEA do), make sure
`load_flat_params()` already handles your format — add a branch there if not.
Weight comparison results are then reported automatically at evaluation time; no other
changes are needed.

---

### Step 6 — Update the README methods table

Add a row for your method in the **Supported Methods** table in `README.md`:

```markdown
| YourMethod | `YourMethod` | Your paradigm | ✓/— | ✓/— | ✓/— |
```

---

## Adding a New Dataset

**Step 1 — Register the name in `parameter_parser.py`:**

```python
parser.add_argument('--dataset_name', ...,
    choices=[
        "cora", "citeseer", "Photo", "Amazon-ratings",
        "Roman-empire", "ogbn-arxiv", "Reddit",
        "YourDataset",   # ← add here
    ])
```

**Step 2 — Implement loading in `dataset/original_dataset.py`:**

Add a branch inside `load_data()`. Follow the pattern of an existing dataset of the
same type:

- PyG built-in homophily (Cora, Citeseer): use `torch_geometric.datasets.Planetoid`
- PyG heterophily (Amazon-ratings, Roman-empire): use `HeterophilousGraphDataset`
- OGB (ogbn-arxiv): use `ogb.nodeproppred.NodePropPredDataset`
- Custom: load manually and populate a `torch_geometric.data.Data` object

The returned `data` object must have at minimum:
`data.x`, `data.edge_index`, `data.y`, `data.num_nodes`, `data.num_features`, `data.num_classes`

**Step 3 — Update `README.md`:**

Add a row for your dataset in the **Datasets** table.

---

## Adding a New Forgetting Attack

**Step 1 — Create `GULib-master/attack/your_attack.py`:**

```python
# attack/your_attack.py
import numpy as np
from sklearn.metrics import roc_auc_score


def YourAttack(
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
    verbose: bool = False,
):
    """
    Returns AUROC (float).
    Values close to 0.5 mean strong forgetting (indistinguishable from random).
    Values above 0.5 indicate residual membership leakage.
    """
    # ... your attack logic ...
    auroc = roc_auc_score(labels, scores)
    return float(auroc)
```

The function signature must match the existing attacks exactly so it plugs in
without changing any other file.

**Step 2 — Register in `evaluate_unlearning.py`:**

```python
# Add import at top
from attack.your_attack import YourAttack

# Add entry to ATTACK_MAP
ATTACK_MAP = {
    "MIattack":    MI_attack,
    "TrendAttack": TrendAttack,
    "MRattack":    MRattack,
    "YourAttack":  YourAttack,   # ← add here
}
```

**Step 3** — Add `"YourAttack"` to the `--attack_type` choices in `evaluate_unlearning.py`.

---

## Code Style

**Log via the existing logger, not bare `print()`.**

```python
# Wrong (for anything important)
print("Unlearning done")

# Correct
self.logger.info("Unlearning done, run %s" % self.run)
```

**Keep method logic self-contained.**  
All of your method's code belongs inside `unlearning/unlearning_methods/YourMethod/`.
Do not add imports for your method to `main.py` as `UnlearningManager` handles dispatch.

**Respect the existing seed.**  
Do not call `random.seed()`, `np.random.seed()`, or `torch.manual_seed()` inside
your method. The global seed is set to `2024` in `main.py` before any method runs.

**Use canonical import locations for attack utilities:**

```python
# MIA AUROC scorer (used at evaluation time)
from attack.MIA_attack import MI_attack

# Inversion and noisy-labeler attacks (evaluation time)
from attack.Trend_attack import TrendAttack
from attack.Membership_Recall_Attack import MRattack
```

---

## Pull Request Checklist

Before opening a PR, confirm **all** of the following:

- [ ] All 6 steps for the new method are complete (or 2-3 steps for dataset/attack)
- [ ] No active `breakpoint()` calls in any changed file
- [ ] No hardcoded absolute paths (`/data/`, `/unlearned_models/`, `/MIA_stats.txt`, etc.)
- [ ] Method runs end-to-end without error on Cora, ratio 0.1, 1 run:
  ```bash
  python GULib-master/main.py \
      --dataset_name cora --base_model GCN \
      --unlearning_methods YourMethod \
      --unlearn_ratio 0.1 --num_runs 1 --cal_mem False
  ```
- [ ] Evaluation runs and prints accuracy, fidelity, logit distance, weight-space distance
  (if your method is in `WEIGHT_COMPARISON_METHODS`), and attack score:
  ```bash
  python GULib-master/evaluate_unlearning.py \
      --dataset_name cora --base_model GCN \
      --unlearning_methods YourMethod \
      --unlearn_ratio 0.1 --num_runs 1
  ```
- [ ] `README.md` methods (or datasets) table updated
- [ ] PR description includes:
  - Paper / reference for the method
  - Which pipeline base class it extends and why
  - Any new `--` CLI arguments with their default values
  - Known OOM or OOT behaviour on large graphs (if applicable)
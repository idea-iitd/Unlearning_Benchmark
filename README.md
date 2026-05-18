# IS GRAPH UNLEARNING READY FOR PRACTICE?
### A Benchmark on Efficiency, Utility, and Forgetting

**ICLR 2026** · [Paper](https://openreview.net/forum?id=gSPkuTTWgU) · [GitHub](https://github.com/idea-iitd/Unlearning_Benchmark)

This repository contains the **official implementation** of the paper:  
**"Is Graph Unlearning Ready for Practice? A Benchmark on Efficiency, Utility, and Forgetting"**

We introduce a unified benchmark framework to evaluate multiple **graph unlearning techniques** across diverse datasets — measuring **efficiency**, **utility**, and **forgetting**.

---

## Overview

This benchmark provides:
- A **standardized evaluation** of graph unlearning methods.
- Comparisons on **time, memory, accuracy, and forgetting behavior**.
- Support for multiple **datasets** and **GNN architectures**.
- Evaluation across three core pillars:

| Pillar | What we measure |
|--------|----------------|
| **Efficiency** | Runtime and peak GPU memory vs. retraining from scratch |
| **Utility** | Accuracy, per-node fidelity, logit-space L2 distance, weight-space distance |
| **Forgetting** | MIA AUROC, unlearning inversion attack, noisy-labeler attack |

---

## Installation

### Prerequisites
- **Python:** 3.8.0
- **CUDA:** Ensure the CUDA version is compatible with your PyTorch installation.

---

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/idea-iitd/Unlearning_Benchmark.git
cd Unlearning_Benchmark
```

---

### 2️⃣ Install in Editable Mode
```bash
pip install -e .
```

---

### 3️⃣ Install Dependencies

#### (a) PyTorch and torchvision (with CUDA Support)

Example for **CUDA 12.1**:
```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Required Versions:
- `torch==2.2.1`
- `torchvision==0.17.1`

---

#### (b) CuPy with CUDA Support (required by ScaleGUN only)

Example for **CUDA 12.x**:
```bash
pip install cupy-cuda12x
```

For other CUDA versions, refer to the official [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html).

---

#### (c) General Dependencies
```bash
pip install -r requirements.txt
```

---

#### (d) Graph Library Dependencies

If you encounter build errors, install the precompiled wheels from the
[PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Example for **CUDA 12.1**:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-sparse  -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-geometric
```

For other CUDA versions, replace `cu121` with your version (e.g., `cu118`).

---

## Running Benchmarks

> **Important:** Always run `GOLD` first for a given dataset + ratio combination.  
> GOLD creates the train/test split, unlearning index files, and the retrained reference  
> model that all evaluation compares against.

### Gold Standard (Retrain from Scratch)

```bash
python GULib-master/main.py \
    --dataset_name cora \
    --base_model GCN \
    --unlearning_methods GOLD \
    --num_epochs 100 \
    --batch_size 64 \
    --unlearn_ratio 0.1 \
    --num_runs 1 \
    --cal_mem True
```

### Unlearning a Model

To unlearn a model, run `unlearn_model.sh` or use the following command:

```bash
python GULib-master/main.py \
    --dataset_name cora \
    --base_model GCN \
    --unlearning_methods MEGU \
    --attack False \
    --num_epochs 100 \
    --batch_size 64 \
    --unlearn_ratio 0.1 \
    --num_runs 1 \
    --cal_mem True
```

This command will **train**, **unlearn**, and **save** the unlearned model.

---

## Optional Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--cuda <device>` | Specify GPU device to use | `--cuda 0` |
| `--dataset_name <name>` | Graph dataset name | `--dataset_name cora` |
| `--base_model <model>` | Base GNN model architecture | `GCN`, `GAT`, `GIN` |
| `--unlearning_methods <method>` | Unlearning method | `MEGU`, `GIF`, `GraphEraser`, `GUIDE`, `GNNDelete`, `IDEA`, `Projector`, `ScaleGUN`, `CGU`, `COGNAC`, `ETR`, `GOLD` |
| `--unlearn_ratio <value>` | Fraction of data to unlearn | `0.1` |
| `--num_unlearned_nodes <N>` | Absolute count of nodes to unlearn | `271` |
| `--unlearn_task <task>` | Unlearning granularity | `node`, `edge`, `feature` |
| `--num_epochs <N>` | Number of training epochs | `100` |
| `--batch_size <N>` | Batch size | `64` |
| `--num_runs <N>` | Independent runs to average over | `5` |
| `--attack <True/False>` | Enable MIA during unlearning | `False` |
| `--attack_type <name>` | Forgetting attack for evaluation | `MIattack`, `TrendAttack`, `MRattack` |
| `--cal_mem <True/False>` | Record time and memory stats | `True` |

---

## Efficiency Evaluation

To record **efficiency metrics** (time and memory usage), pass `--cal_mem True` to `main.py`.

Results are stored in:
```
efficiency_stats.txt
```

---

## Utility Evaluation

### Accuracy, Fidelity, Logit Similarity

Run `utility_stats.sh` or call `evaluate_unlearning.py` directly:

```bash
bash utility_stats.sh
```

```bash
python GULib-master/evaluate_unlearning.py \
    --dataset_name cora \
    --base_model GCN \
    --unlearning_methods MEGU \
    --unlearn_ratio 0.1 \
    --unlearn_task node \
    --num_runs 1
```

This computes:
- **Accuracy**
- **Fidelity** (per-node prediction agreement with GOLD)
- **Logit L2 distance** (output distribution similarity to GOLD)
- **Weight-space distance** — L2, Cosine, and Relative-L2 between the unlearned and GOLD model parameters.  
  Reported automatically for methods that can support it: **MEGU, GIF, IDEA, COGNAC, ETR**.  

---

## Forgetting Evaluation

To evaluate forgetting performance, pass `--attack_type` to `evaluate_unlearning.py`:

```bash
python GULib-master/evaluate_unlearning.py \
    --dataset_name cora \
    --unlearning_methods MEGU \
    --unlearn_ratio 0.1 \
    --attack_type MIattack
```

Where `--attack_type` can be:

| Attack | What it tests |
|--------|--------------|
| `MIattack` | Membership Inference — can an adversary tell if a node was in training? |
| `TrendAttack` | Inversion attack — can deleted edges be reconstructed from logits? |
| `MRattack` | Noisy-labeler — does the model assign high-confidence original labels to deleted nodes? |

An AUROC close to **0.5** indicates strong forgetting. Values above 0.5 indicate residual leakage.

> **Note:** Utility results for **GraphEraser** and **GUIDE** are automatically stored during unlearning time
> in `GraphEraser_utility_stats.txt` and `GUIDE_utility_stats.txt` files. And for getting forgetting results for them 
> pass the `--attack_type` argument to `main.py` instead.

---

## Robustness Analysis over different unlearning workload Distributions 

By default all methods use randomly sampled training nodes. To evaluate on a structured deletion strategy, generate the node set first, then run unlearning and evaluation as normal:

```bash
# Step 1 — overwrite the initial random node set with your chosen strategy
python GULib-master/generate_workload_sets.py \
    --dataset_name cora --unlearn_ratio 0.1 \
    --strategy high_freq   # or: second_freq | low_degree | high_degree | random

# Step 2 — run unlearning (reads the node set as usual)
python GULib-master/main.py \
    --dataset_name cora --base_model GCN \
    --unlearning_methods MEGU --unlearn_ratio 0.1

# Step 3 — evaluate as usual
python GULib-master/evaluate_unlearning.py \
    --dataset_name cora --unlearning_methods MEGU \
    --unlearn_ratio 0.1
```

## Datasets

| Dataset | Nodes | Edges | Type |
|---------|-------|-------|------|
| Cora | 2,708 | 5,278 | Homophily |
| Citeseer | 3,327 | 4,732 | Homophily |
| Photo | 7,487 | 119,043 | Homophily |
| ogbn-arxiv | 169,343 | 1,166,243 | Homophily |
| Amazon-ratings | 24,492 | 93,050 | Heterophily |
| Roman-empire | 22,662 | 32,927 | Heterophily |
| Reddit | 232,965 | 114,615,892 | Homophily (scalability) |

---

## Supported Unlearning Methods

| Method | Paradigm | Model-agnostic | Cont. Training | Train. Mode | Guarantee |
|--------|----------|:--------------:|:--------------:|:-----------:|:---------:|
| MEGU | Learning-based | ✓ | ✓ | Post-hoc | — |
| GIF | Influence function | ✓ | — | Post-hoc | — |
| IDEA | IF + certified | ✓ | — | Post-hoc | — |
| GST | Influence function | ✓ | ✓ | Post-hoc | — |
| ETR | IF + Learning | ✓ | ✓ | Post-hoc | — |
| COGNAC | Corrective | ✓ | ✓ | Post-hoc | — |
| GNNDelete | Learning-based | ✓ | — | Post-hoc | — |
| GraphEraser | SISA / Partition | ✓ | ✓ | Train-time | ✓ |
| GUIDE | SISA / Partition | ✓ | ✓ | Train-time | ✓ |
| Projector | Projection | — | — | Train-time | ✓ |
| ScaleGUN | Certified (linear GNN, binary) | — | ✓ | Train-time | ✓ |
| CGU | Certified (linear GNN, binary) | — | ✓ | Train-time | ✓ |

---

## Project Structure

```
Unlearning_Benchmark/
├── GULib-master/
│   ├── config.py                      # Derived file paths
│   ├── evaluate_unlearning.py         # Compute all metrics (utility and forgetting)
│   ├── main.py                        # Train + unlearn
│   ├── parameter_parser.py            # All CLI arguments and defaults
│   ├── unlearning_manager.py          # method-name → class dispatch
│   ├── unlearning/unlearning_methods/ # One subfolder per method
│   │   ├── MEGU/megu.py
│   │   ├── GIF/gif.py
│   │   └── …
│   ├── pipeline/                      # Base pipeline classes
│   │   ├── Learning_based_pipeline.py
│   │   ├── IF_based_pipeline.py
│   │   └── Shard_based_pipeline.py
│   ├── task/                          # Trainer classes
│   ├── attack/
│   │   ├── MIA_attack.py              # MIA AUROC scorer (evaluation)
│   │   ├── shadow_model.py            # Shadow/attack model classes (training)
│   │   ├── Trend_attack.py
│   │   └── Membership_Recall_Attack.py
│   ├── dataset/                       # Dataset loaders and splits
│   ├── model/                         # GNN architectures and model zoo
│   └── utils/                         # Logging and data utilities
├── unlearn_model.sh                   # Run training + unlearning
├── utility_stats.sh                   # Run evaluation metrics
└── requirements.txt
```

---

## Contributing

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for step-by-step instructions on adding new unlearning methods, datasets, and attack families.

---

## Citation

```bibtex
@inproceedings{jain2026is,
  title     = {Is Graph Unlearning Ready for Practice? A Benchmark on Efficiency, Utility, and Forgetting},
  author    = {Samyak Jain and Ronak Kalvani and Sainyam Galhotra and Sayan Ranu},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=gSPkuTTWgU}
}
```

---

## Contact

For questions, issues, or contributions, please open a GitHub issue or contact the authors.
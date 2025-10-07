# 📘 IS GRAPH UNLEARNING READY FOR PRACTICE?  
### A Benchmark on Efficiency, Utility, and Forgetting

This repository contains the **official implementation** of the paper:  
**"Is Graph Unlearning Ready for Practice? A Benchmark on Efficiency, Utility, and Forgetting"**

We introduce a unified benchmark framework to evaluate multiple **graph unlearning techniques** across diverse datasets — measuring **efficiency**, **utility**, and **forgetting**.

---

## 📚 Overview

This benchmark provides:
- A **standardized evaluation** of graph unlearning methods.  
- Comparisons on **time, memory, accuracy, and forgetting behavior**.  
- Support for multiple **datasets** and **GNN architectures**.

---

## 📥 Installation

### 🧩 Prerequisites
- **Python:** 3.8.0  
- **CUDA:** Ensure the CUDA version is compatible with your PyTorch installation.

---

### 1️⃣ Clone the Repository
```bash
git clone <REPO-URL>
cd GNN_Unlearning
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

✅ Required Versions:
- `torch==2.2.1`  
- `torchvision==0.17.1`

---

#### (b) CuPy with CUDA Support

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
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-geometric
```

For other CUDA versions, replace `cu121` with your version (e.g., `cu118`).

---

## 🚀 Running Benchmarks

### 🔹 Unlearning a Model

To unlearn a model, run the unlearn_model.sh file or use the following command:

```bash
python GULib-master/main.py   --dataset_name cora   --base_model GCN   --unlearning_methods MEGU   --attack False   --num_epochs 100   --batch_size 64   --unlearn_ratio 0.1   --num_runs 1   --cal_mem True
```

This command will **train**, **unlearn**, and **save** the unlearned model.

---

### ⚙️ Optional Arguments

| Argument | Description | Example |
|-----------|--------------|----------|
| `--cuda <device>` | Specify GPU device to use | `--cuda 0` |
| `--dataset_name <name>` | Graph dataset name | `--dataset_name cora` |
| `--base_model <model>` | Base GNN model architecture | `GCN`, `GAT`, `GIN` |
| `--unlearning_methods <method>` | Unlearning method | `MEGU`, `GraphEraser`, `GUIDE`, `GNNDelete`, `GIF`, `IDEA`, `Projector` |
| `--unlearn_ratio <value>` | Fraction of data to unlearn | `0.1` |
| `--num_epochs <N>` | Number of training epochs | `100` |
| `--batch_size <N>` | Batch size | `64` |
| `--attack <True/False>` | Enable membership inference attack | `True` |
| `--cal_mem <True/False>` | Record time and memory stats | `True` |

---

## 📊 Forgetting and Efficiency Evaluation

To evaluate **forgetting performance** using Membership Inference (MI) Attack:
```bash
--attack True
```

To record **efficiency metrics** (time and memory usage):
```bash
--cal_mem True
```

Results are stored in:
```
efficiency_stats.txt
```

---

## 🎯 Utility Evaluation

After Getting the Utility Stats, run:
```bash
bash utility_stats.sh
```

This computes:
- **Accuracy**
- **Fidelity**
- **Logit Similarity**

📝 Note:
- Utility results for **GraphEraser** and **GUIDE** are automatically stored during unlearning:
  - `GraphEraser_utility_stats.txt`
  - `GUIDE_utility_stats.txt`

For Getting Weight Comparsion Results, run:
```bash
python GULib-master/Weight_comparison.py
```
---

## 📦 Datasets

Supported graph datasets:
- **Cora**
- **Citeseer**
- **ogbn-arxiv**
- **Amazon-ratings**
- **Roman-empire**

---

## 🧠 Supported Unlearning Methods

Our benchmark currently supports:
- **MEGU**
- **GraphEraser**
- **GUIDE**
- **GNNDelete**
- **GIF**
- **IDEA**
- **Projector**

---

## 📬 Contact

For questions, issues, or contributions, please open a GitHub issue or contact the authors.



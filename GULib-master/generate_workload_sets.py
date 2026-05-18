"""
generate_workload_sets.py
=========================
Generates a structured unlearning node set for robustness analysis (RQ5 / Table 10).

Saves ONE file to the copy path (--use_copy True path) based on the chosen --strategy:

    Strategy        Selects training nodes from ...
    ─────────────────────────────────────────────────────────
    random          uniform random sample (baseline)
    high_freq       most frequent label class
    second_freq     second most frequent label class
    high_degree     highest-degree nodes (structural hubs)
    low_degree      lowest-degree nodes

The file is saved to:
    ./data/unlearning_task/transductive/imbalanced/
        unlearning_nodes_copy_{ratio}_{dataset}_0_nodes_{N}.txt

Run GOLD first (without --use_copy) so the train/test split exists, then run this
script, then re-run main.py + evaluate_unlearning.py with --use_copy True.

Usage
─────
python GULib-master/generate_workload_sets.py \
    --dataset_name cora --unlearn_ratio 0.1 --strategy high_freq

python GULib-master/generate_workload_sets.py \
    --dataset_name cora --unlearn_ratio 0.1 --strategy high_degree
"""

import os
import numpy as np
import torch

from dataset.original_dataset import original_dataset
from parameter_parser import parameter_parser
from utils.logger import create_logger
from utils.dataset_utils import process_data

# -------------------------------
# Step 1: Load arguments and setup
# -------------------------------
args = parameter_parser()
logger = create_logger(args)

torch.cuda.set_device(args['cuda'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])

strategy    = args["strategy"]
dataset_str = args["dataset_name"]
ratio       = args["unlearn_ratio"]
ratio_str   = str(ratio)

# -------------------------------
# Step 2: Load dataset
# -------------------------------
original_data = original_dataset(args, logger)
data, dataset = original_data.load_data()
data = process_data(logger, data, args)

labels     = data.y.cpu().numpy()
num_nodes  = data.num_nodes
node_count = int(ratio * num_nodes)

train_nodes = np.where(data.train_mask.cpu().numpy())[0]

np.random.seed(42)

# -------------------------------
# Step 3: Select nodes by strategy
# -------------------------------
if strategy == "random":
    selected = np.random.choice(train_nodes, node_count, replace=False)

elif strategy == "high_freq":
    unique_labels, counts = np.unique(labels[train_nodes], return_counts=True)
    top_label = unique_labels[np.argmax(counts)]
    candidates = train_nodes[labels[train_nodes] == top_label]
    selected = np.random.choice(candidates, node_count, replace=False)

elif strategy == "second_freq":
    unique_labels, counts = np.unique(labels[train_nodes], return_counts=True)
    sorted_idx = np.argsort(-counts)
    second_label = unique_labels[sorted_idx[1]]
    candidates = train_nodes[labels[train_nodes] == second_label]
    selected = np.random.choice(candidates, node_count, replace=False)

elif strategy == "high_degree":
    row, col = data.edge_index
    degrees = np.bincount(row.cpu().numpy(), minlength=num_nodes)
    train_degrees = degrees[train_nodes]
    sorted_train_idx = np.argsort(train_degrees)
    selected = train_nodes[sorted_train_idx[-node_count:]]

elif strategy == "low_degree":
    row, col = data.edge_index
    degrees = np.bincount(row.cpu().numpy(), minlength=num_nodes)
    train_degrees = degrees[train_nodes]
    sorted_train_idx = np.argsort(train_degrees)
    selected = train_nodes[sorted_train_idx[:node_count]]

# -------------------------------
# Step 4: Save to copy path
# -------------------------------
save_path = (
    f"./data/unlearning_task/transductive/imbalanced/"
    f"unlearning_nodes_{ratio_str}_{dataset_str}_0_nodes_{node_count}.txt"
)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w") as f:
    for node in selected:
        f.write(f"{node}\n")

print(f"Saved {strategy} nodes to: {save_path}")

# -------------------------------
# Step 5: Print stats
# -------------------------------
print("\n=== Unlearning Stats ===")
print(f"Dataset       : {dataset_str}")
print(f"Strategy      : {strategy}")
print(f"Total nodes   : {num_nodes}")
print(f"Train nodes   : {len(train_nodes)}")
print(f"Unlearn ratio : {ratio}")
print(f"Nodes selected: {len(selected)} / {node_count}")

if strategy in ("high_freq", "second_freq"):
    unique_labels, counts = np.unique(labels[train_nodes], return_counts=True)
    sorted_idx = np.argsort(-counts)
    for rank, (lbl, cnt) in enumerate(zip(unique_labels[sorted_idx], counts[sorted_idx])):
        print(f"  Label {lbl} (rank {rank+1}): {cnt} train nodes")
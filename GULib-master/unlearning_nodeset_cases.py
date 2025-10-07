import os
import random
# import optuna
import numpy as np
import torch
from model.model_zoo import model_zoo
from dataset.original_dataset import original_dataset
from parameter_parser import parameter_parser
from unlearning.unlearning_methods.CEU.ceu import ceu
from utils.logger import create_logger
from task.node_classification import NodeClassifier
from unlearning.unlearning_methods.GNNDelete.gnndelete import gnndelete
from utils.dataset_utils import process_data,save_data
from attack.Attack_methods.GraphEraser_MIA import GraphEraser_Attack
from attack.Attack_methods.GUIDE_MIA import GUIDE_MIA
from unlearning.unlearning_methods.GraphEraser.grapheraser import grapheraser
from unlearning.unlearning_methods.GUIDE.guide import guide
from unlearning.unlearning_methods.GIF.gif import gif
from unlearning.unlearning_methods.CGU.cgu import cgu
from unlearning.unlearning_methods.GST.gst_based import gst
from unlearning.unlearning_methods.SGU import sgu
from unlearning.unlearning_methods.Projector.projector import projector
from unlearning_manager import UnlearningManager
from config import unlearning_path
import sys 
import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
import os


# For Storing the Nodes with Highest frequncy, Second Highest Frequency, Highest Degree, Lowest Degree

# -------------------------------
# Step 1: Load arguments and setup
# -------------------------------
args = parameter_parser()
logger = create_logger(args)

torch.cuda.set_device(args['cuda'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])

# -------------------------------
# Step 2: Load dataset
# -------------------------------
original_data = original_dataset(args,logger)
data,dataset = original_data.load_data()
data = process_data(logger,data,args)

labels = data.y.cpu().numpy()
num_nodes = data.num_nodes
unlearn_ratio = args["unlearn_ratio"]
ratio_str = str(unlearn_ratio)
dataset_str = args["dataset_name"]
node_count = int(args["unlearn_ratio"] * num_nodes)

# ========= Label Analysis =========

# -------------------------------
# Step 3: Identify most frequent labels
# -------------------------------
unique_labels, counts = np.unique(labels, return_counts=True)
sorted_idx = np.argsort(-counts)  # sort descending
top1_label, top2_label = unique_labels[sorted_idx[0]], unique_labels[sorted_idx[1]]
top1_count, top2_count = counts[sorted_idx[0]], counts[sorted_idx[1]]

# -------------------------------
# Step 4: Select nodes to unlearn
# -------------------------------
np.random.seed(42)

train_nodes = np.where(data.train_mask.cpu().numpy())[0]

top1_nodes = [n for n in train_nodes if labels[n] == top1_label]
top2_nodes = [n for n in train_nodes if labels[n] == top2_label]

num_top1_unlearn = node_count
num_top2_unlearn = node_count

unlearn_top1 = np.random.choice(top1_nodes, num_top1_unlearn, replace=False)
unlearn_top2 = np.random.choice(top2_nodes, num_top2_unlearn, replace=False)

# ============= Degree Analysis ==========

# # -------------------------------
# # Step 3: Identify highest and lowest degree nodes
# # -------------------------------
# # Compute node degrees
# row, col = data.edge_index
# degrees = np.bincount(row.cpu().numpy(), minlength=num_nodes)

# # Restrict to train nodes
# train_nodes = np.where(data.train_mask.cpu().numpy())[0]
# train_degrees = degrees[train_nodes]

# # -------------------------------
# # Step 4: Select nodes to unlearn
# # -------------------------------
# np.random.seed(42)

# # Get highest and lowest degree nodes among training set
# sorted_train_idx = np.argsort(train_degrees)
# unlearn_top1 = train_nodes[sorted_train_idx[-node_count:]]
# unlearn_top2 = train_nodes[sorted_train_idx[:node_count]]

# -------------------------------
# Step 5: Save results
# -------------------------------

# First path (save selected nodes)
save_path_top1 = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_nodes_{ratio_str}_{dataset_str}_0_nodes_{node_count}.txt"
save_path_top2 = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_nodes_copy_{ratio_str}_{dataset_str}_0_nodes_{node_count}.txt"

os.makedirs(os.path.dirname(save_path_top1), exist_ok=True)

with open(save_path_top1, "w") as f:
    for node in unlearn_top1:
        f.write(f"{node}\n")

with open(save_path_top2, "w") as f:
    for node in unlearn_top2:
        f.write(f"{node}\n")

print(f"Saved top1 label unlearning nodes to: {save_path_top1}")
print(f"Saved top2 label unlearning nodes to: {save_path_top2}")

# -------------------------------
# Step 6: Print stats
# -------------------------------
print("\n=== Unlearning Stats ===")
print(f"Total nodes: {num_nodes}")
print(f"Unlearn ratio: {unlearn_ratio}")
print(f"Most frequent label: {top1_label} ({top1_count} nodes)")
print(f"Second most frequent label: {top2_label} ({top2_count} nodes)")
print(f"Top1 label ({top1_label}): {len(top1_nodes)} nodes, selected {num_top1_unlearn}")
print(f"Top2 label ({top2_label}): {len(top2_nodes)} nodes, selected {num_top2_unlearn}")
print(f"Percentage removed (Top1): {100*num_top1_unlearn/len(top1_nodes):.2f}%")
print(f"Percentage removed (Top2): {100*num_top2_unlearn/len(top2_nodes):.2f}%")


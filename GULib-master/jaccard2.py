import torch
import torch.nn.functional as F
import pickle
import numpy as np
from model.base_gnn.gcn import GCNNet
from model.base_gnn.deletion import GCNDelete
from model.base_gnn.gat import GATNet
from model.base_gnn.deletion import GATDelete
from model.base_gnn.gin import GINNet
from model.base_gnn.deletion import GINDelete
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
from unlearning.unlearning_methods.Projector.utils.graph_projector_model_utils import Pro_GNN
import copy
import torch_sparse
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm



# === Config ===
parser = argparse.ArgumentParser(description="Run evaluation for MEGU and GOLD with varying unlearning ratios.")
parser.add_argument("--unlearn_ratio", type=float, default=0.1, help="Unlearning ratio (e.g., 0.1, 0.3, 1.0)")
parser.add_argument("--dataset_name", type=str, default="cora", help="Dataset name (default: cora)")
parser.add_argument("--unlearning_methods", type=str, default="MEGU", help="Dataset name (default: MEGU)")
parser.add_argument("--num_runs", type=int, default=1, help="How many times you want to perform the experiment")
parser.add_argument("--unlearn_task", type=str, default="node", help="What are we unlearning?")
parser.add_argument('--use_copy', type=str, default="False", help = "True if Using different unleared set")
parser.add_argument('--base_model', type=str, default='GCN', help = "What is Base GNN Model")


# === Recieving args ===
args_cli = parser.parse_args()
u_ratio = args_cli.unlearn_ratio
dataset = args_cli.dataset_name
unlearn_task = args_cli.unlearn_task
unlearn_ratio = f"ratio_{u_ratio:.2f}"
model_name = args_cli.unlearning_methods
num_runs = args_cli.num_runs
copy_str=""
if args_cli.use_copy=="True":
    copy_str="_copy"
if model_name=="GNNDelete":
    model_name = "gnndelete_nodeemb"
base_model_str=""
if args_cli.base_model!="GCN":
    base_model_str = "_" + args_cli.base_model


# === Load Data ===
data_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/processed/transductive/{dataset}0.8_0_0.2.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)
data = data.to("cuda")
y_true = data.y.cpu().numpy()
test_mask = data.test_mask.cpu().numpy()
train_mask = data.train_mask.cpu().numpy()
edge_index = data.edge_index
num_nodes = data.x.size(0)

# === Paths ===
if model_name=="Projector":
    original_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/Projector.pt"
    gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/Projector.pt"
else:
    original_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/{unlearn_task}/{args_cli.base_model}"
    gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GOLD/{dataset}/{unlearn_task}/{unlearn_ratio}{copy_str}/GOLD_{dataset}_node_{unlearn_ratio}{base_model_str}.pt"
megu_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/{model_name}/{dataset}/{unlearn_task}/{unlearn_ratio}{copy_str}/{model_name}_{dataset}_node_{unlearn_ratio}{base_model_str}.pt"

# === General Model Args ===
args = {
    'dataset_name': dataset,
    'downstream_task': 'node',
    'base_model': args_cli.base_model,
    'hidden_dim': 64,
    'out_dim': data.y.max().item() + 1
}


# === Helper: Load Model & Predict ===
def load_and_predict(model_path, model_type="GOLD", unlearned_param_path=None,run_number=0):
    copy_data = copy.deepcopy(data)
    num_nodes = copy_data.x.size(0)
    edge_index_np = copy_data.edge_index.cpu().numpy()

    if unlearn_task == "node":
        # Remove edges connected to unlearned nodes
        unlearn_idx_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_nodes{copy_str}_{u_ratio}_{dataset}_{run_number}_nodes_{int(u_ratio * num_nodes)}.txt"

        with open(unlearn_idx_path, "r") as f:
            unlearned_indices = list(map(int, f.readlines()))

        # === Masks ===
        unlearned_mask = np.zeros_like(train_mask, dtype=bool)
        unlearned_mask[unlearned_indices] = True
        retained_train_mask = np.logical_and(train_mask, np.logical_not(unlearned_mask))
        unlearned_train_mask = np.logical_and(train_mask, unlearned_mask)
        unlearned_nodes = torch.tensor(unlearned_indices, device='cuda')

        # 1-hop mask
        _, _, _, one_hop_edge_mask = k_hop_subgraph(
            unlearned_nodes,
            num_hops=1,
            edge_index=edge_index,
            num_nodes=num_nodes,
            relabel_nodes=False,
        )

        # 2-hop mask
        _, _, _, two_hop_edge_mask = k_hop_subgraph(
            unlearned_nodes,
            num_hops=2,
            edge_index=edge_index,
            num_nodes=num_nodes,
            relabel_nodes=False,
        )

        # Now get node-level masks from the resulting edge indices
        one_hop_nodes = edge_index[:, one_hop_edge_mask].flatten().unique()
        two_hop_nodes = edge_index[:, two_hop_edge_mask].flatten().unique()

        mask_1hop = torch.zeros(num_nodes, dtype=torch.bool, device='cuda')
        mask_2hop = torch.zeros(num_nodes, dtype=torch.bool, device='cuda')

        mask_1hop[one_hop_nodes] = True
        mask_2hop[two_hop_nodes] = True

        # Optional: exclude unlearned nodes from 2-hop mask
        mask_2hop[unlearned_nodes] = False

        # unlearned_nodes_set = set(unlearned_nodes)
        # mask = [(u not in unlearned_nodes_set and v not in unlearned_nodes_set) 
        #         for u, v in edge_index_np.T]
        
        # unlearned_nodes_set = set(map(int, unlearned_nodes))
        # u, v = edge_index_np
        # mask = ~(np.isin(u, list(unlearned_nodes_set)) | np.isin(v, list(unlearned_nodes_set)))
        # # breakpoint()
        # remain_edges = edge_index_np[:, mask]
        # copy_data.edge_index = torch.tensor(remain_edges, device='cuda')

    elif unlearn_task == "edge":
        # Load unlearned edges from file
        unlearn_edge_path = (
            f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_edges_{u_ratio}_{dataset}_{run_number}_edges_{int(u_ratio * edge_index_np[0].size)}.txt"
        )
        unlearning_edges = np.loadtxt(unlearn_edge_path, dtype=int)
        unlearning_edges = torch.tensor(unlearning_edges, dtype=torch.long, device=edge_index.device)
        nodes_to_unlearn = unlearning_edges.flatten().unique()

        # Build set for fast lookup
        unlearned_edges_set = {tuple(edge) for edge in unlearning_edges}

        # Remaining edges
        remain_edges = [(u, v) for u, v in edge_index_np.T if (u, v) not in unlearned_edges_set]
        remain_edges = np.array(remain_edges).T

        # Convert to tensor

        # Extract unique nodes touched by unlearning edges

        # 1-hop mask
        _, _, _, one_hop_edge_mask = k_hop_subgraph(
            nodes_to_unlearn,
            num_hops=1,
            edge_index=edge_index,
            num_nodes=num_nodes,
            relabel_nodes=False,
        )

        # 2-hop mask
        _, _, _, two_hop_edge_mask = k_hop_subgraph(
            nodes_to_unlearn,
            num_hops=2,
            edge_index=edge_index,
            num_nodes=num_nodes,
            relabel_nodes=False,
        )

        # Now get node-level masks from the resulting edge indices
        one_hop_nodes = edge_index[:, one_hop_edge_mask].flatten().unique()
        two_hop_nodes = edge_index[:, two_hop_edge_mask].flatten().unique()

        mask_1hop = torch.zeros(num_nodes, dtype=torch.bool, device='cuda')
        mask_2hop = torch.zeros(num_nodes, dtype=torch.bool, device='cuda')

        mask_1hop[one_hop_nodes] = True
        mask_2hop[two_hop_nodes] = True

        # copy_data.edge_index = torch.tensor(remain_edges, device='cuda')

    if model_name == 'Projector':
        
        # ---- Remove nodes and edges only for GOLD ----
    
        copy_data.node_inds = torch.arange(copy_data.x.size(0))
        extra_feats = torch.zeros(copy_data.x.size(0), device=copy_data.x.device)
        extra_feats[unlearned_nodes] = 1
        copy_data.x = torch.cat([copy_data.x, extra_feats.view(-1, 1)], dim=1)
        # breakpoint()
        copy_data.y[unlearned_nodes] = copy_data.num_classes
        copy_data.adj_t = SparseTensor(row=copy_data.edge_index[1], col=copy_data.edge_index[0])
        copy_data.adj_t = torch_sparse.fill_diag(copy_data.adj_t.to_symmetric(), 1)
        copy_data.y_one_hot_train = F.one_hot(
            data.y.squeeze(), copy_data.num_classes + 1).float()
        copy_data.y_one_hot_train[copy_data.test_indices, :] = 0
        num_nodes = copy_data.x.size(0)
        copy_data.node_inds = torch.arange(copy_data.x.size(0))


        all_loader = ShaDowKHopSampler(
            copy_data,
            depth=2,
            num_neighbors=20,  # Use your args["hop_neighbors"] here if available
            batch_size=1024,
            shuffle=False
        )


        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint

        y_dims = state_dict['W'].shape[1]
        x_dims = copy_data.x.size(1)
        projector_args = {
            'dataset_name': dataset,
            'downstream_task': 'node',
            'base_model': args_cli.base_model,
            'hidden_dim': 64,
            'out_dim': y_dims,
            'x_iters': 3,
            'y_iters': 3,
            'use_adapt_gcs': False,
            'use_cross_entropy': True,
        }

        model = Pro_GNN(x_dims, y_dims, device="cuda", args=projector_args).to("cuda")
        model.load_state_dict(state_dict)

        model.eval()
        all_pred = []

        copy_data.y_one_hot_train = F.one_hot(copy_data.y, y_dims).float().to("cuda")

        # Setup SparseTensor adjacency
        for subgraph_data in all_loader:
            subgraph_data.adj_t = SparseTensor(row=subgraph_data.edge_index[1], col=subgraph_data.edge_index[0])
            subgraph_data.y_one_hot_train = copy_data.y_one_hot_train[subgraph_data.node_inds]
            pred = model(subgraph_data.to("cuda"))
            all_pred.append(pred.detach().cpu())

        logits = torch.cat(all_pred, dim=0)
        # preds = logits.argmax(dim=1).cpu().numpy()
        return logits

    elif model_type == 'GNNDelete' or model_type == 'gnndelete_nodeemb':
        if args_cli.base_model == "GCN":
            model = GCNDelete(args, data.num_node_features, args['out_dim'])
        elif args_cli.base_model == "GAT":
            model = GATDelete(args, data.num_node_features, args['out_dim'])
        elif args_cli.base_model == "GIN":
            model = GINDelete(args, data.num_node_features, args['out_dim'])

    else:  # Includes "GOLD" and "GIF"
        if args_cli.base_model == "GCN":
            model = GCNNet(args, data.num_node_features, args['out_dim'])
        elif args_cli.base_model == "GAT":
            model = GATNet(args, data.num_node_features, args['out_dim'])
        elif args_cli.base_model == "GIN":
            model = GINNet(args, data.num_node_features, args['out_dim'])

    model = model.to("cuda")
    # breakpoint()

    # Load and apply base weights

    # For GIF: load parameters and unlearned data (x_unlearn + edge_index_unlearn)
    if model_type == "GIF" or model_type == "IDEA":
        assert unlearned_param_path is not None #"GIF model requires a path to unlearned parameters"
        
        # Load unlearned weights
        params_esti = torch.load(unlearned_param_path)
        idx = 0
        for p in model.parameters():
            p.data = params_esti[idx].to(p.device)
            idx += 1

        # Derive the save directory from unlearned_param_path
        save_dir = os.path.dirname(unlearned_param_path)

        # Load x_unlearn and edge_index_unlearn
        run_str=""
        if num_runs>1:
            run_str="_" + str(run_number)
        copy_data.x_unlearn = torch.load(os.path.join(save_dir, f"x_unlearn{run_str}{base_model_str}.pt")).to("cuda")
        copy_data.edge_index_unlearn = torch.load(os.path.join(save_dir, f"edge_index_unlearn{run_str}{base_model_str}.pt")).to("cuda")
        # breakpoint()
    
    else:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
    # breakpoint()
    model.eval()
    with torch.no_grad():
        if model_type == "Projector":
            subgraph_data = copy_data.clone()
            subgraph_data.adj_t = SparseTensor.from_edge_index(copy_data.edge_index).t().to("cuda")
            subgraph_data.y_one_hot_train = F.one_hot(copy_data.y, y_dims).float().to("cuda")
            subgraph_data.root_n_id = torch.arange(copy_data.x.size(0), device='cuda')
            logits = model(subgraph_data)

        elif model_type == 'GNNDelete' or model_type == 'gnndelete_nodeemb':
            logits = model(data.x, data.edge_index, mask_1hop=mask_1hop, mask_2hop=mask_2hop)

        elif model_type == "GIF" or model_type == 'IDEA':
            if model_type == 'IDEA' and args['base_model']=='GCN':
                _, edge_weight_unlearn = gcn_norm(
                    copy_data.edge_index_unlearn, 
                    edge_weight=None, 
                    num_nodes=copy_data.x.size(0),
                    add_self_loops=False)
                logits = model.forward_once_unlearn(copy_data,edge_weight_unlearn)
            else:
                logits = model.reason_once_unlearn(copy_data)

        else:  # "GOLD" or standard
            logits = model(copy_data.x, copy_data.edge_index)
        
    # breakpoint()
    return logits

# === Accuracy Function ===
def accuracy(y_true, pred, mask):
    return accuracy_score(y_true[mask], pred[mask])

def f1(y_true, pred, mask):
    return f1_score(y_true[mask], pred[mask], average='micro')

# === Exact Match Function ===
def exact_match(y1, y2, mask):
    return np.mean(y1[mask] == y2[mask])

# === Final aggregate results ===
def mean_std_str(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"



# === Collect metrics across runs ===
acc_original_runs, acc_gold_runs, acc_megu_runs = [], [], []
f1_original_runs, f1_gold_runs, f1_megu_runs = [], [], []

# Store pairwise metrics
pairwise_results = {"exact": {}, "l2": {}}

if model_name!="Projector":
    logits_original = load_and_predict(original_model_path)
else:
    logits_original = load_and_predict(original_model_path,model_type="Original")
pred_original = torch.argmax(F.softmax(logits_original, dim=-1), dim=1).cpu().numpy()

# loop for multiple runs
for run_number in range(num_runs):

    if unlearn_task == "node":
        num_nodes_local = data.x.size(0)
        unlearn_idx_path = (
            f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_nodes_{u_ratio}_{dataset}_{run_number}_nodes_{int(u_ratio * num_nodes)}.txt"
        )
        with open(unlearn_idx_path, "r") as f:
            unlearned_indices = list(map(int, f.readlines()))

        unlearned_mask = np.zeros_like(train_mask, dtype=bool)
        unlearned_mask[unlearned_indices] = True
        retained_train_mask = np.logical_and(train_mask, np.logical_not(unlearned_mask))
        unlearned_train_mask = np.logical_and(train_mask, unlearned_mask)

        regions = [
            # ("Retained Train Nodes", retained_train_mask),
            # ("Unlearned Train Nodes", unlearned_train_mask),
            ("Test Nodes", test_mask),
            # ("Full Dataset", None),
        ]
    else:
        regions = [
            ("Test Nodes", test_mask), 
            ("Full Dataset", None)
        ]

    # Adjust GOLD model path if multi-run
    if num_runs > 1:
        megu_model_path =  f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/{model_name}/{dataset}/node/{unlearn_ratio}{copy_str}/{model_name}_{dataset}_node_{unlearn_ratio}_{run_number}{base_model_str}.pt"
        if model_name=="Projector":
            gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/Projector_GOLD_{run_number}.pt"
        else:
            gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GOLD/{dataset}/node/{unlearn_ratio}{copy_str}/GOLD_{dataset}_node_{unlearn_ratio}_{run_number}{base_model_str}.pt"

    # === Load Predictions ===
    logits_gold     = load_and_predict(gold_model_path,run_number=run_number)
    logits_megu     = load_and_predict(megu_model_path, model_type=model_name, unlearned_param_path=megu_model_path,run_number=run_number)
    pred_gold       = torch.argmax(F.softmax(logits_gold, dim=-1), dim=1).cpu().numpy()
    pred_megu       = torch.argmax(F.softmax(logits_megu, dim=-1), dim=1).cpu().numpy()

    # === Accuracy & F1 ===
    for acc_list, f1_list, pred in [
        (acc_original_runs, f1_original_runs, pred_original),
        (acc_gold_runs,     f1_gold_runs,     pred_gold),
        (acc_megu_runs,     f1_megu_runs,     pred_megu),
    ]:
        acc_list.append(accuracy(y_true, pred, test_mask))
        f1_list.append(f1(y_true, pred, test_mask))

    # === Pairwise metrics ===
    comparisons = {
        "Gold vs Original":   (logits_gold, logits_original, pred_gold, pred_original),
        f"Gold vs {model_name}": (logits_gold, logits_megu, pred_gold, pred_megu),
        f"Original vs {model_name}": (logits_original, logits_megu, pred_original, pred_megu),
    }
    if False:
        comparisons = {
            f"Gold vs {model_name}": (logits_gold, logits_megu, pred_gold, pred_megu),
        }

    for pair, (log1, log2, p1, p2) in comparisons.items():
        pairwise_results["exact"].setdefault(pair, {r: [] for r, _ in regions})
        pairwise_results["l2"].setdefault(pair, {r: [] for r, _ in regions})
        #Handling Projector case which has extra feature
        if log1.size(1) != log2.size(1): 
            min_dim = min(log1.size(1), log2.size(1))
            log1 = log1[:, :min_dim]
            log2 = log2[:, :min_dim]
        log1, log2 = log1.to("cuda"), log2.to("cuda")

        for region, mask in regions:
            if mask is not None:
                exact = np.mean(p1[mask] == p2[mask])
                l2val = torch.norm((log1[mask] - log2[mask]), p=2, dim=1).mean().item()
            else:
                exact = np.mean(p1 == p2)
                l2val = torch.norm((log1 - log2), p=2, dim=1).mean().item()
            pairwise_results["exact"][pair][region].append(exact)
            pairwise_results["l2"][pair][region].append(l2val)

# === Reporting ===
print("\n=== Aggregate Results over all runs ===")
if True:
    print(f"Original Model Accuracy : {mean_std_str(acc_original_runs)}")
    print(f"Gold Unlearned Accuracy : {mean_std_str(acc_gold_runs)}")
print(f"{model_name} Unlearned Accuracy : {mean_std_str(acc_megu_runs)}")

print("\n=== Pairwise Fideliry (mean ± std) ===")
for pair, regions in pairwise_results["exact"].items():
    print(f"{pair}:")
    for region, vals in regions.items():
        print(f"  {region}: {mean_std_str(vals)}")

print("\n=== Pairwise Logit L2 Distance (mean ± std) ===")
for pair, regions in pairwise_results["l2"].items():
    print(f"{pair}:")
    for region, vals in regions.items():
        print(f"  {region}: {mean_std_str(vals)}")

print("="*60)
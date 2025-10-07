import logging
import torch

torch.cuda.empty_cache()
from utils import dataset_utils
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import numpy as np
import time
from torch_geometric.loader import DataLoader
from unlearning.unlearning_methods.GraphEraser.aggregation.optimal_aggregator import OptimalAggregator
from unlearning.unlearning_methods.GraphEraser.aggregation.optimal_edge_aggregator import OptimalEdgeAggregator
from dataset.original_dataset import original_dataset
from utils.dataset_utils import *
from unlearning.unlearning_methods.GraphEraser.aggregation.contra_aggregator_v2 import ContrastiveAggregator
import sys
import os
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
from sklearn.metrics import accuracy_score
import os
import argparse
from unlearning.unlearning_methods.Projector.utils.graph_projector_model_utils import Pro_GNN
import copy
from torch_sparse import SparseTensor

# Add the GULib-master/ root to sys.path
# sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
# from jaccard_script import load_and_predict




def load_and_predict(model_path, model_type="GOLD", unlearned_param_path=None, data=None,args=None):
    copy_data = copy.deepcopy(data)
    args['out_dim'] = data.y.max().item() + 1

    if model_type == 'Projector':
        extra_feats = torch.zeros(copy_data.x.size(0), device=copy_data.x.device)
        extra_feats[unlearned_nodes] = 1
        copy_data.x = torch.cat([copy_data.x, extra_feats.view(-1, 1)], dim=1)

        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
        W_shape = state_dict['W'].shape
        y_dims = W_shape[1]
        x_iters = 3
        y_iters = 3
        x_dims = copy_data.x.size(1)

        projector_args = {
            # 'dataset_name': dataset,
            'downstream_task': 'node',
            'base_model': args['base_model'],
            'hidden_dim': 64,
            'out_dim': y_dims,
            'x_iters': x_iters,
            'y_iters': y_iters,
            'use_adapt_gcs': False,
            'use_cross_entropy': True,
        }

        model = Pro_GNN(x_dims, y_dims, device="cuda", args=projector_args).to("cuda")
        model.load_state_dict(state_dict)

    elif model_type == 'GNNDelete':
        if args['base_model'] == "GCN":
            model = GCNDelete(args, data.num_node_features, args['out_dim'])
        elif args['base_model'] == "GAT":
            model = GATDelete(args, data.num_node_features, args['out_dim'])
        elif args['base_model'] == "GIN":
            model = GINDelete(args, data.num_node_features, args['out_dim'])

    else:  # Includes "GOLD" and "GIF"
        if args['base_model'] == "GCN":
            model = GCNNet(args, data.num_node_features, args['out_dim'])
        elif args['base_model'] == "GAT":
            model = GATNet(args, data.num_node_features, args['out_dim'])
        elif args['base_model'] == "GIN":
            model = GINNet(args, data.num_node_features, args['out_dim'])

    model = model.to("cuda")

    # Load and apply base weights
    

    # For GIF: load parameters and unlearned data (x_unlearn + edge_index_unlearn)
    if model_type == "GIF" or model_type == "IDEA":
        assert unlearned_param_path is not None, "GIF model requires a path to unlearned parameters"
        
        # Load unlearned weights
        params_esti = torch.load(unlearned_param_path)
        idx = 0
        for p in model.parameters():
            p.data = params_esti[idx].to(p.device)
            idx += 1

        # Derive the save directory from unlearned_param_path
        save_dir = os.path.dirname(unlearned_param_path)

        # Load x_unlearn and edge_index_unlearn
        copy_data.x_unlearn = torch.load(os.path.join(save_dir, "x_unlearn.pt")).to("cuda")
        copy_data.edge_index_unlearn = torch.load(os.path.join(save_dir, "edge_index_unlearn.pt")).to("cuda")
    
    else:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    with torch.no_grad():
        if model_type == "Projector":
            subgraph_data = copy_data.clone()
            subgraph_data.adj_t = SparseTensor.from_edge_index(copy_data.edge_index).t().to("cuda")
            subgraph_data.y_one_hot_train = F.one_hot(copy_data.y, y_dims).float().to("cuda")
            subgraph_data.root_n_id = torch.arange(copy_data.x.size(0), device='cuda')
            logits = model(subgraph_data)

        elif model_type == 'GNNDelete':
            logits = model(data.x, data.edge_index, mask_1hop=mask_1hop, mask_2hop=mask_2hop)

        elif model_type == "GIF" or model_type == 'IDEA':
            logits = model.reason_once_unlearn(copy_data)

        else:  # "GOLD" or standard
            logits = model(data.x, data.edge_index)

        preds = torch.argmax(F.softmax(logits, dim=-1), dim=1).cpu().numpy()

    return preds

def exact_match(y1, y2, mask):
    return np.mean(y1[mask] == y2[mask])

out_file = "/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/GraphEraser_utility_Stats.txt"
# static/global storage for all runs
call_count = 0
all_results = {
    # accuracy buckets
    "retained_acc": [],
    "unlearned_acc": [],
    "test_acc": [],
    "full_acc": [],
    
    # jaccards for 3 comparisons
    "gold_vs_ge": {
        "retained_jaccard": [],
        "unlearned_jaccard": [],
        "test_jaccard": [],
        "full_jaccard": []
    },
    "gold2_vs_ge": {
        "retained_jaccard": [],
        "unlearned_jaccard": [],
        "test_jaccard": [],
        "full_jaccard": []
    },
    "original_vs_ge": {
        "retained_jaccard": [],
        "unlearned_jaccard": [],
        "test_jaccard": [],
        "full_jaccard": []
    }
}


def region_wise_exact(ref, comp, retained_train_mask, unlearned_train_mask, test_mask, comp_type='gold_vs_ge'):
    results_map = {
        "Retained Train Nodes": ("retained_jaccard", retained_train_mask),
        "Unlearned Train Nodes": ("unlearned_jaccard", unlearned_train_mask),
        "Test Nodes": ("test_jaccard", test_mask),
        "Full Dataset": ("full_jaccard", None)
    }

    for name, (key, mask) in results_map.items():
        exact = exact_match(ref, comp, mask) if mask is not None else np.mean(ref == comp)

        # store in correct comparison bucket
        all_results[comp_type][key].append(exact)

    with open(out_file, "a") as f:
        f.write("-" * 40 + "\n")
        


class Aggregator:
    def __init__(self, run, target_model, data, shard_data, args,logger,affected_shard=None):
        self.logger = logger
        self.args = args

        self.data_store = original_dataset(self.args,logger)

        self.run = run
        self.target_model =target_model
        self.data = data
        self.shard_data = shard_data
        self.affected_shard = affected_shard
        self.num_shards = args['num_shards']

        u_ratio = self.args['unlearn_ratio']
        dataset = self.args['dataset_name']
        num_nodes = self.data.y.size(0)

        unlearn_idx_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_nodes_{u_ratio}_{dataset}_0_nodes_{int(u_ratio * num_nodes)}.txt"

        with open(unlearn_idx_path, "r") as f:
            unlearned_indices = list(map(int, f.readlines()))

        # train_mask = self.data.train_mask.cpu().numpy()
        # test_mask = self.data.test_mask.cpu().numpy()
        num_nodes = data.num_nodes

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)

        train_mask[data.train_indices] = True
        val_mask[data.val_indices] = True
        test_mask[data.test_indices] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        unlearned_mask = np.zeros_like(train_mask, dtype=bool)
        unlearned_mask[unlearned_indices] = True

        self.retained_train_mask = np.logical_and(train_mask, np.logical_not(unlearned_mask))
        self.unlearned_train_mask = np.logical_and(train_mask, unlearned_mask)
        self.test_mask = test_mask
        # breakpoint()

        

    def generate_posterior(self, suffix=""):
        
        if self.args["downstream_task"] == "node":




            self.true_label = self.data.y.detach().cpu().numpy()
            self.posteriors = {}
            if self.args['aggregator'] == 'contrastive':
                self.posteriors = []
            self.test_embeddings = []
            for shard in range(self.args['num_shards']):
                self.target_model.data = self.data
                if self.affected_shard is not None and shard in self.affected_shard:
                    load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
                    
                else:
                    load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")
                
                if self.args['aggregator'] == 'contrastive':
                    z, f = self.target_model.posterior_con(return_features=True, use_all=True)
                    self.posteriors.append(z)
                    #self.test_embeddings.append(torch.cat([f, z], 1))
                    self.test_embeddings.append(f)
                else:
                    if self.args["downstream_task"]=="node":
                        self.posteriors[shard] = self.target_model.posterior()
            if self.args['aggregator'] == 'contrastive':        
                self.posteriors = torch.stack(self.posteriors) 
                if len(self.test_embeddings):
                    self.test_embeddings = torch.stack(self.test_embeddings)
            self.logger.info("Saving posteriors.")
            save_posteriors(self.logger,self.args,self.posteriors, self.run, suffix)
                    
                    
        elif self.args["downstream_task"] == "edge":
            pos_edge_labels = torch.ones(self.shard_data[0].test_edge_index.size(1),dtype=torch.float32)
            neg_edge_labels = torch.zeros(self.shard_data[0].test_edge_index.size(1),dtype=torch.float32)
            edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
            self.true_label = edge_labels
            self.posteriors = {}
            if self.args['aggregator'] == 'contrastive':
                self.posteriors = []
            self.test_embeddings = []
            for shard in range(self.args['num_shards']):
                self.target_model.data = self.shard_data[shard]
                if self.affected_shard is not None and shard in self.affected_shard:
                    load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
                else:
                    load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")

                if self.args['aggregator'] == 'contrastive':
                    z, f = self.target_model.posterior_con(return_features=True)
                    self.posteriors.append(z)
                    #self.test_embeddings.append(torch.cat([f, z], 1))
                    self.test_embeddings.append(f)
                else:
                    if self.args["downstream_task"]=="node":
                        self.posteriors[shard] = self.target_model.posterior()
                    elif self.args["downstream_task"]=="edge":
                        self.posteriors[shard] = self.target_model.posterior_edge()
                        
            if self.args['aggregator'] == 'contrastive':        
                self.posteriors = torch.stack(self.posteriors) 
                if len(self.test_embeddings):
                    self.test_embeddings = torch.stack(self.test_embeddings)
            self.logger.info("Saving posteriors.")
            save_posteriors(self.logger,self.args,self.posteriors, self.run, suffix)
        else:
            self.posteriors = []
            self.true_label = []
            test_loader = DataLoader(self.shard_data[0][1], batch_size=64, shuffle=False)
            for tmpdata in test_loader:
                self.true_label.append(tmpdata.y)
            for shard in range(self.args['num_shards']):
                graph_data = self.shard_data[shard]
                if self.affected_shard is not None and shard in self.affected_shard:
                    load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
                else:
                    load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")
                test_loader = DataLoader(graph_data[1], batch_size=64, shuffle=False)
                tmp_list = []
                for tmpdata in test_loader:
                    tmpdata = tmpdata.cuda()
                    tmp_list.append(self.target_model.model(tmpdata.x, tmpdata.edge_index,batch = tmpdata.batch))
                posteriors = torch.cat(tmp_list)
                posteriors =  posteriors.squeeze(0) 
                self.posteriors.append(posteriors)
            self.posteriors = torch.stack(self.posteriors,dim = 0)
                
            save_posteriors(self.logger,self.args,self.posteriors, self.run, suffix)
                

    def aggregate(self, data):
        if self.args['aggregator'] == 'mean':
            aggregate_f1_score, posterior = self._mean_aggregator(data)
        elif self.args['aggregator'] == 'optimal':
            aggregate_f1_score, posterior = self._optimal_aggregator(data)
        elif self.args['aggregator'] == 'majority':
            aggregate_f1_score, posterior = self._majority_aggregator(data)
            posterior = None  # majority doesn't compute posterior directly
        elif self.args['aggregator'] == 'contrastive':
            aggregate_f1_score, t, posterior = self._contrastive_aggregator()
        else:
            raise Exception("unsupported aggregator.")

        global call_count
        if posterior is not None and call_count%2:
            # === GOLD Model Evaluation ===
            dataset = self.args['dataset_name']
            unlearn_ratio = f"ratio_{self.args['unlearn_ratio']:.2f}"
            unlearn_task = self.args['unlearn_task']
            run_str = ""
            if self.args["num_runs"]>1:
                run_str=f"_{self.run}"
            base_model_str=""
            if self.args["base_model"]!="GCN":
                base_model_str = "_" + self.args["base_model"]
            original_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/{unlearn_task}/{self.args['base_model']}"
            gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GOLD/{dataset}/{unlearn_task}/{unlearn_ratio}/GOLD_{dataset}_node_{unlearn_ratio}{run_str}{base_model_str}.pt"
            data_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/processed/transductive/{dataset}0.8_0_0.2.pkl"
            # gold_model_path_copy = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GOLD/{dataset}/{unlearn_task}/{unlearn_ratio}_copy/GOLD_{dataset}_node_{unlearn_ratio}.pt"

            with open(data_path, "rb") as f:
                data = pickle.load(f)
            data = data.to("cuda")
            y_true = data.y.cpu().numpy()
            # test_mask = data.test_mask.cpu().numpy()
            # train_mask = data.train_mask.cpu().numpy()
            num_nodes = data.num_nodes

            train_mask = np.zeros(num_nodes, dtype=bool)
            val_mask = np.zeros(num_nodes, dtype=bool)
            test_mask = np.zeros(num_nodes, dtype=bool)

            train_mask[data.train_indices] = True
            val_mask[data.val_indices] = True
            test_mask[data.test_indices] = True

            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = test_mask

            model_args = {
                'dataset_name': 'cora',
                'downstream_task': 'node',
                'base_model': self.args["base_model"],
                'hidden_dim': 64,
                # 'out_dim': data.y.max().item() + 1
            }

            gold_preds = load_and_predict(gold_model_path, model_type="GOLD", data=data,args=model_args)
            original_preds = load_and_predict(original_model_path, model_type="GOLD", data=data,args=model_args)
            aggregated_preds = posterior.argmax(dim=1).cpu().numpy()
            # gold_preds_copy = load_and_predict(gold_model_path_copy, model_type="GOLD", data=data)

            copy_str=""
            if self.args["use_copy"]:
                copy_str=" 2"

            # save them in static/global dict
            retained_acc = accuracy_score(aggregated_preds[self.retained_train_mask], data.y[self.retained_train_mask].cpu())
            unlearned_acc = accuracy_score(aggregated_preds[self.unlearned_train_mask], data.y[self.unlearned_train_mask].cpu())
            test_acc = accuracy_score(aggregated_preds[test_mask], data.y[self.test_mask].cpu())
            full_acc = accuracy_score(aggregated_preds, data.y.cpu())

            all_results["retained_acc"].append(retained_acc)
            all_results["unlearned_acc"].append(unlearned_acc)
            all_results["test_acc"].append(test_acc)
            all_results["full_acc"].append(full_acc)


            # ======== Gold vs GraphEraser ==========
            region_wise_exact(
                ref=gold_preds,
                comp=aggregated_preds,
                retained_train_mask=self.retained_train_mask,
                unlearned_train_mask=self.unlearned_train_mask,
                test_mask=self.test_mask,
                comp_type='gold_vs_ge'
            )
            # # ======== Gold 2 vs GraphEraser ==========
            # region_wise_exact(
            #     ref=gold_preds_copy,
            #     comp=aggregated_preds,
            #     retained_train_mask=self.retained_train_mask,
            #     unlearned_train_mask=self.unlearned_train_mask,
            #     test_mask=self.test_mask,
            #     comp_type='gold2_vs_ge'
            # )
            # ======== Original vs GraphEraser ==========
            region_wise_exact(
                ref=original_preds,
                comp=aggregated_preds,
                retained_train_mask=self.retained_train_mask,
                unlearned_train_mask=self.unlearned_train_mask,
                test_mask=self.test_mask,
                comp_type='original_vs_ge'
            )

            # final run
            if self.run==self.args["num_runs"]-1:   
                with open(out_file, "a") as f:
                    # --- Accuracy metrics ---
                    f.write(f"=== Dataset: {dataset}, {unlearn_ratio}, {self.args['base_model']} ===\n")
                    f.write("=== Accuracies ===\n")
                    print("\n=== Accuracies ===")
                    for k in ["retained_acc", "unlearned_acc", "test_acc", "full_acc"]:
                        vals = all_results[k]
                        mean_val = np.mean(vals) if vals else float("nan")
                        std_val = np.std(vals) if vals else float("nan")
                        line = f"{k}: mean={mean_val:.4f}, std={std_val:.4f}"
                        print(line)
                        f.write(line + "\n")

                    # --- Jaccards (3 types) ---
                    for comp_type in ["gold_vs_ge", "original_vs_ge"]:
                        header = f"\n=== {comp_type.upper()} Jaccard ===\n"
                        print(header)
                        f.write(header)

                        for k, vals in all_results[comp_type].items():
                            mean_val = np.mean(vals) if vals else float("nan")
                            std_val = np.std(vals) if vals else float("nan")
                            line = f"{k}: mean={mean_val:.4f}, std={std_val:.4f}"
                            print(line)
                            f.write(line + "\n")

        call_count+=1
        return aggregate_f1_score


    def _mean_aggregator(self,data):
        posterior = self.posteriors[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard]

        posterior = posterior / self.num_shards
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), posterior
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, posterior.detach().cpu().numpy(),average="micro"), posterior
        elif self.args["downstream_task"]=="graph":
            posterior = posterior.detach().cpu()
            pred = posterior.argmax(dim=1)
            self.true_label = torch.concat(self.true_label,dim=0).cpu()
            return accuracy_score(self.true_label,pred), posterior

    def _majority_aggregator(self,data):
        pred_labels = []
        for shard in range(self.num_shards):
            edge_pred = torch.where(self.posteriors[shard] > 0.5, torch.tensor(1), torch.tensor(0))
            pred_labels.append(edge_pred.cpu().numpy())
        pred_labels = np.stack(pred_labels)
        pred_label = np.argmax(
            np.apply_along_axis(np.bincount, axis=0, arr=pred_labels, minlength=self.posteriors[0].shape[0]), axis=0)
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, pred_label, average="micro"), posterior
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, pred_label, average="micro"), posterior
        elif self.args["downstream_task"]=="graph":
            return accuracy_score(self.true_label, pred_label, average="micro"), posterior

    def _optimal_aggregator(self,data):
        if self.args["downstream_task"]=="node":
            optimal = OptimalAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        elif self.args["downstream_task"]=="edge":
            optimal = OptimalEdgeAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        optimal.generate_train_data(data)
        weight_para = optimal.optimization()
        save_optimal_weight(self.logger,self.args, weight_para, run=self.run)

        posterior = self.posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            # print(self.posteriors[shard],weight_para[shard])
            posterior += self.posteriors[shard] * weight_para[shard]
        
        # print(posterior,self.true_label_edge)
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), posterior
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, posterior.detach().cpu().numpy(), average="micro"), posterior

    def _contrastive_aggregator(self):
        proj = ContrastiveAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        proj._generate_train_data()
        
        start_time = time.time()
        proj_model = proj.optimization()#.to(self.posteriors.device)
        proj_model.eval()
        if self.args['base_model'] == 'GIN':
            self.test_embeddings = torch.tanh(self.test_embeddings)
        self.test_embeddings = self.test_embeddings.permute(1, 0, 2).to(next(proj_model.parameters()).device)
        #start_time = time.time()
        posterior = proj_model(self.test_embeddings, is_eval=True)
        aggr_time = time.time() - start_time
        
        dataset_utils.save_optimal_weight(self.logger,self.args,proj_model, run=self.run)
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), aggr_time, posterior
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), aggr_time, posterior
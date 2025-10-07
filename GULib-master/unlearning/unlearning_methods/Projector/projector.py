import os
import time

import numpy as np
import torch
import scipy.sparse as sp
import torch_sparse
from ogb.nodeproppred import Evaluator
from config import root_path,unlearning_path
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric import seed_everything
from tqdm import tqdm
import torch_geometric.transforms as T
from unlearning.unlearning_methods.Projector.utils.train_utils import pred_test
from unlearning.unlearning_methods.Projector.utils.graph_projector_model_utils import Pro_GNN
import copy
from config import BLUE_COLOR,RESET_COLOR
from sklearn.metrics import f1_score, accuracy_score,recall_score,roc_auc_score
from memory_profiler import profile
from sklearn.metrics import jaccard_score
class projector():
    """
    Projector class Projects the parameters to a subspace that is irrelevant to the node features that need to be forgotten and forgets the node features.
    It supports node and feature unlearning requests, and downstream tasks of node classification.

    Class Attributes:
        args (dict): Configuration parameters for the unlearning process.

        logger (Logger): Logger instance for recording informational and debugging messages.

        model_zoo (ModelZoo): Collection of pre-trained models available for training and evaluation within the pipeline.
    """
    def __init__(self,args,logger,model_zoo):
        self.args = args
        self.logger =logger
        self.data = model_zoo.data
        self.data_copy = copy.deepcopy(model_zoo.data)
        self.model_zoo = model_zoo
        self.model = self.model_zoo.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.f1 = 0
        self.auc = 0
        self.run = 0
        num_runs = self.args["num_runs"]
        self.direct_average_f1 = np.zeros(num_runs)
        self.reuse_average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)
        self.avg_pro_gnn_time = np.zeros(num_runs)
        self.avg_total_unlearning_time = np.zeros(num_runs)



    # @profile
    def run_exp_mem(self):
        for self.run in range(self.args["num_runs"]):
            self.data = copy.deepcopy(self.data_copy)
            if self.args["dataset_name"] =="ogbn-arxiv":
                self.evaluator = Evaluator(name=self.args["dataset_name"])
            # Generate augment node feats
            # num_train_nodes = len(self.data.train_indices)
            # train_ind_perm = np.random.permutation(self.data.train_indices)
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            delete_nodes_all = np.loadtxt(path_un, dtype=int)
            self.unlearning_nodes = delete_nodes_all

            extra_feats = torch.zeros(self.data.x.size(0))
            extra_feats[delete_nodes_all] = 1

            self.data.x = torch.cat([self.data.x, extra_feats.view(-1, 1)], dim=1)
            self.data.y[delete_nodes_all] = self.data.num_classes
            self.data.adj_t = SparseTensor(row=self.data.edge_index[1], col=self.data.edge_index[0])
            self.data.adj_t = torch_sparse.fill_diag(self.data.adj_t.to_symmetric(), 1)
            self.data.y_one_hot_train = F.one_hot(
                self.data.y.squeeze(), self.data.num_classes + 1).float()
            self.data.y_one_hot_train[self.data.test_indices, :] = 0
            num_nodes = self.data.x.size(0)
            self.data.node_inds = torch.arange(self.data.x.size(0))

            # remove unlearned nodes from training indices
            remain_train_indices = np.setdiff1d(self.data.train_indices, self.unlearning_nodes)

            # ---- Remove edges connected to deleted nodes ----
            edge_index = self.data.edge_index
            mask = ~(
                torch.isin(edge_index[0], torch.tensor(self.unlearning_nodes)) |
                torch.isin(edge_index[1], torch.tensor(self.unlearning_nodes))
            )
            edge_index_gold = edge_index[:, mask]

            # create a clean copy of data for GOLD training
            self.data_gold = copy.deepcopy(self.data)
            self.data_gold.train_indices = remain_train_indices
            self.data_gold.edge_index = edge_index_gold
            self.data_gold.adj_t = SparseTensor(row=edge_index_gold[1],
                                        col=edge_index_gold[0],
                                        sparse_sizes=(self.data_gold.num_nodes, self.data_gold.num_nodes))
            self.data_gold.adj_t = torch_sparse.fill_diag(self.data_gold.adj_t.to_symmetric(), 1)
            self.data_gold.y_one_hot_train = F.one_hot(
                self.data.y.squeeze(), self.data_gold.num_classes + 1).float()
            self.data_gold.y_one_hot_train[self.data_gold.test_indices, :] = 0
            num_nodes = self.data.x.size(0)
            self.data_gold.node_inds = torch.arange(self.data_gold.x.size(0))

            self.gold_train_loader = ShaDowKHopSampler(
                self.data_gold,
                depth=2,
                num_neighbors=self.args["hop_neighbors"],
                batch_size=256,
                shuffle=True,
                node_idx=torch.tensor(remain_train_indices)
            )

            self.train_loader = ShaDowKHopSampler(self.data,
                                            depth=2,
                                            num_neighbors=self.args["hop_neighbors"],
                                            batch_size=256,
                                            shuffle=True,
                                            node_idx=torch.tensor(self.data.train_indices))

            self.all_loader = ShaDowKHopSampler(self.data,
                                        depth=2,
                                        num_neighbors=self.args["hop_neighbors"],
                                        batch_size=1024,
                                        shuffle=False)
            x_dims = self.data.x.size(1)
            y_dims = self.data.y_one_hot_train.size(1)

            x_dims_gold = self.data_gold.x.size(1)
            y_dims_gold = self.data_gold.y_one_hot_train.size(1)

            fn = os.path.join(os.getcwd(),
                            "data","model","node_level",self.args["dataset_name"],"Projector.pt" )
            start_time = time.time()
            if os.path.exists(fn) and not self.args["regen_model"]:
                model_optim = Pro_GNN(x_dims, y_dims, self.device, self.args).to(self.device)
                model_optim.load_state_dict(torch.load(fn))
            else:
                model_optim = self.pre_train()
                torch.save(model_optim.state_dict(), fn)
            print("train model time", time.time() - start_time)
            self.evaluation_reuse_labels(model_optim)

            fn_gold = os.path.join(os.getcwd(),
                            "data","model","node_level",self.args["dataset_name"],f"Projector_GOLD_{str(self.run)}.pt" )
            if os.path.exists(fn_gold) and not self.args["regen_model"]:
                model_optim_gold = Pro_GNN(x_dims_gold, y_dims_gold, self.device, self.args).to(self.device)
                model_optim_gold.load_state_dict(torch.load(fn_gold))
            else:
                model_optim_gold = self.pre_train()
                torch.save(model_optim.state_dict(), fn_gold)

            self.evaluation_reuse_labels(model_optim_gold)

            # projection-based unlearning
            remain_nodes = np.arange(num_nodes)
            feat_dim = self.data.x.size(1)
            label_dim = self.data.y_one_hot_train.size(1)

            W_optim = model_optim.W.data.clone().cpu()

            start_all_time = time.time()
            batch = self.args["parallel_unlearning"]
            delete_node_batch = [[] for _ in range(batch)]
            for i, node_i in enumerate(delete_nodes_all):
                delete_node_batch[i % batch].append(node_i)

            start_time = time.time()
            for cnt, delete_node_batch_i in enumerate(delete_node_batch):
                # get remain node feats
                remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
                remain_node_feats = self.data.x[remain_nodes]
                remain_node_label = self.data.y_one_hot_train[remain_nodes]

                # unlearning
                extra_channel_norm_before = 0
                extra_channel_norm_after = 0
                W_optim_part = torch.split(W_optim, [feat_dim for _ in range(
                    self.args["x_iters"] + 1)] + [label_dim for _ in range(self.args["y_iters"])])
                W_optim_part_unlearn = []

                for W_part in W_optim_part[:self.args["x_iters"] + 1]:
                    XtX = remain_node_feats.T @ remain_node_feats
                    XtX_inv = torch.linalg.pinv(XtX)
                    proj_W_optim = XtX @ XtX_inv @ W_part
                    W_optim_part_unlearn.append(proj_W_optim)
                    extra_channel_norm_before += W_part[-1, :].norm(2).item()
                    extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()

                for W_part in W_optim_part[-self.args["y_iters"]:]:
                    XtX = remain_node_label.T @ remain_node_label
                    XtX_inv = torch.linalg.pinv(XtX)
                    proj_W_optim = XtX @ XtX_inv @ W_part
                    W_optim_part_unlearn.append(proj_W_optim)
                    extra_channel_norm_before += W_part[-1, :].norm(2).item()
                    extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()

                print('extra_channel_norm_before', extra_channel_norm_before,
                    'extra_channel_norm_after', extra_channel_norm_after)
                W_optim = torch.cat(W_optim_part_unlearn, dim=0)

                # evaluate
                print('Unlearning step %d >>>' % (cnt + 1))

            self.avg_training_time[self.run] = time.time() - start_time
            self.avg_total_unlearning_time[self.run] = time.time() - start_all_time
            self.logger.info("Total time:{}".format(self.avg_training_time[self.run]) )
            self.logger.info("Total unl time:{}".format(time.time() - start_all_time) )

            model_unlearn = copy.deepcopy(model_optim)
            model_unlearn.W.data = W_optim

            self.evaluation_reuse_labels(model_unlearn,True)

            self.unlearning_num = self.args["num_unlearned_nodes"]
            # self.mia_attack()
            self.average_auc[self.run] = self.auc

            # unlearning_model_name = self.args["unlearning_model"]  # e.g., 'GNNDelete'
            # dataset = self.args["dataset_name"]                    # e.g., 'cora'
            # unlearn_ratio = self.args["proportion_unlearned_nodes"]  # e.g., 0.01

            # save_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/{unlearning_model_name}/{dataset}/node/ratio_{unlearn_ratio:.2f}"
            # os.makedirs(save_path, exist_ok=True)

            # save_file = f"{unlearning_model_name}_{dataset}_node_ratio_{unlearn_ratio:.2f}.pt"
            # torch.save({'model_state': model_unlearn.state_dict()}, os.path.join(save_path, save_file))

            # self.logger.info(f"Unlearned model saved to {os.path.join(save_path, save_file)}")
            # breakpoint()
        
        self.logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024/1024}MB")
        self.logger.info(f"Max Cached: {torch.cuda.max_memory_reserved()/1024/1024}MB")

        self.logger.info(
        "{}Performance Metrics:\n"
        " - Average Direct F1 Score: {:.4f} ± {:.4f}\n"
        " - Average Reuse F1 Score: {:.4f} ± {:.4f}\n"
        " - Average AUC Score: {:.4f} ± {:.4f}\n"
        " - Average Unlearning Time: {:.4f} ± {:.4f} seconds\n"
        " - Average Training Time: {:.4f} ± {:.4f} seconds\n{}".format(
            BLUE_COLOR,
            np.mean(self.direct_average_f1), np.std(self.direct_average_f1),
            np.mean(self.reuse_average_f1), np.std(self.reuse_average_f1),
            np.mean(self.average_auc), np.std(self.average_auc),
            np.mean(self.avg_training_time), np.std(self.avg_training_time),
            np.mean(self.avg_pro_gnn_time), np.std(self.avg_pro_gnn_time),
            RESET_COLOR
            )
        )

        with open("efficiency_stats.txt", "a") as f:
            f.write("==============  Efficiency Statistics ============== \n")
            f.write(f"Runs: {self.args['num_runs']}, Dataset: {self.args['dataset_name']},Technique: {self.args['unlearning_methods']}\n")
            f.write(f"Total Unlearning time (s): mean={np.mean(self.avg_total_unlearning_time):.6f}, std={np.std(self.avg_total_unlearning_time):.6f}\n")
            f.write(f"Total Unlearning time (s): mean={np.mean(self.avg_training_time):.6f}, std={np.std(self.avg_training_time):.6f}\n")
    
    def run_exp(self):
        """
        Executes the main experimental pipeline, including data preparation, model training, projection-based unlearning, and evaluation. It iterates over multiple runs, performs unlearning on specified nodes, updates model parameters, and records performance metrics such as F1 score and AUC.
        """
        for self.run in range(self.args["num_runs"]):
            self.data = copy.deepcopy(self.data_copy)
            if self.args["dataset_name"] =="ogbn-arxiv":
                self.evaluator = Evaluator(name=self.args["dataset_name"])
            # Generate augment node feats
            # num_train_nodes = len(self.data.train_indices)
            # train_ind_perm = np.random.permutation(self.data.train_indices)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            delete_nodes_all = np.loadtxt(path_un, dtype=int)
            self.unlearning_nodes = delete_nodes_all

            extra_feats = torch.zeros(self.data.x.size(0))
            extra_feats[delete_nodes_all] = 1

            self.data.x = torch.cat([self.data.x, extra_feats.view(-1, 1)], dim=1)
            self.data.y[delete_nodes_all] = self.data.num_classes
            self.data.adj_t = SparseTensor(row=self.data.edge_index[1], col=self.data.edge_index[0])
            self.data.adj_t = torch_sparse.fill_diag(self.data.adj_t.to_symmetric(), 1)
            self.data.y_one_hot_train = F.one_hot(
                self.data.y.squeeze(), self.data.num_classes + 1).float()
            self.data.y_one_hot_train[self.data.test_indices, :] = 0
            num_nodes = self.data.x.size(0)
            self.data.node_inds = torch.arange(self.data.x.size(0))

            # remove unlearned nodes from training indices
            remain_train_indices = np.setdiff1d(self.data.train_indices, self.unlearning_nodes)

            # ---- Remove edges connected to deleted nodes ----
            edge_index = self.data.edge_index
            mask = ~(
                torch.isin(edge_index[0], torch.tensor(self.unlearning_nodes)) |
                torch.isin(edge_index[1], torch.tensor(self.unlearning_nodes))
            )
            edge_index_gold = edge_index[:, mask]

            # create a clean copy of data for GOLD training
            self.data_gold = copy.deepcopy(self.data)
            self.data_gold.train_indices = remain_train_indices
            self.data_gold.edge_index = edge_index_gold
            self.data_gold.adj_t = SparseTensor(row=edge_index_gold[1],
                                        col=edge_index_gold[0],
                                        sparse_sizes=(self.data_gold.num_nodes, self.data_gold.num_nodes))
            self.data_gold.adj_t = torch_sparse.fill_diag(self.data_gold.adj_t.to_symmetric(), 1)
            self.data_gold.y_one_hot_train = F.one_hot(
                self.data.y.squeeze(), self.data_gold.num_classes + 1).float()
            self.data_gold.y_one_hot_train[self.data_gold.test_indices, :] = 0
            num_nodes = self.data.x.size(0)
            self.data_gold.node_inds = torch.arange(self.data_gold.x.size(0))

            self.gold_train_loader = ShaDowKHopSampler(
                self.data_gold,
                depth=2,
                num_neighbors=self.args["hop_neighbors"],
                batch_size=256,
                shuffle=True,
                node_idx=torch.tensor(remain_train_indices)
            )

            self.train_loader = ShaDowKHopSampler(self.data,
                                            depth=2,
                                            num_neighbors=self.args["hop_neighbors"],
                                            batch_size=256,
                                            shuffle=True,
                                            node_idx=torch.tensor(self.data.train_indices))

            self.all_loader = ShaDowKHopSampler(self.data,
                                        depth=2,
                                        num_neighbors=self.args["hop_neighbors"],
                                        batch_size=1024,
                                        shuffle=False)
            x_dims = self.data.x.size(1)
            y_dims = self.data.y_one_hot_train.size(1)

            x_dims_gold = self.data_gold.x.size(1)
            y_dims_gold = self.data_gold.y_one_hot_train.size(1)

            fn = os.path.join(os.getcwd(),
                            "data","model","node_level",self.args["dataset_name"],"Projector.pt" )
            start_time = time.time()
            if os.path.exists(fn) and not self.args["regen_model"]:
                model_optim = Pro_GNN(x_dims, y_dims, self.device, self.args).to(self.device)
                model_optim.load_state_dict(torch.load(fn))
            else:
                model_optim = self.pre_train()
                torch.save(model_optim.state_dict(), fn)
            self.avg_pro_gnn_time[self.run] = time.time() - start_time
            print("train model time", time.time() - start_time)
            self.evaluation_reuse_labels(model_optim)

            fn_gold = os.path.join(os.getcwd(),
                            "data","model","node_level",self.args["dataset_name"],f"Projector_GOLD_{str(self.run)}.pt" )
            if os.path.exists(fn_gold) and not self.args["regen_model"]:
                model_optim_gold = Pro_GNN(x_dims_gold, y_dims_gold, self.device, self.args).to(self.device)
                model_optim_gold.load_state_dict(torch.load(fn_gold))
            else:
                model_optim_gold = self.pre_train()
                torch.save(model_optim.state_dict(), fn_gold)

            self.evaluation_reuse_labels(model_optim_gold)



            # projection-based unlearning
            remain_nodes = np.arange(num_nodes)
            feat_dim = self.data.x.size(1)
            label_dim = self.data.y_one_hot_train.size(1)

            W_optim = model_optim.W.data.clone().cpu()

            start_all_time = time.time()
            batch = self.args["parallel_unlearning"]
            delete_node_batch = [[] for _ in range(batch)]
            for i, node_i in enumerate(delete_nodes_all):
                delete_node_batch[i % batch].append(node_i)

            start_time = time.time()
            for cnt, delete_node_batch_i in enumerate(delete_node_batch):
                # get remain node feats
                remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
                remain_node_feats = self.data.x[remain_nodes]
                remain_node_label = self.data.y_one_hot_train[remain_nodes]

                # unlearning
                extra_channel_norm_before = 0
                extra_channel_norm_after = 0
                W_optim_part = torch.split(W_optim, [feat_dim for _ in range(
                    self.args["x_iters"] + 1)] + [label_dim for _ in range(self.args["y_iters"])])
                W_optim_part_unlearn = []

                for W_part in W_optim_part[:self.args["x_iters"] + 1]:
                    XtX = remain_node_feats.T @ remain_node_feats
                    XtX_inv = torch.linalg.pinv(XtX)
                    proj_W_optim = XtX @ XtX_inv @ W_part
                    W_optim_part_unlearn.append(proj_W_optim)
                    extra_channel_norm_before += W_part[-1, :].norm(2).item()
                    extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()

                for W_part in W_optim_part[-self.args["y_iters"]:]:
                    XtX = remain_node_label.T @ remain_node_label
                    XtX_inv = torch.linalg.pinv(XtX)
                    proj_W_optim = XtX @ XtX_inv @ W_part
                    W_optim_part_unlearn.append(proj_W_optim)
                    extra_channel_norm_before += W_part[-1, :].norm(2).item()
                    extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()

                print('extra_channel_norm_before', extra_channel_norm_before,
                    'extra_channel_norm_after', extra_channel_norm_after)
                W_optim = torch.cat(W_optim_part_unlearn, dim=0)

                # evaluate
                print('Unlearning step %d >>>' % (cnt + 1))

            self.avg_training_time[self.run] = time.time() - start_time
            self.avg_total_unlearning_time[self.run] = time.time() - start_all_time
            self.logger.info("Total time:{}".format(self.avg_training_time[self.run]) )
            self.logger.info("Total unl time:{}".format(time.time() - start_all_time) )
            model_unlearn = copy.deepcopy(model_optim)
            model_unlearn.W.data = W_optim

            self.evaluation_reuse_labels(model_unlearn,True)

            self.unlearning_num = self.args["num_unlearned_nodes"]

            unlearning_model_name = "Projector"  # e.g., 'GNNDelete'
            dataset = self.args["dataset_name"]                    # e.g., 'cora'
            unlearn_ratio = self.args["proportion_unlearned_nodes"]  # e.g., 0.01

            copy_str=""
            if self.args['use_copy']:
                copy_str="_copy"

            run_str=""
            if self.args['num_runs']>1:
                run_str = "_" + str(self.run)

            base_model_str = ""
            if self.args['base_model']!="GCN":  
                base_model_str = "_" + self.args['base_model']

            save_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/{unlearning_model_name}/{dataset}/node/ratio_{unlearn_ratio:.2f}{copy_str}"
            os.makedirs(save_path, exist_ok=True)

            save_file = f"{unlearning_model_name}_{dataset}_node_ratio_{unlearn_ratio:.2f}{run_str}{base_model_str}.pt"
            torch.save({'model_state': model_unlearn.state_dict()}, os.path.join(save_path, save_file))

            # self.logger.info(f"Unlearned model saved to {os.path.join(save_path, save_file)}")
            if self.args["attack"]:
                self.mia_attack()
            self.average_auc[self.run] = self.auc
            # self.logger.info("average_f1:{}".format(self.average_f1[self.run]) )
            # self.logger.info("average_auc:{}".format(self.average_auc[self.run]) )

                # ---- Store AUC and GOLD AUC results in file ----
        with open("/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/MIA_stats.txt", "a") as f:
            f.write(
                "{} Average MIA {} Score: {:.4f} ± {:.4f}\n".format(
                    self.args["dataset_name"],
                    self.args["unlearning_methods"],
                    np.mean(self.average_auc),
                    np.std(self.average_auc)
                )
            )
        self.logger.info(
        "{}Performance Metrics:\n"
        " - Average Direct F1 Score: {:.4f} ± {:.4f}\n"
        " - Average Reuse F1 Score: {:.4f} ± {:.4f}\n"
        " - Average AUC Score: {:.4f} ± {:.4f}\n"
        " - Average Unlearning Time: {:.4f} ± {:.4f} seconds\n"
        " - Average Training Time: {:.4f} ± {:.4f} seconds\n{}".format(
            BLUE_COLOR,
            np.mean(self.direct_average_f1), np.std(self.direct_average_f1),
            np.mean(self.reuse_average_f1), np.std(self.reuse_average_f1),
            np.mean(self.average_auc), np.std(self.average_auc),
            np.mean(self.avg_training_time), np.std(self.avg_training_time),
            np.mean(self.avg_pro_gnn_time), np.std(self.avg_pro_gnn_time),
            RESET_COLOR
            )
        )
        with open("efficiency_stats.txt", "a") as f:
            f.write("==============  Efficiency Statistics ============== \n")
            f.write(f"Runs: {self.args['num_runs']}, Dataset: {self.args['dataset_name']},Technique: {self.args['unlearning_methods']}\n")
            f.write(f"Total Unlearning time (s): mean={np.mean(self.avg_total_unlearning_time):.6f}, std={np.std(self.avg_total_unlearning_time):.6f}\n")
            f.write(f"Total Unlearning time (s): mean={np.mean(self.avg_training_time):.6f}, std={np.std(self.avg_training_time):.6f}\n")
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
                print(f"Peak GPU memory allocated in Projector: {peak_mem:.2f} MB")
                f.write(f"Peak GPU memory allocated in Projector: {peak_mem:.2f} MB\n")

    def pre_train(self):
        """
        Pre-trains the model using the training data and optimizer settings. The function returns the model with the best validation performance.
        """
        best_valid_score = 0

        model = self.model
        model = model.to(self.device)
        if self.args["require_linear_span"]:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args["opt_lr"], momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters())

        for epoch in tqdm(range(1, 1 + self.args["num_epochs"])):
            # training
            seed_everything(self.args["random_seed"])

            # pbar = tqdm(total=len(self.train_loader))
            # pbar.set_description('Epoch %d' % epoch)

            model.train()
            for subgraph_data in self.train_loader:
                subgraph_data.adj_t = SparseTensor(row=subgraph_data.edge_index[1], col=subgraph_data.edge_index[0],sparse_sizes=(subgraph_data.num_nodes, subgraph_data.num_nodes))
                loss = model.loss(subgraph_data.to(self.device), self.args["lam"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # pbar.update(1)
            # pbar.close()

            # evaluate
            model.eval()
            seed_everything(self.args["random_seed"])  # epoch > 0
            with torch.no_grad():
                all_score = []

                for subgraph_data in self.all_loader:
                    subgraph_data.adj_t = SparseTensor(row=subgraph_data.edge_index[1], col=subgraph_data.edge_index[0])
                    score = model(subgraph_data.to(self.device))
                    all_score.append(score.detach().cpu())

                all_score = torch.cat(all_score, dim=0)
            if self.args["dataset_name"] =="ogbn-arxiv":
                train_acc, val_acc, test_acc = pred_test(all_score, self.data,
                                                     self.evaluator)
            else:
                pred = all_score
                mask = self.data.train_mask
                train_acc = pred[mask].max(1)[1].eq(
                    self.data.y[mask]).sum().item() / mask.sum().item()
                mask = self.data.val_mask
                if mask.sum() != 0:
                    val_acc = pred[mask].max(1)[1].eq(
                        self.data.y[mask]).sum().item() / mask.sum().item()
                mask = self.data.test_mask
                test_acc = pred[mask].max(1)[1].eq(
                    self.data.y[mask]).sum().item() / mask.sum().item()

            if (epoch+1) % 10 == 0:
                self.logger.info(
                    f"Epoch: {epoch}, Train: {train_acc:.4f}, Test: {test_acc:.4f}"
                )

            if train_acc > best_valid_score:
                best_valid_score = test_acc
                best_params = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_params)
        return model

    @torch.no_grad()
    def evaluation_reuse_labels(self,model,is_unlearning=False,is_gold=False):
        """
        Evaluates the model's performance by computing both direct predictions and predictions with reused labels.
        Updates the average F1 and AUC scores based on the evaluation results.
        """
        model = model.to(self.device)

        # directly predict
        all_pred = []
        test_acc = 0
        seed_everything(self.args["random_seed"])
        for subgraph_data in self.all_loader:
            subgraph_data.adj_t = SparseTensor(row=subgraph_data.edge_index[1], col=subgraph_data.edge_index[0])
            pred = model(subgraph_data.to(self.device))
            all_pred.append(pred.detach().cpu())

        all_pred = torch.cat(all_pred, dim=0)
        if is_unlearning is False:
            self.original_softlabels =  F.softmax(all_pred,dim=1)
        else:
            self.unlearning_softlabels = F.softmax(all_pred,dim=1)

        if self.args["dataset_name"] =="ogbn-arxiv":
            train_acc, val_acc, test_acc = pred_test(all_pred, self.data,
                                                     self.evaluator)
        else:
            pred = all_pred
            mask = self.data.train_mask
            train_acc = pred[mask].max(1)[1].eq(
                self.data.y[mask]).sum().item() / mask.sum().item()
            mask = self.data.val_mask
            if mask.sum()!=0:
                val_acc = pred[mask].max(1)[1].eq(
                    self.data.y[mask]).sum().item() / mask.sum().item()
            mask = self.data.test_mask
            test_acc = pred[mask].max(1)[1].eq(
                self.data.y[mask]).sum().item() / mask.sum().item()
        print(
            f"Direct predict >>> Train: {train_acc:.4f}, Test: {test_acc:.4f}")
        self.direct_average_f1[self.run] = test_acc

        # reuse predicted labels
        y_one_hot_tmp = copy.deepcopy(self.data.y_one_hot_train)
        y_one_hot_tmp[self.data.test_indices] = F.one_hot(
            all_pred[self.data.test_indices].argmax(dim=-1, keepdim=True).squeeze(),
            self.data.y_one_hot_train.size(1)
        ).float()

        # label reuse
        all_pred = []

        seed_everything(self.args["random_seed"])
        for subgraph_data in self.all_loader:
            subgraph_data.adj_t = SparseTensor(row=subgraph_data.edge_index[1], col=subgraph_data.edge_index[0])
            subgraph_data.y_one_hot_train = y_one_hot_tmp[subgraph_data.node_inds]
            pred = model(subgraph_data.to(self.device))
            all_pred.append(pred.detach().cpu())

        all_pred = torch.cat(all_pred, dim=0)
        if self.args["dataset_name"] =="ogbn-arxiv":
            train_acc, val_acc, test_acc = pred_test(all_pred, self.data,
                                                     self.evaluator)
        else:
            pred = all_pred
            mask = self.data.train_mask
            train_acc = pred[mask].max(1)[1].eq(
                self.data.y[mask]).sum().item() / mask.sum().item()
            mask = self.data.val_mask
            if mask.sum()!=0:
                val_acc = pred[mask].max(1)[1].eq(
                    self.data.y[mask]).sum().item() / mask.sum().item()
            mask = self.data.test_mask
            test_acc = pred[mask].max(1)[1].eq(
                self.data.y[mask]).sum().item() / mask.sum().item()
        print(
            f"Label reuse >>> Train: {train_acc:.4f}, Test: {test_acc:.4f}")

        print(
            ">>> Number of nodes are predicted as the last category",
            torch.sum(
                all_pred[self.data.train_indices].argmax(dim=1) == self.data.num_classes
            ).item(),
        )

        self.reuse_average_f1[self.run] = test_acc

    def mia_attack(self):
        """
        Performs a Membership Inference Attack (MIA) to assess the model's privacy by determining 
        whether specific data points were part of the training dataset. It compares the model's 
        soft labels before and after the unlearning process and calculates the ROC AUC score to 
        evaluate the effectiveness of unlearning.
        """
        self.mia_num = self.unlearning_num
        breakpoint()
        original_softlabels_member = self.original_softlabels[self.unlearning_nodes]
        original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]

        unlearning_softlabels_member = self.unlearning_softlabels[self.unlearning_nodes]
        unlearning_softlabels_non = self.unlearning_softlabels[self.data.test_indices[:self.mia_num]]

        mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        # self.logger.info("posterior:{}".format(posterior))
        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        self.logger.info("auc:{}".format(auc))
        self.auc = auc
        breakpoint()
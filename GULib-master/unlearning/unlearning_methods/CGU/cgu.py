from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression
import pickle
# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch_geometric.nn import SGConv
from torch import Tensor
from torch.nn import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import os.path as osp

from torch.nn import init
from utils.utils import *

from sklearn import preprocessing
from numpy.linalg import norm
from config import root_path,unlearning_path
from config import BLUE_COLOR,RESET_COLOR

from pipeline.IF_based_pipeline import IF_based_pipeline

def plot_auc( y_true, y_score):
    y_true = y_true
    y_score = y_score

    # 计算ROC曲线上的点
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 计算AUC
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

class cgu(IF_based_pipeline):
    """
    CGU class implements the Certified Graph Unlearning (CGU) method.
    It extends the `IF_based_pipeline` class and provides functionalities to handle various unlearning tasks such as
    node, feature, and edge unlearning. The class initializes necessary parameters, manages the unlearning process,
    and records relevant performance metrics throughout multiple runs.

    Class Attributes:
        logger (Logger): Logger object for recording informational messages and debugging.

        args (dict): Dictionary containing configuration arguments for the unlearning process.

        model_zoo (ModelZoo): Collection of models used within the pipeline.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.logger = logger
        self.run = 0
        self.args = args
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self._data = copy.deepcopy(self.data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_runs = self.args["num_runs"]
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)
        assert self.args["base_model"] == 'SGC'
        
        if self.args["unlearn_task"] in ['node', 'feature']:
            self.num_removes = int(self.data.num_nodes * self.args["unlearn_ratio"])
            self.nonmember_id = self.data.test_indices[:self.args["num_unlearned_nodes"]]
        elif self.args["unlearn_task"] == 'edge':
            self.num_removes = int(self.data.edge_index.shape[1]*self.args["unlearn_ratio"])
            
        self.grad_norm_approx = torch.zeros((self.num_removes, self.args["num_runs"])).float()
        self.removal_times = torch.zeros((self.num_removes, self.args["num_runs"])).float()  # record the time of each removal
        self.acc_removal = torch.zeros((3, self.num_removes, self.args["num_runs"])).float()  # record the acc after removal, 0 for val, 1 for test
        self.grad_norm_worst = torch.zeros((self.num_removes, self.args["num_runs"])).float()  # worst case norm bound
        self.grad_norm_real = torch.zeros((self.num_removes, self.args["num_runs"])).float()  # true norm
        # graph retrain
        self.removal_times_graph_retrain = torch.zeros((self.num_removes, self.args["num_runs"])).float()
        self.acc_graph_retrain = torch.zeros((2, self.num_removes, self.args["num_runs"])).float()
    # def run_exp(self):
    #     if self.args["unlearn_task"] == "node" or self.args["unlearn_task"] == "feature":
    #         self.CGU_node_Unlearning(self.args)
    #     elif self.args["unlearn_task"] == "edge":
    #         self.CGU_edge_Unlearning(self.args)

    #     self.logger.info(
    #     "{}Performance Metrics:\n"
    #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average AUC Score: {:.4f} ± {:.4f}\n"
    #     " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}\n".format(
    #         BLUE_COLOR,
    #         np.mean(self.average_f1), np.std(self.average_f1),
    #         np.mean(self.average_auc), np.std(self.average_auc),
    #         np.mean(self.avg_training_time), np.std(self.avg_training_time),
    #         RESET_COLOR
    #         )
    #     )


    def CGU_node_Unlearning(self,args):

        self.logger.info('='*10 + 'Loading data' + '='*10)
        self.logger.info('Dataset:{}'.format( self.args["dataset_name"]))

        self.data = self.data.to(device)
        self.args["num_unlearned_nodes"] = int(self.data.num_nodes * self.args["unlearn_ratio"])
        self.nonmember_id = self.data.test_indices[:self.args["num_unlearned_nodes"]]
        # save the degree of each node for later use
        row = self.data.edge_index[0]
        deg = degree(row)

        
        # process features
        if self.args["featNorm"]:
            X = preprocess_data(self.data.x).to(device)
        else:
            X = self.data.x.to(device)
        # save a copy of X for removal
        X_scaled_copy_guo = X.clone().detach().float()

        # process labels
        if self.args["train_mode"] == 'binary':
            if '+' in self.args["Y_binary"]:
                # two classes are specified
                class1 = int(self.args["Y_binary"].split('+')[0])
                class2 = int(self.args["Y_binary"].split('+')[1])
                Y = self.data.y.clone().detach().float()
                Y[self.data.y == class1] = 1
                Y[self.data.y == class2] = -1
                interested_data_mask = (self.data.y == class1) + (self.data.y == class2)
                train_mask = self.data.train_mask * interested_data_mask
                val_mask = self.data.val_mask * interested_data_mask
                test_mask = self.data.test_mask * interested_data_mask
                self.data.Y = Y
            else:
                # one vs rest
                class1 = int(self.args["Y_binary"])
                Y = self.data.y.clone().detach().float()
                Y[self.data.y == class1] = 1
                Y[self.data.y != class1] = -1
                train_mask = self.data.train_mask
                val_mask = self.data.val_mask
                test_mask = self.data.test_mask
            y_train, y_val, y_test = Y[train_mask].to(device), Y[val_mask].to(device), Y[test_mask].to(device)
            self.data.Y = Y.to(device)
        else:
            # multiclass classification
            train_mask = self.data.train_mask
            val_mask = self.data.val_mask
            test_mask = self.data.test_mask
            y_train = F.one_hot(self.data.y[self.data.train_mask]) * 2 - 1
            y_train = y_train.float().to(device)
            y_val = self.data.y[self.data.val_mask].to(device)
            y_test = self.data.y[self.data.test_mask].to(device)
            self.data.Y = (F.one_hot(self.data.y)* 2 - 1)
            self.data.Y = self.data.Y.float().to(device)


        assert self.args["noise_mode"] == 'data'

        if self.args["compare_gnorm"]:
            # if we want to compare the residual gradient norm of three cases, we should not add noise
            # and make budget very large
            b_std = 0
        else:
            if self.args["noise_mode"] == 'data':
                b_std = self.args["std"]
            elif self.args["noise_mode"] == 'worst':
                b_std = self.args["std"]  # change to worst case sigma
            else:
                raise("Error: Not supported noise model.")

        #############
        # initial training with graph
        self.logger.info('='*10 + 'Training on full dataset with graph' + '='*10)
        start = time.time()
        # Propagation = MyGraphConv(K=self.args["GNN_layer"], add_self_loops=self.args["add_self_loops"],
        #                           alpha=self.args["alpha"], XdegNorm=self.args["XdegNorm"], GPR=self.args["GPR"]).to(device)
        Propagation =SGConv(self.data.num_features,self.data.num_classes,K=3)

        if self.args["GNN_layer"] > 0:
            X = Propagation(X, self.data.edge_index)

        X = X.float()
        X_train = X[train_mask].to(device)
        X_val = X[val_mask].to(device)
        X_test = X[test_mask].to(device)

        self.logger.info("Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(X_train.shape[0], X_val.shape[0],
                                                                                          X_test.shape[0],
                                                                                          self.data.edge_index.shape[1],
                                                                                          X_train.shape[1]))
        ############
        # train removal-enabled linear model
        self.logger.info("With graph, train mode:{}, optimizer:{}".format( self.args["train_mode"], self.args["optimizer"]))

        # reserved for future extension
        weight = None
        # in our case weight should always be None
        assert weight is None
        # record the optimal gradient norm wrt the whole training set
        opt_grad_norm = 0

        if self.args["train_mode"] == 'ovr':
            b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
            if self.args["train_sep"]:
                # train K binary LR models separately
                w = torch.zeros(b.size()).float().to(device)
                for k in range(y_train.size(1)):
                    if weight is None:
                        w[:, k] = lr_optimize(X_train, y_train[:, k], self.args["lam"], b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                    else:
                        w[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train[:, k][weight[:, k].gt(0)], self.args["lam"],
                                              b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            else:
                # train K binary LR models jointly
                w = ovr_lr_optimize(X_train, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            # record the opt_grad_norm
            for k in range(y_train.size(1)):
                opt_grad_norm += lr_grad(w[:, k], X_train, y_train[:, k], self.args["lam"]).norm().cpu()
        else:
            b = b_std * torch.randn(X_train.size(1)).float().to(device)
            w = lr_optimize(X_train, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            opt_grad_norm = lr_grad(w, X_train, y_train, self.args["lam"]).norm().cpu()

        self.logger.info('Time elapsed: %.6fs' % (time.time() - start))
        if self.args["train_mode"] == 'ovr':
            self.logger.info('Val accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w, X_val, y_val))
            self.logger.info('Test accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w, X_test, y_test))
        else:
            self.logger.info('Val accuracy = %.4f' % lr_eval(w, X_val, y_val))
            self.logger.info('Test accuracy = %.4f' % lr_eval(w, X_test, y_test))

        ###########
        if self.args["unlearn_task"] == 'node' or self.args["unlearn_task"] == 'feature':
            if self.args["compare_guo"]:
                # initial training without graph
                self.logger.info('='*10 + 'Training on full dataset without graph' + '='*10)
                start = time.time()

                # only the data preparation part is different
                X_train = X_scaled_copy_guo[train_mask].to(device)
                X_val = X_scaled_copy_guo[val_mask].to(device)
                X_test = X_scaled_copy_guo[test_mask].to(device)

                self.logger.info("Train node:{}, Val node:{}, Test node:{}, Feature dim:{}".format(X_train.shape[0], X_val.shape[0],
                                                                                        X_test.shape[0], X_train.shape[1]))
                ######
                # train removal-enabled linear model without graph
                self.logger.info("Without graph, train mode:", self.args["train_mode"], ", optimizer:", self.args["optimizer"])
                weight = None
                # in our case weight should always be None
                assert weight is None
                opt_grad_norm_guo = 0

                if self.args["train_mode"] == 'ovr':
                    b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                    if self.args["train_sep"]:
                        # train K binary LR models separately
                        w_guo = torch.zeros(b.size()).float().to(device)
                        for k in range(y_train.size(1)):
                            if weight is None:
                                w_guo[:, k] = lr_optimize(X_train, y_train[:, k], self.args["lam"], b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])

                            else:
                                w_guo[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train[:, k][weight[:, k].gt(0)], self.args["lam"],
                                              b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                    else:
                        # train K binary LR models jointly
                        w_guo = ovr_lr_optimize(X_train, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                    # record the opt_grad_norm
                    for k in range(y_train.size(1)):
                        opt_grad_norm_guo += lr_grad(w_guo[:, k], X_train, y_train[:, k],self.args["lam"]).norm().cpu()
                else:
                    b = b_std * torch.randn(X_train.size(1)).float().to(device)
                    w_guo = lr_optimize(X_train, y_train,self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"], opt_choice=self.args["optimizer"],
                                        lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                    opt_grad_norm_guo = lr_grad(w_guo, X_train, y_train, self.args["lam"]).norm().cpu()

                self.logger.info('Time elapsed: %.2fs' % (time.time() - start))
                if self.args["train_mode"] == 'ovr':
                    self.logger.info('Val accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w_guo, X_val, y_val))
                    self.logger.info('Test accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w_guo, X_test, y_test))
                else:
                    self.logger.info('Val accuracy = %.4f' % lr_eval(w_guo, X_val, y_val))
                    self.logger.info('Test accuracy = %.4f' % lr_eval(w_guo, X_test, y_test))


        ###########
        # budget for removal
        c_val = get_c(self.args["delta"])
        # if we need to compute the norms, we should not retrain at all
        if self.args["compare_gnorm"]:
            budget = 1e5
        else:
            if self.args["train_mode"] == 'ovr':
                budget = get_budget(b_std, self.args["eps"], c_val) * y_train.size(1)
            else:
                budget = get_budget(b_std, self.args["eps"], c_val)
        gamma = 1/4  # pre-computed for -logsigmoid loss
        self.logger.info('Budget:{}'.format(budget) )

        ##########
        # our removal
        # all norm here is NOT accumulated, need to use np.cumsum in plots
        grad_norm_approx = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()
        removal_times = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # record the time of each removal
        acc_removal = torch.zeros((3, self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # record the acc after removal, 0 for val, 1 for test
        grad_norm_worst = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # worst case norm bound
        grad_norm_real = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # true norm
        # graph retrain
        removal_times_graph_retrain = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()
        acc_graph_retrain = torch.zeros((2, self.args["num_unlearned_nodes"], self.args["num_runs"])).float()
        if  self.args["unlearn_task"] == 'node' or self.args["unlearn_task"] == 'feature':
            # guo removal
            grad_norm_approx_guo = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()
            removal_times_guo = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # record the time of each removal
            acc_guo = torch.zeros((2, self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # first row for val acc, second row for test acc
            grad_norm_real_guo = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # true norm
            # guo retrain
            removal_times_guo_retrain = torch.zeros((self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # record the time of each removal
            acc_guo_retrain = torch.zeros((2, self.args["num_unlearned_nodes"], self.args["num_runs"])).float()  # first row for val acc, second row for test acc

        if  self.args["unlearn_task"] == 'node' or self.args["unlearn_task"] == 'feature':
            for trail_iter in range(self.args["num_runs"]):
                self.logger.info('*'*10 + str(trail_iter) + '*'*10)
                # if self.args["fix_random_seed"]:
                #     # fix the random seed for perm
                #     np.random.seed(trail_iter)
                train_id = torch.arange(self.data.x.shape[0])[train_mask.cpu()]
                # perm = torch.from_numpy(np.random.permutation(train_id.shape[0]))
                # perm = list(perm)
                #############
                # removal_queue = train_id[perm]
                path_un = unlearning_path + "_" + str(trail_iter) + ".txt"
                removal_queue = np.loadtxt(path_un,dtype=int)
                removal_queue_tensor = torch.from_numpy(removal_queue)

                # 找出train_id中存在但removal_queue_tensor中不存在的元素
                unique_elements = train_id.tolist()  # 将train_id转换为列表
                removal_queue_list = removal_queue_tensor.tolist()  # 将removal_queue_tensor转换为列表
                additional_elements = [item for item in unique_elements if item not in removal_queue_list]

                # 将找到的额外元素添加到removal_queue_tensor中
                updated_removal_queue_tensor = torch.cat((removal_queue_tensor, torch.tensor(additional_elements)))

                # 如果需要将结果转换回NumPy数组
                updated_removal_queue = updated_removal_queue_tensor.numpy()
                removal_queue = updated_removal_queue
                # self.member_id = removal_queue[:args.num_removes]
                self.member_id = removal_queue[:self.args["num_unlearned_nodes"]]


                #############

                edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)

                X_scaled_copy = X_scaled_copy_guo.clone().detach().float()
                w_approx = w.clone().detach()  # copy the parameters to modify
                X_old = X.clone().detach().to(device)

                num_retrain = 0
                grad_norm_approx_sum = 0
                # start the removal process
                self.logger.info('='*10 + 'Testing our removal' + '='*10)
                for i in tqdm(range(self.args["num_unlearned_nodes"]-1)):
                        
                    # First, replace removal features with 0 vector
                    X_scaled_copy[removal_queue[i]] = 0
                    if self.args["unlearn_task"] == 'node':
                        # Then remove the correpsonding edges
                        edge_mask[self.data.edge_index[0] == removal_queue[i]] = False
                        edge_mask[self.data.edge_index[1] == removal_queue[i]] = False
                        # make sure we do not remove self-loops
                        self_loop_idx = torch.logical_and(self.data.edge_index[0] == removal_queue[i],
                                                          self.data.edge_index[1] == removal_queue[i]).nonzero().squeeze(-1)
                        if self_loop_idx.size(0) > 0:
                            edge_mask[self_loop_idx] = True

                    start = time.time()
                    # Get propagated features
                    if self.args["GNN_layer"] > 0:
                        X_new = Propagation(X_scaled_copy, self.data.edge_index[:, edge_mask]).to(device)
                    else:
                        X_new = X_scaled_copy.to(device)

                    X_val_new = X_new[val_mask]
                    X_test_new = X_new[test_mask]
                    X_unlearning_new = X_new[removal_queue]
                    y_unlearning = self.data.y[removal_queue].to(device)

                    # note that the removed data point should still not be used in computing K or H
                    # removal_queue[(i+1):] are the remaining training idx
                    K = get_K_matrix(X_new[removal_queue[(i+1):]]).to(device)
                    spec_norm = sqrt_spectral_norm(K)

                    if self.args["train_mode"] == 'ovr':
                        # removal from all one-vs-rest models
                        X_rem = X_new[removal_queue[(i+1):]]
                        for k in range(y_train.size(1)):
                            y_rem = self.data.Y[removal_queue[(i+1):], k]
                            H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, self.args["lam"])
                            # grad_i is the difference
                            grad_old = lr_grad(w_approx[:, k], X_old[removal_queue[i:]], self.data.Y[removal_queue[(i):], k],  self.args["lam"])
                            grad_new = lr_grad(w_approx[:, k], X_rem, y_rem,  self.args["lam"])
                            grad_i = grad_old - grad_new
                            Delta = H_inv.mv(grad_i)
                            Delta_p = X_rem.mv(Delta)
                            # update w here. If beta exceed the budget, w_approx will be retrained
                            w_approx[:, k] += Delta
                            # data dependent norm
                            grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                            if self.args["compare_gnorm"]:
                                grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem,  self.args["lam"]).norm().cpu()
                                if self.args["unlearn_task"] == 'node':
                                    grad_norm_worst[i, trail_iter] += get_worst_Gbound_node( self.args["lam"], X_rem.shape[0],
                                                                                            self.args["GNN_layer"],
                                                                                            deg[removal_queue[i]]).cpu()
                                elif self.args["unlearn_task"] == 'feature':
                                    grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature( self.args["lam"], X_rem.shape[0],
                                                                                               deg[removal_queue[i]]).cpu()
                        # decide after all classes
                        if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                            # retrain the model
                            grad_norm_approx_sum = 0
                            b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                            w_approx = ovr_lr_optimize(X_rem, self.data.Y[removal_queue[(i+1):]], self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                       opt_choice=self.args["optimizer"],lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            num_retrain += 1
                        else:
                            grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                        # record acc each round
                        acc_removal[0, i, trail_iter] = ovr_lr_eval(w_approx, X_val_new, y_val)[1]
                        acc_removal[1, i, trail_iter] = ovr_lr_eval(w_approx, X_test_new, y_test)[1]
                        acc_removal[2, i, trail_iter] = ovr_lr_eval(w_approx, X_unlearning_new, y_unlearning)[1]
                    else:
                        # removal from a single binary logistic regression model
                        X_rem = X_new[removal_queue[(i+1):]]
                        y_rem = self.data.Y[removal_queue[(i+1):]]
                        H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, self.args["lam"])
                        # grad_i should be the difference
                        grad_old = lr_grad(w_approx, X_old[removal_queue[i:]], Y[removal_queue[i:]], self.args["lam"])
                        grad_new = lr_grad(w_approx, X_rem, y_rem, self.args["lam"])
                        grad_i = grad_old - grad_new
                        Delta = H_inv.mv(grad_i)
                        Delta_p = X_rem.mv(Delta)
                        w_approx += Delta
                        grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                        if self.args["compare_gnorm"]:
                            grad_norm_real[i, trail_iter] += lr_grad(w_approx, X_rem, y_rem, self.args["lam"]).norm().cpu()
                            if self.args["unlearn_task"] == 'node':
                                grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(self.args["lam"], X_rem.shape[0],
                                                                                        self.args["GNN_layer"],
                                                                                        deg[removal_queue[i]]).cpu()
                            elif self.args["unlearn_task"] == 'feature':
                                grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(self.args["lam"], X_rem.shape[0],
                                                                                           deg[removal_queue[i]]).cpu()

                        if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                            # retrain the model
                            grad_norm_approx_sum = 0
                            b = b_std * torch.randn(X_train.size(1)).float().to(device)
                            w_approx = lr_optimize(X_rem, y_rem, self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                   opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            num_retrain += 1
                        else:
                            grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                        # record acc each round
                        acc_removal[0, i, trail_iter] = lr_eval(w_approx, X_val_new, y_val)
                        acc_removal[1, i, trail_iter] = lr_eval(w_approx, X_test_new, y_test)


                    removal_times[i, trail_iter] +=  time.time() - start
                    # Remember to replace X_old with X_new
                    X_old = X_new.clone().detach()
                    unlearning_acc = ovr_lr_eval(w_approx, X_scaled_copy_guo[self.member_id], self.data.y[self.member_id])
                    if i % self.args["disp"] == 0:
                        self.logger.info("unlearning_acc:{}".format(unlearning_acc))
                        self.logger.info('Iteration %d: time = %.6fs, number of retrain = %d' % (i+1, removal_times[i, trail_iter], num_retrain))
                        self.logger.info('Val acc = %.4f, Test acc = %.4f Unlearning acc = %.4f' % (acc_removal[0, i, trail_iter], acc_removal[1, i, trail_iter],acc_removal[2, i, trail_iter]))
                #######
                    if i == self.args["num_unlearned_nodes"]-2:
                        self.average_f1[trail_iter] = acc_removal[1, i, trail_iter]
                        softlabel_original0 = F.softmax(X_scaled_copy_guo[self.nonmember_id].mm(w),dim = 1)
                        softlabel_original1 = F.softmax(X_scaled_copy_guo[self.member_id].mm(w),dim = 1)
                        softlabel_new0 = F.softmax(X_scaled_copy_guo[self.nonmember_id].mm(w_approx),dim = 1)
                        softlabel_new1 = F.softmax(X_scaled_copy_guo[self.member_id].mm(w_approx),dim = 1)
                        mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
                        posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
                        posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
                        posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
                        
                        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                        self.logger.info('auc:{}'.format(auc))
                        self.average_auc[trail_iter] = auc
                        plot_auc(mia_test_y, posterior.reshape(-1, 1))
                        average = removal_times[:, trail_iter].sum()
                        self.avg_training_time[trail_iter] = average

                        self.logger.info('removal_time:{}'.format(average))
                # retrain each round with graph
                if self.args["compare_retrain"]:
                    X_scaled_copy = X_scaled_copy_guo.clone().detach()
                    edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
                    # start the removal process
                    self.logger.info('=' * 10 + 'Testing with graph retrain' + '=' * 10)
                    for i in tqdm(range(self.args["num_unlearned_nodes"])):
                        # First, replace removal features with 0 vector
                        X_scaled_copy[removal_queue[i]] = 0
                        # Then remove the correpsonding edges
                        if self.args["unlearn_task"] == 'node':
                            edge_mask[self.data.edge_index[0] == removal_queue[i]] = False
                            edge_mask[self.data.edge_index[1] == removal_queue[i]] = False
                            # make sure we do not remove self-loops
                            self_loop_idx = torch.logical_and(self.data.edge_index[0] == removal_queue[i],
                                                              self.data.edge_index[1] == removal_queue[
                                                                  i]).nonzero().squeeze(-1)
                            if self_loop_idx.size(0) > 0:
                                edge_mask[self_loop_idx] = True

                        start = time.time()
                        # Get propagated features
                        if self.args["GNN_layer"] > 0:
                            X_new = Propagation(X_scaled_copy, self.data.edge_index[:, edge_mask]).to(device)
                        else:
                            X_new = X_scaled_copy.to(device)

                        X_val_new = X_new[val_mask]
                        X_test_new = X_new[test_mask]

                        if self.args["train_mode"] == 'ovr':
                            # removal from all one-vs-rest models
                            X_rem = X_new[removal_queue[(i + 1):]]
                            y_rem =  self.data.Y[removal_queue[(i + 1):]]
                            # retrain the model
                            # we do not need to add noise if we are retraining every time
                            # b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                            w_graph_retrain = ovr_lr_optimize(X_rem, y_rem, self.args["lam"], weight, b=None,
                                                              num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])


                            acc_graph_retrain[0, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_val_new, y_val)
                            acc_graph_retrain[1, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_test_new,
                                                                              y_test)
                        else:
                            # removal from a single binary logistic regression model
                            X_rem = X_new[removal_queue[(i + 1):]]
                            y_rem =  Y[removal_queue[(i + 1):]]
                            # retrain the model
                            # b = b_std * torch.randn(X_train.size(1)).float().to(device)
                            w_graph_retrain = lr_optimize(X_rem, y_rem, self.args["lam"], b=None,
                                                              num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            acc_graph_retrain[0, i, trail_iter] = lr_eval(w_graph_retrain, X_val_new, y_val)
                            acc_graph_retrain[1, i, trail_iter] = lr_eval(w_graph_retrain, X_test_new, y_test)

                        removal_times_graph_retrain[i, trail_iter] = time.time() - start
                        if i % self.args["disp"] == 0:
                            self.logger.info('Iteration %d, time = %.6fs, val acc = %.4f, test acc = %.4f' % (
                            i + 1, removal_times_graph_retrain[i, trail_iter],
                            acc_graph_retrain[0, i, trail_iter], acc_graph_retrain[1, i, trail_iter]))

                #######
                # guo removal
                if self.args["compare_guo"] and self.args["unlearn_task"] != 'edge':
                    w_approx_guo = w_guo.clone().detach()  # copy the parameters to modify
                    num_retrain = 0
                    grad_norm_approx_sum_guo = 0
                    # prepare the train/val/test sets
                    X_train = X_scaled_copy_guo[train_mask].to(device)
                    X_train_perm = X_scaled_copy_guo[removal_queue]
                    y_train_perm =  Y[removal_queue]
                    K = get_K_matrix(X_train_perm).to(device)
                    X_val = X_scaled_copy_guo[val_mask].to(device)
                    X_test = X_scaled_copy_guo[test_mask].to(device)
                    # start the removal process
                    self.logger.info('=' * 10 + 'Testing Guo et al. removal' + '=' * 10)
                    for i in tqdm(range(self.args["num_unlearned_nodes"])):
                        start = time.time()
                        if self.args["train_mode"] == 'ovr':
                            # removal from all one-vs-rest models
                            X_rem = X_train_perm[(i + 1):]
                            # update matrix K
                            K -= torch.outer(X_train_perm[i], X_train_perm[i])
                            spec_norm = sqrt_spectral_norm(K)
                            for k in range(y_train_perm.size(1)):
                                y_rem = y_train_perm[(i + 1):, k]
                                H_inv = lr_hessian_inv(w_approx_guo[:, k], X_rem, y_rem, self.args["lam"])
                                # grad_i is the difference
                                grad_i = lr_grad(w_approx_guo[:, k], X_train_perm[i].unsqueeze(0),
                                                 y_train_perm[i, k].unsqueeze(0), self.args["lam"])
                                Delta = H_inv.mv(grad_i)
                                Delta_p = X_rem.mv(Delta)
                                # update w here. If beta exceed the budget, w_approx_guo will be retrained
                                w_approx_guo[:, k] += Delta
                                grad_norm_approx_guo[i, trail_iter] += (
                                            Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                                if self.args["compare_gnorm"]:
                                    grad_norm_real_guo[i, trail_iter] += lr_grad(w_approx_guo[:, k], X_rem,
                                                                                 y_rem, self.args["lam"]).norm().cpu()
                            # decide after all classes
                            if grad_norm_approx_sum_guo + grad_norm_approx_guo[i, trail_iter] > budget:
                                # retrain the model
                                grad_norm_approx_sum_guo = 0
                                b = b_std * torch.randn(X_train_perm.size(1), y_train_perm.size(1)).float().to(
                                    device)
                                w_approx_guo = ovr_lr_optimize(X_rem, y_train_perm[(i + 1):], self.args["lam"], weight,
                                                               b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                                num_retrain += 1
                            else:
                                grad_norm_approx_sum_guo += grad_norm_approx_guo[i, trail_iter]
                            # record the acc each round
                            
                            acc_guo[0, i, trail_iter] = ovr_lr_eval(w_approx_guo, X_val, y_val)
                            acc_guo[1, i, trail_iter] = ovr_lr_eval(w_approx_guo, X_test, y_test)
                        else:
                            # removal from a single binary logistic regression model
                            X_rem = X_train_perm[(i + 1):]
                            y_rem = y_train_perm[(i + 1):]
                            H_inv = lr_hessian_inv(w_approx_guo, X_rem, y_rem, self.args["lam"])
                            grad_i = lr_grad(w_approx_guo, X_train_perm[i].unsqueeze(0),
                                             y_train_perm[i].unsqueeze(0), self.args["lam"])
                            K -= torch.outer(X_train_perm[i], X_train_perm[i])
                            spec_norm = sqrt_spectral_norm(K)
                            Delta = H_inv.mv(grad_i)
                            Delta_p = X_rem.mv(Delta)
                            w_approx_guo += Delta
                            grad_norm_approx_guo[i, trail_iter] += (
                                        Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                            if self.args["compare_gnorm"]:
                                grad_norm_real_guo[i, trail_iter] += lr_grad(w_approx_guo, X_rem, y_rem,
                                                                             self.args["lam"]).norm().cpu()
                            if grad_norm_approx_sum_guo + grad_norm_approx_guo[i, trail_iter] > budget:
                                # retrain the model
                                grad_norm_approx_sum_guo = 0
                                b = b_std * torch.randn(X_train_perm.size(1)).float().to(device)
                                w_approx_guo = lr_optimize(X_rem, y_rem, self.args["lam"], b=b,
                                                           num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                                num_retrain += 1
                            else:
                                grad_norm_approx_sum_guo += grad_norm_approx_guo[i, trail_iter]
                            # record the acc each round
                            acc_guo[0, i, trail_iter] = lr_eval(w_approx_guo, X_val, y_val)
                            acc_guo[1, i, trail_iter] = lr_eval(w_approx_guo, X_test, y_test)

                        removal_times_guo[i, trail_iter] = time.time() - start
                        if i % self.args["disp"] == 0:
                            self.logger.info('Iteration %d: time = %.6fs, number of retrain = %d' % (
                            i + 1, removal_times_guo[i, trail_iter], num_retrain))
                            self.logger.info('Val acc = %.4f, Test acc = %.4f' % (
                            acc_guo[0, i, trail_iter], acc_guo[1, i, trail_iter]))

                #######
                # retrain each round without graph
                if self.args["unlearn_task"] != 'edge' and self.args["compare_retrain"] and self.args["compare_guo"]:
                    X_train = X_scaled_copy_guo[train_mask].to(device)
                    X_train_perm = X_train[perm]
                    y_train_perm = y_train[perm]
                    X_val = X_scaled_copy_guo[val_mask].to(device)
                    X_test = X_scaled_copy_guo[test_mask].to(device)

                    # start the removal process
                    self.logger.info('=' * 10 + 'Testing without graph retrain' + '=' * 10)
                    for i in tqdm(range(self.args["num_unlearned_nodes"])):
                        start = time.time()
                        if self.args["train_mode"] == 'ovr':
                            # removal from all one-vs-rest models
                            X_rem = X_train_perm[(i + 1):]
                            y_rem = y_train_perm[(i + 1):]
                            # retrain the model
                            # b = b_std * torch.randn(X_train_perm.size(1), y_train_perm.size(1)).float().to(device)
                            w_guo_retrain = ovr_lr_optimize(X_rem, y_rem, self.args["lam"], weight, b=None,
                                                            num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            acc_guo_retrain[0, i, trail_iter] = ovr_lr_eval(w_guo_retrain, X_val, y_val)
                            acc_guo_retrain[1, i, trail_iter] = ovr_lr_eval(w_guo_retrain, X_test, y_test)
                        else:
                            # removal from a single binary logistic regression model
                            X_rem = X_train_perm[(i + 1):]
                            y_rem = y_train_perm[(i + 1):]
                            # retrain the model
                            # b = b_std * torch.randn(X_train_perm.size(1)).float().to(device)
                            w_guo_retrain = lr_optimize(X_rem, y_rem, self.args["lam"],  b=None,
                                                            num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            acc_guo_retrain[0, i, trail_iter] = lr_eval(w_guo_retrain, X_val, y_val)
                            acc_guo_retrain[1, i, trail_iter] = lr_eval(w_guo_retrain, X_test, y_test)

                        removal_times_guo_retrain[i, trail_iter] = time.time() - start
                        if i % self.args["disp"] == 0:
                            self.logger.info('Iteration %d, time = %.6fs, val acc = %.4f, test acc = %.4f' % (
                            i + 1, removal_times_guo_retrain[i, trail_iter], acc_guo_retrain[0, i, trail_iter],
                            acc_guo_retrain[1, i, trail_iter]))

        else:
            for trail_iter in range(self.args["num_runs"]):
                self.logger.info('*' * 10, trail_iter, '*' * 10)
                if self.args["fix_random_seed"]:
                    np.random.seed(trail_iter)
                # get a random permutation for edge indices for each trail
                perm = torch.from_numpy(np.random.permutation(self.data.edge_index.shape[1]))
                # Note that all edges are used in training, so we just need to decide the order to remove edges
                # the number of training samples will always be m
                edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)

                X_scaled_copy = X_scaled_copy_guo.clone().detach().float()
                w_approx = w.clone().detach()  # copy the parameters to modify
                X_old = X.clone().detach().to(device)

                num_retrain = 0
                grad_norm_approx_sum = 0
                perm_idx = 0
                # start the removal process
                self.logger.info('=' * 10 + 'Testing our edge removal' + '=' * 10)
                for i in tqdm(range(self.args["num_unlearned_nodes"])):
                    # First, check if this is a self-loop or an edge already deleted
                    while (self.data.edge_index[0, perm[perm_idx]] == self.data.edge_index[1, perm[perm_idx]]) or (
                    not edge_mask[perm[perm_idx]]):
                        perm_idx += 1
                    edge_mask[perm[perm_idx]] = False
                    source_idx = self.data.edge_index[0, perm[perm_idx]]
                    dst_idx = self.data.edge_index[1, perm[perm_idx]]
                    # find the other undirected edge
                    rev_edge_idx = torch.logical_and(self.data.edge_index[0] == dst_idx,
                                                     self.data.edge_index[1] == source_idx).nonzero().squeeze(-1)
                    if rev_edge_idx.size(0) > 0:
                        edge_mask[rev_edge_idx] = False

                    perm_idx += 1
                    start = time.time()
                    # Get propagated features
                    if self.args["GNN_layer"] > 0:
                        X_new = Propagation(X_scaled_copy, self.data.edge_index[:, edge_mask]).to(device)
                    else:
                        X_new = X_scaled_copy.to(device)

                    X_val_new = X_new[val_mask]
                    X_test_new = X_new[test_mask]

                    K = get_K_matrix(X_new[train_mask]).to(device)
                    spec_norm = sqrt_spectral_norm(K)
                    
                    if self.args["train_mode"] == 'ovr':
                        # removal from all one-vs-rest models
                        X_rem = X_new[train_mask]
                        for k in range(y_train.size(1)):
                            assert weight is None
                            y_rem = y_train[:, k]
                            H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, self.args["lam"])
                            # grad_i is the difference
                            grad_old = lr_grad(w_approx[:, k], X_old[train_mask], y_rem, self.args["lam"])
                            grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, self.args["lam"])
                            grad_i = grad_old - grad_new
                            Delta = H_inv.mv(grad_i)
                            Delta_p = X_rem.mv(Delta)
                            # update w here. If beta exceed the budget, w_approx will be retrained
                            w_approx[:, k] += Delta
                            grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                            if self.args["compare_gnorm"]:
                                grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem,
                                                                         self.args["lam"]).norm().cpu()
                                grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(self.args["lam"], X_rem.shape[0],
                                                                                        self.args["GNN_layer"])
                        # decide after all classes
                        if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                            # retrain the model
                            grad_norm_approx_sum = 0
                            b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                            w_approx = ovr_lr_optimize(X_rem, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            num_retrain += 1
                        else:
                            grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                        # record acc each round
                        acc_removal[0, i, trail_iter] = ovr_lr_eval(w_approx, X_val_new, y_val)
                        acc_removal[1, i, trail_iter] = ovr_lr_eval(w_approx, X_test_new, y_test)
                    else:
                        # removal from a single binary logistic regression model
                        X_rem = X_new[train_mask]
                        y_rem = y_train
                        H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, self.args["lam"])
                        # grad_i should be the difference
                        grad_old = lr_grad(w_approx, X_old[train_mask], y_train, self.args["lam"])
                        grad_new = lr_grad(w_approx, X_rem, y_rem, self.args["lam"])
                        grad_i = grad_old - grad_new
                        Delta = H_inv.mv(grad_i)
                        Delta_p = X_rem.mv(Delta)
                        w_approx += Delta
                        grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                        if self.args["compare_gnorm"]:
                            grad_norm_real[i, trail_iter] += lr_grad(w_approx, X_rem, y_rem, self.args["lam"]).norm().cpu()
                            grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(self.args["lam"], X_rem.shape[0],
                                                                                    self.args["GNN_layer"])
                        if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                            # retrain the model
                            grad_norm_approx_sum = 0
                            b = b_std * torch.randn(X_train.size(1)).float().to(device)
                            w_approx = lr_optimize(X_rem, y_rem,self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            num_retrain += 1
                        else:
                            grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                        
                        # record acc each round
                        acc_removal[0, i, trail_iter] = lr_eval(w_approx, X_val_new, y_val)
                        acc_removal[1, i, trail_iter] = lr_eval(w_approx, X_test_new, y_test)

                    removal_times[i, trail_iter] = time.time() - start
                    # Remember to replace X_old with X_new
                    X_old = X_new.clone().detach()
                    if i % self.args["disp"] == 0:
                        self.logger.info('Iteration %d: time = %.6fs, number of retrain = %d' % (
                        i + 1, removal_times[i, trail_iter], num_retrain))
                        self.logger.info('Val acc = %.4f, Test acc = %.4f' % (
                        acc_removal[0, i, trail_iter], acc_removal[1, i, trail_iter]))
                #######
                # retrain each round with graph
                if self.args["compare_retrain"]:
                    edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
                    perm_idx = 0
                    # start the removal process
                    self.logger.info('=' * 10 + 'Testing with graph retrain' + '=' * 10)
                    for i in tqdm(range(self.args["num_unlearned_nodes"])):
                        # First, check if this is a self-loop or an edge already deleted
                        while (self.data.edge_index[0, perm[perm_idx]] == self.data.edge_index[1, perm[perm_idx]]) or (
                        not edge_mask[perm[perm_idx]]):
                            perm_idx += 1
                        edge_mask[perm[perm_idx]] = False
                        source_idx = self.data.edge_index[0, perm[perm_idx]]
                        dst_idx = self.data.edge_index[1, perm[perm_idx]]
                        # find the other undirected edge
                        rev_edge_idx = torch.logical_and(self.data.edge_index[0] == dst_idx,
                                                         self.data.edge_index[1] == source_idx).nonzero().squeeze(-1)
                        if rev_edge_idx.size(0) > 0:
                            edge_mask[rev_edge_idx] = False

                        perm_idx += 1
                        start = time.time()
                        # Get propagated features
                        if self.args["GNN_layer"] > 0:
                            X_new = Propagation(X_scaled_copy, self.data.edge_index[:, edge_mask]).to(device)
                        else:
                            X_new = X_scaled_copy.to(device)

                        X_val_new = X_new[val_mask]
                        X_test_new = X_new[test_mask]

                        if self.args["train_mode"] == 'ovr':
                            # removal from all one-vs-rest models
                            X_rem = X_new[train_mask]
                            # retrain the model
                            # we do not need to add noise if we are retraining every time
                            # b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                            w_graph_retrain = ovr_lr_optimize(X_rem, y_train, self.args["lam"], weight, b=None,
                                                              num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            
                            acc_graph_retrain[0, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_val_new, y_val)
                            acc_graph_retrain[1, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_test_new,
                                                                              y_test)
                        else:
                            # removal from a single binary logistic regression model
                            X_rem = X_new[train_mask]
                            # retrain the model
                            # b = b_std * torch.randn(X_train.size(1)).float().to(device)
                            w_graph_retrain = lr_optimize(X_rem, y_train, self.args["lam"], b=None,
                                                              num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                            acc_graph_retrain[0, i, trail_iter] = lr_eval(w_graph_retrain, X_val_new, y_val)
                            acc_graph_retrain[1, i, trail_iter] = lr_eval(w_graph_retrain, X_test_new, y_test)

                        removal_times_graph_retrain[i, trail_iter] = time.time() - start
                        if i % self.args["disp"] == 0:
                            self.logger.info('Iteration %d, time = %.6fs, val acc = %.4f, test acc = %.4f' % (
                            i + 1, removal_times_graph_retrain[i, trail_iter],
                            acc_graph_retrain[0, i, trail_iter], acc_graph_retrain[1, i, trail_iter]))


        # softlabel_original0 = F.softmax(X_scaled_copy_guo[self.nonmember_id].mm(w),dim = 1)
        # softlabel_original1 = F.softmax(X_scaled_copy_guo[self.member_id].mm(w),dim = 1)
        # softlabel_new0 = F.softmax(X_scaled_copy_guo[self.nonmember_id].mm(w_approx),dim = 1)
        # softlabel_new1 = F.softmax(X_scaled_copy_guo[self.member_id].mm(w_approx),dim = 1)
        # mia_test_y = torch.cat((torch.ones(args.num_removes), torch.zeros(args.num_removes)))
        # posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
        # posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
        # posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
        
        # auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        # self.logger.info(auc)
        # plot_auc(mia_test_y, posterior.reshape(-1, 1))
        # average = removal_times[:, 0].mean()

        # self.logger.info(average)

        # save all results
        # if not osp.exists(args.result_dir):
        #     os.makedirs(args.result_dir)
        # save_path = '%s/%s_std_%.0e_lam_%.0e_nr_%d_K_%d_opt_%s_mode_%s_eps_%.1f_delta_%.0e' % (args.result_dir,
        #              args.dataset, b_std, args.lam, args.num_removes, args.prop_step, args.optimizer, args.removal_mode, args.eps, args.delta)


        # if  args.removal_mode == 'node' or args.removal_mode == 'feature':
        #     if args.train_mode == 'binary':
        #         save_path += '_bin_%s' % args.Y_binary
        #     if args.GPR:
        #         save_path += '_gpr'
        #     if args.compare_gnorm:
        #         save_path += '_gnorm'
        #     if args.compare_retrain:
        #         save_path += '_retrain'
        #     if args.compare_guo:
        #         save_path += '_withguo'

        #     save_path += '.pth'
        #     torch.save({'grad_norm_approx': grad_norm_approx, 'removal_times': removal_times, 'acc_removal': acc_removal,
        #                 'grad_norm_worst': grad_norm_worst, 'grad_norm_real': grad_norm_real,
        #                 'removal_times_graph_retrain': removal_times_graph_retrain, 'acc_graph_retrain': acc_graph_retrain,
        #                 'grad_norm_approx_guo': grad_norm_approx_guo, 'removal_times_guo': removal_times_guo, 'acc_guo': acc_guo,
        #                 'removal_times_guo_retrain': removal_times_guo_retrain, 'acc_guo_retrain': acc_guo_retrain,
        #                 'grad_norm_real_guo': grad_norm_real_guo}, save_path)
        # else:
        #     if args.train_mode == 'binary':
        #         save_path += '_bin_%s' % args.Y_binary
        #     if args.GPR:
        #         save_path += '_gpr'
        #     if args.compare_gnorm:
        #         save_path += '_gnorm'
        #     if args.compare_retrain:
        #         save_path += '_retrain'
        #     save_path += '.pth'

        #     torch.save({'grad_norm_approx': grad_norm_approx, 'removal_times': removal_times, 'acc_removal': acc_removal,
        #                 'grad_norm_worst': grad_norm_worst, 'grad_norm_real': grad_norm_real,
        #                 'removal_times_graph_retrain': removal_times_graph_retrain,
        #                 'acc_graph_retrain': acc_graph_retrain}, save_path)

    def CGU_edge_Unlearning(self,args):
        self.logger.info('='*10 + 'Loading data' + '='*10)
        self.logger.info('Dataset:{}'.format( self.args["dataset_name"]))
        num_removes = int(self.data.edge_index.shape[1]*self.args["unlearn_ratio"])
        self.data = self.data.to(device)
        row = self.data.edge_index[0]
        deg = degree(row)
        # process features
        if self.args["featNorm"]:
            X = preprocess_data(self.data.x).to(device)
        else:
            X = self.data.x.to(device)
        X_scaled_copy_guo = X.clone().detach().float()
        if self.args["train_mode"] == 'binary':
            if '+' in self.args["Y_binary"]:
                # two classes are specified
                class1 = int(self.args["Y_binary"].split('+')[0])
                class2 = int(self.args["Y_binary"].split('+')[1])
                Y = self.data.y.clone().detach().float()
                Y[self.data.y == class1] = 1
                Y[self.data.y == class2] = -1
                interested_data_mask = (self.data.y == class1) + (self.data.y == class2)
                train_mask = self.data.train_mask * interested_data_mask
                val_mask = self.data.val_mask * interested_data_mask
                test_mask = self.data.test_mask * interested_data_mask
                self.data.Y = Y
            else:
                # one vs rest
                class1 = int(self.args["Y_binary"])
                Y = self.data.y.clone().detach().float()
                Y[self.data.y == class1] = 1
                Y[self.data.y != class1] = -1
                train_mask = self.data.train_mask
                val_mask = self.data.val_mask
                test_mask = self.data.test_mask
            y_train, y_val, y_test = Y[train_mask].to(device), Y[val_mask].to(device), Y[test_mask].to(device)
            
        else:
            # multiclass classification
            train_mask = self.data.train_mask
            val_mask = self.data.val_mask
            test_mask = self.data.test_mask
            y_train = F.one_hot(self.data.y[self.data.train_mask]) * 2 - 1
            y_train = y_train.float().to(device)
            y_val = self.data.y[self.data.val_mask].to(device)
            y_test = self.data.y[self.data.test_mask].to(device)
        assert self.args["noise_mode"] == 'data'
        if self.args["compare_gnorm"]:
            # if we want to compare the residual gradient norm of three cases, we should not add noise
            # and make budget very large
            b_std = 0
        else:
            if self.args["noise_mode"] == 'data':
                b_std = self.args["std"]
            elif self.args["noise_mode"] == 'worst':
                b_std = self.args["std"]  # change to worst case sigma
            else:
                raise("Error: Not supported noise model.")
        
        self.logger.info('='*10 + 'Training on full dataset with graph' + '='*10)
        start = time.time()
        Propagation = MyGraphConv(K=self.args["GNN_layer"], add_self_loops=self.args["add_self_loops"],
                                  alpha=self.args["alpha"], XdegNorm=self.args["XdegNorm"], GPR=self.args["GPR"]).to(device)

        if self.args["GNN_layer"] > 0:
            X = Propagation(X, self.data.edge_index)
        X = X.float()
        X_train = X[train_mask].to(device)
        X_val = X[val_mask].to(device)
        X_test = X[test_mask].to(device)
        self.logger.info("Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(X_train.shape[0], X_val.shape[0],
                                                                                          X_test.shape[0],
                                                                                          self.data.edge_index.shape[1],
                                                                                          X_train.shape[1]))
        self.logger.info("With graph, train mode:{}, optimizer:{}".format( self.args["train_mode"], self.args["optimizer"]))
        # reserved for future extension
        weight = None
        # in our case weight should always be None
        assert weight is None
        # record the optimal gradient norm wrt the whole training set
        opt_grad_norm = 0
        if self.args["train_mode"] == 'ovr':
            b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
            if self.args["train_sep"]:
                # train K binary LR models separately
                w = torch.zeros(b.size()).float().to(device)
                for k in range(y_train.size(1)):
                    if weight is None:
                        w[:, k] = lr_optimize(X_train, y_train[:, k], self.args["lam"], b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                    else:
                        w[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train[:, k][weight[:, k].gt(0)], self.args["lam"],
                                              b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            else:
                # train K binary LR models jointly
                w = ovr_lr_optimize(X_train, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            # record the opt_grad_norm
            for k in range(y_train.size(1)):
                opt_grad_norm += lr_grad(w[:, k], X_train, y_train[:, k], self.args["lam"]).norm().cpu()
        else:
            b = b_std * torch.randn(X_train.size(1)).float().to(device)
            w = lr_optimize(X_train, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            opt_grad_norm = lr_grad(w, X_train, y_train, self.args["lam"]).norm().cpu()
        self.logger.info('Time elapsed: %.6fs' % (time.time() - start))
        if self.args["train_mode"] == 'ovr':
            self.logger.info('Val accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w, X_val, y_val))
            self.logger.info('Test accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w, X_test, y_test))
        else:
            self.logger.info('Val accuracy = %.4f' % lr_eval(w, X_val, y_val))
            self.logger.info('Test accuracy = %.4f' % lr_eval(w, X_test, y_test))
        
        ########### 
        c_val = get_c(args["delta"])
        if self.args["compare_guo"]:
            budget = 1e5
        else:
            if self.args["train_mode"] == 'ovr':
                budget = get_budget(b_std, args["eps"], c_val) * y_train.size(1)
            else:
                budget = get_budget(b_std, args["eps"], c_val)
                
        gamma = 1/4  # pre-computed for -logsigmoid loss
        self.logger.info('Budget:{}'.format(budget) )
            
        ########### 
        grad_norm_approx = torch.zeros((num_removes, self.args["num_runs"])).float()
        removal_times = torch.zeros((num_removes, self.args["num_runs"])).float()  # record the time of each removal
        acc_removal = torch.zeros((3, num_removes, self.args["num_runs"])).float()  # record the acc after removal, 0 for val, 1 for test
        grad_norm_worst = torch.zeros((num_removes, self.args["num_runs"])).float()  # worst case norm bound
        grad_norm_real = torch.zeros((num_removes, self.args["num_runs"])).float()  # true norm
        # graph retrain
        removal_times_graph_retrain = torch.zeros((num_removes, self.args["num_runs"])).float()
        acc_graph_retrain = torch.zeros((2, num_removes, self.args["num_runs"])).float()
        
        for trail_iter in range(self.args["num_runs"]):
            self.logger.info('*'*10 + str(trail_iter) + '*'*10)
            if self.args["fix_random_seed"]:
                np.random.seed(trail_iter)
            perm = torch.from_numpy(np.random.permutation(self.data.edge_index.shape[1]))
            edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
            X_scaled_copy = X_scaled_copy_guo.clone().detach().float()
            w_approx = w.clone().detach()  # copy the parameters to modify
            X_old = X.clone().detach().to(device)
            num_retrain = 0
            grad_norm_approx_sum = 0
            perm_idx = 0
            
            self.logger.info('='*10 + "Testing our edge removal" + ''*10)
            for i in range(num_removes):
                while (self.data.edge_index[0, perm[perm_idx]] == self.data.edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
                    perm_idx += 1
                    
                edge_mask[perm[perm_idx]] = False
                source_idx = self.data.edge_index[0, perm[perm_idx]]
                dst_idx = self.data.edge_index[1, perm[perm_idx]]
                rev_edge_idx = torch.logical_and(self.data.edge_index[0] == dst_idx,
                                             self.data.edge_index[1] == source_idx).nonzero().squeeze(-1)
                if rev_edge_idx.size(0) > 0:
                    edge_mask[rev_edge_idx] = False
                perm_idx += 1
                start = time.time()
                if self.args["GNN_layer"] > 0:
                    X_new = Propagation(X_scaled_copy, self.data.edge_index[:, edge_mask]).to(device)
                else:
                    X_new = X_scaled_copy.to(device)
                
                X_val_new = X_new[val_mask]
                X_test_new = X_new[test_mask]
                
                K = get_K_matrix(X_new[train_mask]).to(device)
                spec_norm = sqrt_spectral_norm(K)
                if args["train_mode"] == 'ovr':
                    X_rem = X_new[train_mask]
                    for k in range(y_train.size(1)):
                        assert weight is None
                        y_rem = y_train[:, k]
                        H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, self.args["lam"])
                        # grad_i is the difference
                        grad_old = lr_grad(w_approx[:, k], X_old[train_mask], y_rem, self.args["lam"])
                        grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, self.args["lam"])
                        grad_i = grad_old - grad_new
                        Delta = H_inv.mv(grad_i)
                        Delta_p = X_rem.mv(Delta)
                        # update w here. If beta exceed the budget, w_approx will be retrained
                        w_approx[:, k] += Delta
                        grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                        if args["compare_gnorm"]:
                            grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem, self.args["lam"]).norm().cpu()
                            grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(self.args["lam"], X_rem.shape[0],
                                                                                    self.args["prop_step"])
                    if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                        # retrain the model
                        grad_norm_approx_sum = 0
                        b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                        w_approx = ovr_lr_optimize(X_rem, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                        num_retrain += 1
                    else:
                        grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                    
                    acc_removal[0, i, trail_iter] = ovr_lr_eval(w_approx, X_val_new, y_val)[1]
                    acc_removal[1, i, trail_iter] = ovr_lr_eval(w_approx, X_test_new, y_test)[1]
                else:
                    X_rem = X_new[train_mask]
                    y_rem = y_train
                    H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, self.args["lam"])
                    # grad_i should be the difference
                    grad_old = lr_grad(w_approx, X_old[train_mask], y_train, self.args["lam"])
                    grad_new = lr_grad(w_approx, X_rem, y_rem, self.args["lam"])
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    w_approx += Delta
                    grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                    if self.args["compare_gnorm"]:
                        grad_norm_real[i, trail_iter] += lr_grad(w_approx, X_rem, y_rem, self.args["lam"]).norm().cpu()
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(self.args["lam"], X_rem.shape[0], self.args["prop_step"])
                    if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                        grad_norm_approx_sum = 0
                        b = b_std * torch.randn(X_train.size(1)).float().to(device)
                        w_approx = lr_optimize(X_rem, y_rem, self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"], opt_choice=self.args["optimizer"],
                                           lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                        num_retrain += 1
                    else:
                        grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                    acc_removal[0, i, trail_iter] = lr_eval(w_approx, X_val_new, y_val)[1]
                    acc_removal[1, i, trail_iter] = lr_eval(w_approx, X_test_new, y_test)[1]
                removal_times[i, trail_iter] = time.time() - start
                if i==num_removes-1:
                    average = removal_times[:, trail_iter].mean()
                    self.avg_training_time[trail_iter] = average
                X_old = X_new.clone().detach()
                if i % self.args["disp"] == 0:
                    print('Iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_times[i, trail_iter], num_retrain))
                    print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i, trail_iter], acc_removal[1, i, trail_iter]))
            self.average_f1[trail_iter] = acc_removal[1, i, trail_iter]
            if self.args["compare_retrain"]:
                edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
                perm_idx = 0
                self.logger.info('='*10 + 'Testing with graph retrain' + '='*10)
                for i in range(num_removes):
                    while (self.data.edge_index[0, perm[perm_idx]] == self.data.edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
                        perm_idx += 1
                    edge_mask[perm[perm_idx]] = False
                    source_idx = self.data.edge_index[0, perm[perm_idx]]
                    dst_idx = self.data.edge_index[1, perm[perm_idx]]
                    # find the other undirected edge
                    rev_edge_idx = torch.logical_and(self.data.edge_index[0] == dst_idx,
                                                    self.data.edge_index[1] == source_idx).nonzero().squeeze(-1)
                    if rev_edge_idx.size(0) > 0:
                        edge_mask[rev_edge_idx] = False

                    perm_idx += 1
                    start = time.time()
                    # Get propagated features
                    if self.args["GNN_layer"] > 0:
                        X_new = Propagation(X_scaled_copy, self.data.edge_index[:, edge_mask]).to(device)
                    else:
                        X_new = X_scaled_copy.to(device)

                    X_val_new = X_new[val_mask]
                    X_test_new = X_new[test_mask]

                    if self.args["train_mode"] == 'ovr':
                        # removal from all one-vs-rest models
                        X_rem = X_new[train_mask]
                        # retrain the model
                        # we do not need to add noise if we are retraining every time
                        # b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                        w_graph_retrain = ovr_lr_optimize(X_rem, y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                        acc_graph_retrain[0, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_val_new, y_val)
                        acc_graph_retrain[1, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_test_new, y_test)
                    else:
                        # removal from a single binary logistic regression model
                        X_rem = X_new[train_mask]
                        # retrain the model
                        # b = b_std * torch.randn(X_train.size(1)).float().to(device)
                        w_graph_retrain = lr_optimize(X_rem, y_train, self.args["lam"], b=None, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                        acc_graph_retrain[0, i, trail_iter] = lr_eval(w_graph_retrain, X_val_new, y_val)
                        acc_graph_retrain[1, i, trail_iter] = lr_eval(w_graph_retrain, X_test_new, y_test)

                    removal_times_graph_retrain[i, trail_iter] = time.time() - start
                    if i % self.args["disp"] == 0:
                        print('Iteration %d, time = %.2fs, val acc = %.4f, test acc = %.4f' % (i+1, removal_times_graph_retrain[i, trail_iter], acc_graph_retrain[0, i, trail_iter], acc_graph_retrain[1, i, trail_iter]))

    
    def train_original_model(self, run):
        """
        This function implements data preprocessing and original GNN model training by calling functions `prepare_data` and `train_model`.
        """
        self.prepare_data()
        self.train_model()
        pass
    
    def unlearning_request(self):
        """
        This function calculates the deletion budget based on the given parameters and conditions, and updates the queue of nodes or features that need to be deleted.
        """
        # budget for removal
        c_val = get_c(self.args["delta"])
        # if we need to compute the norms, we should not retrain at all
        if self.args["compare_gnorm"]:
            self.budget = 1e5
        else:
            if self.args["train_mode"] == 'ovr':
                self.budget = get_budget(self.b_std, self.args["eps"], c_val) * self.y_train.size(1)
            else:
                self.budget = get_budget(self.b_std, self.args["eps"], c_val)
        self.gamma = 1/4  # pre-computed for -logsigmoid loss
        self.logger.info('Budget:{}'.format(self.budget))
        
        

        if  self.args["unlearn_task"] == 'node' or self.args["unlearn_task"] == 'feature':
            train_id = torch.arange(self.data.x.shape[0])[self.train_mask.cpu()]
            path_un = unlearning_path + "_" + str(self.run) + ".txt"
            self.removal_queue = np.loadtxt(path_un,dtype=int)
            removal_queue_tensor = torch.from_numpy(self.removal_queue)
            unique_elements = train_id.tolist()  # 将train_id转换为列表
            removal_queue_list = removal_queue_tensor.tolist()  # 将removal_queue_tensor转换为列表
            additional_elements = [item for item in unique_elements if item not in removal_queue_list]
            updated_removal_queue_tensor = torch.cat((removal_queue_tensor, torch.tensor(additional_elements)))
            updated_removal_queue = updated_removal_queue_tensor.numpy()
            self.removal_queue = updated_removal_queue
            self.member_id = self.removal_queue[:self.args["num_unlearned_nodes"]]
            
        pass
    
    def unlearn(self):
        """
        This function implements the unlearning process by removing specified nodes or features or edges from the model. It updates the edge masks, propagates the changes, adjusts the model parameters, and records the performance metrics after each removal.
        """
        if self.args["unlearn_task"] in ["node","feature"]:
            self.edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)

            self.X_scaled_copy = self.X_scaled_copy_guo.clone().detach().float()
            self.w_approx = self.w.clone().detach()  # copy the parameters to modify
            self.X_old = self.X.clone().detach().to(device)

            self.num_retrain = 0
            self.grad_norm_approx_sum = 0
            for i in tqdm(range(self.num_removes)):
                self.get_grad(i)
        elif self.args["unlearn_task"] == "edge":
            self.perm = torch.from_numpy(np.random.permutation(self.data.edge_index.shape[1]))
            self.edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
            self.X_scaled_copy = self.X_scaled_copy_guo.clone().detach().float()
            self.w_approx = self.w.clone().detach()  # copy the parameters to modify
            self.X_old = self.X.clone().detach().to(device)
            self.num_retrain = 0
            self.grad_norm_approx_sum = 0
            self.perm_idx = 0
            for i in tqdm(range(self.num_removes)):
                self.get_grad_edge(i)
        
        #--------------- Saving Unlearned Model-------------#
        import os

        # … your existing code that computes grad_norm_approx, removal_times, acc_removal, etc. …

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(self.run)

        # 1) compute fraction of nodes removed
        unlearn_ratio = len(self.nonmember_id) / self.data.num_nodes
        unlearn_ratio_str = f"ratio_{unlearn_ratio:.2f}{copy_str}"

        # 2) build and create your target directory
        save_dir = os.path.join(
            "unlearned_models", "CGU",
            self.args['dataset_name'],
            self.args["downstream_task"],   # or "unlearn_task" if that's what you use
            unlearn_ratio_str
        )
        os.makedirs(save_dir, exist_ok=True)

        # 3) choose a filename that matches what your eval script expects
        #    (Script expects something like CGU_<dataset>_node_ratio_xx.xx.pt)
        model_name = f"CGU_{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}.pt"
        save_path = os.path.join(save_dir, model_name)

        # 4) **SAVE only the raw state_dict** so that your loader does:
        #       checkpoint = torch.load(path)
        #       state_dict = checkpoint  # since no 'model_state' key, it uses checkpoint directly
        torch.save({"model_state": self.w_approx.detach().cpu()}, save_path)
        self.logger.info(f"Saved CGU state_dict to {save_path}")
        pass
    
    def prepare_data(self):
        """
        This function prepares the data for training and unlearning procedures. It loads the dataset, processes the features and labels, applies normalization if specified, and sets up training, validation, and testing masks. Additionally, it initializes the graph convolution propagation and trains the initial model with the full dataset.
        """
        self.logger.info('='*10 + 'Loading data' + '='*10)
        self.logger.info('Dataset:{}'.format( self.args["dataset_name"]))
        
        self.data = copy.deepcopy(self._data)
        self.data = self.data.to(device)
        
        # self.args["num_unlearned_nodes"] = int(self.data.num_nodes * self.args["unlearn_ratio"])
        
        # save the degree of each node for later use
        row = self.data.edge_index[0]
        self.deg = degree(row)

        
        # process features
        if self.args["featNorm"]:
            X = preprocess_data(self.data.x).to(device)
        else:
            X = self.data.x.to(device)
        # save a copy of X for removal
        self.X_scaled_copy_guo = X.clone().detach().float()

        # process labels
        if self.args["train_mode"] == 'binary':
            if '+' in self.args["Y_binary"]:
                # two classes are specified
                class1 = int(self.args["Y_binary"].split('+')[0])
                class2 = int(self.args["Y_binary"].split('+')[1])
                Y = self.data.y.clone().detach().float()
                Y[self.data.y == class1] = 1
                Y[self.data.y == class2] = -1
                interested_data_mask = (self.data.y == class1) + (self.data.y == class2)
                self.train_mask = self.data.train_mask * interested_data_mask
                self.val_mask = self.data.val_mask * interested_data_mask
                self.test_mask = self.data.test_mask * interested_data_mask
                self.data.Y = Y
            else:
                # one vs rest
                class1 = int(self.args["Y_binary"])
                Y = self.data.y.clone().detach().float()
                Y[self.data.y == class1] = 1
                Y[self.data.y != class1] = -1
                self.train_mask = self.data.train_mask
                self.val_mask = self.data.val_mask
                self.test_mask = self.data.test_mask
            self.y_train, self.y_val, self.y_test = Y[self.train_mask].to(device), Y[self.val_mask].to(device), Y[self.test_mask].to(device)
            self.data.Y = Y.to(device)
        else:
            # multiclass classification
            self.train_mask = self.data.train_mask
            self.val_mask = self.data.val_mask
            self.test_mask = self.data.test_mask
            self.y_train = F.one_hot(self.data.y[self.data.train_mask]) * 2 - 1
            self.y_train = self.y_train.float().to(device)
            self.y_val = self.data.y[self.data.val_mask].to(device)
            self.y_test = self.data.y[self.data.test_mask].to(device)
            self.data.Y = (F.one_hot(self.data.y)* 2 - 1)
            self.data.Y = self.data.Y.float().to(device)
        assert self.args["noise_mode"] == 'data'
        
        if self.args["compare_gnorm"]:
            # if we want to compare the residual gradient norm of three cases, we should not add noise
            # and make budget very large
            self.b_std = 0
        else:
            if self.args["noise_mode"] == 'data':
                self.b_std = self.args["std"]
            elif self.args["noise_mode"] == 'worst':
                self.b_std = self.args["std"]  # change to worst case sigma
            else:
                raise("Error: Not supported noise model.")
            
        #############
        # initial training with graph
        self.logger.info('='*10 + 'Training on full dataset with graph' + '='*10)
        
        self.Propagation = MyGraphConv(K=self.args["GNN_layer"], add_self_loops=self.args["add_self_loops"],
                                  alpha=self.args["alpha"], XdegNorm=self.args["XdegNorm"], GPR=self.args["GPR"]).to(device)

        if self.args["GNN_layer"] > 0:
            X = self.Propagation(X, self.data.edge_index)

        X = X.float()
        self.X_train = X[self.train_mask].to(device)
        self.X_val = X[self.val_mask].to(device)
        self.X_test = X[self.test_mask].to(device)
        self.X = X
        self.logger.info("Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(self.X_train.shape[0], self.X_val.shape[0],
                                                                                          self.X_test.shape[0],
                                                                                          self.data.edge_index.shape[1],
                                                                                          self.X_train.shape[1]))
        ############
        # train removal-enabled linear model
        self.logger.info("With graph, train mode:{}, optimizer:{}".format( self.args["train_mode"], self.args["optimizer"]))

    
    def train_model(self):
        """
        Trains the linear model using the preprocessed training data. Depending on the training mode ('ovr' for one-vs-rest or binary classification),
        it initializes model parameters, optionally adds noise for differential privacy, and optimizes the weights using the specified optimizer and learning rate.
        After training, it logs the training duration and evaluates the model's performance on validation and test datasets, reporting metrics such as accuracy, F1 score, and recall.
        """
        self.logger.info("With graph, train mode:{}, optimizer:{}".format( self.args["train_mode"], self.args["optimizer"]))
        start = time.time()
        # reserved for future extension
        self.weight = None
        # in our case weight should always be None
        assert self.weight is None
        # record the optimal gradient norm wrt the whole training set
        opt_grad_norm = 0

        if self.args["train_mode"] == 'ovr':
            b = self.b_std * torch.randn(self.X_train.size(1), self.y_train.size(1)).float().to(device)
            if self.args["train_sep"]:
                # train K binary LR models separately
                self.w = torch.zeros(b.size()).float().to(device)
                for k in range(self.y_train.size(1)):
                    if self.weight is None:
                        self.w[:, k] = lr_optimize(self.X_train, self.y_train[:, k], self.args["lam"], b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                    else:
                        self.w[:, k] = lr_optimize(self.X_train[self.weight[:, k].gt(0)], self.y_train[:, k][self.weight[:, k].gt(0)], self.args["lam"],
                                              b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                              opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            else:
                # train K binary LR models jointly
                self.w = ovr_lr_optimize(self.X_train, self.y_train, self.args["lam"], self.weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            # record the opt_grad_norm
            for k in range(self.y_train.size(1)):
                opt_grad_norm += lr_grad(self.w[:, k], self.X_train, self.y_train[:, k], self.args["lam"]).norm().cpu()
        else:
            b = self.b_std * torch.randn(self.X_train.size(1)).float().to(device)
            self.w = lr_optimize(self.X_train, self.y_train, self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                    opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
            opt_grad_norm = lr_grad(self.w, self.X_train, self.y_train, self.args["lam"]).norm().cpu()

        self.logger.info('Time elapsed: %.6fs' % (time.time() - start))
        if self.args["train_mode"] == 'ovr':
            self.logger.info('Val accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(self.w, self.X_val, self.y_val))
            self.logger.info('Test accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(self.w, self.X_test, self.y_test))
            if self.args["poison"] and self.args["unlearn_task"]=="edge":
                self.poison_f1[self.run] = ovr_lr_eval(self.w, self.X_test, self.y_test)[1]
        else:
            self.logger.info('Val accuracy = %.4f' % lr_eval(self.w, self.X_val, self.y_val))
            self.logger.info('Test accuracy = %.4f' % lr_eval(self.w, self.X_test, self.y_test))
            if self.args["poison"] and self.args["unlearn_task"]=="edge":
                self.poison_f1[self.run] = lr_eval(self.w, self.X_test, self.y_test)
            
        # if self.args["unlearn_task"] == 'node' or self.args["unlearn_task"] == 'feature':
        #     if self.args["compare_guo"]:
        #         # initial training without graph
        #         self.logger.info('='*10 + 'Training on full dataset without graph' + '='*10)
        #         start = time.time()

        #         # only the data preparation part is different
        #         X_train = self.X_scaled_copy_guo[self.train_mask].to(device)
        #         X_val = self.X_scaled_copy_guo[self.val_mask].to(device)
        #         X_test = self.X_scaled_copy_guo[self.test_mask].to(device)

        #         self.logger.info("Train node:{}, Val node:{}, Test node:{}, Feature dim:{}".format(X_train.shape[0], X_val.shape[0],
        #                                                                                 X_test.shape[0], X_train.shape[1]))
        #         ######
        #         # train removal-enabled linear model without graph
        #         self.logger.info("Without graph, train mode:", self.args["train_mode"], ", optimizer:", self.args["optimizer"])
        #         weight = None
        #         # in our case weight should always be None
        #         assert weight is None
        #         opt_grad_norm_guo = 0

        #         if self.args["train_mode"] == 'ovr':
        #             b = self.b_std * torch.randn(X_train.size(1), self.y_train.size(1)).float().to(device)
        #             if self.args["train_sep"]:
        #                 # train K binary LR models separately
        #                 w_guo = torch.zeros(b.size()).float().to(device)
        #                 for k in range(self.y_train.size(1)):
        #                     if weight is None:
        #                         w_guo[:, k] = lr_optimize(X_train, self.y_train[:, k], self.args["lam"], b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
        #                                       opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])

        #                     else:
        #                         w_guo[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], self.y_train[:, k][weight[:, k].gt(0)], self.args["lam"],
        #                                       b=b[:, k], num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
        #                                       opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
        #             else:
        #                 # train K binary LR models jointly
        #                 w_guo = ovr_lr_optimize(X_train, self.y_train, self.args["lam"], weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
        #                             opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
        #             # record the opt_grad_norm
        #             for k in range(self.y_train.size(1)):
        #                 opt_grad_norm_guo += lr_grad(w_guo[:, k], X_train, self.y_train[:, k],self.args["lam"]).norm().cpu()
        #         else:
        #             b = self.b_std * torch.randn(X_train.size(1)).float().to(device)
        #             w_guo = lr_optimize(X_train, self.y_train,self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"], opt_choice=self.args["optimizer"],
        #                                 lr=self.args["opt_lr"], wd=self.args["opt_decay"])
        #             opt_grad_norm_guo = lr_grad(w_guo, X_train, self.y_train, self.args["lam"]).norm().cpu()

        #         self.logger.info('Time elapsed: %.2fs' % (time.time() - start))
        #         if self.args["train_mode"] == 'ovr':
        #             self.logger.info('Val accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w_guo, X_val, self.y_val))
        #             self.logger.info('Test accuracy = %.4f F1 = %.4f Recall = %.4f' % ovr_lr_eval(w_guo, X_test, self.y_test))
        #         else:
        #             self.logger.info('Val accuracy = %.4f' % lr_eval(w_guo, X_val, self.y_val))
        #             self.logger.info('Test accuracy = %.4f' % lr_eval(w_guo, X_test, self.y_test))
                    
    def get_grad(self,index):
        """
        This function computes and updates the gradient for a specific index during the node unlearning process.
        It modifies the feature matrix by zeroing out the features of the node to be unlearned.
        For node or feature unlearning tasks, it also removes corresponding edges and ensures that self-loops are retained.
        The function then propagates the updated features using the graph convolution layer, calculates the residual gradient norms,
        and updates the model parameters accordingly.

        Additionally, it records the time taken for the operation and logs the current state of validation and test accuracies.
        """
        self.X_scaled_copy[self.removal_queue[index]] = 0
        if self.args["unlearn_task"] == 'node':
            # Then remove the correpsonding edges
            self.edge_mask[self.data.edge_index[0] == self.removal_queue[index]] = False
            self.edge_mask[self.data.edge_index[1] == self.removal_queue[index]] = False
            # make sure we do not remove self-loops
            self_loop_idx = torch.logical_and(self.data.edge_index[0] == self.removal_queue[index],
                                                self.data.edge_index[1] == self.removal_queue[index]).nonzero().squeeze(-1)
            if self_loop_idx.size(0) > 0:
                self.edge_mask[self_loop_idx] = True
                
        start = time.time()
        # Get propagated features
        if self.args["GNN_layer"] > 0:
            X_new = self.Propagation(self.X_scaled_copy, self.data.edge_index[:, self.edge_mask]).to(device)
        else:
            X_new = self.X_scaled_copy.to(device)

        X_val_new = X_new[self.val_mask]
        X_test_new = X_new[self.test_mask]
        X_unlearning_new = X_new[self.removal_queue]
        y_unlearning = self.data.y[self.removal_queue].to(device)

        # note that the removed data point should still not be used in computing K or H
        # removal_queue[(i+1):] are the remaining training idx
        K = get_K_matrix(X_new[self.removal_queue[(index+1):]]).to(device)
        spec_norm = sqrt_spectral_norm(K)

        if self.args["train_mode"] == 'ovr':
            # removal from all one-vs-rest models
            X_rem = X_new[self.removal_queue[(index+1):]]
            for k in range(self.y_train.size(1)):
                y_rem = self.data.Y[self.removal_queue[(index+1):], k]
                H_inv = lr_hessian_inv(self.w_approx[:, k], X_rem, y_rem, self.args["lam"])
                # grad_i is the difference
                grad_old = lr_grad(self.w_approx[:, k], self.X_old[self.removal_queue[index:]], self.data.Y[self.removal_queue[(index):], k],  self.args["lam"])
                grad_new = lr_grad(self.w_approx[:, k], X_rem, y_rem,  self.args["lam"])
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                # update w here. If beta exceed the budget, w_approx will be retrained
                self.w_approx[:, k] += Delta
                # data dependent norm
                self.grad_norm_approx[index, self.run] += (Delta.norm() * Delta_p.norm() * spec_norm * self.gamma).cpu()
                if self.args["compare_gnorm"]:
                    self.grad_norm_real[index, self.run] += lr_grad(self.w_approx[:, k], X_rem, y_rem,  self.args["lam"]).norm().cpu()
                    if self.args["unlearn_task"] == 'node':
                        self.grad_norm_worst[index, self.run] += get_worst_Gbound_node( self.args["lam"], X_rem.shape[0],
                                                                                self.args["GNN_layer"],
                                                                                self.deg[self.removal_queue[index]]).cpu()
                    elif self.args["unlearn_task"] == 'feature':
                        self.grad_norm_worst[index, self.run] += get_worst_Gbound_feature( self.args["lam"], X_rem.shape[0],
                                                                                    self.deg[self.removal_queue[index]]).cpu()
            # decide after all classes
            if self.grad_norm_approx_sum + self.grad_norm_approx[index, self.run] > self.budget:
                # retrain the model
                self.grad_norm_approx_sum = 0
                b = self.b_std * torch.randn(self.X_train.size(1), self.y_train.size(1)).float().to(device)
                self.w_approx = ovr_lr_optimize(X_rem, self.data.Y[self.removal_queue[(index+1):]], self.args["lam"], self.weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                            opt_choice=self.args["optimizer"],lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                self.num_retrain += 1
            else:
                self.grad_norm_approx_sum += self.grad_norm_approx[index, self.run]
            # record acc each round
            self.acc_removal[0, index, self.run] = ovr_lr_eval(self.w_approx, X_val_new, self.y_val)[1]
            self.acc_removal[1, index, self.run] = ovr_lr_eval(self.w_approx, X_test_new, self.y_test)[1]
            self.acc_removal[2, index, self.run] = ovr_lr_eval(self.w_approx, X_unlearning_new, y_unlearning)[1]
        else:
            # removal from a single binary logistic regression model
            X_rem = X_new[self.removal_queue[(index+1):]]
            y_rem = self.data.Y[self.removal_queue[(index+1):]]
            H_inv = lr_hessian_inv(self.w_approx, X_rem, y_rem, self.args["lam"])
            # grad_i should be the difference
            grad_old = lr_grad(self.w_approx, self.X_old[self.removal_queue[index:]], self.data.Y[self.removal_queue[index:]], self.args["lam"])
            grad_new = lr_grad(self.w_approx, X_rem, y_rem, self.args["lam"])
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_rem.mv(Delta)
            self.w_approx += Delta
            self.grad_norm_approx[index, self.run] += (Delta.norm() * Delta_p.norm() * spec_norm * self.gamma).cpu()
            if self.args["compare_gnorm"]:
                self.grad_norm_real[index, self.run] += lr_grad(self.w_approx, X_rem, y_rem, self.args["lam"]).norm().cpu()
                if self.args["unlearn_task"] == 'node':
                    self.grad_norm_worst[index, self.run] += get_worst_Gbound_node(self.args["lam"], X_rem.shape[0],
                                                                            self.args["GNN_layer"],
                                                                            self.deg[self.removal_queue[index]]).cpu()
                elif self.args["unlearn_task"] == 'feature':
                    self.grad_norm_worst[index, self.run] += get_worst_Gbound_feature(self.args["lam"], X_rem.shape[0],
                                                                                self.deg[self.removal_queue[index]]).cpu()

            if self.grad_norm_approx_sum + self.grad_norm_approx[index, self.run] > self.budget:
                # retrain the model
                self.grad_norm_approx_sum = 0
                b = self.b_std * torch.randn(self.X_train.size(1)).float().to(device)
                self.w_approx = lr_optimize(X_rem, y_rem, self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                        opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                self.num_retrain += 1
            else:
                self.grad_norm_approx_sum += self.grad_norm_approx[index, self.run]
            # record acc each round
            self.acc_removal[0, index, self.run] = lr_eval(self.w_approx, X_val_new, self.y_val)
            self.acc_removal[1, index, self.run] = lr_eval(self.w_approx, X_test_new, self.y_test)
            
        self.removal_times[index, self.run] +=  time.time() - start
        self.X_old = X_new.clone().detach()
        # unlearning_acc = ovr_lr_eval(self.w_approx, self.X_scaled_copy_guo[self.member_id], self.data.y[self.member_id])
        if index % self.args["disp"] == 0:
            # self.logger.info("unlearning_acc:{}".format(unlearning_acc))
            self.logger.info('Iteration %d: time = %.6fs, number of retrain = %d' % (index+1, self.removal_times[index, self.run], self.num_retrain))
            self.logger.info('Val acc = %.4f, Test acc = %.4f Unlearning acc = %.4f' % (self.acc_removal[0, index, self.run], self.acc_removal[1, index, self.run],self.acc_removal[2, index, self.run]))
        if index == self.num_removes-1:
            self.average_f1[self.run] = self.acc_removal[1, index, self.run]
            if self.args["train_mode"] == 'ovr':
                softlabel_original0 = F.softmax(self.X_scaled_copy_guo[self.nonmember_id].mm(self.w),dim = 1)
                softlabel_original1 = F.softmax(self.X_scaled_copy_guo[self.member_id].mm(self.w),dim = 1)
                softlabel_new0 = F.softmax(self.X_scaled_copy_guo[self.nonmember_id].mm(self.w_approx),dim = 1)
                softlabel_new1 = F.softmax(self.X_scaled_copy_guo[self.member_id].mm(self.w_approx),dim = 1)
            else:
                softlabel_original0 = F.softmax(self.X_scaled_copy_guo[self.nonmember_id].mv(self.w),dim = 0)
                softlabel_original1 = F.softmax(self.X_scaled_copy_guo[self.member_id].mv(self.w),dim = 0)
                softlabel_new0 = F.softmax(self.X_scaled_copy_guo[self.nonmember_id].mv(self.w_approx),dim = 0)
                softlabel_new1 = F.softmax(self.X_scaled_copy_guo[self.member_id].mv(self.w_approx),dim = 0)
            mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
            posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
            posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
            posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
            
            auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            self.logger.info('auc:{}'.format(auc))
            self.average_auc[self.run] = auc
            # plot_auc(mia_test_y, posterior.reshape(-1, 1))
            average = self.removal_times[:, self.run].sum()
            self.avg_training_time[self.run] = average

            self.logger.info('removal_time:{}'.format(average))
        
    def get_grad_edge(self,index):
        """
        Computes and applies gradient updates for edge unlearning. This function removes a specified edge,
        updates the graph's edge mask, propagates updated features using the graph convolution layer,
        calculates the gradient differences, adjusts the model parameters accordingly, checks
        against the defined gradient budget, retrains the model if necessary, and records performance metrics.
        """
        while (self.data.edge_index[0, self.perm[self.perm_idx]] == self.data.edge_index[1, self.perm[self.perm_idx]]) or (not self.edge_mask[self.perm[self.perm_idx]]):
            self.perm_idx += 1
        self.edge_mask[self.perm[self.perm_idx]] = False
        source_idx = self.data.edge_index[0, self.perm[self.perm_idx]]
        dst_idx = self.data.edge_index[1, self.perm[self.perm_idx]]
        rev_edge_idx = torch.logical_and(self.data.edge_index[0] == dst_idx,
                                        self.data.edge_index[1] == source_idx).nonzero().squeeze(-1)
        if rev_edge_idx.size(0) > 0:
            self.edge_mask[rev_edge_idx] = False
        self.perm_idx += 1
        start = time.time()
        if self.args["GNN_layer"] > 0:
            X_new = self.Propagation(self.X_scaled_copy, self.data.edge_index[:, self.edge_mask]).to(device)
        else:
            X_new = self.X_scaled_copy.to(device)
        
        X_val_new = X_new[self.val_mask]
        X_test_new = X_new[self.test_mask]
        
        K = get_K_matrix(X_new[self.train_mask]).to(device)
        spec_norm = sqrt_spectral_norm(K)
        if self.args["train_mode"] == 'ovr':
            X_rem = X_new[self.train_mask]
            for k in range(self.y_train.size(1)):
                assert self.weight is None
                y_rem = self.y_train[:, k]
                H_inv = lr_hessian_inv(self.w_approx[:, k], X_rem, y_rem, self.args["lam"])
                # grad_i is the difference
                grad_old = lr_grad(self.w_approx[:, k], self.X_old[self.train_mask], y_rem, self.args["lam"])
                grad_new = lr_grad(self.w_approx[:, k], X_rem, y_rem, self.args["lam"])
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                # update w here. If beta exceed the budget, w_approx will be retrained
                self.w_approx[:, k] += Delta
                self.grad_norm_approx[index,self.run] += (Delta.norm() * Delta_p.norm() * spec_norm * self.gamma).cpu()
                if self.args["compare_gnorm"]:
                    self.grad_norm_real[index,self.run] += lr_grad(self.w_approx[:, k], X_rem, y_rem, self.args["lam"]).norm().cpu()
                    self.grad_norm_worst[index,self.run] += get_worst_Gbound_edge(self.args["lam"], X_rem.shape[0],
                                                                            self.args["prop_step"])
            if self.grad_norm_approx_sum + self.grad_norm_approx[index,self.run] > self.budget:
                # retrain the model
                self.grad_norm_approx_sum = 0
                b = self.b_std * torch.randn(self.X_train.size(1), self.y_train.size(1)).float().to(device)
                self.w_approx = ovr_lr_optimize(X_rem, self.y_train, self.args["lam"], self.weight, b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"],
                                        opt_choice=self.args["optimizer"], lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                self.num_retrain += 1
            else:
                self.grad_norm_approx_sum += self.grad_norm_approx[index,self.run]
            
            self.acc_removal[0, index,self.run] = ovr_lr_eval(self.w_approx, X_val_new, self.y_val)[1]
            self.acc_removal[1, index,self.run] = ovr_lr_eval(self.w_approx, X_test_new, self.y_test)[1]
        else:
            X_rem = X_new[self.train_mask]
            y_rem = self.y_train
            H_inv = lr_hessian_inv(self.w_approx, X_rem, y_rem, self.args["lam"])
            # grad_i should be the difference
            grad_old = lr_grad(self.w_approx, self.X_old[self.train_mask], self.y_train, self.args["lam"])
            grad_new = lr_grad(self.w_approx, X_rem, y_rem, self.args["lam"])
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_rem.mv(Delta)
            self.w_approx += Delta
            self.grad_norm_approx[index,self.run] += (Delta.norm() * Delta_p.norm() * spec_norm * self.gamma).cpu()
            if self.args["compare_gnorm"]:
                self.grad_norm_real[index,self.run] += lr_grad(self.w_approx, X_rem, y_rem, self.args["lam"]).norm().cpu()
                self.grad_norm_worst[index,self.run] += get_worst_Gbound_edge(self.args["lam"], X_rem.shape[0], self.args["prop_step"])
            if self.grad_norm_approx_sum + self.grad_norm_approx[index,self.run] > self.budget:
                self.grad_norm_approx_sum = 0
                b = self.b_std * torch.randn(self.X_train.size(1)).float().to(device)
                self.w_approx = lr_optimize(X_rem, y_rem, self.args["lam"], b=b, num_steps=self.args["num_epochs"], verbose=self.args["verbose"], opt_choice=self.args["optimizer"],
                                    lr=self.args["opt_lr"], wd=self.args["opt_decay"])
                self.num_retrain += 1
            else:
                self.grad_norm_approx_sum += self.grad_norm_approx[index,self.run]
            self.acc_removal[0, index,self.run] = lr_eval(self.w_approx, X_val_new, self.y_val)[1]
            self.acc_removal[1, index,self.run] = lr_eval(self.w_approx, X_test_new, self.y_test)[1]
        self.removal_times[index,self.run] = time.time() - start
        if index==self.num_removes-1:
            average = self.removal_times[:, self.run].mean()
            self.avg_training_time[self.run] = average
        self.X_old = X_new.clone().detach()
        if index % self.args["disp"] == 0:
            print('Iteration %d: time = %.2fs, number of retrain = %d' % (index+1, self.removal_times[index,self.run], self.num_retrain))
            print('Val acc = %.4f, Test acc = %.4f' % (self.acc_removal[0, index,self.run], self.acc_removal[1, index,self.run]))
        
        
        if index == self.num_removes-1:
            self.average_f1[self.run] = self.acc_removal[1, index,self.run]
            # softlabel_original0 = F.softmax(self.X_scaled_copy_guo[self.nonmember_id].mm(self.w),dim = 1)
            # softlabel_original1 = F.softmax(self.X_scaled_copy_guo[self.member_id].mm(self.w),dim = 1)
            # softlabel_new0 = F.softmax(self.X_scaled_copy_guo[self.nonmember_id].mm(self.w_approx),dim = 1)
            # softlabel_new1 = F.softmax(self.X_scaled_copy_guo[self.member_id].mm(self.w_approx),dim = 1)
            # mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
            # posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
            # posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
            # posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
            
            # auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            # self.logger.info('auc:{}'.format(auc))
            # self.average_auc[self.run] = auc
            # # plot_auc(mia_test_y, posterior.reshape(-1, 1))
            average = self.removal_times[:, self.run].sum()
            self.avg_training_time[self.run] = average

            self.logger.info('removal_time:{}'.format(average))
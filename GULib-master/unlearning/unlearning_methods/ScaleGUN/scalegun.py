import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from tqdm import tqdm
from task import get_trainer
from unlearning.unlearning_methods.ScaleGUN.progation_pkg import propagation
import os
import gc
from unlearning.unlearning_methods.ScaleGUN.utils import *
from unlearning.unlearning_methods.ScaleGUN.linear_unlearn_utils import *
from pipeline.IF_based_pipeline import IF_based_pipeline
from config import root_path,unlearning_path,unlearning_edge_path
from sklearn.metrics import roc_auc_score, roc_curve, auc

#----------------------
import shutil
#-------------- end --------------

class scalegun(IF_based_pipeline):
    """
    ScaleGUN is a class that implements the ScaleGUN unlearning method for graph neural networks.
    It extends the `IF_based_pipeline` class and provides methods for training, unlearning, and evaluating the model. 
    The class handles the preparation of data, training of the model, and the unlearning process, which involves removing specific edges or nodes from the graph and updating the model accordingly.
    
    Class Attributes:
        args (dict): A dictionary of arguments and configurations for the model.

        logger (Logger): A logger instance for logging information and metrics.

        model_zoo (ModelZoo): An instance of ModelZoo containing the model and data.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args["unlearn_trainer"] = 'ScaleGUNTrainer'
        num_runs = self.args["num_runs"]
        self.run = 0
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_unlearning_time = np.zeros(num_runs)
        
        #---------------------
        # Create saved_models directory directly within the ScaleGUN directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"Created saved_models directory at {self.save_dir}")
        #-------------- end --------------
        # self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
    
    # def run_exp(self):
    #     common(args=self.args,data=self.data,dataset=self.args["dataset_name"],result_path="./data/ScaleGUN/unlearning_data",normalized_dim="column")
    #     start = time.perf_counter()
    #     weights = get_prop_weight(self.args["weight_mode"], self.args["prop_step"], self.args["decay"])

    #     feat = preprocess_data(self.data.x, axis_num=self.args["axis_num"])

    #     column_sum_avg = feat.abs().sum(axis=0).mean()
    #     self.logger.info(f"column_sum_avg: {column_sum_avg}")
    #     self.args["rmax"] = self.args["rmax"]*column_sum_avg
    #     feat = feat.T
    #     origin_embedding = np.copy(feat.numpy())
        
    #     if self.args["dataset_name"] in ["ogbn-arxiv", "ogbn-products", "pokec"]:
    #         # transpose due to the discrepancy between Eigen and Python
    #         g = propagation.InstantGNN_transpose()
    #     else:
    #         g = propagation.InstantGNN()
    #     del_path = os.path.join(self.args["path"], self.args["del_path_suffix"])
        
    #     prop_time = g.init_push_graph(del_path, self.args["dataset_name"], origin_embedding,
    #                                 self.data.edge_index.T.numpy(), self.args["prop_step"], self.args["r"], weights, self.args["num_threads"], self.args["rmax"])
    #     # print(origin_embedding)
    #     self.logger.info(f"initial prop time: {prop_time}")
    #     row = self.data.edge_index[0].long()
    #     deg = degree(row, feat.shape[1])
    #     # del edge_index
    #     gc.collect()
    #     init_finish_time = time.perf_counter()
    #     # print(origin_embedding)
    #     # X = torch.FloatTensor(origin_embedding.T)
    #     X = torch.FloatTensor(feat.numpy().T)
    #     # print(X)
    #     # logger.debug(
    #     #     f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
    #     self.data.y = self.data.y.long()
    #     feat_dim = self.data.x.shape[1]
    #     num_classes = self.data.y.max().item() + 1
    #     X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(
    #         self.data, X, self.args["train_mode"], self.args["Y_binary"])
    #     # X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = X[self.data.train_mask], X[self.data.val_mask], X[self.data.test_mask], self.data.y[self.data.train_mask], self.data.y[self.data.val_mask], self.data.y[self.data.test_mask], self.data.train_mask, self.data.val_mask, self.data.test_mask
    #     # print(X_train,y_train.shape)
    #     del X
    #     # del data
    #     self.logger.info(
    #         "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
    #             X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
    #         )
    #     )
    #     train_size = X_train.shape[0]

    #     if self.args["compare_gnorm"]:
    #         b_std = 0
    #     else:
    #         b_std = self.args["std"]
    #     self.logger.info("--------------------------")
    #     self.logger.info("Training...")
    #     train_time = time.perf_counter()
    #     if self.args["train_mode"] == "ovr":
    #         b = b_std * torch.randn(feat_dim, num_classes).float().to(self.device)
    #     else:  # binary classification
    #         b = b_std * torch.randn(feat_dim).float().to(self.device)
    #     best_reg_lambda, best_lr, best_wd = self.args["lam"], self.args["lr"], self.args["wd"]
    #     X_train = X_train.to(self.device)
    #     y_train = y_train.to(self.device)
    #     # logger.info(f"b:{b}")
    #     if self.args["train_mode"] == "ovr":
    #         # print(X_train,y_train,best_reg_lambda,b,best_lr,best_wd)
    #         w = ovr_lr_optimize(
    #             X_train,
    #             y_train,
    #             best_reg_lambda,
    #             weight=None,
    #             b=b,
    #             verbose=self.args["verbose"],
    #             opt_choice=self.args["optimizer"],
    #             lr=best_lr,
    #             wd=best_wd,
    #             # X_val=X_val,
    #             # y_val=y_val,
    #         )
    #     else:
    #         w = lr_optimize(
    #             X_train,
    #             y_train,
    #             best_reg_lambda,
    #             b=b,
    #             num_steps=self.args["epochs"],
    #             verbose=self.args["verbose"],
    #             opt_choice=self.args["optimizer"],
    #             lr=self.args["lr"],
    #             wd=self.args["wd"],
    #         )
    #     # print(w)
    #     train_finish_time = time.perf_counter()
    #     accum_un_grad_norm = 0.0
    #     opt_grad_norm = 0.0
    #     accum_un_grad_norm_arr = torch.zeros(self.args["num_batch_removes"]).float()
    #     accum_un_worst_grad_norm_arr = torch.zeros(self.args["num_batch_removes"]).float()
    #     if self.args["train_mode"] == "ovr":
    #         for k in range(y_train.size(1)):
    #             opt_grad_norm += lr_grad(w[:, k], X_train,
    #                                     y_train[:, k], best_reg_lambda).norm().cpu()
    #     else:
    #         grad_old = lr_grad(w, X_train, y_train, best_reg_lambda)
    #         opt_grad_norm = grad_old.norm().cpu()
    #     accum_un_worst_grad_norm = 0.0
    #     self.logger.info("init cost: %.6fs" % (init_finish_time - start))
    #     self.logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
    #     accum_un_worst_grad_norm_arr[0] = accum_un_grad_norm

    #     X_val = X_val.to(self.device)
    #     y_val = y_val.to(self.device)
    #     X_test = X_test.to(self.device)
    #     y_test = y_test.to(self.device)
    #     if self.args["train_mode"] == "ovr":
    #         val_acc = ovr_lr_eval(w, X_val, y_val)
    #         test_acc = ovr_lr_eval(w, X_test, y_test)
    #     else:
    #         val_acc = lr_eval(w, X_val, y_val)
    #         test_acc = lr_eval(w, X_test, y_test)
    #     self.logger.info("Validation accuracy: %.4f" % val_acc)
    #     self.logger.info("Test accuracy: %.4f" % test_acc)
    #     update_cost = [prop_time,]
    #     unlearn_cost = [train_finish_time - train_time,]
    #     tot_cost = [train_finish_time - train_time+prop_time,]
    #     acc_removal = [[val_acc.item()], [test_acc.item()]]
    #     self.logger.info("first train cost: %.6fs" % (train_finish_time - train_time))

    #     # remove
    #     self.logger.info("start to remove edges...")
    #     self.logger.info("*" * 20)

    #     ###########
    #     # budget for removal
    #     c_val = get_c(self.args["delta"])
    #     if self.args["compare_gnorm"] or self.args["no_retrain"]:
    #         budget = 1e9
    #     else:
    #         if self.args["train_mode"] == "ovr":
    #             budget = get_budget(b_std, self.args["eps"], c_val) * y_train.size(1)
    #         else:
    #             budget = get_budget(b_std, self.args["eps"], c_val)
    #     gamma = 1 / 4  # pre-computed for -logsigmoid loss
    #     self.logger.debug(f"Budget: {budget}")

    #     start_time = time.perf_counter()
    #     grad_norm_approx = torch.zeros(self.args["num_batch_removes"]).float()
    #     grad_norm_worst = torch.zeros(self.args["num_batch_removes"]).float()
    #     grad_norm_real = torch.zeros(self.args["num_batch_removes"]).float()

    #     grad_norm_approx_sum = 0.0
    #     num_retrain = 0

    #     # obtain delete edges
    #     edge_idx_start = self.args["edge_idx_start"]
    #     edge_file = del_path + "/" + self.args["dataset_name"] + "/" + \
    #         self.args["dataset_name"] + f"_del_edges.npy"
    #     del_edges = np.load(edge_file)
    #     self.logger.info(f"read del_edges from {edge_file}")
    #     if del_edges.shape[1] == 2:
    #         del_edges = del_edges.T

    #     w_approx = w.clone().detach().to(self.device)
    #     X_train_old = X_train.clone().detach().to(self.device)
    #     del X_train
    #     del X_val
    #     del X_test
    #     gc.collect()

    #     for i in range(self.args["num_batch_removes"]):
    #         edges = del_edges[
    #             :,
    #             edge_idx_start
    #             + i * self.args["num_removes"]: edge_idx_start
    #             + self.args["num_removes"] * (i + 1),
    #         ].T.tolist()
    #         return_time = g.UpdateEdges(
    #             edges, origin_embedding, self.args["num_threads"], self.args["rmax"])
    #         update_cost.append(return_time)
    #         residue = np.zeros(feat_dim)
    #         g.GetResidueSum(residue)
    #         column_sum_norm = LA.norm(residue, 2)
    #         X_new = torch.FloatTensor(feat.numpy().T)
    #         X_train_new = X_new[train_mask].to(self.device)
    #         update_finish_time = time.perf_counter()
    #         K = get_K_matrix(X_train_new)
    #         spec_norm = sqrt_spectral_norm(K)
    #         if self.args["compare_gnorm"]:
    #             groundtruth = np.copy(feat.numpy())
    #             g.PowerMethod(groundtruth)
    #             X_groundtruth = torch.FloatTensor(groundtruth.T)
    #             X_groundtruth_train = X_groundtruth[train_mask].to(self.device)
    #         if self.args["train_mode"] == "ovr":
    #             for k in range(y_train.size(1)):
    #                 y_rem = y_train[:, k]

    #                 H_inv = lr_hessian_inv(
    #                     w_approx[:, k], X_train_new, y_rem, best_reg_lambda
    #                 )
    #                 grad_old = lr_grad(
    #                     w_approx[:, k], X_train_old, y_rem, best_reg_lambda)
    #                 grad_new = lr_grad(
    #                     w_approx[:, k], X_train_new, y_rem, best_reg_lambda)
    #                 grad_i = grad_old - grad_new
    #                 Delta = H_inv.mv(grad_i)
    #                 w_approx[:, k] += Delta
    #                 Delta_p = X_train_new.mv(Delta)
    #                 # here, grad_norm_approx store the norm induced by unlearning, that is, the second term of data-dependent bound
    #                 grad_norm_approx[i] += (Delta.norm() *
    #                                         Delta_p.norm() * spec_norm * gamma).cpu()
    #                 if self.args['compare_gnorm']:
    #                     grad_gt_k = lr_grad(
    #                         w_approx[:, k], X_groundtruth_train, y_rem, best_reg_lambda)
    #                     grad_norm_real[i] += grad_gt_k.norm().cpu()
    #             if self.args['compare_gnorm']:
    #                 approximation_worst_norm, unlearning_worst_norm = get_worst_Gbound_edge(
    #                     deg[edges[0][0]], deg[edges[0][1]], train_size, feat_dim, self.args.lam, self.args.rmax, self.data.num_nodes, self.args.prop_step)
    #                 accum_un_worst_grad_norm += unlearning_worst_norm * \
    #                     y_train.size(1)
    #                 grad_norm_worst[i] = y_train.size(
    #                     1)*approximation_worst_norm+accum_un_worst_grad_norm
    #                 accum_un_worst_grad_norm_arr[i] = accum_un_worst_grad_norm
    #             approximation_norm = column_sum_norm*2*y_train.shape[1]
    #             accum_un_grad_norm += grad_norm_approx[i]
    #             accum_un_grad_norm_arr[i] = accum_un_grad_norm
    #             grad_norm_approx[i] = approximation_norm + accum_un_grad_norm
    #             if grad_norm_approx[i] > budget:
    #                 self.logger.info(
    #                     f"The {i}-th removal, grad_norm_approx: {grad_norm_approx[i]}, approximation_norm: {approximation_norm}, retraining..."
    #                 )
    #                 accum_un_grad_norm = 0.0
    #                 b = b_std * torch.randn(feat_dim,
    #                                         num_classes).float().to(self.device)
    #                 w_approx = ovr_lr_optimize(
    #                     X_train_new,
    #                     y_train,
    #                     best_reg_lambda,
    #                     weight=None,
    #                     b=b,
    #                     verbose=self.args['verbose'],
    #                     opt_choice=self.args['optimizer'],
    #                     lr=best_lr,
    #                     wd=best_wd,
    #                     # X_val=X_val,
    #                     # y_val=y_val,
    #                 )
    #                 num_retrain += 1
    #             remove_finish_time = time.perf_counter()
    #             X_val_new = X_new[val_mask].to(self.device)
    #             acc_removal[0].append(ovr_lr_eval(
    #                 w_approx, X_val_new, y_val).item())
    #             X_test_new = X_new[test_mask].to(self.device)
    #             acc_removal[1].append(ovr_lr_eval(
    #                 w_approx, X_test_new, y_test).item())
    #         else:
    #             X_train_new = X_new[train_mask].to(self.device)
    #             y_train = y_train.to(self.device)
    #             H_inv = lr_hessian_inv(w_approx, X_train_new, y_train, self.args['lam'])
    #             # grad_i should be the difference
    #             grad_old = lr_grad(w_approx, X_train_old, y_train, self.args['lam'])
    #             grad_new = lr_grad(w_approx, X_train_new, y_train, self.args['lam'])
    #             grad_i = grad_old - grad_new
    #             Delta = H_inv.mv(grad_i)
    #             Delta_p = X_train_new.mv(Delta)
    #             w_approx += Delta
    #             grad_norm_approx[i] += (
    #                 Delta.norm() * Delta_p.norm() * spec_norm * gamma
    #             ).cpu()
    #             grad_old = lr_grad(w_approx, X_train_new, y_train, self.args['lam'])
    #             if self.args['compare_gnorm']:
    #                 grad_norm_real[i] = (
    #                     lr_grad(w_approx, X_groundtruth_train,
    #                             y_train, self.args['lam']).norm().cpu()
    #                 )
    #                 approximation_worst_norm, unlearning_worst_norm = get_worst_Gbound_edge(
    #                     deg[edges[0][0]], deg[edges[0][1]], train_size, feat_dim, self.args['lam'], self.args['rmax'], self.data.num_nodes, self.args['prop_step'])
    #                 accum_un_worst_grad_norm += unlearning_worst_norm
    #                 grad_norm_worst[i] = accum_un_worst_grad_norm + \
    #                     approximation_worst_norm
    #                 approximation_norm = column_sum_norm*2
    #                 accum_un_grad_norm += grad_norm_approx[i]
    #                 grad_norm_approx[i] = approximation_norm + accum_un_grad_norm
    #             if grad_norm_approx[i] > budget:
    #                 # retrain the model
    #                 accum_un_grad_norm = 0
    #                 b = b_std * torch.randn(X_new.size(1)).float().to(self.device)
    #                 w_approx = lr_optimize(
    #                     X_train_new,
    #                     y_train,
    #                     self.args['lam'],
    #                     b=b,
    #                     num_steps=self.args['epochs'],
    #                     verbose=False,
    #                     opt_choice=self.args['optimizer'],
    #                     lr=self.args['lr'],
    #                     wd=self.args['wd'],
    #                 )
    #                 num_retrain += 1

    #             remove_finish_time = time.perf_counter()
    #             acc_removal[0].append(lr_eval(w_approx, X_val_new, y_val).item())
    #             acc_removal[1].append(lr_eval(w_approx, X_test_new, y_test).item())
    #         unlearn_cost.append(remove_finish_time - update_finish_time)
    #         tot_cost.append(remove_finish_time - update_finish_time+return_time)
    #         X_train_old = X_train_new.clone().detach()
    #         if i % self.args['disp'] == 0:
    #             self.logger.info(
    #                 f"Iteration {i}: Edge del = {edges[0]}, grad_norm_approx = {grad_norm_approx[i]}, Val acc = {acc_removal[0][i+1]} Test acc = {acc_removal[1][i+1]}, avg update cost: {update_cost[i+1]}, avg unlearn cost:{unlearn_cost[i+1]}, avg tot cost:{tot_cost[i+1]}, num_retrain: {num_retrain}"
    #             )
    #     end_time = time.perf_counter()
    #     self.logger.info("update cost: %.6fs" %
    #             (sum(update_cost[1:]) / (len(update_cost)-1)))
    #     self.logger.info("unlearn cost: %.6fs" %
    #                 (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    #     self.logger.info("tot cost: %.6fs" % (sum(tot_cost[1:]) / (len(tot_cost)-1)))
    #     self.logger.info("tot cost: %.6fs" % (end_time - start_time))
        
        
        
    def train_original_model(self,run):
        """
        Trains the original model using the provided run configuration.
        This method prepares the data and trains the model based on the 
        specified run configuration. It is a part of the ScaleGUN 
        unlearning method.
        """
        self.prepare_data()
        self.train_model()
        pass
    
    def prepare_data(self):
        """
        Prepares the data for the ScaleGUN model.
        This function performs several preprocessing steps on the input data to prepare it for training and evaluation.
        It includes normalizing the data, calculating weights, and initializing the propagation graph. Additionally, it
        splits the data into training, validation, and test sets.
        """
        common(args=self.args,data=self.data,dataset=self.args["dataset_name"],result_path="./data/ScaleGUN/unlearning_data",normalized_dim="column")
        self.start = time.perf_counter()
        weights = get_prop_weight(self.args["weight_mode"], self.args["prop_step"], self.args["decay"])
        self.feat = preprocess_data(self.data.x, axis_num=self.args["axis_num"])
        column_sum_avg = self.feat.abs().sum(axis=0).mean()
        self.logger.info(f"column_sum_avg: {column_sum_avg}")
        self.args["rmax"] = self.args["rmax"]*column_sum_avg
        self.feat = self.feat.T
        self.origin_embedding = np.copy(self.feat.numpy())
        
        if self.args["dataset_name"] in ["ogbn-arxiv", "ogbn-products", "pokec"]:
            # transpose due to the discrepancy between Eigen and Python
            self.g = propagation.InstantGNN_transpose()
        else:
            self.g = propagation.InstantGNN()
        self.del_path = os.path.join(self.args["path"], self.args["del_path_suffix"])
        
        self.prop_time = self.g.init_push_graph(self.del_path, self.args["dataset_name"], self.origin_embedding,
                                    self.data.edge_index.T.numpy(), self.args["prop_step"], self.args["r"], weights, self.args["num_threads"], self.args["rmax"])
        # print(origin_embedding)
        self.logger.info(f"initial prop time: {self.prop_time}")
        row = self.data.edge_index[0].long()
        self.deg = degree(row, self.feat.shape[1])
        # del edge_index
        gc.collect()
        self.init_finish_time = time.perf_counter()
        # print(origin_embedding)
        # X = torch.FloatTensor(origin_embedding.T)
        X = torch.FloatTensor(self.feat.numpy().T)
        # print(X)
        # logger.debug(
        #     f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
        self.data.y = self.data.y.long()
        self.feat_dim = self.data.x.shape[1]
        self.num_classes = self.data.y.max().item() + 1
        
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = get_split(
            self.data, X, self.args["train_mode"], self.args["Y_binary"])
        # X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = X[self.data.train_mask], X[self.data.val_mask], X[self.data.test_mask], self.data.y[self.data.train_mask], self.data.y[self.data.val_mask], self.data.y[self.data.test_mask], self.data.train_mask, self.data.val_mask, self.data.test_mask
        # print(X_train,y_train.shape)
        # del X
        # del data
        
        # Ensure validation indices are set by taking 100 nodes from training set
        if not hasattr(self.data, 'val_indices') or len(getattr(self.data, 'val_indices', [])) == 0:
            # Get indices of training nodes
            train_indices = torch.where(self.train_mask)[0]
            
            # Randomly select 100 nodes from training set for validation
            val_size = min(100, len(train_indices) // 10)  # Take 100 or 10% of training, whichever is smaller
            perm = torch.randperm(len(train_indices))
            val_from_train_indices = train_indices[perm[:val_size]]
            
            # Update masks - remove selected nodes from training and add to validation
            # self.train_mask[val_from_train_indices] = False
            self.val_mask[val_from_train_indices] = True
            
            # Update the data object
            # self.data.train_mask = self.train_mask
            self.data.val_mask = self.val_mask
            self.data.val_indices = val_from_train_indices
            
            # Update the split data
            # self.X_train = X[self.train_mask]
            self.X_val = X[self.val_mask]
            # self.y_train = self.data.y[self.train_mask]
            self.y_val = self.data.y[self.val_mask]
            
            self.logger.info(f"Set validation indices: {val_size} nodes moved from training to validation")
        
        self.logger.info(
            "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
                self.X_train.shape[0], self.X_val.shape[0], self.X_test.shape[0], self.feat_dim, self.num_classes
            )
        )
        
        self.train_size = self.X_train.shape[0]
        if self.args["compare_gnorm"]:
            self.b_std = 0
        else:
            self.b_std = self.args["std"]
        self.logger.info("--------------------------")
        pass
    
    def train_model(self):
        """
        Trains the model based on the specified training mode and parameters.
        This function handles the training process for a model, supporting both 
        one-vs-rest (ovr) and binary classification modes. It initializes the 
        necessary parameters, performs optimization, and evaluates the model 
        on validation and test datasets. Additionally, it logs various metrics 
        and training costs.
        """
        self.logger.info("Training...")
        train_time = time.perf_counter()
        if self.args["train_mode"] == "ovr":
            b = self.b_std * torch.randn(self.feat_dim, self.num_classes).float().to(self.device)
        else:  # binary classification
            b = self.b_std * torch.randn(self.feat_dim).float().to(self.device)
        self.best_reg_lambda, self.best_lr, self.best_wd = self.args["lam"], self.args["lr"], self.args["wd"]
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        # logger.info(f"b:{b}")
        if self.args["train_mode"] == "ovr":
            # print(X_train,y_train,best_reg_lambda,b,best_lr,best_wd)
            self.w = ovr_lr_optimize(
                self.X_train,
                self.y_train,
                self.best_reg_lambda,
                weight=None,
                b=b,
                verbose=self.args["verbose"],
                opt_choice=self.args["optimizer"],
                lr=self.best_lr,
                wd=self.best_wd,
                # X_val=X_val,
                # y_val=y_val,
            )
        else:
            self.w = lr_optimize(
                self.X_train,
                self.y_train,
                self.best_reg_lambda,
                b=b,
                num_steps=self.args["epochs"],
                verbose=self.args["verbose"],
                opt_choice=self.args["optimizer"],
                lr=self.args["lr"],
                wd=self.args["wd"],
            )
        # print(w)
        train_finish_time = time.perf_counter()
        self.accum_un_grad_norm = 0.0
        opt_grad_norm = 0.0
        self.accum_un_grad_norm_arr = torch.zeros(self.args["num_batch_removes"]).float()
        self.accum_un_worst_grad_norm_arr = torch.zeros(self.args["num_batch_removes"]).float()
        if self.args["train_mode"] == "ovr":
            for k in range(self.y_train.size(1)):
                opt_grad_norm += lr_grad(self.w[:, k], self.X_train,
                                        self.y_train[:, k], self.best_reg_lambda).norm().cpu()
        else:
            grad_old = lr_grad(self.w, self.X_train, self.y_train, self.best_reg_lambda)
            opt_grad_norm = grad_old.norm().cpu()
        self.accum_un_worst_grad_norm = 0.0
        self.logger.info("init cost: %.6fs" % (self.init_finish_time - self.start))
        self.logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
        self.accum_un_worst_grad_norm_arr[0] = self.accum_un_grad_norm

        self.X_val = self.X_val.to(self.device)
        self.y_val = self.y_val.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        if self.args["train_mode"] == "ovr":
            val_acc = ovr_lr_eval(self.w, self.X_val, self.y_val)[1]
            test_acc = ovr_lr_eval(self.w, self.X_test, self.y_test)[1]
        else:
            val_acc = lr_eval(self.w, self.X_val, self.y_val)
            test_acc = lr_eval(self.w, self.X_test, self.y_test)
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = test_acc
        self.logger.info("Validation accuracy: %.4f" % val_acc)
        self.logger.info("Test accuracy: %.4f" % test_acc)
        self.update_cost = [self.prop_time,]
        self.unlearn_cost = [train_finish_time - train_time,]
        self.tot_cost = [train_finish_time - train_time+self.prop_time,]
        self.acc_removal = [[val_acc], [test_acc]]
        self.logger.info("first train cost: %.6fs" % (train_finish_time - train_time))
        # breakpoint()
        
    def unlearning_request(self):
        """
        Handles the process of unlearning specific edges or nodes from the graph data.
        This function initiates the unlearning process by setting up necessary parameters,
        logging the start of the process, and determining the budget for unlearning based
        on the provided arguments. It then identifies the edges or nodes to be removed
        and prepares the model for retraining if necessary.
        """
        self.logger.info("start to remove edges...")
        self.logger.info("*" * 20)
        c_val = get_c(self.args["delta"])
        if self.args["compare_gnorm"] or self.args["no_retrain"]:
            self.budget = 1e9
        else:
            if self.args["train_mode"] == "ovr":
                self.budget = get_budget(self.b_std, self.args["eps"], c_val) * self.y_train.size(1)
            else:
                self.budget = get_budget(self.b_std, self.args["eps"], c_val)
        self.gamma = 1 / 4  # pre-computed for -logsigmoid loss
        self.logger.debug(f"Budget: {self.budget}")
        self.start_time = time.perf_counter()
        self.grad_norm_approx = torch.zeros(self.args["num_batch_removes"]).float()
        self.grad_norm_worst = torch.zeros(self.args["num_batch_removes"]).float()
        self.grad_norm_real = torch.zeros(self.args["num_batch_removes"]).float()

        self.grad_norm_approx_sum = 0.0
        self.num_retrain = 0

        # obtain delete edges
        self.edge_idx_start = self.args["edge_idx_start"]
        # edge_file = self.del_path + "/" + self.args["dataset_name"] + "/" + \
        #     self.args["dataset_name"] + f"_del_edges.npy"
        if self.args['unlearn_task']=='node':
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            self.unlearning_nodes = np.loadtxt(path_un, dtype=int)
            unlearning_nodes = torch.tensor(self.unlearning_nodes)
            mask_start = torch.isin(self.data.edge_index[0], unlearning_nodes)
            mask_end = torch.isin(self.data.edge_index[1],  unlearning_nodes)
            mask = mask_start | mask_end
            self.del_edges = self.data.edge_index[:, mask]
            # breakpoint()
        else:
            #--------------------
            # Check if direct edge deletion file is provided
            if "edge_del_file" in self.args and self.args["edge_del_file"]:
                edge_del_file = self.args["edge_del_file"]
                self.logger.info(f"Using directly specified edge deletion file: {edge_del_file}")
                if os.path.exists(edge_del_file):
                    self.del_edges = np.loadtxt(edge_del_file, dtype=int).T
                    self.logger.info(f"Loaded {self.del_edges.shape[1]} edges for deletion from file")
                else:
                    self.logger.error(f"Edge deletion file not found: {edge_del_file}")
                    raise FileNotFoundError(f"Edge deletion file not found: {edge_del_file}")
            else:
                # Use the traditional path construction
                path_un_edge = unlearning_edge_path + "_" + str(self.run) + ".txt"
                if os.path.exists(path_un_edge):
                    self.del_edges = np.loadtxt(path_un_edge, dtype=int).T
                    self.logger.info(f"Loaded edges from default path: {path_un_edge}")
                else:
                    self.logger.error(f"Default edge deletion file not found: {path_un_edge}")
                    raise FileNotFoundError(f"Default edge deletion file not found: {path_un_edge}")
            #-------------- end --------------
        # self.del_edges = np.load(edge_file)
        # self.logger.info(f"read del_edges from {edge_file}")
        # if self.del_edges.shape[1] == 2:
        #     self.del_edges = self.del_edges.T

        self.w_approx = self.w.clone().detach().to(self.device)
        self.X_train_old = self.X_train.clone().detach().to(self.device)
        del self.X_train
        del self.X_val
        del self.X_test
        gc.collect()
        pass
        
    def unlearn(self):
        """
        Unlearns specific edges from the graph and updates the model accordingly.
        This function iteratively removes edges from the graph, updates the model's weights, and evaluates the performance 
        after each removal. It also compares the gradient norms and retrains the model if necessary.
        """
        # breakpoint()
        X_train_early_old = self.X_train_old.clone().detach().to(self.device)
        for i in range(self.args["num_batch_removes"]):
            edges = self.del_edges[
                :,
                self.edge_idx_start
                + i * self.args["num_removes"]: self.edge_idx_start
                + self.args["num_removes"] * (i + 1),
            ].T.tolist()
            return_time = self.g.UpdateEdges(
                edges, self.origin_embedding, self.args["num_threads"], self.args["rmax"])
            self.update_cost.append(return_time)
            residue = np.zeros(self.feat_dim)
            self.g.GetResidueSum(residue)
            column_sum_norm = LA.norm(residue, 2)
            X_new = torch.FloatTensor(self.feat.numpy().T)
            X_train_new = X_new[self.train_mask].to(self.device)
            update_finish_time = time.perf_counter()
            K = get_K_matrix(X_train_new)
            spec_norm = sqrt_spectral_norm(K)
            if self.args["compare_gnorm"]:
                groundtruth = np.copy(self.feat.numpy())
                self.g.PowerMethod(groundtruth)
                X_groundtruth = torch.FloatTensor(groundtruth.T)
                X_groundtruth_train = X_groundtruth[self.train_mask].to(self.device)
            if self.args["train_mode"] == "ovr":
                for k in range(self.y_train.size(1)):
                    y_rem = self.y_train[:, k]

                    H_inv = lr_hessian_inv(
                        self.w_approx[:, k], X_train_new, y_rem, self.best_reg_lambda
                    )
                    grad_old = lr_grad(
                        self.w_approx[:, k], self.X_train_old, y_rem, self.best_reg_lambda)
                    grad_new = lr_grad(
                        self.w_approx[:, k], X_train_new, y_rem, self.best_reg_lambda)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    self.w_approx[:, k] += Delta
                    Delta_p = X_train_new.mv(Delta)
                    # here, grad_norm_approx store the norm induced by unlearning, that is, the second term of data-dependent bound
                    self.grad_norm_approx[i] += (Delta.norm() *
                                            Delta_p.norm() * spec_norm * self.gamma).cpu()
                    if self.args['compare_gnorm']:
                        grad_gt_k = lr_grad(
                            self.w_approx[:, k], X_groundtruth_train, y_rem, self.best_reg_lambda)
                        self.grad_norm_real[i] += grad_gt_k.norm().cpu()
                if self.args['compare_gnorm']:
                    approximation_worst_norm, unlearning_worst_norm = get_worst_Gbound_edge(
                        self.deg[edges[0][0]], self.deg[edges[0][1]], self.train_size, self.feat_dim, self.best_reg_lambda, self.args["rmax"], self.data.num_nodes, self.args["prop_step"])
                    self.accum_un_worst_grad_norm += unlearning_worst_norm * self.y_train.size(1)
                    self.grad_norm_worst[i] = self.y_train.size(1)*approximation_worst_norm+self.accum_un_worst_grad_norm
                    self.accum_un_worst_grad_norm_arr[i] = self.accum_un_worst_grad_norm
                approximation_norm = column_sum_norm*2*self.y_train.shape[1]
                self.accum_un_grad_norm += self.grad_norm_approx[i]
                self.accum_un_grad_norm_arr[i] = self.accum_un_grad_norm
                self.grad_norm_approx[i] = approximation_norm + self.accum_un_grad_norm
                if self.grad_norm_approx[i] > self.budget:
                    self.logger.info(
                        f"The {i}-th removal, grad_norm_approx: {self.grad_norm_approx[i]}, approximation_norm: {approximation_norm}, retraining..."
                    )
                    self.accum_un_grad_norm = 0.0
                    b = self.b_std * torch.randn(self.feat_dim,
                                            self.num_classes).float().to(self.device)
                    self.w_approx = ovr_lr_optimize(
                        X_train_new,
                        self.y_train,
                        self.best_reg_lambda,
                        weight=None,
                        b=b,
                        verbose=self.args['verbose'],
                        opt_choice=self.args['optimizer'],
                        lr=self.best_lr,
                        wd=self.best_wd,
                        # X_val=X_val,
                        # y_val=y_val,
                    )
                    self.num_retrain += 1
                remove_finish_time = time.perf_counter()
                X_val_new = X_new[self.val_mask].to(self.device)
                self.acc_removal[0].append(ovr_lr_eval(
                    self.w_approx, X_val_new, self.y_val)[1])
                X_test_new = X_new[self.test_mask].to(self.device)
                self.acc_removal[1].append(ovr_lr_eval(
                    self.w_approx, X_test_new, self.y_test)[1])
                # Update X_train_old for next iteration
                self.X_train_old = X_train_new.clone().detach()
            else:
                X_train_new = X_new[self.train_mask].to(self.device)
                self.y_train = self.y_train.to(self.device)
                H_inv = lr_hessian_inv(self.w_approx, X_train_new, self.y_train, self.args['lam'])
                # grad_i should be the difference
                grad_old = lr_grad(self.w_approx, self.X_train_old, self.y_train, self.args['lam'])
                grad_new = lr_grad(self.w_approx, X_train_new, self.y_train, self.args['lam'])
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_train_new.mv(Delta)
                self.w_approx += Delta
                self.grad_norm_approx[i] += (
                    Delta.norm() * Delta_p.norm() * spec_norm * self.gamma
                ).cpu()
                grad_old = lr_grad(self.w_approx, X_train_new, self.y_train, self.args['lam'])
                if self.args['compare_gnorm']:
                    self.grad_norm_real[i] = (
                        lr_grad(self.w_approx, X_groundtruth_train,
                                self.y_train, self.args['lam']).norm().cpu()
                    )
                    approximation_worst_norm, unlearning_worst_norm = get_worst_Gbound_edge(
                        self.deg[edges[0][0]], self.deg[edges[0][1]], self.train_size, self.feat_dim, self.args['lam'], self.args['rmax'], self.data.num_nodes, self.args['prop_step'])
                    self.accum_un_worst_grad_norm += unlearning_worst_norm
                    self.grad_norm_worst[i] = self.accum_un_worst_grad_norm + \
                        approximation_worst_norm
                    approximation_norm = column_sum_norm*2
                    self.accum_un_grad_norm += self.grad_norm_approx[i]
                    self.grad_norm_approx[i] = approximation_norm + self.accum_un_grad_norm
                if self.grad_norm_approx[i] > self.budget:
                    # retrain the model
                    self.accum_un_grad_norm = 0
                    b = self.b_std * torch.randn(X_new.size(1)).float().to(self.device)
                    self.w_approx = lr_optimize(
                        X_train_new,
                        self.y_train,
                        self.args['lam'],
                        b=b,
                        num_steps=self.args['epochs'],
                        verbose=False,
                        opt_choice=self.args['optimizer'],
                        lr=self.args['lr'],
                        wd=self.args['wd'],
                    )
                    self.num_retrain += 1

                remove_finish_time = time.perf_counter()
                self.acc_removal[0].append(lr_eval(self.w_approx, X_val_new, self.y_val).item())
                self.acc_removal[1].append(lr_eval(self.w_approx, X_test_new, self.y_test).item())
            self.unlearn_cost.append(remove_finish_time - update_finish_time)
            self.tot_cost.append(remove_finish_time - update_finish_time+return_time)
            # Update X_train_old for next iteration
            self.X_train_old = X_train_new.clone().detach()
            if i % self.args['disp'] == 0:
                self.logger.info(
                    f"Iteration {i}: Edge del = {edges[0]}, grad_norm_approx = {self.grad_norm_approx[i]}, Val acc = {self.acc_removal[0][i+1]} Test acc = {self.acc_removal[1][i+1]}, avg update cost: {self.update_cost[i+1]}, avg unlearn cost:{self.unlearn_cost[i+1]}, avg tot cost:{self.tot_cost[i+1]}, num_retrain: {self.num_retrain}"
                )
        
        end_time = time.perf_counter()
        self.logger.info("update cost: %.6fs" %
                (sum(self.update_cost[1:]) / (len(self.update_cost)-1)))
        self.logger.info("unlearn cost: %.6fs" %
                    (sum(self.unlearn_cost[1:]) / (len(self.unlearn_cost)-1)))
        self.logger.info("tot cost: %.6fs" % (sum(self.tot_cost[1:]) / (len(self.tot_cost)-1)))
        self.logger.info("tot cost: %.6fs" % (end_time - self.start_time))
        self.average_f1[self.run] = self.acc_removal[1][-1]
        self.avg_unlearning_time[self.run] = sum(self.tot_cost[1:]) / (len(self.tot_cost)-1)
        
        # Evaluate final unlearned model stats (similar to base model)
        X_new_final = torch.FloatTensor(self.feat.numpy().T)
        X_val_new_final = X_new_final[self.val_mask].to(self.device)
        X_test_new_final = X_new_final[self.test_mask].to(self.device)
        
        if self.args["train_mode"] == "ovr":
            unlearned_val_acc, unlearned_val_f1, unlearned_val_recall = ovr_lr_eval(self.w_approx, X_val_new_final, self.y_val)
            unlearned_test_acc, unlearned_test_f1, unlearned_test_recall = ovr_lr_eval(self.w_approx, X_test_new_final, self.y_test)
            self.logger.info("=" * 60)
            self.logger.info("Final Unlearned Model Stats:")
            self.logger.info("Validation accuracy: %.4f, F1: %.4f, Recall: %.4f" % (unlearned_val_acc, unlearned_val_f1, unlearned_val_recall))
            self.logger.info("Test accuracy: %.4f, F1: %.4f, Recall: %.4f" % (unlearned_test_acc, unlearned_test_f1, unlearned_test_recall))
            self.logger.info("=" * 60)
        else:
            unlearned_val_acc = lr_eval(self.w_approx, X_val_new_final, self.y_val)
            unlearned_test_acc = lr_eval(self.w_approx, X_test_new_final, self.y_test)
            self.logger.info("=" * 60)
            self.logger.info("Final Unlearned Model Stats:")
            self.logger.info("Validation accuracy: %.4f" % unlearned_val_acc)
            self.logger.info("Test accuracy: %.4f" % unlearned_test_acc)
            self.logger.info("=" * 60)
        
        #----------------------
        # Save the models after unlearning
        # breakpoint()
        self.save_models()
        #-------------- end --------------
        
        if self.args['unlearn_task'] == "node":
            self.member_id = self.unlearning_nodes
            self.nonmember_id = self.data.test_indices[:self.args["num_unlearned_nodes"]]
            if self.args['unlearn_task'] == "node" and self.args['downstream_task'] == "node":
                X_ = torch.FloatTensor(self.feat.numpy().T).cuda()
                softlabel_original0 = F.softmax(X_[self.nonmember_id].mm(self.w),dim = 1)
                softlabel_original1 = F.softmax(X_[self.member_id].mm(self.w),dim = 1)
                softlabel_new0 = F.softmax(X_[self.nonmember_id].mm(self.w_approx),dim = 1)
                softlabel_new1 = F.softmax(X_[self.member_id].mm(self.w_approx),dim = 1)
                mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
                posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
                posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
        
        #------------------
    def save_models(self):
        """
        Saves the original model, unlearnt model, and retrained model (from scratch) to disk.
        
        The models are saved in the 'saved_models' directory in the parent folder.
        The original model is saved after initial training, the unlearnt model is saved
        after the unlearning process, and a retrained model is created by training from
        scratch on the dataset with deleted datapoints.
        """
        dataset_name = self.args["dataset_name"]
        unlearn_task = self.args["unlearn_task"]
        run_id = self.run
        
        # Create run-specific directory
        run_save_dir = os.path.join(self.save_dir, f"{dataset_name}_{unlearn_task}_run{run_id}")
        os.makedirs(run_save_dir, exist_ok=True)
        
        # Save original model
        original_model_path = os.path.join(run_save_dir, "original_model.pt")
        torch.save(self.w, original_model_path)
        self.logger.info(f"Original model saved to {original_model_path}")
        
        # Save unlearnt model
        unlearnt_model_path = os.path.join(run_save_dir, "unlearnt_model.pt")
        torch.save(self.w_approx, unlearnt_model_path)
        self.logger.info(f"Unlearnt model saved to {unlearnt_model_path}")
        
        # Train and save a model from scratch on dataset without deleted datapoints
        self.logger.info("Training model from scratch on dataset without deleted datapoints...")
        retrained_model = self.train_from_scratch()
        retrained_model_path = os.path.join(run_save_dir, "retrained_model.pt")
        torch.save(retrained_model, retrained_model_path)
        self.logger.info(f"Retrained model saved to {retrained_model_path}")
        
    def train_from_scratch(self):
        """
        Trains a model from scratch on the dataset with deleted datapoints removed.
        
        Returns:
            torch.Tensor: The trained model weights
        """
        self.logger.info("Preparing dataset without deleted datapoints...")
        
        # Get the current features after unlearning (with deleted edges/nodes removed)
        X_new = torch.FloatTensor(self.feat.numpy().T)
        X_train_new = X_new[self.train_mask].to(self.device)
        
        # Initialize model parameters
        if self.args["train_mode"] == "ovr":
            b = self.b_std * torch.randn(self.feat_dim, self.num_classes).float().to(self.device)
            # Train model from scratch
            retrained_model = ovr_lr_optimize(
                X_train_new,
                self.y_train,
                self.best_reg_lambda,
                weight=None,
                b=b,
                verbose=self.args['verbose'],
                opt_choice=self.args['optimizer'],
                lr=self.best_lr,
                wd=self.best_wd,
            )
        else:
            b = self.b_std * torch.randn(self.feat_dim).float().to(self.device)
            # Train model from scratch
            retrained_model = lr_optimize(
                X_train_new,
                self.y_train,
                self.best_reg_lambda,
                b=b,
                num_steps=self.args['epochs'],
                verbose=self.args['verbose'],
                opt_choice=self.args['optimizer'],
                lr=self.args['lr'],
                wd=self.args['wd'],
            )
        
        # Evaluate retrained model
        X_val_new = X_new[self.val_mask].to(self.device)
        X_test_new = X_new[self.test_mask].to(self.device)
        
        if self.args["train_mode"] == "ovr":
            val_acc = ovr_lr_eval(retrained_model, X_val_new, self.y_val)[1]
            test_acc = ovr_lr_eval(retrained_model, X_test_new, self.y_test)[1]
        else:
            val_acc = lr_eval(retrained_model, X_val_new, self.y_val)
            test_acc = lr_eval(retrained_model, X_test_new, self.y_test)
            
        self.logger.info(f"Retrained model - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")
        
        return retrained_model
        #-------------- end --------------
        
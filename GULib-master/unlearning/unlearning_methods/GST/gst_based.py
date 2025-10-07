import argparse
from itertools import product
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import time
from unlearning.unlearning_methods.GST.gst_main import *
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj, add_self_loops, contains_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader
import numpy as np
from utils import *
from utils.utils import *
# from unlearning.unlearning_methods.GST.train_unlearn import *
from config import root_path,unlearning_path
import ipdb
from config import RESET_COLOR,BLUE_COLOR
from task import get_trainer
from memory_profiler import profile
class gst():
    """
    GST-based Unlearning Method.
    This class implements the GST-based unlearning method for graph neural networks.
    It initializes with the necessary arguments, logger, and model zoo, and manages
    the unlearning process across multiple runs and folds. The class records performance
    metrics such as F1 score, AUC score, and training time.
    
    Class Attributes:
        args (dict): Configuration parameters for the unlearning process, including
            number of unlearned nodes, number of runs, budget, etc.

        model_zoo (ModelZoo): A container for the models and data used in the experiments.
        
        logger (Logger): Logger for recording information and metrics.
    """
    def __init__(self,args,logger,model_zoo):
        self.args = args
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self.logger  = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nonmember_id = self.data.test_indices[:args["num_unlearned_nodes"]]
        num_runs = self.args["num_runs"]
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)
        
    @profile
    def run_exp_mem(self):
        self.args["unlearn_trainer"] = 'GSTTrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        scattering = self.model_zoo.model

        # grad_norm_approx = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # Data dependent res grad norm
        # grad_norm_real = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # true res grad norm
        # grad_norm_worst = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # worst case res grad norm
        grad_norm_approx = torch.zeros( self.args["folds"]).float() # Data dependent res grad norm
        grad_norm_real = torch.zeros(self.args["folds"]).float() # true res grad norm
        grad_norm_worst = torch.zeros( self.args["folds"]).float() # worst case res grad norm
        removal_times = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # record the time of each removal
        acc_removal = torch.zeros((2, self.args["num_unlearned_nodes"], self.args["folds"])).float() # record the acc after removal
        num_retrain = torch.zeros((self.args["folds"],)).int()

        acc = torch.zeros((2, self.args["folds"])).float() # record the standard acc
        times = torch.zeros((self.args["folds"],)).float() # record the standard time


        for fold in range(self.args["num_runs"]):
            self.logger.info('='*20 + '  fold=' + str(fold) + '  ' + '='*20)
            if self.args["downstream_task"] != "graph":
                path_un = unlearning_path + "_" + str(fold) + ".txt"
                self.unlearning_nodes = np.loadtxt(path_un, dtype=int)
            
                train_idx, val_idx, test_idx = self.data.train_indices,self.data.test_indices,self.data.val_indices
                
                w, durations, acc[0,fold], acc[1,fold],softlabel_original0,softlabel_original1 = self.target_model.train_GST(self.logger,self.args, self.data, scattering, self.device,self.unlearning_nodes,self.nonmember_id)
            else:
                train_list = [self.data[i] for i in self.data.train_indices]
                test_list = [self.data[i] for i in self.data.test_indices]
                w, durations, acc[0,fold], acc[1,fold] = self.target_model.train_GST_graph(self.logger,self.args, train_list,test_list, scattering, self.device)
            
            times[fold] = durations[0]+durations[1]

         #=================Unlearning process====================#
            print('='*5+f'Start Unlearning Process for {self.args["base_model"]}'+'='*5)

            c_val = get_c(self.args["GST_delta"])
            budget = get_budget(self.args["std"], self.args["eps"], c_val) * self.data.num_classes
            budget = 1e5
            print('Budget:', budget)
            if not self.args["base_model"] == "GIN":
                w_approx = w.clone().detach() 

            if self.args["downstream_task"] != "graph":
                self.avg_training_time[fold], num_retrain[fold], self.average_f1[fold], grad_norm_approx[fold], grad_norm_real[fold], grad_norm_worst[fold], removal_queue,softlabel_new1, softlabel_new0 = self.target_model.Unlearn_GST(self.logger,self.args, scattering, self.data, device, w_approx, budget,self.unlearning_nodes,self.nonmember_id,nonlin=True, gamma=1/4,removal_queue = self.unlearning_nodes)
                mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
                posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
                posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
                auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                # self.logger.info("AUC:{}".format(auc))
                # self.logger.info("F1:{}".format(self.average_f1[fold]))
                self.average_auc[fold] = auc 
            else:
                self.avg_training_time[fold],num_retrain[fold],self.average_f1[fold], grad_norm_approx[fold], grad_norm_real[fold], grad_norm_worst[fold] = self.target_model.Unlearn_GST_graph(self.logger,self.args, scattering,train_list, device, w_approx, budget,nonlin=True, gamma=1/4,test_list = test_list)
        
        
    def run_exp(self):
        """
        Excutes the GST-based unlearning pipeline for the specified number of runs.
        The method initializes the GST trainer, trains the model, and then unlearns
        the specified number of nodes. It records the performance metrics for each run.
        """
        self.args["unlearn_trainer"] = 'GSTTrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        scattering = self.model_zoo.model

        # grad_norm_approx = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # Data dependent res grad norm
        # grad_norm_real = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # true res grad norm
        # grad_norm_worst = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # worst case res grad norm
        grad_norm_approx = torch.zeros( self.args["folds"]).float() # Data dependent res grad norm
        grad_norm_real = torch.zeros(self.args["folds"]).float() # true res grad norm
        grad_norm_worst = torch.zeros( self.args["folds"]).float() # worst case res grad norm
        removal_times = torch.zeros(self.args["num_unlearned_nodes"], self.args["folds"]).float() # record the time of each removal
        acc_removal = torch.zeros((2, self.args["num_unlearned_nodes"], self.args["folds"])).float() # record the acc after removal
        num_retrain = torch.zeros((self.args["folds"],)).int()

        acc = torch.zeros((2, self.args["folds"])).float() # record the standard acc
        times = torch.zeros((self.args["folds"],)).float() # record the standard time


        for fold in range(self.args["num_runs"]):
            self.logger.info('='*20 + '  fold=' + str(fold) + '  ' + '='*20)
            if self.args["downstream_task"] != "graph":
                path_un = unlearning_path + "_" + str(fold) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
                self.unlearning_nodes = np.loadtxt(path_un, dtype=int)
            
                train_idx, val_idx, test_idx = self.data.train_indices,self.data.test_indices,self.data.val_indices
                
                w, durations, acc[0,fold], acc[1,fold],softlabel_original0,softlabel_original1 = self.target_model.train_GST(self.logger,self.args, self.data, scattering, self.device,self.unlearning_nodes,self.nonmember_id)
            else:
                train_list = [self.data[i] for i in self.data.train_indices]
                test_list = [self.data[i] for i in self.data.test_indices]
                w, durations, acc[0,fold], acc[1,fold] = self.target_model.train_GST_graph(self.logger,self.args, train_list,test_list, scattering, self.device)
            
            times[fold] = durations[0]+durations[1]

         #=================Unlearning process====================#
            print('='*5+f'Start Unlearning Process for {self.args["base_model"]}'+'='*5)

            c_val = get_c(self.args["GST_delta"])
            budget = get_budget(self.args["std"], self.args["eps"], c_val) * self.data.num_classes
            budget = 1e5
            print('Budget:', budget)
            if not self.args["base_model"] == "GIN":
                w_approx = w.clone().detach() 

            if self.args["downstream_task"] != "graph":
                self.avg_training_time[fold], num_retrain[fold], self.average_f1[fold], grad_norm_approx[fold], grad_norm_real[fold], grad_norm_worst[fold], removal_queue,softlabel_new1, softlabel_new0 = self.target_model.Unlearn_GST(self.logger,self.args, scattering, self.data, device, w_approx, budget,self.unlearning_nodes,self.nonmember_id,nonlin=True, gamma=1/4,removal_queue = self.unlearning_nodes)
                mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
                posterior1 = torch.cat((softlabel_original1, softlabel_original0), 0).cpu().detach()
                posterior2 = torch.cat((softlabel_new1, softlabel_new0), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
                auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                # self.logger.info("AUC:{}".format(auc))
                # self.logger.info("F1:{}".format(self.average_f1[fold]))
                self.average_auc[fold] = auc 
            else:
                self.avg_training_time[fold],num_retrain[fold],self.average_f1[fold], grad_norm_approx[fold], grad_norm_real[fold], grad_norm_worst[fold] = self.target_model.Unlearn_GST_graph(self.logger,self.args, scattering,train_list, device, w_approx, budget,nonlin=True, gamma=1/4,test_list = test_list)
            
            

        # plot_auc(mia_test_y, posterior.reshape(-1, 1))
        self.logger.info(
            "{}Performance Metrics:\n"
            " - Average F1 Score: {:.4f} ± {:.4f}\n"
            " - Average AUC Score: {:.4f} ± {:.4f}\n"
            " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
                BLUE_COLOR,
                np.mean(self.average_f1), np.std(self.average_f1),
                np.mean(self.average_auc), np.std(self.average_auc),
                np.mean(self.avg_training_time), np.std(self.avg_training_time),
                RESET_COLOR
                )
            )

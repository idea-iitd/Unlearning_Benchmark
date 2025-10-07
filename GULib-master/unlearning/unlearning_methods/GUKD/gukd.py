from task import get_trainer
from config import BLUE_COLOR,RESET_COLOR
import numpy as np
import torch
import time
import torch.nn.functional as F
from utils.dataset_utils import *
from model.base_gnn.gukd_teacher import GCNII
import copy
from sklearn.metrics import roc_auc_score
from attack.MIA_attack import train_attack_model,train_shadow_model,generate_shadow_model_output,evaluate_attack_model,GCNShadowModel,AttackModel
from pipeline.Learning_based_pipeline import Learning_based_pipeline
class gukd(Learning_based_pipeline):
    """
    GUKD class implements a learning-based pipeline for performing unlearning tasks on GNNs. 
    It supports both node and edge unlearning, executing unlearning requests, training models, and evaluating the impact through membership inference attacks.
    It gets the knowledge from the teacher model and trains the target model using teacher knowledge distillation. 

    Class Attributes:
        args (dict): Configuration parameters for the unlearning process.

        logger (Logger): Logger instance for recording informational and debugging messages.

        model_zoo (ModelZoo): Collection of pre-trained models available for training and evaluation within the pipeline.

        strategy (int): Indicator of the unlearning strategy employed (e.g., 1 for KL divergence, 2 for MSE loss).
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.num_classes = self.data.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unlearning_number = args["num_unlearned_nodes"]
        num_runs = self.args["num_runs"]
        self.avg_partition_time = np.zeros(num_runs)
        self.average_f1 = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)
        self.avg_updating_time = np.zeros(num_runs)
        self.average_attack_auc = np.zeros(num_runs)

    def determine_target_model(self):
        """
        Determines and initializes the target model for the unlearning process.
        This method sets the 'unlearn_trainer' argument to 'GUKDTrainer' and obtains the corresponding trainer instance
        by calling 'get_trainer' with the current arguments, logger, model from the model zoo, and data.
        """
        self.args["unlearn_trainer"] = 'GUKDTrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        # self.train_test_split()
        
    def train_original_model(self):
        """
        Train the original teacher model.
        This method initializes and trains the teacher model using the GCNII architecture.
        It records the average training time, computes the original features from the trained model,
        and, if data poisoning is enabled for an edge unlearning task, evaluates and stores
        the poisoning F1 score.
        """
        start_time = time.time()
        teacher_model = GCNII(self.args,self.data.num_features, self.data.num_classes).to(self.device)
        self.teacher_model = get_trainer(self.args,self.logger,teacher_model,self.data)
        self.teacher_model.train()
        self.avg_training_time[self.run] = time.time() - start_time
        self.original_feature = self.teacher_model.model(self.data.x, self.data.edge_index)
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.teacher_model.evaluate()
        # out = self.teacher_model.model(self.retain_data.x, self.retain_data.edge_index)
        # # self.z_t = out[self.retain_data.test_mask]
        # self.z_t = out.detach()
    
        
    # def run_exp(self):
    #     for run in range(self.args['num_runs']):
    #         self.determine_target_model()
            
    #         # self.train_test_split()
    #         self.unlearn()
    #         self.pre_train()
    #         best_f1,avg_training_time=self.target_model.gukd_train(self.z_t)
    #         attack_auc_score = self.mia_attack()

            
    #         self.average_f1[run] = best_f1
    #         self.avg_training_time[run]=avg_training_time
    #         self.average_attack_auc[run] = attack_auc_score

    #     self.logger.info(
    #     "{}Performance Metrics:\n"
    #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average AUC Score: {:.4f} ± {:.4f}\n"
    #     " - Average Training Time: {:.4f} ± {:.4f} seconds{}\n".format(
    #         BLUE_COLOR,
    #         np.mean(self.average_f1), np.std(self.average_f1),
    #         np.mean(self.average_attack_auc), np.std(self.average_attack_auc),
    #         np.mean(self.avg_training_time), np.std(self.avg_training_time),
    #         RESET_COLOR
    #         )
    #     )
    
    def train_test_split(self):
        if not self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.data.train_indices, self.data.test_indices = train_test_split(np.arange((self.data.num_nodes)),train_size = 0.6,test_size=self.args['test_ratio'], random_state=100)
            save_train_test_split(self.logger,self.args,self.data.train_indices, self.data.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.test_indices))
            print(self.data.train_indices.size, self.data.test_indices.size)
        else:
            self.data = load_train_test_split(self.logger)
            # self.data.train_indices, self.data.test_indices

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.test_indices))

    # def unlearn(self):
    #     self.retain_data = copy.deepcopy(self.data)
    #     if self.args["downstream_task"]=="node":
    #         path_un = unlearning_path + "_" + str(self.run) + ".txt"
    #         unlearn_nodes = np.loadtxt(path_un, dtype=int)
    #         # unlearn_nodes = np.random.choice(np.arange((self.data.num_nodes)), self.unlearning_number, replace=False)
    #         keep_mask = np.ones(self.data.num_nodes, dtype=bool)
    #         keep_mask[unlearn_nodes] = False
            
    #         # Update train indices and mask
    #         retain_indices = keep_mask[self.retain_data.train_indices]

    #         self.retain_data.train_indices = np.array(self.retain_data.train_indices)

    #         self.retain_data.train_indices = self.retain_data.train_indices[retain_indices]
    #         self.retain_data.train_mask = torch.from_numpy(np.isin(np.arange(self.retain_data.num_nodes), self.retain_data.train_indices))
            
    #         # Remove edges connected to unlearned nodes
    #         edge_index = self.retain_data.edge_index.numpy()
    #         keep_edges = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    #         self.retain_data.edge_index = torch.tensor(edge_index[:, keep_edges], dtype=torch.long)
    #     elif self.args["downstream_task"]=="edge":
    #         path_un_edge = unlearning_edge_path + "_" + str(self.run) + ".txt"
    #         if os.path.exists(path_un_edge):
    #             remove_edges = np.loadtxt(path_un_edge, dtype=int)
    #         remove_edges_indices = []
    #         for edge in remove_edges:
    #             remove_edges_indices.append(np.where((self.data.edge_index[0] == edge[0]) & (self.data.edge_index[1] == edge[1]))[0][0])
    #         # 删除这些边
    #         edge_index = self.data.edge_index.numpy()
    #         keep_edges = np.ones(edge_index.shape[1], dtype=bool)
    #         keep_edges[remove_edges_indices] = False
    #         self.retain_data.edge_index = torch.tensor(edge_index[:, keep_edges], dtype=torch.long)
    def pre_train(self):
        """
        Pre-trains the teacher GCNII model on the provided dataset.
        This function initializes a GCNII teacher model with the specified arguments and data parameters,
        sets up the training environment, and trains the model using the training data. After training,
        it computes the output of the teacher model on the dataset and stores the detached output for further use.
        """
        teacher_model = GCNII(self.args,self.retain_data.num_features, self.retain_data.num_classes).to(self.device)
        self.teacher_model = get_trainer(self.args,self.logger,teacher_model,self.data)
        self.teacher_model.train()
        out = self.teacher_model.model(self.retain_data.x, self.retain_data.edge_index)
        # self.z_t = out[self.retain_data.test_mask]
        self.z_t = out.detach()
        

    def mia_attack(self):
        #------------------------------------------------------------------------------------------
        # """
        # Perform a Membership Inference Attack (MIA) to evaluate the vulnerability of the target model.
        # This method assesses whether specific nodes were part of the training data by comparing
        # the soft labels produced by the target model before and after unlearning certain nodes. It
        # computes the posterior differences between original and unlearning soft labels and calculates
        # the Area Under the Receiver Operating Characteristic Curve (AUC) to quantify the attack's effectiveness.
        # """
        # self.mia_num = self.unlearning_number
        # self.original_softlabels = F.softmax(self.target_model.model(
        #     self.data.x,self.data.edge_index),dim=1).clone().detach().float()
        # original_softlabels_member = self.original_softlabels[self.unlearn_nodes]
        # original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
        # # if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args[
        # #     "base_model"] == "SIGN":
        # #     unlearning_softlabels_member = F.softmax(self.target_model.model(self.features[self.unlearning_nodes]),dim=1)
        # #     unlearning_softlabels_non = self.target_model.model.get_softlabel(
        # #         self.features[self.data.test_indices[:self.mia_num]])
        # # else:
        # unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x,self.data.edge_index)[self.unlearn_nodes],dim=1)
        # unlearning_softlabels_non = F.softmax(self.target_model.model(
        #     self.data.x,self.data.edge_index)[self.data.test_indices[:self.mia_num]],dim=1)

        # mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        # posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        # posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        # posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        # # self.logger.info("posterior:{}".format(posterior))
        # auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        # self.average_auc[self.run] = auc
        # # self.logger.info("auc:{}".format(auc))
        # # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
        # return auc
        #----------------------------------------------------------------- end ------------------------
        pass

    # def mia_attack_edge(self):
    #     self.mia_num = self.unlearning_edges.shape[1]
    #     neg_edge_index = negative_sampling(edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,num_neg_samples=self.mia_num)
    #     original_softlabels_member = self.target_model.decode(self.original_feature,self.unlearning_edges)
    #     original_softlabels_non = self.target_model.decode(self.original_feature,neg_edge_index)
        
    #     out_unlearn = self.target_model.model(self.data.x,self.retain_data.edge_index)
    #     unlearning_softlabels_member = self.target_model.decode(out_unlearn,self.unlearning_edges)
    #     unlearning_softlabels_non = self.target_model.decode(out_unlearn,neg_edge_index)
    #     mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
    #     posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
    #     posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
    #     posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
    #     print(mia_test_y, posterior.reshape(-1, 1))
    #     auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))

    #     self.average_auc[self.run] = auc
    
    
    # def mia_attack(self):
    #     shadow_model = GCNShadowModel(self.data.num_features,64 ,self.data.num_classes)
    #     train_shadow_model(shadow_model,self.data)
    #     soft_labels_train,soft_labels_test = generate_shadow_model_output(shadow_model,self.data)

    #     attack_mdoel = train_attack_model(soft_labels_train,self.data.num_classes)
    #     attack_auc_score = evaluate_attack_model(attack_mdoel,soft_labels_test)

    #     return attack_auc_score


    def unlearning_request(self):
        """
        Performs an unlearning operation by removing specified nodes or edges from the dataset.
        """
        self.retain_data = copy.deepcopy(self.data)
        if self.args["unlearn_task"]=="node":
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            self.unlearn_nodes = np.loadtxt(path_un, dtype=int)
            # unlearn_nodes = np.random.choice(np.arange((self.data.num_nodes)), self.unlearning_number, replace=False)
            keep_mask = np.ones(self.data.num_nodes, dtype=bool)
            keep_mask[self.unlearn_nodes] = False
            
            # Update train indices and mask
            retain_indices = keep_mask[self.retain_data.train_indices]

            self.retain_data.train_indices = np.array(self.retain_data.train_indices)

            self.retain_data.train_indices = self.retain_data.train_indices[retain_indices]
            self.retain_data.train_mask = torch.from_numpy(np.isin(np.arange(self.retain_data.num_nodes), self.retain_data.train_indices))
            
            # Remove edges connected to unlearned nodes
            edge_index = self.retain_data.edge_index.detach().cpu().numpy()
            keep_edges = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
            self.retain_data.edge_index = torch.tensor(edge_index[:, keep_edges], dtype=torch.long)
        elif self.args["unlearn_task"]=="edge":
            path_un_edge = unlearning_edge_path + "_" + str(self.run) + ".txt"
            if os.path.exists(path_un_edge):
                remove_edges = np.loadtxt(path_un_edge, dtype=int)
            remove_edges_indices = []
            for edge in remove_edges:
                
                remove_edges_indices.append(torch.where((self.data.edge_index[0] == edge[0]) & (self.data.edge_index[1] == edge[1]))[0][0])
            
            # 删除这些边
            edge_index = self.data.edge_index
            keep_edges = torch.ones(edge_index.shape[1], dtype=bool)
            keep_edges[remove_edges_indices] = False
            self.retain_data.edge_index = torch.tensor(edge_index[:, keep_edges], dtype=torch.long)
            self.unlearning_edges = remove_edges.T
        out = self.teacher_model.model(self.retain_data.x.cuda(), self.retain_data.edge_index)
        # self.z_t = out[self.retain_data.test_mask]
        self.z_t = out.detach()
        
    def unlearn(self):
        """
        Performs the unlearning process by training the target model using the GUKD method.
        Stores the resulting best F1 score and the average unlearning time for the current run.
        """
        best_f1,avg_unlearning_time=self.target_model.gukd_train(self.z_t)
        self.average_f1[self.run] = best_f1
        self.avg_unlearning_time[self.run]=avg_unlearning_time
        

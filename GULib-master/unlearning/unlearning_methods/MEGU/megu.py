import torch
import torch.nn as nn
import numpy as np
import time
import copy
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import CorrectAndSmooth
from task.node_classification import NodeClassifier
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from utils.utils import sparse_mx_to_torch_sparse_tensor,normalize_adj,propagate,criterionKD,calc_f1
from config import BLUE_COLOR,RESET_COLOR
from config import unlearning_path,unlearning_edge_path
from sklearn.metrics import roc_auc_score
from task import get_trainer
from torch_geometric.transforms import SIGN
from torch_geometric.utils import negative_sampling
from sklearn.metrics import f1_score,roc_auc_score
from attack.MIA_attack import train_attack_model,train_shadow_model,generate_shadow_model_output,evaluate_attack_model,GCNShadowModel,AttackModel
from pipeline.Learning_based_pipeline import Learning_based_pipeline
from task import BaseTrainer
import os


class GATE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = torch.nn.Linear(dim, dim)

    def forward(self, x):
        t = x.clone()
        return self.lr(t)



class megu(Learning_based_pipeline):
    """
    The `megu` class is a specialized implementation of the `Learning_based_pipeline` class designed for graph unlearning tasks. It provides methods for training, unlearning, and evaluating graph neural network models. 
    The class handles different unlearning tasks such as node, edge, and feature unlearning, and performs membership inference attacks to assess the privacy of the unlearning process.

    Class Attributes:
        args (dict): Configuration arguments for the unlearning process.

        logger (Logger): Logger for recording the process.

        model_zoo (ModelZoo): Collection of pre-trained models and data.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self._data = copy.deepcopy(self.data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_feats = self.data.num_features
        self.num_layers = self.args["GNN_layer"]
        num_runs = self.args["num_runs"]
        self.run = 0
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_unlearning_time = np.zeros(num_runs)
        self.train_indices,self.test_indices = self.data.train_indices,self.data.test_indices
        self.train_mask,self.test_mask = self.data.train_mask,self.data.test_mask
        self.avg_unlearning_time_gold = np.zeros(num_runs)
        self.average_f1_gold = np.zeros(num_runs)        
        
        
        
    def determine_target_model(self):
        """
        Determines and sets the target model for the unlearning process.
        This function sets the 'unlearn_trainer' argument to 'MEGUTrainer' and 
        initializes the target model using the get_trainer function with the 
        provided arguments, logger, model from the model zoo, and data.
        """
        self.args["unlearn_trainer"] = 'MEGUTrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self._data)
        
    def train_original_model(self):
        """
        Trains the original model and evaluates it if poisoning is enabled.
        This method trains the original model by calling the internal _train_model method.
        If the 'poison' argument is set to True, it evaluates the target model and stores
        the F1 score for the current run.
        """
        self.logger.info('training target models, run %s' % self.run)
        run_training_time,_ = self._train_model(self.run)
        self.avg_training_time[self.run] = run_training_time
        # if self.args["poison"] and self.args["unlearn_task"]=="edge":
        #     self.poison_f1[self.run] = self.target_model.evaluate()
    def unlearning_request(self):
        """
        Handles the unlearning request based on the specified unlearning task.
        This function performs different unlearning operations on the graph data 
        depending on the task specified in `self.args["unlearn_task"]`. The tasks 
        can be 'node', 'edge', or 'feature', and the function updates the graph 
        data accordingly.
        """
        # self.data.x_unlearn = self.data.x.clone()
        # self.data.edge_index_unlearn = self.data.edge_index.clone()
        # edge_index = self.data.edge_index.cpu().numpy()
        # unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args["unlearn_task"] == 'node':
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            unique_nodes = np.loadtxt(path_un, dtype=int)
            self.unlearing_nodes = unique_nodes
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        if self.args["unlearn_task"] == 'edge':
            # num_unlearned_edges = int(self.data.edge_index_unlearn.shape[1] * self.args["unlearn_ratio"])
            path_un = unlearning_edge_path + "_" + str(self.run) + "_edges_" + str(self.args["num_unlearned_edges"]) + ".txt"
            remove_edges = np.loadtxt(path_un,dtype=int)
            unique_nodes = np.unique(remove_edges)
            # breakpoint()
            # self.data.edge_index = self.data.train_edge_index  #commented because we need all edge not just training
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_edges)

            
        if self.args["unlearn_task"] == 'feature':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.x_unlearn[unique_nodes] = 0.

        self.temp_node = unique_nodes
        # self.target_model.data = self.data
        # Create a copy of data with unlearned nodes/edges removed
        self.gold_data = self._create_gold_standard_data()

    def unlearn(self):
        """
        Unlearns specific nodes from the graph data.
        This function performs the following steps:
        1. Converts the adjacency matrix of the graph to a sparse tensor and normalizes it.
        2. Selects k-hop neighbors for the nodes in the graph.
        3. Calls the target model's `megu_unlearning` method to unlearn the specified nodes and records the average unlearning time and F1 score.
        4. Performs a membership inference attack (MIA) and records the average AUC score.
        """
        # self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(to_scipy_sparse_matrix(self.data.edge_index,num_nodes=self.data.num_nodes))).cuda()
        # self.neighbor_khop = self.neighbor_select(self.data.x.cuda())
        # self.avg_unlearning_time[self.run], self.average_f1[self.run] = self.target_model.megu_unlearning(self.temp_node,self.neighbor_khop,self.run)
        self.avg_unlearning_time_gold[self.run], self.average_f1_gold[self.run] = self.gold_standard_unlearning()
        if self.args["attack"]:
            self.average_auc[self.run] = self.mia_attack()
        

    def get_softlabels(self):
        """
        Computes and returns the soft labels for the unlearning nodes and a subset of test indices.
        This function generates soft labels (probability distributions over classes) for two sets of nodes:
        1. Unlearning nodes specified in `self.unlearing_nodes`.
        2. A subset of test indices specified in `self.data.test_indices`.
        The function first checks the base model type specified in `self.args["base_model"]`. If the base model is "SIGN",
        it computes the output using `self.data.xs`. Otherwise, it uses `self.data.x` and `self.data.edge_index`.
        """
        if self.args["base_model"] == "SIGN":
            out = self.target_model.model(self.data.xs)
        else:
            out = self.target_model.model(self.data.x,self.data.edge_index)
        mem_labels = F.softmax(out,dim = 1)[self.unlearing_nodes]
        non_labels = F.softmax(out,dim = 1)[self.data.test_indices[:self.args["num_unlearned_nodes"]]]
        return mem_labels,non_labels

    def _train_model(self, run):
        """
        Trains the target model using the provided data and measures the training time.
        """
        # self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        res = self.target_model.train()
        train_time = time.time() - start_time
        
        # self.data_store.save_target_model(run, self.target_model)
        # self.logger.info(f"Model training time: {train_time:.4f}")

        return train_time, res
        

    def neighbor_select(self, features):
        """
        Selects neighboring nodes based on cosine similarity of propagated features.
        This function identifies neighboring nodes that are influenced by unlearning nodes
        by comparing the cosine similarity between the propagated features and the reversed
        propagated features. It iteratively adjusts the similarity threshold to find the 
        most influential nodes and then filters out the nodes that are within a k-hop 
        subgraph but not part of the unlearning nodes.
        """
        temp_features = features.clone()
        pfeatures = propagate(temp_features, self.num_layers, self.adj)
        reverse_feature = self.reverse_features(temp_features)
        re_pfeatures = propagate(reverse_feature, self.num_layers, self.adj)

        cos = nn.CosineSimilarity()
        sim = cos(pfeatures, re_pfeatures)
        
        alpha = 0.1
        gamma = 0.1
        max_val = 0.
        while True:
            influence_nodes_with_unlearning_nodes = torch.nonzero(sim <= alpha).flatten().cpu()
            if len(influence_nodes_with_unlearning_nodes.view(-1)) > 0:
                temp_max = torch.max(sim[influence_nodes_with_unlearning_nodes])
            else:
                alpha = alpha + gamma
                continue

            if temp_max == max_val:
                break

            max_val = temp_max
            alpha = alpha + gamma

        # influence_nodes_with_unlearning_nodes = torch.nonzero(sim < 0.5).squeeze().cpu()
        neighborkhop, _, _, two_hop_mask = k_hop_subgraph(
            torch.tensor(self.temp_node,dtype=torch.long),
            self.num_layers,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        neighborkhop = neighborkhop[~np.isin(neighborkhop.cpu(), self.temp_node)]
        neighbor_nodes = []
        for idx in influence_nodes_with_unlearning_nodes:
            if idx in neighborkhop and idx not in self.temp_node:
                neighbor_nodes.append(idx.item())
        
        neighbor_nodes_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), neighbor_nodes))

        return neighbor_nodes_mask


    # def megu_training(self):
    #     operator = GATE(self.data.num_classes).to(self.device)

    #     optimizer = torch.optim.SGD(self.target_model.model.parameters(), lr=self.target_model.model.config.lr, weight_decay=self.target_model.model.config.decay)

            

    #     with torch.no_grad():
    #         self.target_model.model.eval()
    #         if self.args["base_model"] == "SIGN":
    #             preds = self.target_model.model(self.data.xs)
    #         else:
    #             preds = self.target_model.model(self.data.x,self.data.edge_index)
    #             preds_edge = preds
    #         # preds = self.target_model.model(self.data.x, self.data.edge_index)
    #         if self.args['dataset_name'] == 'ppi':
    #             preds = torch.sigmoid(preds).ge(0.5)
    #             preds = preds.type_as(self.data.y)
    #         else:           
    #             preds = torch.argmax(preds, axis=1).type_as(self.data.y)
    #             # neg_edge_index_pred = negative_sampling(
    #             #     edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,
    #             #     num_neg_samples=self.data.edge_index.size(1)
    #             # )
    #             # edge_preds = self.decode(z=preds_edge, pos_edge_index=self.data.edge_index,neg_edge_index=neg_edge_index_pred)
            
    #     if self.args["base_model"] == "SIGN":
    #         self.data.x = self.data.x_unlearn
    #         self.data.edge_index = self.data.edge_index_unlearn
    #         self.data = SIGN(self.args["GNN_layer"])(self.data)
    #         self.data.xs = [self.data.x] + [self.data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
    #         self.data.xs = torch.stack(self.data.xs).to('cuda')
    #         self.data.xs = self.data.xs.transpose(0,1)

    #     start_time = time.time()
    #     for epoch in tqdm(range(self.args["unlearning_epochs"]),desc="Unlearning"):
    #         self.target_model.model.train()
    #         operator.train()
    #         optimizer.zero_grad()
    #         if self.args["base_model"] == "SIGN":
    #             out_ori =  self.target_model.model(self.data.xs)
    #         else:
    #             out_ori = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
    #         out = operator(out_ori)
    #         # if self.args["downstream_task"]=="node":
    #         if self.args['dataset_name'] == 'ppi':
    #             loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.binary_cross_entropy_with_logits(out[self.temp_node], preds[self.temp_node])
    #             loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.binary_cross_entropy_with_logits(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
    #         else:
    #             loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.cross_entropy(out[self.temp_node], preds[self.temp_node])
    #             loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.cross_entropy(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
    #         # elif self.args["downstream_task"]=="edge":
    #         #     neg_edge_index = negative_sampling(
    #         #         edge_index=self.data.edge_index_unlearn,num_nodes=self.data.num_nodes,
    #         #         num_neg_samples=self.data.edge_index_unlearn.size(1)
    #         #     )
    #         #     mask = np.isin(self.data.edge_index_unlearn.cpu().numpy(), self.data.edge_index.cpu().numpy()).astype(np.uint8)
    #         #     neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)
    #         #     pos_edge_label = torch.ones(neg_edge_index.size(1),dtype=torch.float32)
    #         #     edge_labels = torch.cat((pos_edge_label,neg_edge_label),dim=-1)
    #         #     edge_logits = self.decode(z=out, pos_edge_index=self.data.edge_index_unlearn,neg_edge_index=neg_edge_index)
                
    #         #     if self.args['dataset_name'] == 'ppi':
    #         #         loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.binary_cross_entropy_with_logits(edge_logits, edge_preds[mask])
    #         #         loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.binary_cross_entropy_with_logits(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
    #         loss = self.args['kappa'] * loss_u + loss_r

    #         loss.backward()
    #         optimizer.step()

    #     unlearn_time = time.time() - start_time
    #     self.target_model.model.eval()
    #     if self.args["base_model"] == "SIGN":
    #         test_out =  self.target_model.model(self.data.xs)
    #     else:
    #         test_out = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
    #     if self.args['dataset_name'] == 'ppi':
    #         out = torch.sigmoid(test_out)
    #     else:
    #         out = self.correct_and_smooth(F.softmax(test_out, dim=-1), preds)
    #     if self.args["downstream_task"]=="node":
    #         y_hat = out.cpu().detach().numpy()
    #         y = self.data.y.cpu()
    #         if self.args['dataset_name'] == 'ppi':
    #             test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
    #         else:
    #             test_f1 = calc_f1(y, y_hat, self.data.test_mask)
    #     elif self.args["downstream_task"]=="edge":
    #         neg_edge_index = negative_sampling(
    #             edge_index=self.data.test_edge_index,num_nodes=self.data.num_nodes,
    #             num_neg_samples=self.data.test_edge_index.size(1)
    #         )
    #         edge_pred_logits = self.target_model.decode(z=out, pos_edge_index=self.data.test_edge_index,neg_edge_index=neg_edge_index)
    #         edge_pred_logits = torch.sigmoid(edge_pred_logits)
    #         edge_pred_logits = edge_pred_logits.cpu()
    #         edge_pred = torch.where(edge_pred_logits > 0.5, torch.tensor(1), torch.tensor(0))
            
    #         # edge_pred = torch.argmax(edge_pred_logits)
    #         pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
    #         neg_edge_labels = torch.zeros(neg_edge_index.size(1),dtype=torch.float32)
    #         edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
    #         test_f1 = roc_auc_score(edge_labels.cpu(), edge_pred.cpu())

    #     return unlearn_time, test_f1


    # def update_edge_index_unlearn(self, delete_nodes, remove_edges=None):
    #     """
    #     Updates the edge index of a graph by removing specified nodes or edges.
    #     This function is used to update the edge index of a graph by unlearning, 
    #     i.e., removing certain nodes or edges from the graph. It handles two 
    #     unlearning tasks: 'edge' and 'node'. For the 'edge' task, it removes 
    #     specified edges. For the 'node' task, it removes all edges connected 
    #     to the specified nodes.
    #     """
    #     edge_index = self.data.edge_index.cpu().numpy()

    #     unique_indices = np.where(edge_index[0] < edge_index[1])[0]
    #     unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
    #     if self.args["unlearn_task"] == 'edge':
    #         # ----- changed code becuase input are edges not the indices -----
    #         # remain_indices = np.setdiff1d(unique_indices, delete_edge_index)

    #         # normalize all edges (min, max) for undirected comparison
    #         edge_index_np = edge_index.T  # shape [num_edges, 2]
    #         edge_index_norm = np.sort(edge_index_np, axis=1)
    #         remove_edges = np.sort(np.array(remove_edges), axis=1)
    #         remove_set = set(map(tuple, remove_edges))

    #         # keep only edges not in remove_set
    #         mask = np.array([tuple(e) not in remove_set for e in edge_index_norm])
    #         edge_index = edge_index_np[mask].T
    #         # breakpoint()
    #         return torch.from_numpy(edge_index).long()
    #     else:
    #         unique_edge_index = edge_index[:, unique_indices]
    #         delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
    #                                             np.isin(unique_edge_index[1], delete_nodes))
    #         remain_indices = np.logical_not(delete_edge_indices)
    #         remain_indices = np.where(remain_indices == True)

    #     remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
    #     unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[
    #         0, unique_indices_not]
    #     # sort_indices = np.argsort(unique_encode_not)
    #     # remain_indices_not = unique_indices_not[
    #     #     sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
    #     # remain_indices = np.union1d(remain_indices, remain_indices_not)
        
    #     sort_indices = np.argsort(unique_encode_not)
    #     # print(unique_encode_not,len(remain_encode[0]))
    #     temp_index = np.searchsorted(unique_encode_not, remain_encode[0], sorter=sort_indices)-1
    #     temp_indices = sort_indices[temp_index]
    #     remain_indices_not = unique_indices_not[temp_indices]
    #     remain_indices = np.union1d(remain_indices, remain_indices_not)

    #     return torch.from_numpy(edge_index[:, remain_indices])

    #----------- new version of above update edge index unlearn function
    def update_edge_index_unlearn(self, delete_nodes, remove_edges=None):
        """
        Efficiently updates the edge_index by removing nodes or edges.
        Memory-efficient version for large graphs.
        """
        edge_index = self.data.edge_index  # [2, num_edges]
        device = edge_index.device

        if self.args["unlearn_task"] == "edge":
            # remove specific edges (CPU/GPU safe)
            edge_index_norm = torch.sort(edge_index, dim=0).values  # shape [2, num_edges]
            remove_edges = torch.tensor(remove_edges, dtype=edge_index.dtype, device=device).T
            remove_edges_norm = torch.sort(remove_edges, dim=0).values

            # create a mask using a hash trick (no big boolean tensor)
            edge_hash = edge_index_norm[0] * edge_index.shape[1] + edge_index_norm[1]
            remove_hash = remove_edges_norm[0] * edge_index.shape[1] + remove_edges_norm[1]
            mask = ~torch.isin(edge_hash, remove_hash)
            return edge_index[:, mask]

        else:
            # remove edges connected to delete_nodes using torch.isin
            delete_nodes = torch.tensor(delete_nodes, dtype=edge_index.dtype, device=device)
            mask0 = ~torch.isin(edge_index[0], delete_nodes)
            mask1 = ~torch.isin(edge_index[1], delete_nodes)
            mask = mask0 & mask1
            return edge_index[:, mask]


    
    def reverse_features(self, features):
        """
        Reverse the features of specified nodes.
        This function takes a tensor of features and reverses the features of the nodes
        specified in the `self.temp_node` list. For each node in `self.temp_node`, the 
        feature value is subtracted from 1, effectively reversing it.
        """
        reverse_features = features.clone()
        for idx in self.temp_node:
            reverse_features[idx] = 1 - reverse_features[idx]

        return reverse_features


    
    # def mia_attack(self,mem_labels_o,non_labels_o,mem_labels,non_labels):
    #     mia_test_y = torch.cat((torch.ones(self.args["num_unlearned_nodes"]), torch.zeros(self.args["num_unlearned_nodes"])))
    #     posterior1 = torch.cat((mem_labels_o, non_labels_o), 0).cpu().detach()
    #     posterior2 = torch.cat((mem_labels, non_labels), 0).cpu().detach()
    #     posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
    #     # self.logger.info("posterior:{}".format(posterior))
    #     auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
    #     self.average_auc[self.run] = auc
    #     # self.logger.info("auc:{}".format(auc))
    #     # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
    #     return auc
    
    def mia_attack(self):
        if self.unlearing_nodes.shape[0]>len(self.data.test_indices):
            self.mia_num = len(self.data.test_indices)   # changed: use full test set size
            self.original_softlabels = F.softmax(self.target_model.model(
                self.data.x,self.data.edge_index),dim=1).clone().detach().float()
            original_softlabels_non = self.original_softlabels[self.data.test_indices]
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.x,self.data.edge_index)[self.data.test_indices],dim=1)
            original_softlabels_member_full = self.original_softlabels[self.unlearing_nodes]
            unlearning_softlabels_member_full = F.softmax(self.target_model.model(
                self.data.x,self.data.edge_index)[self.unlearing_nodes],dim=1)
            aucs = []
            for i in range(0, len(self.unlearing_nodes), self.mia_num):
                if i + self.mia_num > len(self.unlearing_nodes): break
                original_softlabels_member = original_softlabels_member_full[i:i+self.mia_num]
                unlearning_softlabels_member = unlearning_softlabels_member_full[i:i+self.mia_num]
                mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
                posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
                posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[j]-posterior2[j]) for j in range(len(posterior1))])
                auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                aucs.append(auc)
            self.average_auc[self.run] = np.mean(aucs)
        else:
            self.mia_num = self.unlearing_nodes.shape[0]
            self.original_softlabels = F.softmax(self.target_model.model(
                self.data.x,self.data.edge_index),dim=1).clone().detach().float()
            original_softlabels_member = self.original_softlabels[self.unlearing_nodes]
            original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x,self.data.edge_index)[self.unlearing_nodes],dim=1)
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.x,self.data.edge_index)[self.data.test_indices[:self.mia_num]],dim=1)

            mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
            posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
            posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
            posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
            # self.logger.info("posterior:{}".format(posterior))
            auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            self.average_auc[self.run] = auc
        return auc

    def gold_standard_unlearning(self):
        """
        Gold standard unlearning by retraining from scratch on remaining data.
        
        Returns:
            tuple: (unlearning time, test F1 score)
        """
        # Create a new model instance (same architecture)
        model_class = type(self.model_zoo.model)
        new_model = model_class(self.args,self.num_feats,self._data.num_classes).to(self.device)
        
        # breakpoint()
        # Create a new trainer for gold standard
        trainer = BaseTrainer(self.args, self.logger, new_model, self.gold_data)
        
        # Train from scratch on remaining data
        start_time = time.time()
        trainer.train_node(save=False)
        unlearn_time = time.time() - start_time
        
        # Evaluate
        test_f1, acc = trainer.evaluate()
        if self.args["attack"]:
            self.mia_num = self.unlearing_nodes.shape[0]
            if self.unlearing_nodes.shape[0]>len(self.data.test_indices):
                self.mia_num = len(self.data.test_indices)   # changed: use full test set size
                self.original_softlabels = F.softmax(self.target_model.model(
                    self.data.x,self.data.edge_index),dim=1).clone().detach().float()
                original_softlabels_non = self.original_softlabels[self.data.test_indices]
                unlearning_softlabels_non = F.softmax(trainer.model(
                    self.data.x,self.data.edge_index)[self.data.test_indices],dim=1)

                original_softlabels_member_full = self.original_softlabels[self.unlearing_nodes]
                unlearning_softlabels_member_full = F.softmax(trainer.model(
                    self.data.x,self.data.edge_index)[self.unlearing_nodes],dim=1)

                aucs = []
                for i in range(0, len(self.unlearing_nodes), self.mia_num):
                    if i + self.mia_num > len(self.unlearing_nodes): break
                    original_softlabels_member = original_softlabels_member_full[i:i+self.mia_num]
                    unlearning_softlabels_member = unlearning_softlabels_member_full[i:i+self.mia_num]

                    mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
                    posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
                    posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
                    posterior = np.array([np.linalg.norm(posterior1[j]-posterior2[j]) for j in range(len(posterior1))])
                    auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                    aucs.append(auc)

                self.average_gold_auc[self.run] = np.mean(aucs)
            else:
                self.original_softlabels = F.softmax(self.target_model.model(
                    self.data.x,self.data.edge_index),dim=1).clone().detach().float()
                original_softlabels_member = self.original_softlabels[self.unlearing_nodes]
                unlearning_softlabels_member = F.softmax(trainer.model(self.data.x,self.data.edge_index)[self.unlearing_nodes],dim=1)

                original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
                unlearning_softlabels_non = F.softmax(trainer.model(
                    self.data.x,self.data.edge_index)[self.data.test_indices[:self.mia_num]],dim=1)

                mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
                posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
                posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
                # self.logger.info("posterior:{}".format(posterior))
                auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                self.average_gold_auc[self.run] = auc
        # if self.args["attack"]:
        #     self.mia_num = len(self.data.test_indices)   # use full test set size
        #     self.original_softlabels = F.softmax(self.target_model.model(
        #         self.data.x,self.data.edge_index),dim=1).clone().detach().float()
        #     original_softlabels_non = self.original_softlabels[self.data.test_indices]
        #     unlearning_softlabels_non = F.softmax(trainer.model(
        #         self.data.x,self.data.edge_index)[self.data.test_indices],dim=1)

        #     original_softlabels_member_full = self.original_softlabels[self.unlearing_nodes]
        #     unlearning_softlabels_member_full = F.softmax(trainer.model(
        #         self.data.x,self.data.edge_index)[self.unlearing_nodes],dim=1)

        #     aucs = []
        #     for i in range(0, len(self.unlearing_nodes), self.mia_num):
        #         original_softlabels_member = original_softlabels_member_full[i:i+self.mia_num]
        #         unlearning_softlabels_member = unlearning_softlabels_member_full[i:i+self.mia_num]

        #         mia_test_y = torch.cat((torch.ones(original_softlabels_member.shape[0]), 
        #                                 torch.zeros(self.mia_num)))
        #         posterior1 = torch.cat((original_softlabels_member, original_softlabels_non[:mia_test_y.shape[0]-original_softlabels_member.shape[0] + self.mia_num]), 0).cpu().detach()
        #         posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non[:mia_test_y.shape[0]-original_softlabels_member.shape[0] + self.mia_num]), 0).cpu().detach()

        #         posterior = np.array([np.linalg.norm(posterior1[j]-posterior2[j]) for j in range(len(posterior1))])
        #         auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        #         aucs.append(auc)

        #     self.average_gold_auc[self.run] = np.mean(aucs)


        
        # Save gold standard model
        # unlearn_ratio = len(self.temp_node) / self.data.num_nodes
        unlearn_ratio = self.args['unlearn_ratio']

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(self.run)

        base_model_str = ""
        if self.args['base_model']!="GCN":  
            base_model_str = "_" + self.args['base_model']
            
        save_dir = os.path.join(
            "unlearned_models", "GOLD",
            self.args['dataset_name'],
            self.args["unlearn_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}"
        )
        os.makedirs(save_dir, exist_ok=True)

        model_name = f"GOLD_{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}{base_model_str}.pt"
        torch.save(new_model.state_dict(), os.path.join(save_dir, model_name))
        print(f" Saving this retrained model at: {os.path.abspath(os.path.join(save_dir, model_name))}")
        
        self.logger.info(f"Gold Standard Unlearning time: {unlearn_time:.4f}, Test F1: {test_f1:.4f}, Test Acc: {acc:.4f}")
        return unlearn_time, test_f1

    def _create_gold_standard_data(self):
        """Creates dataset with unlearned nodes/edges removed"""
        gold_data = copy.deepcopy(self.data).cpu()
        
        if self.args["unlearn_task"] == 'node':
            # Create mask for nodes to keep
            keep_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
            keep_mask[self.temp_node] = False
            
            # Update node features and labels
            gold_data.x = self.data.x[keep_mask]
            gold_data.y = self.data.y[keep_mask]
            
            # Update masks
            for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                if hasattr(gold_data, mask_name):
                    mask = getattr(gold_data, mask_name)
                    setattr(gold_data, mask_name, mask[keep_mask])
            
            # Update edges
            edge_index = self.data.edge_index.cpu().numpy()
            edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
            gold_data.edge_index = self.data.edge_index[:, edge_mask]
            
            # Reindex nodes
            node_mapping = torch.full((self.data.num_nodes,), -1, dtype=torch.long)
            new_idx = torch.arange(keep_mask.sum().item())
            node_mapping[keep_mask] = new_idx
            gold_data.edge_index = node_mapping[gold_data.edge_index.cpu()]

            
            gold_data.num_nodes = keep_mask.sum().item()
            
        elif self.args["unlearn_task"] == 'edge':
            # Use the edge index after unlearning
            gold_data.edge_index = self.data.edge_index_unlearn.clone()
            
            # Keep node features and labels intact
            gold_data.x = self.data.x.clone()
            gold_data.y = self.data.y.clone()
            
            # Masks also remain the same since no nodes are dropped
            for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                if hasattr(self.data, mask_name):
                    setattr(gold_data, mask_name, getattr(self.data, mask_name).clone())
            
            # Node count stays the same
            gold_data.num_nodes = self.data.num_nodes
            
        return gold_data

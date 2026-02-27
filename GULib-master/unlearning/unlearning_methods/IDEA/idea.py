import torch
import numpy as np
import os
import time
import copy
from task import get_trainer
from utils.dataset_utils import *
import torch.nn.functional as F
from config import BLUE_COLOR,RESET_COLOR
from torch.autograd import grad
# from attack.MIA_attack import train_attack_model,train_shadow_model,generate_shadow_model_output,evaluate_attack_model,GCNShadowModel,AttackModel
from pipeline.IF_based_pipeline import IF_based_pipeline
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
class idea(IF_based_pipeline):
    """
    IDEA class implements a IF-based pipeline for performing unlearning tasks on GNNs, enabling efficient removal of specific data points, edges, or features from
    trained graph-based models without the need for retraining from scratch.

    Class Attributes:
        args (dict): Configuration parameters for the GIF pipeline, including settings for the number of runs, unlearning ratios, and method choices.

        logger (Logger): Logger instance for logging information, debugging, and tracking the pipeline's progress and performance metrics.

        model_zoo (ModelZoo): Collection of models and related data resources used within the pipeline.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.args["unlearn_trainer"] = 'IDEATrainer'
        
        self.num_classes = self.data.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unlearning_number = args["num_unlearned_nodes"]
        num_runs = self.args["num_runs"]

        self.attack_preparations = {}

        self.deleted_nodes = np.array([])     
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])

        self.certification_alpha1 = 0.0
        self.certification_alpha2 = 0.0
        self.samples_to_be_unlearned = 0.0

        self.originally_trained_model_params = None
        

    # def run_exp(self):

    #     run_f1 = np.empty((0))
    #     run_f1_unlearning = np.empty((0))
    #     run_f1_retrain = np.empty((0))
    #     unlearning_times = np.empty((0))
    #     training_times = np.empty((0))
    #     my_bounds = np.empty((0))
    #     certified_edge_bounds = np.empty((0))
    #     certified_edge_worst_bounds = np.empty((0))
    #     actual_diffs = np.empty((0))
    #     attack_auc_scores = np.empty((0))

    #     # self.train_test_split()
        
    #     self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
    #     for self.run in range(self.args['num_runs']):
    #         self.logger.info("Run %f" % self.run)
    #         self.unlearning_request()
    #         run_training_time, result_tuple = self.train_unlearning_model(self.run)
    #         self.originally_trained_model_params = [p.clone() for p in self.target_model.model.parameters() if p.requires_grad]
    #         f1_score = self.target_model.evaluate(self.run)
 
    #         run_f1 = np.append(run_f1, f1_score)
    #         training_times = np.append(training_times, run_training_time)

    #         # unlearn and compute certification stats
    #         unlearning_time, f1_score_unlearning, params_change = self.approxi(result_tuple)
    #         my_bound, certified_edge_bound, certified_edge_worst_bound, actual_diff, f1_score_retrain = self.alpha_computation(params_change, result_tuple)
    #         self.attack_preparations["unlearned_feature_pre"] = self.target_model.attack_preparations["unlearned_feature_pre"]
    #         self.attack_preparations["predicted_prob"] = self.target_model.attack_preparations["predicted_prob"]
                       

    #         unlearning_times = np.append(unlearning_times, unlearning_time)
    #         run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)

    #         run_f1_retrain = np.append(run_f1_retrain, f1_score_retrain)

    #         my_bounds = np.append(my_bounds, my_bound)
    #         certified_edge_bounds = np.append(certified_edge_bounds, certified_edge_bound)
    #         certified_edge_worst_bounds = np.append(certified_edge_worst_bounds, certified_edge_worst_bound)
    #         actual_diffs = np.append(actual_diffs, actual_diff)

    #         self.attack_preparations["train_indices"] = self.data.train_mask
    #         self.attack_preparations["test_indices"] = self.data.test_mask  
    #         exp_marker = [self.args['dataset_name'], self.args['unlearn_task'], str(self.args['unlearn_ratio']), self.args['base_model'], str(self.args['unlearn_feature_partial_ratio'])]
    #         exp_marker_string = "_".join(exp_marker)

    #         current_dir = os.getcwd()
    #         attack_materials_dir = os.path.join(current_dir, 'attack_materials')
    #         if not os.path.exists(attack_materials_dir):
    #             os.makedirs(attack_materials_dir)
    #         if self.args["downstream_task"]=="node":
    #             attack_auc_score = self.mia_attack()
    #             attack_auc_scores = np.append(attack_auc_scores,attack_auc_score)
    #         # torch.save(self.attack_preparations, 'attack_materials/' + 'IDEA_' + exp_marker_string + '.pth')

        
    #     self.logger.info("Completed. \n")
    #     f1_score_avg = np.average(run_f1)
    #     f1_score_std = np.std(run_f1)

    #     training_time_avg = np.average(training_times)
    #     training_time_std = np.std(training_times)

    #     f1_score_unlearning_avg = np.average(run_f1_unlearning)
    #     f1_score_unlearning_std = np.std(run_f1_unlearning)

    #     f1_retrain_avg = np.average(run_f1_retrain)
    #     f1_retrain_std = np.std(run_f1_retrain)

    #     unlearning_time_avg = np.average(unlearning_times)
    #     unlearning_time_std = np.std(unlearning_times)

    #     my_bounds_avg = np.average(my_bounds)
    #     my_bounds_std = np.std(my_bounds)

    #     certified_edge_bounds_avg = np.average(certified_edge_bounds)
    #     certified_edge_bounds_std = np.std(certified_edge_bounds)

    #     certified_edge_worst_bounds_avg = np.average(certified_edge_worst_bounds)
    #     certified_edge_worst_bounds_std = np.std(certified_edge_worst_bounds)

    #     actual_diffs_avg = np.average(actual_diffs)
    #     actual_diffs_std = np.std(actual_diffs)

    #     attack_auc_score_avg = np.average(attack_auc_scores)
    #     attack_auc_score_std = np.std(attack_auc_scores)
            

    #     self.logger.info(
    #     "{}Performance Metrics:\n"
    #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average Training Time: {:.4f} ± {:.4f} seconds\n"
    #     " - Average Retain F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average Unlearn F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average Unlearning Time: {:.4f} ± {:.4f} seconds\n"
    #     " - My Bound: {:.4f} ± {:.4f}\n"
    #     " - Certified Edge Bound: {:.4f} ± {:.4f}\n"
    #     " - Certified Edge Worst Bounds: {:.4f} ± {:.4f}\n"
    #     " - Actual Diffs: {:.4f} ± {:.4f}\n"
    #     " - Attack AUC Score: {:.4f} ± {:.4f}\n{}".format(
    #         BLUE_COLOR,
    #         f1_score_avg, f1_score_std,
    #         training_time_avg, training_time_std,
    #         f1_retrain_avg, f1_retrain_std,
    #         f1_score_unlearning_avg, f1_score_unlearning_std,
    #         unlearning_time_avg, unlearning_time_std,
    #         my_bounds_avg, my_bounds_std,
    #         certified_edge_bounds_avg, certified_edge_bounds_std,
    #         certified_edge_worst_bounds_avg, certified_edge_worst_bounds_std,
    #         actual_diffs_avg, actual_diffs_std,
    #         attack_auc_score_avg,attack_auc_score_std,
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
            # print(self.data.train_indices.size, self.data.test_indices.size)
        else:
            self.data = load_train_test_split(self.logger)
            # self.data.train_indices, self.data.test_indices

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.test_indices))

    def determine_target_model(self):
        """
        Determines and initializes the target model based on the provided configuration.
        Logs the target model's name, sets the unlearning trainer to 'IDEATrainer',
        and retrieves the trainer instance using the get_trainer function.
        """
        self.logger.info('target model: %s' % (self.args['base_model'],))
        self.args["unlearn_trainer"] = "IDEATrainer"
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)

    def unlearning_request(self):
        """
        Handles unlearning requests based on the specified task.
        This method processes the data for unlearning by selecting nodes, edges, or features to remove or modify.
        It updates the model's data accordingly and logs relevant information. Depending on the downstream task,
        it also identifies k-hop neighborhoods related to the unlearning operation.
        """
        if self.args["downstream_task"]=="graph":
            self.data = graph_cls_process(self.data,train_ratio=0.8,val_ratio=0,test_ratio=0.2)
            self.target_model.data = self.data
        self.logger.debug("Train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))

        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.cpu().numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args["unlearn_task"] == 'node':
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            if os.path.exists(path_un):
                unique_nodes = np.loadtxt(path_un, dtype=int)
            else:
                unique_nodes = np.random.choice(len(self.data.train_indices),
                                                int(len(self.data.train_indices) * self.args['unlearn_ratio']),
                                                replace=False)
            unique_edges = edge_index[:,np.where(np.isin(edge_index,unique_nodes))[1]]
            self.unlearning_nodes = unique_nodes
            
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)
            self.samples_to_be_unlearned = float(int(len(self.data.train_indices) * self.args['unlearn_ratio']))

            self.attack_preparations["removed_nodes"] = unique_nodes

        if self.args["unlearn_task"] == 'edge':
            path_un_edge = unlearning_edge_path + "_" + str(self.run) + "_edges_" + str(self.args["num_unlearned_edges"]) + ".txt"
            if os.path.exists(path_un_edge):
                self.unlearning_edges = np.loadtxt(path_un_edge, dtype=int).T
            else:
                remove_indices = np.random.choice(
                    unique_indices,
                    int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                    replace=False)
                self.unlearning_edges = edge_index[:, remove_indices]
            # setting 1
            # remove_indices = np.random.choice(
            #     unique_indices,
            #     int(unique_indices.shape[0] * self.args['unlearn_ratio']),
            #     replace=False)
            # remove_edges = edge_index[:, remove_indices]
            # unique_edges = remove_edges
            unique_nodes = np.unique(self.unlearning_edges)

            self.attack_preparations["removed_edges"] = self.unlearning_edges
            
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, self.unlearning_edges)
            self.samples_to_be_unlearned = 0.0


            # # setting 2
            # # sample non-existed edges
            # remove_edges = negative_sampling(self.data.edge_index, num_neg_samples=int(unique_indices.shape[0] * self.args['unlearn_ratio']))

            # # change the original graph topology to add these edges
            # self.data.edge_index = torch.cat((self.data.edge_index, remove_edges), 1)
            # self.data.edge_index_unlearn = self.data.edge_index_unlearn  # no change
            # unique_nodes = np.unique(remove_edges.numpy())

            # self.attack_preparations["removed_edges"] = remove_edges
            # self.samples_to_be_unlearned = 0.0

        if self.args["unlearn_task"] == 'feature':
            # unique_nodes = np.random.choice(len(self.data.train_indices),
            #                                 int(len(self.data.train_indices) * self.args['unlearn_ratio']),
            #                                 replace=False)
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            unique_nodes = np.loadtxt(path_un, dtype=int)
            self.unlearing_nodes = unique_nodes
            self.unlearning_edges = None
            self.data.x_unlearn[unique_nodes] = 0.
            self.samples_to_be_unlearned = 0.0
            self.attack_preparations["unlearned_feature_node_idx"] = unique_nodes

        if self.args["unlearn_task"] == 'partial_feature':
            unique_nodes = np.random.choice(len(self.data.train_indices),
                                            int(len(self.data.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)

            unlearn_feature_dims = np.random.randint(0, self.data.x.shape[1], size=int(self.data.x.shape[1] * self.args['unlearn_feature_partial_ratio']))
            self.data.x_unlearn[unique_nodes[:, None], unlearn_feature_dims] = 0.0
            self.samples_to_be_unlearned = 0.0

            self.attack_preparations["unlearned_feature_node_idx"] = unique_nodes
            self.attack_preparations["unlearned_feature_dim_idx"] = unlearn_feature_dims

        if self.args["downstream_task"] in["node","graph"]:
            self.find_k_hops(unique_nodes)
        elif self.args["downstream_task"]=="edge":
            self.find_k_hops_edge(unique_nodes,self.unlearning_edges)

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        """
        Updates the edge index by removing specified edges or edges connected to specified nodes based on the unlearning task.
        Depending on the 'unlearn_task' parameter, this function either deletes specific edges provided in `delete_edge_index` or removes all edges connected to nodes listed in `delete_nodes`. The updated edge index is returned as a PyTorch tensor.
        """
        edge_index = self.data.edge_index.cpu().numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args["unlearn_task"] == 'edge':
            # remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
            edge_set = set(map(tuple, edge_index.T))
            delete_edge_set = set(map(tuple, delete_edge_index.T))
            remaining_edges = edge_set - delete_edge_set
            remain_indices = np.array([i for i, edge in enumerate(edge_index.T) if tuple(edge) in remaining_edges])
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        if len(remain_indices) == 0:
            return torch.empty((2,0), dtype=torch.long)
        remain_encode = (edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]).squeeze()
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        # print(unique_encode_not)
        # print(remain_encode,remain_encode.shape)
        # print(sort_indices)
        index_temp = np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)
        sort_indices = sort_indices[index_temp-1]
        remain_indices_not = unique_indices_not[sort_indices]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])
    
    def find_k_hops(self, unique_nodes):
        """
        Finds and sets the influenced nodes within a specified number of hops from the given unique nodes based on the unlearning task.
        """
        edge_index = self.data.edge_index.cpu().numpy()

        ## finding influenced neighbors
        hops = 2
        if self.args["unlearn_task"] == 'node':
            hops = 3
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        if self.args["unlearn_task"] == 'feature':
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        if self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        if self.args["unlearn_task"] == 'edge':
            self.influence_nodes = influenced_nodes
    
    def find_k_hops_edge(self, unique_nodes,unique_edges):
        """
        Finds and categorizes edges within a specified number of hops from given unique nodes and edges.
        This function identifies edges influenced by the provided unique nodes and edges by exploring their neighbors up to a defined number of hops. The number of hops is determined based on the type of unlearning task. It categorizes the influenced nodes and edges, and determines which nodes and edges should be deleted or influenced according to the task requirements.
        """
        edge_index = self.data.edge_index.cpu().numpy()

        hops = 2
        if self.args["unlearn_task"] == 'node':
            hops = 3
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        influenced_edges = edge_index[:,np.where(np.isin(edge_index,influenced_nodes))[1]]
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        neighbor_edges = influenced_edges
        neighbor_edges = neighbor_edges[:, ~np.isin(neighbor_edges.T, unique_edges.T).all(axis=1)]
        if self.args["unlearn_task"] == 'feature' or 'partial_feature':
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
            self.influence_edges = neighbor_edges
        if self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.deleted_edges = unique_edges
            self.influence_nodes = neighbor_nodes
            self.influence_edges = influenced_edges
        if self.args["unlearn_task"] == 'edge':
            self.influence_nodes = influenced_nodes
            self.deleted_edges = unique_edges
            self.influence_edges = neighbor_edges

    
    def alpha_computation(self, params_change, result_tuple):

        # bound given by alpha 1 + alpha 2
        m = self.samples_to_be_unlearned
        t = self.influence_nodes.shape[0]

        self.certification_alpha1 = (m * self.args['l'] + (m ** 2 + self.args['l'] ** 2 + 4 * self.args['lambda'] * len(self.data.train_indices) * t * self.args['c']) ** 0.5) / (self.args['lambda'] * len(self.data.train_indices))
        params_change_flatten = [item.flatten() for item in params_change]
        self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2)
        self.logger.info("Certification related stats:  ")
        self.logger.info("certification_alpha1 (bound): %s" % self.certification_alpha1)
        self.logger.info("certification_alpha2 (l2 of modification): %s" % self.certification_alpha2)
        total_bound = self.certification_alpha1 + self.certification_alpha2
        self.logger.info("total bound given by alpha1 + alpha2: %s" % total_bound)

        # bound given by certified edge
        certified_edge_bound = self.certification_alpha2 ** 2 * self.args['M'] / self.args['l']
        self.logger.info("data-dependent bound given by certified edge: %s" % certified_edge_bound)

        # worset bound given by certified edge  
        certified_edge_worst_bound = self.args['M'] * (self.args['gamma_2'] ** 2) * (self.args['c1'] ** 2) * (t ** 2) / ((self.args['lambda_edge_unlearn'] ** 4) * len(self.data.train_indices))
        self.logger.info("worst bound given by certified edge: %s" % certified_edge_worst_bound)

        # recover the originally trained model
        idx = 0
        for p in self.target_model.model.parameters():
            p.data = self.originally_trained_model_params[idx].clone()
            idx = idx + 1

        # continue optimizing the model with data already updated
        self.target_model.train_model_continue((self.deleted_nodes, self.feature_nodes, self.influence_nodes))

        # actual difference
        original_params = [p.flatten() for p in self.params_esti]
        retraining_model_params = [p.flatten() for p in self.target_model.model.parameters() if p.requires_grad]
        actual_param_difference = torch.norm((torch.cat(original_params) - torch.cat(retraining_model_params)), 2).detach()
        self.logger.info("actual params difference: %s" % actual_param_difference)

        test_F1_retrain = self.target_model.evaluate_unlearn_F1([p for p in self.target_model.model.parameters() if p.requires_grad])
        self.logger.info("retrain f1 score: %s" % test_F1_retrain)

        return total_bound.cpu().numpy(), certified_edge_bound.cpu().numpy(), certified_edge_worst_bound, actual_param_difference.cpu().numpy(), test_F1_retrain

    def train_unlearning_model(self, run):
        self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        if self.args["downstream_task"]=="node":
            res = self.target_model.idea_train(
                (self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        elif self.args["downstream_task"]=="edge":
            res = self.target_model.idea_train(
                (self.deleted_edges,self.feature_nodes,self.influence_edges)
            )
        elif self.args["downstream_task"]=="graph":
            res = self.target_model.idea_train((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        train_time = time.time() - start_time
        self.logger.info("Model training time: %s" % (train_time))

        return train_time, res
    
    def approxi(self, res_tuple):
        """
        Approximates parameter changes for model unlearning using gradient information.
        This function processes a tuple of gradients based on the specified unlearning method ('GIF' or 'IF')
        and iteratively updates an estimated parameter change. It adjusts the model's parameters accordingly
        and evaluates the unlearned model's performance by calculating the test F1 score.
        """
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''

        start_time = time.time()
        iteration, damp, scale = self.args['iteration'], self.args['damp'], self.args['scale']
        if self.args["dataset_name"] in ["Photo","Computers","Physics","Amazon-ratings","Questions"]:
            iteration =int(iteration/10)
        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):

            model_params  = [p for p in self.target_model.model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        self.params_esti = params_esti

        end_time = time.time()

        # add Gaussian Noise
        gaussian_noise = [(torch.randn(item.size()) * self.args['gaussian_std'] + self.args['gaussian_mean']).to(item.device) for item in params_esti]
        params_esti = [item1 + item2 for item1, item2 in zip(gaussian_noise, params_esti)]
    
        test_F1 = self.target_model.evaluate_unlearn_F1(params_esti,edge_weight_unlearn=self.edge_weight_unlearn)

        import os

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(self.run)

        base_model_str = ""
        if self.args['base_model']!="GCN":  
            base_model_str = "_" + self.args['base_model']

        # Inside the megu_unlearning method, before the return statement
        unlearn_ratio = (self.args["num_unlearned_nodes"]) / self.data.num_nodes
        save_dir = os.path.join(
            "unlearned_models", "IDEA",
            self.args['dataset_name'],
            self.args["unlearn_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}"
        )
        os.makedirs(save_dir, exist_ok=True)


        model_name = f"IDEA_{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}{base_model_str}.pt"
        torch.save(params_esti, os.path.join(save_dir, model_name))
        torch.save(self.data.x_unlearn, os.path.join(save_dir, f"x_unlearn{run_str}{base_model_str}.pt"))
        torch.save(self.data.edge_index_unlearn, os.path.join(save_dir, f"edge_index_unlearn{run_str}{base_model_str}.pt"))

        # return end_time - start_time, test_F1, params_change
        return end_time - start_time, test_F1
    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads
    
    # def mia_attack(self):
    #     shadow_model = GCNShadowModel(self.data.num_features,64 ,self.data.num_classes)
    #     train_shadow_model(shadow_model,self.data)
    #     soft_labels_train,soft_labels_test = generate_shadow_model_output(shadow_model,self.data)

    #     attack_mdoel = train_attack_model(soft_labels_train,self.data.num_classes)
    #     attack_auc_score = evaluate_attack_model(attack_mdoel,soft_labels_test)

    #     return attack_auc_score
    def mia_attack(self):
        
        #-------------------------------------------------------------------------------
        if self.target_model_name in ['GCN','SGC',"S2GC"]:
            out = self.target_model.model.forward_once(self.data, self.edge_weight)
        else:
            out = F.softmax(self.target_model.model.reason_once(self.data),dim=-1)

        self.original_softlabels = out
        if len(self.data.test_indices)<len(self.unlearning_nodes):
            # use full test set size
            self.mia_num = len(self.data.test_indices)

            original_softlabels_non = self.original_softlabels[self.data.test_indices]
            original_softlabels_member_full = self.original_softlabels[self.unlearning_nodes]

            # get unlearning model outputs
            out = self.target_model.model.reason_once_unlearn(self.data)
            unlearning_softlabels_non = out[self.data.test_indices]
            unlearning_softlabels_member_full = out[self.unlearning_nodes]

            aucs = []
            # split unlearned set into test-sized chunks
            for i in range(0, len(self.unlearning_nodes), self.mia_num):
                if i + self.mia_num > len(self.unlearning_nodes):
                    break
                original_softlabels_member = original_softlabels_member_full[i:i+self.mia_num]
                unlearning_softlabels_member = unlearning_softlabels_member_full[i:i+self.mia_num]

                mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
                posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
                posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[j] - posterior2[j]) for j in range(len(posterior1))])

                auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                aucs.append(auc)

            self.average_auc[self.run] = np.mean(aucs)
        else:
            self.mia_num = self.unlearning_nodes.size
            original_softlabels_member = self.original_softlabels[self.unlearning_nodes]
            original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
            # if self.target_model_name in ['GCN','SGC',"S2GC"]:
            #     out = self.target_model.model.forward_once(self.data, self.target_model.edge_weight)

            # else:
            #     out = self.target_model.model.forward_once(self.data)
            
            out = self.target_model.model.reason_once(self.data)

            unlearning_softlabels_member = out[self.unlearning_nodes]
            unlearning_softlabels_non = out[self.data.test_indices[:self.mia_num]]

            mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
            posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
            posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
            posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
            # self.logger.info("posterior:{}".format(posterior))
            auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            # self.logger.info("auc:{}".format(auc))
            # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
            self.average_auc[self.run] = auc
        #-------------------------------------------- end-------------------------------------------------------
        pass
    
    # def mia_attack_edge(self):
    #     out = self.target_model.model.forward_once(self.data, self.edge_weight)

    #     self.mia_num = self.unlearning_edges.shape[1]
    #     neg_edge_index = negative_sampling(edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,num_neg_samples=self.mia_num)
    #     original_softlabels_member = self.target_model.decode(out,self.data.train_edge_index[:,:self.mia_num])
    #     original_softlabels_non = self.target_model.decode(out,neg_edge_index)
        
    #     out_unlearn = self.target_model.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)
    #     unlearning_softlabels_member = self.target_model.decode(out_unlearn,self.data.train_edge_index[:,:self.mia_num])
    #     unlearning_softlabels_non = self.target_model.decode(out_unlearn,neg_edge_index)
    #     mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
    #     posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
    #     posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
    #     posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])

    #     auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
    #     self.average_auc[self.run] = auc
        
    def train_original_model(self, run):
        """
        Trains the original target model on the dataset without performing any unlearning operations.
        Logs the training process, records the training time, and updates performance metrics.
        For graph-based downstream tasks, it prepares the training and testing datasets accordingly.
        If poisoning is enabled and the unlearning task involves edges, it also evaluates the poisoned F1 score.
        """
        self.logger.info('training target models, run %s' % run)
        self.target_model.data.y = self.data.y.squeeze().to(self.device)
        if self.args["downstream_task"]=="graph":
            temp_data = copy.deepcopy(self.model_zoo.data)
            train_dataset = [temp_data[i] for i in temp_data.train_indices]
            test_dataset = [temp_data[i] for i in temp_data.test_indices]
            self.train_graph = None
            temp_data = [train_dataset,test_dataset]
            self.target_model.data = temp_data
            self.data = train_dataset
        start_time = time.time()
        # self.target_model.model.reset_parameters()
        # self.target_model.model, self.target_model.data = self.target_model.model.to(self.device), self.target_model.data.to(self.device)
        
        # self.target_model._gen_train_loader()
        self.target_model.train(save = False)
        train_time = time.time() - start_time
        self.avg_training_time[self.run] = train_time
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.target_model.evaluate()
    def get_if_grad(self, run):
        """
        Retrieves the gradients of the target model based on the current unlearning task and downstream task.
        This method processes the necessary unlearning information and computes the corresponding gradients
        required for the unlearning process, supporting node, edge, and graph downstream tasks.
        """
        self.logger.info('training target models, run %s' % run)
        self.target_model.model = self.target_model.model.to(self.device)
        self.data = self.data.to(self.device)
        self._gen_train_loader()
        if self.args["downstream_task"]=="node":
            res = self.get_grad((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        elif self.args["downstream_task"]=="edge":
            res = self.get_grad_edge((self.deleted_edges,self.feature_nodes,self.influence_edges))
        elif self.args["downstream_task"]=="graph":
            res = self.get_grad_graph((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        return  res
    
    def get_grad(self,unlearn_info=None):
        """
        Computes the gradients of the loss functions with respect to the model parameters for the entire dataset,
        as well as for the subsets of data affected by the unlearning request. This is used to estimate how the
        model parameters should be updated to effectively unlearn the specified data without retraining from scratch.
        """
        grad_all, grad1, grad2 = None, None, None
        if self.args["base_model"] in ['GCN','SGC']:
            out1 = self.target_model.model.forward_once(self.data, self.edge_weight)
            out2 = self.target_model.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)
        else:
            out1 = F.softmax(self.target_model.model.reason_once(self.data),dim=-1)
            out2 = F.softmax(self.target_model.model.reason_once_unlearn(self.data),dim=-1)
        if self.args["unlearn_task"] == "edge":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[2]] = True
            mask2 = mask1
        if self.args["unlearn_task"] == "node":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[0]] = True
            mask1[unlearn_info[2]] = True
            mask2 = np.array([False] * out2.shape[0])
            mask2[unlearn_info[2]] = True
        if self.args["unlearn_task"] == "feature":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[1]] = True
            mask1[unlearn_info[2]] = True
            mask2 = mask1
            
        loss = F.cross_entropy(out1[self.data.train_mask], self.data.y[self.data.train_mask],reduction='sum')
        loss1 = F.cross_entropy(out1[mask1], self.data.y[mask1], reduction='sum')
        loss2 = F.cross_entropy(out2[mask2], self.data.y[mask2], reduction='sum')

        model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True,allow_unused=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True,allow_unused=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True,allow_unused=True)      
        return (grad_all, grad1, grad2)
    
    def get_grad_edge(self,unlearn_info=None):
        """
        Computes gradients for unlearning specific components in the target model.
        This method calculates the gradients of the loss with respect to the model parameters
        to facilitate the unlearning of certain elements such as edges, nodes, or features
        based on the specified unlearn_task. It performs forward passes using both the existing
        data and the modified data after unlearning, computes the corresponding losses, and then
        derives the gradients necessary for updating the model parameters accordingly.
        """
        grad_all, grad1, grad2 = None, None, None
        edge_index_r =None
        if self.args["base_model"] in ['GCN','SGC']:
            out1 = self.target_model.model.forward_once(self.data, self.edge_weight)
            out2 = self.target_model.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)
        else:
            out1 = F.softmax(self.target_model.model.reason_once(self.data),dim=-1)
            out2 = F.softmax(self.target_model.model.reason_once_unlearn(self.data),dim=-1)
        if self.args["unlearn_task"] == "edge":
            # print(unlearn_info[0].shape,unlearn_info[2].shape)
            edge_index_r = unlearn_info[2]
        elif self.args["unlearn_task"]=="node":
            edge_index_r = np.concatenate((unlearn_info[0], unlearn_info[2]),axis=1)
        elif self.args["unlearn_task"]=="feature":
            edge_index_r = unlearn_info[2]
        # print(edge_index_r.shape)
        loss = self.get_loss(out1,reduction="sum")
        loss1 = self.get_edge_loss(out1,edge_index=edge_index_r,reduction="sum")
        loss2 = self.get_edge_loss(out2,edge_index=unlearn_info[2],reduction="sum")


        model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
        return (grad_all, grad1, grad2)
    
    def get_loss(self, out, reduction="none"):
        neg_edge_index = negative_sampling(
                edge_index=self.data.train_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_index.size(1)
            )
        neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)
        pos_edge_label = torch.ones(self.data.train_edge_index.size(1),dtype=torch.float32)
        edge_logits = self.target_model.decode(z=out, pos_edge_index=self.data.train_edge_index,neg_edge_index=neg_edge_index)
        edge_labels = torch.cat((pos_edge_label,neg_edge_label),dim=-1)
        edge_labels = edge_labels.to(self.device)
        loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels, reduction=reduction)
        return loss

    def get_edge_loss(self,out,edge_index,reduction):
        """
        Computes the binary cross-entropy loss for predicted edges in a graph neural network.
        """
        out_decode = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=-1)

        edge_label = torch.ones(out_decode.shape,dtype=torch.float32,device=self.device)
        # print(out_decode.shape,edge_label.shape)
        loss = F.binary_cross_entropy_with_logits(out_decode,edge_label,reduction=reduction)
        return loss
    
    def get_grad_graph(self,unlearn_info=None):
        """
        Computes gradients for the target model based on the specified unlearning task.
        """
        grad_all, grad1, grad2 = None, None, None
        out1 = self.target_model.model.reason_once(self.data)
        out2 = self.target_model.model.reason_once_unlearn(self.data)
        if self.args["unlearn_task"] == "edge":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[2]] = True
            mask2 = mask1
        if self.args["unlearn_task"] == "node":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[0]] = True
            mask1[unlearn_info[2]] = True
            mask2 = np.array([False] * out2.shape[0])
            mask2[unlearn_info[2]] = True
        if self.args["unlearn_task"] == "feature":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[1]] = True
            mask1[unlearn_info[2]] = True
            mask2 = mask1
        
        loss = self.get_graph_loss(out1,self.data.train_mask)
        loss1 = self.get_graph_loss(out1,mask1)
        loss2 = self.get_graph_loss(out2,mask2)
        
        model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
        return (grad_all, grad1, grad2)
    def get_graph_loss(self,out,mask):
        """
        Compute the total cross-entropy loss for graph-level predictions.
        This function aggregates the loss over all training graphs by selecting node
        embeddings based on the provided mask, computing the mean embedding for each
        graph, and calculating the cross-entropy loss against the target labels.
        """
        total_loss = 0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.to(out.device)
        for gid in self.data.train_ids:
            graph_mask = (self.data.graph_id == gid) & mask
            graph_nodes = out[graph_mask]
            
            if graph_nodes.size(0) == 0:
                continue
            graph_logits = self.target_model.model.linear(graph_nodes.mean(dim=0, keepdim=True))
            # print(graph_logits, self.data.y[gid])
            graph_loss = F.cross_entropy(graph_logits, self.data.y[gid].unsqueeze(0))
            total_loss += graph_loss
        return total_loss
    
    
    def _gen_train_loader(self):
        self.logger.info("generate train loader")
        train_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]
        edge_index = self.filter_edge_index(self.data.edge_index, train_indices, reindex=False)
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 2], [2, 1]])

        self.train_loader = NeighborSampler(
            edge_index, node_idx=self.data.train_mask,
            sizes=[5, 5], num_nodes=self.data.num_nodes,
            batch_size=self.args['batch_size'], shuffle=True,
            num_workers=0)
        # if self.target_model_name in ['GCN','SGC']:
        _, self.edge_weight = gcn_norm(
            self.data.edge_index, 
            edge_weight=None, 
            num_nodes=self.data.x.shape[0],
            add_self_loops=False)

        _, self.edge_weight_unlearn = gcn_norm(
            self.data.edge_index_unlearn, 
            edge_weight=None, 
            num_nodes=self.data.x.shape[0],
            add_self_loops=False)

        self.logger.info("generate train loader finish")
        
    def filter_edge_index(self,edge_index, node_indices, reindex=True):
        
        assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu()

        node_index = np.isin(edge_index, node_indices)
        col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
        edge_index = edge_index[:, col_index]

        if reindex:
            return np.searchsorted(node_indices, edge_index)
        else:
            return edge_index
        
    def unlearn(self):
        """
        Perform the unlearning process by calculating the gradient influence and approximating the unlearning metrics.
        This method retrieves the gradient information using the current run identifier, computes the unlearning time and 
        F1 score approximation, and updates the respective averages for F1 score and unlearning time.
        """
        result_tuple = self.get_if_grad(self.run)
        unlearning_time, f1_score_unlearning = self.approxi(result_tuple)
        self.average_f1[self.run] = f1_score_unlearning
        self.avg_unlearning_time[self.run] = unlearning_time
        
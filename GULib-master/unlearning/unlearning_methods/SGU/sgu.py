import random
import heapq
import numpy
import torch
import copy
import numpy as np
from numba import njit, prange, jit
import os
import torch.nn.functional as F
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score,recall_score,roc_auc_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_undirected,to_scipy_sparse_matrix
import time
from task.node_classification import NodeClassifier
from utils import utils
from utils.utils import aug_normalized_adjacency, cal_distance
from sklearn.utils import shuffle
from torch_geometric.transforms import SIGN
from config import root_path,unlearning_path,unlearning_edge_path
from model.base_gnn.Convs import S2GConv,SGConv,GCNConv
from config import BLUE_COLOR,RESET_COLOR
from tqdm import tqdm
from task import get_trainer
from task.edge_prediction import EdgePredictor
from torch_geometric.utils import negative_sampling
from pipeline.Learning_based_pipeline import Learning_based_pipeline
class sgu(Learning_based_pipeline):
    """
    The `sgu` class implements a learning-based pipeline for Graph unlearning.
    It extends the `Learning_based_pipeline` class and provides methods for training, unlearning, and evaluating GNN models. 
    The class handles both node and edge unlearning tasks, calculates influence scores, performs positive and negative sampling, and evaluates the effectiveness 
    of the unlearning process using Membership Inference Attack (MIA).

    Class Attributes:
        args (dict): Configuration arguments for the unlearning process.

        logger (Logger): Logger for recording information and debugging.

        model_zoo (ModelZoo): Collection of models and data for the unlearning process.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.model_zoo = model_zoo
        self.data = model_zoo.data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = 1
        self.Budget = int(self.data.num_nodes *self.args["Budget"])
        self.num_nodes = self.data.num_nodes
        self.timecount = {}
        num_runs = self.args["num_runs"]
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_activating_time = np.zeros(num_runs)
        self.avg_sampling_time = np.zeros(num_runs)
        self.avg_unlearning_time = np.zeros(num_runs)
        self.run = 0
        self.best = 0
        self.final_auc = 0


    def determine_target_model(self):
        """
        Determines and sets up the target model for the unlearning process.
        """
        self.args["unlearn_trainer"] = 'SGUTrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        self.features = self.preprocess_feature()
        self.target_model.data.pre_features = self.features
        if self.args['dataset_name'] == "ogbn-products":
            self.target_model.model.adj = self.data.adj
        
    def train_original_model(self):
        """
        Trains the original model or loads an existing model from a specified path.
        This method checks if a pre-trained model exists at the specified path. If the model exists, it loads the model.
        If the model does not exist, it trains a new model, saves it to the specified path, and creates necessary directories
        if they do not exist. Additionally, if the 'poison' argument is set and the 'unlearn_task' is 'edge', it evaluates the model
        and stores the F1 score.
        """
        model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/"+self.args["downstream_task"]+"/" + \
                         self.args["base_model"] + "_SGU"
        # if os.path.exists(model_path):
        #     print(f"File {model_path} already exists.")
        #     self.target_model.load_model(model_path)
        # else:
        self.target_model.train()
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # with open(model_path, 'w'):
        #     print(f"File {model_path} created successfully.")
            # self.target_model.load_model(model_path)
            # self.target_model.save_model(model_path)
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.target_model.evaluate()
    def unlearning_request(self):
        """
        Handles the unlearning request based on the specified unlearning task.
        This function processes the unlearning request by loading the nodes or edges to be unlearned 
        from the corresponding files, and then updates the unlearning mask and count accordingly.
        """
        if self.args["unlearn_task"] == "node":
            path_un = unlearning_path + "_" + str(self.run) + ".txt"
            self.unlearning_nodes = np.loadtxt(path_un, dtype=int)
        elif self.args["unlearn_task"] == "edge":
            path_un_edge = unlearning_edge_path + "_" + str(self.run) + ".txt"
            self.unlearning_edges = np.loadtxt(path_un_edge, dtype=int)
            self.unlearning_nodes = np.unique(self.unlearning_edges)
            # F1_score, Accuracy, Recall = self.target_model.eval_unlearning(self.features, self.unlearning_nodes)
            # self.logger.info(
            #     'Original Model Unlearning : F1_score = {}  Accuracy = {}  Recall = {}'.format(F1_score, Accuracy, Recall))
        self.unlearning_num = self.unlearning_nodes.size
        self.unlearning_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.unlearning_mask[self.unlearning_nodes] = True
        print("Unlearning nodes:{}".format(self.unlearning_num))
        print("All nodes:{}".format(self.num_nodes))
    def unlearn(self):
        """
        Unlearns specific nodes or edges from the graph neural network model based on the specified task.
        This function handles two types of unlearning tasks: 'node' and 'edge'. Depending on the task, it calculates 
        the necessary embeddings, soft labels, and influence scores, and then identifies the nodes or edges to be 
        unlearned. It also performs positive and negative sampling for the unlearning process and updates the model 
        accordingly.
        """
        if self.args["unlearn_task"] == "node":
            start_time = time.time()
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
                self.original_emb = F.relu(self.target_model.model(self.features,return_all_emb = True)[0]).clone().detach().float()
                self.original_softlabels = F.softmax(self.target_model.model(self.features),dim = 1).clone().detach().float()
            else:
                self.data = self.data.to(self.device)
                self.original_emb = F.relu(self.target_model.model(self.data.x,self.data.edge_index,return_all_emb = True)[0]).clone().detach().float()
                self.original_softlabels = F.softmax(self.target_model.model(self.data.x,self.data.edge_index),dim = 1).clone().detach().float()

            self.feature_similarity = self.cal_similarity(self.features, self.unlearning_nodes, True).cuda()
            self.label_similarity = self.cal_similarity(self.original_softlabels, self.unlearning_nodes, False).cuda()
            self.influence_score = (self.feature_similarity + self.alpha * self.label_similarity).clone().detach()
            del self.feature_similarity
            del self.label_similarity

            self.activated_nodes = self.activate_nodes()
            self.avg_activating_time[self.run] = time.time()-start_time
            del self.influence_score
            self.activated_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            self.activated_mask[self.activated_nodes] = True

            parameter_list = []
            for param_tensor in self.target_model.model.parameters():
                param_clone = param_tensor.clone().detach().float()
                parameter_list.append(param_clone)
            self.original_w = parameter_list
            self.prototype_embedding = self.get_prototype()
            activated_label = np.argmax(self.original_softlabels[self.activated_nodes].cpu().numpy(), axis=1)
            start_time = time.time()
            self.pos_pair = self.pos_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
            self.neg_pair = self.neg_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
            self.avg_sampling_time[self.run] = time.time()-start_time
            self.best = self.target_model.sgu_unlearning(self.original_softlabels,
                                                        self.original_w,
                                                        self.unlearning_nodes,
                                                        self.activated_nodes,
                                                        self.pos_pair,
                                                        self.neg_pair,
                                                        self.features,
                                                        self.prototype_embedding,
                                                        self.avg_unlearning_time,
                                                        self.average_f1,
                                                        self.run)
            # self.target_model.load_model(root_path + "/data/model/node_level/"+self.args["dataset_name"]+ "/"+self.args["downstream_task"]+"/" + self.args["base_model"]+"_unlearning_best.pt")
            # self.final_auc = self.mia_attack()
            # self.logger.info("Budget: {}".format(self.Budget))
        elif self.args["unlearn_task"] == "edge":
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
                self.original_emb = F.relu(self.target_model.model(self.features,return_all_emb = True)[0]).clone().detach().float()
                self.original_softlabels = F.softmax(self.target_model.model(self.features),dim = 1).clone().detach().float()
            else:
                self.data = self.data.to(self.device)
                self.original_emb = F.relu(self.target_model.model(self.data.x,self.data.edge_index,return_all_emb = True)[0]).clone().detach().float()
                self.original_softlabels = F.softmax(self.target_model.model(self.data.x,self.data.edge_index),dim = 1).clone().detach().float()

            self.adj = to_scipy_sparse_matrix(self.data.edge_index, num_nodes=self.data.num_nodes)
            self.adj_sp = self.adj.tocoo()
            self.adj_torch = utils.sparse_mx_to_torch_sparse_tensor(self.adj_sp)
            self.adj_norm_sp = utils.aug_normalized_adjacency(self.adj)
            self.adj_norm_torch = utils.sparse_mx_to_torch_sparse_tensor(self.adj_norm_sp)
            degree_torch = torch.sparse.sum(self.adj_torch, dim=0).values()
            self.deg = np.array(degree_torch)
            #cal the influence score
            self.node_cnt_sum = torch.zeros(self.data.num_nodes,requires_grad=False)
            self.adj_lil = self.adj_sp.tolil()
            self.adj_norm_lil = self.adj_norm_sp.tolil()
            self.node_degree = torch.sparse.sum(self.adj_torch.to(self.device), [0]).to_dense().cpu().numpy()
            node_cnt = np.array(self.adj_norm_lil[self.unlearning_nodes].tocsr().max(axis=0).todense()).flatten()
            self.degree_heap = []
            start_time = time.time()
            for i in self.unlearning_nodes:
                node_cnt[i] = 1
                heapq.heappush(self.degree_heap, (self.node_degree[i], i))

            self.__prep_cnt_sum(self.node_cnt_sum, node_cnt, self.adj_norm_sp.row, self.adj_norm_sp.col,
                                self.adj_norm_sp.data)

            self.activated_nodes = torch.topk(self.node_cnt_sum, self.Budget)[1].numpy()
            self.avg_activating_time[self.run] = time.time()-start_time
            self.activated_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            self.activated_mask[self.activated_nodes] = True

            parameter_list = []
            for param_tensor in self.target_model.model.parameters():
                param_clone = param_tensor.clone().detach().float()
                parameter_list.append(param_clone)
            self.original_w = parameter_list
            self.prototype_embedding = self.get_prototype()
            activated_label = np.argmax(self.original_softlabels[self.activated_nodes].cpu().numpy(), axis=1)
            start_time = time.time()
            self.pos_pair = self.pos_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
            self.neg_pair = self.neg_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
            self.avg_sampling_time[self.run] = time.time()-start_time
            self.best = self.target_model.sgu_unlearning_edge(self.original_softlabels,
                                                        self.original_w,
                                                        self.unlearning_nodes,
                                                        self.activated_nodes,
                                                        self.pos_pair,
                                                        self.neg_pair,
                                                        self.features,
                                                        self.prototype_embedding,
                                                        self.avg_unlearning_time,
                                                        self.average_f1,
                                                        self.run)
            # self.target_model.load_model(root_path + "/data/model/edge_level/"+self.args["dataset_name"]+ "/"+self.args["downstream_task"]+"/" + self.args["base_model"]+"_unlearning_best.pt")
            
        
    


    def preprocess_feature(self,edge_mask=None):
        """
        Preprocesses the features of the graph data based on the specified base model.
        This function applies different propagation methods to the input features of the graph
        data depending on the base model specified in the arguments.
        """
        start_time = time.time()
        if self.args["base_model"] == "SGC":
            propagation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
            features_pre = propagation.forward_SGU(self.data.x,self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "S2GC":
            propagation = S2GConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "SIGN":
            return self.data.xs
        elif self.args["base_model"] == "GCN":
            propagation = propagation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "GAT":
            propagation = propagation = SGConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "SAGE":
            propagation = propagation = SGConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "GIN":
            propagation = propagation = SGConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "SAINT":
            propagation = propagation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "Cluster_GCN":
            propagation = propagation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        timesum = time.time()-start_time
        self.timecount["process_feaure_time"] = timesum

            # adj = aug_normalized_adjacency(self.data.adj)
            # adj_matrix = torch.FloatTensor(adj.todense()).cuda()
            # adj_matrix_last = torch.FloatTensor(adj.todense()).cuda()
            # for i in range(self.args["GNN_layer"]):
            #     adj_matrix_last = torch.mm(adj_matrix_last,adj_matrix).cuda()
            # features = self.data.x.cuda()
            # features_last = np.array(torch.mm(adj_matrix_last,features).cpu())
            # adj_matrix_last = np.array(adj_matrix_last.cpu())

    def cal_similarity(self,vectors,unleaning_nodes,isfeature):
        """
        Calculate the similarity matrix between vectors and unlearning nodes.
        This function computes the similarity between a set of vectors and a subset of vectors identified as unlearning nodes.
        It normalizes the vectors and computes the similarity matrix using a specified method.
        """
        # vectors = torch.FloatTensor(vectors)
        # similarity_matrix = np.zeros((self.num_nodes,self.num_nodes))
        # # similarity_matrix = torch.FloatTensor(self.num_nodes,self.num_nodes)
        # for i in range(self.num_nodes-1):
        #     for j in range(i+1,self.num_nodes):
        #         similarity_matrix[i][j] = cal_distance(vectors[i,:],vectors[j,:])
        #         similarity_matrix[j][i] = similarity_matrix[i][j]
        if self.args["base_model"] == "SIGN" and isfeature:
            vectors = self.target_model.model.get_features(vectors)

        if isinstance(vectors, np.ndarray):
            vectors = torch.FloatTensor(vectors)
        unlearning_vector = vectors[unleaning_nodes,:]
        unlearning_vector = F.normalize(unlearning_vector, p=2, dim=1).clone().detach().float()
        vectors = F.normalize(vectors, p=2, dim=1).clone().detach().float()
        # similarity_matrix = torch.matmul(unlearning_vector, vectors.T)
        # similarity_matrix = torch.sum(torch.matmul(unlearning_vector, vectors.T),dim=0)
        similarity_matrix = self.compute_similarity_in_chunks(unlearning_vector, vectors, 10000)

        return similarity_matrix
    
    def compute_similarity_in_chunks(self,unlearning_vector, vectors, chunk_size):
        """
        Computes the similarity between an unlearning vector and a set of vectors in chunks.
        This function calculates the similarity between a given unlearning vector and a set of vectors by processing the vectors in chunks to manage memory usage efficiently. The similarity is computed using the dot product between the unlearning vector and each chunk of vectors.
        """
        num_vectors = vectors.size(0)
        similarity_matrix = torch.zeros(num_vectors, device=unlearning_vector.device)

        # Iterate through the vectors in chunks
        for start in range(0, num_vectors, chunk_size):
            end = min(start + chunk_size, num_vectors)
            chunk = vectors[start:end]
            
            # Compute the chunk of the similarity matrix
            similarity_chunk = torch.matmul(unlearning_vector, chunk.T)
            
            # Sum the results along the dimension
            similarity_chunk_sum = torch.sum(similarity_chunk, dim=0)
            
            # Store the results in the corresponding part of the similarity matrix
            similarity_matrix[start:end] = similarity_chunk_sum

        return similarity_matrix

    def activate_nodes(self):
        """
        Activates a subset of nodes based on their influence scores.
        This method calculates the influence scores of nodes and selects the top-k nodes with the highest scores to be activated. 
        The influence scores of the nodes to be unlearned are set to zero before the selection process.
        """
        activated_node  = list()
        total_influence = torch.zeros(self.num_nodes)
        self.influence_score[self.unlearning_nodes] = 0
        # influence_mask = torch.zeros(self.num_nodes,dtype=torch.bool)
        # influence_mask[self.data.train_indices] = True
        # influence_mask[self.unlearning_nodes] = False
        # for i in range(self.args["num_unlearned_nodes"]):
        #     self.influence_score[i, ~influence_mask] = 0
        #     node_influence_score = self.influence_score[i,:]
        #     #compute the influence by all the unlearning node
        #     total_influence+= node_influence_score.cpu()

        max_score,max_node,count = 0, 0, 0
        # topk_val,activated_node = torch.topk(total_influence,self.Budget)
        topk_val,activated_node = torch.topk(self.influence_score,self.Budget)
        return np.array(activated_node.cpu(),dtype=int)

    def pos_sampling(self, activated_nodes,activated_labels,unlearning_nodes):
        """
        Perform positive sampling for the given activated nodes.
        This function generates positive samples for each activated node by finding nodes with the same label
        that are not in the unlearning nodes or activated nodes. It calculates the mean embedding of up to 5 
        such nodes for each activated node.
        """
        # pos_pair = {}
        # count = 0
        # number = 0
        # for node in activated_nodes:
        #     number = 0
        #     pos_pair[node] = []
        #     labels = self.data.y[self.data.train_mask].cpu().numpy()
        #     indice = np.where(labels == activated_labels[count])
        #     np.random.shuffle(indice[0])
        #     count += 1
        #     if indice == None:
        #         pos_pair[node].append(numpy.zeros_like(self.original_emb[0]))
        #     for index in indice[0]:
        #         if index not in unlearning_nodes and index not in activated_nodes:
        #             pos_pair[node].append(self.original_emb[index].cpu().numpy())
        #             number += 1
        #             if number >= 5:
        #                 break
        #     if pos_pair[node] == []:
        #         pos_pair[node].append(numpy.zeros_like(self.original_emb[0].cpu()))
        #     pos_pair[node] = np.mean(pos_pair[node],axis=0)
        pos_pair = {}
        labels = self.data.y[self.data.train_mask].cpu().numpy()  # 提前将labels转为numpy数组
        unlearning_set = set(unlearning_nodes)  # 转为集合加速查找
        activated_set = set(activated_nodes)  # 转为集合加速查找

        for count, node in enumerate(activated_nodes):
            # if count % 10000 == 0:
                # print("count!{}".format(count))
            pos_pair[node] = []
            indice = np.where(labels == activated_labels[count])[0]  # 获取索引并直接使用数组
            np.random.shuffle(indice)  # 打乱索引
            number = 0
            embeddings = []
            for index in indice:
                if index not in unlearning_set and index not in activated_set:
                    embeddings.append(self.original_emb[index].cpu().numpy())
                    number += 1
                    if number >= 5:
                        break

            if not embeddings:
                pos_pair[node].append(np.zeros_like(self.original_emb[0].cpu()))
            else:
                pos_pair[node] = np.mean(embeddings, axis=0)
        pos_pair = {k: np.array(v) for k, v in pos_pair.items()}
            


        return pos_pair

    def neg_sampling(self, activated_nodes, activated_labels, unlearning_nodes):
        """
        Perform negative sampling for the given activated nodes.
        This function generates negative samples for each activated node by selecting embeddings
        from nodes with the same label that are not in the unlearning set. The negative samples
        are created by averaging the embeddings of the selected nodes.
        """
        # neg_pair = {}
        # count = 0
        # for node in activated_nodes:
        #     number = 1
        #     neg_pair[node] = []
        #     for i in range(5):
        #         neg_pair[node].append(self.original_emb[node].cpu().numpy())
        #     count = 5
        #     labels = self.data.y[activated_nodes].cpu().numpy()
        #     indice = np.where(labels == activated_labels[count])
        #     np.random.shuffle(indice[0])
        #     if indice == None:
        #         continue
        #     for index in indice[0]:
        #         if index not in unlearning_nodes:
        #             neg_pair[node].append(self.original_emb[index].cpu().numpy())
        #             number += 1
        #             if number >= 10:
        #                 break
        #     neg_pair[node] = np.mean(neg_pair[node],axis=0)
        neg_pair = {}
        labels = self.data.y[activated_nodes].cpu().numpy()  # 提前将labels转为numpy数组
        unlearning_set = set(unlearning_nodes)  # 转为集合加速查找

        for count, node in enumerate(activated_nodes):
            # if count % 10000 == 0:
            #     print("count!{}".format(count))
            neg_pair[node] = []
            embeddings = []

            # 添加当前节点的embedding 5次
            current_emb = self.original_emb[node].cpu().numpy()
            embeddings.extend([current_emb] * 5)

            # 获取符合条件的节点索引
            indice = np.where(labels == activated_labels[count])[0]
            np.random.shuffle(indice)  # 打乱索引

            number = 5
            for index in indice:
                if index not in unlearning_set:
                    embeddings.append(self.original_emb[index].cpu().numpy())
                    number += 1
                    if number >= 10:
                        break

            # 计算均值
            neg_pair[node] = np.mean(embeddings, axis=0)
            
        return neg_pair
    #5个自身向量，5个负样本

    def delete_node(self):
        """
        Deletes nodes specified in the `unlearning_nodes` list from the graph data.
        This function updates the `edge_mask` attribute to exclude edges connected to the nodes
        that need to be unlearned. It sets the mask to `False` for all edges where either the 
        source or target node is in the `unlearning_nodes` list, effectively removing these edges 
        from the graph.
        """
        self.edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
        for node in self.unlearning_nodes:
            self.edge_mask[self.data.edge_index[0] == node] = False
            self.edge_mask[self.data.edge_index[1] == node] = False

    def retrain(self, unlearning_nodes, edge_mask):
        """
        Retrains the target model by removing specific edges and then training the model again.
        """
        self.target_model.data.edge_index = self.target_model.data.edge_index[:,edge_mask]
        self.target_model.train()
        edge_mask = True
        # F1 = self.target_model.eval_unlearning(unlearning_nodes,edge_mask)
        # self.logger.info('unlearning F1_score : {}'.format(F1))

    def mia_attack(self):
        """
        Perform a Membership Inference Attack (MIA) to evaluate the effectiveness of the unlearning method.
        This function calculates the AUC (Area Under the Curve) score to determine how well the model can distinguish 
        between members (nodes that were part of the training set) and non-members (nodes that were not part of the training set) 
        after the unlearning process.
        """
        self.mia_num = self.unlearning_num
        original_softlabels_member = self.original_softlabels[self.unlearning_nodes]
        original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args[
            "base_model"] == "SIGN":
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.features[self.unlearning_nodes]),dim = 1)
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.features[self.data.test_indices[:self.mia_num]]),dim = 1)
        else:
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x,self.data.edge_index),dim = 1)[self.unlearning_nodes]
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.x,self.data.edge_index),dim = 1)[self.data.test_indices[:self.mia_num]]

        mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        # self.logger.info("posterior:{}".format(posterior))
        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        self.average_auc[self.run] = auc
        # self.logger.info("auc:{}".format(auc))
        # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
        return auc

    def get_prototype(self):
        """
        Computes the prototype embeddings for each class based on the training data, excluding the unlearning and activated nodes.
        """
        train_indices = torch.tensor(self.data.train_indices,requires_grad=False)
        prototype_mask = torch.ones_like(train_indices, dtype=torch.bool)
        u_nodes = torch.from_numpy(self.unlearning_nodes)
        a_nodes = torch.from_numpy(self.activated_nodes)
        prototype_mask[torch.isin(train_indices, torch.cat((u_nodes,a_nodes)))] = False
        prototype_indices = train_indices[prototype_mask]
        tmp_emb = self.original_emb[prototype_indices]
        tmp_y = self.data.y[prototype_indices].cpu()
        unique_labels = np.unique(tmp_y)
        # 初始化一个数组用于存储每个类别的平均嵌入向量
        class2embeddings = torch.zeros((self.data.num_classes, tmp_emb.shape[1]),requires_grad=False)


        # 遍历每个类别
        for label in unique_labels:
            # 获取属于当前类别的样本索引
            indices = np.where(tmp_y == label)[0]
            # 获取属于当前类别的嵌入向量
            class_emb = tmp_emb[indices]
            # 计算当前类别的平均嵌入向量
            class_avg_emb = torch.mean(class_emb, dim=0)
            class2embeddings[label.item()] = class_avg_emb

        return class2embeddings

    def plot_auc(self,y_true,y_score):
        y_true = y_true
        y_score = y_score

        # 计算ROC曲线上的点
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # 计算AUC
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()



    def edge_unlearning(self):
        """
        Perform edge unlearning on the graph data.
        This function modifies the adjacency matrix of the graph by removing and adding edges, 
        reprocesses the data, and prepares the model for unlearning. It also calculates the 
        influence score of nodes, identifies activated nodes, and prepares features for the 
        target model. The function includes steps for saving and loading the model, as well 
        as performing contrastive learning sampling.
        """
        self.adj = to_scipy_sparse_matrix(self.data.edge_index, num_nodes=self.data.num_nodes)
        self.adj_sp = self.adj.tocoo()
        self.adj_torch = utils.sparse_mx_to_torch_sparse_tensor(self.adj_sp)
        self.adj_norm_sp = utils.aug_normalized_adjacency(self.adj)
        self.adj_norm_torch = utils.sparse_mx_to_torch_sparse_tensor(self.adj_norm_sp)
        degree_torch = torch.sparse.sum(self.adj_torch, dim=0).values()
        self.deg = np.array(degree_torch)
        self.data=self.data.to(self.device)
        #find the initailly activated nodes
        # rand = torch.randint(high=self.data.edge_index.size(1),size=(self.args["num_unlearned_edges"],))
        # remove_edge = self.data.edge_index[:,rand]
        if os.path.exists(root_path + "/data/SGU/unlearning_edge.pt"):
            remove_edge = torch.load(root_path+ "/data/SGU/unlearning_edge.pt")
            updated_edge_index = torch.hstack([self.data.edge_index.to(self.device), remove_edge.to(self.device)])
        else:
            remove_edge,updated_edge_index = self.generate_heterophilous_edges(self.data.edge_index,self.data.num_nodes,self.args["proportion_unlearned_edges_num"])
            torch.save(remove_edge,root_path + "/data/SGU/unlearning_edge.pt")

        self.data.edge_index = updated_edge_index
        self.reprocess()
        # print(self.data.train_edge_index[0,:].max(),self.data.train_edge_index[1,:].max())

        self.unlearning_nodes = np.union1d(remove_edge[0],remove_edge[1])
        self.logger.info("add edges:{}".format(remove_edge.size(1)))

        self.args["unlearn_trainer"] = 'SGUTrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)


        #cal the influence score
        self.node_cnt_sum = torch.zeros(self.data.num_nodes,requires_grad=False)
        self.adj_lil = self.adj_sp.tolil()
        self.adj_norm_lil = self.adj_norm_sp.tolil()
        self.node_degree = torch.sparse.sum(self.adj_torch.to(self.device), [0]).to_dense().cpu().numpy()
        node_cnt = np.array(self.adj_norm_lil[self.unlearning_nodes].tocsr().max(axis=0).todense()).flatten()
        self.degree_heap = []
        for i in self.unlearning_nodes:
            node_cnt[i] = 1
            heapq.heappush(self.degree_heap, (self.node_degree[i], i))

        self.__prep_cnt_sum(self.node_cnt_sum, node_cnt, self.adj_norm_sp.row, self.adj_norm_sp.col,
                            self.adj_norm_sp.data)

        self.activated_nodes = torch.topk(self.node_cnt_sum, self.Budget)[1].numpy()

        self.features = self.preprocess_feature()
        # self.target_model = NodeClassifier(self.args, self.data, self.model_zoo, self.logger)
        # self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        self.target_model.data.pre_features = self.features
        # model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/"+self.args["downstream_task"]+"/" + \
        #                  self.args["base_model"] + str(self.args["proportion_unlearned_edges"])
        # if os.path.exists(model_path):
        #     print(f"File {model_path} already exists.")
        #     self.target_model.load_model(model_path)
        # else:
            # os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # with open(model_path, 'w'):
            #     print(f"File {model_path} created successfully.")
        self.target_model.train(save=False)
                # self.target_model.save_model(model_path)
        ##############train################
        if self.args["base_model"] in ["SGC","S2GC","SIGN"] :
            self.softlabels = F.softmax(self.target_model.model(self.features.cuda()),dim = 1).cpu().detach().numpy()
            self.original_emb = F.relu(self.target_model.model(self.features.cuda(),return_all_emb = True)[0]).clone().detach().float()
            self.original_softlabels = F.softmax(self.target_model.model(
                self.features.cuda()),dim = 1).clone().detach().float()
        else:
            self.softlabels = F.softmax(self.target_model.model(self.features.cuda(),self.data.edge_index.cuda()),dim = 1)
            self.original_emb = F.relu(self.target_model.model(self.features.cuda(),self.data.edge_index.cuda(),return_all_emb = True)[0]).clone().detach().float()
            self.original_softlabels = F.softmax(self.target_model.model(
                self.features.cuda(),self.data.edge_index.cuda()),dim = 1).clone().detach().float()
        
        
        # get frozen model param
        
        parameter_list = []
        for param_tensor in self.target_model.model.parameters():
            param_clone = param_tensor.clone().detach().float()
            parameter_list.append(param_clone)
        self.original_w = parameter_list

        # get prototype embedding
        self.prototype_embedding = self.get_prototype()
        
        # pred = self.target_model.model.emb2softlable(
        #     self.prototype_embedding.to(self.device)).cpu().detach().numpy()
        # pred = np.argmax(pred, axis=1)
        # self.logger.info(
        #     "proto_acc:{}".format(accuracy_score(y_true=np.arange(0, self.data.num_classes), y_pred=pred)))

        # contrastive learning sampling
        activated_label = np.argmax(self.original_softlabels[self.activated_nodes].cpu().numpy(), axis=1)
        self.pos_pair = self.pos_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
        self.neg_pair = self.neg_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)



        self.best = self.SGU_unlearning_edge(self.original_softlabels,
                                                self.original_w,
                                                remove_edge,
                                                self.activated_nodes,
                                                self.pos_pair,
                                                self.neg_pair,
                                                self.features,
                                                self.prototype_embedding,
                                                remove_edge)
        # self.target_model.load_model(root_path + "/data/model/node_level/"+self.args["dataset_name"]+ "/"+self.args["downstream_task"]+"/" + self.args["base_model"]+"_unlearning_best.pt")
        # self.final_auc = self.mia_attack()
        self.logger.info("Budget: {}".format(self.Budget))










    def __prep_cnt_sum(self,node_cnt_sum, K_arr, row, col, data):
        """
        Prepares and updates the node count sum based on the given arrays.
        This function identifies the non-zero indices in the K_arr array and creates a mask to compare these indices with the row array. 
        It then updates the node_cnt_sum array by iterating through the relevant indices and applying a condition to update the values.
        """
        non_zero_indices = np.where(K_arr > 0)[0]

        # 广播non_zero_indices到row数组的形状，然后比较是否相等
        # 这样会自动对每个元素进行比较，而不需要显式的循环
        mask = np.isin(row, non_zero_indices)

        tx_values = np.where(mask)[0]

        for tx in prange(tx_values[0]):
            if self.node_cnt_sum[col[tx]] < K_arr[row[tx]] * data[tx]:
                self.node_cnt_sum[col[tx]] = K_arr[row[tx]] * data[tx]

    def generate_heterophilous_edges(self,edge_index, node_num, proportion):
        """
        Generate heterophilous edges and update edge_index.
        This function generates heterophilous edges (edges between nodes of different classes) 
        and updates the given edge_index with these new edges.
        """

        # 计算全连接图的边数量
        full_connected_edges = node_num * (node_num - 1) // 2

        # 根据比例计算需要生成的异配性边的数量
        num_heterophilous_edges = int(full_connected_edges * proportion)

        heterophilous_edges = []
        for _ in range(num_heterophilous_edges):
            # 随机选择两个不同类别的节点
            node1 = np.random.choice(node_num)
            while True:
                node2 = np.random.choice(node_num)
                if self.data.y[node1] != self.data.y[node2] and node1 != node2:
                    break
            heterophilous_edges.append((node1, node2))

        # 将新的异配性边添加到现有的edge_index中
        heterophilous_edges_array = torch.tensor(heterophilous_edges,requires_grad=False).T
        heterophilous_edges_array = to_undirected(heterophilous_edges_array)
        updated_edge_index = torch.hstack([edge_index, heterophilous_edges_array])

        return heterophilous_edges_array, updated_edge_index


    def print_time(self):
        for key,value in self.timecount.items():
            if key == "training_time":
                value = value/self.args["unlearning_epochs"]
            self.logger.info("{} : {}s".format(key,value))

        
    def reprocess(self):
        """
        Reprocesses the graph data by reindexing the nodes and edges for the training, validation, and test sets.
        """
        train_new_index = 0
        val_new_index = 0
        test_new_index = 0
        train_dict = {}
        val_dict = {}
        test_dict = {}
        for node in range(self.data.num_nodes):
            if self.data.train_mask[node]:
                train_dict[node] = train_new_index
                train_new_index += 1
            elif self.data.val_mask[node]:
                val_dict[node] = val_new_index
                val_new_index += 1
            elif self.data.test_mask[node]:
                test_dict[node] = test_new_index
                test_new_index += 1
        # 删除连接训练、验证和测试集之间的边
        train_edge_index = self.data.edge_index.clone()
        val_edge_index = self.data.edge_index.clone()
        test_edge_index = self.data.edge_index.clone()
        
        train_edge_mask = self.data.train_mask[train_edge_index[0]] & self.data.train_mask[train_edge_index[1]]
        self.data.train_edge_index = train_edge_index[:, train_edge_mask]
        for edge in range(self.data.train_edge_index.size(1)):
            self.data.train_edge_index[0][edge] = train_dict[self.data.train_edge_index[0][edge].item()]
            self.data.train_edge_index[1][edge] = train_dict[self.data.train_edge_index[1][edge].item()]
        # 删除连接验证集之外的边
        val_edge_mask = self.data.val_mask[val_edge_index[0]] & self.data.val_mask[val_edge_index[1]]
        self.data.val_edge_index = val_edge_index[:, val_edge_mask]
        for edge in range(self.data.val_edge_index.size(1)):
            self.data.val_edge_index[0][edge] = val_dict[self.data.val_edge_index[0][edge].item()]
            self.data.val_edge_index[1][edge] = val_dict[self.data.val_edge_index[1][edge].item()]

            # 删除连接测试集之外的边
        test_edge_mask = self.data.test_mask[test_edge_index[0]] & self.data.test_mask[test_edge_index[1]]
        self.data.test_edge_index = test_edge_index[:, test_edge_mask]
        for edge in range(self.data.test_edge_index.size(1)):
            self.data.test_edge_index[0][edge] = test_dict[self.data.test_edge_index[0][edge].item()]
            self.data.test_edge_index[1][edge] = test_dict[self.data.test_edge_index[1][edge].item()]
    
    
    
        
    















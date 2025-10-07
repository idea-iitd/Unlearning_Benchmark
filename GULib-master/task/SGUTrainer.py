import torch.nn.functional as F
import time
import torch
import copy
import numpy as np
from tqdm import tqdm
from task import BaseTrainer
from config import root_path
from sklearn.metrics import f1_score, accuracy_score,recall_score,roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from utils.utils import sparse_mx_to_torch_sparse_tensor,normalize_adj
class SGUTrainer(BaseTrainer):
    """
    SGUTrainer class for training and performing SGU unlearning method on Graph Neural Networks (GNNs).

    This class includes methods for:
        - Training the GNN model on node-level or edge-level tasks.

        - Evaluating the model's performance.

        - Performing the SGU unlearning process for nodes and edges.

        - Saving the best-performing model states.

    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, 
                    training hyperparameters, unlearning settings, and other relevant settings.
        
        logger (logging.Logger): Logger object used to log training progress, metrics, 
                                 and other important information.
        
        model (torch.nn.Module): The neural network model that will be trained and evaluated.
        
        data (torch_geometric.data.Data): The dataset containing edge and node information 
                                          for training, validation, and testing.
        
        device (torch.device): The computation device (CPU or GPU) on which the model 
                               and data are loaded for training and evaluation.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the SGUTrainer with the provided configuration, logger, model, and data.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications, 
                        training hyperparameters, unlearning settings, and other relevant settings.
            
            logger (logging.Logger): Logger object used to log training progress, metrics, 
                                     and other important information.
            
            model (torch.nn.Module): The neural network model that will be trained and evaluated.
            
            data (torch_geometric.data.Data): The dataset containing edge and node information 
                                              for training, validation, and testing.
        """
        super().__init__(args, logger, model, data)

    def train(self,save=False,model_path=None):
        """
        Trains the GNN model based on the specified downstream task.
        
        Depending on the `downstream_task` configuration, this method delegates the training process to 
        either `train_node` or `train_edge`.

        Args:
            save (bool, optional): Flag indicating whether to save the trained model. Defaults to `False`.
            
            model_path (str, optional): Path to save the trained model if `save` is `True`. Defaults to `None`.
        
        Returns:
            tuple or None: Returns the result of `train_node` or `train_edge` method based on the downstream task.
                           If the downstream task is neither 'node' nor 'edge', returns `None`.
        """
        if self.args["downstream_task"] == 'node':
            return self.train_node(save,model_path)
        elif self.args["downstream_task"] == 'edge':
            return self.train_edge(save,model_path)
    
    def train_node_fullbatch(self,save = False, retrain=False):
        """
        Trains the GNN model for node-level classification tasks using full-batch training.

        Args:
            save (bool, optional): Flag indicating whether to save the trained model. Defaults to `False`.
            
            retrain (bool, optional): Flag indicating whether to retrain the model. Defaults to `False`.
        
        Returns:
            None
        """
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        # if self.args['dataset_name'] == "ogbn-products":
        #     adj = to_scipy_sparse_matrix(self.data.edge_index,num_nodes=self.data.num_nodes)
        #     adj = normalize_adj(adj)
        #     self.model.adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        start_time = time.time()
        best_acc = 0
        best_w = 0
        for epoch in tqdm(range(self.args['num_epochs'])):
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
                out = self.model(self.data.pre_features[self.data.train_indices])
                loss = F.cross_entropy(out, self.data.y[self.data.train_indices])
            else:
                out = self.model(self.data.x,self.data.edge_index)
                loss = F.cross_entropy(out[self.data.train_indices], self.data.y[self.data.train_indices])

            self.logger.info("epoch:{} loss:{}".format(epoch,loss))
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % self.args["test_freq"] == 0:
                F1_score, Accuracy, Recall = self.test_node_fullbatch(self.data.pre_features[self.data.test_indices])
                self.logger.info(
                    'epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score, Accuracy, Recall))
                if Accuracy > best_acc :
                    best_acc = Accuracy
                    best_w = copy.deepcopy(self.model.state_dict())
        if self.args["unlearn_task"] == "edge":
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] +"/"+ self.args["downstream_task"]+"/" + \
                         self.args["base_model"] + str(self.args["proportion_unlearned_edges"])
        else:
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] +"/"+ self.args["downstream_task"]+ "/" + \
                         self.args["base_model"]
        if save:
            self.save_model(model_path,best_w)
        self.logger.info("best:{}".format(best_acc))


    @torch.no_grad()
    def test_node_fullbatch(self,feature):
        """
        Evaluates the GNN model on node-level classification tasks using full-batch evaluation.

        Args:
            feature (torch.Tensor): The input features for the test nodes.
        
        Returns:
            tuple: A tuple containing:
                - F1_score (float): The micro-averaged F1 score on the test dataset.
                - Accuracy (float): The accuracy on the test dataset.
                - Recall (float): The micro-averaged recall on the test dataset.
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = F.softmax(self.model(feature),dim = 1).cpu()
        else:
            y_pred = self.model(self.data.x, self.data.edge_index).cpu()
            y_pred = y_pred[self.data.test_indices]
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[self.data.test_indices], y_pred, average="micro")
        Acc_score = accuracy_score(y_true=y[self.data.test_indices], y_pred=y_pred)
        Recall_score = recall_score(y_true=y[self.data.test_indices], y_pred=y_pred, average="micro")
        return F1_score, Acc_score, Recall_score
    
    def train_edge(self, save,model_path):
        """
        Trains the GNN model for edge-level tasks.
        
        This method resets the model's parameters, initializes the optimizer, and trains the model for a specified 
        number of epochs. It performs negative sampling for edge classification, computes the loss, and evaluates 
        the model's performance at regular intervals. The model state with the best ROC AUC score is saved.

        Args:
            save (bool): Flag indicating whether to save the trained model.
            
            model_path (str): Path to save the trained model.
        
        Returns:
            None
        """
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr,
                                          weight_decay=self.model.config.decay)
        start_time = time.time()
        best_auc = 0
        best_w = 0
        for epoch in range(self.args['num_epochs']):
            self.model.train()
            self.optimizer.zero_grad()
            neg_edge_index = negative_sampling(
                edge_index=self.data.train_edge_index, num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_index.size(1)
            )
            if self.args["base_model"] in ["SGC","S2GC","SIGN"]:
                # out = self.model(self.data.x)
                out = self.model(self.data.pre_features)
            else:
                out = self.model(self.data.x, self.data.train_edge_index)
            edge_logits = self.decode(z=out, pos_edge_index=self.data.train_edge_index,neg_edge_index=neg_edge_index)
            edge_labels = self.get_edge_labels(self.data.train_edge_index, neg_edge_index)
            # edge_labels = edge_labels.to(self.device)
            loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % self.args["test_freq"] == 0:
                AUC_SCORE = self.evaluate_edge_model()
                self.logger.info(
                    'epoch: {}  AUC_Score = {}'.format(epoch, AUC_SCORE))
                if AUC_SCORE > best_auc:
                    best_auc = AUC_SCORE
                    best_w = copy.deepcopy(self.model.state_dict())

        self.logger.info("best:{}".format(best_auc))
        
    def sgu_unlearning(self,original_softlabels,
                            original_w,
                            unlearning_nodes,
                            activated_nodes,
                            pos_pair,
                            neg_pair,
                            features,
                            prototype_embedding,
                            avg_training_time,
                            average_f1,
                            run):
        """
        Performs SGU unlearning method on node-level tasks.
        
        This method targets specific nodes and their activated neighbors for unlearning. It adjusts the model's parameters 
        to forget the influence of the targeted nodes while preserving the model's performance on other parts of the graph.

        Args:
            original_softlabels (torch.Tensor): The original soft labels before unlearning.
            
            original_w (list): The original model parameters before unlearning.
            
            unlearning_nodes (list or torch.Tensor): Indices of nodes targeted for unlearning.
            
            activated_nodes (list or torch.Tensor): Indices of neighboring nodes activated for unlearning.
            
            pos_pair (dict): Dictionary containing positive pairs for activated nodes.
            
            neg_pair (dict): Dictionary containing negative pairs for activated nodes.
            
            features (torch.Tensor): Input features for the nodes.
            
            prototype_embedding (torch.Tensor): Prototype embeddings for the classes.
            
            avg_training_time (list): List to accumulate average training time per run.
            
            average_f1 (list): List to store average F1 scores per run.
            
            run (int): Index representing the current run.
        
        Returns:
            float: The best F1 score achieved during unlearning.
        """
        self.model.train()
        avg_training_time[run] = 0
        self.lam = 1
        self.tau = 0.5
        self.para1 = self.args["para1"]
        self.para2 = self.args["para2"]
        self.para3 = self.args["para3"]
        self.para4 = self.args["para4"]
        self.para5 = self.args["para5"]
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        prototype_embedding = prototype_embedding.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr,
                                          weight_decay=self.model.config.decay)

        indices = torch.randperm(self.data.y[unlearning_nodes].size(0))

        # 使用这些索引重新排列原始张量
        random_class = self.data.y[unlearning_nodes][indices]
        random_emb_proto = prototype_embedding[random_class]
        average_probs = torch.ones(self.data.num_classes) / self.data.num_classes
        stacked_probs = average_probs.repeat(self.args["num_unlearned_nodes"], 1).detach()
        best = 0
        best_w = 0
        F1_score = 0
        if self.args["downstream_task"]=="node":
            F1_score,Accuracy,Recall = self.test_node_fullbatch(features[self.data.test_indices])
        elif self.args["downstream_task"]=="edge":
            F1_score = self.evaluate_edge_model()
        self.logger.info("Original F1 Score: {:.4f} ".format(F1_score) )
        
        pos_tensors = [torch.tensor(pos_pair[node]) for node in activated_nodes]
        positive_tensors = torch.stack([F.normalize(pos_tensor, p=2, dim=0) for pos_tensor in pos_tensors], dim=0).cuda()
        
        neg_tensors = [torch.tensor(neg_pair[node]) for node in activated_nodes]
        negtive_tensors = torch.stack([F.normalize(negtive_tensor, p=2, dim=0) for negtive_tensor in neg_tensors],
                                       dim=0).cuda()
        embedding_E_proto = prototype_embedding[self.data.y[activated_nodes]]
        original_output = original_softlabels[activated_nodes]
        for epoch in tqdm(range(self.args['unlearning_epochs']), desc="Unlearning", unit="epoch"):
            self.model.train()
            self.optimizer.zero_grad()

            start_time = time.time()
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args[
                "base_model"] == "SIGN":
                softlabels_all = self.model(features,return_all_emb = True)
                softlabels_U = F.softmax(softlabels_all[-1][unlearning_nodes],dim =1)
                softlabels_E = F.softmax(softlabels_all[-1][activated_nodes],dim =1)
                embedding_U = F.relu(softlabels_all[0][unlearning_nodes])
                embedding_E = F.relu(softlabels_all[0][activated_nodes])
                # softlabels_U = F.softmax(self.model(features[unlearning_nodes]),dim =1)
                # softlabels_E = F.softmax(self.model(features[activated_nodes]),dim=1)
                # embedding_E = F.relu(self.model(features[activated_nodes],return_all_emb = True)[0])
                # embedding_U = F.relu(self.model(features[unlearning_nodes],return_all_emb = True)[0])
            else:
                softlabels_all = self.model(self.data.x,self.data.edge_index,return_all_emb = True)
                softlabels_U = F.softmax(softlabels_all[-1][unlearning_nodes],dim =1)
                softlabels_E = F.softmax(softlabels_all[-1][activated_nodes],dim =1)
                embedding_U = F.relu(softlabels_all[0][unlearning_nodes])
                embedding_E = F.relu(softlabels_all[0][activated_nodes])
                # softlabels_U = F.softmax(self.model(self.data.x,self.data.edge_index),dim = 1)[unlearning_nodes]
                # softlabels_E = F.softmax(self.model(self.data.x,self.data.edge_index),dim = 1)[activated_nodes]
                # embedding_E = F.relu(self.model(self.data.x,self.data.edge_index,return_all_emb = True)[0])[activated_nodes]
                # embedding_U = F.relu(self.model(self.data.x,self.data.edge_index,return_all_emb = True)[0])[unlearning_nodes]
            w_U = []
            ### cal loss###
            loss_w = 0
            for param_tensor in self.model.parameters():
                param_clone = param_tensor
                w_U.append(param_tensor)
            for i in range(len(original_w)):
                delta_w = (w_U[i] - original_w[i])
                loss_w += torch.norm(delta_w)
            creterion_MSE = torch.nn.MSELoss(reduction="mean")
            smooth_factor = 1e-9
            loss_pE = F.kl_div(torch.log(softlabels_E + smooth_factor),original_output + smooth_factor, reduction='batchmean')
            pos_scores = torch.exp(torch.einsum('ij,ij->i', F.normalize(embedding_E, p=2, dim=1),
                                                positive_tensors )/ self.tau)
            neg_scores = torch.exp(torch.einsum('ij,ij->i', F.normalize(embedding_E, p=2, dim=1),
                                                negtive_tensors) / self.tau)
            loss_hE = -torch.log(pos_scores /50*neg_scores).sum()
            
            loss_proto = creterion_MSE(embedding_E,embedding_E_proto)

            loss_pU = F.cross_entropy(softlabels_U, random_class.to(self.device))

            loss_emb_U = creterion_MSE(embedding_U, random_emb_proto)

            loss = ( self.para1 * loss_w +  self.para3 * loss_pE ) + (self.para2 * loss_hE +  self.para4 * (loss_pU+ loss_emb_U)  + self.para5 * loss_proto)

            loss.backward()
            avg_training_time[run] += (time.time() - start_time)
            if (epoch+1) % self.args["test_freq"] == 0:
                if self.args["downstream_task"]=="node":
                    F1_score,Accuracy,Recall = self.test_node_fullbatch(features[self.data.test_indices])
                elif self.args["downstream_task"]=="edge":
                    F1_score = self.evaluate_edge_model()
                self.logger.info("Epoch: {:03d} | F1 Score: {:.4f} | Loss: {:.4f}".format(epoch, F1_score, loss) )
            
            self.optimizer.step()
            if F1_score > best and epoch > 30:
                best = F1_score
                best_w = copy.deepcopy(self.model.state_dict())
        self.logger.info("Parameters - para1: {:.4f} | para2: {:.4f} | para3: {:.4f} | para4: {:.4f} | para5: {:.4f}".format(self.para1, self.para2, self.para3, self.para4, self.para5))

        self.logger.info('Best Unlearning: F1_score: {:.4f}'.format(best))
        average_f1[run] = best
        
        # save_path = root_path + "/data/model/node_level/"+self.args["dataset_name"]+ "/"+self.args["downstream_task"]+"/" + self.args["base_model"]+"_unlearning_best.pt"
        # with open(save_path,'w') as file:
        #     self.save_model(save_path,best_w)
        self.model.load_state_dict(best_w)

        return best
    
    def sgu_unlearning_edge(self,
                        original_softlabels,
                        original_w,
                        unlearning_nodes,
                        activated_nodes,
                        pos_pair,
                        neg_pair,
                        original_feaures,
                        prototype_embedding,
                        avg_training_time,
                        average_f1,
                        run):
        """
        Performs SGU unlearning method on edge-level tasks.
        
        This method targets specific edges and their activated components for unlearning. It adjusts the model's parameters 
        to forget the influence of the targeted edges while preserving the model's performance on other parts of the graph.

        Args:
            original_softlabels (torch.Tensor): The original soft labels before unlearning.
            
            original_w (list): The original model parameters before unlearning.
            
            unlearning_nodes (list or torch.Tensor): Indices of nodes targeted for unlearning.
            
            activated_nodes (list or torch.Tensor): Indices of activated nodes for unlearning.
            
            pos_pair (dict): Dictionary containing positive pairs for activated nodes.
            
            neg_pair (dict): Dictionary containing negative pairs for activated nodes.
            
            original_features (torch.Tensor): Original input features for the nodes.
            
            prototype_embedding (torch.Tensor): Prototype embeddings for the classes.
            
            avg_training_time (list): List to accumulate average training time per run.
            
            average_f1 (list): List to store average F1 scores per run.
            
            run (int): Index representing the current run.
        
        Returns:
            float: The best F1 score achieved during unlearning.
        """
        avg_training_time[run] = 0
        self.tau = 0.5
        self.para1 = self.args["para1"]
        self.para2 = self.args["para2"]
        self.para3 = self.args["para3"]
        self.para4 = self.args["para4"]
        self.para5 = self.args["para5"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr,
                                          weight_decay=self.model.config.decay)
        best = 0
        prototype_embedding = prototype_embedding.cuda()


        for epoch in range(self.args['unlearning_epochs']):
            
            self.model.train()
            self.optimizer.zero_grad()
            start_time = time.time()
            
            if self.args["base_model"] in ["SGC","S2GC","SIGN"] :
                z = self.model(original_feaures)
            else:
                z = self.model(original_feaures,self.data.train_edge_index)
            neg_edge_index = negative_sampling(
                edge_index=self.data.train_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_index.size(1)
            )
            neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)
            pos_edge_label = torch.ones(self.data.train_edge_index.size(1),dtype=torch.float32)
            
            logits = self.decode(z=z, pos_edge_index=self.data.train_edge_index,neg_edge_index=neg_edge_index)
            label = torch.cat((pos_edge_label,neg_edge_label),dim=-1)
            loss_U_edge = F.binary_cross_entropy_with_logits(logits.cpu(), label.cpu())
            
            if self.args["base_model"] in ["SGC","S2GC","SIGN"] :
                softlabels_E = F.softmax(self.model(original_feaures[activated_nodes]),dim=1)
                embedding_E = F.relu(self.model(original_feaures[activated_nodes],return_all_emb = True)[0])
            else:
                softlabels_E = F.softmax(self.model(original_feaures,self.data.edge_index),dim=1)[activated_nodes]
                embedding_E = F.relu(self.model(original_feaures,self.data.edge_index,return_all_emb = True)[0][activated_nodes])
            

            w_U = []
            loss_w = 0
            for param_tensor in self.model.parameters():
                param_clone = param_tensor
                w_U.append(param_tensor)
            for i in range(len(original_w)):
                delta_w = (w_U[i] - original_w[i])
                loss_w += torch.norm(delta_w)
            creterion_MSE = torch.nn.MSELoss(reduction="mean")
            smooth_factor = 1e-9

            loss_pE = F.kl_div(torch.log(softlabels_E + smooth_factor),original_softlabels[activated_nodes] + smooth_factor, reduction='batchmean')
            pos_tensors = [torch.tensor(pos_pair[node]) for node in activated_nodes]
            positive_tensors = torch.stack([F.normalize(pos_tensor, p=2, dim=0) for pos_tensor in pos_tensors], dim=0).cuda()
            pos_scores = torch.exp(torch.einsum('ij,ij->i', F.normalize(embedding_E, p=2, dim=1),
                                                positive_tensors )/ self.tau)
            neg_scores = torch.zeros_like(pos_scores)
            
            neg_tensors = [torch.tensor(neg_pair[node]) for node in activated_nodes]
            negtive_tensors = torch.stack([F.normalize(neg_tensor, p=2, dim=0) for neg_tensor in neg_tensors], dim=0).cuda()
            neg_scores = torch.exp(torch.einsum('ij,ij->i', F.normalize(embedding_E, p=2, dim=0),
                                negtive_tensors) / self.tau)
            
            loss_hE = -torch.log(pos_scores / 50*neg_scores).sum()


            embedding_E_proto = prototype_embedding[self.data.y[activated_nodes].cuda()]


            loss_proto = creterion_MSE(embedding_E,embedding_E_proto)

            loss = self.para1 * loss_w +  self.para2 * loss_hE + self.para3 * loss_pE + self.para4 * loss_U_edge  + self.para5 * loss_proto

            # loss = loss_pE + loss_proto + loss_pU

            self.logger.info("loss_w: {}  loss_hE: {} loss_pE: {} loss_pU: {} loss_proto:{}".format(self.para1*loss_w,
                                                                                                    self.para2 * loss_hE,
                                                                                                    self.para3 * loss_pE,
                                                                                                    self.para4 * loss_U_edge,
                                                                                                    self.para5 * loss_proto))
            self.logger.info("epoch:{} loss:{}".format(epoch, loss))
            loss.backward()
            self.optimizer.step()
            avg_training_time[run] += (time.time() - start_time)
            if (epoch+1) % self.args["test_freq"] == 0:
                F1_score = self.SGU_eval_edge(original_feaures.to(self.device))
                self.logger.info('epoch: {}  F1_score = {}'.format(epoch, F1_score))
                if F1_score > best and epoch > 30:
                    best = F1_score
                    best_w = copy.deepcopy(self.model.state_dict())
            
        self.logger.info("para1:{}   para2:{}   para3:{}   para4:{}   para5:{}".format(self.para1,
                                                                                       self.para2,
                                                                                       self.para3,
                                                                                       self.para4,
                                                                                       self.para5))
        self.logger.info('Unlearning: F1_score = {}  best: {}'.format(F1_score,best))
        self.logger.info('Best Unlearning: F1_score: {:.4f}'.format(best))
        average_f1[run] = best
        

        self.model.load_state_dict(best_w)
        return best
    
    def SGU_eval_edge(self,test_features):
        """
        Evaluates the GNN model on edge-level tasks.
        
        This method sets the model to evaluation mode, performs a forward pass on the test features, and computes 
        the ROC AUC score for edge classification. It utilizes negative sampling to generate negative edges for 
        evaluation.

        Args:
            test_features (torch.Tensor): The input features for the test edges.
        
        Returns:
            float: The ROC AUC score for edge-level classification on the test dataset.
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            z = test_features
        else:
            z = self.model(self.data.x,self.data.edge_index)
        neg_edge_index = negative_sampling(
                edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.test_edge_index.size(1)
            )
        edge_pred_logits = self.decode(z=z, pos_edge_index=self.data.test_edge_index,neg_edge_index=neg_edge_index).sigmoid()
        # edge_pred_logits = torch.sigmoid(edge_pred_logits)
        edge_pred = edge_pred_logits.cpu()
        edge_pred = torch.where(edge_pred_logits > 0.5, torch.tensor(1), torch.tensor(0))
        # edge_pred = torch.argmax(edge_pred_logits)
        pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
        neg_edge_labels = torch.zeros(neg_edge_index.size(1),dtype=torch.float32)
        edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
        F1_score = roc_auc_score(edge_labels.cpu(), edge_pred.detach().cpu().numpy())
        return F1_score
    
    
    
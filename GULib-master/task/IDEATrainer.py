import torch
import numpy as np
from task.BaseTrainer import BaseTrainer
import time
from sklearn.metrics import f1_score,roc_auc_score
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import grad
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import negative_sampling
class IDEATrainer(BaseTrainer):
    """
    IDEATrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for applying the IDEA (Inductive Graph Unlearning) unlearning method.
    
    This class manages the training and evaluation of GNN models, setting the foundation for the subsequent IDEA unlearning process.
    It includes methods for evaluating model performance, computing gradients for unlearning, continuing training after unlearning, 
    and managing data loaders. 
    
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
    
        target_model_name (str): Name of the target model architecture (e.g., 'GCN', 'SGC').
    
        attack_preparations (dict): Dictionary to store preparations related to attacks or evaluations.
    
        loss_all (Any): Placeholder for storing loss values or related information.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the IDEATrainer with the provided configuration, logger, model, and data.
        
        This constructor sets up the necessary attributes for training and evaluating the GNN model. It initializes
        the target model name based on the configuration, prepares a dictionary for attack-related preparations,
        and initializes a placeholder for storing loss information.
        
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
        self.target_model_name = self.args["base_model"]
        self.attack_preparations = {}
        self.loss_all = None
    
    
    def evaluate(self):
        """
        Evaluates the GNN model's performance based on the specified downstream task.
        
        This method computes the F1 score for node classification, edge prediction, or graph classification 
        depending on the `downstream_task` argument. It logs the evaluation time and the final test F1 score.
        
        Returns:
            float: The F1 score of the model on the test set.
        """
        self.logger.info('model evaluation')

        start_time = time.time()
        if self.args["downstream_task"]=="node":
            
            posterior = self.posterior()
            test_f1 = f1_score(
                self.data.y[self.data['test_mask']].cpu().numpy(), 
                posterior.argmax(axis=1).cpu().numpy(), 
                average="micro"
            )
        elif self.args["downstream_task"]=="edge":
            # out = self.model(self.data.x,self.data.test_edge_index)
            test_f1 = self.evaluate_edge_model()
        elif self.args["downstream_task"]=="graph":
            test_f1 = self.evaluate_graph_model()
        evaluate_time = time.time() - start_time
        self.logger.info("Evaluation cost %s seconds." % evaluate_time)

        self.logger.info("Final Test F1: %s" % (test_f1,))
        return test_f1
    
    def eval_unlearn_edge(self,out):
        """
        Evaluates the model's performance on edge prediction after unlearning.
        
        This method performs negative sampling, decodes edge predictions, computes sigmoid probabilities, 
        and calculates the ROC AUC score to assess the effectiveness of the unlearning process on edge predictions.
        
        Args:
            out (torch.Tensor): Model outputs used for edge prediction.
        
        Returns:
            float: The ROC AUC score for edge prediction after unlearning.
        """
        neg_edge_index = negative_sampling(
            edge_index=self.data.test_edge_index,num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.test_edge_index.size(1)
        )

        edge_pred_logits = self.decode(z=out, pos_edge_index=self.data.test_edge_index,neg_edge_index=neg_edge_index)
        
        edge_pred_logits = torch.sigmoid(edge_pred_logits)

        edge_pred = edge_pred_logits.cpu()

        pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
        neg_edge_labels = torch.zeros(neg_edge_index.size(1),dtype=torch.float32)
        edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
        AUC_score = roc_auc_score(edge_labels.detach().cpu(), edge_pred.detach().cpu())
        return AUC_score
    
    def idea_train(self,unlearn_info=None):
        """
        Initiates the training process to gather gradients necessary for the IDEA unlearning method.
        
        Depending on the specified downstream task (node, edge, or graph), this method computes the gradients 
        required for unlearning specific nodes, edges, or graph features.
        
        Args:
            unlearn_info (Any, optional): Information about the nodes, edges, or features to be unlearned. Defaults to `None`.
        
        Returns:
            tuple: Gradients computed for the entire model, specific unlearning targets, and additional components.
        """
        self._gen_train_loader()
        # self.train()
        if self.args["downstream_task"]=="node":
            res = self.get_grad(unlearn_info)
        elif self.args["downstream_task"]=="edge":
            res = self.get_grad_edge(unlearn_info)
        elif self.args["downstream_task"]=="graph":
            res = self.get_grad_graph(unlearn_info)
        return res
    
    def get_grad(self,unlearn_info=None):
        """
        Computes gradients for IDEA unlearning based on different downstream tasks.
        
        This method calculates the gradients of the loss with respect to the model parameters for the entire training set, 
        as well as for specific subsets of nodes targeted for unlearning.
        
        Args:
            unlearn_info (Any, optional): Information about the nodes to be unlearned. Defaults to `None`.
        
        Returns:
            tuple: Gradients for the entire training set, gradients for targeted nodes before unlearning, 
                   and gradients after unlearning.
        """
        grad_all, grad1, grad2 = None, None, None
        if self.target_model_name in ['GCN','SGC']:
            out1 = self.model.forward_once(self.data, self.edge_weight)
            out2 = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)

        else:
            out1 = self.model.forward_once(self.data)
            out2 = self.model.forward_once_unlearn(self.data)
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

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        # model_params = [p for p in self.model.parameters() if p.requires_grad]
        # print(loss,loss1,loss2)
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True,allow_unused=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True,allow_unused=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True,allow_unused=True)      
        # print(grad1)
        return (grad_all, grad1, grad2)
    
    def get_grad_edge(self,unlearn_info=None):
        """
        Computes gradients for edge-level unlearning.
        
        This method calculates the gradients of the loss with respect to the model parameters for edge predictions, 
        specifically targeting edges to be unlearned.
        
        Args:
            unlearn_info (Any, optional): Information about the edges to be unlearned. Defaults to `None`.
        
        Returns:
            tuple: Gradients for the entire edge set, gradients for targeted edges before unlearning, 
                   and gradients after unlearning.
        """
        grad_all, grad1, grad2 = None, None, None
        edge_index_r =None
        if self.target_model_name in ['GCN','SGC']:
            out1 = self.model.forward_once(self.data, self.edge_weight)
            out2 = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)

        else:
            out1 = self.model.forward_once(self.data)
            out2 = self.model.forward_once_unlearn(self.data)
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


        model_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
        return (grad_all, grad1, grad2)
    
    def get_loss(self, out, reduction="none"):
        """
        Computes the binary cross-entropy loss for edge predictions.
        
        This method performs negative sampling, decodes edge scores, and calculates the binary cross-entropy loss 
        between predicted logits and true labels for both positive and negative edges.
        
        Args:
            out (torch.Tensor): Model outputs used for edge prediction.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to `"none"`.
        
        Returns:
            torch.Tensor: Calculated loss value.
        """
        neg_edge_index = negative_sampling(
                edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.edge_index.size(1)
            )
        neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)
        pos_edge_label = torch.ones(self.data.edge_index.size(1),dtype=torch.float32)
        edge_logits = self.decode(z=out, pos_edge_index=self.data.edge_index,neg_edge_index=neg_edge_index)
        edge_labels = torch.cat((pos_edge_label,neg_edge_label),dim=-1)
        edge_labels = edge_labels.to(self.device)
        loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels, reduction=reduction)
        return loss

    def get_edge_loss(self,out,edge_index,reduction):
        """
        Computes the binary cross-entropy loss for specific edge predictions.
        
        This method decodes edge scores for given edge indices and calculates the binary cross-entropy loss 
        against true positive labels.
        
        Args:
            out (torch.Tensor): Model outputs used for edge prediction.
            edge_index (torch.Tensor): Indices of edges to compute loss for.
            reduction (str): Specifies the reduction to apply to the output.
        
        Returns:
            torch.Tensor: Calculated loss value for the specified edges.
        """
        out_decode = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=-1)

        edge_label = torch.ones(out_decode.shape,dtype=torch.float32,device=self.device)
        # print(out_decode.shape,edge_label.shape)
        loss = F.binary_cross_entropy_with_logits(out_decode,edge_label,reduction=reduction)
        return loss
    
    def get_grad_graph(self,unlearn_info=None):
        """
        Computes gradients for graph-level unlearning.
        
        This method calculates the gradients of the loss with respect to the model parameters for graph-level 
        predictions, specifically targeting entire graphs or graph features to be unlearned.
        
        Args:
            unlearn_info (Any, optional): Information about the graphs or features to be unlearned. Defaults to `None`.
        
        Returns:
            tuple: Gradients for the entire graph set, gradients for targeted graphs before unlearning, 
                   and gradients after unlearning.
        """
        grad_all, grad1, grad2 = None, None, None
        out1 = self.model.reason_once(self.data)
        out2 = self.model.reason_once_unlearn(self.data)
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
        
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
        return (grad_all, grad1, grad2)
    def get_graph_loss(self,out,mask):
        """
        Computes the cross-entropy loss for graph-level predictions.
        
        This method calculates the cross-entropy loss for graphs specified by the mask, aggregating node embeddings 
        to obtain graph-level representations and computing the loss against true graph labels.
        
        Args:
            out (torch.Tensor): Model outputs used for graph prediction.
            mask (np.ndarray or torch.Tensor): Boolean mask indicating which graphs to compute loss for.
        
        Returns:
            float: Aggregated cross-entropy loss for the specified graphs.
        """
        total_loss = 0
        mask = torch.tensor(mask,device=self.device)
        for gid in self.data.train_ids:
            graph_mask = (self.data.graph_id == gid) & mask
            graph_nodes = out[graph_mask]
            
            if graph_nodes.size(0) == 0:
                continue
            graph_logits = self.model.linear(graph_nodes.mean(dim=0, keepdim=True))
            # print(graph_logits, self.data.y[gid])
            graph_loss = F.cross_entropy(graph_logits, self.data.y[gid].unsqueeze(0))
            total_loss += graph_loss
        return total_loss
    
    
    def train_model_continue(self, unlearn_info=None):
        """
        Continues training the GNN model after unlearning specific targets.
        
        This method performs additional training epochs on the model, excluding the influence of unlearned nodes, 
        edges, or features as specified by `unlearn_info`. It adjusts the training mask accordingly and optimizes 
        the model parameters based on the updated loss.
        
        Args:
            unlearn_info (Any, optional): Information about the nodes, edges, or features to be excluded from training. Defaults to `None`.
        
        Returns:
            None
        """
        self.logger.info("training model continue")
        self.model.train()
        self._gen_train_loader()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=(self.model.config.lr / 1e2), weight_decay=self.model.config.decay)
        training_mask = self.data.train_mask.clone()
        if unlearn_info[0] is not np.array([]):
            training_mask[unlearn_info[0]] = False

        for epoch in tqdm(range(int(self.args['num_epochs'] * 0.1))):
            optimizer.zero_grad()
            if self.args["downstream_task"]=="graph":
                out = self.model.reason_once(self.data)
                loss = self.get_graph_loss(out,training_mask)
            else:
                if self.target_model_name in ['GCN']:
                    out = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)

                else:
                    out = self.model.reason_once_unlearn(self.data)

                loss = F.nll_loss(out[training_mask], self.data.y[training_mask])
            
            loss.backward()
            optimizer.step()

    def evaluate_unlearn_F1(self, new_parameters=None,edge_weight_unlearn=None):
        """
        Evaluates the F1 score of the GNN model after performing unlearning operations.
        
        This method updates the model parameters if new parameters are provided, performs evaluation based on the 
        specified downstream task (node, edge, or graph), and computes the corresponding F1 score to assess 
        the effectiveness of the unlearning process.
        
        Args:
            new_parameters (torch.Tensor, optional): Updated model parameters after unlearning. Defaults to `None`.
            edge_weight_unlearn (torch.Tensor, optional): Updated edge weights after unlearning. Defaults to `None`.
        
        Returns:
            float: The F1 score of the model after unlearning.
        """
        if new_parameters is not None:
            idx = 0
            for p in self.model.parameters():
                p.data = new_parameters[idx]
                idx = idx + 1

        self.model.eval()

        if self.target_model_name in ['GCN']:
            out = self.model.forward_once_unlearn(self.data, edge_weight_unlearn)
        else:
            out = self.model.reason_once_unlearn(self.data)
        self.attack_preparations["predicted_prob"] = 0

        self.attack_preparations["unlearned_feature_pre"] = 0
        if self.args["downstream_task"]=="node":
            test_f1 = f1_score(
                self.data.y[self.data['test_mask']].cpu().numpy(), 
                out[self.data['test_mask']].argmax(axis=1).cpu().numpy(), 
                average="micro"
            )
        elif self.args["downstream_task"]=="edge":
            out = self.model.reason_once_unlearn(self.data)
            test_f1 = self.eval_unlearn_edge(out)
        elif self.args["downstream_task"]=="graph":
            test_f1 = self.evaluate_graph_model()
        return test_f1
    
    @torch.no_grad()
    def evaluate_model(self):
        """
        Evaluates the GNN model's performance on both training and test datasets.
        
        This method computes the accuracy of the model on the training set and the test set by comparing 
        predicted labels with true labels. It returns a list containing the training accuracy and test accuracy.
        
        Returns:
            list: A list containing training accuracy and test accuracy.
        """
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        if self.target_model_name in ['GCN','SGC']:
            out = self.model(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            out = self.model(self.data.x, self.test_loader, self.device)

        y_true = self.data.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        results = []
        for mask in [self.data.train_mask, self.data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

        return results
    

    
    def eval_hessian(self, loss_grad, model):
        """
        Computes the Hessian matrix of the loss with respect to the model parameters.
        
        This method calculates the second-order gradients (Hessian) for the given loss gradients and model parameters.
        
        Args:
            loss_grad (tuple): Gradients of the loss with respect to the model parameters.
            model (torch.nn.Module): The GNN model for which the Hessian is computed.
        
        Returns:
            torch.Tensor: The computed Hessian matrix.
        """
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l).to(loss_grad[0].device)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian
    
    def _gen_train_loader(self):
        """
        Generates the training data loader with appropriate sampling strategies.
        
        This method creates a NeighborSampler for batching training data, performs edge normalization if required, 
        and prepares the training loader for efficient data processing during model training.
        
        Returns:
            None
        """
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
        if self.target_model_name in ['GCN','SGC']:
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
        """
        Filters the edge indices to include only those edges that connect specified nodes.
        
        This method selects edges where both endpoints are within the provided `node_indices`. Optionally, 
        it reindexes the edge indices based on the positions of the nodes in `node_indices`.
        
        Args:
            edge_index (torch.Tensor): Original edge indices of the graph.
            node_indices (list or np.ndarray): List of node indices to retain.
            reindex (bool, optional): Whether to reindex the filtered edge indices. Defaults to `True`.
        
        Returns:
            torch.Tensor or np.ndarray: Filtered (and optionally reindexed) edge indices.
        """
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
import torch.nn.functional as F
import torch
import time
import numpy as np
import copy
import os
from task.BaseTrainer import BaseTrainer
from config import root_path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import GraphSAINTNodeSampler
class GUIDETrainer(BaseTrainer):
    """
    GUIDETrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for applying the GUIDE (Guided Inductive Graph Unlearning) unlearning method.
    
    This class extends the `BaseTrainer` to implement specific training and evaluation routines required for the GUIDE unlearning methodology.
    It includes methods for training the GNN model using full-batch or mini-batch approaches, evaluating the model's performance,
    and obtaining prediction information.


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
        Initializes the GUIDETrainer with the provided configuration, logger, model, and data.

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
    

    def train_node_fullbatch(self,save=False,model_path =False):
        """
        Trains the GNN model using a full-batch approach.

        This method performs full-batch training of the GNN model, where the entire graph data is processed in each training epoch.
        It supports different base models (e.g., SIGN) and logs the training loss and F1-score at specified intervals.

        Args:
            save (bool, optional): If `True`, saves the best model weights during training. Defaults to `False`.
            
            model_path (str or bool, optional): Path to save the trained model. If `False`, no model is saved unless `save` is `True`. Defaults to `False`.

        Returns:
            tuple:
                float: Best F1-score achieved during training.
                
                float: Average training time per epoch (in seconds).
        """
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
            self.model.train()
            labels = self.data.y

            self.optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out = self.model(self.data.xs)
                loss = F.cross_entropy(out[self.data.train_mask], labels[self.data.train_mask[:labels.size(0)]])
            else:
                out = self.model(self.data.x,self.data.edge_index)
                loss = F.cross_entropy(out[self.data.train_mask], labels[self.data.train_mask[:labels.size(0)]])
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % self.args["test_freq"] == 0:
                y_pred,y = out[self.data.test_mask.cpu()],self.data.y[self.data.test_mask[:labels.size(0)].cpu()]
                y_pred = y_pred.detach().cpu().numpy()              
                y = y.detach().cpu().numpy()
                f1macro = f1_score(y, np.argmax(y_pred, axis=1), average='macro')
                
                self.logger.info('Epoch: {:03d} | Loss: {:.4f} | F1: {:.4f}'.format(epoch + 1, loss, f1macro))

    def train_node_minibatch(self,save=False,model_path=None):
        """
        Trains the GNN model using a mini-batch approach with GraphSAINT sampling.

        This method performs mini-batch training of the GNN model using the GraphSAINTNodeSampler.
        It updates the model's parameters based on sampled subgraphs and evaluates the model's performance
        on the test set at specified intervals. The method also supports saving the best model weights.

        Args:
            save (bool, optional): If `True`, saves the best model weights during training. Defaults to `False`.
            
            model_path (str, optional): Path to save the trained model. If `None`, a default path is used. Defaults to `None`.

        Returns:
            tuple:
                float: Best F1-score achieved during training.
                
                float: Average training time per epoch (in seconds).
        """
        time_sum  = 0
        best_f1 = 0
        best_w = 0
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data.num_nodes = self.data.x.size(0)
        self.data = self.data.to('cpu')
        padding = torch.full((self.data.num_nodes - self.data.y.size(0),), -1, dtype=self.data.y.dtype)  # 创建填充张量
        self.data.y = torch.cat([self.data.y, padding], dim=0) 
        loader = GraphSAINTNodeSampler(self.data,batch_size=256,num_steps = 10)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        for epoch in tqdm(range(self.args['num_epochs']), desc="BaseTraining", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            for data in loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                self.optimizer.step()
            time_sum += time.time() - start_time
            
            if (epoch + 1) % self.args["test_freq"] == 0:
                f1 = self.test_node_minibatch()
                if f1 > best_f1:
                    best_f1 = f1
                    if save:
                        best_w = copy.deepcopy(self.model.state_dict())
                self.logger.info('Epoch: {:03d} | F1 Score: {:.4f} | Loss: {:.4f}'.format(epoch + 1, f1, loss))
                
                avg_training_time = time_sum / self.args['num_epochs']
        if save:
            if not model_path:
                model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"]  +"/"+self.args["downstream_task"]+"/" + self.args["base_model"]
            os.makedirs(root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"], exist_ok=True)
            self.save_model(model_path,best_w)        
        return best_f1,avg_training_time

    @torch.no_grad()
    def test_node_minibatch(self):
        """
        Evaluates the GNN model's performance on the test set using a mini-batch approach.

        This method performs inference on the test set using the GraphSAINTNodeSampler and computes the F1-score
        based on the model's predictions. It is designed to work in a mini-batch fashion for scalability.

        Returns:
            float: The micro-averaged F1-score of the model on the test set.
        """
        self.model.eval()
        loader = GraphSAINTNodeSampler(self.data, batch_size=256) 
        all_preds = [] 
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1) 

            all_preds.append(pred[data.test_mask].cpu())
            all_labels.append(data.y[data.test_mask].cpu())

        all_preds = torch.cat(all_preds, dim=0).detach().cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()

        f1 = f1_score(all_labels, all_preds, average='micro')

        return f1
    
    @torch.no_grad()
    def prediction_info(self):
        """
        Retrieves prediction information for the test set.

        This method performs a forward pass on the entire graph data to obtain predictions for the test nodes.
        It returns both the predicted labels and the true labels for further analysis or evaluation.

        Returns:
            tuple:
                torch.Tensor: Predicted logits for the test nodes.
                
                torch.Tensor: True labels for the test nodes.
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        y_pred = self.model(self.data.x, self.data.edge_index).cpu()
        y = self.data.y.cpu()
        return y_pred[self.data.test_mask.cpu()], y[self.data.test_mask.cpu()]
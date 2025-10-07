from task.BaseTrainer import BaseTrainer
from sklearn.metrics import f1_score
import torch
from tqdm import trange, tqdm
import time
import torch.nn.functional as F
from config import root_path
import copy
import os
from torch_geometric.utils import negative_sampling
class GUKDTrainer(BaseTrainer):
    """
    GUKDTrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for applying the Graph Unlearning using Knowledge Distillation (GUKD) method.

    This class extends the `BaseTrainer` to implement specific training and evaluation routines required for the GUKD unlearning methodology.
    It includes methods for training the GNN model using full-batch or mini-batch approaches, evaluating the model's performance,
    and obtaining prediction information. The GUKD method leverages knowledge distillation techniques to effectively forget 
    specific nodes or subgraphs from the trained model while maintaining overall performance on the remaining data.
    
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
        Initializes the GUKDTrainer with the provided configuration, logger, model, and data.
    
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

    def gukd_train(self,z_t,save=False):
        """
        Initiates the training process based on the specified downstream task.
    
        This method determines whether to perform node-level or edge-level training based on the `downstream_task` argument
        and delegates the training process to the corresponding method.
    
        Args:
            z_t (torch.Tensor): Target logits or embeddings used for knowledge distillation during unlearning.
            save (bool, optional): If `True`, saves the best model weights during training. Defaults to `False`.
    
        Returns:
            tuple:
                float: Best F1-score achieved during training.
                float: Total training time (in seconds).
        """
        if self.args["downstream_task"] == 'node':
            return self.gukd_train_node(z_t,save)
        else:
            return self.gukd_train_edge(z_t,save)
    def gukd_train_node(self,z_t,save):
        """
        Trains the GNN model in preparation for node-level unlearning using knowledge distillation.
    
        This method performs node-level training where the model learns to mimic the target logits (`z_t`) for the test nodes.
        It computes the cross-entropy loss between the model's predictions and the target logits, updates the model parameters,
        and evaluates the model's F1-score at specified intervals. Optionally, it saves the best-performing model weights.
    
        Args:
            z_t (torch.Tensor): Target logits for test nodes, obtained from the teacher model before unlearning.
            save (bool, optional): If `True`, saves the best model weights based on F1-score. Defaults to `False`.
    
        Returns:
            tuple:
                float: Best F1-score achieved during training.
                float: Total training time (in seconds).
    
        """
        time_sum = 0
        best_f1 = 0
        best_w = 0
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        logits_t = F.softmax(z_t[self.data.test_mask],dim=-1).to(self.device)
        for epoch in tqdm(range(self.args['num_epochs']), desc="GUKD_Training", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out = self.model(self.data.xs)
            else:
                out = self.model(self.data.x, self.data.edge_index)
            # logits_s = F.log_softmax(out[self.data.test_mask],dim=-1)
            # loss = F.mse_loss(logits_s, logits_t)
            loss = F.cross_entropy(out[self.data.test_mask],logits_t)
            loss.backward()
            self.optimizer.step()
            time_sum += time.time() - start_time
            
            #test#
            if (epoch + 1) % self.args["test_freq"] == 0:
                f1 = self.test_node_fullbatch()
                if f1 > best_f1:
                    best_f1 = f1
                    if save:
                        best_w = copy.deepcopy(self.model.state_dict())
                self.logger.info('Epoch: {:03d} | F1 Score: {:.4f} | Loss: {:.4f}'.format(epoch + 1, f1, loss))

        avg_training_time = time_sum 
        self.logger.info("Average training time per epoch: {:.4f}s".format(avg_training_time))
        # if save:
        #     model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + self.args["base_model"]
        #     os.makedirs(root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"], exist_ok=True)
        #     self.save_model(model_path,best_w)

        import os

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(self.run)

        # Inside the megu_unlearning method, before the return statement
        unlearn_ratio = (self.args["num_unlearned_nodes"]) / self.data.num_nodes
        save_dir = os.path.join(
            "unlearned_models", "GUKD",
            self.args['dataset_name'],
            self.args["downstream_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}"
        )
        os.makedirs(save_dir, exist_ok=True)

        breakpoint()
        model_name = f"GUKD_{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}.pt"
        torch.save(self.model.state_dict(), os.path.join(save_dir, model_name))

        return best_f1,avg_training_time

    def gukd_train_edge(self,z_t,save):
        """
        Trains the GNN model in preparation for edge-level unlearning using knowledge distillation.
    
        This method performs edge-level training where the model learns to mimic the target logits (`z_t`) for edge predictions.
        It computes the mean squared error (MSE) loss between the model's decoded edge scores and the target logits, updates the model parameters,
        and evaluates the model's F1-score at specified intervals. Optionally, it saves the best-performing model weights.
    
        Args:
            z_t (torch.Tensor): Target logits for edge predictions, obtained from the teacher model before unlearning.
            save (bool, optional): If `True`, saves the best model weights based on F1-score. Defaults to `False`.
    
        Returns:
            tuple:
                float: Best F1-score achieved during training.
                float: Total training time (in seconds).
    
        """
        time_sum = 0
        best_f1 = 0
        best_w = 0
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        z_t = z_t.to(self.device)

        neg_edge_index = negative_sampling(
                edge_index=self.data.train_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_index.size(1)
            )
        logits_t = self.decode(z_t,self.data.train_edge_index,neg_edge_index).to(self.device)
        for epoch in tqdm(range(self.args['num_epochs']), desc="GUKD_Training", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out = self.model(self.data.xs)
            else:
                out = self.model(self.data.x, self.data.train_edge_index)
            
            logits_s = self.decode(out,self.data.train_edge_index,neg_edge_index)
            loss = F.mse_loss(logits_s, logits_t)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            time_sum += time.time() - start_time

            #test#
            if (epoch + 1) % self.args["test_freq"] == 0:
                f1 = self.evaluate_edge_model()
                if f1 > best_f1:
                    best_f1 = f1
                    if save:
                        best_w = copy.deepcopy(self.model.state_dict())
                self.logger.info('Epoch: {:03d} | F1 Score: {:.4f} | Loss: {:.4f}'.format(epoch + 1, f1, loss))

        avg_training_time = time_sum 
        self.logger.info("Average training time per epoch: {:.4f}s".format(avg_training_time))
        if save:
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + self.args["base_model"]
            os.makedirs(root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"], exist_ok=True)
            self.save_model(model_path,best_w)
        return best_f1,avg_training_time
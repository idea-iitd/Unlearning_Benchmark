from task.BaseTrainer import BaseTrainer
import torch
from tqdm import tqdm
import time
import torch.nn.functional as F
from config import root_path
import copy
import os
from torch_geometric.utils import negative_sampling
class D2DGNTrainer(BaseTrainer):
    """
    D2DGNTrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for applying the Distills to Delete in GNN (D2DGN) unlearning method.
    
    This class extends the BaseTrainer to implement specific training and evaluation routines 
    required for the D2DGN methodology. It includes methods for training node-level, edge-level, 
    and graph-level tasks, evaluating model performance, and handling model persistence.
    
    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, 
                     training hyperparameters, unlearning settings, and other relevant settings.
    
        logger (logging.Logger): Logger object used to log training progress, metrics, 
                                 and other important information.
    
        model (torch.nn.Module): The neural network model that will be trained.
    
        data (dict): A dictionary containing datasets for training, validation, and testing.
                     Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.
    
        device (torch.device): The computation device (CPU or GPU) on which the model 
                               and data are loaded for training.
    
        alpha (float): Weighting factor used in the loss function to balance between 
                       preserver and destroyer knowledge distillation.
    """
    def __init__(self, args, logger, model, data,alpha=0.5):
        """
        Initializes the D2DGNTrainer with the provided configuration, logger, model, data, and alpha.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications, 
                        training hyperparameters, unlearning settings, and other relevant settings.
                        
            logger (logging.Logger): Logger object used to log training progress, metrics, 
                                     and other important information.
                        
            model (torch.nn.Module): The neural network model that will be trained.
                        
            data (dict): A dictionary containing datasets for training, validation, and testing.
                         Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.
                         
            alpha (float, optional): Weighting factor for balancing preserver and destroyer losses. 
                                     Defaults to 0.5.
        """
        super().__init__(args, logger, model, data)
        self.alpha = alpha

    def d2dgn_train(self,preserver_knowledge,destroyer_knowledge,loss_fn,save=False):
        """
        Trains the model based on the downstream task.

        This method delegates the training process to specific methods depending on the 
        downstream task, which can be node-level, edge-level, or graph-level.

        Args:
            preserver_knowledge (torch.Tensor): Knowledge to be preserved during unlearning.
            
            destroyer_knowledge (torch.Tensor): Knowledge to be destroyed during unlearning.
            
            loss_fn (str): The type of loss function to use ("KL" for Kullback-Leibler divergence, 
                           "MSE" for Mean Squared Error).
            
            save (bool, optional): Whether to save the best model weights during training. 
                                   Defaults to False.
        
        Returns:
            tuple: A tuple containing the best F1 score achieved during training and 
                   the average training time.
        """
        if self.args["downstream_task"] == 'node':
            return self.d2dgn_train_node(preserver_knowledge,destroyer_knowledge,loss_fn,save)
        elif self.args["downstream_task"] == 'edge':
            return self.d2dgn_train_edge(preserver_knowledge,destroyer_knowledge,loss_fn,save)
        elif self.args["downstream_task"]=="graph":
            return self.d2dgn_train_graph(preserver_knowledge,destroyer_knowledge,loss_fn,save)
    def d2dgn_train_node(self,preserver_knowledge,destroyer_knowledge,loss_fn,save=False):
        """
        Trains the model for node-level tasks.

        This method handles the training loop for node classification tasks, including loss 
        computation, backpropagation, optimizer steps, evaluation, and model saving based 
        on validation F1 score improvements.

        Args:
            preserver_knowledge (torch.Tensor): Knowledge to be preserved during unlearning.
            
            destroyer_knowledge (torch.Tensor): Knowledge to be destroyed during unlearning.
            
            loss_fn (str): The type of loss function to use ("KL" for Kullback-Leibler divergence, 
                           "MSE" for Mean Squared Error).
            
            save (bool, optional): Whether to save the best model weights during training. 
                                   Defaults to False.
        
        Returns:
            tuple: A tuple containing the best F1 score achieved during training and 
                   the average training time.
        """
        time_sum = 0
        best_f1 = 0
        best_w = 0
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        self.alpha = torch.tensor(self.alpha,dtype=torch.float,device=self.device)
        for epoch in tqdm(range(self.args['num_epochs']), desc="D2DGNTraining", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if loss_fn== "KL":
                if self.args["base_model"] == "SIGN":
                    out = self.model(self.data.xs)
                else:
                    out = self.model(self.data.x, self.data.edge_index)
                out = F.log_softmax(out,dim=-1)
                # print(out[self.data.keep_mask].shape)
                loss1 = F.kl_div(out[self.data.keep_mask],preserver_knowledge[self.data.keep_mask],reduction='batchmean')
                loss2 = F.kl_div(out[self.data.unlearn_mask],destroyer_knowledge[self.data.unlearn_mask],reduction='batchmean')
            elif loss_fn == "MSE":
                if self.args["base_model"] == "SIGN":
                    out = self.model(self.data.xs,return_all_emb=True)
                else:
                    out = self.model(self.data.x, self.data.edge_index,return_all_emb=True)

                loss_func = torch.nn.MSELoss(reduction="sum")
                loss1 = 0
                loss2 = 0
                for conv_out,know in zip(out,preserver_knowledge):
                    # conv_out=F.softmax(conv_out,dim=-1)
                    
                    loss1 += loss_func(conv_out[self.data.keep_mask],know[self.data.keep_mask])
                for conv_out,know in zip(out,destroyer_knowledge):
                    # conv_out=F.softmax(conv_out,dim=-1)
                    loss2 += loss_func(conv_out[self.data.unlearn_mask],know[self.data.unlearn_mask])
            
            loss = loss1 * self.alpha + loss2 * (1 - self.alpha)
            # print(loss)
            loss.backward(retain_graph=True)
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
        if save:
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + self.args["base_model"]
            os.makedirs(root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"], exist_ok=True)
            self.save_model(model_path,best_w)
        return best_f1,avg_training_time
    

    def d2dgn_train_edge(self,preserver_knowledge,destroyer_knowledge,loss_fn,save=False):
        """
        Trains the model for edge-level tasks.

        This method handles the training loop for edge classification tasks, including loss 
        computation, backpropagation, optimizer steps, evaluation, and model saving based 
        on validation F1 score improvements.

        Args:
            preserver_knowledge (torch.Tensor): Knowledge to be preserved during unlearning.
            
            destroyer_knowledge (torch.Tensor): Knowledge to be destroyed during unlearning.
            
            loss_fn (str): The type of loss function to use ("KL" for Kullback-Leibler divergence, 
                           "MSE" for Mean Squared Error).
            
            save (bool, optional): Whether to save the best model weights during training. 
                                   Defaults to False.
        
        Returns:
            tuple: A tuple containing the best F1 score achieved during training and 
                   the average training time.
        """
        time_sum = 0
        best_f1 = 0
        best_w = 0
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr/10, weight_decay=self.model.config.decay)
        self.alpha = torch.tensor(self.alpha,dtype=torch.float,device=self.device)
        neg_edge_index = negative_sampling(
                edge_index=self.data.train_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_index.size(1)
            )
        if loss_fn == "KL":
            logits_preserver = self.decode(preserver_knowledge,self.data.keep_edge_index,neg_edge_index)
            logits_preserver = torch.sigmoid(logits_preserver)
            logits_preserver = torch.clamp(logits_preserver, min=1e-5, max=1-1e-5)
            logits_preserver = torch.stack([logits_preserver,1-logits_preserver],dim=1)

            logits_destroyer = self.decode(destroyer_knowledge,self.data.unlearn_edge_index,neg_edge_index)
            logits_destroyer = torch.sigmoid(logits_destroyer)
            logits_destroyer = torch.clamp(logits_destroyer, min=1e-5, max=1-1e-5)
            logits_destroyer = torch.stack([logits_destroyer,1-logits_destroyer],dim=1)
        else:
            logits_preserver = []
            logits_destroyer = []
            for i,know in enumerate(preserver_knowledge):
                logits_preserver.append(self.decode(know,self.data.keep_edge_index,neg_edge_index))
            for i,know in enumerate(destroyer_knowledge):
                logits_destroyer.append(self.decode(know,self.data.unlearn_edge_index,neg_edge_index)) 
        for epoch in tqdm(range(self.args['num_epochs']), desc="D2DGNTraining", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if loss_fn== "KL":
                if self.args["base_model"] == "SIGN":
                    out = self.model(self.data.xs)
                else:
                    
                    out = self.model(self.data.x, self.data.train_edge_index)

                logits_keep = self.decode(out,self.data.keep_edge_index,neg_edge_index)
                logits_keep = torch.sigmoid(logits_keep)
                logits_keep = torch.clamp(logits_keep, min=1e-5, max=1-1e-5)
                logits_keep = torch.log(torch.stack([logits_keep,1-logits_keep],dim=1))
                # print(logits_keep)

                logits_unlearn = self.decode(out,self.data.unlearn_edge_index,neg_edge_index)
                logits_unlearn = torch.sigmoid(logits_unlearn)
                logits_unlearn = torch.clamp(logits_unlearn, min=1e-5, max=1-1e-5)
                logits_unlearn = torch.log(torch.stack([logits_unlearn,1-logits_unlearn],dim=1))
                
                loss1 = F.kl_div(logits_keep,logits_preserver,reduction="batchmean")
                loss2 = F.kl_div(logits_unlearn,logits_destroyer,reduction="batchmean")
            elif loss_fn == "MSE":
                if self.args["base_model"] == "SIGN":
                    out = self.model(self.data.xs,return_all_emb=True)
                else:
                    out = self.model(self.data.x, self.data.edge_index,return_all_emb=True)

                loss_func = torch.nn.MSELoss(reduction="mean")
                loss1 = 0
                loss2 = 0
                for conv_out,logit_preserver in zip(out,logits_preserver):
                    logit_keep = self.decode(conv_out,self.data.keep_edge_index,neg_edge_index)
                    loss1 += loss_func(logit_keep,logit_preserver)
                for conv_out,logit_destroyer in zip(out,logits_destroyer):
                    logit_unlearn = self.decode(conv_out,self.data.unlearn_edge_index,neg_edge_index)
                    loss2 += loss_func(logit_unlearn,logit_destroyer)
            # print(loss1,loss2)
            loss = loss1 * self.alpha + loss2 * (1 - self.alpha)
            
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
        # model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + self.args["base_model"]
        # os.makedirs(root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"], exist_ok=True)
        # self.save_model(model_path,best_w)
        return best_f1,avg_training_time
    
        

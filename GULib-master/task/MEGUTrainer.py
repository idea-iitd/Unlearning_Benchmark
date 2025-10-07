import torch.nn.functional as F
import time
import torch
import copy
import numpy as np
from tqdm import tqdm
from task import BaseTrainer
from config import root_path
from sklearn.metrics import f1_score, accuracy_score,recall_score,roc_auc_score
from torch_geometric.transforms import SIGN
from utils.utils import sparse_mx_to_torch_sparse_tensor,normalize_adj,propagate,criterionKD,calc_f1
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import CorrectAndSmooth
class GATE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = torch.nn.Linear(dim, dim)

    def forward(self, x):
        t = x.clone()
        return self.lr(t)

class MEGUTrainer(BaseTrainer):
    """
    MEGUTrainer class for performing Mutual Evolution Graph Unlearning (MEGU) on Graph Neural Networks (GNNs).

    This class manages the unlearning process by:
        - Targeting specific nodes, edges, or features for unlearning.

        - Applying a gating mechanism to modify model parameters accordingly.

        - Ensuring that the model retains its predictive performance on non-targeted data.

        - Evaluating the effectiveness of the unlearning process through appropriate metrics.

    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, 
                    training hyperparameters, unlearning settings, and other relevant settings.
        
        logger (logging.Logger): Logger object used to log training progress, metrics, 
                                 and other important information.
        
        model (torch.nn.Module): The neural network model that will undergo unlearning.
        
        data (torch_geometric.data.Data): The dataset containing edge and node information 
                                          for training, validation, and testing.
        
        device (torch.device): The computation device (CPU or GPU) on which the model 
                               and data are loaded for training and unlearning.
        
        temp_node (list or torch.Tensor): Indices of nodes targeted for unlearning.
        
        neighbor_khop (list or torch.Tensor): Indices of neighboring nodes within k hops of the targeted nodes.
        
        attack_preparations (dict): Dictionary to store preparations related to attacks or evaluations.
        
        loss_all (Any): Placeholder for storing loss values or related information.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the MEGUTrainer with the provided configuration, logger, model, and data.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications, 
                        training hyperparameters, unlearning settings, and other relevant settings.

            logger (logging.Logger): Logger object used to log training progress, metrics, 
                                     and other important information.
                                     
            model (torch.nn.Module): The neural network model that will undergo unlearning.

            data (torch_geometric.data.Data): The dataset containing edge and node information 
                                              for training, validation, and testing.
        """
        super().__init__(args, logger, model, data)

    def megu_unlearning(self,temp_node,neighbor_khop,run=0):
        """
        Performs the Mutual Evolution Graph Unlearning (MEGU) process on the GNN model.
        
        This method targets specific nodes and their neighboring nodes within k hops for unlearning.
        It utilizes a gating mechanism (`GATE`) to adjust the model's parameters, effectively removing the 
        influence of the targeted nodes and their neighbors. The unlearning process involves multiple epochs 
        of training where the model learns to forget the specified elements while retaining performance on 
        the remaining data.

        Args:
            temp_node (list or torch.Tensor): Indices of nodes targeted for unlearning.

            neighbor_khop (list or torch.Tensor): Indices of neighboring nodes within k hops of the targeted nodes.

        Returns:
            tuple: A tuple containing:
                - unlearn_time (float): Total time taken to perform the unlearning process.

                - test_f1 (float): Evaluation metric (F1 score for node-level tasks or ROC AUC score for edge-level tasks) 
                                    on the test dataset after unlearning.
        """
        self.temp_node = temp_node
        self.neighbor_khop = neighbor_khop
        operator = GATE(self.data.num_classes).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        self.data = self.data.to(self.device)
            

        with torch.no_grad():
            self.model.eval()
            if self.args["base_model"] == "SIGN":
                preds = self.model(self.data.xs)
            else:
                preds = self.model(self.data.x,self.data.edge_index)
                preds_edge = preds
            if self.args['dataset_name'] == 'ppi':
                preds = torch.sigmoid(preds).ge(0.5)
                preds = preds.type_as(self.data.y)
            else:           
                preds = torch.argmax(preds, axis=1).type_as(self.data.y)
            
        if self.args["base_model"] == "SIGN":
            self.data.x = self.data.x_unlearn
            self.data.edge_index = self.data.edge_index_unlearn
            self.data = SIGN(self.args["GNN_layer"])(self.data)
            self.data.xs = [self.data.x] + [self.data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
            self.data.xs = torch.stack(self.data.xs).to('cuda')
            self.data.xs = self.data.xs.transpose(0,1)

        start_time = time.time()
        for epoch in tqdm(range(self.args["unlearning_epochs"]),desc="Unlearning"):
            self.model.train()
            operator.train()
            optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out_ori =  self.model(self.data.xs)
            else:
                out_ori = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            out = operator(out_ori)
            # if self.args["downstream_task"]=="node":
            if self.args['dataset_name'] == 'ppi':
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.binary_cross_entropy_with_logits(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.binary_cross_entropy_with_logits(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
            else:
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.cross_entropy(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.cross_entropy(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
            
            loss = self.args['kappa'] * loss_u + loss_r

            loss.backward()
            optimizer.step()
        unlearn_time = time.time() - start_time
        self.model.eval()
        if self.args["base_model"] == "SIGN":
            test_out =  self.model(self.data.xs)
        else:
            # test_out = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            test_out = self.model(self.data.x, self.data.edge_index)
        if self.args['dataset_name'] == 'ppi':
            out = torch.sigmoid(test_out)
        else:
            # out = self.correct_and_smooth(F.softmax(test_out, dim=-1), preds)
            out = F.softmax(test_out, dim=-1)
        if self.args["downstream_task"]=="node":
            y_hat = out.cpu().detach().numpy()
            y = self.data.y.cpu()
            test_mask = self.data.test_mask.cpu().numpy()
            y_pred_classes = y_hat.argmax(axis=1)
            # breakpoint()
            test_accuracy = accuracy_score(y_pred_classes[test_mask], y[test_mask])
            if self.args['dataset_name'] == 'ppi':
                test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
            else:
                test_f1 = calc_f1(y, y_hat, self.data.test_mask)
                
        elif self.args["downstream_task"]=="edge":
            neg_edge_index = negative_sampling(
                edge_index=self.data.test_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.test_edge_index.size(1)
            )
            edge_pred_logits = self.decode(z=out, pos_edge_index=self.data.test_edge_index,neg_edge_index=neg_edge_index)
            edge_pred_logits = torch.sigmoid(edge_pred_logits)
            edge_pred_logits = edge_pred_logits.cpu()
            edge_pred = torch.where(edge_pred_logits > 0.5, torch.tensor(1), torch.tensor(0))
            
            # edge_pred = torch.argmax(edge_pred_logits)
            pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
            neg_edge_labels = torch.zeros(neg_edge_index.size(1),dtype=torch.float32)
            edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
            test_f1 = roc_auc_score(edge_labels.cpu(), edge_pred.cpu())
            test_accuracy = accuracy_score(edge_labels.cpu(), edge_pred.cpu())
        self.logger.info("Unlearning time: %.4f, Test F1: %.4f, Test accuracy: %.4f"%(unlearn_time, test_f1, test_accuracy))

        # Add this import at the top of your file if not already present
        import os

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(run)

        base_model_str = ""
        if self.args['base_model']!="GCN":  
            base_model_str = "_" + self.args['base_model']
        # breakpoint()
        # edge_str=""
        # if self.args['unlearn_task']=="edge":
        #     edge_str="_unlearning_edge"

        # Inside the megu_unlearning method, before the return statement
        # unlearn_ratio = len(self.temp_node) / self.data.num_nodes
        unlearn_ratio = self.args["unlearn_ratio"]
        save_dir = os.path.join(
            "unlearned_models", "MEGU",
            self.args['dataset_name'],
            self.args["unlearn_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}"
        )
        os.makedirs(save_dir, exist_ok=True)

        model_name = f"MEGU_{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}{base_model_str}.pt"
        torch.save(self.model.state_dict(), os.path.join(save_dir, model_name))
        print("Unlearned model saved at: ",os.path.join(save_dir, model_name))
        
        return unlearn_time, test_f1
    
    
    def correct_and_smooth(self, y_soft, preds):
        """
        Applies correction and smoothing to the soft predictions using the CorrectAndSmooth method.
        
        Args:
            y_soft (torch.Tensor): The soft predictions (e.g., probabilities) from the model.
            
            preds (torch.Tensor): The original predictions from the model before correction and smoothing.

        Returns:
            torch.Tensor: The corrected and smoothed predictions.
        """
        pos = CorrectAndSmooth(num_correction_layers=80, correction_alpha=self.args['alpha1'],
                               num_smoothing_layers=80, smoothing_alpha=self.args['alpha2'],
                               autoscale=False, scale=1.)

        y_soft = pos.correct(y_soft, preds[self.data.train_mask], self.data.train_mask,
                                  self.data.edge_index_unlearn)
        y_soft = pos.smooth(y_soft, preds[self.data.train_mask], self.data.train_mask,
                                 self.data.edge_index_unlearn)
        
        return y_soft

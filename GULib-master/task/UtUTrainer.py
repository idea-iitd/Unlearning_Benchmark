import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from utils.utils import member_infer_attack,get_link_labels
from task.BaseTrainer import BaseTrainer
class UtUTrainer(BaseTrainer):
    """
    UtUTrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for Unlink to Unlearn (UtU) method.

    The `UtUTrainer` class manages the training process of GNN models, incorporating mechanisms for unlearning specific data points 
    and evaluating the model's resilience against membership inference attacks. It supports both full-batch and mini-batch 
    training approaches and facilitates the evaluation of model performance on node-level and edge-level tasks.

    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, training hyperparameters, 
                    unlearning settings, and other relevant settings.

        trainer_log (dict): Dictionary for logging training metrics and attack evaluations. Contains keys such as 
                            'unlearning_model', 'dataset', and 'log'.

        logit_all_pair (Any): Placeholder for storing logits of all node pairs, used in evaluation and attack assessments.

        df_pos_edge (list): List to store positive edge indices for differential privacy or unlearning purposes.

        device (torch.device): The computation device (CPU or GPU) on which the model and data are loaded for training and evaluation.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the UtUTrainer with the provided configuration.

        Sets up the necessary attributes for training and evaluating the GNN model, including logging structures and device setup.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications, training hyperparameters, 
                        unlearning settings, and other relevant settings.
        """
        super().__init__(args, logger, model, data)
        self.args = args
        self.trainer_log = {
            'unlearning_model': args["unlearning_model"],
            'dataset': args["dataset_name"],
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def train_UTU_model(self, optimizer, logits_ori=None, attack_model_all=None,
                        attack_model_sub=None):
        """
        Trains the GNN model using a full-batch training approach.

        Resets the model's parameters, initializes the optimizer, and iterates through the training epochs. It performs 
        member inference attacks before unlearning, saves model checkpoints, and logs the best performance metrics.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters during training.
            
            logits_ori (torch.Tensor, optional): The original logits from the model before any unlearning or attack processes. Defaults to `None`.
            
            attack_model_all (Any, optional): The attack model used for evaluating membership inference attacks on all data. Defaults to `None`.
            
            attack_model_sub (Any, optional): The attack model used for evaluating membership inference attacks on a subset of data. Defaults to `None`.
        
        Returns:
            None
        """
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

        best_metric = 0
        loss_fct = nn.MSELoss()

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(self.model, attack_model_all, self.data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(self.model, attack_model_sub, self.data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before
        z = self.model(self.data.x, self.data.train_pos_edge_index[:, self.data.dr_mask])
        # Save
        ckpt = {
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
        torch.save(z, os.path.join(self.args["checkpoint_dir"], 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric

    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None,
                        attack_model_sub=None):
        """
        Trains the GNN model using a mini-batch training approach.

        Currently, this method mirrors the functionality of `train_fullbatch` but is structured for potential mini-batch extensions.

        Args:
            model (torch.nn.Module): The GNN model to be trained.
            
            data (torch_geometric.data.Data): The dataset containing node features, edge indices, and other relevant information.
            
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters during training.
            
            args (dict): Configuration parameters, including model type, dataset specifications, training hyperparameters, 
                        unlearning settings, and other relevant settings.
            
            logits_ori (torch.Tensor, optional): The original logits from the model before any unlearning or attack processes. Defaults to `None`.
            
            attack_model_all (Any, optional): The attack model used for evaluating membership inference attacks on all data. Defaults to `None`.
            
            attack_model_sub (Any, optional): The attack model used for evaluating membership inference attacks on a subset of data. Defaults to `None`.
        
        Returns:
            None
        """
        start_time = time.time()
        best_metric = 0

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args["checkpoint_dir"], 'model_best.pt'))
        # torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        """
        Evaluates the GNN model's performance and its resilience against membership inference attacks.

        Loads the best model checkpoint, performs evaluations on node-level or edge-level tasks, and conducts 
        membership inference attacks post-unlearning to assess privacy preservation.

        Args:
            model (torch.nn.Module): The GNN model to be evaluated.
            
            data (torch_geometric.data.Data): The dataset containing node features, edge indices, and other relevant information.
            
            model_retrain (torch.nn.Module, optional): The retrained model after unlearning, used for comparison. Defaults to `None`.
            
            attack_model_all (Any, optional): The attack model used for evaluating membership inference attacks on all data. Defaults to `None`.
            
            attack_model_sub (Any, optional): The attack model used for evaluating membership inference attacks on a subset of data. Defaults to `None`.
            
            ckpt (str, optional): Specifies which checkpoint to load for evaluation. Defaults to `'best'`.
        
        Returns:
            tuple: A tuple containing evaluation metrics:
                   - loss (float): The loss on the test dataset.

                   - dt_auc (float): AUC score for direct testing.

                   - dt_aup (float): Average precision score for direct testing.

                   - df_auc (float): AUC score for differential privacy testing.

                   - df_aup (float): Average precision score for differential privacy testing.

                   - df_logit (Any): Logits for differential privacy testing.

                   - logit_all_pair (Any): Logits for all node pairs (if applicable).

                   - test_log (dict): Dictionary containing additional test metrics.
        """
        if ckpt == 'best' and self.args["unlearning_model"] != 'simple':  # Load best ckpt
            ckpt = torch.load(os.path.join(self.args["checkpoint_dir"], 'model_best.pt'), map_location=self.device)
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args["dataset"]:
            pred_all = False
        else:
            pred_all = True
        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test',
                                                                                             pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        # self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['auc_sum'] = dt_auc + df_auc
        self.trainer_log['aup_sum'] = dt_aup + df_aup
        self.trainer_log['auc_gap'] = abs(dt_auc - df_auc)
        self.trainer_log['aup_gap'] = abs(dt_aup - df_aup)

        # if model_retrain is not None:    # Deletion
        #     self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
        # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in
                                                        zip(self.trainer_log['mi_logit_all_after'],
                                                            self.trainer_log['mi_logit_all_before'])])
            test_log['mi_sucrate'], test_log['mi_ratio'] = mi_sucrate_all_after, self.trainer_log['mi_ratio_all']
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
            self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after

            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in
                                                        zip(self.trainer_log['mi_logit_all_after'],
                                                            self.trainer_log['mi_logit_all_before'])])
            self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in
                                                        zip(self.trainer_log['mi_logit_sub_after'],
                                                            self.trainer_log['mi_logit_sub_before'])])
            print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'],
                  self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
            print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])
        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log

    def eval(self, model, data, stage='val', pred_all=False):
        """
        Evaluates the GNN model's performance on a specified stage (validation or test).

        Computes loss, AUC scores, and other relevant metrics for both direct testing (DT) and differential privacy testing (DF).
        Optionally logs logits for all node pairs if `pred_all` is enabled.

        Args:
            model (torch.nn.Module): The GNN model to be evaluated.
            
            data (torch_geometric.data.Data): The dataset containing node features, edge indices, and other relevant information.
            
            stage (str, optional): The evaluation stage, either 'val' or 'test'. Defaults to `'val'`.
            
            pred_all (bool, optional): Flag indicating whether to predict on all node pairs. Defaults to `False`.
        
        Returns:
            tuple: A tuple containing evaluation metrics:
                   - loss (float): The binary cross-entropy loss on the specified stage.

                   - dt_auc (float): AUC score for direct testing.

                   - dt_aup (float): Average precision score for direct testing.

                   - df_auc (float): AUC score for differential privacy testing.

                   - df_aup (float): Average precision score for differential privacy testing.

                   - df_logit (list): Logits for differential privacy testing.

                   - logit_all_pair (Any): Logits for all node pairs (if applicable).

                   - log (dict): Dictionary containing detailed evaluation metrics.
        """
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = negative_sampling(
            edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.test_pos_edge_index.size(1)
        )

        if self.args["eval_on_cpu"]:
            model = model.to('cpu')

        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        logits = self.decode(z, pos_edge_index, neg_edge_index)
        label = get_link_labels(pos_edge_index, neg_edge_index)
        print(logits,label)
        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        if self.args["unlearning_model"] in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = self.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []

            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(5):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)

            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = self.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

                logit = df_logit + pos_logit
                label = [0] * len(df_logit) + [1] * len(pos_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))

            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)

        else:
            df_auc = np.nan
            df_aup = np.nan


        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        if self.args["eval_on_cpu"]:
            model = model.to(self.device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, None, log
    
    def decode(self, z, pos_edge_index, neg_edge_index=None):
        """
        Decodes edge logits from node embeddings for link prediction tasks.

        Computes the dot product between pairs of node embeddings to predict the existence of edges. If negative edge 
        indices are provided, it concatenates the positive and negative edge logits.

        Args:
            z (torch.Tensor): Node embeddings obtained from the GNN model.
            
            pos_edge_index (torch.Tensor): Edge indices for positive (existing) edges.
            
            neg_edge_index (torch.Tensor, optional): Edge indices for negative (non-existing) edges. Defaults to `None`.
        
        Returns:
            torch.Tensor: Logits indicating the presence of edges. Higher values suggest a higher likelihood of edge existence.
        """
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits
    
    

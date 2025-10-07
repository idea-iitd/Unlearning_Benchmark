import torch.nn.functional as F
import torch
import os
import time
import numpy as np
import copy
from task.BaseTrainer import BaseTrainer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,recall_score,average_precision_score
from attack.Attack_methods.GNNDelete_MIA import member_infer_attack
from torch_geometric.utils import negative_sampling
from utils.utils import get_loss_fct, trange, Reverse_CE
from config import root_path

class GNNDeleteTrainer(BaseTrainer):
    """
    GNNDeleteTrainer class for training and evaluating GNNs in preparation for applying the GNNDelete method.

    This class extends the BaseTrainer to implement specific training and evaluation routines 
    required for the GNNDelete methodology.It handles training procedures for different downstream tasks such as node classification and edge prediction,
    including methods to evaluate model performance before and after deletion operations. The class also integrates 
    member inference attacks to assess the privacy implications of the unlearning process.

    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, 
                     training hyperparameters, unlearning settings, and other relevant settings.
        
        logger (logging.Logger): Logger object used to log training progress, metrics, 
                                 and other important information.
        
        model (torch.nn.Module): The neural network model to be trained and evaluated.
        
        data (torch_geometric.data.Data): The dataset containing graph information for training, 
                                         validation, and testing.
        
        df_pos_edge (list): A list to store positive edges for defensive purposes during training.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the GNNDeleteTrainer with the provided configuration, logger, model, and data.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications,
                        training hyperparameters, unlearning settings, and other relevant settings.
                        
            logger (logging.Logger): Logger object used to log training progress, metrics,
                                    and other important information.
                        
            model (torch.nn.Module): The neural network model to be trained and evaluated.
                        
            data (torch_geometric.data.Data): The dataset containing graph information for training,
                                            validation, and testing.
        """
        super().__init__(args, logger, model, data)
        self.df_pos_edge = []

    def gnndelete_train(self, avg_time, run, optimizer, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        """
        Initiates the training process based on the specified downstream task.

        This method delegates the training process to specialized methods depending on whether the
        downstream task is node classification or edge prediction.

        Args:
            avg_time (dict): A dictionary to store average training times.
            
            run (int): The current run or experiment index.
            
            optimizer (list of torch.optim.Optimizer): A list containing optimizers for different parts of the model.
            
            logits_ori (torch.Tensor, optional): Original logits before deletion. Defaults to None.
            
            attack_model_all (torch.nn.Module, optional): Model used for member inference attacks on all data. Defaults to None.
            
            attack_model_sub (torch.nn.Module, optional): Model used for member inference attacks on a subset of data. Defaults to None.

        Returns:
            None
        """
        if self.args["downstream_task"]=="node":
            return self.train_node_fullbatch_del(avg_time,run,optimizer, logits_ori,attack_model_all,attack_model_sub)
        elif self.args["downstream_task"]=="edge":
            return self.gnndelete_train_edge(avg_time,run,optimizer, logits_ori,attack_model_all,attack_model_sub)

    def gnndelete_train_edge(self, avg_time, run, optimizer, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        """
        Trains the GNN model for edge-level tasks.

        This method handles the training loop for edge prediction tasks, including loss computation,
        backpropagation, optimizer steps, and periodic evaluation. It also integrates member inference
        attacks to assess privacy before the unlearning process.

        Args:
            avg_time (dict): A dictionary to store average training times.
            
            run (int): The current run or experiment index.
            
            optimizer (list of torch.optim.Optimizer): A list containing optimizers for different parts of the model.
            
            logits_ori (torch.Tensor, optional): Original logits before deletion. Defaults to None.
            
            attack_model_all (torch.nn.Module, optional): Model used for member inference attacks on all data. Defaults to None.
            
            attack_model_sub (torch.nn.Module, optional): Model used for member inference attacks on a subset of data. Defaults to None.

        Returns:
            None
        """
        self.model = self.model.to('cuda')
        self.data = self.data.to('cuda')
        self.trainer_log = {
            'unlearning_model': self.args["unlearning_model"],
            'dataset': self.args["dataset_name"],
            'log': []}
        best_metric = 0

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(self.model, attack_model_all, self.data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(self.model, attack_model_sub, self.data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before


        non_df_node_mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.data.x.device)
        non_df_node_mask[self.data.directed_df_edge_index.flatten().unique()] = False

        self.data.sdf_node_1hop_mask_non_df_mask = self.data.sdf_node_1hop_mask & non_df_node_mask
        self.data.sdf_node_2hop_mask_non_df_mask = self.data.sdf_node_2hop_mask & non_df_node_mask
        
        # Original node embeddings
        with torch.no_grad():
            if self.args["base_model"] in ["SIGN","SGC","S2GC"]:
                z1_ori, z2_ori = self.model.get_original_embeddings(self.data.x, return_all_emb=True)
            else:
                z1_ori, z2_ori = self.model.get_original_embeddings(self.data.x, self.data.edge_index[:, self.data.dr_mask], return_all_emb=True)

        loss_fct = get_loss_fct(self.args["loss_fct"])

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=self.data.edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.df_mask.sum())

        for epoch in trange(self.args['num_epochs'], desc='Unlerning'):
            self.model.train()

            start_time = time.time()
            z1, z2 = self.model(self.data.x, self.data.edge_index[:, self.data.sdf_mask], return_all_emb=True)

            # Randomness
            pos_edge = self.data.edge_index[:, self.data.df_mask]
            # neg_edge = torch.randperm(data.num_nodes)[:pos_edge.view(-1).shape[0]].view(2, -1)

            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)

            # Local causality
            loss_l1 = loss_fct(z1[self.data.sdf_node_1hop_mask_non_df_mask], z1_ori[self.data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[self.data.sdf_node_2hop_mask_non_df_mask], z2_ori[self.data.sdf_node_2hop_mask_non_df_mask])


            # Total loss
            '''both_all, both_layerwise, only2_layerwise, only2_all, only1'''
            loss_l = loss_l1 + loss_l2
            loss_r = loss_r1 + loss_r2

            loss1 = self.args["alpha"] * loss_r1 + (1 - self.args["alpha"]) * loss_l1
            loss1.backward(retain_graph=True)
            optimizer[0].step()
            optimizer[0].zero_grad()

            loss2 = self.args["alpha"] * loss_r2 + (1 - self.args["alpha"]) * loss_l2
            loss2.backward(retain_graph=True)
            optimizer[1].step()
            optimizer[1].zero_grad()

            loss = loss1 + loss2

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args["test_freq"] == 0:
                valid_loss, dt_f1,dt_acc ,_,_,_,_,_,_, valid_log = self.eval_edge('test')

                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time,
                }
                
                for log in [train_log, valid_log]:
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                if dt_acc + dt_f1 > best_metric:
                    best_metric = dt_acc + dt_f1
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': self.model.state_dict(),
                        # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
                    }
                    torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
        avg_time[run] = epoch_time
        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in self.model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_final.pt'))


    def train_node_fullbatch(self,save=False,model_path=False):
        """
        Trains the GNN model for node-level tasks using full-batch training.

        This method manages the training loop for node classification tasks, including loss computation,
        backpropagation, optimizer steps, and periodic evaluation. It tracks the best F1 score and
        optionally saves the best model weights.

        Args:
            save (bool, optional): Whether to save the best model weights during training.
                                    Defaults to False.
            
            model_path (str, optional): The path to save the model. If False, a default path is used.
                                        Defaults to False.

        Returns:
            tuple:
                best_f1 (float): The best F1 score achieved during training.
                avg_training_time (float): The average training time per epoch in seconds.
        """
        time_sum = 0
        best_f1 = 0
        best_w = 0
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.lr, weight_decay=self.model.config.decay)
        for epoch in tqdm(range(self.args['num_epochs']), desc="BaseTraining", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] in ["SIGN","SGC","S2GC"]:
                out = self.model(self.data.features_pre)
            else:
                out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask]).to(self.device)
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

        avg_training_time = time_sum / self.args['num_epochs']
        self.logger.info("Average training time per epoch: {:.4f}s".format(avg_training_time))
        if save:
            if not model_path:
                model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"]  +"/"+self.args["downstream_task"]+"/" + self.args["base_model"]
            os.makedirs(root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"], exist_ok=True)
            self.save_model(model_path,best_w)
        return best_f1,avg_training_time
    

    @torch.no_grad()
    def test_node_fullbatch(self):
        """
        Tests the GNN model for node-level tasks.

        This method evaluates the trained model on the test set for node classification tasks,
        computing the F1 score based on the model's predictions.

        Returns:
            float:
                f1 (float): The F1 score on the test set.
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] in ["SIGN","SGC","S2GC"]:
            y_pred = self.model(self.data.features_pre).cpu()
        else:
            y_pred = self.model(self.data.x, self.data.edge_index).cpu()
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        f1 = f1_score(y[self.data.test_mask.cpu()], y_pred[self.data.test_mask.cpu()], average="micro")
        return f1


    def train_node_fullbatch_del(self, avg_time, run, optimizer, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        """
        Trains the GNN model for node-level tasks with deletion capabilities using the GNNDelete method.

        This method manages the training loop for node classification tasks, incorporating the unlearning
        process to remove the influence of specific nodes. It handles loss computation, backpropagation,
        optimizer steps, and periodic evaluation. Additionally, it integrates member inference attacks
        to assess privacy before and after the unlearning process.

        Args:
            avg_time (dict): A dictionary to store average training times.
            
            run (int): The current run or experiment index.
            
            optimizer (list of torch.optim.Optimizer): A list containing optimizers for different parts of the model.
            
            logits_ori (torch.Tensor, optional): Original logits before deletion. Defaults to None.
            
            attack_model_all (torch.nn.Module, optional): Model used for member inference attacks on all data. Defaults to None.
            
            attack_model_sub (torch.nn.Module, optional): Model used for member inference attacks on a subset of data. Defaults to None.

        Returns:
            None
        """
        self.trainer_log = {
            'unlearning_model': self.args["unlearning_model"],
            'dataset': self.args["dataset_name"],
            'log': []}
        self.model = self.model.to('cuda')
        self.data = self.data.to('cuda')

        best_metric = 0
        
        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(self.model, attack_model_all, self.data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(self.model, attack_model_sub, self.data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        non_df_node_mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.data.x.device)
        non_df_node_mask[self.data.directed_df_edge_index.flatten().unique()] = False

        self.data.sdf_node_1hop_mask_non_df_mask = self.data.sdf_node_1hop_mask & non_df_node_mask
        self.data.sdf_node_2hop_mask_non_df_mask = self.data.sdf_node_2hop_mask & non_df_node_mask

        # Original node embeddings
        with torch.no_grad():
            if self.args["base_model"] in ["SIGN","SGC","S2GC"]:
                z1_ori, z2_ori = self.model.get_original_embeddings(self.data.features_pre,return_all_emb=True)
            else:
                z1_ori, z2_ori = self.model.get_original_embeddings(self.data.x, self.data.edge_index[:, self.data.dr_mask],
                                                           return_all_emb=True)

        loss_fct = get_loss_fct(self.args["loss_fct"])

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=self.data.edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.df_mask.sum())
        epoch_time = 0

        for epoch in trange(self.args["unlearning_epochs"], desc='Unlerning'):
            self.model.train()

            start_time = time.time()
            if self.args["base_model"] in ["SIGN","SGC","S2GC"]:
                z1, z2 = self.model(self.data.features_pre, return_all_emb=True)
            else:    
                z1, z2 = self.model(self.data.x, self.data.edge_index[:, self.data.sdf_mask], return_all_emb=True)

            # Randomness
            pos_edge = self.data.edge_index[:, self.data.df_mask]


            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)

            # Local causality
            loss_l1 = loss_fct(z1[self.data.sdf_node_1hop_mask_non_df_mask], z1_ori[self.data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[self.data.sdf_node_2hop_mask_non_df_mask], z2_ori[self.data.sdf_node_2hop_mask_non_df_mask])

            # Total loss
            '''both_all, both_layerwise, only2_layerwise, only2_all, only1'''
            loss_l = loss_l1 + loss_l2
            loss_r = loss_r1 + loss_r2

            loss1 = self.args["alpha"] * loss_r1 + (1 - self.args["alpha"]) * loss_l1
            loss1.backward(retain_graph=True)
            optimizer[0].step()
            optimizer[0].zero_grad()

            loss2 = self.args["alpha"] * loss_r2 + (1 - self.args["alpha"]) * loss_l2
            loss2.backward(retain_graph=True)
            optimizer[1].step()
            optimizer[1].zero_grad()

            loss = loss1 + loss2

            end_time = time.time()
            epoch_time += end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time/self.args["unlearning_epochs"]
            }
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))
            self.logger.info("time:{}".format(epoch_time/self.args["num_epochs"]))
            if (epoch + 1) % self.args["test_freq"] == 0:
                valid_loss, dt_acc,recall, dt_f1, valid_log = self.eval_node_fullbatch_del('val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time/self.args["unlearning_epochs"],
                }

                for log in [train_log, valid_log]:
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_acc + dt_f1 > best_metric:
                    best_metric = dt_acc + dt_f1
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': self.model.state_dict(),
                        # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
                    }
                    torch.save(ckpt, os.path.join(self.args["checkpoint_dir"],'model_best.pt'))
        avg_time[run] = epoch_time
        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in self.model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_final.pt'))

        #         # === Compute unlearn ratio ===
        # if hasattr(self, 'temp_node'):
        #     unlearn_ratio = len(self.temp_node) / self.data.num_nodes
        # else:
        #     # Fallback: compute based on sdf_mask (common in MEGU logic)
        #     unlearn_ratio = self.data.df_mask.sum().item() / self.data.num_nodes

        # # === Construct save directory ===
        # save_dir = os.path.join(
        #     "/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU",
        #     "unlearned_models", "GNNDelete",
        #     self.args['dataset_name'],
        #     self.args["downstream_task"],
        #     f"ratio_{unlearn_ratio:.2f}"
        # )
        # os.makedirs(save_dir, exist_ok=True)save

        # # === Save best model if available ===
        # best_model_path = os.path.join(save_dir, f"GNNDelete{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}.pt")
        # if os.path.exists(os.path.join(self.args["checkpoint_dir"], 'model_best.pt')):
        #     best_ckpt = torch.load(os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
        #     torch.save(best_ckpt['model_state'], best_model_path)
        # else:
        #     # If no best checkpoint, save current model state
        #     torch.save(self.model.state_dict(), best_model_path)
        unlearning_model_name = self.args["unlearning_model"]  # e.g., 'GNNDelete'
        dataset = self.args["dataset_name"]                    # e.g., 'cora'
        unlearn_ratio = self.args["proportion_unlearned_nodes"]  # e.g., 0.01

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(run)

        base_model_str = ""
        if self.args['base_model']!="GCN":  
            base_model_str = "_" + self.args['base_model']

        unlearn_task = self.args['unlearn_task']

        save_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/{unlearning_model_name}/{dataset}/{unlearn_task}/ratio_{unlearn_ratio:.2f}{copy_str}"
        os.makedirs(save_path, exist_ok=True)

        save_file = f"{unlearning_model_name}_{dataset}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}{base_model_str}.pt"
        torch.save({'model_state': self.model.state_dict()}, os.path.join(save_path, save_file))

        self.logger.info(f"Unlearned model saved to {os.path.join(save_path, save_file)}")


    @torch.no_grad()
    def test_node_fullbatch_del(self, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        """
        Tests the GNN model for node-level tasks after deletion operations.

        This method loads the best model checkpoint, evaluates the model on the test set for node classification,
        and optionally performs member inference attacks to assess privacy after unlearning.

        Args:
            model_retrain (torch.nn.Module, optional): The retrained model for deletion verification.
                                                    Defaults to None.
            
            attack_model_all (torch.nn.Module, optional): Model used for member inference attacks on all data.
                                                        Defaults to None.
            
            attack_model_sub (torch.nn.Module, optional): Model used for member inference attacks on a subset of data.
                                                        Defaults to None.
            
            ckpt (str, optional): Specifies which checkpoint to load ('best' or other). Defaults to 'best'.

        Returns:
            tuple:
                loss (float): The loss value on the test set.

                dt_acc (float): The accuracy on the test set.

                recall (float): The recall on the test set.

                dt_f1 (float): The F1 score on the test set.

                test_log (dict): A dictionary containing additional test metrics.
        """
        if ckpt == 'best':  # Load best ckpt
            ckpt = torch.load(os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
            self.model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args["dataset_name"]:
            pred_all = False
        else:
            pred_all = True
        loss, dt_acc, recall,dt_f1, test_log = self.eval_node_fullbatch_del('test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_acc'] = dt_acc
        self.trainer_log['dt_f1'] = dt_f1
        # self.trainer_log['df_logit'] = df_logit
        # self.logit_all_pair = logit_all_pair
        # self.trainer_log['df_auc'] = df_auc
        # self.trainer_log['df_aup'] = df_aup

        if model_retrain is not None:  # Deletion
            self.trainer_log['ve'] = self.verification_error(self.model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, self.data=self.data).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(self.model, attack_model_all, self.data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(self.model, attack_model_sub, self.data)
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
        self.logger.info("loss:{}  ,dt_acc:{} ,recall:{}  ,dt_f1:{}  ,test_log:{}  ".format(loss, dt_acc, recall, dt_f1, test_log))
        return loss, dt_acc, recall, dt_f1, test_log

    def eval_del(self,stage="val",pred_all=False):
        """
        Evaluates the model's performance on deletion tasks based on the downstream task.

        Depending on whether the downstream task is node classification or edge prediction,
        this method delegates the evaluation process to the corresponding specialized evaluation method.

        Args:
            stage (str, optional): The evaluation stage ('val' or 'test'). Defaults to "val".
            
            pred_all (bool, optional): Whether to predict logits for all node pairs. Defaults to False.

        Returns:
            tuple:
                If downstream_task is "node":
                    loss (float): The loss value.

                    dt_acc (float): The accuracy.

                    dt_auc (float): The AUC score.

                    df_auc (float): The defensive AUC score.

                    test_log (dict): A dictionary containing evaluation metrics.
                
                If downstream_task is "edge":
                    loss (float): The loss value.

                    acc (float): The accuracy.

                    dt_auc (float): The AUC score.

                    df_auc (float): The defensive AUC score.

                    test_log (dict): A dictionary containing evaluation metrics.
        """
        if self.args["downstream_task"]=="node":
            return self.eval_node_fullbatch_del(stage,pred_all)
        elif self.args["downstream_task"]=="edge":
            loss, f1,acc,dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval_edge(stage, pred_all)
            return loss,acc,dt_auc,df_auc,test_log

    @torch.no_grad()
    def eval_node_fullbatch_del(self,stage='val', pred_all=False):
        """
        Evaluates the model's performance on node-level tasks after deletion.

        This method computes loss, accuracy, recall, and F1 score for node classification tasks
        based on the specified evaluation stage.

        Args:
            stage (str, optional): The evaluation stage ('val' or 'test'). Defaults to 'val'.
            
            pred_all (bool, optional): Whether to predict logits for all node pairs. Defaults to False.

        Returns:
            tuple:
                loss (float): The loss value.

                dt_acc (float): The accuracy score.

                recall (float): The recall score.

                dt_f1 (float): The F1 score.

                log (dict): A dictionary containing evaluation metrics.
        """
        self.model.eval()

        # DT AUC AUP
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            z = self.model(self.data.features_pre)
        else:
            z = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(z[self.data.test_mask], self.data.y[self.data.test_mask]).cpu().item()
        pred = torch.argmax(z[self.data.test_mask], dim=1).cpu()
        true_lable = self.data.y[self.data.test_mask]
        dt_acc = accuracy_score(self.data.y[self.data.test_mask].cpu(), pred)
        recall = recall_score(self.data.y[self.data.test_mask].cpu(), pred,average='micro')
        dt_f1 = f1_score(self.data.y[self.data.test_mask].cpu(), pred, average='micro')

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_acc': dt_acc,
            f'{stage}_dt_f1': dt_f1,
        }

        if self.device == 'cpu':
            self.model = self.model.to(self.device)


        return loss, dt_acc,recall, dt_f1, log
    
    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        """
        General evaluation method for node classification tasks.

        This method computes loss, accuracy, and F1 score based on the model's predictions
        and the true labels of the specified evaluation stage.

        Args:
            model (torch.nn.Module): The trained model to be evaluated.
            
            data (torch_geometric.data.Data): The dataset containing graph information.
            
            stage (str, optional): The evaluation stage ('val' or 'test'). Defaults to 'val'.
            
            pred_all (bool, optional): Whether to predict logits for all node pairs. Defaults to False.

        Returns:
            tuple:
                loss (float): The loss value.

                dt_acc (float): The accuracy score.

                dt_f1 (float): The F1 score.

                log (dict): A dictionary containing evaluation metrics.
        """
        model.eval()

        z = F.log_softmax(model(data.x, data.edge_index), dim=1)

        # DT AUC AUP
        loss = F.nll_loss(z[data.test_mask], data.y[data.test_mask]).cpu().item()
        pred = torch.argmax(z[data.test_mask], dim=1).cpu()
        dt_acc = accuracy_score(data.y[data.test_mask].cpu(), pred)
        dt_f1 = f1_score(data.y[data.test_mask].cpu(), pred, average='micro')

        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_acc': dt_acc,
            f'{stage}_dt_f1': dt_f1,
        }

        return loss, dt_acc, dt_f1, log
    
    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        """
        Tests the GNN model for node-level tasks after deletion operations.

        This method loads the best model checkpoint, evaluates the model on the test set for node classification,
        and optionally performs member inference attacks to assess privacy after unlearning.

        Args:
            model (torch.nn.Module): The trained model to be evaluated.
            
            data (torch_geometric.data.Data): The dataset containing graph information.
            
            model_retrain (torch.nn.Module, optional): The retrained model for deletion verification.
                                                    Defaults to None.
            
            attack_model_all (torch.nn.Module, optional): Model used for member inference attacks on all data.
                                                        Defaults to None.
            
            attack_model_sub (torch.nn.Module, optional): Model used for member inference attacks on a subset of data.
                                                        Defaults to None.
            
            ckpt (str, optional): Specifies which checkpoint to load ('best' or other). Defaults to 'best'.

        Returns:
            tuple:

                loss (float): The loss value on the test set.

                dt_acc (float): The accuracy on the test set.

                dt_f1 (float): The F1 score on the test set.
                
                test_log (dict): A dictionary containing additional test metrics.
        """
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args["dataset_name"]:
            pred_all = False
        else:
            pred_all = True
        loss, dt_acc, dt_f1, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_acc'] = dt_acc
        self.trainer_log['dt_f1'] = dt_f1

        if model_retrain is not None: 
            self.trainer_log['ve'] = self.verification_error(model, model_retrain).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
            self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after
            
            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_all_after'], self.trainer_log['mi_logit_all_before'])])
            self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_sub_after'], self.trainer_log['mi_logit_sub_before'])])
            print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'], self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
            print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])

        return loss, dt_acc, dt_f1, test_log

    @torch.no_grad()
    def verification_error(self,model1, model2):
        """
        Computes the verification error between two models.

        This method calculates the L2 distance between the parameters of two models to measure
        how much the model has changed after the unlearning process.

        Args:
            model1 (torch.nn.Module): The original model.
            
            model2 (torch.nn.Module): The retrained or modified model.

        Returns:
            torch.Tensor:

                diff (torch.Tensor): The computed verification error (L2 distance).
        """
        '''L2 distance between aproximate model and re-trained model'''

        model1 = model1.to('cpu')
        model2 = model2.to('cpu')

        modules1 = {n: p for n, p in model1.named_parameters()}
        modules2 = {n: p for n, p in model2.named_parameters()}

        all_names = set(modules1.keys()) & set(modules2.keys())

        print(all_names)

        diff = torch.tensor(0.0).float()
        for n in all_names:
            diff += torch.norm(modules1[n] - modules2[n])
        
        return diff
    
    @torch.no_grad()
    def eval_edge(self, stage='val', pred_all=False):
        """
        Evaluates the model's performance on edge-level tasks.

        This method computes loss, F1 score, accuracy, AUC scores, and other metrics for edge prediction
        based on the specified evaluation stage. It handles both positive and negative edges and
        integrates defensive edges for comprehensive evaluation.

        Args:
            stage (str, optional): The evaluation stage ('val' or 'test'). Defaults to 'val'.
            
            pred_all (bool, optional): Whether to predict logits for all node pairs. Defaults to False.

        Returns:
            tuple:

                loss (float): The loss value.

                f1 (float): The F1 score.

                acc (float): The accuracy score.

                dt_auc (float): The AUC score.

                dt_aup (float): The average precision score.

                df_auc (float): The defensive AUC score.

                df_aup (float): The defensive average precision score.

                df_logit (list): A list of defensive logits.

                logit_all_pair (torch.Tensor or None): Logits for all node pairs if `pred_all` is True.

                log (dict): A dictionary containing evaluation metrics.
        """
        self.model.eval()
        pos_edge_index = self.data[f'{stage}_edge_index']
        neg_edge_index = negative_sampling(
                edge_index=self.data.test_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.test_edge_index.size(1)
            )
        
        z = self.model(self.data.x, self.data.edge_index)
        logits = self.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)
        pred = torch.where(logits > 0.5, torch.tensor(1), torch.tensor(0))
        f1 = f1_score(label.cpu(),pred.cpu())
        acc = accuracy_score(label.cpu(),pred.cpu())
        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()

        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        if self.args["unlearning_model"] in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = torch.sigmoid(self.decode_val(z, self.data.directed_df_edge_index)).tolist()

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []
        
            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(self.data.edge_index[:, self.data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(self.data.edge_index[:, self.data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)
            
            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = self.decode(z, self.data.edge_index[:, self.data.dr_mask][:, mask]).sigmoid().tolist()
                
                logit = df_logit + pos_logit
                label = [0] * len(df_logit) +  [1] * len(df_logit)

                # df_auc.append(roc_auc_score(label, logit))
                # df_aup.append(average_precision_score(label, logit))
        
            df_auc = np.mean(df_auc) if len(df_auc) else np.nan
            df_aup = np.mean(df_aup) if len(df_aup) else np.nan

        else:
            df_auc = np.nan
            df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        return loss, f1,acc,dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, log
    
    @torch.no_grad()
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        """
        Generates labels for positive and negative edges.

        This method creates a label tensor where positive edges are labeled as 1 and
        negative edges are labeled as 0.

        Args:
            pos_edge_index (torch.Tensor): Indices of the positive (existing) edges.
            
            neg_edge_index (torch.Tensor): Indices of the negative (non-existing) edges.

        Returns:
            torch.Tensor:

                link_labels (torch.Tensor): A tensor of edge labels.
        """
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels
    
    @torch.no_grad()
    def test_edge(self, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        """
        Tests the GNN model for edge-level tasks after deletion operations.

        This method loads the best model checkpoint, evaluates the model on the test set for edge prediction,
        and optionally performs member inference attacks to assess privacy after unlearning.

        Args:
            model_retrain (torch.nn.Module, optional): The retrained model for deletion verification.
                                                    Defaults to None.
            
            attack_model_all (torch.nn.Module, optional): Model used for member inference attacks on all data.
                                                        Defaults to None.
            
            attack_model_sub (torch.nn.Module, optional): Model used for member inference attacks on a subset of data.
                                                        Defaults to None.
            
            ckpt (str, optional): Specifies which checkpoint to load ('best' or other). Defaults to 'best'.

        Returns:
            tuple:

                loss (float): The loss value on the test set.

                f1 (float): The F1 score on the test set.

                acc (float): The accuracy on the test set.

                dt_auc (float): The AUC score on the test set.

                dt_aup (float): The average precision score on the test set.

                df_auc (float): The defensive AUC score.

                df_aup (float): The defensive average precision score.

                df_logit (list): A list of defensive logits.

                logit_all_pair (torch.Tensor or None): Logits for all node pairs if `pred_all` is True.

                test_log (dict): A dictionary containing additional test metrics.
        """
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
            self.model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args["dataset_name"]:
            pred_all = False
        else:
            pred_all = True
        loss, f1,acc,dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval_edge('test', pred_all)

        return loss, f1,acc,dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log
    
    
    
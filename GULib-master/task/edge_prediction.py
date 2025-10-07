import torch
import time
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from utils.utils import get_link_labels
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


class EdgePredictor:
    def __init__(self, args, data, model_zoo, logger):
        self.args = args
        self.data = data
        self.data = self.split_edge(self.data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_zoo = model_zoo
        self.model = model_zoo.model
        self.model_name = self.args['base_model']
        self.logger = logger

    def train_model(self, retrain=False):
        self.data = self.split_edge(self.data)
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        # self.train_data=self.train_data.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model.config.lr,
                                          weight_decay=self.model_zoo.model.config.decay)
        time_sum = 0
        for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
            start_time = time.time()
            self.model.train()

            self.optimizer.zero_grad()
            if self.model_name == "SIGN":
                out = self.model(self.data.xs)
            else:
                out = self.model(self.data.x, self.data.train_edge_index)

            neg_edge_index = negative_sampling(
                edge_index=self.data.train_edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_label_index.size(1)
            )
            neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)
            pos_edge_label = torch.ones(self.data.train_edge_index.size(1),dtype=torch.float32)
            # edge_logits = self.decode(z=out, edge_index=self.data.train_edge_label_index)
            edge_logits = self.decode(z=out, pos_edge_index=self.data.train_edge_index,neg_edge_index=neg_edge_index)
            
            edge_labels = torch.cat((pos_edge_label,neg_edge_label),dim=-1)
            edge_labels = edge_labels.to(self.device)
            # edge_labels = edge_labels.to(self.device)
            loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            # loss = self.get_loss(out)
            loss.backward()
            self.optimizer.step()
            self.best_valid_acc = 0
            time_sum += time.time() - start_time
            # print(loss.item())
            if (epoch + 1) % self.args["test_freq"] == 0:
                F1_score = self.evaluate_model()
                self.logger.info('Epoch: {:03d} | F1 Score: {:.4f} | Loss: {:.4f}'.format(epoch + 1, F1_score, loss))
        avg_training_time = time_sum / self.args['num_epochs']
        self.logger.info("Average training time per epoch: {:.4f}s".format(avg_training_time))


    @torch.no_grad()
    def evaluate_model(self):

        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

        if self.args["base_model"] == "SIGN":
            out = self.model(self.data.xs)
        else:
            out = self.model(self.data.x, self.data.val_edge_index)
        edge_label_index = self.data.val_edge_label_index
        edge_pred_logits = self.decode_val(z=out, edge_label_index=edge_label_index)
        edge_pred_logits = edge_pred_logits.cpu()
        edge_pred = torch.where(edge_pred_logits > 0.5, torch.tensor(1), torch.tensor(0))

        edge_labels = self.data.val_edge_label
        F1_score = roc_auc_score(edge_labels.cpu(), edge_pred.cpu())

        return F1_score

    @torch.no_grad()
    def eval_unlearning(self, features, unlearning_nodes, edge_mask=None):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = F.softmax(self.model(features[unlearning_nodes].cuda()),dim = 1).cpu()
        else:
            y_pred = F.softmax(self.model(self.data.x, self.data.edge_index),dim = 1).cpu()
            y_pred = y_pred[unlearning_nodes]
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[unlearning_nodes], y_pred, average="micro")
        Acc_score = accuracy_score(y_true=y[unlearning_nodes], y_pred=y_pred)
        Recall_score = recall_score(y_true=y[unlearning_nodes], y_pred=y_pred, average="micro")
        return F1_score, Acc_score, Recall_score

    def split_edge(self, data):
        temp = Data(x=data.x, y=data.y, edge_index=data.edge_index)
        train_data, val_data, test_data = RandomLinkSplit(num_val=0.2, num_test=0.1, is_undirected=True,
                                                          add_negative_train_samples=True)(temp)

        data.train_edge_index = train_data.edge_index
        data.test_edge_index = test_data.edge_index
        data.val_edge_index = val_data.edge_index

        data.train_edge_label = train_data.edge_label
        data.test_edge_label = test_data.edge_label
        data.val_edge_label = val_data.edge_label

        data.train_edge_label_index = train_data.edge_label_index
        data.test_edge_label_index = test_data.edge_label_index
        data.val_edge_label_index = val_data.edge_label_index

        return data


    def decode_val(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def get_edge_labels(self, pos_edge_index, neg_edge_index):
        num_edges = pos_edge_index.size(1) + neg_edge_index.size(1)
        edge_labels = torch.zeros(num_edges, dtype=torch.float32, device=self.device)  # float32 or float
        edge_labels[:pos_edge_index.size(1)] = 1
        return edge_labels

    def save_model(self, save_path, model_dict=None):
        with open(save_path, mode='w') as file:
            if model_dict is not None:
                self.logger.info('saving best model {}'.format(save_path))
                torch.save(model_dict, save_path)
            else:
                self.logger.info('saving model {}'.format(save_path))
                torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        # self.logger.info('loading model {}'.format(save_path))
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(save_path, map_location=device))

    def posterior_other(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SIGN":
            posteriors = F.log_softmax(self.model(self.data.xs))
        else:
            posteriors = F.log_softmax(self.model(self.data.x, self.data.edge_index))
        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask.cpu()], dim=-1)

        return posteriors.detach()

    @torch.no_grad()
    def evaluate_Del_model(self):
        # self.data = self.split_edge(self.data)
        self.model.eval()
        self.model = self.model.to(self.device)
        # self.data = self.data.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = self.model(self.data.features_pre).cpu()
        else:
            y_pred = self.model(self.data.x, self.data.val_edge_index).cpu()
        edge_label_index = self.data.val_edge_label_index
        edge_pred_logits = self.decode_val(z=y_pred, edge_label_index=edge_label_index.cpu())
        edge_pred_logits = edge_pred_logits.cpu()
        edge_pred = torch.where(edge_pred_logits > 0.5, torch.tensor(1), torch.tensor(0))
        # edge_pred = torch.argmax(edge_pred_logits)
        edge_labels = self.data.val_edge_label
        F1_score = f1_score(edge_labels.cpu(), edge_pred.cpu())
        Acc_score = accuracy_score(y_true=edge_labels.cpu(), y_pred=edge_pred.cpu())
        Recall_score = recall_score(y_true=edge_labels.cpu(), y_pred=edge_pred.cpu())
        return F1_score, Acc_score, Recall_score

    def GIF_evaluate_unlearn_F1(self, new_parameters):
        idx = 0
        for p in self.model.parameters():
            p.data = new_parameters[idx]
            idx = idx + 1

        out = self.model.reason_once_unlearn(self.data)

        neg_edge_index = negative_sampling(
            edge_index=self.data.edge_index, num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.edge_index.size(1)
        )
        edge_label = self.get_edge_labels(self.data.edge_index, neg_edge_index)
        edge_index = torch.cat([self.data.edge_index, neg_edge_index], dim=-1)
        edge_logits = self.decode(out, edge_index=edge_index).cpu()
        edge_pred = torch.where(edge_logits > 0.5, torch.tensor(1), torch.tensor(0))
        F1_score = f1_score(edge_label.cpu(), edge_pred.cpu())
        return F1_score

    @torch.no_grad()
    def evaluate_SGU_model(self, test_features):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = F.softmax(self.model(test_features,return_all_emb = True)[-1],dim=1).cpu()
        else:
            y_pred = F.softmax(self.model(self.data.x, self.data.edge_index,return_all_emb = True)[-1],dim=1).cpu()
        edge_label_index = self.data.val_edge_label_index
        edge_pred_logits = self.decode_val(z=y_pred, edge_label_index=edge_label_index.cpu())
        edge_pred_logits = edge_pred_logits.cpu()
        edge_pred = torch.where(edge_pred_logits > 0.5, torch.tensor(1), torch.tensor(0))
        # edge_pred = torch.argmax(edge_pred_logits)
        edge_labels = self.data.val_edge_label
        F1_score = f1_score(edge_labels.cpu(), edge_pred.cpu())
        Acc_score = accuracy_score(y_true=edge_labels.cpu(), y_pred=edge_pred.cpu())
        Recall_score = recall_score(y_true=edge_labels.cpu(), y_pred=edge_pred.cpu())
        return F1_score, Acc_score, Recall_score

    def train_GUIDE_model(self):
        self.data = self.split_edge(self.data)
        for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
            self.model.train()

            self.optimizer.zero_grad()
            if self.model_name == "SIGN":
                out = self.model(self.data.xs)
            else:
                out = self.model(self.data.x, self.data.train_edge_index)

            edge_logits = self.decode(z=out, edge_index=self.data.train_edge_label_index)
            edge_labels = self.data.train_edge_label
            # edge_labels = edge_labels.to(self.device)
            loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            loss.backward()
            self.optimizer.step()
            self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss))

    def train_UTU_model(self):
        self.data = self.data.to(self.device)
        self.trainer_log = {
            'unlearning_model': self.args["unlearning_model"],
            'dataset': self.args["dataset"],
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []
        start_time = time.time()
        best_valid_loss = 1000000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model.config.lr,
                                          weight_decay=self.model_zoo.model.config.decay)
        for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
            self.model.train()
            self.model.reset_parameters()
            neg_edge_index = negative_sampling(
                edge_index=self.data.train_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.dtrain_mask.sum())
            z = self.model(self.data.x, self.data.train_pos_edge_index)
            logits = self.decode(z, self.data.train_pos_edge_index, neg_edge_index)
            label = get_link_labels(self.data.train_pos_edge_index, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if (epoch + 1) % self.args["test_freq"] == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.UTU_eval(
                    self.model, self.data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                }
                log = {**train_log, **valid_log}
                for log in [train_log, valid_log]:
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
                    torch.save(z, os.path.join(self.args["checkpoint_dir"], 'node_embeddings.pt'))

        self.trainer_log['training_time'] = time.time() - start_time
        ckpt = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_final.pt'))

        print(
            f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss

    @torch.no_grad()
    def UTU_eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        logits = self.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = get_link_labels(pos_edge_index, neg_edge_index)

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

        # Logits for all node pairs
        # if pred_all and stage == 'test':
        #     logit_all_pair = (z @ z.t()).cpu()
        # else:
        #     logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, None, log
    
    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits


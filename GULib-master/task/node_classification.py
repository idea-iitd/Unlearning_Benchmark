import logging
import os
import random
import time
import copy
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score,recall_score
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch_geometric.loader import NeighborLoader
from attack.Attack_methods.GNNDelete_MIA import member_infer_attack
from model.base_gnn.ceu_model import CEU_GNN
from utils import  utils
from torch.autograd import grad
import config
import torch_geometric.transforms as T
from tqdm import tqdm
from utils.utils import get_loss_fct, trange, Reverse_CE
from config import root_path


class NodeClassifier:
    def __init__(self,args,data,model_zoo,logger):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.data = data
        self.model_zoo = model_zoo
        self.model = model_zoo.model
        self.logger = logger
        self.model_name = self.args['base_model']

    def train_model(self,retrain=False):
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model.config.lr, weight_decay=self.model_zoo.model.config.decay)
        time_sum = 0

        for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out = self.model(self.data.xs)
            else:
                out = self.model(self.data.x,self.data.edge_index)

            loss = F.cross_entropy(out[self.data.train_mask],self.data.y[self.data.train_mask])
            
            
            loss.backward()
            self.optimizer.step()
            self.best_valid_acc = 0
            time_sum += time.time() - start_time

            if (epoch+1) % self.args["test_freq"] == 0:
                
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
            y_pred = self.model(self.data.xs).cpu()
        else:
            y_pred = self.model(self.data.x,self.data.edge_index).cpu()
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[self.data.test_mask.cpu()], y_pred[self.data.test_mask.cpu()], average="micro")
        
        return F1_score


    def train_SGU_model(self,retrain=False):
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model.config.lr, weight_decay=self.model_zoo.model.config.decay)
        start_time = time.time()
        best_acc = 0
        best_w = 0
        for epoch in range(self.args['num_epochs']):
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
                F1_score, Accuracy, Recall = self.evaluate_SGU_model(self.data.pre_features[self.data.test_indices])
                self.logger.info(
                    'epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score, Accuracy, Recall))
                if Accuracy > best_acc :
                    best_acc = Accuracy
                    best_w = copy.deepcopy(self.model.state_dict())
        if self.args["unlearn_task"] == "edge":
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + \
                         self.args["base_model"] + self.args["proportion_unlearned_edges"]
        else:
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + \
                         self.args["base_model"]

        self.logger.info("best:{}".format(best_acc))


    @torch.no_grad()
    def evaluate_SGU_model(self,test_features):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = self.model.get_softlabel(test_features).cpu()
        else:
            y_pred = self.model.get_softlabel(self.data.x,self.data.edge_index).cpu()
            y_pred = y_pred[self.data.test_indices]
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[self.data.test_indices], y_pred, average="micro")
        Acc_score = accuracy_score(y_true=y[self.data.test_indices], y_pred=y_pred)
        Recall_score = recall_score(y_true=y[self.data.test_indices], y_pred=y_pred, average="micro")
        return F1_score, Acc_score, Recall_score



    def SGU_unlearning_edge(self,
                        original_softlabels,
                        original_w,
                        unlearning_nodes,
                        activated_nodes,
                        pos_pair,
                        neg_pair,
                        original_feaures,
                        prototype_embedding,
                        unlearning_edge_index):
        self.tau = 0.5
        self.para1 = 0.8
        self.para2 = 0.001
        self.para3 = 100
        self.para4 = 0.01
        self.para5 = 15
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model.config.lr,
                                          weight_decay=self.model_zoo.model.config.decay)
        best = 0
        prototype_embedding = prototype_embedding.cuda()


        for epoch in range(self.args['unlearning_epochs']):
            if (epoch+1) % self.args["test_freq"] == 0:
                F1_score,Accuracy,Recall = self.evaluate_SGU_model(original_feaures[self.data.test_indices])
                self.logger.info('epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score,Accuracy,Recall))
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(original_feaures)
            logits = self.model.decode(z, unlearning_edge_index)
            label = torch.zeros(unlearning_edge_index.size(1)).to(self.device)
            loss_U_edge = F.binary_cross_entropy_with_logits(logits, label)
            softlabels_E = F.softmax(self.model(original_feaures[activated_nodes]),dim=1)
            embedding_E = self.model.get_embedding(original_feaures[activated_nodes])
            embedding_U = self.model.get_embedding(original_feaures[unlearning_nodes])

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
            for i, node in enumerate(activated_nodes):
                neg_tensors = [torch.tensor(neg_emb) for neg_emb in neg_pair[node]]
                negtive_tensors = torch.stack([F.normalize(neg_tensor, p=2, dim=0) for neg_tensor in neg_tensors], dim=0).cuda()
                neg_scores[i] = torch.exp(torch.einsum('d,nd->n', F.normalize(embedding_E[i], p=2, dim=0),
                                 negtive_tensors) / self.tau).sum()
            loss_hE = -torch.log(pos_scores / neg_scores).sum()


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

            if F1_score > best and epoch > 40:
                best = F1_score
                F1_score,Accuracy,Recall = self.eval_unlearning(original_feaures,unlearning_nodes)
                self.logger.info('Unlearning: F1_score = {}  Accuracy = {}  Recall = {}'.format(F1_score,Accuracy,Recall))
        F1_score, Accuracy, Recall = self.eval_unlearning(original_feaures,unlearning_nodes)
        self.logger.info("para1:{}   para2:{}   para3:{}   para4:{}   para5:{}".format(self.para1,
                                                                                       self.para2,
                                                                                       self.para3,
                                                                                       self.para4,
                                                                                       self.para5))
        self.logger.info('Unlearning: F1_score = {}  Accuracy = {}  Recall = {}  best: {}'.format(F1_score, Accuracy, Recall,best))






    @torch.no_grad()
    def eval_unlearning(self,features,unlearning_nodes,edge_mask=None):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = self.model.get_softlabel(features[unlearning_nodes].cuda()).cpu()
        else:
            y_pred = self.model.get_softlabel(self.data.x,self.data.edge_index).cpu()
            y_pred = y_pred[unlearning_nodes]
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[unlearning_nodes], y_pred, average="micro")
        Acc_score = accuracy_score(y_true=y[unlearning_nodes], y_pred=y_pred)
        Recall_score = recall_score(y_true=y[unlearning_nodes], y_pred=y_pred,average="micro")
        return F1_score,Acc_score,Recall_score


    def GIF_evaluate_unlearn_F1(self, new_parameters):
        idx = 0
        for p in self.model.parameters():
            p.data = new_parameters[idx]
            idx = idx + 1

        out = self.model.reason_once_unlearn(self.data)

        test_f1 = f1_score(
            self.data.y[self.data['test_mask']].cpu().numpy(),
            out[self.data['test_mask']].argmax(axis=1).cpu().numpy(),
            average="micro"
        )
        return test_f1

    def train_GUIDE_model(self):
        for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
            self.model.train()
            labels = self.data.y

            self.optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out = self.model(self.data.xs)[self.data.train_mask]
                loss = F.cross_entropy(out, labels[self.data.train_mask[:labels.size(0)]])

            else:
                out = self.model(self.data.x,self.data.edge_index)[self.data.train_mask]
                loss = F.cross_entropy(out, labels[self.data.train_mask[:labels.size(0)]])
            self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss))

            loss.backward()
            self.optimizer.step()


    def CEU_train(self,data,eval=True, verbose=True,return_epoch=False):
        if verbose:
            t0 = time.time()
            print(f'Start to train a {self.args["model_name"]} model...')

        model = self.model_zoo.model.to(self.device)
        train_loader = DataLoader(data['train_set'], batch_size=self.args["train_batch"], shuffle=True)
        valid_loader = DataLoader(data['valid_set'], batch_size=self.args["test_batch"])
        edge_index = torch.tensor(data['edges'], device=self.device).t()
        num_epochs = self.args['num_epochs']
        lr = self.args["unlearn_lr"]
        l2 = self.args["l2"]
        patience = self.args["patience"]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        best_valid_loss = 999999.
        best_epoch = 0
        trail_count = 0

        for e in range(1, num_epochs + 1):

            train_losses = []

            model.train()
            iterator = tqdm(train_loader, f'  Epoch {e}') if verbose else train_loader
            for nodes, labels in iterator:
                model.zero_grad()

                nodes = nodes.to(self.device)
                labels = labels.to(self.device)

                y_hat = model(nodes, edge_index)
                loss = model.loss(y_hat, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.cpu().item())

            train_loss = np.mean(train_losses)

            valid_losses = []

            model.eval()
            with torch.no_grad():
                for nodes, labels in valid_loader:
                    nodes = nodes.to(self.device)
                    labels = labels.to(self.device)

                    y_hat = model(nodes, edge_index)
                    loss = model.loss(y_hat, labels)

                    valid_losses.append(loss.cpu().item())

            valid_loss = np.mean(valid_losses)

            if verbose:
                print(f'  Epoch {e}, training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}.')

            if self.args["early_stop"]:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    trail_count = 0
                    best_epoch = e
                    path = os.path.join('./data/CEU/checkpoint', 'tmp')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(model.state_dict(), path + f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt')
                else:
                    trail_count += 1
                    if trail_count > patience:
                        if verbose:
                            print(
                                f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                        break
            else:
                path = os.path.join('./data/CEU/checkpoint','tmp')
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(),path+f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt')
        path = os.path.join('./data/CEU/checkpoint', 'tmp')
        model.load_state_dict(torch.load(path + f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt'))

        if eval:
            self.CEU_evaluate()

        # os.remove(os.path.join('./checkpoint', 'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))

        if verbose:
            print(f'{self.args["base_model"]} model training finished. Duration:', int(time.time() - t0))

        if return_epoch:
            return model, num_epochs
        else:
            return model

    @torch.no_grad()
    def CEU_test(self,model, test_loader, edge_index):
        y_preds = []
        y_trues = []
        test_loss = []
        model.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes, labels = nodes.to(self.device), labels.to(self.device)
                y_hat = model(nodes, edge_index)
                test_loss.append(model.loss(y_hat, labels).cpu().item())
                y_pred = torch.argmax(y_hat, dim=1)
                y_preds.extend(y_pred.cpu().tolist())
                y_trues.extend(labels.cpu().tolist())
        # del model
        # torch.cuda.empty_cache()
        res = classification_report(y_trues, y_preds, digits=4, output_dict=True)
        return res, np.mean(test_loss)

    @torch.no_grad()
    def CEU_evaluate(self):
        test_loader = DataLoader(self.data['test_set'], batch_size=self.args['test_batch'], shuffle=False)
        edge_index = torch.tensor(self.data['edges'], device=self.device).t()
        y_preds = []
        y_true = []

        self.model.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes = nodes.to(self.device)
                labels = labels.to(self.device)

                y_hat = self.model(nodes, edge_index)
                y_pred = torch.argmax(y_hat, dim=1)

                y_preds.extend(y_pred.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

        results = classification_report(y_true, y_preds, digits=4)
        print('  Result:')
        print(results)




    def GNNDelete_train(self,logger, avg_time,run,model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        self.trainer_log = {
            'unlearning_model': self.args["unlearning_model"],
            'dataset': self.args["dataset_name"],
            'log': []}
        model = model.to('cuda')
        data = data.to('cuda')

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

        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask

        # Original node embeddings
        with torch.no_grad():
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
                z1_ori, z2_ori = model.get_original_embeddings(data.features_pre,return_all_emb=True)
            else:
                z1_ori, z2_ori = model.get_original_embeddings(data.x, data.edge_index[:, data.dr_mask],
                                                           return_all_emb=True)

        loss_fct = get_loss_fct(self.args["loss_fct"])

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.df_mask.sum())
        epoch_time = 0

        for epoch in trange(self.args["unlearning_epochs"], desc='Unlerning'):
            model.train()

            start_time = time.time()
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
                z1, z2 = model(data.features_pre, return_all_emb=True)
            else:    
                z1, z2 = model(data.x, data.edge_index[:, data.sdf_mask], return_all_emb=True)

            pos_edge = data.edge_index[:, data.df_mask]

            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)

            # Local causality
            loss_l1 = loss_fct(z1[data.sdf_node_1hop_mask_non_df_mask], z1_ori[data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[data.sdf_node_2hop_mask_non_df_mask], z2_ori[data.sdf_node_2hop_mask_non_df_mask])

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
            logger.info("time:{}".format(epoch_time/50))
            if (epoch + 1) % self.args["test_freq"] == 0:
                valid_loss, dt_acc,recall, dt_f1, valid_log = self.eval(model, data, 'val')
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
                        'model_state': model.state_dict(),
                        # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
                    }
                    torch.save(ckpt, os.path.join(self.args["checkpoint_dir"],'model_best.pt'))
        avg_time[run] = epoch_time/self.args["unlearning_epochs"]  
        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(self.args["checkpoint_dir"], 'model_final.pt'))

    @torch.no_grad()
    def GNNDelete_test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):

        if ckpt == 'best':  # Load best ckpt
            ckpt = torch.load(os.path.join(self.args["checkpoint_dir"], 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args["dataset_name"]:
            pred_all = False
        else:
            pred_all = True
        loss, dt_acc, recall,dt_f1, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_acc'] = dt_acc
        self.trainer_log['dt_f1'] = dt_f1

        if model_retrain is not None:  # Deletion
            self.trainer_log['ve'] = self.verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
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
        self.logger.info("loss:{}  ,dt_acc:{} ,recall:{}  ,dt_f1:{}  ,test_log:{}  ".format(loss, dt_acc, recall, dt_f1, test_log))
        return loss, dt_acc, recall, dt_f1, test_log

    @torch.no_grad()
    def GNNDelete_eval(self,epoch,retrain):
        valid_loss, dt_acc, recall, dt_f1, valid_log = self.eval(self.model, self.data, 'val')
        if dt_acc > self.best_valid_acc:
            best_valid_acc = dt_acc
            best_epoch = epoch

            self.logger.info(f'Save best checkpoint at epoch {epoch:04d}. Valid Acc = {dt_acc:.4f} ')
            ckpt = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
            }
            if retrain == False:
                original_path = os.path.join(self.args["checkpoint_dir"], self.args["dataset_name"], self.args["gnn"],
                                             'original',
                                             '-'.join([str(i) for i in [self.args["df"], self.args["df_size"],
                                                                        self.args["random_seed"]]]))
                os.makedirs(original_path, exist_ok=True)
                torch.save(ckpt, original_path + '/model_best.pt')
            else:
                retrain_path = os.path.join(self.args["checkpoint_dir"], self.args["dataset_name"], self.args["gnn"],
                                            'retrain',
                                            '-'.join([str(i) for i in [self.args["df"], self.args["df_size"],
                                                                       self.args["random_seed"]]]))
                os.makedirs(retrain_path, exist_ok=True)
                torch.save(ckpt, retrain_path + '/model_best.pt')
        
        self.logger.info('Epoch: {:03d} | F1 Score: {:.4f} | Loss: {:.4f}'.format(epoch + 1, dt_f1, valid_loss))


    
    
    @torch.no_grad()
    def evaluate_Del_model(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = self.model(self.data.features_pre).cpu()
        else:
            y_pred = self.model(self.data.x,self.data.edge_index).cpu()
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[self.data.test_mask.cpu()], y_pred[self.data.test_mask.cpu()], average="micro")
        Acc_score = accuracy_score(y_true=y[self.data.test_mask.cpu()], y_pred=y_pred[self.data.test_mask.cpu()])
        Recall_score = recall_score(y_true=y[self.data.test_mask.cpu()], y_pred=y_pred[self.data.test_mask.cpu()],average="micro")
        return F1_score,Acc_score,Recall_score



    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()

        if self.device == 'cpu':
            model = model.to('cpu')

        # DT AUC AUP
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            z = self.model(self.data.features_pre)
        else:
            z = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(z[data.val_mask], data.y[data.val_mask]).cpu().item()
        pred = torch.argmax(z[data.val_mask], dim=1).cpu()
        true_lable = data.y[data.val_mask]
        dt_acc = accuracy_score(data.y[data.val_mask].cpu(), pred)
        recall = recall_score(data.y[data.val_mask].cpu(), pred,average='micro')
        dt_f1 = f1_score(data.y[data.val_mask].cpu(), pred, average='micro')

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_acc': dt_acc,
            f'{stage}_dt_f1': dt_f1,
        }

        if self.device == 'cpu':
            model = model.to(self.device)

        return loss, dt_acc,recall, dt_f1, log

    @torch.no_grad()
    def prediction_info(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        y_pred = self.model(self.data.x, self.data.edge_index).cpu()
        y = self.data.y.cpu()
        return y_pred[self.data.test_mask.cpu()], y[self.data.test_mask.cpu()]



    def posterior(self,return_features=False):
        self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()
        
        if self.args["unlearning_methods"] == "GraphRevoker":
            z, f = self._inference()

            if return_features:
                return z[self.data_full.test_mask, :], f[self.data_full.test_mask, :]
            return z[self.data_full.test_mask, :]
        posteriors = self.model.inference(self.data.x, self.test_loader, self.device)
        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask.cpu()], dim=-1)

        return posteriors.detach()
    
    @torch.no_grad()
    def _inference(self, no_test_edges=False):
        assert not self.data is None and not self.data_full is None

        self.model.eval()
        self.model = self.model.to(self.device)
        self.data_full = self.data.to(self.device) if no_test_edges else self.data_full.to(self.device)

        z, feat = self.model(self.data_full.x, self.data_full.edge_index, return_feature=True)

        return F.log_softmax(z,dim=1), feat

    def posterior_other(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["unlearning_methods"] == "GraphRevoker":
            posteriors = self.model(self.data.x, self.data.edge_index)
            return F.log_softmax(posteriors[self.data.test_mask, :]).detach()
        else:
            if self.args["base_model"] == "SIGN":
                posteriors = F.log_softmax(self.model(self.data.xs))
            else:
                posteriors = F.log_softmax(self.model(self.data.x,self.data.edge_index))
            for _, mask in self.data('test_mask'):
                posteriors = F.log_softmax(posteriors[mask.cpu()], dim=-1)

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        if self.model_name == 'GCN':
            logits = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            logits = self.model.inference(self.data.x, self.test_loader, self.device)
        return logits
    

    def prepare_data(self, input_data):
        data_full = input_data.clone()
        data = input_data.clone()
        
        data.edge_index = data.edge_index_train
        
        data.edge_index_train = None
        data_full.edge_index_train = None

        self.data.edge_index = input_data.edge_index_train
        self.data.edge_index_train = None
        self.data_full = data_full
        
        if self.args['is_use_train_batch']:
            self.gen_train_loader()
        if self.args['is_use_test_batch']:
            self.gen_test_loader()

    def gen_train_loader(self):
        assert not self.data is None

        self.train_loader = NeighborLoader(self.data, input_nodes=self.data.train_mask,
                                           num_neighbors=[15, 10], batch_size=self.args['batch_size'], 
                                           shuffle=True, num_workers=0, drop_last=True)

    def gen_test_loader(self):
        assert not self.data_full is None

        self.test_loader = NeighborLoader(self.data_full, input_nodes=None,
                                          num_neighbors=[15, 10], batch_size=self.args['test_batch_size'], 
                                          shuffle=False, num_workers=0)

    @torch.no_grad()
    def verification_error(self,model1, model2):
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

    def save_model(self, save_path,model_dict=None):
        with open(save_path,mode='w') as file:
            if model_dict is not None:
                self.logger.info('saving best model {}'.format(save_path))
                torch.save(model_dict, save_path)
            else:
                self.logger.info('saving model {}'.format(save_path))
                torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(save_path, map_location=device))




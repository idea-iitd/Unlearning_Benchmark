import time
import os
from sklearn.metrics import f1_score, accuracy_score,recall_score,roc_auc_score
import torch
import copy
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from task.node_classification import NodeClassifier
from utils.dataset_utils import save_train_test_split,graph_cls_process,negative_sampling
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils.dataset_utils import load_train_test_split
import torch.nn.functional as F
from config import BLUE_COLOR,RESET_COLOR
from config import root_path,unlearning_path,unlearning_edge_path
from task.edge_prediction import EdgePredictor
from task import get_trainer
from pipeline.IF_based_pipeline import IF_based_pipeline
class gif(IF_based_pipeline):
    """
    GIF (Graph Influence Function) class implements a IF-based pipeline for performing unlearning tasks on GNNs, enabling efficient removal of specific data points, edges, or features from
    trained graph-based models without the need for retraining from scratch.

    Class Attributes:
        args (dict): Configuration parameters for the GIF pipeline, including
            settings for the number of runs, unlearning ratios, and method choices.

        logger (Logger): Logger instance for logging information, debugging, and
            tracking the pipeline's progress and performance metrics.

        model_zoo (ModelZoo): Collection of models and related data resources used
            within the pipeline.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.model_zoo = model_zoo
        self.data= self.model_zoo.data
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run = 0
        self.num_runs = self.args["num_runs"]
        self.average_f1 = np.zeros(self.num_runs)
        self.average_auc = np.zeros(self.num_runs)
        self.avg_unlearning_time = np.zeros(self.num_runs)
        self.training_time = np.zeros(self.num_runs)



    # def run_exp(self):
    #     for self.run in range(self.num_runs):
    #         if self.args["GIF_exp"].lower() == "unlearning":
    #             if self.args["GIF_method"] .lower() == "retrain":
    #                 self.GIF_retrain()
    #             elif self.args["GIF_method"].lower() in ["gif", "if"]:
    #                 self.GraphInfluenceFunction()
    #             else:
    #                 raise NotImplementedError
    #         elif self.args["GIF_exp"].lower() == "attack_unlearning":
    #             if self.args["GIF_method"] == "Retrain":
    #                 self.GIF_attack()
    #             if self.args["GIF_method"].lower() in ["gif", "if"]:
    #                 self.GIF_attack()
    #         self.GraphInfluenceFunction()
    #     self.logger.info(
    #     "{}Performance Metrics:\n"
    #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average AUC Score: {:.4f} ± {:.4f}\n"
    #     " - Average Training Time: {:.4f} ± {:.4f}\n"
    #     " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
    #         BLUE_COLOR,
    #         np.mean(self.average_f1), np.std(self.average_f1),
    #         np.mean(self.average_auc), np.std(self.average_auc),
    #         np.mean(self.training_time), np.std(self.training_time),
    #         np.mean(self.avg_unlearning_time), np.std(self.avg_unlearning_time),
    #         RESET_COLOR
    #         )
    #     )

    def GIF_retrain(self):
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.gen_train_graph()
        self.determine_target_model()

        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        training_times = np.empty((0))
        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)

            run_training_time, grad_all = self._train_model(run)

            f1_score = self.evaluate(run)
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)

        f1_score_avg = np.average(run_f1)
        f1_score_std = np.std(run_f1)
        self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))
        self.logger.info("model training time: avg=%s seconds" % np.average(training_times))

    def GraphInfluenceFunction(self):
        self.deleted_nodes = np.array([])
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])
        self.deleted_edges = np.array([])
        self.influence_edges = np.array([])
        self.num_feats = self.data.num_features
        # self.train_test_split()
        self.unlearning_request()
        self.target_model_name = self.args['base_model']
        self.determine_target_model()
        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        unlearning_times = np.empty((0))
        training_times = np.empty((0))
        # for run in range(self.args['num_runs']):
            # self.logger.info("Run %f" % run)

        run_training_time, result_tuple = self.train_unlearning_model(self.run)
        if self.target_model_name in ['GCN','SGC',"S2GC"]:
            # out = self.target_model.model.forward_once(self.data, self.target_model.edge_weight)
            out = self.target_model.model.reason_once(self.data)
        else:
            # out = self.target_model.model.forward_once(self.data)
            out = self.target_model.model.reason_once(self.data)
        self.original_softlabels = out

        # if self.args["base_model"] != "SIGN":
        #     f1_score = self.evaluate(run)
        #     run_f1 = np.append(run_f1, f1_score)
        # training_times = np.append(training_times, run_training_time)
        self.training_time[self.run] = run_training_time
        # unlearning with GIF
        unlearning_time, f1_score_unlearning = self.approxi(result_tuple)
        
        unlearning_times = np.append(unlearning_times, unlearning_time)
        run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)
            

        # f1_score_avg = np.average(run_f1)
        # f1_score_std = np.std(run_f1)
        # self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))
        # self.logger.info("model training time: avg=%s seconds" % np.average(training_times))

        f1_score_unlearning_avg = np.average(run_f1_unlearning)
        # f1_score_unlearning_std = np.std(run_f1_unlearning)
        unlearning_time_avg = np.average(unlearning_times)
        # self.logger.info("f1_score of GIF: avg=%s, std=%s" % (f1_score_unlearning_avg, f1_score_unlearning_std))
        # self.logger.info("GIF unlearing time: avg=%s seconds" % np.average(unlearning_time_avg))
        self.average_f1[self.run] = f1_score_unlearning_avg
        if self.args["unlearn_task"]=="node" and self.args["downstream_task"]=="node":
            self.mia_attack()
        # elif self.args["unlearn_task"]=="edge":
        #     self.mia_attack_edge()
        self.avg_unlearning_time[self.run] = unlearning_time_avg
        

    def mia_attack(self):
        #--------------------------------------------------------------------------------
        if self.target_model_name in ['GCN','SGC',"S2GC"]:
            # out = self.target_model.model.forward_once(self.data, self.target_model.edge_weight)
            out = self.target_model.model.reason_once(self.data)
        else:
            # out = self.target_model.model.forward_once(self.data)
            out = self.target_model.model.reason_once(self.data)
        self.original_softlabels = out

        if len(self.data.test_indices)<len(self.unlearning_nodes):
            # use full test set size
            self.mia_num = len(self.data.test_indices)

            original_softlabels_non = self.original_softlabels[self.data.test_indices]
            original_softlabels_member_full = self.original_softlabels[self.unlearning_nodes]

            # get unlearning model outputs
            out = self.target_model.model.reason_once_unlearn(self.data)
            unlearning_softlabels_non = out[self.data.test_indices]
            unlearning_softlabels_member_full = out[self.unlearning_nodes]

            aucs = []
            # split unlearned set into test-sized chunks
            for i in range(0, len(self.unlearning_nodes), self.mia_num):
                if i + self.mia_num > len(self.unlearning_nodes):
                    break
                original_softlabels_member = original_softlabels_member_full[i:i+self.mia_num]
                unlearning_softlabels_member = unlearning_softlabels_member_full[i:i+self.mia_num]

                mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
                posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
                posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
                posterior = np.array([np.linalg.norm(posterior1[j] - posterior2[j]) for j in range(len(posterior1))])

                auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
                aucs.append(auc)

            self.average_auc[self.run] = np.mean(aucs)
        else:
            self.mia_num = self.unlearning_nodes.size
            original_softlabels_member = self.original_softlabels[self.unlearning_nodes]
            original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
            # if self.target_model_name in ['GCN','SGC',"S2GC"]:
            #     out = self.target_model.model.forward_once(self.data, self.target_model.edge_weight)

            # else:
            #     out = self.target_model.model.forward_once(self.data)
            
            out = self.target_model.model.reason_once_unlearn(self.data)

            unlearning_softlabels_member = out[self.unlearning_nodes]
            unlearning_softlabels_non = out[self.data.test_indices[:self.mia_num]]

            mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
            posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
            posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
            posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
            # self.logger.info("posterior:{}".format(posterior))
            auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            # self.logger.info("auc:{}".format(auc))
            # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
            self.average_auc[self.run] = auc
            return auc
        #---------------------------------------------------- end ----------------------------------------------
        pass



    # def mia_attack_edge(self):

    #     self.mia_num = self.unlearning_edges.shape[1]
    #     neg_edge_index = negative_sampling(edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,num_neg_samples=self.mia_num)
        
    #     original_softlabels_member = F.sigmoid(self.target_model.decode(self.original_feature,self.unlearning_edges)*2-1)
    #     original_softlabels_non = F.sigmoid(self.target_model.decode(self.original_feature,neg_edge_index)*2-1)
        
    #     out_unlearn = self.target_model.model.reason_once_unlearn(self.data)
    #     unlearning_softlabels_member = F.sigmoid(self.target_model.decode(out_unlearn,self.unlearning_edges)*2-1)
    #     unlearning_softlabels_non = F.sigmoid(self.target_model.decode(out_unlearn,neg_edge_index)*2-1)
        
    #     mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
    #     posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
    #     posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
    #     posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        
    #     auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
    #     self.average_auc[self.run] = auc

    def plot_auc(self,y_true,y_score):
        y_true = y_true
        y_score = y_score

        # 计算ROC曲线上的点
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # 计算AUC
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()


    def GIF_attack(self):
        self.deleted_nodes = np.array([])
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])
        self.deleted_edges = np.array([])
        self.influence_edges = np.array([])
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()
        self.target_model_name = self.args['base_model']
        self.determine_target_model()
        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        unlearning_times = np.empty((0))
        training_times = np.empty((0))
        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)

            run_training_time, result_tuple = self.train_unlearning_model(run)
            f1_score = self.evaluate(run)
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)

            # unlearning with GIF
            if self.args["GIF_method"] in ["IF", "GIF"]:
                ## TODO: implement GIF core, return runing time and test f1 score
                unlearning_time, f1_score_unlearning = self.approxi(result_tuple)
                unlearning_times = np.append(unlearning_times, unlearning_time)
                run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)

        f1_score_avg = np.average(run_f1)
        f1_score_std = np.std(run_f1)
        self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))
        self.logger.info("model training time: avg=%s seconds" % np.average(training_times))

        if self.args["GIF_method"] in ["IF", "GIF"]:
            f1_score_unlearning_avg = np.average(run_f1_unlearning)
            f1_score_unlearning_std = np.std(run_f1_unlearning)
            unlearning_time_avg = np.average(unlearning_times)
            self.logger.info("f1_score of %s: avg=%s, std=%s" % (
            self.args["GIF_method"], f1_score_unlearning_avg, f1_score_unlearning_std))
            self.logger.info(
                "%s unlearing time: avg=%s seconds" % (self.args["GIF_method"], np.average(unlearning_time_avg)))



    def gen_train_graph(self):
        self.logger.debug("Before deletion. train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))

        if self.args["unlearn_ratio"] != 0:
            if self.args["unlearn_task"] == 'feature':
                unique_nodes = np.random.choice(len(self.train_indices),
                                                int(len(self.train_indices) * self.args['unlearn_ratio']),
                                                replace=False)
                feature_mask = np.random.choice(a=[0.0, 1.0], size=(len(unique_nodes), self.num_feats),
                                                p=[1.0, 0.0]).astype(np.float32)
                # self.data.x[unique_nodes, :] = self.data.x[unique_nodes, :] * feature_mask
                self.data.x[unique_nodes] = 0.

            else:
                self.data.edge_index = self._ratio_delete()

    def train_test_split(self):
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            # use the dataset's default split
            if self.data.name in ['ogbn-arxiv', 'ogbn-products']:
                # self.train_indices, self.test_indices = self.data.train_indices.numpy(), self.data.test_indices.numpy()
                self.train_indices, self.test_indices = np.array(self.data.train_indices), np.array(self.data.test_indices)
            else:
                self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)),
                                                                         test_size=self.args['test_ratio'],
                                                                         random_state=100)
                # self.train_indices, self.test_indices = np.array(self.data.train_indices), np.array(self.data.test_indices)

            save_train_test_split(self.logger,self.args,self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = load_train_test_split(self.logger)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))


    def _ratio_delete(self):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
        if self.args["unlearn_task"] == 'edge':
            remain_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * (1.0 - self.args['unlearn_ratio'])),
                replace=False)
        else:
            delete_nodes = np.random.choice(
                len(self.train_indices),
                int(len(self.train_indices) * self.args['unlearn_ratio']),
                replace=False)
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    def determine_target_model(self):
        """
        Determines and initializes the target model based on the provided configuration.
        Logs the target model's name, sets the unlearning trainer to 'GIFTrainer',
        and retrieves the trainer instance using the get_trainer function.
        """
        self.logger.info('target model: %s' % (self.args['base_model'],))
        self.args["unlearn_trainer"] = "GIFTrainer"
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
    def _train_model(self, run):
        self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        self.target_model.data = self.data
        # grad_all = self.target_model.GIF_train()
        # self.target_model.train_model()
        self.target_model.train()
        grad_all = self.get_grad((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        train_time = time.time() - start_time

        self.logger.info("Model training time: %s" % (train_time))

        return train_time, grad_all
    
    def train_original_model(self, run):
        """
        Trains the original target model on the dataset without performing any unlearning operations.
        Logs the training process, records the training time, and updates performance metrics.
        For graph-based downstream tasks, it prepares the training and testing datasets accordingly.
        If poisoning is enabled and the unlearning task involves edges, it also evaluates the poisoned F1 score.
        """
        self.logger.info('training target models, run %s' % run)
        if self.args["downstream_task"]=="graph":
            temp_data = copy.deepcopy(self.model_zoo.data)
            train_dataset = [temp_data[i] for i in temp_data.train_indices]
            test_dataset = [temp_data[i] for i in temp_data.test_indices]
            self.train_graph = None
            temp_data = [train_dataset,test_dataset]
            self.target_model.data = temp_data
            self.data = train_dataset
        start_time = time.time()
        # model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"]  +"/"+self.args["downstream_task"]+"/" + self.args["base_model"]
        # if os.path.exists(model_path):
        #     self.target_model.load_model(model_path)
        # else:
        self.target_model.train(save = False)
        train_time = time.time() - start_time
        self.avg_training_time[self.run] = train_time
        # self.original_feature = self.target_model.model.reason_once(self.data).detach()
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.target_model.evaluate()
    
    def get_if_grad(self, run):
        """
        Retrieves the gradients of the target model based on the current unlearning task and downstream task.
        This method processes the necessary unlearning information and computes the corresponding gradients
        required for the unlearning process, supporting node, edge, and graph downstream tasks.
        """
        self.logger.info('training target models, run %s' % run)
        self.target_model.model = self.target_model.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["downstream_task"]=="node":
            res = self.get_grad((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        elif self.args["downstream_task"]=="edge":
            res = self.get_grad_edge((self.deleted_edges,self.feature_nodes,self.influence_edges))
        elif self.args["downstream_task"]=="graph":
            res = self.get_grad_graph((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        return  res
    
    def get_grad(self,unlearn_info=None):
        """
        Computes the gradients of the loss functions with respect to the model parameters for the entire dataset,
        as well as for the subsets of data affected by the unlearning request. This is used to estimate how the
        model parameters should be updated to effectively unlearn the specified data without retraining from scratch.
        """
        grad_all, grad1, grad2 = None, None, None
        self.data = self.data.to(self.device)
        if self.args["GIF_method"] in ["GIF", "IF"]:
            out1 = self.target_model.model.reason_once(self.data)
            out2 = self.target_model.model.reason_once_unlearn(self.data)
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

            model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
            grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
            grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
            grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)      

        return (grad_all, grad1, grad2)
    
    def get_grad_edge(self,unlearn_info=None):
        """
        Computes gradients for unlearning specific components in the target model.
        This method calculates the gradients of the loss with respect to the model parameters
        to facilitate the unlearning of certain elements such as edges, nodes, or features
        based on the specified unlearn_task. It performs forward passes using both the existing
        data and the modified data after unlearning, computes the corresponding losses, and then
        derives the gradients necessary for updating the model parameters accordingly.
        """
        grad_all, grad1, grad2 = None, None, None
        edge_index_r =None
        if self.args["GIF_method"] in ["GIF", "IF"]:
            out1 = self.target_model.model.reason_once(self.data)
            out2 = self.target_model.model.reason_once_unlearn(self.data)
            if self.args["unlearn_task"] == "edge":
                # print(unlearn_info[0].shape,unlearn_info[2].shape)
                edge_index_r = unlearn_info[2]
            elif self.args["unlearn_task"]=="node":
                edge_index_r = np.concatenate((unlearn_info[0], unlearn_info[2]),axis=1)
            elif self.args["unlearn_task"]=="feature":
                edge_index_r = unlearn_info[2]
            # print(edge_index_r.shape)
            loss = self.target_model.get_loss(out1,reduction="sum")
            loss1 = self.get_edge_loss(out1,edge_index=edge_index_r,reduction="sum")
            loss2 = self.get_edge_loss(out2,edge_index=unlearn_info[2],reduction="sum")


            model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
            grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
            grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
            grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
            
                

        return (grad_all, grad1, grad2)

    def get_edge_loss(self,out,edge_index,reduction):
        """
        Computes the binary cross-entropy loss for predicted edges in a graph neural network.
        """
        out_decode = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=-1)

        edge_label = torch.ones(out_decode.shape,dtype=torch.float32,device=self.device)
        # print(out_decode.shape,edge_label.shape)
        loss = F.binary_cross_entropy_with_logits(out_decode,edge_label,reduction=reduction)
        return loss
    # def evaluate(self, run):
    #     self.logger.info('model evaluation')

    #     start_time = time.time()
    #     posterior = self.target_model.posterior()
    #     test_f1 = f1_score(
    #         self.data.y[self.data['test_mask']].cpu().numpy(),
    #         posterior.argmax(axis=1).cpu().numpy(),
    #         average="micro"
    #     )

    #     evaluate_time = time.time() - start_time
    #     self.logger.info("Evaluation cost %s seconds." % evaluate_time)

    #     self.logger.info("Final Test F1: %s" % (test_f1,))
    #     return test_f1

    def get_grad_graph(self,unlearn_info=None):
        """
        Computes gradients for the target model based on the specified unlearning task.
        """
        grad_all, grad1, grad2 = None, None, None
        loss = 0
        loss1 = 0
        loss2 = 0
        # for subgraph in self.data:
        #     print("get_grad_graph",subgraph)
        out1 = self.target_model.model.reason_once(self.data)
        out2 = self.target_model.model.reason_once_unlearn(self.data)
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
    
        loss += self.get_graph_loss(out1,self.data.train_mask)
        loss1 += self.get_graph_loss(out1,mask1)
        loss2 += self.get_graph_loss(out2,mask2)
        
        model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
        return (grad_all, grad1, grad2)
    def get_graph_loss(self,out,mask):
        """
        Compute the total cross-entropy loss for graph-level predictions.
        This function aggregates the loss over all training graphs by selecting node
        embeddings based on the provided mask, computing the mean embedding for each
        graph, and calculating the cross-entropy loss against the target labels.
        """
        total_loss = 0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.to(out.device)
        for gid in self.data.train_ids:
            graph_mask = (self.data.graph_id == gid) & mask
            graph_nodes = out[graph_mask]
            
            if graph_nodes.size(0) == 0:
                continue
            graph_logits = self.target_model.model.linear(graph_nodes.mean(dim=0, keepdim=True))
            # print(graph_logits, self.data.y[gid])
            graph_loss = F.cross_entropy(graph_logits, self.data.y[gid].unsqueeze(0))
            total_loss += graph_loss
        return total_loss

    def unlearning_request(self):
        """
        Handles unlearning requests based on the specified task.
        This method processes the data for unlearning by selecting nodes, edges, or features to remove or modify.
        It updates the model's data accordingly and logs relevant information. Depending on the downstream task,
        it also identifies k-hop neighborhoods related to the unlearning operation.
        """
        if self.args["downstream_task"]=="graph":
            # print(self.data)
            self.data = graph_cls_process(self.data,train_ratio=0.8,val_ratio=0,test_ratio=0.2)
            self.target_model.data = self.data
        self.logger.debug("Train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))

        self.data.x_unlearn = self.data.x.clone()
        if self.args["base_model"] == "SIGN":

            self.data.xs_unlearn = self.data.xs.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.cpu().numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args["unlearn_task"] == 'node':
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            if os.path.exists(path_un):
                unique_nodes = np.loadtxt(path_un, dtype=int)
            else:
                unique_nodes = np.random.choice(len(self.data.train_indices),
                                                int(len(self.data.train_indices) * self.args['unlearn_ratio']),
                                                replace=False)
            unique_edges = edge_index[:,np.where(np.isin(edge_index,unique_nodes))[1]]
            self.unlearning_nodes = unique_nodes
            
            if self.args["base_model"] == "SIGN":
                self.data.edge_index = self.update_edge_index_unlearn(unique_nodes)
            else:
                self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        elif self.args["unlearn_task"] == 'edge':
            path_un_edge = unlearning_edge_path + "_" + str(self.run) + "_edges_" + str(self.args["num_unlearned_edges"]) + ".txt"
            if os.path.exists(path_un_edge):
                self.unlearning_edges = np.loadtxt(path_un_edge, dtype=int).T
            else:
                remove_indices = np.random.choice(
                    unique_indices,
                    int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                    replace=False)
                self.unlearning_edges = edge_index[:, remove_indices]

            # unique_edges = self.unlearning_edges
            unique_nodes = np.unique(self.unlearning_edges)

            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, self.unlearning_edges)

        elif self.args["unlearn_task"] == 'feature':
            unique_nodes = np.random.choice(len(self.data.train_indices),
                                            int(len(self.data.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            # unique_nodes = np.loadtxt(unlearning_path, dtype=int)
            self.unlearning_edges = None
            self.data.x_unlearn[unique_nodes] = 0
        if self.args["downstream_task"] in ["node","graph"]:
            self.find_k_hops(unique_nodes)
        elif self.args["downstream_task"]=="edge":
            self.find_k_hops_edge(unique_nodes,self.unlearning_edges)

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        """
        Updates the edge index by removing specified edges or edges connected to specified nodes based on the unlearning task.
        Depending on the 'unlearn_task' parameter, this function either deletes specific edges provided in `delete_edge_index` or removes all edges connected to nodes listed in `delete_nodes`. The updated edge index is returned as a PyTorch tensor.
        """
        edge_index = self.data.edge_index.cpu().numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args["unlearn_task"] == 'edge':
            edge_set = set(map(tuple, edge_index.T))
            delete_edge_set = set(map(tuple, delete_edge_index.T))
            remaining_edges = edge_set - delete_edge_set
            remain_indices = np.array([i for i, edge in enumerate(edge_index.T) if tuple(edge) in remaining_edges])
            # remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        if len(remain_indices) == 0:
            return torch.empty((2,0), dtype=torch.long)
        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        # print(unique_encode_not,len(remain_encode[0]))
        temp_index = np.searchsorted(unique_encode_not, remain_encode[0], sorter=sort_indices)-1
        temp_indices = sort_indices[temp_index]
        remain_indices_not = unique_indices_not[temp_indices]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    def find_k_hops(self, unique_nodes):
        """
        Finds and sets the influenced nodes within a specified number of hops from the given unique nodes based on the unlearning task.
        """
        edge_index = self.data.edge_index.cpu().numpy()

        ## finding influenced neighbors
        hops = 2
        if self.args["unlearn_task"] == 'node':
            hops = 3
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        if self.args["unlearn_task"] == 'feature':
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        if self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        if self.args["unlearn_task"] == 'edge':
            self.influence_nodes = influenced_nodes

    def find_k_hops_edge(self,unique_nodes,unique_edges):
        """
        Finds and categorizes edges within a specified number of hops from given unique nodes and edges.
        This function identifies edges influenced by the provided unique nodes and edges by exploring their neighbors up to a defined number of hops. The number of hops is determined based on the type of unlearning task. It categorizes the influenced nodes and edges, and determines which nodes and edges should be deleted or influenced according to the task requirements.
        """
        edge_index = self.data.edge_index.cpu().numpy()

        ## finding influenced neighbors
        hops = 2
        if self.args["unlearn_task"] == 'node':
            hops = 3
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        influenced_edges = edge_index[:,np.where(np.isin(edge_index,influenced_nodes))[1]]
        # print("find_k_hops_edge",influenced_edges.shape)
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        neighbor_edges = influenced_edges
        neighbor_edges = neighbor_edges[:, ~np.isin(neighbor_edges.T, unique_edges.T).all(axis=1)]
        # neighbor_edges = np.setdiff1d(influenced_edges,unique_edges)
        if self.args["unlearn_task"] == 'feature':
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
            self.influence_edges = neighbor_edges
        if self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.deleted_edges = unique_edges
            self.influence_nodes = neighbor_nodes
            self.influence_edges = influenced_edges
        if self.args["unlearn_task"] == 'edge':
            self.influence_nodes = influenced_nodes
            self.deleted_edges = unique_edges
            self.influence_edges = neighbor_edges

        

    def approxi(self, res_tuple):
        """
        Approximates parameter changes for model unlearning using gradient information.
        This function processes a tuple of gradients based on the specified unlearning method ('GIF' or 'IF')
        and iteratively updates an estimated parameter change. It adjusts the model's parameters accordingly
        and evaluates the unlearned model's performance by calculating the test F1 score.
        """
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''
        start_time = time.time()
        iteration, damp, scale = self.args['iteration'], self.args['damp'], self.args['scale']
        if self.args["dataset_name"] in ["Photo","Computers","Physics","Questions"]:
            iteration =int(iteration/10)
            # scale *=10
        if self.args["GIF_method"] =="GIF":
            v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        if self.args["GIF_method"] =="IF":
            v = res_tuple[1]
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):

            model_params  = [p for p in self.target_model.model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        test_F1 = self.target_model.eval_unlearn(params_esti)

        import os

        copy_str=""
        if self.args['use_copy']:
            copy_str="_copy"

        run_str=""
        if self.args['num_runs']>1:
            run_str = "_" + str(self.run)

        base_model_str = ""
        if self.args['base_model']!="GCN":  
            base_model_str = "_" + self.args['base_model']

        # Inside the megu_unlearning method, before the return statement
        unlearn_ratio = (self.args["num_unlearned_nodes"]) / self.data.num_nodes
        save_dir = os.path.join(
            "unlearned_models", "GIF",
            self.args['dataset_name'],
            self.args["unlearn_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}"
        )
        os.makedirs(save_dir, exist_ok=True)


        model_name = f"GIF_{self.args['dataset_name']}_{self.args['downstream_task']}_ratio_{unlearn_ratio:.2f}{run_str}{base_model_str}.pt"
        torch.save(params_esti, os.path.join(save_dir, model_name))
        torch.save(self.data.x_unlearn, os.path.join(save_dir, f"x_unlearn{run_str}{base_model_str}.pt"))
        torch.save(self.data.edge_index_unlearn, os.path.join(save_dir, f"edge_index_unlearn{run_str}{base_model_str}.pt"))


        return time.time() - start_time, test_F1

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)

        return_grads = grad(element_product, model_params, create_graph=True)
        return return_grads
    
    def unlearn(self):
        """
        Perform the unlearning process by calculating the gradient influence and approximating the unlearning metrics.
        This method retrieves the gradient information using the current run identifier, computes the unlearning time and 
        F1 score approximation, and updates the respective averages for F1 score and unlearning time.
        """
        result_tuple = self.get_if_grad(self.run)
        unlearning_time, f1_score_unlearning = self.approxi(result_tuple)
        self.average_f1[self.run] = f1_score_unlearning
        self.avg_unlearning_time[self.run] = unlearning_time
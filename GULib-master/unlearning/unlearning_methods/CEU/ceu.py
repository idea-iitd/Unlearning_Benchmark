import copy
import os
import random
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import DataLoader
from collections import defaultdict
from config import unlearning_edge_path, unlearning_path
from attack.Attack_methods.CEU_MIA import _mia_attack
from task import get_trainer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.utils import filter_edge_index_1
from scipy.sparse import csr_matrix
from deeprobust.graph.utils import preprocess
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack, MinMax, Metattack
from unlearning.unlearning_methods.CEU.unlearn import unlearn
from utils.utils import JSD, remove_undirected_edges
from pipeline.IF_based_pipeline import IF_based_pipeline

class ceu(IF_based_pipeline):
    """
    CEU (Certified Edge Unlearning) class for implementing unlearning methods in graph neural networks.
    This class inherits from IF_based_pipeline.

    Class Attributes:
        args (dict): Arguments for the CEU pipeline.

        logger (Logger): Logger for logging information.

        data (dict): Data used in the model.

        model_zoo (ModelZoo): Model zoo containing various models.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self._data = copy.deepcopy(self.data)
        self.model_zoo = model_zoo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.NodeClassifier = NodeClassifier(args,self.data,self.model_zoo,self.logger)
        # self.args["unlearn_trainer"] = 'CEUTrainer'
        # self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        # self.fidelity()
        # self.efficacy()
        

    # def fidelity(self):
    #     self.delete_edges = [100,200,400,800,1000]
    #     test_loader = DataLoader(self.data['test_set'], shuffle=False, batch_size=self.args["test_batch"])
    #     edge_index = torch.tensor(self.data['edges'], device=self.device).t()
    #     result = defaultdict(list)

    #     for _ in tqdm(range(1), desc=f'{self.args["dataset_name"]}-{self.args["base_model"]}'):
    #         original_model = self.target_model.CEU_train(self.data,eval=False, verbose=False,return_epoch=False)
    #         original_res, _ = self.target_model.CEU_test(original_model, test_loader, edge_index)

    #         for num_edges in self.delete_edges:
    #             _, _, adv_res, _ = self.adv_retrain_unlearn(self.data, num_edges)

    #             # Benign
    #             random_edges = []
    #             while len(random_edges) < num_edges * 2:
    #                 v1 = random.randint(0, self.data['num_nodes'] - 1)
    #                 v2 = random.randint(0, self.data['num_nodes'] - 1)
    #                 if (v1, v2) in self.data['edges'] or (v2, v1) in self.data['edges']:
    #                     continue
    #                 random_edges.append((v1, v2))
    #                 random_edges.append((v2, v1))

    #             _data = copy.deepcopy(self.data)
    #             _data['edges'] += random_edges
    #             benign_orig = self.target_model.CEU_train(_data,eval=False, verbose=False)
    #             benign_unlearn, _ = unlearn(self.args,_data, benign_orig, random_edges, device=self.device)

    #             benign_res, _ = self.target_model.CEU_test(benign_unlearn, test_loader, edge_index)
    #             result['# edges'].append(num_edges)
    #             result['setting'].append('Retrain')
    #             result['accuracy'].append(original_res['accuracy'])

    #             result['# edges'].append(num_edges)
    #             result['setting'].append('Benign')
    #             result['accuracy'].append(benign_res['accuracy'])

    #             result['# edges'].append(num_edges)
    #             result['setting'].append('Advesarial')
    #             result['accuracy'].append(adv_res['accuracy'])
        
    #     df = pd.DataFrame(data=result)
    #     print(df.groupby(['# edges', 'setting']).mean())
    #     if not os.path.exists('./result/CEU'):
    #         os.mkdir('./result/CEU')
    #     df.to_csv(f'./result/CEU/fidelity_{self.args["dataset_name"]}_{self.args["base_model"]}.csv')

    # def efficacy(self):
    #     self.delete_edges = [100, 200, 400, 800, 1000]
    #     test_loader = DataLoader(self.data['test_set'], shuffle=False, batch_size=self.args["test_batch"])
    #     edge_index = torch.tensor(self.data['edges'], device=self.device).t()
    #     nodes = torch.tensor(self.data['nodes'], device=self.device)

    #     jsd_result = defaultdict(list)
    #     mia_result = defaultdict(list)


    #     for _ in tqdm(range(1), desc=f'{self.args["dataset_name"]}-{self.args["base_model"]}'):
    #         original_model = self.target_model.CEU_train(self.data,eval=False, verbose=False,return_epoch=False)
    #         original_res, _ = self.target_model.CEU_test(original_model, test_loader, edge_index)

    #         for num_edges in self.delete_edges:
    #             def _efficacy(retrain_model, unlearn_model, edge_index_prime):
    #                 retrain_model.eval()
    #                 with torch.no_grad():
    #                     retrain_post = retrain_model(nodes, edge_index_prime)
    #                 unlearn_model.eval()
    #                 with torch.no_grad():
    #                     unlearn_post = unlearn_model(nodes, edge_index_prime)

    #                 retrain_post = F.softmax(retrain_post, dim=1).cpu().numpy().astype(np.float64)
    #                 unlearn_post = F.softmax(unlearn_post, dim=1).cpu().numpy().astype(np.float64)

    #                 _jsd = JSD(retrain_post, unlearn_post)
    #                 if np.sum(_jsd < 0) > 0:
    #                     _jsd[_jsd < 0] = 0
    #                 return np.mean(_jsd)

    #             adv_model, adv_unlearn_model, A = self.adv_unlearn(self.data, num_edges)

    #             adv_o_pl, adv_r_pl, adv_u_pl = _mia_attack(self.data, adv_model, original_model,
    #                                                        adv_unlearn_model, A, self.device)
    #             edge_index_prime = torch.tensor(remove_undirected_edges(self.data['edges'], A), device=self.device).t()

    #             adv_jsd = _efficacy(original_model, adv_unlearn_model, edge_index_prime)

    #             random_edges = []
    #             while len(random_edges) < num_edges * 2:
    #                 v1 = random.randint(0, self.data['num_nodes'] - 1)
    #                 v2 = random.randint(0, self.data['num_nodes'] - 1)
    #                 if (v1, v2) in self.data['edges'] or (v2, v1) in self.data['edges']:
    #                     continue
    #                 random_edges.append((v1, v2))
    #                 random_edges.append((v2, v1))

    #             _data = copy.deepcopy(self.data)
    #             _data['edges'] += random_edges
    #             benign_orig = self.target_model.CEU_train(_data, eval=False, verbose=False)
    #             benign_unlearn, _ = unlearn(self.args, _data, benign_orig, random_edges, device=self.device)

    #             _edges = remove_undirected_edges(self.data['edges'], random_edges)
    #             edge_index_prime = torch.tensor(_edges, device=self.device).t()
    #             beni_o_pl, beni_r_pl, beni_u_pl = _mia_attack(self.data, benign_orig, original_model,
    #                                                           benign_unlearn, random_edges, self.device)
    #             benign_jsd = _efficacy(original_model, benign_unlearn, edge_index_prime)

    #             jsd_result['# edges'].append(num_edges)
    #             jsd_result['setting'].append('adv')
    #             jsd_result['jsd'].append(np.mean(adv_jsd))

    #             jsd_result['# edges'].append(num_edges)
    #             jsd_result['setting'].append('benign')
    #             jsd_result['jsd'].append(np.mean(benign_jsd))

    #             mia_result['# edges'].append(num_edges)
    #             mia_result['setting'].append('adv')
    #             mia_result['privacy leakage'].append(adv_u_pl)

    #             mia_result['# edges'].append(num_edges)
    #             mia_result['setting'].append('benign')
    #             mia_result['privacy leakage'].append(beni_u_pl)

    #         jsd_df = pd.DataFrame(jsd_result)
    #         jsd_df.to_csv(os.path.join('./result/CEU', f'rq3_jsd_{self.args["dataset_name"]}_{self.args["base_model"]}.csv'))

    #         mia_df = pd.DataFrame(mia_result)
    #         mia_df.to_csv(os.path.join('./result/CEU', f'rq3_mia_{self.args["dataset_name"]}_{self.args["base_model"]}.csv'))

    def adv_retrain_unlearn(self,data, num_edges):
        """
        performs adversarial retraining to facilitate the unlearning process.
        """
        test_loader = DataLoader(data['test_set'])

        A = self.adversarial_adjacency_mat(data, n_perturbations=int(num_edges))
        print()
        print('The number we asked:', num_edges)
        print('The number of adv edges:', len(A) / 2)

        A = [(v1, v2) for v1, v2 in A]

        D_prime = data['edges'] + A

        data_ = copy.deepcopy(data)
        data_['edges'] = D_prime
        orig_model_prime = self.target_model.CEU_train(data_,eval=False, verbose=False)
        edge_index_d_prime = torch.tensor(D_prime, device=self.device).t()
        orig_res_prime, orig_loss_prime = self.target_model.CEU_test(orig_model_prime, test_loader, edge_index_d_prime)

        unlearn_model_prime, _ = unlearn(self.args, data_, orig_model_prime, A, device=self.device)
        edge_index = torch.tensor(data['edges'], device=self.device).t()
        unlearn_res_prime, unlearn_loss_prime = self.target_model.CEU_test(unlearn_model_prime, test_loader, edge_index)

        return orig_res_prime, orig_loss_prime, unlearn_res_prime, unlearn_loss_prime

    def adv_unlearn(self, data, num_edges):
        """
        performs adversarial unlearning. It generates a set of adversarial edges by perturbing the original adjacency matrix with a specified number of edges.
        These adversarial edges are added to the existing edge data to create a modified dataset. The function then trains a new model using this modified data and performs an unlearning process based on the adversarial edges. 
        Finally, it returns the original trained model, the unlearned model, and the list of adversarial edges.
        """
        A = self.adversarial_adjacency_mat(data, n_perturbations=int(num_edges))
        print()
        print('The number we asked:', num_edges)
        print('The number of adv edges:', len(A))
        A = [(v1, v2) for v1, v2 in A]

        D_prime = data['edges'] + A

        data_ = copy.deepcopy(data)
        data_['edges'] = D_prime
        orig_model_prime = self.target_model.CEU_train(data_,eval=False, verbose=False)
        unlearn_model_prime, _ = unlearn(self.args, data_, orig_model_prime, A, device=self.device)
        return orig_model_prime, unlearn_model_prime, A
    def adversarial_adjacency_mat(self,data, n_perturbations=500):
        """
        Generates an adversarial adjacency matrix by introducing perturbations to the original graph structure. 
        The function leverages a surrogate Graph Convolutional Network (GCN) model to identify and apply 
        the most impactful modifications, aiming to degrade the performance of the victim model. It performs 
        a specified number of perturbations, either by adding or removing edges, based on the gradients 
        computed from the chosen loss function. The resulting adversarial adjacency matrix highlights the 
        altered connections that most significantly affect the model's predictions.
        """
        # data = load_data(args)
        n_feat = data['features'].shape[1]

        adj_data, row, col = [], [], []
        for v1, v2 in data['edges']:
            row.append(v1)
            col.append(v2)
            adj_data.append(1)
        adj = csr_matrix((adj_data, (row, col)), shape=(data['num_nodes'], data['num_nodes']))

        features = csr_matrix(data['features'])
        # features = data['features']
        labels = data['labels']
        idx_train = [node for node in data['train_set'].nodes]
        idx_val = [node for node in data['valid_set'].nodes]
        idx_test = [node for node in data['test_set'].nodes]
        idx_unlabeled = np.union1d(idx_val, idx_test)
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)  # 将adj和feature转化为tensor
        # adj = adj.to(device)
        # features = features.to(device)
        # labels = labels.to(device)
        victim_model = GCN(nfeat=n_feat, nclass=data['num_classes'],
                           nhid=16, dropout=0.5, weight_decay=5e-4, device=self.device).to(self.device)
        victim_model.fit(features, adj, labels, idx_train, idx_val=idx_val, patience=10, train_iter=20)

        # Setup Attack Model
        model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=self.device).to(self.device)
        # model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        # model = Metattack(victim_model, nnodes=adj.shape[0], feature_shape=features.shape, device=device).to(device)
        # Attack
        try:
            model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations, attack=20, epochs=20)
            # model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations)
        except AssertionError as err:
            print('Error:', err)
        modified_adj = model.modified_adj.cpu()  # modified_adj is a torch.tensor
        A = torch.cat(torch.where(modified_adj - adj == 1)).view(2, -1).t().tolist()
        return A

    def determine_target_model(self):
        """
        Determines and sets the target model for the CEU pipeline.
        This method logs the base model being used and initializes the target model
        using the provided arguments, logger, model zoo, and data.
        """
        self.logger.info('target model: %s' % (self.args['base_model'],))
        self.args["unlearn_trainer"] = "CEUTrainer"
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self._data)

    def train_original_model(self, run):
        """
        Trains the target model using the provided data and logs the training process. 
        It also evaluates the model on the test set and stores the accuracy if the model is poisoned.
        """
        self.logger.info('training target models, run %s' % run)
        # test_loader = DataLoader(self.data['test_set'], shuffle=False, batch_size=self.args["test_batch"])
        # edge_index = torch.tensor(self.data['edges'], device=self.device).t()
        # result = defaultdict(list)
        start_time = time.time()
        best_f1,avg_training_time = self.target_model.train()
        self.avg_training_time[self.run] = time.time() - start_time
        # self.target_model.model = self.target_model.CEU_train(self.data,eval=False, verbose=False,return_epoch=False)
        # original_res, _ = self.target_model.CEU_test(self.target_model.model, test_loader, edge_index)
        f1 = self.target_model.evaluate()
        self.original_softlabels = self.target_model.get_softlabels()
        if self.args["poison"] and self.args["downstream_task"]=="edge":
            self.poison_f1[self.run] = f1
    def unlearning_request(self):
        """
        This method handles the unlearning request by calculating the number of edges to unlearn, 
        performing adversarial retraining, and generating random edges that are not present in the original data.
        """
        if self.args["unlearn_task"]=="node":
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            if os.path.exists(path_un):
                print("unlearning nodes")
                self.unlearning_nodes = np.loadtxt(path_un)
                self.unlearning_num = self.unlearning_nodes.shape[0]
                self.unlearning_edges = filter_edge_index_1(self.data, self.unlearning_nodes).T
        elif self.args["unlearn_task"]=="edge":
            path_un = unlearning_edge_path + "_"+str(self.run)+".txt"
            if os.path.exists(path_un):
                print("unlearning edges")
                self.unlearning_edges = np.loadtxt(path_un)
        # _, _, adv_res, _ = self.adv_retrain_unlearn(self.data, num_edges)
        self._data = copy.deepcopy(self.data)
        edge_index_set = set(map(tuple, self.data.edge_index.T.tolist()))
        unlearn_edge_set = set(map(tuple, self.unlearning_edges.tolist()))
        retain_edge_set = edge_index_set - unlearn_edge_set
        retain_edge_index = np.array(list(retain_edge_set)).T
        self._data.edge_index = torch.from_numpy(retain_edge_index)
       
    
    def unlearn(self):
        """
        Performs the unlearning process by inferencing without the specified random edges.
        Evaluates the unlearned model on the test set and updates the average accuracy metric.
        """
        self.target_model.data = self._data
        # self.target_model.train()
        start_time = time.time()
        self.target_model.model, _ = unlearn(self.args,self._data, self.target_model.model, self.unlearning_edges, device=self.device)
        breakpoint()
        self.avg_unlearning_time[self.run] = time.time() - start_time
        f1 = self.target_model.evaluate()
        self.average_f1[self.run] = f1
        
    def mia_attack(self):
        # """
        # This function performs a Membership Inference Attack (MIA) to evaluate the model's vulnerability to such attacks after the unlearning process. It calculates the AUC score by comparing the model's predictions on unlearned nodes versus test nodes, thereby assessing the effectiveness of the unlearning method in protecting against membership inference.
        # """
        # self.data = self.data.to(self.device)
        # self.mia_num = self.unlearning_num
        # original_softlabels_member = self.original_softlabels[self.unlearning_nodes]
        # original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]

        # unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x,self.data.edge_index),dim = 1)[self.unlearning_nodes]
        # unlearning_softlabels_non = F.softmax(self.target_model.model(
        #         self.data.x,self.data.edge_index),dim = 1)[self.data.test_indices[:self.mia_num]]

        # mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        # posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        # posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        # posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        # # self.logger.info("posterior:{}".format(posterior))
        # auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        # self.average_auc[self.run] = auc
        # # self.logger.info("auc:{}".format(auc))
        # # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
        # return auc
        pass
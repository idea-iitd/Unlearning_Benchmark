import numpy as np
import time
import torch

from utils.dataset_utils import *
import copy
from torch_geometric.utils import negative_sampling
from config import BLUE_COLOR,RESET_COLOR
import torch.nn.functional as F
from task import get_trainer
from sklearn.metrics import roc_auc_score
from attack.MIA_attack import train_attack_model,train_shadow_model,generate_shadow_model_output,evaluate_attack_model,GCNShadowModel,AttackModel
from pipeline.Learning_based_pipeline import Learning_based_pipeline
class d2dgn(Learning_based_pipeline):
    """
    D2DGN class implements a learning-based pipeline for performing unlearning tasks on GNNs. 
    It supports both node and edge unlearning, executing unlearning requests, training models, and evaluating the impact through membership inference attacks.
    It gets the knowledge from the teacher model and trains the target model using teacher knowledge distillation. 

    Class Attributes:
        args (dict): Configuration parameters for the unlearning process.

        logger (Logger): Logger instance for recording informational and debugging messages.

        model_zoo (ModelZoo): Collection of pre-trained models available for training and evaluation within the pipeline.

        strategy (int): Indicator of the unlearning strategy employed (e.g., 1 for KL divergence, 2 for MSE loss).
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.args["unlearn_trainer"] = 'D2DGNTrainer'
        
        self.num_classes = self.data.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unlearning_number = args["num_unlearned_nodes"]
        num_runs = self.args["num_runs"]
        self.avg_partition_time = np.zeros(num_runs)
        self.average_f1 = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)
        self.avg_updating_time = np.zeros(num_runs)
        self.attack_auc_scores = np.zeros(num_runs)
        self.strategy = 1
        # self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')


    # def run_exp(self):
    #     # self.train_test_split()
    #     self.unlearn()
    #     self.pre_train()

    #     for run in range(self.args['num_runs']):
    #         self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
    #         # print(self.data)
    #         best_f1,avg_training_time=self.target_model.d2dgn_train(self.preserver_knowledge,self.destroyer_knowledge,self.loss_fn)
    #         attack_auc_score = self.mia_attack()
    #         self.average_f1[run] = best_f1
    #         self.avg_training_time[run]=avg_training_time
    #         self.attack_auc_scores[run] = attack_auc_score
    #     self.logger.info(
    #     "{}Performance Metrics:\n"
    #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average Training Time: {:.4f} ± {:.4f} seconds\n"
    #     " - Average Attack AUC Score: {:.4f} ± {:.4f}\n".format(
    #         BLUE_COLOR,
    #         np.mean(self.average_f1), np.std(self.average_f1),
    #         np.mean(self.avg_training_time), np.std(self.avg_training_time),
    #         np.mean(self.attack_auc_scores), np.std(self.attack_auc_scores),
    #         RESET_COLOR
    #         )
    #     )
        

    def train_test_split(self):
        """
        Splits the dataset into training and testing sets based on the specified ratios.
        If a split has not been performed, it randomly selects training and testing indices,
        saves the split, and creates corresponding masks. If a split already exists, it
        loads the precomputed training and testing indices and updates the masks accordingly.
        """
        if not self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.data.train_indices, self.data.test_indices = train_test_split(np.arange((self.data.num_nodes)),train_size = 0.6,test_size=self.args['test_ratio'], random_state=100)
            save_train_test_split(self.logger,self.args,self.data.train_indices, self.data.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.test_indices))
            print(self.data.train_indices.size, self.data.test_indices.size)
        else:
            self.data = load_train_test_split(self.logger)
            # self.data.train_indices, self.data.test_indices

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.test_indices))


    def unlearning_request(self):
        """
        Handles unlearning requests by removing specified nodes or edges from the graph data.
        Depending on the unlearning task, it updates the training and testing masks,
        removes the relevant edges or nodes, and prepares the data for retraining.
        """
        if self.args["unlearn_task"]=="node":
            
            path_un = config.unlearning_path + "_" + str(self.run) + ".txt"
            if os.path.exists(path_un):
                with open(path_un) as file:
                    unlearn_indices = [int(line.rstrip()) for line in file]
            else:
                unlearn_indices = np.random.choice(self.data.train_indices, self.unlearning_number, replace=False)
            
            # Create masks for the nodes to keep and to unlearn
            keep_mask = ~np.isin(self.data.train_indices, unlearn_indices)
            unlearn_mask = np.isin(self.data.train_indices, unlearn_indices)
            
            # Create subgraphs for the kept nodes and unlearned nodes
            self.data.train_indices = np.array(self.data.train_indices)
            self.data.unlearn_indices = self.data.train_indices[unlearn_mask]
            self.data.keep_indices = self.data.train_indices[keep_mask]

            self.data.keep_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.keep_indices))
            self.data.unlearn_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.unlearn_indices))

            # Remove edges connected to the unlearned nodes
            edge_index = self.data.edge_index.cpu().numpy()
            keep_edges_mask = ~np.isin(edge_index[0], self.data.unlearn_indices) & ~np.isin(edge_index[1], self.data.unlearn_indices)
            self.data.keep_edge_index = torch.tensor(edge_index[:, keep_edges_mask], dtype=torch.long)
            self.data.unlearn_edge_index = torch.tensor(edge_index[:, ~keep_edges_mask], dtype=torch.long)
            
            self.logger.info(f'Unlearned {self.unlearning_number} nodes.')
            # self.logger.info(f'Kept {len(self.data.keep_indices)} nodes.')
        elif self.args["unlearn_task"]=="edge":
            self.data.keep_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            self.data.unlearn_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.data.train_indices))
            
            path_un_edge = unlearning_edge_path + "_" + str(self.run) + ".txt"
            if os.path.exists(path_un_edge):
                unlearn_edge_index = np.loadtxt(path_un_edge, dtype=int)
                
            edge_index_set = set(map(tuple, self.data.train_edge_index.t().cpu().numpy()))
            unlearn_edge_set = set(map(tuple, unlearn_edge_index))
            keep_edge_set = edge_index_set - unlearn_edge_set

            # Update edge_index to remove unlearned edges
            self.data.keep_edge_index = torch.tensor(list(keep_edge_set)).t().contiguous()
            self.data.unlearn_edge_index = torch.tensor(list(unlearn_edge_set)).t().contiguous()
            
            self.logger.info(f'Unlearned {self.unlearning_number} edges.')
        self.del_data = copy.deepcopy(self.data)
        neg_edge_index = negative_sampling(
                edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.edge_index.size(1)
            )
        self.del_data.edge_index = neg_edge_index
    def pre_train(self):
        """
        Initializes the preserver and destroyer trainers based on the chosen strategy, retrieves their knowledge representations, and prepares the models for the unlearning process by setting up loss functions and device configurations.
        """
        self.preserver = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        self.preserver.train()
        # for name, param in self.preserver.model.named_parameters():
        #     param.requires_grad = False
        if self.strategy == 1:
            self.destroyer = get_trainer(self.args,self.logger,self.model_zoo.model,self.del_data)

            self.loss_fn = "KL"
            self.preserver_knowledge = self.preserver.model(self.data.x, self.data.edge_index,return_all_emb=False)
            self.destroyer_knowledge = self.destroyer.model(self.data.x,self.data.edge_index,return_all_emb=False)
            if self.args["downstream_task"]=="node":
                self.preserver_knowledge = F.softmax(self.preserver_knowledge.detach(),dim=-1).to(self.device)
                self.destroyer_knowledge = F.softmax(self.destroyer_knowledge.detach(),dim=-1).to(self.device)
            else:
                self.preserver_knowledge = self.preserver_knowledge.detach().to(self.device)
                self.destroyer_knowledge = self.destroyer_knowledge.detach().to(self.device)
        elif self.strategy ==2:
            self.destroyer = get_trainer(self.args,self.logger,self.model_zoo.model,self.del_data)

            self.loss_fn = "MSE"
            self.preserver_knowledge = self.preserver.model(self.data.x, self.data.edge_index,return_all_emb=True)
            self.destroyer_knowledge = self.destroyer.model(self.data.x,self.data.edge_index,return_all_emb=True)
            for i,know in enumerate(self.preserver_knowledge):
                self.preserver_knowledge[i] = know.detach().to(self.device)
            for i,know in enumerate(self.destroyer_knowledge):
                self.destroyer_knowledge[i] = know.detach().to(self.device)
        elif self.strategy ==3:
            self.destroyer = get_trainer(self.args,self.logger,self.model_zoo.model,self.del_data)
            self.destroyer.train()

            self.loss_fn = "MSE"
            self.preserver_knowledge = self.preserver.model(self.data.x, self.data.edge_index,return_all_emb=True)
            self.destroyer_knowledge = self.destroyer.model(self.data.x,self.data.edge_index,return_all_emb=True)
            for i,know in enumerate(self.preserver_knowledge):
                self.preserver_knowledge[i] = know.detach().to(self.device)
            for i,know in enumerate(self.destroyer_knowledge):
                self.destroyer_knowledge[i] = know.detach().to(self.device)
        

    def mia_attack(self):
        """
        This function performs a Membership Inference Attack (MIA) to evaluate the model's vulnerability to such attacks after the unlearning process. It calculates the AUC score by comparing the model's predictions on unlearned nodes versus test nodes, thereby assessing the effectiveness of the unlearning method in protecting against membership inference.
        """
        self.mia_num = self.unlearning_number
        self.original_softlabels = F.softmax(self.target_model.model(
            self.data.x,self.data.edge_index),dim=1).clone().detach().float()
        original_softlabels_member = self.original_softlabels[self.data.unlearn_indices]
        original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
        # if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args[
        #     "base_model"] == "SIGN":
        #     unlearning_softlabels_member = F.softmax(self.target_model.model(self.features[self.unlearning_nodes]),dim=1)
        #     unlearning_softlabels_non = self.target_model.model.get_softlabel(
        #         self.features[self.data.test_indices[:self.mia_num]])
        # else:
        unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x,self.data.edge_index)[self.data.unlearn_indices],dim=1)
        unlearning_softlabels_non = F.softmax(self.target_model.model(
            self.data.x,self.data.edge_index)[self.data.test_indices[:self.mia_num]],dim=1)

        mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        # self.logger.info("posterior:{}".format(posterior))
        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        self.average_auc[self.run] = auc
        # self.logger.info("auc:{}".format(auc))
        # self.plot_auc(mia_test_y, posterior.reshape(-1, 1))
        return auc
    

    # def mia_attack(self):
    #     shadow_model = GCNShadowModel(self.data.num_features,64 ,self.data.num_classes)
    #     shadow_model = train_shadow_model(shadow_model,self.data)
    #     soft_labels_train,soft_labels_test = generate_shadow_model_output(shadow_model,self.data)

    #     attack_mdoel = train_attack_model(soft_labels_train,self.data.num_classes)
    #     attack_auc_score = evaluate_attack_model(attack_mdoel,soft_labels_test)
    #     self.average_auc[self.run] = attack_auc_score
    #     # return attack_auc_score
        
    def train_original_model(self):
        """
        Trains the original target model. Initializes the target model using the provided trainer, sets it to training mode, and evaluates its performance if poisoning is enabled for edge unlearning tasks.
        """
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        self.target_model.train()
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.target_model.evaluate()
        pass
    def unlearn(self):
        """
        Handles the unlearning process by updating the model to eliminate the influence of specified nodes or edges.
        This method supports different unlearning strategies, measures the time taken for training and unlearning,
        and updates the model's performance metrics accordingly.
        """
        start_time = time.time()
        self.preserver = copy.deepcopy(self.target_model)
        if self.strategy == 1:
            self.destroyer = get_trainer(self.args,self.logger,self.model_zoo.model,self.del_data)
            if not self.data.x.is_cuda:
                self.data.x = self.data.x.cuda()
            if not self.data.edge_index.is_cuda:
                self.data.edge_index = self.data.edge_index.cuda()
            self.loss_fn = "KL"
            self.preserver_knowledge = self.preserver.model(self.data.x, self.data.edge_index,return_all_emb=False)
            self.destroyer_knowledge = self.destroyer.model(self.data.x,self.data.edge_index,return_all_emb=False)
            if self.args["downstream_task"]=="node":
                self.preserver_knowledge = F.softmax(self.preserver_knowledge.detach(),dim=-1).to(self.device)
                self.destroyer_knowledge = F.softmax(self.destroyer_knowledge.detach(),dim=-1).to(self.device)
            else:
                self.preserver_knowledge = self.preserver_knowledge.detach().to(self.device)
                self.destroyer_knowledge = self.destroyer_knowledge.detach().to(self.device)
        elif self.strategy ==2:
            self.destroyer = get_trainer(self.args,self.logger,self.model_zoo.model,self.del_data)

            self.loss_fn = "MSE"
            self.preserver_knowledge = self.preserver.model(self.data.x, self.data.edge_index,return_all_emb=True)
            self.destroyer_knowledge = self.destroyer.model(self.data.x,self.data.edge_index,return_all_emb=True)
            for i,know in enumerate(self.preserver_knowledge):
                self.preserver_knowledge[i] = know.detach().to(self.device)
            for i,know in enumerate(self.destroyer_knowledge):
                self.destroyer_knowledge[i] = know.detach().to(self.device)
        elif self.strategy ==3:
            self.destroyer = get_trainer(self.args,self.logger,self.model_zoo.model,self.del_data)
            self.destroyer.train()

            self.loss_fn = "MSE"
            self.preserver_knowledge = self.preserver.model(self.data.x, self.data.edge_index,return_all_emb=True)
            self.destroyer_knowledge = self.destroyer.model(self.data.x,self.data.edge_index,return_all_emb=True)
            for i,know in enumerate(self.preserver_knowledge):
                self.preserver_knowledge[i] = know.detach().to(self.device)
            for i,know in enumerate(self.destroyer_knowledge):
                self.destroyer_knowledge[i] = know.detach().to(self.device)
        self.avg_training_time[self.run] = time.time()-start_time
        
        best_f1,avg_unlearning_time=self.target_model.d2dgn_train(self.preserver_knowledge,self.destroyer_knowledge,self.loss_fn)
        self.average_f1[self.run] = best_f1
        self.avg_unlearning_time[self.run]=avg_unlearning_time
        
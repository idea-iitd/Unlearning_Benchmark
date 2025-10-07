import numpy as np
import networkx as nx
import time
import torch


from task.node_classification import NodeClassifier
from unlearning.unlearning_methods.GraphEraser.aggregation.aggregator import Aggregator
from utils.dataset_utils import *
from pipeline.Shard_based_pipeline import Shard_based_pipeline
import config
from utils.utils import *
from torch_geometric.data import Data
from unlearning.unlearning_methods.GraphEraser.partition.graph_partition import GraphPartition
from utils import utils, dataset_utils
from dataset.original_dataset import *
from attack.Attack_methods.GraphEraser_MIA import GraphEraser_Attack
from task.edge_prediction import EdgePredictor
from collections import defaultdict
from task import get_trainer
BLUE_COLOR = "\033[34m"
RESET_COLOR = "\033[0m"

class grapheraser(Shard_based_pipeline):
    """
    GraphEraser Class handles tasks such as generating training graphs, partitioning the graph into shards, managing shard data, 
    performing unlearning operations on nodes, edges, or features, and evaluating the effectiveness of these unlearning operations. 
    This class extends the `Shard_based_pipeline` to leverage shard-based processing for scalability and efficiency.

    Class Attributes:
    args (dict): Configuration parameters for the unlearning process, including methods for partitioning, number of shards, base model specifications, and unlearning task parameters.
    
    logger (Logger): Logger instance for logging informational and debugging messages.
    
    model_zoo (ModelZoo): Collection of models and associated data used for training and unlearning.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        # self.original_data = original_data
        self.data = copy.deepcopy(model_zoo.data)
        self.model_zoo = model_zoo
        self.logger = logger
        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']
        self.target_model_name = self.args['base_model']
        num_runs = self.args["num_runs"]
        self.run = 0
        self.affected_shard = []
        
    def gen_train_graph(self):
        """
        This function is designed to generate a training graph based on the specified downstream task.
        """
        if self.args["downstream_task"] != "graph":
            edge_index = self.data.edge_index.detach().cpu().numpy()
            if self.args["downstream_task"] == "node":
                test_edge_indices = np.logical_or(np.isin(edge_index[0], self.data.test_indices),
                                                np.isin(edge_index[1], self.data.test_indices))
                train_edge_indices = np.logical_not(test_edge_indices)
                edge_index_train = edge_index[:, train_edge_indices]
            else:
                edge_index_train = self.data.train_edge_index.detach().cpu().numpy()
            self.train_graph = nx.Graph()
            self.train_graph.add_nodes_from(self.data.train_indices)
            # use largest connected graph as train graph
            if self.args['is_prune']:
                self._prune_train_set()
            # reconstruct a networkx train graph
            for u, v in np.transpose(edge_index_train):
                self.train_graph.add_edge(u, v)

            self.logger.info("After edge deletion. train graph  #.Nodes: %f, #.Edges: %f" % (
                self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
            self.logger.info("After edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
                self.data.num_nodes, self.data.num_edges))
            save_train_data(self.logger,self.data,config.train_data_file)
            save_train_graph(self.logger,self.train_graph,config.train_graph_file)
        else:
            assert self.args['partition_method'] == "graph_km" and self.args["aggregator"] == "mean"
            self.data = copy.deepcopy(self.model_zoo.data)
            train_dataset = [self.data[i] for i in self.data.train_indices]
            test_dataset = [self.data[i] for i in self.data.test_indices]
            self.train_graph = None
            self.data = [train_dataset,test_dataset]
            save_train_data(self.logger,self.data,config.train_data_file)
            save_train_graph(self.logger,self.train_graph,config.train_graph_file)
        
    def graph_partition(self):
        """
        This function is designed to partition the training graph into shards based on the specified partition.
        """
        if self.args['is_partition']:
            self.logger.info('graph partitioning')

            start_time = time.time()
            partition = GraphPartition(self.logger, self.args,self.train_graph, self.data,model_zoo = self.model_zoo)
            self.community_to_node = partition.graph_partition()
            partition_time = time.time() - start_time
            self.logger.info("Partition cost %s seconds." % partition_time)
            self.avg_partition_time[self.run]  = partition_time

            save_community_data(self.logger,self.community_to_node,config.community_path)
        else:
            self.community_to_node = load_community_data(self.logger,config.community_path)

        self.logger.info(partition)
            

    def generate_shard_data(self):
        """
        This function is designed to generate shard data based on the partitioned graph and the specified downstream task.
        """
        if self.args["downstream_task"] != "graph":
            self.shard_path = config.shard_file
            self.shard_data = {}
            # test_edge_index = torch.tensor(utils.filter_edge_index_1(self.data,self.data.test_indices))
            
            for shard in range(self.args['num_shards']):
                train_shard_indices = list(self.community_to_node[shard])
                shard_indices = np.union1d(train_shard_indices, self.data.test_indices)
                if self.args["downstream_task"]=="node":
                    x = self.data.x[shard_indices]
                    edge_index = utils.filter_edge_index_1(self.data, shard_indices)
                    y = self.data.y[shard_indices]
                    data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
                    data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
                    data.test_mask = torch.from_numpy(np.isin(shard_indices, self.data.test_indices))
                else:
                    x = self.data.x
                    train_edge_index = torch.from_numpy(utils.filter_edge_index_3(self.data,shard_indices))
                    test_edge_index = self.data.test_edge_index
                    edge_index = torch.cat([train_edge_index.cuda(),test_edge_index.cuda()],dim=1)
                    y = self.data.y
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.train_edge_index = train_edge_index
                    data.test_edge_index = test_edge_index
                
                if self.args["base_model"] == "SIGN":
                    data = SIGN(self.args["GNN_layer"])(data)
                    data.xs = [data.x] + [data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                    # data.xs = torch.tensor([x.detach().numpy() for x in data.xs]).cuda()
                    data.xs = torch.stack(data.xs).cuda()
                    data.xs = data.xs.transpose(0, 1)

                self.shard_data[shard] = data
                # self.logger.info(data)
            save_shard_data(self.logger,self.shard_data,self.shard_path)
        else:
            self.shard_path = config.shard_file
            self.shard_data = {}
            for community_id, graphs in self.community_to_node.items():
                self.shard_data[community_id] = [self.data[0][i] for i in graphs]
            for shard,train_data in self.shard_data.items():
                self.shard_data[shard] = [train_data,self.data[1]]
            save_shard_data(self.logger,self.shard_data,self.shard_path)

    def load_data(self):
        """
        This function is designed to load the preprocessed data for the unlearning process.
        """
        self.shard_data = dataset_utils.load_shard_data(self.logger)
        self.train_data = dataset_utils.load_saved_data(self.logger,config.train_data_file)
        self.unlearned_shard_data = self.shard_data
        self.logger.info(self.shard_data)
        
    def determine_target_model(self):
        """
        This function is designed to determine the target model for the unlearning process based on the specified downstream task.
        """
        self.target_model = get_trainer(self.args, self.logger, self.model_zoo.model,self.data)
        
    def run_exp_train(self):
        """
        This is the pipeline for training the target model on the shard data.
        """
        self.train_target_models(self.run)
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.aggregate(self.run)
    # def run_exp(self):
    #     for self.run in range(self.args["num_runs"]):
    #         self.args["exp"] = "partition"
    #         self.train_test_split()
    #         self.gen_train_graph()
    #         self.graph_partition()
    #         self.generate_shard_data()
    #         self.args["exp"] = "unlearning"
    #         self.shard_data = dataset_utils.load_shard_data(self.logger)
    #         self.train_data = dataset_utils.load_saved_data(self.logger,config.train_data_file)
    #         self.unlearned_shard_data = self.shard_data
    #         self.logger.info(self.shard_data)
    #         self.target_model = get_trainer(self.args, self.logger, self.model_zoo.model,self.data)
    #         self.run_exp_train()
    #         self.args["exp"] = "attack_unlearning"
    #         self.train_test_split()
    #         self.model_zoo.data = self.data
    #         GraphEraser_Attack(self.args, self.logger, self.original_data, self.model_zoo,self.avg_unlearning_time,self.average_f1,self.average_auc,self.run)




        # # 输出带有红色文字的日志
        # self.logger.info(
        #     "{}Performance Metrics:\n"
        #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
        #     " - Average AUC Score: {:.4f} ± {:.4f}\n"
        #     " - Average Partition Time: {:.4f} ± {:.4f} seconds\n"
        #     " - Average Training Time: {:.4f} ± {:.4f} seconds\n"
        #     " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
        #         BLUE_COLOR,
        #         np.mean(self.average_f1), np.std(self.average_f1),
        #         np.mean(self.average_auc), np.std(self.average_auc),
        #         np.mean(self.avg_partition_time), np.std(self.avg_partition_time),
        #         np.mean(self.avg_training_time), np.std(self.avg_training_time),
        #         np.mean(self.avg_unlearning_time), np.std(self.avg_unlearning_time),
        #         RESET_COLOR
        #     )
        # )
        # self.logger.info(self.avg_partition_time)
        # self.logger.info(self.avg_unlearning_time)



    def train_test_split(self):
        """
        This function is designed to split the training and testing data based on the specified test ratio.
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

    # def graph_partition(self):

    #     if self.args['is_partition']:
    #         self.logger.info('graph partitioning')

    #         start_time = time.time()
    #         partition = GraphPartition(self.logger, self.args,self.train_graph, self.data,model_zoo = self.model_zoo)
    #         self.community_to_node = partition.graph_partition()
    #         partition_time = time.time() - start_time
    #         self.logger.info("Partition cost %s seconds." % partition_time)
    #         self.avg_partition_time[self.run]  = partition_time

    #         save_community_data(self.logger,self.community_to_node,config.community_path)
    #     else:
    #         self.community_to_node = load_community_data(self.logger,config.community_path)

    #     # self.logger.info(partition)

    def train_shard_model(self):
        """
        This function is designed to train the shard model based on the specified downstream task.
        """
        self.train_target_models(self.run)
        
    def aggregate_shard_model(self):
        """
        This function is designed to aggregate the shard models based on the specified downstream task and calculate the score.
        """
        aggregate_f1_score = self.aggregate(self.run)
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = aggregate_f1_score
        
    def generate_requests(self):
        """
        This function is designed to generate unlearning requests, and record the unlearning indices or edges.
        """
        self.load_preprocessed_data()
        if self.args["downstream_task"] == "graph":
            return
        if self.args["unlearn_task"] == "node":
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            if os.path.exists(path_un):
                with open(path_un) as file:
                    self.node_unlearning_indices = [int(line.rstrip()) for line in file]
            else:
                shuffle_num = torch.randperm(self.data.train_indices.size)
                self.node_unlearning_indices = self.data.train_indices[shuffle_num][:self.args["num_unlearned_nodes"]]
                
                with open(path_un,mode="w") as file:
                    for node in self.node_unlearning_indices:
                        file.write(str(node) + '\n')
            # self.logger.info(self.node_unlearning_indices)
        elif self.args["unlearn_task"] == "edge":
            path_un_edge = config.unlearning_edge_path + "_" + str(self.run) + "_edges_" + str(self.args["num_unlearned_edges"]) + ".txt"
            if os.path.exists(path_un_edge):
                self.unlearning_edges = np.loadtxt(path_un_edge, dtype=int).T
            else:
                train_edges = np.array(self.data.train_edge_index)
                shuffle_num = torch.randperm(self.train_edges.shape[1])
                num_unlearned_edges = int(train_edges.shape[1] * self.args["unlearn_ratio"])
                self.unlearning_edges = train_edges[:, shuffle_num][:, :num_unlearned_edges]
                np.savetxt(path_un_edge, self.unlearning_edges.T, fmt="%d")
                
        elif self.args["unlearn_task"] == "feature":
            path_un = config.unlearning_path + "_" + str(self.run) + ".txt"
            if os.path.exists(path_un):
                with open(path_un) as file:
                    self.node_unlearning_indices = [int(line.rstrip()) for line in file]
            else:
                shuffle_num = torch.randperm(self.data.train_indices.size)
                self.node_unlearning_indices = self.data.train_indices[shuffle_num][:self.args["num_unlearned_nodes"]]
                with open(path_un,mode="w") as file:
                    for node in self.node_unlearning_indices:
                        file.write(str(node) + '\n')
    
    def unlearn(self):
        """
        This function is designed to perform unlearning operations based on the specified downstream task and unlearning task. And then calculate the score after unlearning.
        """
        start_time = time.time()
        if self.args["downstream_task"] != "graph":
            if self.args["unlearn_task"] == "node":
                self.graph_node_unlearning_request_respond(self.node_unlearning_indices)
            elif self.args["unlearn_task"] == "edge":
                self.graph_edge_unlearning_request_respond(self.unlearning_edges)
            elif self.args["unlearn_task"] == "feature":
                self.graph_feature_unlearning_request_respond(self.node_unlearning_indices)
            self.average_f1[self.run] = self.f1_score

        else:
            self.graph_graph_unlearning_request_respond()
            for shard in self.affected_shard:
                shard = int(shard)
                suffix = "_unlearned"
                self._train_shard_model(shard, suffix)

            aggregate_acc = self.aggregate(self.run)
            self.average_acc[self.run] = aggregate_acc
        unlearning_time = time.time() - start_time
        self.avg_unlearning_time[self.run] = unlearning_time
        self.logger.info("unlearning_time:{}".format(unlearning_time))
                
        
        
    def attack_unlearning(self):
        """
        Performs attack-based unlearning based on specified tasks.
        """
        if self.args["downstream_task"] == "graph":
            return
        if self.args["unlearn_task"] == "node":
            self.logger.info("start cal AUC")
            self.attack_graph_unlearning(self.average_auc)
            
        
    #############
    
    def load_preprocessed_data(self):
        """
        Loads and prepares preprocessed data for the model based on the specified downstream task.
        This method retrieves shard data, training data, and the training graph. It also handles the train-test split,
        loads community-to-node mappings, and initializes the target model for node classification or other tasks.
        """
        if self.args["downstream_task"] != "graph":
            self.shard_data = dataset_utils.load_shard_data(self.logger)
            # self.raw_data = self.original_data.load_data()
            self.train_data = dataset_utils.load_saved_data(self.logger,config.train_data_file)
            self.train_graph = dataset_utils.load_train_graph(self.logger)
            data = dataset_utils.load_train_test_split(self.logger)
            self.train_indices, self.test_indices = data.train_indices,data.test_indices
            self.community_to_node = dataset_utils.load_community_data(self.logger)
            num_feats = self.train_data.num_features
            num_classes = len(self.train_data.y.unique())
            #9.20
            # self.target_model = NodeClassifier(self.args,self.shard_data,self.model_zoo,self.logger)
            self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.shard_data)

        
    def attack_graph_unlearning(self,average_auc):
        """
        Performs an attack to evaluate the effectiveness of graph unlearning.
        This function loads unlearned node indices from a specified path, queries the target model to obtain
        posterior probabilities for both member (positive) and non-member (negative) samples, and evaluates
        the attack's performance using the average AUC metric.
        """
        # load unlearned indices
        # path_un = config.unlearning_path + "_" + str(self.run) + ".txt"
        path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
        with open(path_un) as file:
            unlearned_indices = [int(line.rstrip()) for line in file]

        unlearned_indices = unlearned_indices[:100]
        # member sample query, label as 1
        positive_posteriors = self._query_target_model(unlearned_indices, unlearned_indices)
        # non-member sample query, label as 0
        # negative_posteriors = self._query_target_model(self.data.test_indices[0:self.args["num_unlearned_nodes"]],
        #                                                self.data.test_indices[0:self.args["num_unlearned_nodes"]])
        negative_posteriors = self._query_target_model(self.data.test_indices[0:100],
                                                       self.data.test_indices[0:100])
        
        # evaluate attack performance, train multiple shadow models, or calculate posterior entropy, or directly calculate AUC.
        self.evaluate_attack_performance(positive_posteriors, negative_posteriors,average_auc)

    def _query_target_model(self, unlearned_indices, test_indices):
        """
        Queries the target model to obtain posterior probabilities for specified indices.
        This function loads the unlearned training data and aggregates posterior
        probabilities from the target model's submodels for each index in 
        `unlearned_indices`. Depending on the configuration, it may also perform
        repartitioning of shard data and collect additional posterior probabilities.
        """
        # load unlearned data
        train_data = dataset_utils.load_unlearned_data(self.logger,'train_data')
        # load optimal weight score
        # optimal_weight=self.data_store.load_optimal_weight(0)

        # calculate the final posterior, save as attack feature
        self.logger.info('aggregating submodels')
        posteriors_a, posteriors_b, posteriors_c = [], [], []

        for i in tqdm(unlearned_indices, desc="MIA Progress"):
            community_to_node = dataset_utils.load_community_data(self.logger,config.load_community_data,'')
            shard_data = self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, int(i))

            posteriors_a.append(self._generate_posteriors(shard_data, ''))

            shard_num = self.node_to_com.get(i)
            posteriors_b.append(self._generate_posteriors_unlearned(shard_data))

            if self.args['repartition']:
                suffix = "_repartition_unlearned_" + str(i)
                community_to_node = dataset_utils.load_community_data(self.logger,config.load_community_data,suffix)
                shard_data = self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node,
                                                                               int(i))
                suffix = "_repartition_unlearned_" + str(i)
                posteriors_c.append(self._generate_posteriors(shard_data, suffix))
        return posteriors_a, posteriors_b, posteriors_c

    def _generate_posteriors_unlearned(self, shard_data):
        """
        Generates and returns the averaged posterior probabilities for each shard of the dataset after unlearning.
        This function iterates over all data shards and determines whether each shard is affected by the unlearning process.
        For affected shards, it loads the corresponding unlearned model; otherwise, it loads the standard model. 
        """
        posteriors = []
        for shard in range(self.args['num_shards']):
            if shard in self.affected_shard:
                suffix = "_unlearned"
                # load the retrained the shard model
                dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard, suffix)
            else:
                # self.target_model.model.reset_parameters()
                # load unaffected shard model
                dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard, '')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.target_model.model = self.target_model.model.to(self.device)
            self.target_model.data = shard_data[shard].to(self.device)
            # if self.args['base_model'] == "SAGE":
            #     posteriors.append(self.target_model.posterior())
            # else:
            #9.20
            # posteriors.append(self.target_model.posterior_other())
            posteriors.append(self.target_model.posterior())

        return torch.mean(torch.cat(posteriors, dim=0), dim=0)
    
    def _generate_unlearned_repartitioned_shard_data(self, train_data, community_to_node, test_indices):
        """
        Generates unlearned and repartitioned shard data for training.
        This function partitions the training data into multiple shards based on the provided community-to-node mapping.
        For each shard, it combines the training shard indices with the test indices to create a subset of the data.
        It processes the subset by filtering the relevant edges and preparing the data for the specified base model.
        """
        # self.logger.info('generating shard data')

        shard_data = {}
        for shard in range(self.args['num_shards']):
            train_shard_indices = list(community_to_node[shard])
            shard_indices = np.union1d(train_shard_indices, test_indices)

            x = self.train_data.x[shard_indices]
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(train_data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            if self.args["base_model"] == "SIGN":
                data = SIGN(self.args["GNN_layer"])(data)
                data.xs = [data.x] + [data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                # data.xs = torch.tensor([x.detach().numpy() for x in data.xs]).cuda()
                data.xs = torch.stack(data.xs).cuda()
                data.xs = data.xs.transpose(0, 1)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, test_indices))

            shard_data[shard] = data

        # self.data_store.save_unlearned_data(shard_data, 'shard_data_repartition')
        return shard_data


    def evaluate_attack_performance(self, positive_posteriors, negative_posteriors,average_auc):
        """
        Evaluates the performance of attack models using posterior probabilities.
        This function constructs attack data by combining positive and negative posterior probabilities, calculates the L2 distance between different model posteriors, and evaluates the attack performance by computing the Area Under the Curve (AUC) metrics. The AUC results for each attack model are logged, and the primary attack AUC is stored in the provided average_auc dictionary.
        """
        # constrcut attack data
        label = torch.cat((torch.ones(len(positive_posteriors[0])), torch.zeros(len(negative_posteriors[0]))))
        data={}
        for i in range(2):
             data[i] = torch.cat((torch.stack(positive_posteriors[i]), torch.stack(negative_posteriors[i])),0)

        # calculate l2 distance
        model_b_distance = self._calculate_distance(data[0], data[1])
        # directly calculate AUC with feature and labels
        attack_auc_b = self.evaluate_attack_with_AUC(model_b_distance, label)
        attack_auc_c = 0
        if self.args['repartition']:
            model_c_distance = self._calculate_distance(data[0], data[2])
            attack_auc_c = self.evaluate_attack_with_AUC(model_c_distance, label)

        self.logger.info("Attack_Model_B AUC: %s | Attack_Model_C AUC: %s" % (attack_auc_b, attack_auc_c))
        average_auc[self.run] = attack_auc_b


    def _generate_posteriors(self, shard_data, suffix):
        """
        Generates and aggregates posterior distributions across data shards.
        This method iterates over the specified number of data shards, loading the target model for each shard, 
        and computing the posterior distribution using the target model. 
        It then aggregates all posterior distributions by concatenating them and computing their mean.
        """
        posteriors = []
        for shard in range(self.args['num_shards']):
            # self.target_model.model.reset_parameters()
            dataset_utils.load_target_model(self.logger,self.args, self.run, self.target_model, shard, suffix)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.target_model.model = self.target_model.model.to(self.device)
            self.target_model.data = shard_data[shard].to(self.device)

            # if self.args['base_model'] == "SAGE":
            #     posteriors.append(self.target_model.posterior())
            # else:
            #9.20
            # posteriors.append(self.target_model.posterior_other())
            posteriors.append(self.target_model.posterior())
        return torch.mean(torch.cat(posteriors, dim=0), dim=0)


    def evaluate_attack_with_AUC(self, data, label):
        """
        Calculates the ROC AUC score to evaluate attack performance.
        This function computes the AUC score to assess the effectiveness of an attack by comparing the predicted data against the true labels.
        """
        from sklearn.metrics import roc_auc_score
        self.logger.info("Directly calculate the attack AUC")
        return roc_auc_score(label, data.reshape(-1, 1))

    def _calculate_distance(self, data0, data1, distance='l2_norm'):
        """
        Calculate the distance between two datasets using the specified metric.
        This method computes the distance between corresponding elements of `data0` and `data1`
        based on the chosen distance metric. Supported metrics include:
        - 'l2_norm': Calculates the Euclidean (L2) norm between each pair of elements.
        - 'direct_diff': Computes the direct difference between elements.
        """
        if distance == 'l2_norm':
            return np.array([np.linalg.norm(data0[i] - data1[i]) for i in range(len(data0))])
        elif distance == 'direct_diff':
            return data0 - data1
        else:
            raise Exception("Unsupported distance")
        
    def graph_graph_unlearning_request_respond(self):
        """
        Processes graph unlearning requests by randomly selecting and modifying shards.
        This method selects a subset of shards based on the specified number of shards and performs
        unlearning operations on each selected shard. Depending on the unlearning task, it either
        nullifies a percentage of node features or removes a percentage of nodes entirely from the
        graph data. The updated shard data reflects the unlearning modifications, ensuring that
        the specified unlearning requirements are met.
        """
        shard_index  = int(0.5 * self.args["num_shards"])
        remove_shard = torch.randperm(self.args["num_shards"])[:shard_index]
        for shard in remove_shard:
            shard = int(shard)
            self.affected_shard.append(shard)
            data = self.unlearned_shard_data[shard][0]
            for i in range(len(data)):
                data2update = data[i]
                num_nodes_to_remove = int(0.1 * data2update.num_nodes)  # 计算5%的节点数
                remove_indices = torch.randperm(data2update.num_nodes)[:num_nodes_to_remove]  # 随机选择节点索引
                if self.args["unlearn_task"] == 'feature':
                    data2update.x[remove_indices] = 0
                else:
                    node_mask = torch.ones(data2update.num_nodes, dtype=torch.bool)
                    node_mask[remove_indices] = False  # 删除的节点设为 False

                    # 3. 更新节点特征和标签
                    x_new = data2update.x[node_mask]
                    y = data2update.y

                    # 4. 更新边索引
                    # 找出与删除节点无关的边
                    edge_mask = node_mask[data2update.edge_index[0]] & node_mask[data2update.edge_index[1]]
                    edge_index_new = data2update.edge_index[:, edge_mask]

                    # 重新映射边的索引到新的节点编号
                    mapping = torch.zeros(data2update.num_nodes, dtype=torch.long)
                    mapping[node_mask] = torch.arange(node_mask.sum())
                    edge_index_new = mapping[edge_index_new]

                    # 5. 构建新的 data 对象
                    data_new = Data(x=x_new, edge_index=edge_index_new, y=y)
                    data2update = data_new
                data[i] = data2update
        self.shard_data_after_unlearning = self.unlearned_shard_data
            
    def graph_node_unlearning_request_respond(self, node_unlearning_request=None):
        """
        Processes node unlearning requests by removing specified nodes from the training graph,
        updating data shards, and retraining affected models. This function handles the reindexing
        of node IDs, updates training and testing masks, manages shard data after unlearning, and
        optionally repartitions the graph before retraining shard models.
        """
        # reindex the node ids
        self.node_to_com = dataset_utils.c2n_to_n2c(self.args,self.community_to_node)
        train_indices_prune = list(self.node_to_com.keys())
        # breakpoint()
        if len(node_unlearning_request) == 0:
            # generate node unlearning requests
            node_unlearning_indices = np.random.choice(train_indices_prune, self.args['num_unlearned_nodes'])
        else:
            node_unlearning_indices = np.array([node_unlearning_request])[0]
        self.num_unlearned_edges = 0
        self.unlearning_indices = defaultdict(list)
        for node in node_unlearning_indices:
            node = int(node)
            node_key = self.node_to_com.get(node)
            if node_key is not None:
                self.unlearning_indices[node_key].append(node)
            else:
                # 处理键不存在的情况，可能需要打印一些信息或者采取其他措施
                print(f"The key {node} does not exist in the dictionary.")

            # unlearning_indices[node_to_com[node]].append(node)

        # delete a list of revoked nodes from train_graph
        self.train_graph.remove_nodes_from(node_unlearning_indices)

        # delete the revoked nodes from train_data
        # by building unlearned data from unlearned train_graph
        self.train_data.train_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), self.train_indices))
        self.train_data.test_mask = torch.from_numpy(
            np.isin(np.arange(self.train_data.num_nodes), np.append(self.test_indices, node_unlearning_indices)))

        # delete the revoked nodes from shard_data
        self.shard_data_after_unlearning = {}
        for shard in range(self.args["num_shards"]):
            train_shard_indices = list(self.community_to_node[shard])
            # node unlearning
            train_shard_indices = np.setdiff1d(train_shard_indices, self.unlearning_indices[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

            x = self.train_data.x[shard_indices]
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.train_data, shard_indices)
            
            edge_index_set = set(tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1]))
            test_edge_index = torch.tensor(utils.filter_edge_index_1(self.data,self.data.test_indices))
            test_edge_index_set = set(tuple(test_edge_index[:, i].tolist()) for i in range(test_edge_index.shape[1]))
            train_edge_index_set = edge_index_set - test_edge_index_set
            train_edge_index = torch.tensor(list(train_edge_index_set)).T
            
            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            
            if self.args["base_model"] == "SIGN":
                data = SIGN(self.args["GNN_layer"])(data)
                data.xs = [data.x] + [data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                # data.xs = torch.tensor([x.detach().numpy() for x in data.xs]).cuda()
                data.xs = torch.stack(data.xs).cuda()
                data.xs = data.xs.transpose(0, 1)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))
            data.train_edge_index = train_edge_index
            data.test_edge_index = test_edge_index
            # utils.get_inductive_edge(data)
            self.logger.info(data)
            self.shard_data_after_unlearning[shard] = data
            self.num_unlearned_edges += self.shard_data[shard].num_edges - self.shard_data_after_unlearning[
                shard].num_edges

            # find the affected shard model
            if self.shard_data_after_unlearning[shard].num_nodes != self.shard_data[shard].num_nodes:
                self.affected_shard.append(shard)
    
        dataset_utils.save_unlearned_data(self.logger,self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger,self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger,self.shard_data_after_unlearning, 'shard_data')

        # retrain the correponding shard model
        # if not self.args['repartition']:
        #     for shard in self.affected_shard:
        #         tmp = ""
        #         for node_id in self.unlearning_indices[shard]:
        #             tmp += str(node_id)
        #             tmp += "&"
        #         tmp = tmp[0:-1]
        #         suffix = "_unlearned_" + str(tmp)
        #         self._train_shard_model(shard, suffix)
        if not self.args['repartition']:
            for shard in self.affected_shard:
                suffix = "_unlearned"
                self._train_shard_model(shard, suffix)

        # (if re-partition, re-partition the remaining graph)
        # re-train the shard model, save model and optimal weight score
        if self.args['repartition']:
            suffix = "_repartition_unlearned_" + str(node_unlearning_indices[0])
            self._repartition(suffix)
            for shard in range(self.args["num_shards"]):
                self._train_shard_model(shard, suffix)


        f1 = self.aggregate(self.run)
        # self.logger.info("F1: {}".format(f1))
        self.f1_score = f1

    def graph_edge_unlearning_request_respond(self,all_edges_to_remove):
        """
        Processes an edge unlearning request by removing specified edges from the graph. 
        This function updates the relevant data shards, identifies and retrains affected shard models, 
        saves the updated data, and aggregates evaluation metrics after unlearning.
        """
        self.node_to_com = dataset_utils.c2n_to_n2c(self.args,self.community_to_node)
        self.num_unlearned_edges = 0
        self.shard_data_after_unlearning = {}
        # print(self.train_data.train_edge_index,all_edges_to_remove)
        for shard in range(self.args["num_shards"]):
            train_shard_indices = list(self.community_to_node[shard])
            # node unlearning
            shard_indices = np.union1d(train_shard_indices, self.test_indices)
            if self.args["downstream_task"]=="edge":
                x = self.train_data.x
            else:
                x = self.train_data.x[shard_indices]
            
            
            y = self.train_data.y[shard_indices]

            train_edge_index = utils.filter_edge_index_3(self.train_data, shard_indices,all_edges_to_remove)
            # test_edge_index = self.train_data.test_edge_index
            # edge_index = np.concatenate((train_edge_index,test_edge_index.detach().cpu()),axis=1)
            test_edge_index = utils.filter_edge_index_3(self.train_data, shard_indices,all_edges_to_remove,edge_set="test")
            edge_index = np.concatenate((train_edge_index,test_edge_index),axis=1)
            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            
            if self.args["base_model"] == "SIGN":
                data = SIGN(self.args["GNN_layer"])(data)
                data.xs = [data.x] + [data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                # data.xs = torch.tensor([x.detach().numpy() for x in data.xs]).cuda()
                data.xs = torch.stack(data.xs).cuda()
                data.xs = data.xs.transpose(0, 1)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))
            data.train_edge_index = torch.from_numpy(train_edge_index)
            data.test_edge_index = test_edge_index
            # utils.get_inductive_edge(data)
            print("unlearning_data",data)
            self.shard_data_after_unlearning[shard] = data
            self.num_unlearned_edges += self.shard_data[shard].num_edges - self.shard_data_after_unlearning[
                shard].num_edges
            # find the affected shard model
            print("num_edges_unlearn:",self.shard_data_after_unlearning[shard].num_edges,"num_edges:",self.shard_data[shard].num_edges)
            if self.shard_data_after_unlearning[shard].num_edges != self.shard_data[shard].num_edges:
                self.affected_shard.append(shard)

        dataset_utils.save_unlearned_data(self.logger,self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger,self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger,self.shard_data_after_unlearning, 'shard_data')
        for shard in self.affected_shard:
            print("shard:",shard)
            suffix = "_unlearned"
            self._train_shard_model(shard, suffix)


        f1 = self.aggregate(self.run)
        # self.logger.info("F1: {}".format(f1))
        self.f1_score = f1

    def graph_feature_unlearning_request_respond(self, node_unlearning_request=None):
        """
        Handles feature unlearning requests by nullifying specified node features. 
        """
        # reindex the node ids
        self.node_to_com = dataset_utils.c2n_to_n2c(self.args,self.community_to_node)
        train_indices_prune = list(self.node_to_com.keys())

        if len(node_unlearning_request) == 0:
            # generate node unlearning requests
            node_unlearning_indices = np.random.choice(train_indices_prune, self.args['num_unlearned_nodes'])
        else:
            node_unlearning_indices = np.array([node_unlearning_request])[0]
        self.num_unlearned_edges = 0
        self.unlearning_indices = defaultdict(list)
        for node in node_unlearning_indices:
            node = int(node)
            node_key = self.node_to_com.get(node)
            if node_key is not None:
                self.unlearning_indices[node_key].append(node)
            else:
                # 处理键不存在的情况，可能需要打印一些信息或者采取其他措施
                print(f"The key {node} does not exist in the dictionary.")

            # unlearning_indices[node_to_com[node]].append(node)

        # delete a list of revoked nodes from train_graph
        self.train_graph.remove_nodes_from(node_unlearning_indices)

        # delete the revoked nodes from train_data
        # by building unlearned data from unlearned train_graph
        self.train_data.train_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), self.train_indices))
        self.train_data.test_mask = torch.from_numpy(
            np.isin(np.arange(self.train_data.num_nodes), np.append(self.test_indices, node_unlearning_indices)))

        # delete the revoked nodes from shard_data
        self.shard_data_after_unlearning = {}
        for shard in range(self.args["num_shards"]):
            train_shard_indices = list(self.community_to_node[shard])
            # node unlearning
            # train_shard_indices = np.setdiff1d(train_shard_indices, self.unlearning_indices[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)
            self.train_data.x[self.unlearning_indices[shard]] = torch.zeros_like(self.train_data.x[self.unlearning_indices[shard]])
            x = self.train_data.x[shard_indices]
            
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.train_data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            
            if self.args["base_model"] == "SIGN":
                data = SIGN(self.args["GNN_layer"])(data)
                data.xs = [data.x] + [data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                # data.xs = torch.tensor([x.detach().numpy() for x in data.xs]).cuda()
                data.xs = torch.stack(data.xs).cuda()
                data.xs = data.xs.transpose(0, 1)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))
            # utils.get_inductive_edge(data)
            self.shard_data_after_unlearning[shard] = data
            self.num_unlearned_edges += self.shard_data[shard].num_edges - self.shard_data_after_unlearning[
                shard].num_edges

            # find the affected shard model
            if self.shard_data_after_unlearning[shard].num_nodes != self.shard_data[shard].num_nodes:
                self.affected_shard.append(shard)

        dataset_utils.save_unlearned_data(self.logger,self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger,self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger,self.shard_data_after_unlearning, 'shard_data')

        # retrain the correponding shard model
        # if not self.args['repartition']:
        #     for shard in self.affected_shard:
        #         tmp = ""
        #         for node_id in self.unlearning_indices[shard]:
        #             tmp += str(node_id)
        #             tmp += "&"
        #         tmp = tmp[0:-1]
        #         suffix = "_unlearned_" + str(tmp)
        #         self._train_shard_model(shard, suffix)
        if not self.args['repartition']:
            for shard in self.affected_shard:
                suffix = "_unlearned"
                self._train_shard_model(shard, suffix)

        # (if re-partition, re-partition the remaining graph)
        # re-train the shard model, save model and optimal weight score
        if self.args['repartition']:
            suffix = "_repartition_unlearned_" + str(node_unlearning_indices[0])
            self._repartition(suffix)
            for shard in range(self.args["num_shards"]):
                self._train_shard_model(shard, suffix)


        f1 = self.aggregate(self.run)
        # self.logger.info("F1: {}".format(f1))
        self.f1_score = f1
        
    def _repartition(self, suffix):
        """
        Repartitions the training graph and data using the provided suffix.
        """
        # load unlearned train_graph and train_data
        train_graph = dataset_utils.load_unlearned_data(self.logger,'train_graph')
        train_data = dataset_utils.load_unlearned_data(self.logger,'train_data')
        # repartition
        start_time = time.time()
        partition = GraphPartition(self.args,self.logger, train_graph, train_data)
        community_to_node = partition.graph_partition()
        partition_time = time.time() - start_time
        self.logger.info("Partition cost %s seconds." % partition_time)
        # save the new partition and shard
        dataset_utils.save_community_data(self.logger,community_to_node,config.load_community_data, suffix)
        self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, self.test_indices)
        
        
    def _train_shard_model(self, shard, suffix="_unlearned"):
        """
        Trains a shard-specific model using the unlearned training data.
        """
        self.logger.info('training target models, shard %s' % shard)

        # load shard data
        self.target_model.data = self.shard_data_after_unlearning[shard]
        # breakpoint()
        print("unlearned train data:",self.shard_data_after_unlearning[shard])
        # retrain shard model

        self.target_model.train(needs_retrain=True)
        # auc = self.target_model.evaluate()
        # print("auc:",auc)
        # replace shard model
        # move Data object (PyG Data supports .to)
        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.target_model.device = device
        dataset_utils.save_target_model(self.logger,self.args, self.run, self.target_model, shard, suffix)
        # self.data_store.save_unlearned_target_model(0, self.target_model, shard, suffix)
        
        

    def _ratio_delete_edges(self, edge_index):
        """
        Deletes a specified ratio of edges from the given edge index.
        """
        edge_index = edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
        remain_indices = np.random.choice(unique_indices,
                                           int(unique_indices.shape[0] * (1.0 - self.args['ratio_deleted_edges'])),
                                           replace=False)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        # self.data.edge_index = torch.from_numpy(edge_index[:, remain_indices])
        return torch.from_numpy(edge_index[:, remain_indices])

    def _prune_train_set(self):
        """
        Prune the training set by extracting the largest connected component from the training graph.
        This method logs the number of nodes and edges before pruning, identifies the maximum
        connected component within the training graph, updates the training graph to this component,
        and logs the number of nodes and edges after pruning.
        """
        # extract the the maximum connected component
        self.logger.debug("Before Prune...  #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

        self.train_graph = max(connected_component_subgraphs(self.train_graph), key=len)

        self.logger.debug("After Prune... #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

    def train_target_models(self, run):
        """
        Trains the target models based on the provided run configuration.
        """
        if self.args['is_train_target_model']:
            self.logger.info('training target models')

            self.time = {}
            for shard in range(self.args['num_shards']):
                self.time[shard] = self._train_model(run, shard)
                self.avg_training_time[self.run] += self.time[shard]
            self.avg_training_time[self.run] = self.avg_training_time[self.run]/ self.args['num_shards']/self.args["num_epochs"]

    def _train_model(self, run, shard):
        """
        Trains the target model using the specified run and data shard.
        This method logs the training initiation, assigns the unlearned shard data to the target model, and sets the model to training mode. It measures the training duration, saves the updated model, and returns the time taken for the training process.
        """
        self.logger.info('training target models, run %s, shard %s' % (run, shard))

        start_time = time.time()
        self.target_model.data = self.unlearned_shard_data[shard]
        # print("train_data:",self.target_model.data.edge_index)
        self.target_model.train()


        #nodeClassifier = NodeClassifier(self.args, self.target_model.data, self.model_zoo, self.logger)
        train_time = time.time() - start_time
        save_target_model(self.logger,self.args,run, self.target_model, shard)
        
        return train_time

    def aggregate(self, run):
        """
        Aggregates submodel results and computes the final F1 score.
        This method initializes an Aggregator with the current run configuration, target model, training data, 
        and unlearned shard data. It generates posterior probabilities, performs the aggregation of data to 
        compute the F1 score, logs the final test F1 score, and returns the aggregated F1 score.
        """
        self.logger.info('aggregating submodels')

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(run, self.target_model, self.train_data, self.unlearned_shard_data, self.args,self.logger,self.affected_shard)
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate(self.data)

        self.logger.info("Final Test F1: %s" % (self.aggregate_f1_score,))
        return self.aggregate_f1_score


    def construct_graph(self):
        """
        Constructs a comprehensive graph by aggregating node features, edge indices, labels, and graph IDs from multiple communities.
        This function iterates through each community and its associated graphs, adjusts edge indices to maintain global consistency, and concatenates all node attributes into a single large graph represented by a `Data` object. The resulting graph includes node features (`x`), edge indices (`edge_index`), labels (`y`), and graph IDs (`graph_id`) to identify the origin of each node.
        """           
        all_x = []             # 节点特征
        all_edge_index = []    # 边索引
        all_y = []             # 节点标签
        all_graph_id = []      # 每个节点所属的图ID标签，用于标识节点来源

        node_offset = 0        # 用于跟踪节点索引偏移量

        for community,graphs in self.community_to_node.items():
            for i, data in enumerate(graphs):
                # 收集节点特征和标签
                all_x.append(data.x)
                all_y.append(data.y)
                
                # 调整边索引以适应全局大图
                edge_index = data.edge_index + node_offset
                all_edge_index.append(edge_index)
                
                # 将每个节点的图ID记录下来
                all_graph_id.append(torch.full((data.num_nodes,), i, dtype=torch.long))
                
                # 更新节点偏移量
                node_offset += data.num_nodes

            # 将所有节点特征、边索引和标签进行拼接
            big_x = torch.cat(all_x, dim=0)
            big_edge_index = torch.cat(all_edge_index, dim=1)
            big_y = torch.cat(all_y, dim=0)
            big_graph_id = torch.cat(all_graph_id, dim=0)

        # 构建大图的 Data 对象
        big_graph = Data(x=big_x, edge_index=big_edge_index, y=big_y, graph_id=big_graph_id)
        data = big_graph
import numpy as np
import networkx as nx
import time
import torch

import config
from task.node_classification import NodeClassifier
from unlearning.unlearning_methods.GraphEraser.aggregation.aggregator import Aggregator
from utils.dataset_utils import *
import config
from utils.utils import *
from torch_geometric.data import Data
from unlearning.unlearning_methods.GraphEraser.partition.graph_partition import GraphPartition
from utils import utils, dataset_utils
from dataset.original_dataset import *
from attack.Attack_methods.GraphEraser_MIA import GraphEraser_Attack

class grapheraser:
    def __init__(self,args,original_data,logger,model_zoo):
        self.args = args
        self.original_data = original_data
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.logger = logger
        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']
        self.target_model_name = self.args['base_model']
        self.run = 0
        self.average_f1 = np.zeros(3)
        self.average_auc = np.zeros(3)
        self.avg_partition_time = np.zeros(3)
        self.avg_training_time = np.zeros(3)
        self.avg_unlearning_time = np.zeros(3)
    

        #ExpGraphPartition
        # if self.args["exp"] == "partition":
        #     self.train_test_split()
        #     self.gen_train_graph()
        #     self.graph_partition()
        #     self.generate_shard_data()
        # #ExpUnlearning
        # elif self.args["exp"] == "unlearning":
        #     self.shard_data = dataset_utils.load_shard_data(self.logger)
        #     self.train_data = load_saved_data(self.logger,config.train_data_file)
        #     self.unlearned_shard_data = self.shard_data
        #     self.logger.info(self.shard_data)
        #     self.target_model = NodeClassifier(args,self.data,model_zoo,logger)
        #     self.run_exp()
        # elif self.args["exp"] == "attack_unlearning":
        #     #要将is_split设置为true
        #     self.train_test_split()
        #     model_zoo.data = self.data
        #     GraphEraser_Attack(args, logger, original_data, model_zoo,self.avg_unlearning_time,self.average_f1,self.average_auc,self.run)
        # elif self.args["exp"] == "sequence":
        #     self.args["exp"] == "partition"
        #     self.train_test_split()
        #     self.gen_train_graph()
        #     self.graph_partition()
        #     self.generate_shard_data()
        #     self.args["exp"] = "unlearning"
        #     self.shard_data = dataset_utils.load_shard_data(self.logger)
        #     self.train_data = load_saved_data(self.logger,config.train_data_file)
        #     self.unlearned_shard_data = self.shard_data
        #     self.logger.info(self.shard_data)
        #     self.target_model = NodeClassifier(args,self.data,model_zoo,logger)
        #     self.run_exp()
        #     self.args["exp"] = "attack_unlearning"
        #     for self.run in range(3):
        #         self.train_test_split()
        #         model_zoo.data = self.data
        #         GraphEraser_Attack(args, logger, original_data, model_zoo,self.avg_unlearning_time,self.average_f1,self.average_auc,self.run)

        #     self.logger.info("average_f1:{}±{}".format(np.mean(self.average_f1),np.std(self.average_f1)))
        #     self.logger.info("average_auc:{}±{}".format(np.mean(self.average_auc),np.std(self.average_auc)))
        #     self.logger.info("avg_partition_time:{}±{}".format(np.mean(self.avg_partition_time),np.std(self.avg_partition_time)))
        #     self.logger.info("avg_training_time:{}±{}".format(np.mean(self.avg_training_time),np.std(self.avg_training_time)))
        #     self.logger.info("avg_unlearning_time:{}±{}".format(np.mean(self.avg_unlearning_time),np.std(self.avg_unlearning_time)))

    def run_exp(self):
        self.args["exp"] == "partition"
        self.train_test_split()
        self.gen_train_graph()
        self.graph_partition()
        self.generate_shard_data()
        self.args["exp"] = "unlearning"
        self.shard_data = dataset_utils.load_shard_data(self.logger)
        self.train_data = load_saved_data(self.logger, config.train_data_file)
        self.unlearned_shard_data = self.shard_data
        self.logger.info(self.shard_data)
        self.target_model = NodeClassifier(self.args,self.data,self.model_zoo,self.logger)
        self.run_exp_train()
        self.args["exp"] = "attack_unlearning"
        for self.run in range(self.args["num_runs"]):
            self.train_test_split()
            self.model_zoo.data = self.data
            GraphEraser_Attack(self.args, self.logger, self.original_data, self.model_zoo,self.avg_unlearning_time,self.average_f1,self.average_auc,self.run)

        self.logger.info("average_f1:{}±{}".format(np.mean(self.average_f1),np.std(self.average_f1)))
        self.logger.info("average_auc:{}±{}".format(np.mean(self.average_auc),np.std(self.average_auc)))
        self.logger.info("avg_partition_time:{}±{}".format(np.mean(self.avg_partition_time),np.std(self.avg_partition_time)))
        self.logger.info("avg_training_time:{}±{}".format(np.mean(self.avg_training_time),np.std(self.avg_training_time)))
        self.logger.info("avg_unlearning_time:{}±{}".format(np.mean(self.avg_unlearning_time),np.std(self.avg_unlearning_time)))
    


    def train_test_split(self):
        if not self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.data.train_indices, self.data.test_indices = train_test_split(np.arange((self.data.num_node)),train_size = 0.6,test_size=self.args['test_ratio'], random_state=100)
            save_train_test_split(self.logger,self.args,self.data.train_indices, self.data.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_node), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_node), self.data.test_indices))
            print(self.data.train_indices.size, self.data.test_indices.size)
        else:
            self.data = load_train_test_split(self.logger)
            # self.data.train_indices, self.data.test_indices

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_node), self.data.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_node), self.data.test_indices))



    def gen_train_graph(self):
        if self.args['ratio_deleted_edges'] != 0:
            self.logger.info("Before edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
                self.data.num_node, self.data.num_edges))

            # self._ratio_delete_edges()
            self.data.edge_index = self._ratio_delete_edges(self.data.edge_index)

        # decouple train test edges.
        edge_index = self.data.edge_index.numpy()

        test_edge_indices = np.logical_or(np.isin(edge_index[0], self.data.test_indices),
                                        np.isin(edge_index[1], self.data.test_indices))
        train_edge_indices = np.logical_not(test_edge_indices)
        edge_index_train = edge_index[:, train_edge_indices]

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
            self.data.num_node, self.data.num_edges))
        save_train_data(self.logger, self.data, config.train_data_file)
        save_train_graph(self.logger, self.train_graph, config.train_graph_file)

    def graph_partition(self):

        if self.args['is_partition']:
            self.logger.info('graph partitioning')

            start_time = time.time()
            partition = GraphPartition(self.logger, self.args,self.train_graph, self.data)
            self.community_to_node = partition.graph_partition()
            partition_time = time.time() - start_time
            self.logger.info("Partition cost %s seconds." % partition_time)
            self.avg_partition_time[self.run]  = partition_time

            save_community_data(self.logger, self.community_to_node, config.community_path)
        else:
            self.community_to_node = load_community_data(self.logger, config.community_path)

        self.logger.info(partition)

    def generate_shard_data(self):
        self.shard_path = config.shard_file
        # self.logger.info('generating shard data')

        self.shard_data = {}
        for shard in range(self.args['num_shards']):
            train_shard_indices = list(self.community_to_node[shard])
            shard_indices = np.union1d(train_shard_indices, self.data.test_indices)

            x = self.data.x[shard_indices]
            y = self.data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            if self.args["base_model"] == "SIGN":
                data = SIGN(self.args["GNN_layer"])(data)
                data.xs = [data.x] + [data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                # data.xs = torch.tensor([x.detach().numpy() for x in data.xs]).cuda()
                data.xs = torch.stack(data.xs).cuda()
                data.xs = data.xs.transpose(0, 1)

            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.data.test_indices))
            utils.get_inductive_edge(data)

            self.shard_data[shard] = data
            self.logger.info(data)
        save_shard_data(self.logger,self.shard_data,self.shard_path)

    def run_exp_train(self):
        run_f1 = np.empty((0))
        unlearning_time = np.empty((0))
        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)
            self.train_target_models(run)
            aggregate_f1_score = self.aggregate(run)
            # node_unlearning_time = self.unlearning_time_statistic()
            node_unlearning_time = 0
            run_f1 = np.append(run_f1, aggregate_f1_score)
            unlearning_time = np.append(unlearning_time, node_unlearning_time)
        self.num_unlearned_edges = 0
        # model utility
        self.f1_score_avg = np.average(run_f1)
        self.f1_score_std = np.std(run_f1)
        self.unlearning_time_avg = np.average(unlearning_time)
        self.unlearning_time_std = np.std(unlearning_time)
        
    

    def _ratio_delete_edges(self, edge_index):
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
        # extract the the maximum connected component
        self.logger.debug("Before Prune...  #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

        self.train_graph = max(connected_component_subgraphs(self.train_graph), key=len)

        self.logger.debug("After Prune... #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

    def train_target_models(self, run):
        if self.args['is_train_target_model']:
            self.logger.info('training target models')

            self.time = {}
            for shard in range(self.args['num_shards']):
                self.time[shard] = self._train_model(run, shard)
                self.avg_training_time[self.run] += self.time[shard]
            self.avg_training_time[self.run] = self.avg_training_time[self.run]/ self.args['num_shards']/self.args["num_epochs"]

    def _train_model(self, run, shard):
        self.logger.info('training target models, run %s, shard %s' % (run, shard))

        start_time = time.time()
        self.target_model.data = self.unlearned_shard_data[shard]
        # if self.args["base_model"] == "SIGN":
        #     self.target_model.data = SIGN(self.args["GNN_layer"])(self.target_model.data)
        #     self.target_model.data.xs = [self.target_model.data.x] + [self.target_model.data[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
        #     self.target_model.data.xs = torch.tensor([x.detach().numpy() for x in self.target_model.data.xs]).cuda()
        #     self.target_model.data.xs = self.target_model.data.xs.transpose(0, 1)

        self.target_model.train_model()
        #nodeClassifier = NodeClassifier(self.args, self.target_model.data, self.model_zoo, self.logger)
        train_time = time.time() - start_time
        save_target_model(self.logger,self.args,run, self.target_model, shard)

        return train_time

    def aggregate(self, run):
        self.logger.info('aggregating submodels')

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(run, self.target_model, self.train_data, self.unlearned_shard_data, self.args,self.logger)
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate(self.data)

        self.logger.info("aggregate_f1_score: %s" % (self.aggregate_f1_score,))
        return self.aggregate_f1_score




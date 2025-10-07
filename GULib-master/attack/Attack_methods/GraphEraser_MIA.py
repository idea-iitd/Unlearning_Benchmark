import os
import time

from torch_geometric.transforms import SIGN
from tqdm import tqdm

import config
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data

from task.node_classification import NodeClassifier
from utils import dataset_utils
from utils import utils
from unlearning.unlearning_methods.GraphEraser.partition.graph_partition import GraphPartition
from unlearning.unlearning_methods.GraphEraser.aggregation.aggregator import Aggregator
from task import get_trainer
class GraphEraser_Attack:
    def __init__(self,args,logger,original_data,model_zoo,avg_unlearning_time,average_f1,average_auc,run):
        self.args = args
        self.logger = logger
        self.original_data = original_data
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self.affected_shard = []
        self.run = run
        self.f1_score = 0

        self.load_preprocessed_data()
        if self.args["unlearn_task"] == "node":
            path_un = config.unlearning_path + "_" + str(self.run) + ".txt"
            if os.path.exists(path_un):
                with open(path_un) as file:
                    node_unlearning_indices = [int(line.rstrip()) for line in file]
            else:
                shuffle_num = torch.randperm(self.data.train_indices.size)
                node_unlearning_indices = self.data.train_indices[shuffle_num][:self.args["num_unlearned_nodes"]]
                with open(path_un,mode="w") as file:
                    for node in node_unlearning_indices:
                        file.write(str(node) + '\n')
            start_time = time.time()
            self.graph_unlearning_request_respond(node_unlearning_indices)
            unlearning_time = time.time() - start_time
            avg_unlearning_time[run] = unlearning_time
            average_f1[run] = self.f1_score

            
            self.logger.info("start cal AUC")
            self.attack_graph_unlearning(average_auc)
            self.logger.info("unlearning_time:{}".format(unlearning_time))
        elif self.args["unlearn_task"] == "edge":
            start_time = time.time()
            self.graph_edge_unlearning_request_respond()
            unlearning_time = time.time() - start_time
            avg_unlearning_time[run] = unlearning_time
            average_f1[run] = self.f1_score
        elif self.args["unlearn_task"] == "feature":
            path_un = config.unlearning_path + "_" + str(self.run) + ".txt"
            if os.path.exists(path_un):
                with open(path_un) as file:
                    node_unlearning_indices = [int(line.rstrip()) for line in file]
            else:
                shuffle_num = torch.randperm(self.data.train_indices.size)
                node_unlearning_indices = self.data.train_indices[shuffle_num][:self.args["num_unlearned_nodes"]]
                with open(path_un,mode="w") as file:
                    for node in node_unlearning_indices:
                        file.write(str(node) + '\n')
            start_time = time.time()
            self.graph_feature_unlearning_request_respond(node_unlearning_indices)
            unlearning_time = time.time() - start_time
            avg_unlearning_time[run] = unlearning_time
            average_f1[run] = self.f1_score





    def load_preprocessed_data(self):
        self.shard_data = dataset_utils.load_shard_data(self.logger)
        self.raw_data = self.original_data.load_data()
        self.train_data = dataset_utils.load_saved_data(self.logger, config.train_data_file)
        self.train_graph = dataset_utils.load_train_graph(self.logger)
        data = dataset_utils.load_train_test_split(self.logger)
        self.train_indices, self.test_indices = data.train_indices,data.test_indices
        self.community_to_node = dataset_utils.load_community_data(self.logger)
        num_feats = self.train_data.num_features
        num_classes = len(self.train_data.y.unique())
        #9.20
        # self.target_model = NodeClassifier(self.args,self.shard_data,self.model_zoo,self.logger)
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.shard_data)


    def graph_unlearning_request_respond(self, node_unlearning_request=None):
        # reindex the node ids
        self.node_to_com = dataset_utils.c2n_to_n2c(self.args, self.community_to_node)
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
            train_shard_indices = np.setdiff1d(train_shard_indices, self.unlearning_indices[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

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

        dataset_utils.save_unlearned_data(self.logger, self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger, self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger, self.shard_data_after_unlearning, 'shard_data')

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

    def graph_edge_unlearning_request_respond(self):
        edge_index = self.data.edge_index.numpy()
        self.num_unlearned_edges = 0
        train_edge_indices = np.logical_and(np.isin(edge_index[0], self.data.train_indices),
                                np.isin(edge_index[1], self.data.train_indices))

        # 过滤出满足 edge_index[0] < edge_index[1] 的边，构成单向边
        directed_train_edges = np.logical_and(train_edge_indices, edge_index[0] < edge_index[1])

        # 获取这些单向边的索引
        train_edges = np.where(directed_train_edges)[0]

        # 计算需要删除的边的数量 (5%)
        num_edges_to_remove = int(len(train_edges) * 0.05)

        # 随机选择 5% 的单向边
        edges_to_remove = np.random.choice(train_edges, size=num_edges_to_remove, replace=False)

        # 找到与这些单向边对应的反向边
        reverse_edges_to_remove = np.where(
            (edge_index[0][None, :] == edge_index[1][edges_to_remove][:, None]) &
            (edge_index[1][None, :] == edge_index[0][edges_to_remove][:, None])
        )[1]

        # 合并正向边和反向边的索引
        all_edges_to_remove = np.concatenate([edges_to_remove, reverse_edges_to_remove])
        self.shard_data_after_unlearning = {}
        for shard in range(self.args["num_shards"]):
            train_shard_indices = list(self.community_to_node[shard])
            # node unlearning
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

            x = self.train_data.x[shard_indices]
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_3(self.train_data, shard_indices, all_edges_to_remove)

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
            if self.shard_data_after_unlearning[shard].num_edges != self.shard_data[shard].num_edges:
                self.affected_shard.append(shard)

        dataset_utils.save_unlearned_data(self.logger, self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger, self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger, self.shard_data_after_unlearning, 'shard_data')
        for shard in self.affected_shard:
            suffix = "_unlearned"
            self._train_shard_model(shard, suffix)


        f1 = self.aggregate(self.run)
        # self.logger.info("F1: {}".format(f1))
        self.f1_score = f1

    def graph_feature_unlearning_request_respond(self, node_unlearning_request=None):
        # reindex the node ids
        self.node_to_com = dataset_utils.c2n_to_n2c(self.args, self.community_to_node)
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

        dataset_utils.save_unlearned_data(self.logger, self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger, self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger, self.shard_data_after_unlearning, 'shard_data')

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

    def aggregate(self, run):
        self.logger.info('aggregating submodels')

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(run, self.target_model, self.train_data, self.shard_data_after_unlearning, self.args,self.logger,self.affected_shard)
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate(self.data)

        self.logger.info("Final Test F1 Score: {:.4f}".format(self.aggregate_f1_score))
        return self.aggregate_f1_score


    def _train_shard_model(self, shard, suffix="_unlearned"):
        self.logger.info('training target models, shard %s' % shard)

        # load shard data
        self.target_model.data = self.shard_data_after_unlearning[shard]
        # retrain shard model
        self.target_model.train()
        # replace shard model
        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.target_model.device = device
        dataset_utils.save_target_model(self.logger, self.args, self.run, self.target_model, shard, suffix)
        # self.data_store.save_unlearned_target_model(0, self.target_model, shard, suffix)

    def _repartition(self, suffix):
        # load unlearned train_graph and train_data
        train_graph = dataset_utils.load_unlearned_data(self.logger, 'train_graph')
        train_data = dataset_utils.load_unlearned_data(self.logger, 'train_data')
        # repartition
        start_time = time.time()
        partition = GraphPartition(self.args,self.logger, train_graph, train_data)
        community_to_node = partition.graph_partition()
        partition_time = time.time() - start_time
        self.logger.info("Partition cost %s seconds." % partition_time)
        # save the new partition and shard
        dataset_utils.save_community_data(self.logger, community_to_node, config.load_community_data, suffix)
        self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, self.test_indices)

    def _generate_unlearned_repartitioned_shard_data(self, train_data, community_to_node, test_indices):
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

    def attack_graph_unlearning(self,average_auc):

        # load unlearned indices
        path_un = config.unlearning_path + "_" + str(self.run) + ".txt"
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
        # load unlearned data
        train_data = dataset_utils.load_unlearned_data(self.logger, 'train_data')
        # load optimal weight score
        # optimal_weight=self.data_store.load_optimal_weight(0)

        # calculate the final posterior, save as attack feature
        self.logger.info('aggregating submodels')
        posteriors_a, posteriors_b, posteriors_c = [], [], []

        for i in tqdm(unlearned_indices, desc="MIA Progress"):
            community_to_node = dataset_utils.load_community_data(self.logger, config.load_community_data, '')
            shard_data = self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, int(i))

            posteriors_a.append(self._generate_posteriors(shard_data, ''))

            shard_num = self.node_to_com.get(i)
            posteriors_b.append(self._generate_posteriors_unlearned(shard_data))

            if self.args['repartition']:
                suffix = "_repartition_unlearned_" + str(i)
                community_to_node = dataset_utils.load_community_data(self.logger, config.load_community_data, suffix)
                shard_data = self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node,
                                                                               int(i))
                suffix = "_repartition_unlearned_" + str(i)
                posteriors_c.append(self._generate_posteriors(shard_data, suffix))

        return posteriors_a, posteriors_b, posteriors_c

    def _generate_posteriors_unlearned(self, shard_data):
        posteriors = []
        for shard in range(self.args['num_shards']):
            if shard in self.affected_shard:
                suffix = "_unlearned"
                # load the retrained the shard model
                dataset_utils.load_target_model(self.logger, self.args, self.run, self.target_model, shard, suffix)
            else:
                # self.target_model.model.reset_parameters()
                # load unaffected shard model
                dataset_utils.load_target_model(self.logger, self.args, self.run, self.target_model, shard, '')
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


    def evaluate_attack_performance(self, positive_posteriors, negative_posteriors,average_auc):
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
        posteriors = []
        for shard in range(self.args['num_shards']):
            # self.target_model.model.reset_parameters()
            dataset_utils.load_target_model(self.logger, self.args, self.run, self.target_model, shard, suffix)
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
        from sklearn.metrics import roc_auc_score
        self.logger.info("Directly calculate the attack AUC")
        return roc_auc_score(label, data.reshape(-1, 1))

    def _calculate_distance(self, data0, data1, distance='l2_norm'):
        if distance == 'l2_norm':
            return np.array([np.linalg.norm(data0[i] - data1[i]) for i in range(len(data0))])
        elif distance == 'direct_diff':
            return data0 - data1
        else:
            raise Exception("Unsupported distance")
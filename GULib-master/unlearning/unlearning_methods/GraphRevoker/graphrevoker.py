import numpy as np
import torch
import time
from collections import defaultdict
from dataset.original_dataset import *
from utils import utils, dataset_utils
from utils.dataset_utils import *
from utils.utils import *
from config import community_file,shard_file,train_data_file
from unlearning.unlearning_methods.GraphRevoker.lib_partition.graph_partition import GraphPartition
from task.node_classification import NodeClassifier
from unlearning.unlearning_methods.GraphRevoker.lib_aggregator.aggregator import Aggregator
from config import BLUE_COLOR,RESET_COLOR
from task.GraphRevokerTrainer import GraphRevokerTrainer
from pipeline.Shard_based_pipeline import Shard_based_pipeline
class graphrevoker(Shard_based_pipeline):
    """
    The GraphRevoker class implements a pipeline for graph unlearning methods within a sharded architecture.
    It handles the partitioning of the graph, training of target models, aggregation of results, and execution
    of unlearning attacks. This class is designed to facilitate efficient unlearning in large-scale graph data
    by leveraging shard-based processing.
    
    Class Attributes:
        args (dict): Configuration parameters for the pipeline, including model settings and experiment details.

        logger (Logger): Logger instance for tracking the workflow and debugging information.
        
        model_zoo (ModelZoo): Collection of models used for training and evaluation.
    """

    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger 
        self.model_zoo = model_zoo
        # self.data = self.model_zoo.data
        self.data_copy = self.model_zoo.data
        self.target_model_name = self.args['base_model']
        self.num_opt_samples = self.args['num_opt_samples']
        self.num_shards = self.args['num_shards']
        num_runs = self.args["num_runs"]
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_partition_time = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)
        self.avg_unlearning_time = np.zeros(num_runs)
        self.run = 0
    # def run_exp(self):
    #     for self.run in range(self.args["num_runs"]):
    #         self.data = self.data_copy.clone()
    #         self.datafull = self.data_copy.clone()
    #         self.args["exp"] == "partition"
    #         self.train_test_split()
    #         self.gen_train_graph()
    #         self.graph_partition()
    #         self.generate_shard_data()
            
    #         self.load_data()
    #         self.determine_target_model()
    #         self.args["exp"] = "unlearning"
    #         self.train_target_models(self.run)
    #         aggregate_f1_score = self.aggregate(self.run)
    #         node_unlearning_time = self.unlearning_time_statistic()
    #         # self.average_f1[self.run] = aggregate_f1_score

    #         # self.logger.info("Run %f" % self.run)
    #         self.args["exp"] = "attack_unlearning"
    #         retrained_shards = self.generate_requests()
    #         self.update_shard()
    #         # is_in = np.all(np.isin(self.unlearning_nodes, self.train_data.train_indices))
    #         t = self.train_sequential(retrained_shards)
    #         self.avg_unlearning_time[self.run] = t

    #         f1 = self.aggregate(self.run)
    #         self.average_f1[self.run] = f1
    #         self.logger.info("Final f1:{}".format(f1))
    #         self.attack_graph_unlearning()

    #     self.logger.info(
    #         "{}Performance Metrics:\n"
    #         " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #         " - Average AUC Score: {:.4f} ± {:.4f}\n"
    #         " - Average Partition Time: {:.4f} ± {:.4f} seconds\n"
    #         " - Average Training Time: {:.4f} ± {:.4f} seconds\n"
    #         " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
    #             BLUE_COLOR,
    #             np.mean(self.average_f1), np.std(self.average_f1),
    #             np.mean(self.average_auc), np.std(self.average_auc),
    #             np.mean(self.avg_partition_time), np.std(self.avg_partition_time),
    #             np.mean(self.avg_training_time), np.std(self.avg_training_time),
    #             np.mean(self.avg_unlearning_time), np.std(self.avg_unlearning_time),
    #             RESET_COLOR
    #         )
    #     )
        


    def train_test_split(self):
        self.train_indices, self.val_indices, self.test_indices = np.array(self.data.train_indices),np.array(self.data.val_indices),np.array(self.data.test_indices)


    def gen_train_graph(self):
        # Decouple train test edges. Edges between train and validation are preserved.
        self.data.edge_index_train = utils.filter_test_edges(self.data.edge_index.detach().cpu().numpy(), self.test_indices)
        # self.data.edge_index_train = self.data.edge_index
        # The train graph will be deleted. NetworkX is not scalable.
        self.train_graph = None
        #self.train_graph = nx.Graph()
        #self.train_graph.add_nodes_from(sorted(self.train_indices))
        #self.train_graph.add_edges_from(cp.asnumpy(cp.transpose(edge_index_train)))

        #self.logger.debug("Train graph  #.Nodes: %f, #.Edges: %f" % (
        #    self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        self.logger.debug("Train graph  #.Nodes: %f, #.Edges: %f" % (
            len(self.train_indices), self.data.edge_index_train.shape[1]))
        self.logger.debug("Entire dataset  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))
        
        if self.args['dataset_name'] == 'Flickr':
            print('The train graph is too large in this  dataset. Ignore train data saving.')
        else:
            save_train_data(self.logger,self.data,train_data_file)


    def graph_partition(self):
        if self.args['is_partition']:
            self.logger.info('graph partitioning')

            start_time = time.time()
            partition = GraphPartition(self.args, self.train_graph, self.logger,self.model_zoo,self.data)
            
            self.community_to_node = partition.graph_partition()
            partition_time = time.time() - start_time
            self.logger.info("Partition cost %s seconds." % partition_time)
            self.avg_partition_time[self.run] = partition_time
            save_community_data(self.logger,self.community_to_node,community_file)
        else:
            load_community_data(self.logger,community_file)

    def generate_shard_data(self):
        self.logger.info('generating shard data')

        self.shard_data = {}

        # Generate shard data and record shard characteristics
        shard_samples = []
        shard_edges = []
        shard_label_dist = []
        # self.datafull.edge_index_train = utils.filter_test_edges(self.datafull.edge_index.detach().cpu().numpy(), self.test_indices)
        self.datafull.edge_index_train = self.datafull.edge_index
        for shard in tqdm(range(self.args['num_shards'])):
            train_shard_indices = list(self.community_to_node[shard])
            self.shard_data[shard] = build_shard_data(self.datafull, train_shard_indices, self.val_indices, self.test_indices)

            shard_nodes = self.community_to_node[shard]
            shard_samples.append(len(shard_nodes))

            shard_edge_mask = np.logical_and(np.isin(self.datafull.edge_index[0], train_shard_indices),
                                             np.isin(self.datafull.edge_index[1], train_shard_indices))
            num_shard_edges = np.sum(shard_edge_mask)
            shard_edges.append(num_shard_edges)
            
            label_cnt = [0] * len(np.unique(self.datafull.y.cpu().numpy()))
            for label in self.datafull.y[train_shard_indices].cpu().numpy():
                label_cnt[label] += 1
            shard_label_dist.append(np.array(label_cnt))
        
        # Calculate the label distribution 
        shard_label_dist = np.array(shard_label_dist)
        #total_dist = np.sum(shard_label_dist, axis=0)
        #total_dist = total_dist / np.sum(total_dist)
        shard_label_dist = shard_label_dist / np.expand_dims(np.sum(shard_label_dist, axis=1), 1) + 1e-5
        shard_label_gini = np.mean(1 - np.sum(np.power(shard_label_dist, 2), axis=1))
        #mean_kl_div = np.mean([np.sum(rel_entr(line, total_dist)) for line in shard_label_dist])
        self.logger.info('Mean Label Gini = {}'.format(shard_label_gini))

        # Calculate the Unbalancedness
        m = np.mean(shard_samples)
        u = np.sqrt(np.mean((np.array(shard_samples) - m) ** 2))
        self.logger.info('Unbalanceness = {}'.format(u))
        self.logger.info('Samples in each shard = {}'.format(shard_samples))
        self.logger.info('Edges in each shard = {}'.format(shard_edges))

        self.logger.info('{} of {} edges are inside train shards.'.format(
            np.sum(shard_edges), self.data.edge_index.shape[1]))

        save_shard_data(self.logger,self.shard_data,shard_file)

    def load_data(self):
        self.shard_data = dataset_utils.load_shard_data(self.logger)


    def determine_target_model(self):
        num_feats = self.data.num_features
        num_classes = len(self.data.y.unique())
        # if self.args["downstream_task"] == "node":
        #     self.target_model = NodeClassifier(self.args,self.data,self.model_zoo, self.logger)
        # elif self.args["downstream_task"] == "edge":
        #     pass
        self.target_model = GraphRevokerTrainer(self.args,self.logger,self.model_zoo.model,self.data)

    def train_target_models(self, run):
        if self.args['is_train_target_model']:
            self.logger.info('training target models')

            self.time = {}
            for shard in range(self.num_shards):
                # self.target_model.data = self.shard_data[shard]
                self.time[shard] = self._train_model(run, shard)

    def _train_model(self, run, shard):
        start_time = time.time()
        self.target_model.prepare_data(self.shard_data[shard])

        self.logger.info('training target model, run %s, shard %s' % (run, shard))

        #start_time = time.time()
        # print(self.target_model.data)
        train_time = self.target_model.train()
        train_time = time.time() - start_time

        dataset_utils.save_target_model(self.logger,self.args,run, self.target_model, shard)
        self.logger.info("Model training time: %s" % (train_time))

        return train_time

    def aggregate(self, run):
        self.logger.info('aggregating submodels')

        aggregator = Aggregator(run, self.target_model, self.data, self.shard_data, self.args,self.logger)
        aggregator.generate_posterior()
        start_time = time.time()
        self.aggregate_f1_score, aggregate_time = aggregator.aggregate()
        aggregate_time = time.time() - start_time
        self.logger.info("Aggregation cost %s seconds." % aggregate_time)

        self.logger.info("Final Test F1: %s" % (self.aggregate_f1_score,))
        return self.aggregate_f1_score
    
    def unlearning_time_statistic(self):
        if self.args['is_train_target_model'] and self.num_shards != 1:
            self.community_to_node = dataset_utils.load_community_data(self.logger)
            node_list = []
            for key, value in self.community_to_node.items():
                node_list.extend(value)

            # random sample 5% nodes, find their belonging communities
            sample_nodes = np.random.choice(node_list, int(0.05 * len(node_list)))
            belong_community = []
            for sample_node in range(len(sample_nodes)):
                for community, node in self.community_to_node.items():
                    if np.in1d(sample_nodes[sample_node], node).any():
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = []
            node_unlearning_time = []
            for shard in range(self.num_shards):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard])
                    node_unlearning_time.extend([float(self.time[shard]) for j in range(belong_community.count(shard))])

            return node_unlearning_time

        elif self.args['is_train_target_model'] and self.num_shards == 1:
            return self.time[0]

        else:
            return 0
        
    def generate_requests(self):
        self.community_to_node = dataset_utils.load_community_data(self.logger)
        node_list = []
        for key, value in self.community_to_node.items():
            node_list.extend(value)
        
        # Ratio of nodes to unlearn
        # sample_nodes = np.random.choice(node_list, int(self.args['ratio_unlearned_nodes'] * len(node_list)))
        path_un = unlearning_path + "_" + str(self.run) + ".txt"
        self.unlearning_nodes = np.loadtxt(path_un, dtype=int)
        sample_nodes = self.unlearning_nodes 
        belong_community = []
        for sample_node in sample_nodes:
            for community, node in self.community_to_node.items():
                if np.in1d(sample_node, node).any():
                    belong_community.append(community)
        self.retrained_shards = list(np.unique(belong_community))
        # return list(np.unique(belong_community))
    
    def train_sequential(self, retrained_shards):
        t = time.time()
        args = ([self.shard_data_after_unlearning[i] for i in retrained_shards], [i for i in retrained_shards], 'cuda:{}'.format(self.args['cuda']))
        res = self.train_single( *args, self.args)
        return time.time() - t
    


    def train_single(self,data_list, shard_n_list, device, args):
        res_dict = dict()
        for data, shard_n in zip(data_list, shard_n_list):
            # print(data)
            self.target_model.data = data
            self.target_model.train()
            suffix = "_unlearned"
            dataset_utils.save_target_model(self.logger,self.args, self.run, self.target_model, shard_n, suffix)
            res_dict[shard_n] =  self.target_model.model.state_dict()
            print('Finish retraining on shard: {}'.format(shard_n))

    def update_shard(self):
        self.node_to_com = dataset_utils.c2n_to_n2c(self.args,self.community_to_node)
        self.unlearning_indices = defaultdict(list)
        self.train_data = dataset_utils.load_saved_data(self.logger,config.train_data_file)
        self.train_data.edge_index = self.train_data.edge_index_train
        self.train_data.edge_index_train = None
        # self.train_graph = dataset_utils.load_train_graph(self.logger)
        self.affected_shard = []

        for node in self.unlearning_nodes:
            node = int(node)
            node_key = self.node_to_com.get(node)
            if node_key is not None:
                self.unlearning_indices[node_key].append(node)
            else:
                # 处理键不存在的情况，可能需要打印一些信息或者采取其他措施
                print(f"The key {node} does not exist in the dictionary.")

        # self.train_graph.remove_nodes_from(self.unlearning_nodes)

        # delete the revoked nodes from train_data
        # by building unlearned data from unlearned train_graph
        self.train_data.train_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), self.train_indices))
        self.train_data.val_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), self.val_indices))
        self.train_data.test_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), np.append(self.test_indices, self.unlearning_nodes)))

        
        self.shard_data_after_unlearning = {}
        for shard in range(self.args["num_shards"]):
            train_shard_indices = list(self.community_to_node[shard])
            # node unlearning
            train_shard_indices = np.setdiff1d(train_shard_indices, self.unlearning_indices[shard])
            shard_indices = np.union1d(np.union1d(train_shard_indices, self.val_indices), 
                               self.test_indices)

            x = self.train_data.x[shard_indices]
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.datafull, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_edge_index = torch.from_numpy(edge_index)
            data.test_edge_index = torch.from_numpy(edge_index)
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
            # self.num_unlearned_edges += self.shard_data[shard].num_edges - self.shard_data_after_unlearning[
                # shard].num_edges

            # find the affected shard model
            if self.shard_data_after_unlearning[shard].num_nodes != self.shard_data[shard].num_nodes:
                self.affected_shard.append(shard)

        # dataset_utils.save_unlearned_data(self.logger,self.train_graph, 'train_graph')
        dataset_utils.save_unlearned_data(self.logger,self.train_data, 'train_data')
        dataset_utils.save_unlearned_data(self.logger,self.shard_data_after_unlearning, 'shard_data')

    def aggregate_unlearned(self,):
        self.logger.info('aggregating unlearned submodels')

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(self.run, self.target_model, self.train_data, self.shard_data_after_unlearning, self.args,self.logger,self.affected_shard)
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate(self.data)

        self.logger.info("Final Test F1 Score: {:.4f}".format(self.aggregate_f1_score))
        return self.aggregate_f1_score
    
    def attack_graph_unlearning(self):
        unlearned_indices = self.unlearning_nodes[:100] 
        positive_posteriors = self._query_target_model(unlearned_indices)
        negative_posteriors = self._query_target_model(self.data.test_indices[0:100])

        self.evaluate_attack_performance(positive_posteriors, negative_posteriors)

    def _query_target_model(self, unlearned_indices):
        # train_data = dataset_utils.load_unlearned_data(self.logger,'train_data')
        community_to_node = dataset_utils.load_community_data(self.logger,config.load_community_data,'')
        # community_to_node = dataset_utils.load_community_data(self.logger,"./data/GraphEraser/processed/cora/community_lpa_10_0",'')
        posteriors_a, posteriors_b = [], []
        for i in tqdm(unlearned_indices, desc="MIA Progress"):
            shard_data = self._generate_unlearned_repartitioned_shard_data(self.datafull, community_to_node, int(i))

            posteriors_a.append(self._generate_posteriors(shard_data, ''))
            # shard_num = self.node_to_com.get(i)
            posteriors_b.append(self._generate_posteriors_unlearned(shard_data))

        return posteriors_a, posteriors_b

            


    def _generate_unlearned_repartitioned_shard_data(self, train_data, community_to_node, test_indices):
        # self.logger.info('generating shard data')

        shard_data = {}

        for shard in range(self.args['num_shards']):
            train_shard_indices = list(community_to_node[shard])
            train_shard_indices = np.setdiff1d(train_shard_indices, self.unlearning_indices[shard])
            shard_indices = np.union1d(np.union1d(train_shard_indices, self.val_indices), 
                               test_indices)
            shard_indices = shard_indices.astype(int)
            ################
            # shard_indices_tensor = torch.tensor(shard_indices)

            # connected_edges = torch.nonzero(
            #     ((self.train_data.edge_index[0] == test_indices) & torch.isin(self.train_data.edge_index[1], shard_indices_tensor)) |
            #     (torch.isin(self.train_data.edge_index[0], shard_indices_tensor) & (self.train_data.edge_index[1] == test_indices)),
            #     as_tuple=False
            # )
            # connected_edges = torch.nonzero(
            #     ((self.train_data.edge_index[0] == test_indices)) |
            #     ((self.train_data.edge_index[1] == test_indices) ),
            #     as_tuple=False
            # )
            # edges = self.train_data.edge_index[:, connected_edges].squeeze()
            ###############

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

            # test_indices_list = list()
            # test_indices_list.append(test_indices)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, test_indices))

            # 判断 test_indices 是否是独立节点
            test_node_index = torch.nonzero(data.test_mask, as_tuple=True)[0].item()
            connected_edges = torch.nonzero(
                (data.edge_index[0] == test_node_index) | (data.edge_index[1] == test_node_index),
                as_tuple=False
            )

            # 如果没有找到与 test_indices 相关的边，说明它是独立节点
            # if connected_edges.size(0) == 0:
            #     print(f"Node {test_indices} is an isolated node in shard {shard}.")
            # else:

            #     print(f"Node {test_indices} change to {test_node_index} has {connected_edges.size(0)} connected edges in shard {shard}.")

            shard_data[shard] = data

        # self.data_store.save_unlearned_data(shard_data, 'shard_data_repartition')
        return shard_data
    
    def _generate_posteriors(self, shard_data, suffix):
        posteriors = []
        for shard in range(self.args['num_shards']):
            # self.target_model.model.reset_parameters()
            dataset_utils.load_target_model(self.logger,self.args, self.run, self.target_model, shard, '')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.target_model.model = self.target_model.model.to(self.device)
            self.target_model.data = shard_data[shard].to(self.device)
            ###debug
            # test_node_index = torch.nonzero(self.target_model.data.test_mask, as_tuple=True)[0].item()
            # connected_edges = torch.nonzero((self.target_model.data.edge_index[0] == test_node_index) | (self.target_model.data.edge_index[1] == test_node_index), as_tuple=False)
            # edges = self.target_model.data.edge_index[:, connected_edges].squeeze()
            ######
            # if self.args['base_model'] == "SAGE":
            #     posteriors.append(self.target_model.posterior())
            # else:
            #9.20
            # posteriors.append(self.target_model.posterior_other())
            posteriors.append(self.target_model.posterior(mia=True))
        return torch.mean(torch.cat(posteriors, dim=0), dim=0)
    
    def _generate_posteriors_unlearned(self, shard_data):
        posteriors = []
        for shard in range(self.args['num_shards']):
            if shard in self.affected_shard:
                suffix = "_unlearned"
                dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard, suffix)
                #####test
                # dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model2, shard, '')
                # params1 = list(self.target_model.model.parameters())
                # params2 = list(self.target_model2.model.parameters())
                # for p1, p2 in zip(params1, params2):
                #     if not torch.equal(p1, p2):
                #         print("not equal!")

            else:
                dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard, '')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.target_model.model = self.target_model.model.to(self.device)
            self.target_model.data = shard_data[shard].to(self.device)
            #9.20
            # posteriors.append(self.target_model.posterior_other())
            posteriors.append(self.target_model.posterior(mia=True))

        return torch.mean(torch.cat(posteriors, dim=0), dim=0)
    
    def evaluate_attack_performance(self, positive_posteriors, negative_posteriors):
        # constrcut attack data
        label = torch.cat((torch.ones(len(positive_posteriors[0])), torch.zeros(len(negative_posteriors[0]))))
        data={}
        for i in range(2):
             data[i] = torch.cat((torch.stack(positive_posteriors[i]), torch.stack(negative_posteriors[i])),0)

        # calculate l2 distance
        model_b_distance = utils._calculate_distance(data[0], data[1])
        # directly calculate AUC with feature and labels
        attack_auc_b = utils.evaluate_attack_with_AUC(model_b_distance, label)

        self.logger.info("Attack_Model_B AUC: %s " % (attack_auc_b))

        self.average_auc[self.run] = attack_auc_b
        
    def train_shard_model(self):
        self.train_target_models(self.run)
    
    def aggregate_shard_model(self):
        aggregate_f1_score = self.aggregate(self.run)
        node_unlearning_time = self.unlearning_time_statistic()
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = aggregate_f1_score
            
    def attack_unlearning(self):
        if self.args["downstream_task"] == "graph":
            return
        if self.args["unlearn_task"] == "node":
            self.logger.info("start cal AUC")
            self.attack_graph_unlearning(self.average_auc)
            
    def unlearn(self):
        self.update_shard()
        # is_in = np.all(np.isin(self.unlearning_nodes, self.train_data.train_indices))
        t = self.train_sequential(self.retrained_shards)
        self.avg_unlearning_time[self.run] = t

        f1 = self.aggregate(self.run)
        self.average_f1[self.run] = f1
        self.logger.info("Final f1:{}".format(f1))

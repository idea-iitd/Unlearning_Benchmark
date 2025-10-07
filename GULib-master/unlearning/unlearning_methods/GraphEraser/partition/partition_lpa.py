import math
import os
import numpy as np
import networkx as nx
import logging
import pickle

from unlearning.unlearning_methods.GraphEraser.partition.constrained_lpa_base import ConstrainedLPABase
from unlearning.unlearning_methods.GraphEraser.partition.partition import Partition
from unlearning.unlearning_methods.GraphEraser.partition.constrained_lpa import ConstrainedLPA
import config





class PartitionLPA(Partition):
    def __init__(self, args,logger, graph):
        super(PartitionLPA, self).__init__(args, graph)

        self.logger = logger

    def partition(self):
        # implement LPA by hand, refer to https://github.com/benedekrozemberczki/LabelPropagation
        community_generator = nx.algorithms.community.label_propagation.label_propagation_communities(self.graph)
        self.logger.info("Generating LPA communities.")
        community_to_node = {key: c for key, c in zip(range(self.graph.number_of_nodes()), community_generator)}
        print("Found %s communities by unconstrained LPA", len(community_to_node.keys()))
        return community_to_node


class PartitionConstrainedLPA(Partition):
    def __init__(self, args,logger, graph):
        super(PartitionConstrainedLPA, self).__init__(args, graph)
        self.args = args

        self.logger = logger

    def partition(self):
        # adj_array = nx.linalg.adjacency_matrix(self.graph).toarray().astype(np.bool_)  # is old version
        adj_array = nx.to_numpy_array(self.graph, dtype=bool)

        # node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards']) + 0.05 * self.graph.number_of_nodes()
        # node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards'])
        node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards'] +
                                   self.args['shard_size_delta'] * (self.graph.number_of_nodes()-self.graph.number_of_nodes() / self.args['num_shards']))

        self.logger.info(" #. nodes: %s. LPA shard threshold: %s." % (self.graph.number_of_nodes(), node_threshold))
        lpa = ConstrainedLPA(adj_array, self.num_shards, node_threshold, self.args['terminate_delta'])

        lpa.initialization()
        community_to_node, lpa_deltas = lpa.community_detection()
        if not os.path.exists(config.ANALYSIS_PATH + "partition"):
            os.mkdir(config.ANALYSIS_PATH + "partition")

        pickle.dump(lpa_deltas, open(config.ANALYSIS_PATH + "partition/blpa_" + self.args['dataset_name'], 'wb'))

        return self.idx2id(community_to_node, np.array(self.graph.nodes))


class PartitionConstrainedLPABase(Partition):
    def __init__(self, args,logger, graph):
        super(PartitionConstrainedLPABase, self).__init__(args, graph)
        self.args = args

        self.logger = logger

    def partition(self):
        # adj_array = nx.linalg.adj_matrix(self.graph).toarray().astype(np.bool)  #is old so changed to below
        adj_array = nx.to_numpy_array(self.graph, dtype=bool)

        node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards'] + self.args['shard_size_delta'] * (self.graph.number_of_nodes()-self.graph.number_of_nodes() / self.args['num_shards']))

        self.logger.info(" #. nodes: %s. LPA shard threshold: %s." % (self.graph.number_of_nodes(), node_threshold))
        lpa = ConstrainedLPABase(self.logger,adj_array, self.num_shards, node_threshold, self.args['terminate_delta'])

        lpa.initialization()
        community_to_node, lpa_deltas = lpa.community_detection()

        # pickle.dump(lpa_deltas, open(config.ANALYSIS_PATH + "partition/base_blpa_" + self.args['dataset_name'], 'wb')) #Change file writing
        import os

        # Ensure directory exists
        output_dir = os.path.join(config.ANALYSIS_PATH, "partition")
        os.makedirs(output_dir, exist_ok=True)

        # Save file safely
        output_file = os.path.join(output_dir, "base_blpa_" + self.args['dataset_name'])
        with open(output_file, 'wb') as f:
            pickle.dump(lpa_deltas, f)

        return self.idx2id(community_to_node, np.array(self.graph.nodes))

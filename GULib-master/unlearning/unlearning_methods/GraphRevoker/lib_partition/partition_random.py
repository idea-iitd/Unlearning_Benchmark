import numpy as np

from unlearning.unlearning_methods.GraphRevoker.lib_partition.partition import Partition
import torch

class PartitionRandom(Partition):
    def __init__(self, args, graph, dataset,logger,model_zoo):
        super(PartitionRandom, self).__init__(args, graph)

        self.dataset = dataset
        self.logger = logger
        self.model_zoo = model_zoo

    def partition(self):
        #graph_nodes = np.array(self.graph.nodes)
        graph_nodes = torch.nonzero(self.dataset.train_mask).squeeze(1).numpy()
        np.random.shuffle(graph_nodes)
        train_shard_indices = np.array_split(graph_nodes, self.args['num_shards'])

        return dict(zip(range(self.num_shards), train_shard_indices))

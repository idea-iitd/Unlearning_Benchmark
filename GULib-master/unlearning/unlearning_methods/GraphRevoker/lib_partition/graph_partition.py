import logging

from unlearning.unlearning_methods.GraphRevoker.lib_partition.partition_random import PartitionRandom
from unlearning.unlearning_methods.GraphRevoker.lib_partition.partition_gpa import PartitionGPA

class GraphPartition:
    def __init__(self, args, graph,logger,model_zoo,dataset=None):
        self.logger = logging.getLogger(__name__)

        self.args = args
        self.graph = graph
        self.dataset = dataset
        self.logger = logger
        self.model_zoo = model_zoo

        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']

    def graph_partition(self):
        self.logger.info('graph partition, method: %s' % self.partition_method)

        if self.partition_method == 'random':
            partition_method = PartitionRandom(self.args, self.graph, self.dataset,self.logger,self.model_zoo)
        elif self.partition_method == 'gpa':
            partition_method = PartitionGPA(self.args, self.graph, self.dataset,self.logger,self.model_zoo)
        else:
            raise Exception('Unsupported partition method')

        return partition_method.partition()

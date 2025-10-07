import logging

from unlearning.unlearning_methods.GraphEraser.partition.partition_kmeans import PartitionKMeans
from unlearning.unlearning_methods.GraphEraser.partition.partition_lpa import PartitionConstrainedLPA, PartitionLPA, PartitionConstrainedLPABase
from unlearning.unlearning_methods.GraphEraser.partition.metis_partition import MetisPartition
from unlearning.unlearning_methods.GraphEraser.partition.partition_random import PartitionRandom
from unlearning.unlearning_methods.GraphEraser.partition.partition_gpa import PartitionGPA
from unlearning.unlearning_methods.GraphEraser.partition.graph_kmeans import PartitionGraphKM
class GraphPartition:
    def __init__(self,logger, args, graph, dataset=None,model_zoo = None):
        self.logger = logger

        self.args = args
        self.graph = graph
        self.dataset = dataset
        self.model_zoo = model_zoo

        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']

    def graph_partition(self):
        self.logger.info('graph partition, method: %s' % self.partition_method)

        if self.partition_method == 'random':
            partition_method = PartitionRandom(self.args, self.graph)
        elif self.partition_method in ['sage_km', 'sage_km_base']:
            partition_method = PartitionKMeans(self.args, self.logger,self.graph, self.dataset,model_zoo = self.model_zoo)
        elif self.partition_method == 'lpa' and not self.args['is_constrained']:
            partition_method = PartitionLPA(self.args, self.logger,self.graph)
        elif self.partition_method == 'lpa' and self.args['is_constrained']:
            partition_method = PartitionConstrainedLPA(self.args,self.logger, self.graph)
        elif self.partition_method == 'lpa_base':
            partition_method = PartitionConstrainedLPABase(self.args,self.logger, self.graph)
        elif self.partition_method == 'metis':
            partition_method = MetisPartition(self.args, self.graph, self.dataset)
        elif self.partition_method == 'gpa':
            partition_method = PartitionGPA(self.args, self.graph, self.dataset,self.logger,self.model_zoo)
        elif self.partition_method == 'graph_km':
            partition_method = PartitionGraphKM(self.args,self.logger,self.dataset)
        else:
            raise Exception('Unsupported partition method')

        return partition_method.partition()

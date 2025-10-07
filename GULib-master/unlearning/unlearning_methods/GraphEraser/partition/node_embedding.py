import logging
import torch.nn.functional as F
import config
from model.base_gnn.graphsage import SAGENet
from utils.dataset_utils import *
from utils.dataset_utils import _extract_embedding_method
from model.base_gnn.graphsage import SAGENet
from model.base_gnn.gat import GATNet
from model.base_gnn.gcn import GCNNet
from model.base_gnn.gin import GINNet
from model.base_gnn.sgc import SGCNet
from model.base_gnn.s2gc import S2GCNet
from model.base_gnn.sign import SIGNNet
from task import BaseTrainer
from utils.dataset_utils import save_embeddings,load_embeddings
from config import embedding_file
class NodeEmbedding:
    def __init__(self, args,logger, graph, data,model_zoo=None):
        super(NodeEmbedding, self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.graph = graph
        self.data = data
        self.logger  = logger
        self.model_zoo = model_zoo

    def sage_encoder(self):
        embedding_name = '_'.join(('embedding', _extract_embedding_method(self.args['partition_method']),
                                   str(self.args['ratio_deleted_edges'])))
        embedding_path = config.PROCESSED_DATA_PATH + self.data.name + "/" + embedding_name
        if self.args['is_gen_embedding']:
            self.logger.info("generating node embeddings with GraphSage...")

            node_to_embedding = {}
            # run sage

            #self.target_model = SAGENet(self.data.num_features, len(self.data.y.unique()), self.data)
            self.target_model = BaseTrainer(self.args,self.logger,self.model_zoo.model,self.data)
            self.target_model.data = self.data
            target_model_name = '_'.join((self.args['base_model'], 'random_1',
                                          str(self.args['shard_size_delta']),
                                          str(self.args['ratio_deleted_edges']), '0_0_1'))
            target_model_file = config.MODEL_PATH + self.args['dataset_name'] + '/' + target_model_name
            self.target_model.train()
            self.target_model.save_model(target_model_file)

            # load a pretrained GNN model for generating node embeddings
            self.target_model.load_model(target_model_file)
            # self.target_model.train_model(50)

            # load a pretrained GNN models for generating node embeddings
            # target_model_name = '_'.join((self.args['base_model'], 'random_1',
            #                               str(self.args['shard_size_delta']),
            #                               str(self.args['ratio_deleted_edges']), '0_0_1'))

            # logits = self.target_model.generate_embeddings().detach().cpu().numpy()
            logits = F.log_softmax(self.target_model.model(self.data.x,self.data.edge_index),dim = 1).detach().cpu().numpy()
            for node in self.graph.nodes:
                node_to_embedding[node] = logits[node]

            save_embeddings(node_to_embedding,embedding_path)
        else:
            node_to_embedding = load_embeddings(embedding_path)

        return node_to_embedding
    
    def encoder(self, hidden_channels=256, embed_layer=None):
        if self.args['is_gen_embedding']:
            self.data = self.data.to(self.device)
            self.logger.info("Generating node embeddings for graph partitioning...")

            node_to_embedding = {}
            # Run SAGE as an encoder
            self.target_model = BaseTrainer(self.args,self.logger,self.model_zoo.model,self.data)
            # index_tmp  = self.data.edge_index_train
            # self.target_model.prepare_data(self.data)
            # self.data.edge_index_train = index_tmp
            self.target_model.data = self.data
            target_model_name = '_'.join((self.args['base_model'], 'random_1',
                                          str(self.args['shard_size_delta']),
                                          str(self.args['ratio_deleted_edges']), '0_0_1'))
            target_model_file = config.MODEL_PATH + self.args['dataset_name'] + '/' + target_model_name
            self.target_model.train()
            self.target_model.save_model(target_model_file)

            # load a pretrained GNN model for generating node embeddings
            self.target_model.load_model(target_model_file)
            embs = F.log_softmax(self.target_model.model(self.data.x,self.data.edge_index)[self.data.train_mask],dim = 1).detach().cpu().numpy()
            train_nodes = torch.nonzero(self.data.train_mask.cpu()).squeeze(1).numpy()
            for i, node in enumerate(train_nodes):
                node_to_embedding[node] = embs[i]

            if self.args['exp'] != 'attack_unlearning':
                save_embeddings(node_to_embedding,embedding_file)
        else:
            node_to_embedding = load_embeddings(self.logger,embedding_file)

        return node_to_embedding

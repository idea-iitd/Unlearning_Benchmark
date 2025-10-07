import logging
import config

import torch
import torch.nn.functional as F
from task.node_classification import NodeClassifier
from utils.dataset_utils import save_embeddings,load_embeddings
from config import embedding_file

class NodeEmbedding:
    def __init__(self, args,logger, graph, data,model_zoo):
        super(NodeEmbedding, self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.logger = logger
        self.graph = graph 
        self.data = data
        self.model_zoo = model_zoo

    def encoder(self, hidden_channels=256, embed_layer=None):
        if self.args['is_gen_embedding']:
            self.data = self.data.to(self.device)
            self.logger.info("Generating node embeddings for graph partitioning...")

            node_to_embedding = {}
            # Run SAGE as an encoder
            self.target_model = NodeClassifier(self.args,self.data,self.model_zoo,self.logger)
            index_tmp  = self.data.edge_index_train
            self.target_model.prepare_data(self.data)
            self.data.edge_index_train = index_tmp
            target_model_name = '_'.join((self.args['base_model'], 'random_1',
                                          str(self.args['shard_size_delta']),
                                          str(self.args['ratio_deleted_edges']), '0_0_1'))
            target_model_file = config.MODEL_PATH + self.args['dataset_name'] + '/' + target_model_name
            self.target_model.train_model()
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
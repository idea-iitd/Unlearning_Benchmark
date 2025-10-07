import logging
import torch

torch.cuda.empty_cache()

from sklearn.metrics import accuracy_score, f1_score,roc_auc_score
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from dataset.original_dataset import original_dataset
from unlearning.unlearning_methods.GraphRevoker.lib_aggregator.optimal_aggregator import OptimalAggregator
from unlearning.unlearning_methods.GraphRevoker.lib_aggregator.contra_aggregator_v2 import ContrastiveAggregator
from unlearning.unlearning_methods.GraphRevoker.lib_aggregator.optimal_edge_aggregator import OptimalEdgeAggregator
from utils.utils import *
from utils import dataset_utils

class Aggregator:
    def __init__(self, run, target_model, data, shard_data, args,logger,affected_shard=None):
        self.logger = logger
        self.args = args

        self.data_store = original_dataset(self.args,logger)


        self.run = run
        self.target_model = target_model
        self.data = data
        self.shard_data = shard_data
        self.affected_shard = affected_shard

        self.num_shards = args['num_shards']

    def generate_posterior(self, suffix=""):
        self.true_label = self.shard_data[0].y[self.shard_data[0]['test_mask']].detach().cpu().numpy()
        self.posteriors = []
        self.test_embeddings = []
        self.true_label = self.shard_data[0].y[self.shard_data[0]['test_mask']].detach().cpu().numpy()
        pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
        neg_edge_labels = torch.zeros(self.data.test_edge_index.size(1),dtype=torch.float32)
        edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
        self.true_label_edge = edge_labels
        for shard in range(self.args['num_shards']):
            if self.affected_shard is not None and shard in self.affected_shard:
                dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
            else:
                dataset_utils.load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")
            # dataset_utils.load_target_model(self.logger,self.args,self.run, self.target_model, shard, suffix)
            self.target_model.prepare_data(self.shard_data[shard])
            if self.args['aggregator'] == 'contrastive':
                z, f = self.target_model.posterior(return_features=True)
                self.posteriors.append(z)
                #self.test_embeddings.append(torch.cat([f, z], 1))
                self.test_embeddings.append(f)
            else:
                if self.args["downstream_task"]=="node":
                    self.posteriors.append(self.target_model.posterior()) 
                elif self.args["downstream_task"]=="edge":
                    self.posteriors.append(self.target_model.posterior_edge()) 

        self.posteriors = torch.stack(self.posteriors)
        if len(self.test_embeddings):
            self.test_embeddings = torch.stack(self.test_embeddings)
        #self.logger.info("Saving posteriors.")
        #self.data_store.save_posteriors(self.posteriors, self.run, suffix)

    def aggregate(self):
        if self.args['aggregator'] == 'mean':
            aggregate_f1_score, t = self._mean_aggregator()
        elif self.args['aggregator'] == 'optimal':
            aggregate_f1_score, t = self._optimal_aggregator()
        elif self.args['aggregator'] == 'majority':
            aggregate_f1_score, t = self._majority_aggregator()
        elif self.args['aggregator'] == 'contrastive':
            aggregate_f1_score, t = self._contrastive_aggregator()
        else:
            raise Exception("unsupported aggregator.")

        return aggregate_f1_score, t

    def _mean_aggregator(self):
        posterior = self.posteriors[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard]

        posterior = posterior / self.num_shards

        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"),0
        elif self.args["downstream_task"]=="edge":
            return roc_auc_score(self.true_label_edge, posterior.detach().cpu().numpy(),average="micro"),0
    def _majority_aggregator(self):
        pred_labels = []
        edge_pred_labels = []
        for shard in range(self.num_shards):
            pred_labels.append(self.posteriors[shard].argmax(axis=1).cpu().numpy())
            edge_pred = torch.where(self.posteriors[shard] > 0.5, torch.tensor(1), torch.tensor(0))
            edge_pred_labels.append(edge_pred.cpu().numpy())
        if self.args["downstream_task"]=="node":
            pred_labels = np.stack(pred_labels)
            pred_label = np.argmax(
                np.apply_along_axis(np.bincount, axis=0, arr=pred_labels, minlength=self.posteriors[0].shape[1]), axis=0)
            return f1_score(self.true_label, pred_label, average="micro"), 0
        elif self.args["downstream_task"]=="edge":
            edge_pred_labels = np.stack(edge_pred_labels)
            edge_pred_label = np.argmax(
                np.apply_along_axis(np.bincount, axis=0, arr=edge_pred_labels, minlength=self.posteriors[0].shape[0]), axis=0)
            return roc_auc_score(self.true_label_edge, edge_pred_label, average="micro")
    def _optimal_aggregator(self):
        if self.args["downstream_task"]=="node":
            optimal = OptimalAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        elif self.args["downstream_task"]=="edge":
            optimal = OptimalEdgeAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        optimal.generate_train_data()
        start_time = time.time()
        weight_para = optimal.optimization()
        
        #start_time = time.time()
        posterior = self.posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard] * weight_para[shard]
        aggr_time = time.time() - start_time

        dataset_utils.save_optimal_weight(self.logger,self.args,weight_para, run=self.run)
        
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"),aggr_time
        elif self.args["downstream_task"]=="edge":
            return roc_auc_score(self.true_label_edge, posterior.detach().cpu().numpy(), average="micro"),aggr_time
    
    def _contrastive_aggregator(self):
        proj = ContrastiveAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        proj.generate_train_data()
        
        start_time = time.time()
        proj_model = proj.optimization()#.to(self.posteriors.device)
        proj_model.eval()
        if self.args['base_model'] == 'GIN':
            self.test_embeddings = torch.tanh(self.test_embeddings)
        self.test_embeddings = self.test_embeddings.permute(1, 0, 2).to(next(proj_model.parameters()).device)
        #start_time = time.time()
        posterior = proj_model(self.test_embeddings, is_eval=True)
        aggr_time = time.time() - start_time
        
        dataset_utils.save_optimal_weight(self.logger,self.args,proj_model, run=self.run)

        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), aggr_time
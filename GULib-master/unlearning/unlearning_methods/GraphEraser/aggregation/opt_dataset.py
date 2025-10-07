from torch.utils.data import Dataset
from random import randint
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class OptDataset(Dataset):
    def __init__(self, posteriors, labels):
        self.posteriors = posteriors
        self.labels = labels

    def __getitem__(self, index):
        ret_posterior = {}

        for shard, post in self.posteriors.items():
            ret_posterior[shard] = post[index]

        return ret_posterior, self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


class NegSamplingDataset(Dataset):
    def __init__(self, posteriors, labels, adj_m, shard_nums):
        self.posteriors = torch.stack(list(posteriors.values()))
        #input(self.posteriors.shape)
        self.labels = labels
        self.shard_nums = shard_nums
        self.c_neg_indices = []

        # Generate negative sample indices for Contrastive Loss
        for i in range(labels.shape[0]):
            neg_index = randint(0, labels.shape[0] - 1)
            while labels[neg_index] == labels[i]:
                neg_index = randint(0, labels.shape[0] - 1)
            self.c_neg_indices.append(neg_index)
        
        # Generate Postive/Negative Samples for Triplet Loss
        self.a = adj_m
        self.t_pos_indices = []
        self.t_neg_indices = []
        
        num_shards = len(np.unique(shard_nums))
        shard_one_hot = F.embedding(input=torch.LongTensor(shard_nums), weight=torch.eye(num_shards))
        I = torch.eye(self.labels.shape[0])
        same_shard_m = shard_one_hot @ shard_one_hot.T - I
        dif_shard_m = torch.logical_not(same_shard_m)
        a_d = self.a.to_dense()
        valid_pos_m = dif_shard_m * a_d + I
        valid_neg_m = torch.logical_not(a_d) * 1 - I

        for i in range(labels.shape[0]):
            valid_pos = torch.nonzero(valid_pos_m[i])
            valid_neg = torch.nonzero(valid_neg_m[i])
            pos_index = valid_pos[randint(0, valid_pos.shape[0] - 1)]
            neg_index = valid_neg[randint(0, valid_neg.shape[0] - 1)]
            self.t_pos_indices.append(pos_index.item())
            self.t_neg_indices.append(neg_index.item())

    def __getitem__(self, index):
        #e = []
        #contra_neg_e = []
        #triplet_pos_e = []
        #triplet_neg_e = []

        #neg_index = randint(0, self.labels.shape[0] - 1)
        '''
        for shard, post in self.posteriors.items():
            e.append(post[index])
            contra_neg_e.append(post[self.c_neg_indices[index]])
            triplet_pos_e.append(post[self.t_pos_indices[index]])
            triplet_neg_e.append(post[self.t_neg_indices[index]])
        '''
        e = self.posteriors[:, index, :]
        contra_neg_e = self.posteriors[:, self.c_neg_indices[index], :]
        triplet_pos_e = self.posteriors[:, self.t_pos_indices[index], :]
        triplet_neg_e = self.posteriors[:, self.t_neg_indices[index], :]

        #return torch.stack(e), torch.stack(contra_neg_e), torch.stack(triplet_pos_e), torch.stack(triplet_neg_e), self.labels[index], self.shard_nums[index]
        return e, contra_neg_e, triplet_pos_e, triplet_neg_e, self.labels[index], self.shard_nums[index]
    
    def __len__(self):
        return self.labels.shape[0]
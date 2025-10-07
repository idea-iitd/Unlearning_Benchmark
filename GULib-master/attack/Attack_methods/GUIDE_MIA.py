import collections
import os
import random
import time
import numpy as np
import torch
from itertools import compress
import copy
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import SIGN
from unlearning.unlearning_methods.GUIDE.kernel_vector import PyramidMatchVector
import torch.nn.functional as F
import shutil
from config import unlearning_path




class GUIDE_MIA:
    def __init__(self,run,average_auc,avg_updating_time,args,data,looger,guide):
        self.average_auc = average_auc
        self.avg_updating_time = avg_updating_time
        self.args = args
        self.run = run
        self.logger = looger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unlearning_number = args["num_unlearned_nodes"]
        self.data = data
        self.guide = guide
        self.G_ny0 = [data.edge_index.cpu()]
        self.pm_kernel = PyramidMatchVector(normalize=True, with_labels=False)
        self.G_ny0 = self.pm_kernel.parse_input(self.G_ny0)
        self.data = self.data.cuda()
        if self.args["base_model"] == "SIGN":
            self.data.xs = torch.tensor([np.array(x.detach().cpu()) for x in self.data.xs]).cuda()

        
        self.updata_shard()
        self.attack()

    def updata_shard(self):
        id_gi_part_innerid = []
        ids = list(range(self.data.x.size(0)))
        gis = [0] * len(ids)
        inday_ids = range(len(ids))
        part_ids = self.guide.partations_fast[0].ids_shards.values()
        id_gi_part_innerid += list(zip(ids, gis, part_ids,inday_ids))
        torch.save(id_gi_part_innerid, self.guide.partations_fast[0].DPATH.split('partid')[0] + 'id_gi_part_innerid.pt')


        average_countpart = []
        train_labeled_idx = list(range(len(self.data.train_indices)))
        # self.unlearning_id = random.sample(train_labeled_idx, k=int(self.unlearning_number))
        path_un = unlearning_path + "_" + str(self.run) + ".txt"
        self.unlearning_id = np.loadtxt(path_un, dtype=int)
        is_in = np.intersect1d(self.unlearning_id, np.array(self.data.train_indices))
        sub_tuples = [id_gi_part_innerid[i] for i in self.unlearning_id]

        gi_tuples = collections.defaultdict(list)
        for tup in sub_tuples:
            gi_tuples[tup[1]].append(tup[2:])
        gi_part_uid = {}
        average_countpart_ = []
        for i in gi_tuples.keys():
            gi_tuples_in = collections.defaultdict(list)
            for j in gi_tuples[i]:
                gi_tuples_in[j[0]].append(j[1])

            gi_part_uid[i] = gi_tuples_in
            average_countpart_ += list(gi_tuples_in.keys())
        average_countpart.append(len(set(average_countpart_)))

        G_nx0 = []
        gis_keys_graph = {}
        gis_keys_graph[0] = {}
        for part_id in range(self.args['num_shards']):
            loadname = self.guide.partations_fast[0].DPATH.replace('partid', 'part' + str(part_id))
            sub_graph = torch.load(loadname)
            gis_keys_graph[0][part_id] = sub_graph

        gis_keys_graph_update = copy.deepcopy(gis_keys_graph)
        part_set = []
        time_sum = 0
        for gi in gi_part_uid.keys():
            start_time = time.time()
            for part in gi_part_uid[gi].keys():
                part_set.append(part)
                #self.logger.info("part  {}, train number: {}".format(part, gis_keys_graph_update[gi][part].train_mask.sum().item()))
                delete_subindex = []
                for id in gi_part_uid[gi][part]:
                    newindex = self.index_to_subindex(part_ids,id)
                    gis_keys_graph_update[gi][part].y = tensor = torch.cat((gis_keys_graph_update[gi][part].y[:newindex], gis_keys_graph_update[gi][part].y[newindex+1:]))
                    delete_subindex.append(self.find_uid_2(newindex,gis_keys_graph_update[gi][part].train_mask))

                reserve_mask = torch.tensor([False if id in delete_subindex else True for id in gis_keys_graph_update[gi][part].uids])

                sub_graph = gis_keys_graph_update[gi][part].subgraph(reserve_mask)
                sub_graph.uids = list(compress(sub_graph.uids, reserve_mask.tolist()))
                gis_keys_graph_update[gi][part] = sub_graph
                #self.logger.info("part  {}, after delete train number: {}".format(part, sub_graph.train_mask.sum().item()))
            time_sum+= time.time()-start_time
        part_set = set(part_set)
        self.logger.info("update_time:{}".format(time_sum/10))

        
        for part in part_set:
            #self.logger.info("part  {} ".format(part))
            time_sum = 0
            for gi in gis_keys_graph_update.keys():
                G_nx0.append(gis_keys_graph_update[gi][part].edge_index.to('cpu'))

            submodel = self.guide.target_model
            submodel.model.to(self.device)
            optimizer = torch.optim.Adam(
                submodel.model.parameters(), lr=0.01, weight_decay=1e-5)
            #self.logger.info("gis_keys_graph_update {}".format(gis_keys_graph_update[0][part]))
            
            for epoch in range(1, 8):
                start_time = time.time()
                submodel.model.train()
                sub_graph = gis_keys_graph_update[0][part]
                sub_graph = sub_graph.to(self.device)
                labels = sub_graph.y
                optimizer.zero_grad()
                if self.args["base_model"] == "SIGN":
                    sub_graph = SIGN(self.args["GNN_layer"])(sub_graph)
                    sub_graph.xs = [sub_graph.x] + [sub_graph[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                    sub_graph.xs = torch.tensor([x.detach().cpu().numpy() for x in sub_graph.xs]).cuda()
                    sub_graph.xs = sub_graph.xs.transpose(0,1)
                    out = submodel.model(sub_graph.xs)
                else:
                    out = submodel.model(sub_graph.x, sub_graph.edge_index)
                out = out[sub_graph.train_mask]
                #self.logger.info("labels  {},  out  {}".format(labels.shape,out.shape))
                loss = F.cross_entropy(out, labels[sub_graph.train_mask[:labels.size(0)]])


                loss.backward()
                optimizer.step()
                time_sum+= time.time()-start_time
            self.logger.info("training_time:{}".format(time_sum/7))
            self.avg_updating_time[self.run] = time_sum/7
            submodel.model.to('cpu')
            sub_graph.to('cpu')
            savemodeln = self.guide.partations_fast[0].MPATH.replace('/partid', '_copy/partid').replace('partid',
                                                                                              'part' + str(part))
            os.makedirs(os.path.dirname(savemodeln), exist_ok=True)
            torch.save(submodel.model.state_dict(), savemodeln)

        G_nx0 = self.pm_kernel.parse_input(G_nx0)
        self.method = self.args["GUIDE_methods"]
        self.positive0, self.negative0 = {}, {}
        self.positive0[self.method], self.negative0[self.method] = [], []
        submodellist = []
        part_set_all = range(self.args['num_shards'])
        for part in part_set_all:
            submodel = self.guide.target_model
            loadname = self.guide.partations_fast[0].MPATH.replace('partid', 'part' + str(part_id))

            submodel.load_model(loadname)
            submodel.model.eval()
            submodellist.append(submodel.model.cuda())
        subouts_train = []
        subout_tests = []
        part_set_all = range(self.args['num_shards'])
        for part in part_set_all:
            subout_train = []
            if self.args["base_model"] == "SIGN":
                subout_train_daily = torch.softmax(submodellist[part](self.data.xs),dim=1)
            else:
                subout_train_daily = torch.softmax(submodellist[part](self.data.x, self.data.edge_index), dim=1)
            subout_train.append(subout_train_daily)

            subout_train = torch.concat(subout_train)
            subout_train_ = subout_train[self.unlearning_id]
            subouts_train.append(subout_train_.detach())

            subout_test = []
            if self.args["base_model"] == "SIGN":
                subout_test_daily = torch.softmax(submodellist[part](self.data.xs),dim=1)[self.data.test_mask]
            else:
                subout_test_daily = torch.softmax(submodellist[part](self.data.x, self.data.edge_index), dim=1)[self.data.test_mask]
            subout_test.append(subout_test_daily)

            subout_test = torch.concat(subout_test)
            subout_tests.append(subout_test.detach())

        self.suboutTensor = torch.stack(subouts_train)
        self.suboutTensorn = torch.stack(subout_tests)

        weights_gi = torch.load(
            self.guide.partations_fast[0].MPATH.replace('partid/submodels/', '').replace('model_partid', '_weight'))

        summed = torch.tensordot(
            self.suboutTensor, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
        self.positive0[self.method].append(summed.cpu())

        summed = torch.tensordot(
            self.suboutTensorn, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
        self.negative0[self.method].append(summed.cpu())

        tmp = time.strftime("%Y%m%d-%H%M%S")
        savename = self.guide.partations_fast[0].MPATH.split('partid')[
                       0] + f'evaluation_attack_partition_{self.method}_' + self.args[
                       "dataset_name"] + '_unlearning_idrnd'
        savename += ''.join(tmp)
        savename += ''.join('.pt')
        torch.save(self.unlearning_id, savename)

        savename = self.guide.partations_fast[0].MPATH.split('partid')[
                       0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_positive0'
        savename += ''.join(tmp)
        savename += ''.join('.pt')
        torch.save(self.positive0, savename)
        savename = self.guide.partations_fast[0].MPATH.split('partid')[
                       0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_negative0'
        savename += ''.join(tmp)
        savename += ''.join('.pt')
        torch.save(self.negative0, savename)

        self.positive1, self.negative1 = {}, {}
        self.positive1[self.method], self.negative1[self.method] = [], []

        submodellist = []
        part_set_all = range(self.args['num_shards'])
        for part in part_set_all:
            submodel = self.guide.target_model

            loadname = self.guide.partations_fast[0].MPATH.replace('partid', 'part' + str(part_id)).replace(self.args["dataset_name"], self.args["dataset_name"]+'_copy')

            submodel.load_model(loadname)
            submodel.model.eval()
            submodellist.append(submodel.model)

        subouts_train = []
        subout_tests = []
        part_set_all = range(self.args['num_shards'])
        for part in part_set_all:
            subout_train = []
            if self.args["base_model"] == "SIGN":
                subout_train_daily = torch.softmax(submodellist[part](self.data.xs),dim=1)
            else:
                subout_train_daily = torch.softmax(submodellist[part](self.data.x, self.data.edge_index), dim=1)
            subout_train.append(subout_train_daily)

            subout_train = torch.concat(subout_train)
            subout_train_ = subout_train[self.unlearning_id]
            subouts_train.append(subout_train_.detach())

            subout_test = []
            if self.args["base_model"] == "SIGN":
                subout_test_daily = torch.softmax(submodellist[part](self.data.xs),dim=1)[self.data.test_mask]
            else:
                subout_test_daily = torch.softmax(submodellist[part](self.data.x, self.data.edge_index), dim=1)[self.data.test_mask]
            subout_test.append(subout_test_daily)

            subout_test = torch.concat(subout_test)
            subout_tests.append(subout_test.detach())

        self.suboutTensor = torch.stack(subouts_train)
        self.suboutTensorn = torch.stack(subout_tests)

        # weights_gi = torch.tensor([1.0/args['shards_number']
        #                         for _ in range(args['shards_number'])])
        weights_gi = torch.load(
            self.guide.partations_fast[0].MPATH.replace('partid/submodels/', '').replace('model_partid', '_weight'))

        summed = torch.tensordot(
            self.suboutTensor, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
        self.positive1[self.method].append(summed.cpu())

        summed = torch.tensordot(
            self.suboutTensorn, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
        self.negative1[self.method].append(summed.cpu())

        savename = self.guide.partations_fast[0].MPATH.split('partid')[0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_positive1'
        savename += ''.join(tmp)
        savename += ''.join('.pt')
        torch.save(self.positive1, savename)
        savename = self.guide.partations_fast[0].MPATH.split('partid')[0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_negative1'
        savename += ''.join(tmp)
        savename += ''.join('.pt')
        torch.save(self.negative1, savename)

    def attack(self):

        positive_posteriors, negative_posteriors = [], []
        positive_posteriors.append(self.positive0[self.method])
        positive_posteriors.append(self.positive1[self.method])
        negative_posteriors.append(self.negative0[self.method])
        negative_posteriors.append(self.negative1[self.method])

        size_number = self.args["num_unlearned_nodes"]
        # break

        positive_posteriors[0] = torch.cat(positive_posteriors[0], dim=0)
        positive_posteriors[1] = torch.cat(positive_posteriors[1], dim=0)
        negative_posteriors[0] = torch.cat(negative_posteriors[0], dim=0)
        negative_posteriors[1] = torch.cat(negative_posteriors[1], dim=0)

        attack_auc_b = []
        for _ in range(1000):
            batch_test = random.choices(range(negative_posteriors[0].shape[0]), k=size_number)
            label = torch.cat((torch.ones(size_number), torch.zeros(size_number)))
            data = {}
            for i in range(2):
                data[i] = torch.cat((positive_posteriors[i], negative_posteriors[i][batch_test, :]), 0)

            # calculate l2 distance
            model_b_distance = self.calculate_distance(data[0], data[1])
            # directly calculate AUC with feature and labels
            attack_auc_b_ = self.evaluate_attack_with_AUC(model_b_distance.cpu().numpy(), label)
        # print("Attack_Model_B AUC: %s " % (attack_auc_b_))
            attack_auc_b.append(attack_auc_b_)

        print(self.args["dataset_name"]+
            f' Pos:Neg Average Attack_Model_B  AUC: {self.method} {(np.array(attack_auc_b).mean())}   {(np.array(attack_auc_b).var())}')
        
        self.average_auc[self.run] = np.array(attack_auc_b).mean()

    def calculate_distance(self,data0, data1, distance='l2_norm'):
        if distance == 'l2_norm':
            return torch.norm(data0 - data1, dim=1)
        elif distance == 'direct_diff':
            return data0 - data1
        else:
            raise Exception("Unsupported distance")

    def evaluate_attack_with_AUC(self,data, label):
        return roc_auc_score(label, data.reshape(-1, 1))


    def copy_file(self):
        src_dir = './checkpoints/cora'
        dst_dir = './checkpoints/cora_copy'

        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        shutil.copytree(src_dir, dst_dir)

    def index_to_subindex(self,part_ids,index):
        part_ids = list(part_ids)
        subindex = part_ids[:index].count(part_ids[index])
        return subindex

    def reverse_y(self,y,train_mask):
        ori_y = torch.zeros(len(train_mask))
        count = 0
        for i in range(len(train_mask)):
            if train_mask[i]:
                ori_y[i] = y[count]
                count += 1
            else:
                ori_y[i] = -1
        return  ori_y

    def find_uid(self,id,train_mask):
        for uid in range(len(train_mask)):
            if sum(train_mask[:uid]) == id:
                return uid
            
    def find_uid_2(self,id,train_mask):
        return id

import os
import time
import collections
import copy
from torch_geometric.transforms import SIGN
import torch
import random
from itertools import compress
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from unlearning.unlearning_methods.GUIDE.guide_func import GUIDE_FUNC
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from task.node_classification import NodeClassifier
from unlearning.unlearning_methods.GUIDE.kernel_vector import PyramidMatchVector
import shutil
from config import unlearning_path
from tqdm import tqdm
from config import BLUE_COLOR,RESET_COLOR
from task.edge_prediction import EdgePredictor
from task import get_trainer
from pipeline.Shard_based_pipeline import Shard_based_pipeline
from torch_geometric.utils import negative_sampling
import sys
import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from model.base_gnn.gcn import GCNNet
from model.base_gnn.deletion import GCNDelete
from sklearn.metrics import accuracy_score
import os
import argparse
from unlearning.unlearning_methods.Projector.utils.graph_projector_model_utils import Pro_GNN
import copy
from torch_sparse import SparseTensor

filename = "GUIDE_utility_stats.txt"

def append_to_file(filename: str, text: str):
    with open(filename, "a") as f:
        f.write(text + "\n")

class guide(Shard_based_pipeline):
    """
    The `guide` class is a specialized implementation of the `Shard_based_pipeline` class designed for 
    graph unlearning tasks using the GUIDE method. 
    This method partitions the graph using the GUIDE method, which is different from the other methods like GraphEraser. It repairs the subgraphs and trains models on each subgraph.

    Attributes:
        args (dict): Configuration arguments for the GUIDE method and unlearning tasks.

        logger (Logger): Logger for recording information and debugging.

        model_zoo (ModelZoo): Collection of models and data used in the pipeline.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.num_classes = self.data.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pm_kernel = PyramidMatchVector()
        self.run = 0
        self.unlearning_number = args["num_unlearned_nodes"]
        num_runs = self.args["num_runs"]
        self.G_ny0 = [self.data.edge_index.cpu()]
        u_ratio = self.args['unlearn_ratio']
        dataset = self.args['dataset_name']
        num_nodes = self.data.y.size(0)
        unlearn_idx_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/unlearning_task/transductive/imbalanced/unlearning_nodes_{u_ratio}_{dataset}_0_nodes_{int(u_ratio * num_nodes)}.txt"

        with open(unlearn_idx_path, "r") as f:
            unlearned_indices = list(map(int, f.readlines()))

        train_mask = self.data.train_mask.cpu().numpy()
        test_mask = self.data.test_mask.cpu().numpy()

        unlearned_mask = np.zeros_like(train_mask, dtype=bool)
        unlearned_mask[unlearned_indices] = True

        self.retained_train_mask = np.logical_and(train_mask, np.logical_not(unlearned_mask))
        self.unlearned_train_mask = np.logical_and(train_mask, unlearned_mask)
        self.test_mask = test_mask

        
        

    # def run_exp(self):
    #     for self.run in range(self.args['num_runs']):
    #         self.logger.info("run :{}".format(self.run))
    #         self.partition()
    #         self.train_shard_model()
    #         self.aggregate_shard_model()
    #         self.update_shard()
    #         if self.args["unlearn_task"] == "node":
    #             self.mia_attack()
    #     self.logger.info(
    #     "{}Performance Metrics:\n"
    #     " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #     " - Average AUC Score: {:.4f} ± {:.4f}\n"
    #     " - Average Partition Time: {:.4f} ± {:.4f} seconds\n"
    #     " - Average Training Time: {:.4f} ± {:.4f} seconds\n"
    #     " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
    #         BLUE_COLOR,
    #         np.mean(self.average_f1), np.std(self.average_f1),
    #         np.mean(self.average_auc), np.std(self.average_auc),
    #         np.mean(self.avg_partition_time), np.std(self.avg_partition_time),
    #         np.mean(self.avg_training_time), np.std(self.avg_training_time),
    #         np.mean(self.avg_updating_time), np.std(self.avg_updating_time),
    #         RESET_COLOR
    #         )
    #     )
    #     self.logger.info("avg_updating_time:{}".format(self.avg_updating_time))

    
    def graph_partition(self):
        """
        Override of Shard_based_pipeline.graph_partition. Partitions the graph using the GUIDE method.
        This function overrides the `graph_partition` method from the `Shard_based_pipeline` class.
        It initializes the GUIDE methods with the provided arguments and data, and then fits the 
        partitioning model using the specified GUIDE method.
        """
        self.GUIDE_methods = GUIDE_FUNC(self.args,self.data,edge_indexs=self.data.edge_index, labels=self.data.y, k = self.args['num_shards'])
        self.p1 = self.GUIDE_methods.fit(self.avg_partition_time,self.run,method=self.args["GUIDE_methods"], alpha_=1e-3)
        
    def generate_shard_data(self):
        """
        Generates shard data for the graph neural network model based on the partitioning results.
        """
        # if self.args["dataset_name"] in ["flickr","Chameleon","Minesweeper","Tolokers"]:
        #     self.p1_saved = self.GUIDE_methods.subgraph_repair(x=self.data.x, REPAIR_METHOD=self.args["GUIDE_repair_methods"], PATH='./data/GUIDE/checkpoints/',
        #                                                    DATA_NAME='{}'.format(self.args['dataset_name']), MULTI_GRAPH=0,no_repair=True)
        # else:
        self.p1_saved = self.GUIDE_methods.subgraph_repair(x=self.data.x, REPAIR_METHOD=self.args["GUIDE_repair_methods"], PATH='./data/GUIDE/checkpoints/',
                                                           DATA_NAME='{}'.format(self.args['dataset_name']), MULTI_GRAPH=0)
        # self.p1_saved = self.GUIDE_methods.subgraph_repair(x=self.data.x, REPAIR_METHOD=self.args["GUIDE_repair_methods"], PATH='./data/GUIDE/checkpoints/',
        #                                                   DATA_NAME='{}'.format(self.args['dataset_name']), MULTI_GRAPH=0)
        savename = self.p1_saved.DPATH.split('partid')[0] + self.p1_saved.method + '_saved.pt'
        torch.save(self.p1_saved, savename)
        print("SAVED AT ", savename)

        self.gis_keys_graph = {}
        for part_id in range(self.args['num_shards']):
            loadname = self.p1_saved.DPATH.replace('partid', 'part' + str(part_id))
            sub_graph = torch.load(loadname)
            self.gis_keys_graph[part_id] = sub_graph
            # kernel feature extraction for further use
            subg_kfea = self.pm_kernel.parse_input([sub_graph.edge_index.cpu()])
            savekfean = loadname.replace('subgraphs', 'subgkfeas').replace('.pt', '_PM' + '.pt')
            os.makedirs(os.path.dirname(savekfean), exist_ok=True)
            torch.save(subg_kfea, savekfean)
            self.p1_saved.FPATH = savekfean.replace('part' + str(part_id), 'partid')
            # model save path
            savemodeln = loadname.replace('subgraphs', 'submodels').replace('.pt', '_' + self.args[ "base_model"] + '.pt').replace(
                'part' + str(part_id), 'partid')
            savemodeln = savemodeln.split('graph')[0] + 'model_part' + savemodeln.split('_part')[1]
            self.p1_saved.MPATH = savemodeln.replace('part' + str(part_id), 'partid')

    def determine_target_model(self):
        """
        Determines and sets the target model for the unlearning process.
        This method sets the 'unlearn_trainer' argument to 'GUIDETrainer' and 
        initializes the target model using the get_trainer function with the 
        provided arguments, logger, model from the model zoo, and data.
        """
        self.args["unlearn_trainer"] = 'GUIDETrainer'
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        
        
    def train_shard_model(self):
        """
        Trains a shard model for each part ID in the saved shards.
        This function iterates over each part ID in the saved shards, trains a submodel on the corresponding subgraph, 
        and saves the trained submodel to disk. The training process includes resetting model parameters, moving the 
        model and subgraph to the appropriate device, and optimizing the model using the Adam optimizer. If the base 
        model is "SIGN", the subgraph is processed accordingly. The average training time per epoch is recorded, and 
        the trained model is saved to a specified path.
        """
        for part_id in self.p1_saved.shards_ids.keys():
            # submodel training
            start = time.time()
            submodel = self.target_model
            submodel.model.reset_parameters()
            submodel.model.to(self.device)
            submodel.optimizer = torch.optim.Adam(submodel.model.parameters(), lr=0.01, weight_decay=1e-5)
            sub_graph = self.gis_keys_graph[part_id]
            sub_graph = sub_graph.to(self.device)
            if self.args["base_model"] == "SIGN":
                    sub_graph = SIGN(self.args["GNN_layer"])(sub_graph)
                    sub_graph.xs = [sub_graph.x] + [sub_graph[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                    sub_graph.xs = torch.tensor([x.detach().cpu().numpy() for x in sub_graph.xs]).cuda()
                    sub_graph.xs = sub_graph.xs.transpose(0,1)
            submodel.data = sub_graph
            submodel.train_node(save=False)

            self.avg_training_time[self.run] = (time.time()-start)/self.args['num_epochs']
            submodel.model.to('cpu')
            sub_graph = sub_graph.to('cpu')
            savemodeln = self.p1_saved.MPATH.replace('partid', 'part' + str(part_id))
            os.makedirs(os.path.dirname(savemodeln), exist_ok=True)
            torch.save(submodel.model.state_dict(), savemodeln)
            self.logger.info("save model {}".format(savemodeln))
            print(f"saving model of SHARD at {savemodeln}")
    # def train_model(self):
    #     self.target_model.train()
    #     y_pred,y =self.target_model.prediction_info()
    #     y_pred = y_pred.numpy()
    #     y = y.numpy()
    #     accuracy, f1macro, aucroc = self.store_metrics(y_pred,y)
    #     self.logger.info("accuracy {:} f1macro: {:.4f} aucroc: {:.4f}".format(
    #         accuracy, f1macro, aucroc))

    def store_metrics(self,pred_scores, target_labels):
        """
        Calculate and return evaluation metrics for given prediction scores and target labels.
        This function computes the accuracy, F1 macro score, and AUC-ROC score for the provided 
        prediction scores and target labels.
        """
        # calculate metrics
        pred_labels = pred_scores.argmax(axis=1)
        one_hot_labels = np.eye(self.num_classes)[target_labels]
        accuracy = accuracy_score(target_labels, pred_labels)
        f1macro = f1_score(target_labels, pred_labels, average='macro')
        aucroc = roc_auc_score(one_hot_labels, pred_scores,multi_class='ovr',average='macro')

        return accuracy, f1macro, aucroc

    # def partition(self):
    #     self.partations_fast = []
    #     # if not self.args["is_transductive"]:
    #     #     self.GUIDE_methods = GUIDE_FUNC(self.args,self.data,edge_indexs=self.data.train_edge_index, labels=self.data.y[self.data.train_indices], k = self.args['num_shards'])
    #     #     self.p1 = self.GUIDE_methods.fit(self.avg_partition_time,self.run,method=self.args["GUIDE_methods"], alpha_=1e-3)
    #     # else:
    #     self.GUIDE_methods = GUIDE_FUNC(self.args,self.data,edge_indexs=self.data.edge_index, labels=self.data.y, k = self.args['num_shards'])
    #     self.p1 = self.GUIDE_methods.fit(self.avg_partition_time,self.run,method=self.args["GUIDE_methods"], alpha_=1e-3)
    #     # self.logger.info("avg_partition_time:{}".format(self.avg_partition_time[self.run]))
    #     # if not self.args["is_transductive"]:
    #     #     self.p1_saved = self.GUIDE_methods.subgraph_repair(x=self.data.x[self.data.train_indices], REPAIR_METHOD=self.args["GUIDE_repair_methods"], PATH='./data/GUIDE/checkpoints/',
    #     #                                                    DATA_NAME='{}'.format(self.args['dataset_name']), MULTI_GRAPH=0)
    #     # else:
    #     self.p1_saved = self.GUIDE_methods.subgraph_repair(x=self.data.x, REPAIR_METHOD=self.args["GUIDE_repair_methods"], PATH='./data/GUIDE/checkpoints/',
    #                                                        DATA_NAME='{}'.format(self.args['dataset_name']), MULTI_GRAPH=0)
    #     savename = self.p1_saved.DPATH.split('partid')[0] + self.p1_saved.method + '_saved.pt'
    #     torch.save(self.p1_saved, savename)
    #     self.partations_fast.append(self.p1_saved)

    #     self.gis_keys_graph = {}
    #     for part_id in range(self.args['num_shards']):
    #         loadname = self.partations_fast[0].DPATH.replace('partid', 'part' + str(part_id))
    #         sub_graph = torch.load(loadname)
    #         self.gis_keys_graph[part_id] = sub_graph
    #         # kernel feature extraction for further use
    #         subg_kfea = self.pm_kernel.parse_input([sub_graph.edge_index.cpu()])
    #         savekfean = loadname.replace('subgraphs', 'subgkfeas').replace('.pt', '_PM' + '.pt')
    #         os.makedirs(os.path.dirname(savekfean), exist_ok=True)
    #         torch.save(subg_kfea, savekfean)
    #         self.partations_fast[0].FPATH = savekfean.replace('part' + str(part_id), 'partid')
    #         # model save path
    #         savemodeln = loadname.replace('subgraphs', 'submodels').replace('.pt', '_' + self.args[ "base_model"] + '.pt').replace(
    #             'part' + str(part_id), 'partid')
    #         savemodeln = savemodeln.split('graph')[0] + 'model_part' + savemodeln.split('_part')[1]
    #         self.partations_fast[0].MPATH = savemodeln.replace('part' + str(part_id), 'partid')


    def aggregate_shard_model(self,after_unlearning=False):
        """
        Aggregates the shard models to produce a final prediction.
        This function loads the shard models, computes their predictions, and aggregates these predictions
        using a weighted sum based on kernel similarity. The aggregated predictions are then evaluated
        based on the specified downstream task.
        """
        results_Fast = {
            'acc': np.zeros((16, 1)),
            'f1macro': np.zeros((16, 1)),
            'aucroc': np.zeros((16, 1))
        }
        # if(self.args["is_transductive"]):
        #     subgt_kfea = self.pm_kernel.parse_input([self.data.edge_index.cpu()])
        # else:
        subgt_kfea = self.pm_kernel.parse_input([self.data.edge_index.cpu()])
        #self.logger.info("self.data.test_edge_index {}".format(self.data.test_edge_index.shape))
        subg_kfea = []
        subouts = []
        for part_id in self.p1_saved.shards_ids.keys():
            loadname = self.p1_saved.FPATH.replace('partid', 'part' + str(part_id))
            print(f"currently loading {loadname}")
            subg_kfea += torch.load(loadname)

            submodel = self.target_model
            savemodeln = self.p1_saved.MPATH.replace('/partid', '/partid').replace('partid', 'part' + str(part_id))
            print(f"currently loading model {savemodeln}")
            submodel.load_model(savemodeln)
            
            
            submodel.model.to(self.device)
            submodel.model.eval()
            subgraph = self.gis_keys_graph[part_id]
            if self.args["downstream_task"] == "node":
                if self.args["base_model"] == "SIGN":
                    subout = torch.softmax(submodel.model(self.data.xs[self.data.test_indices]),dim=1)
                else:
                    subout = torch.softmax(submodel.model(self.data.x.to(self.device),self.data.edge_index.to(self.device)),dim = 1)
                    subout = subout
                #subout = subout[self.data.test_mask]
                #self.logger.info("subout {}".format(subout.shape))
                subouts.append(subout.detach().to('cpu'))
                submodel.model.to('cpu')
            elif self.args["downstream_task"] == "edge":
                # submodel.data = subgraph
                # AUC_score = submodel.evaluate_edge_model()
                # print(AUC_score)
                # print(self.data)
                neg_edge_index = negative_sampling(
                    edge_index=self.data.edge_index,num_nodes=self.data.num_nodes,
                    num_neg_samples=self.data.test_edge_index.size(1)
                )
                # neg_edge_index = self.data.test_neg_edge_index
                if self.args["base_model"] == "SIGN":
                    subout = submodel.model(self.data.xs)
                else:
                    subout = submodel.model(self.data.x.to(self.device),self.data.train_edge_index.to(self.device))
                subout = submodel.decode(subout,self.data.test_edge_index,neg_edge_index)
                
                # pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
                # neg_edge_labels = torch.zeros(neg_edge_index.size(1),dtype=torch.float32)
                # edge_pred = torch.where(torch.sigmoid(subout) > 0.5, torch.tensor(1), torch.tensor(0))
                # target_labels = torch.cat((pos_edge_labels,neg_edge_labels))
                # AUC_score = roc_auc_score(target_labels.cpu(), edge_pred.detach().cpu().numpy())
                # print(AUC_score)
                
                subouts.append(subout.detach().to('cpu'))
                submodel.model.to('cpu')
            del submodel, subout
            

        suboutTensor = torch.stack(subouts)

        weights = self.pm_kernel.kernel_similarity(subgt_kfea, subg_kfea)
        weights = torch.tensor(weights).to(torch.float32).reshape(len(self.p1_saved.shards_ids.keys()),
                                                                              -1).mean(dim=-1)
        saveweightn = self.p1_saved.MPATH.replace('partid/submodels/', '').replace('model_partid',
                                                                                         f'_weight')
        os.makedirs(os.path.dirname(saveweightn), exist_ok=True)
        torch.save(weights, saveweightn)

        weights = F.softmax(weights, dim=0)
        weighted_pred = torch.tensordot(suboutTensor, weights, dims=([0], [0]))
        # print(suboutTensor,weights,weighted_pred)
        #self.logger.info("weighted_pred shape{}".format(weighted_pred.detach().cpu().numpy().shape))
        if self.args["downstream_task"] == "node":
            if 'labeled_mask' in self.data:
                target_labels = self.data.y[self.data.labeled_mask].detach().cpu().numpy()
            else:
                target_labels = self.data.y.detach().cpu().numpy()
            test_acc, test_f1macro, test_aucroc = self.store_metrics(weighted_pred.detach().cpu().numpy(), target_labels)
            self.logger.info(f"Test F1 Score: {test_f1macro:.4f}")
            if after_unlearning:
                self.average_f1[self.run]  = test_f1macro
                self.average_auc[self.run] = test_aucroc

                #--------------------------------------------------
                dataset = self.args['dataset_name']
                unlearn_ratio_str = f"ratio_{self.args['unlearn_ratio']:.2f}"
                gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GOLD/{dataset}/node/{unlearn_ratio_str}/GOLD_{dataset}_node_{unlearn_ratio_str}.pt"
                data_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/processed/transductive/{dataset}0.8_0_0.2.pkl"
                original_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/node/GCN"
                
                append_to_file(filename, f"-----------------    {dataset} --------------- GUIDE --------------------- {self.args['unlearn_ratio']:.2f}")

                with open(data_path, "rb") as f:
                    gold_data = pickle.load(f)
                gold_data = gold_data.to("cuda")



                args = {
                    'dataset_name': f"{self.args['dataset_name']}",
                    'downstream_task': 'node',
                    'base_model': 'GCN',
                    'hidden_dim': 64,
                    # 'out_dim': data.y.max().item() + 1
                }

                def load_and_predict(model_path, model_type="GOLD", unlearned_param_path=None, data=None):
                    copy_data = copy.deepcopy(data)
                    args['out_dim'] = data.y.max().item() + 1

                    if model_type == 'Projector':
                        extra_feats = torch.zeros(copy_data.x.size(0), device=copy_data.x.device)
                        extra_feats[unlearned_nodes] = 1
                        copy_data.x = torch.cat([copy_data.x, extra_feats.view(-1, 1)], dim=1)

                        checkpoint = torch.load(model_path)
                        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
                        W_shape = state_dict['W'].shape
                        y_dims = W_shape[1]
                        x_iters = 3
                        y_iters = 3
                        x_dims = copy_data.x.size(1)

                        projector_args = {
                            # 'dataset_name': dataset,
                            'downstream_task': 'node',
                            'base_model': 'GCN',
                            'hidden_dim': 64,
                            'out_dim': y_dims,
                            'x_iters': x_iters,
                            'y_iters': y_iters,
                            'use_adapt_gcs': False,
                            'use_cross_entropy': True,
                        }

                        model = Pro_GNN(x_dims, y_dims, device="cuda", args=projector_args).to("cuda")
                        model.load_state_dict(state_dict)

                    elif model_type == 'GNNDelete':
                        model = GCNDelete(args, data.num_node_features, args['out_dim'])

                    else:  # Includes "GOLD" and "GIF"
                        model = GCNNet(args, data.num_node_features, args['out_dim'])

                    model = model.to("cuda")

                    # Load and apply base weights
                    

                    # For GIF: load parameters and unlearned data (x_unlearn + edge_index_unlearn)
                    if model_type == "GIF" or model_type == "IDEA":
                        assert unlearned_param_path is not None, "GIF model requires a path to unlearned parameters"
                        
                        # Load unlearned weights
                        params_esti = torch.load(unlearned_param_path)
                        idx = 0
                        for p in model.parameters():
                            p.data = params_esti[idx].to(p.device)
                            idx += 1

                        # Derive the save directory from unlearned_param_path
                        save_dir = os.path.dirname(unlearned_param_path)

                        # Load x_unlearn and edge_index_unlearn
                        copy_data.x_unlearn = torch.load(os.path.join(save_dir, "x_unlearn.pt")).to("cuda")
                        copy_data.edge_index_unlearn = torch.load(os.path.join(save_dir, "edge_index_unlearn.pt")).to("cuda")
                    
                    else:
                        checkpoint = torch.load(model_path)
                        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
                        model.load_state_dict(state_dict, strict=True)

                    model.eval()
                    with torch.no_grad():
                        if model_type == "Projector":
                            subgraph_data = copy_data.clone()
                            subgraph_data.adj_t = SparseTensor.from_edge_index(copy_data.edge_index).t().to("cuda")
                            subgraph_data.y_one_hot_train = F.one_hot(copy_data.y, y_dims).float().to("cuda")
                            subgraph_data.root_n_id = torch.arange(copy_data.x.size(0), device='cuda')
                            logits = model(subgraph_data)

                        elif model_type == 'GNNDelete':
                            logits = model(data.x, data.edge_index, mask_1hop=mask_1hop, mask_2hop=mask_2hop)

                        elif model_type == "GIF" or model_type == 'IDEA':
                            logits = model.reason_once_unlearn(copy_data)

                        else:  # "GOLD" or standard
                            logits = model(data.x, data.edge_index)

                        preds = torch.argmax(F.softmax(logits, dim=-1), dim=1).cpu().numpy()
                    return preds

                def exact_match(y1, y2, mask):
                    return np.mean(y1[mask] == y2[mask])

                def region_wise_exact(ref, comp, retained_train_mask, unlearned_train_mask, test_mask):
                    for name, mask in [
                        ("Retained Train Nodes", retained_train_mask),
                        ("Unlearned Train Nodes", unlearned_train_mask),
                        ("Test Nodes", test_mask),
                        ("Full Dataset", None)
                    ]:
                        exact = exact_match(ref, comp, mask) if mask is not None else np.mean(ref == comp)
                        print(f"[{name}] Exact Match: {exact:.4f}")
                        append_to_file(filename, f"[{name}] Exact Match: {exact:.4f}")



                gold_preds = load_and_predict(
                    model_path=gold_model_path,
                    model_type="GOLD",
                    data=gold_data
                )

                orz_preds = load_and_predict(
                    model_path=original_model_path,
                    model_type="GOLD",
                    data=gold_data
                )
                # breakpoint()
                aggregated_preds = weighted_pred.argmax(dim=1).cpu().numpy()

                # Region-wise exact match
                append_to_file(filename, "----------------------------------------- GOLD VS GUIDE---------------------------------------------------")
                region_wise_exact(
                    ref=gold_preds,
                    comp=aggregated_preds,
                    retained_train_mask=self.retained_train_mask,
                    unlearned_train_mask=self.unlearned_train_mask,
                    test_mask=self.test_mask
                )

                append_to_file(filename, "----------------------------------------- GOLD VS ORG---------------------------------------------------")
                region_wise_exact(
                    ref=gold_preds,
                    comp=orz_preds,
                    retained_train_mask=self.retained_train_mask,
                    unlearned_train_mask=self.unlearned_train_mask,
                    test_mask=self.test_mask
                )

                append_to_file(filename, "----------------------------------------- ORG VS GUIDE---------------------------------------------------")
                region_wise_exact(
                    ref=orz_preds,
                    comp=aggregated_preds,
                    retained_train_mask=self.retained_train_mask,
                    unlearned_train_mask=self.unlearned_train_mask,
                    test_mask=self.test_mask
                )

                # Global match on test mask
                jaccard = np.mean(gold_preds[self.test_mask] == aggregated_preds[self.test_mask])
                test_acc_g = accuracy_score(aggregated_preds[self.test_mask], self.data.y[self.test_mask])
                test_acc_o = accuracy_score(orz_preds[self.test_mask], self.data.y[self.test_mask])
                test_acc_r = accuracy_score(gold_preds[self.test_mask], self.data.y[self.test_mask])

                print("\ntest acc guide", test_acc_g)
                append_to_file(filename, f"test accuracy of GUIDE is {test_acc_g}")
                print("test acc retrain gold", test_acc_r)
                append_to_file(filename, f"test accuracy of retrain GOLD is {test_acc_r}")
                print("test acc original", test_acc_o)
                append_to_file(filename, f"test accuracy of original is {test_acc_o}")
                self.logger.info(f"[Unlearning] Jaccard similarity (test mask) with gold retrain: {jaccard:.4f}")
                #----------------------end---------------------------------------




        elif self.args["downstream_task"] == "edge":
            
            pos_edge_labels = torch.ones(self.data.test_edge_index.size(1),dtype=torch.float32)
            neg_edge_labels = torch.zeros(self.data.test_edge_index.size(1),dtype=torch.float32)
            edge_pred = torch.where(torch.sigmoid(weighted_pred) > 0.5, torch.tensor(1), torch.tensor(0))
            # edge_pred = weighted_pred
            target_labels = torch.cat((pos_edge_labels,neg_edge_labels))
            AUC_score = roc_auc_score(target_labels.cpu(), edge_pred.detach().cpu().numpy())

            if after_unlearning:
                self.average_f1[self.run] = AUC_score
            else:
                self.poison_f1[self.run] = AUC_score
        

    def update_shard(self):
        """
        Updates the shard by performing unlearning tasks on edges, nodes, or features based on the specified unlearning task.
        """
        self.start_time = time.time()
        if self.args["unlearn_task"] == "edge":
            #get the deleting edges
            edge_index = self.data.edge_index.numpy()
            self.num_unlearned_edges = 0
            train_edge_indices = np.logical_and(np.isin(edge_index[0], self.data.train_indices),
                                    np.isin(edge_index[1], self.data.train_indices))

            # 过滤出满足 edge_index[0] < edge_index[1] 的边，构成单向边
            directed_train_edges = np.logical_and(train_edge_indices, edge_index[0] < edge_index[1])

            # 获取这些单向边的索引
            train_edges = np.where(directed_train_edges)[0]

            # 计算需要删除的边的数量 (5%)
            num_edges_to_remove = int(len(train_edges) * 0.05)

            # 随机选择 5% 的单向边
            edges_to_remove = np.random.choice(train_edges, size=num_edges_to_remove, replace=False)

            # 找到与这些单向边对应的反向边
            reverse_edges_to_remove = np.where(
                (edge_index[0][None, :] == edge_index[1][edges_to_remove][:, None]) &
                (edge_index[1][None, :] == edge_index[0][edges_to_remove][:, None])
            )[1]

            # 合并正向边和反向边的索引
            gis_keys_graph = {}
            gis_keys_graph[0] = {}
            self.part_set = []
            all_edges_to_remove = np.concatenate([edges_to_remove, reverse_edges_to_remove])
            for part_id, edge_index in self.p1_saved.shards_edges.items():
                loadname = self.p1_saved.DPATH.replace('partid', 'part' + str(part_id))
                sub_graph = torch.load(loadname)
                gis_keys_graph[0][part_id] = sub_graph

                
                # 1. 获取该 shard 中节点的原始索引
                shard_original_nodes = self.p1_saved.shards_ids[part_id]
                
                # 2. 建立 shard 中节点索引到原始节点索引的映射
                shard_to_original_map = {new_id: orig_id for new_id, orig_id in enumerate(shard_original_nodes)}
                
                # 3. 反向映射 shard 的 edge_index 到原始图的 edge_index
                original_edge_index = torch.clone(edge_index)
                original_edge_index[0] = torch.tensor([shard_to_original_map[node] for node in edge_index[0].tolist()])
                original_edge_index[1] = torch.tensor([shard_to_original_map[node] for node in edge_index[1].tolist()])

                # 4. 构建删除边的 mask，找到原始图中的边在该 shard 中的匹配位置
                mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
                for del_idx in all_edges_to_remove:
                    # 原始图中需要删除的边
                    del_edge = self.data.edge_index[:, del_idx].unsqueeze(1)
                    
                    # 找到在该 shard 中对应的边（需要映射回原始图）
                    delete_edge = (original_edge_index == del_edge).all(dim=0)
                    mask = mask | delete_edge  # 更新mask，标记删除的边
                if not mask.sum() == edge_index.shape[1]:
                    self.part_set.append(part_id)
                # 5. 使用 mask 更新 shard 的边
                edge_index_del = edge_index[:,mask]

                new_mask = torch.ones(gis_keys_graph[0][part_id].edge_index.shape[1], dtype=torch.bool)
                for i in range(edge_index_del.shape[1]):
                    edge_to_check = edge_index_del[:, i].unsqueeze(1)  # 取出当前边

                    # 检查 edge_to_check 是否在 shard 的 edge_index 中
                    matching_edges = (gis_keys_graph[0][part_id].edge_index == edge_to_check).all(dim=0)
                    
                    # 找到所有匹配的边的索引
                    matching_indices = matching_edges.nonzero(as_tuple=True)[0]

                    # 将对应的 new_mask 中的这些索引设置为 False
                    new_mask[matching_indices] = False
                gis_keys_graph[0][part_id].edge_index = gis_keys_graph[0][part_id].edge_index[:, new_mask]
                
            self.part_set = set(self.part_set)
            self.gis_keys_graph_update = copy.deepcopy(gis_keys_graph)

            
        elif self.args["unlearn_task"] == "node":
            id_gi_part_innerid = []
            ids = list(range(self.data.x.size(0)))
            gis = [0] * len(ids)
            inday_ids = range(len(ids))
            part_ids = self.p1_saved.ids_shards.values()
            id_gi_part_innerid += list(zip(ids, gis, part_ids,inday_ids))
            torch.save(id_gi_part_innerid, self.p1_saved.DPATH.split('partid')[0] + 'id_gi_part_innerid.pt')


            average_countpart = []
            # train_labeled_idx = list(range(len(self.data.train_indices)))
            # self.unlearning_id = random.sample(train_labeled_idx, k=int(self.unlearning_number))
            path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
            # breakpoint()
            self.unlearning_id = np.loadtxt(path_un, dtype=int)

            # 假设 self.data.train_indices 和 self.unlearning_id 是列表或张量
            train_indices = list(self.data.train_indices)
            unlearning_id = list(self.unlearning_id)

            # 对 self.data.train_indices 进行排序，获取排序后的索引
            # sorted_train_indices = sorted(train_indices)

            # # 获取 self.unlearning_id 中每个元素在 sorted_train_indices 中的排名
            # self.unlearning_id = [sorted_train_indices.index(id) for id in unlearning_id]

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

            
            gis_keys_graph = {}
            gis_keys_graph[0] = {}
            for part_id in range(self.args['num_shards']):
                loadname = self.p1_saved.DPATH.replace('partid', 'part' + str(part_id))
                sub_graph = torch.load(loadname)
                gis_keys_graph[0][part_id] = sub_graph

            self.gis_keys_graph_update = copy.deepcopy(gis_keys_graph)
            self.part_set = []
            time_sum = 0
            for gi in gi_part_uid.keys():
                for part in gi_part_uid[gi].keys():
                    self.part_set.append(part)
                    #self.logger.info("part  {}, train number: {}".format(part, self.gis_keys_graph_update[gi][part].train_mask.sum().item()))
                    delete_subindex = []
                    for id in gi_part_uid[gi][part]:
                        newindex = self.index_to_subindex(part_ids,id)
                        # self.gis_keys_graph_update[gi][part].y = tensor = torch.cat((self.gis_keys_graph_update[gi][part].y[:newindex], self.gis_keys_graph_update[gi][part].y[newindex+1:]))
                        delete_subindex.append(self.find_uid_2(newindex,self.gis_keys_graph_update[gi][part].train_mask))
  
                    reserve_mask = torch.tensor([False if id in delete_subindex else True for id in self.gis_keys_graph_update[gi][part].uids])
                    sub_graph = self.gis_keys_graph_update[gi][part].subgraph(reserve_mask)
                    # if  sub_graph.x.size(0)!=sub_graph.y.size(0):
                    #     self.logger.info("not match!")
                    sub_graph.uids = list(compress(sub_graph.uids, reserve_mask.tolist()))
                    self.gis_keys_graph_update[gi][part] = sub_graph
                    #self.logger.info("part  {}, after delete train number: {}".format(part, sub_graph.train_mask.sum().item()))
            self.part_set = set(self.part_set)
            # self.logger.info("update_time:{}".format(time_sum/10))
        elif self.args["unlearn_task"] == "feature":
            id_gi_part_innerid = []
            ids = list(range(self.data.x.size(0)))
            gis = [0] * len(ids)
            inday_ids = range(len(ids))
            part_ids = self.p1_saved.ids_shards.values()
            id_gi_part_innerid += list(zip(ids, gis, part_ids,inday_ids))
            torch.save(id_gi_part_innerid, self.p1_saved.DPATH.split('partid')[0] + 'id_gi_part_innerid.pt')
            gis_keys_graph = {}
            gis_keys_graph[0] = {}
            self.part_set = []
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
            self.gis_keys_graph_update = copy.deepcopy(gis_keys_graph)

            for part_id, edge_index in self.p1_saved.shards_ids.items():
                loadname = self.p1_saved.DPATH.replace('partid', 'part' + str(part_id))
                sub_graph = torch.load(loadname)
                gis_keys_graph[0][part_id] = sub_graph

            for gi in gi_part_uid.keys():
                
                for part in gi_part_uid[gi].keys():
                    self.part_set.append(part)
                    #self.logger.info("part  {}, train number: {}".format(part, self.gis_keys_graph_update[gi][part].train_mask.sum().item()))
                    delete_subindex = []
                    for id in gi_part_uid[gi][part]:
                        newindex = self.index_to_subindex(part_ids,id)
                        self.gis_keys_graph_update[0][part].x[newindex] = torch.zeros_like(self.gis_keys_graph_update[0][part].x[newindex])
            
                

    # def unlearn(self):
    #     """
    #     Unlearns the model by updating the shard and retraining the model on the updated graph data.
    #     """
    #     self.update_shard()
    #     self.G_nx0 = []
    #     subouts=[]
    #     for part in self.part_set:
    #         #self.logger.info("part  {} ".format(part))
    #         time_sum = 0
    #         for gi in self.gis_keys_graph_update.keys():
    #             self.G_nx0.append(self.gis_keys_graph_update[gi][part].edge_index.to('cpu'))

    #         submodel = self.target_model
    #         breakpoint()
    #         submodel.model.to(self.device)
    #         optimizer = torch.optim.Adam(
    #             submodel.model.parameters(), lr=0.01, weight_decay=1e-5)
    #         #self.logger.info("self.gis_keys_graph_update {}".format(self.gis_keys_graph_update[0][part]))
    #         if self.args["downstream_task"] == "node":
    #             for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
                    
    #                 submodel.model.train()
    #                 sub_graph = self.gis_keys_graph_update[0][part]
    #                 sub_graph = sub_graph.to(self.device)
    #                 labels = sub_graph.y
    #                 optimizer.zero_grad()
    #                 if self.args["base_model"] == "SIGN":
    #                     sub_graph = SIGN(self.args["GNN_layer"])(sub_graph)
    #                     sub_graph.xs = [sub_graph.x] + [sub_graph[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
    #                     sub_graph.xs = torch.stack(sub_graph.xs).to('cuda')
    #                     sub_graph.xs = sub_graph.xs.transpose(0,1)
    #                     out = submodel.model(sub_graph.xs)
    #                 else:
    #                     out = submodel.model(sub_graph.x, sub_graph.edge_index)
    #                 out = out[sub_graph.train_mask]
    #                 #self.logger.info("labels  {},  out  {}".format(labels.shape,out.shape))
    #                 loss = F.cross_entropy(out, labels[sub_graph.train_mask[:labels.size(0)]])
    #                 self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss))

    #                 loss.backward()
    #                 optimizer.step()
    #         elif self.args["downstream_task"] == "edge":
    #             for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
    #                 submodel.model.train()
    #                 sub_graph = self.gis_keys_graph_update[0][part]
    #                 sub_graph = sub_graph.to(self.device)
    #                 # print(sub_graph)
    #                 optimizer.zero_grad()
    #                 if self.args["base_model"] == "SIGN":
    #                     sub_graph = SIGN(self.args["GNN_layer"])(sub_graph)
    #                     sub_graph.xs = [sub_graph.x] + [sub_graph[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
    #                     sub_graph.xs = torch.stack(sub_graph.xs).to('cuda')
    #                     sub_graph.xs = sub_graph.xs.transpose(0,1)
    #                     out = submodel.model(sub_graph.xs)
    #                 else:
    #                     out = submodel.model(sub_graph.x, sub_graph.train_edge_index)
                    
    #                 neg_edge_index = negative_sampling(
    #                     edge_index=sub_graph.edge_index,num_nodes=sub_graph.num_nodes,
    #                     num_neg_samples=sub_graph.train_edge_index.size(1),force_undirected=True)
    #                 edge_pred = submodel.decode(out,sub_graph.train_edge_index,neg_edge_index)
    #                 pos_edge_labels = torch.ones(sub_graph.train_edge_index.size(1),dtype=torch.float32)
    #                 neg_edge_labels = torch.zeros(neg_edge_index.size(1),dtype=torch.float32)
    #                 edge_labels = torch.cat((pos_edge_labels,neg_edge_labels),dim=-1)
    #                 edge_labels = edge_labels.to(self.device)
    #                 # print(edge_labels,edge_pred)
    #                 loss = F.binary_cross_entropy_with_logits(edge_pred,edge_labels)
    #                 # print(loss)
    #                 self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss))
    #                 loss.backward()
    #                 optimizer.step()
                    
    #         # self.logger.info("training_time:{}".format(time_sum/self.args['num_epochs']))
    #         submodel.model.eval()
    #         # subouts = []
    #         if self.args["base_model"] == "SIGN":
    #             x_input = self.data.xs[self.data.test_indices].to(self.device)
    #             subout = torch.softmax(submodel.model(x_input), dim=1)
    #         else:
    #             x = self.data.x.to(self.device)
    #             edge_index = self.data.edge_index.to(self.device)
    #             subout = torch.softmax(submodel.model(x, edge_index), dim=1)

    #         subouts.append(subout.detach().cpu())



            
    #         submodel.model.to('cpu')
    #         sub_graph.to('cpu')
    #         savemodeln = self.p1_saved.MPATH.replace('/partid', '_copy/partid').replace('partid',
    #                                                                                           'part' + str(part))
    #         os.makedirs(os.path.dirname(savemodeln), exist_ok=True)
    #         torch.save(submodel.model.state_dict(), savemodeln)
    #         print(f"in unlearn function, saving at {savemodeln}")

    #     self.aggregate_shard_model(True)
    #     time_sum+= time.time()-self.start_time
    #     self.avg_unlearning_time[self.run] = time_sum



    def unlearn(self):
        """
        Unlearns the model by updating the shard and retraining the model on the updated graph data.
        """
        self.update_shard()
        self.G_nx0 = []
        subouts = []
        subg_kfea = []

        for part in self.part_set:
            time_sum = 0
            for gi in self.gis_keys_graph_update.keys():
                self.G_nx0.append(self.gis_keys_graph_update[gi][part].edge_index.to('cpu'))

            submodel = self.target_model
            submodel.model.to(self.device)
            optimizer = torch.optim.Adam(
                submodel.model.parameters(), lr=0.01, weight_decay=1e-5)

            if self.args["downstream_task"] == "node":
                # breakpoint()
                for epoch in tqdm(range((self.args['num_epochs'])*10), desc="Training", unit="epoch"):
                    submodel.model.train()
                    sub_graph = self.gis_keys_graph_update[0][part]
                    sub_graph = sub_graph.to(self.device)
                    labels = sub_graph.y
                    optimizer.zero_grad()

                    if self.args["base_model"] == "SIGN":
                        sub_graph = SIGN(self.args["GNN_layer"])(sub_graph)
                        sub_graph.xs = [sub_graph.x] + [sub_graph[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                        sub_graph.xs = torch.stack(sub_graph.xs).to('cuda')
                        sub_graph.xs = sub_graph.xs.transpose(0,1)
                        out = submodel.model(sub_graph.xs)
                    else:
                        out = submodel.model(sub_graph.x, sub_graph.edge_index)
                    # breakpoint()
                    out = out[sub_graph.train_mask]
                    loss = F.cross_entropy(out, labels[sub_graph.train_mask[:labels.size(0)]])
                    self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss))
                    loss.backward()
                    optimizer.step()

            elif self.args["downstream_task"] == "edge":
                for epoch in tqdm(range(self.args['num_epochs']), desc="Training", unit="epoch"):
                    submodel.model.train()
                    sub_graph = self.gis_keys_graph_update[0][part]
                    sub_graph = sub_graph.to(self.device)
                    optimizer.zero_grad()

                    if self.args["base_model"] == "SIGN":
                        sub_graph = SIGN(self.args["GNN_layer"])(sub_graph)
                        sub_graph.xs = [sub_graph.x] + [sub_graph[f'x{i}'] for i in range(1, self.args["GNN_layer"] + 1)]
                        sub_graph.xs = torch.stack(sub_graph.xs).to('cuda')
                        sub_graph.xs = sub_graph.xs.transpose(0,1)
                        out = submodel.model(sub_graph.xs)
                    else:
                        out = submodel.model(sub_graph.x, sub_graph.train_edge_index)

                    neg_edge_index = negative_sampling(
                        edge_index=sub_graph.edge_index, num_nodes=sub_graph.num_nodes,
                        num_neg_samples=sub_graph.train_edge_index.size(1), force_undirected=True)
                    edge_pred = submodel.decode(out, sub_graph.train_edge_index, neg_edge_index)

                    pos_edge_labels = torch.ones(sub_graph.train_edge_index.size(1), dtype=torch.float32)
                    neg_edge_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)
                    edge_labels = torch.cat((pos_edge_labels, neg_edge_labels), dim=-1).to(self.device)

                    loss = F.binary_cross_entropy_with_logits(edge_pred, edge_labels)
                    self.logger.info('Epoch: {:03d} | Loss: {:.4f}'.format(epoch + 1, loss))
                    loss.backward()
                    optimizer.step()

            # === Save retrained model ===
            submodel.model.eval()
            if self.args["base_model"] == "SIGN":
                x_input = self.data.xs[self.data.test_indices].to(self.device)
                subout = torch.softmax(submodel.model(x_input), dim=1)
            else:
                x = self.data.x.to(self.device)
                edge_index = self.data.edge_index.to(self.device)
                subout = torch.softmax(submodel.model(x, edge_index), dim=1)

            subouts.append(subout.detach().cpu())

            savemodeln = self.p1_saved.MPATH.replace('/partid', '_copy/partid').replace('partid', f'part{part}')
            os.makedirs(os.path.dirname(savemodeln), exist_ok=True)
            torch.save(submodel.model.state_dict(), savemodeln)
            print(f"in unlearn function, saving at {savemodeln}")

            # === Load feature for the part (for kernel sim) ===
            loadname = self.p1_saved.FPATH.replace('partid', f'part{part}')
            print(f"currently loading {loadname}")
            subg_kfea += torch.load(loadname)

        # === Aggregate all submodel predictions ===
        subgt_kfea = self.pm_kernel.parse_input([self.data.edge_index.cpu()])
        suboutTensor = torch.stack(subouts)
        weights = self.pm_kernel.kernel_similarity(subgt_kfea, subg_kfea)
        weights = torch.tensor(weights).to(torch.float32).reshape(len(self.p1_saved.shards_ids.keys()), -1).mean(dim=-1)
        # saveweightn = self.p1_saved.MPATH.replace('partid/submodels/', '').replace('model_partid', f'_weight')
        # os.makedirs(os.path.dirname(saveweightn), exist_ok=True)
        # torch.save(weights, saveweightn)

        weights = F.softmax(weights, dim=0)
        weighted_pred = torch.tensordot(suboutTensor, weights, dims=([0], [0]))
        aggregated_preds = weighted_pred.argmax(dim=1).cpu().numpy()

        test_labels = self.data.y[self.data.test_mask].cpu().numpy()
        from sklearn.metrics import accuracy_score
        test_acc = accuracy_score(test_labels, aggregated_preds[self.data.test_mask])
        print(f"[Unlearned Aggregated Model] Test Accuracy: {test_acc:.4f}")

        # === Wrap up ===
        self.aggregate_shard_model(True)
        time_sum += time.time() - self.start_time
        self.avg_unlearning_time[self.run] = time_sum



    def attack_unlearning(self):
        """
        Executes the unlearning attack process for a node classification task.
        """
        if self.args["unlearn_task"] == "node":
            self.G_nx0 = self.pm_kernel.parse_input(self.G_nx0)
            self.method = self.args["GUIDE_methods"]
            self.positive0, self.negative0 = {}, {}
            self.positive0[self.method], self.negative0[self.method] = [], []
            submodellist = []
            part_set_all = range(self.args['num_shards'])
            for part in part_set_all:
                submodel = self.target_model
                loadname = self.p1_saved.MPATH.replace('partid', 'part' + str(part))

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
                    subout_train_daily = torch.softmax(submodellist[part](self.data.x.to(self.device), self.data.edge_index.to(self.device)), dim=1)
                subout_train.append(subout_train_daily)

                subout_train = torch.concat(subout_train)
                subout_train_ = subout_train[self.unlearning_id]
                subouts_train.append(subout_train_.detach())

                subout_test = []
                if self.args["base_model"] == "SIGN":
                    subout_test_daily = torch.softmax(submodellist[part](self.data.xs),dim=1)[self.data.test_mask]
                else:
                    subout_test_daily = torch.softmax(submodellist[part](self.data.x.to(self.device), self.data.edge_index.to(self.device)), dim=1)[self.data.test_mask]
                subout_test.append(subout_test_daily)

                subout_test = torch.concat(subout_test)
                subout_tests.append(subout_test.detach())

            self.suboutTensor = torch.stack(subouts_train)
            self.suboutTensorn = torch.stack(subout_tests)

            weights_gi = torch.load(
                self.p1_saved.MPATH.replace('partid/submodels/', '').replace('model_partid', '_weight'))

            summed = torch.tensordot(
                self.suboutTensor, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
            self.positive0[self.method].append(summed.cpu())

            summed = torch.tensordot(
                self.suboutTensorn, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
            self.negative0[self.method].append(summed.cpu())

            tmp = time.strftime("%Y%m%d-%H%M%S")
            savename = self.p1_saved.MPATH.split('partid')[
                        0] + f'evaluation_attack_partition_{self.method}_' + self.args[
                        "dataset_name"] + '_unlearning_idrnd'
            savename += ''.join(tmp)
            savename += ''.join('.pt')
            torch.save(self.unlearning_id, savename)

            savename = self.p1_saved.MPATH.split('partid')[
                        0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_positive0'
            savename += ''.join(tmp)
            savename += ''.join('.pt')
            torch.save(self.positive0, savename)
            savename = self.p1_saved.MPATH.split('partid')[
                        0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_negative0'
            savename += ''.join(tmp)
            savename += ''.join('.pt')
            torch.save(self.negative0, savename)

            self.positive1, self.negative1 = {}, {}
            self.positive1[self.method], self.negative1[self.method] = [], []

            submodellist = []
            part_set_all = range(self.args['num_shards'])
            for part in part_set_all:
                submodel = self.target_model

                loadname = self.p1_saved.MPATH.replace('partid', 'part' + str(part)).replace(self.args["dataset_name"], self.args["dataset_name"]+'_copy')

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
                    subout_train_daily = torch.softmax(submodellist[part](self.data.x.to(self.device), self.data.edge_index.to(self.device)), dim=1)
                subout_train.append(subout_train_daily)

                subout_train = torch.concat(subout_train)
                subout_train_ = subout_train[self.unlearning_id]
                subouts_train.append(subout_train_.detach())

                subout_test = []
                if self.args["base_model"] == "SIGN":
                    subout_test_daily = torch.softmax(submodellist[part](self.data.xs),dim=1)[self.data.test_mask]
                else:
                    subout_test_daily = torch.softmax(submodellist[part](self.data.x.to(self.device), self.data.edge_index.to(self.device)), dim=1)[self.data.test_mask]
                subout_test.append(subout_test_daily)

                subout_test = torch.concat(subout_test)
                subout_tests.append(subout_test.detach())

            self.suboutTensor = torch.stack(subouts_train)
            self.suboutTensorn = torch.stack(subout_tests)

            # weights_gi = torch.tensor([1.0/args['shards_number']
            #                         for _ in range(args['shards_number'])])
            weights_gi = torch.load(
                self.p1_saved.MPATH.replace('partid/submodels/', '').replace('model_partid', '_weight'))

            summed = torch.tensordot(
                self.suboutTensor, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
            self.positive1[self.method].append(summed.cpu())

            summed = torch.tensordot(
                self.suboutTensorn, torch.softmax(weights_gi.cuda(), dim=0), dims=([0], [0]))
            self.negative1[self.method].append(summed.cpu())

            savename = self.p1_saved.MPATH.split('partid')[0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_positive1'
            savename += ''.join(tmp)
            savename += ''.join('.pt')
            torch.save(self.positive1, savename)
            savename = self.p1_saved.MPATH.split('partid')[0] + f'evaluation_attack_partition_{self.method}_' + self.args["dataset_name"] + '_negative1'
            savename += ''.join(tmp)
            savename += ''.join('.pt')
            torch.save(self.negative1, savename)
        if self.args["unlearn_task"] == "node" and self.args['downstream_task'] == "node":
            self.mia_attack()

    def mia_attack(self):
        """
        Perform a membership inference attack (MIA) on the model.
        This function simulates a membership inference attack by comparing the 
        posterior probabilities of positive and negative samples. It calculates 
        the AUC (Area Under the Curve) score to evaluate the attack's effectiveness.
        """
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

        self.logger.info(
            f"Dataset: {self.args['dataset_name']} | Method: {self.method} | Pos:Neg Average Attack_Model_B AUC: "
            f"Mean: {np.mean(attack_auc_b):.4f} | Variance: {np.var(attack_auc_b):.4f}"
        )
        
        self.average_auc[self.run] = np.array(attack_auc_b).mean()

    def calculate_distance(self,data0, data1, distance='l2_norm'):
        """
        Calculate the distance between two data points using the specified distance metric.
        """
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
        """
        Converts a global index to a subindex within a list of partition IDs.
        This function takes a list of partition IDs and a global index, and returns the subindex of the element at the given index within its partition. The subindex is determined by counting the occurrences of the partition ID up to the given index.
        """
        part_ids = list(part_ids)
        subindex = part_ids[:index].count(part_ids[index])
        return subindex

    def reverse_y(self,y,train_mask):
        """
        Reverses the labels in `y` based on the `train_mask`.
        This function takes a tensor `y` containing labels and a boolean mask `train_mask` indicating 
        which elements in `y` are part of the training set. It creates a new tensor `ori_y` where 
        the labels from `y` are placed at the positions indicated by `train_mask`, and all other 
        positions are filled with -1.
        """
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
        """
        Finds the unique identifier (uid) corresponding to a given id within a training mask.
        """
        for uid in range(len(train_mask)):
            if sum(train_mask[:uid]) == id:
                return uid
            
    def find_uid_2(self,id,train_mask):
        return id
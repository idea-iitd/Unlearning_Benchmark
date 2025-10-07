import copy
import math
import os
import pickle
import numpy as np
import torch
import time
import torch.nn.functional as F
from ogb.linkproppred import PygLinkPropPredDataset
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.datasets import CitationFull, Coauthor, Flickr
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.utils import train_test_split_edges, k_hop_subgraph, negative_sampling, to_undirected, is_undirected, to_networkx
from tqdm import tqdm
from model.base_gnn.Convs import S2GConv,SGConv,GCNConv
from task.node_classification import NodeClassifier
from utils.dataset_utils import load_saved_data
from utils.utils import negative_sampling_kg, plot_auc
from utils.utils import to_directed
from config import root_path,unlearning_path,unlearning_edge_path
from config import BLUE_COLOR,RESET_COLOR
from task import get_trainer
from task.edge_prediction import EdgePredictor
from pipeline.Learning_based_pipeline import Learning_based_pipeline

class gnndelete(Learning_based_pipeline):
    """
    GNNDelete Class, a pipeline for performing unlearning operations on Graph Neural Networks. It extends the `Learning_based_pipeline` and provides methods to delete nodes, edges, or features from the training data, retrain the model accordingly, and evaluate the impact of these deletions. 
    This class is designed to support experiments involving the removal of specific data points from a trained GNN model while maintaining model performance and assessing vulnerabilities to membership inference attacks.
    
    Class Attributes:
        args (dict): Configuration parameters and arguments for the GNNDelete.

        logger (Logger): Logger for tracking and recording the pipeline's operations and metrics.

        model_zoo (ModelZoo): Collection of models available for training and evaluation.
    """
    def __init__(self,args,logger,model_zoo):
        super().__init__(args,logger,model_zoo)
        self.args= args
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_runs = self.args["num_runs"]
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_time = np.zeros(num_runs)
        self.run = 0
        self.args["checkpoint_dir"] = root_path + '/data/GNNDelete/checkpoint_node'
        self.args['in_dim'] = self.data.x.shape[1]
        self.args['out_dim'] = self.data.num_classes
        self.args['unlearning_model'] = 'original'
    
    def determine_target_model(self):
        """
        Determines and initializes the target model based on the specified base model.
        This method selects the appropriate propagation method (SGC, S2GC, or SIGN) based on the
        'base_model' argument, computes preprocessed features, and sets up the target model
        for unlearning by configuring the unlearning trainer and loading the model from the
        model zoo.
        """
        if self.args["base_model"] == "SGC":
            propagation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
            features_pre = propagation.forward_SGU(self.data.x,self.data.edge_index)
            self.data.features_pre = features_pre.cuda()
        elif self.args["base_model"] == "S2GC":
            propagation = S2GConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
            features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
            self.data.features_pre = features_pre.cuda()
        elif self.args["base_model"] == "SIGN":
            self.data.features_pre = self.data.xs
        self.args["unlearn_trainer"] = "GNNDeleteTrainer"
        self.data = self.data.to(self.device)
        self.args['unlearning_model'] = 'original'
        model = self.model_zoo.get_model(num_nodes=self.data.num_nodes).to(self.device)
        self.model_zoo.model = model
        self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)

    def train_original_model(self):
        """
        Trains the original target model and get the original soft labels for the dataset.
        """
        self.target_model.train(save=False)
        self.data = self.data.to(self.device)
        if self.args["poison"] and self.args["unlearn_task"]=="edge":
            self.poison_f1[self.run] = self.target_model.evaluate()
        if  self.args["base_model"] in ["SIGN","SGC","S2GC"]:
            self.original_softlabels = F.softmax(self.target_model.model(self.data.features_pre), dim=1)
        else:
            self.original_softlabels = F.softmax(self.target_model.model(self.data.x, self.data.edge_index), dim=1)
        self.args["unlearning_model"] = 'gnndelete_nodeemb'
        
    def unlearn(self):
        """
        Executes the unlearning process based on the specified task.
        This method determines the type of unlearning operation to perform by checking the 
        'unlearn_task' argument. Depending on its value, it delegates the operation to the 
        """
        if self.args["unlearn_task"] == "node":
            self.delete_node()
        elif self.args["unlearn_task"] == "edge":
            self.delete_edge()
        elif self.args["unlearn_task"] == "feature":
            self.delete_feature()
    
    
    # def run_exp(self):
        # if self.args["base_model"] == "SGC":
        #     propagation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
        #     features_pre = propagation.forward_SGU(self.data.x,self.data.edge_index)
        #     self.data.features_pre = features_pre.cuda()
        # elif self.args["base_model"] == "S2GC":
        #     propagation = S2GConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
        #     features_pre = propagation.forward_SGU(self.data.x, self.data.edge_index)
        #     self.data.features_pre = features_pre.cuda()
        # elif self.args["base_model"] == "SIGN":
        #     self.data.features_pre = self.data.xs
    #     for self.run in range(self.args["num_runs"]):
    #         self.args["unlearn_trainer"] = "GNNDeleteTrainer"
    #         self.train_node(self.data,retrain=True)
    #         # self.args["unlearn_trainer"] = "GNNDeleteTrainer"
    #         if self.args["unlearn_task"] == "node":
    #             self.delete_node()
    #         elif self.args["unlearn_task"] == "edge":
    #             self.delete_edge()
    #         elif self.args["unlearn_task"] == "feature":
    #             self.delete_feature()

    #     self.logger.info(
    #         "{}Performance Metrics:\n"
    #         " - Average F1 Score: {:.4f} ± {:.4f}\n"
    #         " - Average AUC Score: {:.4f} ± {:.4f}\n"
    #         " - Average Training Time: {:.4f} ± {:.4f} seconds{}\n".format(
    #             BLUE_COLOR,
    #             np.mean(self.average_f1), np.std(self.average_f1),
    #             np.mean(self.average_auc), np.std(self.average_auc),
    #             np.mean(self.avg_time), np.std(self.avg_time),
    #             RESET_COLOR
    #         )
    #     )

    def prepare_dataset(self):
        df_size = [i / 100 for i in range(10)] + [i / 10 for i in range(10)] + [i for i in range(10)]  # Df_size in percentage
        seeds = [42, 21, 13, 87, 100]
        self.graph = to_networkx(self.data)

        # Get two hop degree for all nodes
        node_to_neighbors = {}
        for n in tqdm(self.graph.nodes(), desc='Two hop neighbors'):
            neighbor_1 = set(self.graph.neighbors(n))
            neighbor_2 = sum([list(self.graph.neighbors(i)) for i in neighbor_1], [])
            neighbor_2 = set(neighbor_2)
            neighbor = neighbor_1 | neighbor_2

            node_to_neighbors[n] = neighbor

        two_hop_degree = []
        row, col = self.data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        for r, c in tqdm(zip(row, col), total=len(row)):
            neighbor_row = node_to_neighbors[r.item()]
            neighbor_col = node_to_neighbors[c.item()]
            neighbor = neighbor_row | neighbor_col

            num = len(neighbor)

            two_hop_degree.append(num)

        two_hop_degree = torch.tensor(two_hop_degree)

        for s in seeds:
            seed_everything(s)

            # D
            data = self.data
            if self.args["dataset_name"] == 'ogbl':
                data = self.train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.1, two_hop_degree=two_hop_degree)
            else:
                data = self.train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.2)
            print(s, data)

            # with open(root_path + "/data/GNNDelete/" + self.args["dataset_name"]+ "/" + f'd_{s}.pkl', 'wb') as f:
            #     pickle.dump((self.dataset, data), f)

            _, local_edges, _, mask = k_hop_subgraph(
                data.test_pos_edge_index.flatten().unique(),
                2,
                data.train_pos_edge_index,
                num_nodes=self.data.num_nodes)
            distant_edges = data.train_pos_edge_index[:, ~mask]
            print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

            in_mask = mask
            out_mask = ~mask

            # df_in_mask = torch.zeros_like(mask)
            # df_out_mask = torch.zeros_like(mask)

            # df_in_all_idx = in_mask.nonzero().squeeze()
            # df_out_all_idx = out_mask.nonzero().squeeze()
            # df_in_selected_idx = df_in_all_idx[torch.randperm(df_in_all_idx.shape[0])[:df_size]]
            # df_out_selected_idx = df_out_all_idx[torch.randperm(df_out_all_idx.shape[0])[:df_size]]

            # df_in_mask[df_in_selected_idx] = True
            # df_out_mask[df_out_selected_idx] = True

            # assert (in_mask & out_mask).sum() == 0
            # assert (df_in_mask & df_out_mask).sum() == 0

            # local_edges = set()
            # for i in range(data.test_pos_edge_index.shape[1]):
            #     edge = data.test_pos_edge_index[:, i].tolist()
            #     subgraph = get_enclosing_subgraph(graph, edge)
            #     local_edges = local_edges | set(subgraph[2])

            # distant_edges = graph.edges() - local_edges

            # print('aaaaaaa', len(local_edges), len(distant_edges))
            # local_edges = torch.tensor(sorted(list([i for i in local_edges if i[0] < i[1]])))
            # distant_edges = torch.tensor(sorted(list([i for i in distant_edges if i[0] < i[1]])))

            # df_in = torch.randperm(local_edges.shape[1])[:df_size]
            # df_out = torch.randperm(distant_edges.shape[1])[:df_size]

            # df_in = local_edges[:, df_in]
            # df_out = distant_edges[:, df_out]

            # df_in_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
            # df_out_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)

            # for row in df_in:
            #     i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
            #     df_in_mask[i] = True

            # for row in df_out:
            #     i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
            #     df_out_mask[i] = True

            save_path = os.path.join("./data/GNNDelete", self.args["dataset_name"])
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                {'out': out_mask, 'in': in_mask},
                os.path.join(save_path, f'df_{s}.pt')
            )

    def train_test_split_edges_no_neg_adj_mask(self,data, val_ratio: float = 0.2, test_ratio: float = 0.2,
                                               two_hop_degree=None, kg=False):
        '''Avoid adding neg_adj_mask'''
        num_nodes = data.num_nodes
        row, col = data.edge_index
        edge_attr = data.edge_attr

        if not kg:
            # Return upper triangular portion.
            mask = row < col
            row, col = row[mask], col[mask]

            if edge_attr is not None:
                edge_attr = edge_attr[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        if two_hop_degree is not None:  # Use low degree edges for test sets
            low_degree_mask = two_hop_degree < 50

            low = low_degree_mask.nonzero().squeeze()
            high = (~low_degree_mask).nonzero().squeeze()

            low = low[torch.randperm(low.size(0))]
            high = high[torch.randperm(high.size(0))]

            perm = torch.cat([low, high])

        else:
            perm = torch.randperm(row.size(0))

        row = row[perm]
        col = col[perm]

        # Train
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[r] = train_mask[c] = True
        data.train_mask = train_mask

        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        if edge_attr is not None:
            out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
            data.train_pos_edge_index, data.train_pos_edge_attr = out
        else:
            data.train_pos_edge_index = data.train_pos_edge_index
            # data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        assert not is_undirected(data.train_pos_edge_index)

        # Test
        r, c = row[:n_t], col[:n_t]
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[r] = test_mask[c] = True
        data.test_mask = test_mask
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        neg_edge_index = negative_sampling(
            edge_index=data.test_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.test_pos_edge_index.shape[1])

        data.test_neg_edge_index = neg_edge_index

        # Valid
        r, c = row[n_t:n_t + n_v], col[n_t:n_t + n_v]
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[r] = val_mask[c] = True
        data.val_mask = val_mask
        data.val_pos_edge_index = torch.stack([r, c], dim=0)

        neg_edge_index = negative_sampling(
            edge_index=data.val_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.val_pos_edge_index.shape[1])

        data.val_neg_edge_index = neg_edge_index

        return data

    # def train_node(self,data = None,retrain=False):
    #     self.args["checkpoint_dir"] = root_path + '/data/GNNDelete/checkpoint_node'
    #     data = data.to(self.device)
    #     self.args['in_dim'] = data.x.shape[1]
    #     self.args['out_dim'] = self.data.num_classes
    #     self.args['unlearning_model'] = 'original'
    #     model = self.model_zoo.get_model(num_nodes=data.num_nodes).to(self.device)
    #     self.model_zoo.model = model
    #     #9.20
    #     # if self.args["downstream_task"]=="node":
    #     #     self.NodeClassifier = NodeClassifier(self.args, data, self.model_zoo, self.logger)
    #     # elif self.args["downstream_task"]=="edge":
    #     #     self.NodeClassifier = EdgePredictor(self.args, data, self.model_zoo, self.logger)
    #     self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)

    #     # if retrain:
    #     #     self.NodeClassifier.train_model(retrain=retrain)
    #     # else:
    #     #     self.NodeClassifier.train_model()
        
    #     # if os.path.exists(os.path.join(root_path + "/data/model/node_level/",self.args["dataset_name"],self.args["base_model"])):
    #     #     model_ckpt = torch.load(os.path.join(root_path + "/data/model/node_level/",self.args["dataset_name"],self.args["base_model"]),
    #     #                             map_location=self.device)
    #     #     # model.load_state_dict(model_ckpt['model_state'], strict=False)
    #     #     self.target_model.model.load_state_dict(model_ckpt, strict=False)
    #     #     self.target_model.model.to(self.device)
            
    #     # else:
    #     self.target_model.train(save=True)
        
    #     # self.logger.info("original:{}".format(F1))
    #     if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
    #         self.original_softlabels = F.softmax(self.target_model.model(self.data.features_pre), dim=1)
    #     else:
    #         self.original_softlabels = F.softmax(self.target_model.model(self.data.x, self.data.edge_index), dim=1)
    #     self.args["unlearning_model"] = 'gnndelete_nodeemb'





    def delete_node(self):
        """
        Deletes specified nodes from the graph neural network and updates the model accordingly.
        This method performs node unlearning by removing designated nodes and their associated edges from the training dataset. 
        It updates the model's masks to exclude these nodes, handles the creation of subgraphs, and manages the retraining or adjustment of the GNN model based on the selected unlearning strategy. 
        Additionally, it evaluates the updated model's performance and conducts membership inference attacks to assess the effectiveness of the unlearning process.
        """
        self.args["checkpoint_dir"] = root_path + '/data/GNNDelete/checkpoint_node'
        original_path = os.path.join(self.args["checkpoint_dir"],self.args["dataset_name"],self.args["base_model"],'original',
                                                          '-'.join([str(i) for i in [self.args["df"], self.args["df_size"], self.args["random_seed"]]]))
        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)
        df_size = int(self.data.num_nodes * self.args["proportion_unlearned_nodes"])
        path_un = unlearning_path + "_" + str(self.run) + "_nodes_" + str(self.args["num_unlearned_nodes"])+ ".txt"
        df_nodes = np.loadtxt(path_un, dtype=int)
        self.unlearning_nodes = df_nodes
        df_nodes_set = set(df_nodes)
        all_exist = df_nodes_set.issubset(self.data.train_indices)
        global_node_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
        global_node_mask[df_nodes] = False

        dr_mask_node = global_node_mask
        df_mask_node = ~global_node_mask
        # breakpoint()
        assert df_mask_node.sum() == df_size

        # Delete edges associated with deleted nodes from training set
        res = [torch.eq(self.data.edge_index, aelem).logical_or_(torch.eq(self.data.edge_index, aelem)) for aelem in df_nodes]
        df_mask_edge = torch.any(torch.stack(res, dim=0), dim=0)
        df_mask_edge = df_mask_edge.sum(0).bool()
        dr_mask_edge = ~df_mask_edge

        df_edge = self.data.edge_index[:, df_mask_edge]
        self.data.directed_df_edge_index = to_directed(df_edge)

        # self.logger.info('Deleting the following nodes:{}'.format(df_nodes) )

        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            2,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        # Nodes in S_Df
        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            1,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)
        sdf_node_1hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        sdf_node_2hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)

        sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        sdf_node_2hop[two_hop_edge.flatten().unique()] = True

        assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
        assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

        self.data.sdf_node_1hop_mask = sdf_node_1hop
        self.data.sdf_node_2hop_mask = sdf_node_2hop

        # print(is_undirected(self.data.edge_index))

        two_hop_mask = two_hop_mask.bool()
        df_mask_edge = df_mask_edge.bool()
        dr_mask_edge = ~df_mask_edge

        # print('Undirected dataset:', self.data)evaluate_Del_model
        # print(is_undirected(train_pos_edge_index), train_pos_edge_index.shape, two_hop_mask.shape, df_mask.shape, two_hop_mask.shape)

        self.data.sdf_mask = two_hop_mask
        self.data.df_mask = df_mask_edge
        self.data.dr_mask = dr_mask_edge
        self.data.dtrain_mask = dr_mask_edge
        
        model = self.model_zoo.get_model(sdf_node_1hop, sdf_node_2hop, num_nodes=self.data.num_nodes)

        if self.args["unlearning_model"] != 'retrain':  # Start from trained GNN model
            if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
                logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))
                if logits_ori is not None:
                    logits_ori = logits_ori.to(self.device)
            else:
                logits_ori = None
            
            model_ckpt = self.target_model.model.state_dict()
            model.load_state_dict(model_ckpt, strict=False)
            self.target_model.model = model
            # model_ckpt = torch.load(os.path.join(root_path + "/data/model/node_level/" ,self.args["dataset_name"],self.args["downstream_task"] ,self.args["base_model"]), map_location=self.device)
            # model.load_state_dict(model_ckpt['model_state'], strict=False)
            # self.target_model.model.load_state_dict(model_ckpt, strict=False)

        else:  # Initialize a new GNN model
            retrain = None
            logits_ori = None
        self.target_model.model.to(self.device)


        if 'gnndelete' in self.args["unlearning_model"] and 'nodeemb' in self.args["unlearning_model"]:
            parameters_to_optimize = [
                {'params': [p for n, p in self.target_model.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters() if 'del' in n])
            if 'layerwise' in self.args["loss_type"]:
                optimizer1 = torch.optim.Adam(self.target_model.model.deletion1.parameters(), lr=self.args["unlearn_lr"])
                optimizer2 = torch.optim.Adam(self.target_model.model.deletion2.parameters(), lr=self.args["unlearn_lr"])
                optimizer = [optimizer1, optimizer2]
            else:
                optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args["unlearn_lr"])

        else:
            if 'gnndelete' in self.args["unlearning_model"]:
                parameters_to_optimize = [
                    {'params': [p for n, p in self.target_model.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
                ]
                print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters() if 'del' in n])

            else:
                parameters_to_optimize = [
                    {'params': [p for n, p in self.target_model.model.named_parameters()], 'weight_decay': 0.0}
                ]
                print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters()])

            optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args["unlearn_lr"])  # , weight_decay=args.weight_decay)

        # MI attack model
        attack_model_all = None
        # attack_model_all = MLPAttacker(args)
        # attack_ckpt = torch.load(os.path.join(attack_path_all, 'attack_model_best.pt'))
        # attack_model_all.load_state_dict(attack_ckpt['model_state'])
        # attack_model_all = attack_model_all.to(device)

        attack_model_sub = None
        # attack_model_sub = MLPAttacker(args)
        # attack_ckpt = torch.load(os.path.join(attack_path_sub, 'attack_model_best.pt'))
        # attack_model_sub.load_state_dict(attack_ckpt['model_state'])
        # attack_model_sub = attack_model_sub.to(device)

        #Train
        self.model_zoo.model = self.target_model.model
        self.args["unlearn_trainer"] = "GNNDeleteTrainer"
        self.target_model = get_trainer(self.args, self.logger,self.model_zoo.model, self.data)
        start_time  = time.time()
        # self.NodeClassifier_Del.GNNDelete_train(self.logger,self.avg_time,self.run,self.NodeClassifier.model, self.data, optimizer, self.args, logits_ori, attack_model_all, attack_model_sub)
        self.target_model.gnndelete_train(self.avg_time, self.run, optimizer,logits_ori, attack_model_all, attack_model_sub)
        self.avg_unlearning_time[self.run] = time.time() - start_time

        #Test
        # if self.args["unlearning_model"] != 'retrain':
        #     retrain_path = os.path.join(self.args["checkpoint_dir"],self.args["dataset_name"],self.args["gnn"],'retrain',
        #                                                   '-'.join([str(i) for i in [self.args["df"], self.args["df_size"], self.args["random_seed"]]]))
        #     retrain_ckpt = torch.load(os.path.join(retrain_path, 'model_best.pt'), map_location=self.device)
        #     retrain_args = copy.deepcopy(self.args)
        #     original_model = self.args["unlearning_model"]
        #     self.args["unlearning_model"] = 'retrain'
        #     retrain = self.model_zoo.get_model(num_nodes=self.data.num_nodes)
        #     self.args["unlearning_model"] =original_model
        #     retrain.load_state_dict(retrain_ckpt['model_state'])
        #     retrain = retrain.to(self.device)
        #     retrain.eval()


        # else:
        #     retrain = None

        # self.NodeClassifier_Del.GNNDelete_test(self.data, model_retrain=None, attack_model_all=attack_model_all,
        #              attack_model_sub=attack_model_sub)
        self.target_model.test_node_fullbatch_del(model_retrain=None, attack_model_all=attack_model_all,attack_model_sub=attack_model_sub)
        self.target_model.model.to(self.device)
        loss, dt_acc, recall, dt_f1, log = self.target_model.eval_del( 'test')
        self.logger.info(
            "Loss: {:.4f} | Accuracy: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
                loss, dt_acc, recall, dt_f1
            )
        )
        self.average_f1[self.run] = dt_acc
        # F1_score, Accuracy, Recall = self.NodeClassifier_Del.eval_unlearning(self.data.features_pre, self.unlearning_nodes)
        # self.logger.info(
            # 'Original Model Unlearning : F1_score = {}  Accuracy = {}  Recall = {}'.format(F1_score, Accuracy, Recall))

        ###MIA

        self.mia_num = df_size
        original_softlabels_member = self.original_softlabels[df_nodes]
        original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]

        if  self.args["base_model"] in ["SIGN","SGC","S2GC"]:
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.features_pre[df_nodes],sdf_node_1hop[df_nodes],sdf_node_2hop[df_nodes]),dim =1)
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.features_pre[self.data.test_indices[:self.mia_num]],
                sdf_node_1hop[self.data.test_indices[:self.mia_num]],
                sdf_node_2hop[self.data.test_indices[:self.mia_num]]),dim = 1)
        else:
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x, self.data.edge_index)[
                df_nodes],dim = 1)
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.x, self.data.edge_index)[self.data.test_indices[:self.mia_num]],dim=1)

        mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])


        # self.logger.info("posterior:{}".format(posterior))
        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        # self.logger.info("auc:{}".format(auc))
        self.average_auc[self.run] = auc
        plot_auc(mia_test_y, posterior.reshape(-1, 1))
        # === Save the updated model in the standardized format ===
        


    def delete_edge(self):
        """
        Deletes specified edges from the graph and updates the model accordingly.
        This function removes edges defined in `self.unlearning_edges` from the graph data.
        It updates various masks related to edges and nodes, recalculates subgraphs up to two hops,
        and prepares the target model for retraining or evaluation based on the unlearning strategy.
        Additionally, it sets up optimizers for the deletion process, logs relevant metrics, 
        and ensures the model is correctly loaded and moved to the appropriate device.
        """
        self.args["checkpoint_dir"] = root_path + '/data/GNNDelete/checkpoint_edge'
        original_path = os.path.join(self.args["checkpoint_dir"],self.args["dataset_name"],self.args["base_model"],'original',
                                                          '-'.join([str(i) for i in [self.args["df"], self.args["df_size"], self.args["random_seed"]]]))
        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)
        df_size = int(self.args["df_size"] / 100 * self.data.edge_index.shape[1])
        
        path_un = unlearning_edge_path + "_" + str(self.run) + "_edges_" + str(self.args["num_unlearned_edges"]) + ".txt"
        self.unlearning_nodes = None
        self.unlearning_edges = np.loadtxt(path_un, dtype=int)

        # Create a mask for edges to be deleted
        df_mask_edge = torch.zeros(self.data.edge_index.shape[1], dtype=torch.bool,device=self.device)
        for edge in self.unlearning_edges:
            mask = (self.data.edge_index[0] == edge[0]) & (self.data.edge_index[1] == edge[1])
            mask = mask.to(self.device)
            df_mask_edge |= mask

        # Ensure the mask is symmetric for undirected graphs
        if is_undirected(self.data.edge_index):
            df_mask_edge |= (self.data.edge_index[0] == self.unlearning_edges[:, 1]) & (self.data.edge_index[1] == self.unlearning_edges[:, 0])
        dr_mask_edge = ~df_mask_edge
        df_edge = self.data.edge_index[:, df_mask_edge]
        self.data.directed_df_edge_index = df_edge
        global_node_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
        dr_mask_node = global_node_mask
        df_mask_node = ~global_node_mask

        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            2,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        # Nodes in S_Df
        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            1,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)
        
        sdf_node_1hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        sdf_node_2hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)

        sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        sdf_node_2hop[two_hop_edge.flatten().unique()] = True

        assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
        assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

        self.data.sdf_node_1hop_mask = sdf_node_1hop
        self.data.sdf_node_2hop_mask = sdf_node_2hop

        print(is_undirected(self.data.edge_index))

        two_hop_mask = two_hop_mask.bool()
        df_mask_edge = df_mask_edge.bool()
        dr_mask_edge = ~df_mask_edge
        
        self.data.sdf_mask = two_hop_mask
        self.data.df_mask = df_mask_edge
        self.data.dr_mask = dr_mask_edge
        self.data.dtrain_mask = dr_mask_edge

        self.target_model.model = self.model_zoo.get_model(sdf_node_1hop, sdf_node_2hop, num_nodes=self.data.num_nodes)
        model_path  = os.path.join(root_path + "/data/model/edge_level/" ,self.args["dataset_name"], self.args['downstream_task'],self.args["base_model"])
        if self.args["unlearning_model"] != 'retrain':  # Start from trained GNN model
            if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
                logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))
                if logits_ori is not None:
                    logits_ori = logits_ori.to(self.device)
            else:
                logits_ori = None
                logits_ori = torch.load(model_path, map_location=self.device)
            # model.load_state_dict(model_ckpt['model_state'], strict=False)
            self.target_model.model.load_state_dict(logits_ori, strict=False)
            
        else:  # Initialize a new GNN model
            retrain = None
            logits_ori = None
        self.target_model.model.to(self.device)
        
        if 'gnndelete' in self.args["unlearning_model"] and 'nodeemb' in self.args["unlearning_model"]:
            parameters_to_optimize = [
                {'params': [p for n, p in self.target_model.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters() if 'del' in n])
            if 'layerwise' in self.args["loss_type"]:
                optimizer1 = torch.optim.Adam(self.target_model.model.deletion1.parameters(), lr=self.args["unlearn_lr"])
                optimizer2 = torch.optim.Adam(self.target_model.model.deletion2.parameters(), lr=self.args["unlearn_lr"])
                optimizer = [optimizer1, optimizer2]
            else:
                optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args["unlearn_lr"])
        else:
            if 'gnndelete' in self.args["unlearning_model"]:
                parameters_to_optimize = [
                    {'params': [p for n, p in self.target_model.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
                ]
                print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters() if 'del' in n])

            else:
                parameters_to_optimize = [
                    {'params': [p for n, p in self.target_model.model.named_parameters()], 'weight_decay': 0.0}
                ]
                print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters()])

            optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args["unlearn_lr"])  # , weight_decay=args.weight_decay)

        attack_model_all = None
        attack_model_sub = None

        self.model_zoo.model = self.target_model.model
        self.args["unlearn_trainer"] = "GNNDeleteTrainer"
        self.target_model = get_trainer(self.args, self.logger,self.model_zoo.model, self.data)
        # self.NodeClassifier_Del.GNNDelete_train(self.logger,self.avg_time,self.run,self.NodeClassifier.model, self.data, optimizer, self.args, logits_ori, attack_model_all, attack_model_sub)
        self.target_model.gnndelete_train(self.avg_time, self.run, optimizer,logits_ori, attack_model_all, attack_model_sub)
        self.target_model.test_edge(model_retrain=None, attack_model_all=attack_model_all,attack_model_sub=attack_model_sub)
        self.target_model.model.to(self.device)
        loss,acc,dt_auc,df_auc,log, = self.target_model.eval_del( 'test')
        self.logger.info(
            "Loss: {:.4f} | Accuracy: {:.4f}  | AUC Score: {:.4f}".format(
                loss, acc, dt_auc
            )
        )
        self.average_f1[self.run] = dt_auc

    def delete_feature(self):
        """
        Deletes specified node features and updates the GNN model accordingly.
        This method performs feature deletion on a subset of nodes designated for unlearning in the graph
        neural network. It sets the features of these nodes to zero, updates relevant masks for nodes
        and edges, and prepares the data for unlearning operations.
        """
        self.args["checkpoint_dir"] = root_path + '/data/GNNDelete/checkpoint_node_feature'
        original_path = os.path.join(self.args["checkpoint_dir"],self.args["dataset_name"],self.args["base_model"],'original',
                                                          '-'.join([str(i) for i in [self.args["df"], self.args["df_size"], self.args["random_seed"]]]))
        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)
        df_size = int(self.data.num_nodes * self.args["proportion_unlearned_nodes"])
        path_un = unlearning_path + "_" + str(self.run) + ".txt"
        df_nodes = np.loadtxt(path_un, dtype=int)
        self.unlearning_nodes = df_nodes
        df_nodes_set = set(df_nodes)
        all_exist = df_nodes_set.issubset(self.data.train_indices)
        self.data.x[df_nodes] = 0
        assert self.data.x[df_nodes].sum() == 0

        global_node_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
        # global_node_mask[df_nodes] = False

        dr_mask_node = global_node_mask
        df_mask_node = ~global_node_mask

        res = [torch.eq(self.data.edge_index, aelem).logical_or_(torch.eq(self.data.edge_index, aelem)) for aelem in df_nodes]
        df_mask_edge = torch.any(torch.stack(res, dim=0), dim=0)
        df_mask_edge = df_mask_edge.sum(0).bool()
        dr_mask_edge = ~df_mask_edge

        df_edge = self.data.edge_index[:, df_mask_edge]
        self.data.directed_df_edge_index = to_directed(df_edge)

        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            2,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        # Nodes in S_Df
        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            1,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)
        
        sdf_node_1hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        sdf_node_2hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)

        sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        sdf_node_2hop[two_hop_edge.flatten().unique()] = True

        assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
        assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

        self.data.sdf_node_1hop_mask = sdf_node_1hop
        self.data.sdf_node_2hop_mask = sdf_node_2hop

        print(is_undirected(self.data.edge_index))

        two_hop_mask = two_hop_mask.bool()
        df_mask_edge = df_mask_edge.bool()
        dr_mask_edge = ~df_mask_edge

        self.data.sdf_mask = two_hop_mask
        self.data.df_mask = df_mask_edge
        self.data.dr_mask = dr_mask_edge
        self.data.dtrain_mask = dr_mask_edge

        self.target_model.model = self.model_zoo.get_model(sdf_node_1hop, sdf_node_2hop, num_nodes=self.data.num_nodes)

        if self.args["unlearning_model"] != 'retrain':  # Start from trained GNN model
            if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
                logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))
                if logits_ori is not None:
                    logits_ori = logits_ori.to(self.device)
            else:
                logits_ori = None
            model_ckpt = torch.load(os.path.join(root_path + "/data/model/node_level/" ,self.args["dataset_name"], self.args["base_model"]), map_location=self.device)
            # model.load_state_dict(model_ckpt['model_state'], strict=False)
            self.target_model.model.load_state_dict(model_ckpt, strict=False)
            
        else:  # Initialize a new GNN model
            retrain = None
            logits_ori = None
        self.target_model.model.to(self.device)

        if 'gnndelete' in self.args["unlearning_model"] and 'nodeemb' in self.args["unlearning_model"]:
            parameters_to_optimize = [
                {'params': [p for n, p in self.target_model.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters() if 'del' in n])
            if 'layerwise' in self.args["loss_type"]:
                optimizer1 = torch.optim.Adam(self.target_model.model.deletion1.parameters(), lr=self.args["unlearn_lr"])
                optimizer2 = torch.optim.Adam(self.target_model.model.deletion2.parameters(), lr=self.args["unlearn_lr"])
                optimizer = [optimizer1, optimizer2]
            else:
                optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args["unlearn_lr"])
        else:
            if 'gnndelete' in self.args["unlearning_model"]:
                parameters_to_optimize = [
                    {'params': [p for n, p in self.target_model.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
                ]
                print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters() if 'del' in n])

            else:
                parameters_to_optimize = [
                    {'params': [p for n, p in self.target_model.model.named_parameters()], 'weight_decay': 0.0}
                ]
                print('parameters_to_optimize', [n for n, p in self.target_model.model.named_parameters()])

            optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args["unlearn_lr"])  # , weight_decay=args.weight_decay)

        attack_model_all = None
        attack_model_sub = None

        self.model_zoo.model = self.target_model.model
        self.args["unlearn_trainer"] = "GNNDeleteTrainer"
        self.target_model = get_trainer(self.args, self.logger,self.model_zoo.model, self.data)
        # self.NodeClassifier_Del.GNNDelete_train(self.logger,self.avg_time,self.run,self.NodeClassifier.model, self.data, optimizer, self.args, logits_ori, attack_model_all, attack_model_sub)
        self.target_model.gnndelete_train(self.avg_time, self.run, optimizer,logits_ori, attack_model_all, attack_model_sub)
        self.target_model.test_node_fullbatch_del(model_retrain=None, attack_model_all=attack_model_all,attack_model_sub=attack_model_sub)
        self.target_model.model.to(self.device)
        loss, dt_acc, recall, dt_f1, log = self.target_model.eval_del( 'test')
        self.logger.info(
            "Loss: {:.4f} | Accuracy: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
                loss, dt_acc, recall, dt_f1
            )
        )
        self.average_f1[self.run] = dt_acc
        self.mia_num = df_size
        original_softlabels_member = self.original_softlabels[df_nodes]
        original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]

        if self.args["base_model"] in ["SIGN","SGC","S2GC"]:
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.features_pre[df_nodes]),dim = 1)
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.features_pre[self.data.test_indices[:self.mia_num]]),dim = 1)
        else:
            unlearning_softlabels_member = F.softmax(self.target_model.model(self.data.x, self.data.edge_index)[
                df_nodes],dim = 1)
            unlearning_softlabels_non = F.softmax(self.target_model.model(
                self.data.x, self.data.edge_index)[self.data.test_indices[:self.mia_num]],dim=1)

        mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        posterior = np.array([np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))])
        # self.logger.info("posterior:{}".format(posterior))
        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        # self.logger.info("auc:{}".format(auc))
        self.average_auc[self.run] = auc
        plot_auc(mia_test_y, posterior.reshape(-1, 1))
        

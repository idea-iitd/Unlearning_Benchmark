import logging
import os
from scipy.sparse import coo_matrix
from os import path as osp
from typing import Dict, List, Optional, Tuple,Callable
from scipy.sparse import csr_matrix
import torch
import pickle
import torch_geometric.transforms as T
import networkx as nx
import torch_geometric
from torch_geometric.datasets import MNISTSuperpixels,ShapeNet
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, Flickr, Amazon, CitationFull,PPI,Reddit,TUDataset
from utils.dataset_utils import is_data_exists, load_saved_data, save_data
from torch_geometric.transforms import SIGN
from config import root_path,split_ratio
from torch_geometric.io import fs
from dataset.ppi_data import ppi_data
from torch_geometric.datasets import Actor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from utils.utils import sparse_mx_to_torch_sparse_tensor,normalize_adj

import json
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor


def load_saint_dataset(name: str, *, root: str="datasets") -> Data:
    r"""Loads Graph Saint provided dataset and converts it into proper
    torch_geometric.data.Data object.

    Params:
    name: str - Name of the dataset. One of flickr, ogbn-arxiv, or reddit
    root: str - Root directory of the dataset.
    returns: torch_geometric.data.Data
    """
    with open(f"{root}/{name}/role.json", "r") as jsonfile:
        roles = json.load(jsonfile)
    train_nodes = roles["tr"]
    val_nodes = roles["va"]
    test_nodes = roles["te"]
    adj = sp.load_npz(f"{root}/{name}/adj_full.npz")
    if name == "ogbn-arxiv":
        adj = adj + adj.T
        adj[adj > 1] = 1
    adj = adj.tocoo()
    rows = adj.row
    cols = adj.col
    edge_index = np.stack((rows, cols), axis=0)
    feats = np.load(f"{root}/{name}/feats.npy")
    feats = feats.astype(np.float32)
    with open(f"{root}/{name}/class_map.json", "r") as jsonfile:
        class_map = json.load(jsonfile)
    num_nodes = feats.shape[0]
    ys = np.zeros((num_nodes,))
    for node, cls in class_map.items():
        ys[int(node)] = cls
    ys = ys.astype(np.int64)  # Long.
    nc = np.unique(ys).shape[0]
    train_mask = np.zeros((num_nodes,))
    train_mask[train_nodes] = 1
    train_mask = train_mask.astype(bool)
    val_mask = np.zeros((num_nodes,))
    val_mask[val_nodes] = 1
    val_mask = val_mask.astype(bool)
    test_mask = np.zeros((num_nodes,))
    test_mask[test_nodes] = 1
    test_mask = test_mask.astype(bool)
    #
    adj = adj.tolil()
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    adj = adj.dot(r_mat_inv)
    adj = adj.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(adj.row).unsqueeze(1)
    sparsecol = torch.LongTensor(adj.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(adj.data)
    adj = torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(adj.shape))
    adj = SparseTensor(
        row=adj._indices()[0],
        col=adj._indices()[1],
        value=adj._values(),
        sparse_sizes=adj.size(),
    )
    data = Data(
        x=torch.tensor(feats),
        edge_index=torch.tensor(edge_index).long(),
        y=torch.tensor(ys),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
        num_nodes=torch.tensor(num_nodes),
        num_classes=torch.tensor(nc),
        adj=adj,
    )
    return data


class original_dataset:
    """
    This class handles loading and processing of various graph datasets. It supports both inductive and 
    transductive settings for dataset splits and can handle different types of datasets (e.g., Planetoid, 
    Amazon, PPI, etc.). The class supports loading and preprocessing the dataset, handling different splits 
    (train, validation, test), and managing dataset-specific attributes like the number of features and classes.

    Class Attributes:
        args (dict): A dictionary containing various configuration options like dataset name, split type, 
                     and base model type.
        
        dataset_name (str): The name of the dataset to load.
        
        num_features (dict): A dictionary that maps dataset names to the number of features in each dataset.
        
        base_model (str): The base model to use (e.g., GCN, GraphSAGE, etc.).
        
        logger (logging.Logger): A logger object for logging information.
    """
    def __init__(self,args,logger):
        """
        Initializes the original_dataset object with the provided arguments and logger.

        Args:
            
            args (dict): A dictionary containing configuration options.
            
            logger (logging.Logger): A logger object used to log dataset loading information.

        """
        self.args = args
        self.dataset_name = self.args['dataset_name']
        self.num_features = {
            "cora": 1433,
            "pubmed": 500,
            "citeseer": 3703,
            "CS": 6805,
            "Physics": 8415,
            'flickr': 500,
            "ppi": 50,
            "Computers": 767,
            "Photo": 745,
            "Reddit": 602,
            "ogbn-arxiv":128,
            "DBLP":334,
            "ogbn-products":100,
            'Squirrel': 2089,
            'Chameleon': 2325,
            'Actor': 931,
            'Minesweeper':7,
            'Tolokers': 10,
            'Roman-empire': 300,
            'Amazon-ratings': 300,
            'Questions':301,
            'MUTAG':7,
            'COX2':38,
            "BZR":56,
            "AIDS":42,
            "DD":89,
            "PROTEINS":4,
            "ENZYMES":21,
            "PTC_MR":9,
            "NCI1":37,
            "DHFR": 56,
            "MNISTSuperpixels":1
        }
        self.base_model = self.args['base_model'] 
        self.logger = logger


    def load_data(self):
        """
        Loads the dataset based on the dataset name and splits it according to the inductive or 
        transductive setting. It checks if the data has already been processed and saved; if not, 
        it loads and processes the raw data from the appropriate source.

        The method performs the following:

            - Checks if the dataset already exists (via file paths) and loads it if available.

            - If the dataset does not exist, it loads the dataset from raw data sources based on 
              the specified dataset name.

            - Preprocesses the dataset according to its type and settings (e.g., normalizing features, 
              converting to undirected graphs, handling different splits).

            - Saves the processed data for future use.

        Args:
            None

        Returns:
            tuple: A tuple containing two elements:

                - data (Data): The processed graph data object.

                - dataset (Dataset): The dataset object containing additional dataset-specific information.

        Raises:
            Exception: If the dataset name is not supported or invalid.
        """
        if self.args["is_transductive"]:
            if self.args['is_balanced']:
                data_filename = './data/processed/transductive/'+self.args['dataset_name']+ split_ratio +'_balanced.pkl'
                dataset_filename = './data/processed/transductive/'+self.args['dataset_name'] +  split_ratio +"dataset" +'_balanced.pkl'
            else:
                data_filename = './data/processed/transductive/'+self.args['dataset_name']+ split_ratio +'.pkl'
                dataset_filename = './data/processed/transductive/'+self.args['dataset_name'] +  split_ratio +"dataset" +'.pkl'
        else:
            if self.args['is_balanced']:
                data_filename = './data/processed/inductive/'+self.args['dataset_name']+ split_ratio +'_balanced.pkl'
                dataset_filename = './data/processed/inductive/'+self.args['dataset_name'] +  split_ratio +"dataset" +'_balanced.pkl'
            else:
                data_filename = './data/processed/inductive/' + self.args['dataset_name'] +  split_ratio +'.pkl'
                dataset_filename = './data/processed/inductive/' + self.args['dataset_name'] +  split_ratio +"dataset" + '.pkl'
        if is_data_exists(data_filename) and is_data_exists(dataset_filename):
            self.logger.info("Data already saved! "+ data_filename)
            data = load_saved_data(self.logger,data_filename)
            dataset = load_saved_data(self.logger, dataset_filename)
            self.args["num_unlearned_nodes"] = int(data.num_nodes * self.args["unlearn_ratio"])
            self.args["num_unlearned_edges"] = int(data.edge_index.shape[1] * self.args["unlearn_ratio"])
            return data, dataset

        #---------------------------------------------------------------------------------
        root_path = "/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/data"  
        
        #-------------------------------------------------------------------------------------
        if self.dataset_name in ["cora", "pubmed", "citeseer"]:
            dataset = Planetoid(root_path , self.dataset_name, transform=T.NormalizeFeatures())
            data = dataset._data
        elif self.dataset_name in ["Tolokers", "Roman-empire", "Amazon-ratings", "Questions", "Minesweeper"]:
            dataset =  HeterophilousGraphDataset(root=root_path , name=self.dataset_name)
            data = dataset._data
        elif self.dataset_name in ["Chameleon", "Squirrel"]:
            dataset =  WikiPages(root=root_path , name=self.dataset_name)
            data = dataset._data
        elif self.dataset_name in ["CS", "Physics"]:
            dataset = Coauthor(root_path , self.dataset_name, pre_transform=T.NormalizeFeatures())
            data = dataset._data
        elif self.dataset_name == 'flickr':
            dataset = Flickr(root_path  ,self.dataset_name, pre_transform=T.NormalizeFeatures())
            data = dataset._data
            data.num_classes = 7
        elif self.args["dataset_name"] == 'Reddit':
            # dataset = Reddit(root_path )
            # data = dataset._data
            # from load_saint_dataset import load_saint_dataset

        # data = load_saint_dataset(dataset_name, root=root)
            dataset = load_saint_dataset('reddit',root_path)
            data = dataset[0]  # this loads / uses processed files
        elif self.args["dataset_name"] == "ppi":
            train_datasets = PPI(root='./data/raw/ppi', split='train')
            val_datasets = PPI(root='./data/raw/ppi', split='val')
            test_datasets = PPI(root='./data/raw/ppi', split='test')
            all_data = []
            for train_dataset in train_datasets:
                all_data.append(train_dataset)
            for val_dataset in val_datasets:
                all_data.append(val_dataset)
            for test_dataset in test_datasets:
                all_data.append(test_dataset)
            ppi_ = ppi_data(all_data)
            ppi_.train_y = torch.cat([data.y for data in all_data[:20]], dim=0)
            ppi_.test_y = torch.cat([data.y for data in all_data[22:24]], dim=0)
            return ppi_,ppi_
        elif self.dataset_name in ['Computers','Photo']:
            dataset = Amazon(root_path , self.dataset_name, transform=T.NormalizeFeatures())
            data = dataset._data
        elif self.dataset_name in ['DBLP']:
            dataset = CitationFull(root_path + '/data/raw', self.args["dataset_name"], transform=T.NormalizeFeatures())
            data = dataset._data
        elif self.dataset_name in ['Actor']:
            dataset = Actor(root= root_path + '/data/raw')
            data = dataset._data
        elif self.dataset_name == 'AmazonRatings':
            dataset = Amazon(root=root_path , name='Ratings')
            data = dataset[0]
        elif self.args["dataset_name"] == 'obgl' and self.args["unlearning_methods"] == "CEU":
            dataset = PygLinkPropPredDataset(root_path + '/data/raw')
            data = dataset._data
        elif self.args["dataset_name"] in ['ogbn-arxiv', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=self.dataset_name, root = root_path)
            ogb_data = dataset[0]
            ogb_data = T.ToUndirected()(ogb_data)
            split_idx = dataset.get_idx_split()


            mask = torch.zeros(ogb_data.x.size(0))
            mask[split_idx['train']] = 1
            ogb_data.train_mask = mask.to(torch.bool)
            ogb_data.train_indices = split_idx['train']

            mask = torch.zeros(ogb_data.x.size(0))
            mask[split_idx['valid']] = 1
            ogb_data.val_mask = mask.to(torch.bool)
            ogb_data.val_indices = split_idx['valid']

            mask = torch.zeros(ogb_data.x.size(0))
            mask[split_idx['test']] = 1
            ogb_data.test_mask = mask.to(torch.bool)
            ogb_data.test_indices = split_idx['test']

            ogb_data.y = ogb_data.y.flatten()
            data = ogb_data
        elif self.args["dataset_name"] in ["ogbg-molhiv","ogbg-molpcba","ogbg-ppa"]:
            dataset = PygGraphPropPredDataset(name=self.dataset_name, root = root_path + '/data/raw')
            data = dataset._data
            data = dataset
            return data,dataset
        elif self.args["dataset_name"] == "MNISTSuperpixels":
            data = MNISTSuperpixels(name=self.dataset_name, root = root_path + '/data/raw')
            data = dataset._data
            return data,dataset
        elif self.args["dataset_name"] == "ShapeNet":
            data = ShapeNet(name=self.dataset_name, root = root_path + '/data/raw')
            data = dataset._data
            return data,dataset
            
        elif self.args["dataset_name"] in ["AIDS","BZR","COX2","DD","MUTAG","PROTEINS","PTC_MR","ENZYMES","NCI1","DHFR","IMDB-BINARY","IMDB-MULTI","COLLAB"]:
            dataset = TUDataset(root=root_path + '/data/raw',name=self.args["dataset_name"],use_node_attr=True,use_edge_attr=True)
            
            data = dataset
            return data,dataset
        else:
            raise Exception('unsupported dataset')

        data = T.ToUndirected()(data)
        data.name = self.dataset_name
        if not hasattr(data, 'num_classes'):
            data.num_classes = dataset.num_classes
        data.num_nodes = data.x.size(0)
        data.num_features = self.num_features[data.name]
        data.num_edges = data.edge_index.size(1)
        data.x = data.x.to(torch.float32)
        self.args["num_unlearned_nodes"] = int(data.num_nodes * self.args["unlearn_ratio"])
        self.args["num_unlearned_edges"] = int(data.edge_index.shape[0] * self.args["unlearn_ratio"])
        # breakpoint()
        # save_data(self.logger, data, data_filename)
        if self.args['dataset_name'] == "ogbn-products":
            adj = to_scipy_sparse_matrix(data.edge_index,num_nodes=data.num_nodes)
            adj = normalize_adj(adj)
            data.adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
        save_data(self.logger, dataset, dataset_filename)


        return data,dataset

    def edge2graph(self,edge_index):
        """
        Converts an edge index to an undirected graph and returns its adjacency matrix.

        This method takes an edge index, constructs an undirected graph from it, and returns the corresponding adjacency matrix.

        Args:
            
            edge_index (Tensor): The edge index tensor representing the graph's edges, with shape (2, num_edges), where each column represents an edge.

        Returns:
            
            scipy.sparse.csr_matrix: The adjacency matrix of the graph, represented in sparse CSR format.

        """
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        adj_matrix = nx.adjacency_matrix(G)


        return adj_matrix


class WikiPages(InMemoryDataset):
    """
    A dataset class for loading and processing the WikiPages dataset.

    This class handles the loading, processing, and saving of the WikiPages dataset,
    which consists of node features, labels, and graph edges. The dataset is loaded 
    into memory from processed files or downloaded and processed from the raw files.

    Class Attributes:
    
    url (str): The URL from which the dataset can be downloaded.
    
    name (str): The name of the dataset (e.g., "chameleon", "squirrel").
    
    raw_dir (str): The directory containing the raw dataset files.
    
    processed_dir (str): The directory where processed dataset files are stored.
    
    raw_file_names (List[str]): The list of raw data file names.
    
    processed_file_names (str): The name of the processed dataset file.
    """
    url = "https://data.dgl.ai/dataset"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Initializes the WikiPages dataset class.

        Args:
            root (str): The root directory where the dataset will be stored.

            name (str): The name of the dataset (e.g., "chameleon", "squirrel").

            transform (Optional[Callable], optional): A function/transform that takes in a data object and returns a transformed version.

            pre_transform (Optional[Callable], optional): A function/transform that is applied before saving the dataset to disk.

            force_reload (bool, optional): Whether to force reload the dataset even if it is already processed. Defaults to False.
        """
        self.name = name # [chameleon, squirrel]

        super().__init__(root, transform, pre_transform,force_reload=force_reload)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """
        Returns the directory path containing the raw data files.

        Returns:
            str: The path to the raw data directory.
        """
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        """
        Returns the directory path where the processed data is stored.

        Returns:
            str: The path to the processed data directory.
        """
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the list of raw data file names.

        Returns:
            List[str]: The list of raw data files.
        """
        return ["out1_graph_edges.txt", "out1_node_feature_label.txt"]

    @property
    def processed_file_names(self) -> str:
        """
        Returns the name of the processed dataset file.

        Returns:
            str: The name of the processed file.
        """
        return 'data.pt'

    def download(self) -> None:
        """
        Downloads the dataset from the specified URL and extracts it.

        The dataset is downloaded and extracted into the `raw_dir` directory.
        """
        fs.cp(f"{self.url}/{self.name.lower()}.zip", self.raw_dir, extract=True)

    def process(self) -> None:
        """
        Processes the raw dataset and saves it in the processed directory.

        Reads the raw files, extracts the edge index, node features, and labels, 
        then saves them into a PyTorch `Data` object, which is subsequently saved 
        to the processed directory.
        """
        edge_index_path = osp.join(self.raw_dir, "out1_graph_edges.txt")
        data_list = []
        with open(edge_index_path, 'r') as file:
            # Skip the header
            next(file)
            for line in file:
                data_list.append([int(number) for number in line.split()])
        edge_index = torch.tensor(data_list).long().T

        node_feature_label_path = osp.join(self.raw_dir, "out1_node_feature_label.txt")
        node_feature_list = []
        node_label_list = []
        with open(node_feature_label_path, 'r') as file:
            next(file)
            for line in file:
                node_id, feature, label = line.strip().split('\t')
                node_feature_list.append([int(num) for num in feature.split(',')])
                node_label_list.append(int(label))
        x = torch.tensor(node_feature_list)
        y = torch.tensor(node_label_list)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return f'{self.name}()'

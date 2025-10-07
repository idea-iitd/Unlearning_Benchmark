import torch
import pickle
import os
import numpy as np
import config
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import root_path
from torch_geometric.transforms import SIGN
from config import root_path,unlearning_path,split_ratio,unlearning_edge_path
from utils.utils import filter_edge_index_2
from torch_geometric.utils import negative_sampling
def process_data(logger,data,args):
    """
    Processes the data based on the specified arguments.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        data:
            
            The input data object to be processed.
        
        args (dict):
            
            Configuration parameters that determine the processing steps.
    
    Returns:
        torch_geometric.data.Data:
            The processed data object.
    """
    # Store original data properties that might be needed later
    has_train_edge_index = hasattr(data, "train_edge_index")
    original_train_edge_index = data.train_edge_index if has_train_edge_index else None
    
    #--------------start ---------------
    # Create train_edge_index based on train_mask if it doesn't exist
    if not hasattr(data, "train_edge_index") and hasattr(data, "train_mask"):
        # Get the nodes that are in the training set
        print("h1")
        train_nodes = torch.where(data.train_mask)[0]
        
        # Filter edges where both nodes are in the training set
        edge_index = data.edge_index
        train_mask = torch.isin(edge_index[0], train_nodes) & torch.isin(edge_index[1], train_nodes)    
        data.train_edge_index = edge_index[:, train_mask]
        print("h2")
        
        logger.info(f"Created train_edge_index from training mask with shape {data.train_edge_index.shape}")
    elif not hasattr(data, "train_edge_index"):
        # Fallback if train_mask is not available
        print("h3")
        data.train_edge_index = data.edge_index.clone()
        logger.info(f"Created train_edge_index as a clone of edge_index with shape {data.edge_index.shape}")
        print("h4")
    #-----------end ---------------
    
    if args["downstream_task"]=="graph":
        data.train_mask = torch.zeros(len(data))
        data.test_mask = torch.zeros(len(data))
        data.train_mask[:int(len(data)*0.8)] = True
        data.test_mask[int(len(data)*0.8):] = True
        data.train_indices =data.train_mask.nonzero(as_tuple=True)[0].tolist()
        data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
        
        return data
    #     data = graph_cls_process(args,data)
    #     return data
    if args["unlearning_methods"] == "CEU":
        data = ceu_process(args,data)
        return data

    if args['is_transductive']:
        if args['is_balanced']:
            filename = root_path + '/data/processed/transductive/' + args['dataset_name'] +split_ratio+ '_balanced.pkl'
            if is_data_exists(filename):
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
            else:
                data = transductive_split_node_balanced(logger,args,data,train_ratio=args["train_ratio"], val_ratio=args["val_ratio"], test_ratio=args["test_ratio"])
                save_data(logger, data, filename)
        else:
            filename = root_path + '/data/processed/transductive/' + args['dataset_name'] + split_ratio + '.pkl'
            if is_data_exists(filename):
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
                #-------------start ---------------

                #add num classes attribute

                if not hasattr(data, "num_classes") and hasattr(data, "y"):
                    if data.y.dim() > 0:  # Handle non-empty tensor
                        data.num_classes = int(data.y.max().item() + 1)
                        logger.info(f"Added num_classes={data.num_classes} based on data.y after loading from pickle")
                    else:
                        logger.warning("Could not determine num_classes from data.y after loading from pickle")

                # Re-create train_edge_index after loading from pickle if it was created before
                # or if it doesn't exist in the loaded data
                if ( not hasattr(data, "train_edge_index")):
                    # If we had created it before, restore it
                   
                    # Otherwise create it again
                    if hasattr(data, "train_mask"):
                        # Get the nodes that are in the training set
                        train_nodes = torch.where(data.train_mask)[0]
                        
                        # Filter edges where both nodes are in the training set
                        edge_index = data.edge_index

                        #----------------------------
                        # mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
                        
                        # for i in range(edge_index.shape[1]):
                        #     src, dst = edge_index[0, i], edge_index[1, i]
                        #     if src in train_nodes and dst in train_nodes:
                        #         mask[i] = True
                        
                        # # Create train_edge_index with only the edges between training nodes
                        # data.train_edge_index = edge_index[:, mask]
                        #-------------------------
                        train_mask = torch.isin(edge_index[0], train_nodes) & torch.isin(edge_index[1], train_nodes)    
                        data.train_edge_index = edge_index[:, train_mask]

                        
                        logger.info(f"Re-created train_edge_index from training mask with shape {data.train_edge_index.shape}")
                        print("edges recreated")
                        #---------------end ---------------
            else:
                data = transductive_split_node(logger,args,data,train_ratio=args["train_ratio"], val_ratio=args["val_ratio"], test_ratio=args["test_ratio"])
                save_data(logger, data, filename)
            
    else:
        if args['is_balanced']:
            filename = root_path + '/data/processed/inductive/' + args['dataset_name'] + split_ratio + '_balanced.pkl'
            if is_data_exists(filename):
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
            else:
                data = inductive_split_node_balanced(logger,args,data,train_ratio=args["train_ratio"], val_ratio=args["val_ratio"], test_ratio=args["test_ratio"])
                save_data(logger, data, filename)
        else:
            filename = root_path + '/data/processed/inductive/' + args['dataset_name'] + split_ratio +'.pkl'
            if is_data_exists(filename):
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
            else:
                data = inductive_split_node(logger,args,data,train_ratio=args["train_ratio"], val_ratio=args["val_ratio"], test_ratio=args["test_ratio"])
                save_data(logger, data, filename)
    if args["unlearn_task"]=="node":
        # breakpoint()
        if hasattr(data, "train_indices"):
            train_nodes = np.array(data.train_indices)
        elif hasattr(data, "train_mask"):
            train_nodes = torch.where(data.train_mask)[0].cpu().numpy()
            data.train_indices = train_nodes
            test_nodes = torch.where(data.test_mask)[0].cpu().numpy()
            data.test_indices = test_nodes
        for i in range(args["num_runs"]):
            shuffle_num = torch.randperm(train_nodes.size)
            unlearning_nodes = train_nodes[shuffle_num][:args["num_unlearned_nodes"]]
            path_un = unlearning_path + "_" + str(i) + "_nodes_" + str(args["num_unlearned_nodes"])+ ".txt"
            #-----------------------------------------------------------
            os.makedirs(os.path.dirname(path_un), exist_ok=True)
            #-------------------------done-------------------------------------------

            if not os.path.isfile(path_un):
                np.savetxt(path_un, unlearning_nodes, fmt="%d")
    elif args["unlearn_task"]=="edge":
        # edge_index_to_use = data.train_edge_index if hasattr(data, "train_edge_index") else data.edge_index
        edge_index_to_use = data.edge_index 
        train_edges = np.array(edge_index_to_use)
        # train_edges = np.array(data.train_edge_index)
        num_unlearned_edges = int(train_edges.shape[1] * args["unlearn_ratio"])
        for i in range(args["num_runs"]):
            path_un_edge = unlearning_edge_path + "_" + str(i) + "_edges_" + str(args["num_unlearned_edges"]) + ".txt"
            if args["poison"]:
                if not os.path.isfile(path_un_edge):
                    unlearning_edges = negative_sampling(
                        edge_index=data.edge_index, num_nodes=data.num_nodes,
                        num_neg_samples=num_unlearned_edges,force_undirected=True
                    )
                    data.edge_index = torch.cat([data.edge_index, unlearning_edges], dim=1)
                    data.train_edge_index = torch.cat([data.train_edge_index, unlearning_edges], dim=1)
                    np.savetxt(path_un_edge, unlearning_edges.T, fmt="%d")
                else:
                    unlearning_edges = np.loadtxt(path_un_edge).T
                    data.edge_index = torch.cat([data.edge_index, torch.tensor(unlearning_edges,dtype=torch.int)], dim=1)
                    data.train_edge_index = torch.cat([data.train_edge_index, torch.tensor(unlearning_edges,dtype=torch.int)], dim=1)
            else:
                shuffle_num = torch.randperm(train_edges.shape[1])
                unlearning_edges = train_edges[:, shuffle_num][:, :num_unlearned_edges]
            
                if not os.path.isfile(path_un_edge):
                    np.savetxt(path_un_edge, unlearning_edges.T, fmt="%d")
                
    return data

class BasicDataset(Dataset):
    """
    A basic dataset class for handling nodes and their corresponding labels.

    Args:
        nodes (list or torch.Tensor): The nodes in the dataset.

        labels (list or torch.Tensor): The labels corresponding to each node.
    """
    def __init__(self, nodes, labels):
        """
        Initializes the BasicDataset with nodes and labels.

        Args:
            nodes (list or torch.Tensor): The nodes in the dataset.

            labels (list or torch.Tensor): The labels corresponding to each node.
        """
        self.nodes = nodes
        self.labels = labels

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx], self.labels[idx]

    def remove(self, node):
        """
        Removes a node and its corresponding label from the dataset.

        Args:
            node:

                The node to be removed from the dataset.

        Returns:
            None
        """
        index = self.nodes.index(node)
        del self.nodes[index]
        del self.labels[index]



def ceu_process(args,data):
    """
    Processes the dataset for the CEU (Contrastive Edge Unlearning) method.
    
    This function prepares the dataset by extracting nodes, edges, features, and labels from the
    full dataset. It handles feature initialization if specified, splits the data into training,
    validation, and test sets, and saves these subsets for future use. The processed data is then
    organized into a dictionary for downstream tasks.
    
    Args:
        args (dict):
        
            A dictionary containing configuration parameters.
            
            - 'feature' (bool): Flag indicating whether to use existing features or initialize new ones.
            
            - 'emb_dim' (int): The dimensionality of the initialized feature embeddings.
            
            - 'dataset_name' (str): The name of the dataset being processed.
        
        data (torch_geometric.data.Data):
        
            The full dataset containing node features, edge indices, labels, and other relevant information.
    
    Returns:
        dict:
        
            A dictionary containing the processed dataset with the following keys:
            
                - 'nodes' (list):
                    
                    A list of node indices in the dataset.
                
                - 'edges' (list of tuples):
                    
                    A list of edge tuples representing connections between nodes.
                
                - 'features' (numpy.ndarray):
                    
                    The feature matrix for all nodes. Initialized or extracted based on the `args['feature']` flag.
                
                - 'labels' (list):
                    
                    A list of labels corresponding to each node.
                
                - 'num_nodes' (int):
                    
                    The total number of nodes in the dataset.
                
                - 'num_edges' (int):
                    
                    The total number of edges in the dataset.
                
                - 'num_classes' (int):
                    
                    The number of unique classes in the dataset, determined by the maximum label value.
                
                - 'num_features' (int):
                    
                    The number of features per node.
                
                - 'train_set' (BasicDataset):
                    
                    The training subset of the dataset.
                
                - 'valid_set' (BasicDataset):
                    
                    The validation subset of the dataset.
                
                - 'test_set' (BasicDataset):
                    
                    The test subset of the dataset.
    """
    nodes = list(range(len(data.x)))
    edges = [(e[0],e[1]) for e in data.edge_index.t().tolist()]
    features = data.x.numpy()
    num_features = data.x.size(1)
    labels = data.y.tolist()
    num_nodes = data.num_nodes
    if not args['feature']:
        features = initialize_features(data, num_nodes, args['emb_dim'],args)
        num_features = args['emb_dim']
    train_set_path = os.path.join('./data/CEU', args["dataset_name"], 'train_set.pt')
    valid_set_path = os.path.join('./data/CEU', args["dataset_name"], 'valid_set.pt')
    test_set_path = os.path.join('./data/CEU', args["dataset_name"], 'test_set.pt')
    if os.path.exists(train_set_path) and os.path.exists(valid_set_path) and os.path.exists(test_set_path):
        train_set = torch.load(os.path.join('./data/CEU', args["dataset_name"], 'train_set.pt'))
        valid_set = torch.load(os.path.join('./data/CEU', args["dataset_name"], 'valid_set.pt'))
        test_set = torch.load(os.path.join('./data/CEU', args["dataset_name"], 'test_set.pt'))
    else:
        nodes_train, nodes_test, labels_train, labels_test = train_test_split(nodes, labels, test_size=0.2)
        nodes_train, nodes_valid, labels_train, labels_valid = train_test_split(
            nodes_train, labels_train, test_size=0.2)
        train_set = BasicDataset(nodes_train, labels_train)
        valid_set = BasicDataset(nodes_valid, labels_valid)
        test_set = BasicDataset(nodes_test, labels_test)
        torch.save(train_set, train_set_path)
        torch.save(valid_set, valid_set_path)
        torch.save(test_set, test_set_path)
        
    data = {
        'nodes': nodes,
        'edges': edges,
        'features': features,
        'labels': labels,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'num_classes': np.max(labels) + 1,
        'num_features': num_features,
        # 'node2idx': node2idx,
        # 'label2idx': label2idx,
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
    }
    return data


def initialize_features(dataset, num_nodes, emb_dim,args):
    """
    Initializes node features for a dataset, loading from disk if available.
    
    This function checks if pre-initialized features exist for the specified embedding dimension.
    If they do, it loads them; otherwise, it initializes features using Xavier normal initialization,
    saves them to disk, and returns them as a NumPy array.
    
    Args:
        dataset:
        
            The dataset for which features are being initialized.
        
        num_nodes (int):
        
            The total number of nodes in the dataset.
        
        emb_dim (int):
        
            The dimensionality of the node feature embeddings.
        
        args (dict):
        
            Configuration parameters, including 'dataset_name'.
    
    Returns:
        numpy.ndarray: The initialized node feature matrix.
    """
    if emb_dim == 32:
        features_path = os.path.join('./data/CEU', args["dataset_name"], 'features.pt')
    else:
        features_path = os.path.join('./data/CEU', args["dataset_name"], f'features_{emb_dim}.pt')

    # import os
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    if os.path.exists(features_path):
        features = torch.load(features_path)
    else:
        features = torch.zeros(num_nodes, emb_dim)
        nn.init.xavier_normal_(features)
        torch.save(features, features_path)
    return features.numpy()



def inductive_split_node(logger,args,data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    create three graph:
    train: data.x[data.train_mask] data.train_edge_index
    val: the same
    test: the same

    :param logger:
    :param args:
    :param data:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """

    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    edge_index = data.edge_index

    train_mask = indices[:int(train_ratio * num_nodes)]
    val_mask = indices[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)]
    test_mask = indices[int((train_ratio + val_ratio) * num_nodes):]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_mask] = 1
    data.val_mask[val_mask] = 1
    data.test_mask[test_mask] = 1

    data.train_indices =data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    data.val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask


    train_edge_index = data.edge_index.clone()

    train_new_index = 0
    train_dict = {}
    for node in range(num_nodes):
        if data.train_mask[node]:
            train_dict[node] = train_new_index
            train_new_index += 1

    train_edge_mask = data.train_mask[train_edge_index[0]] & data.train_mask[train_edge_index[1]]
    data.train_edge_index = train_edge_index[:, train_edge_mask]
    for edge in range(data.train_edge_index.size(1)):
        data.train_edge_index[0][edge] = train_dict[data.train_edge_index[0][edge].item()]
        data.train_edge_index[1][edge] = train_dict[data.train_edge_index[1][edge].item()]


    test_neg_edge_index = negative_sampling(
            edge_index=data.test_edge_index,num_nodes=data.num_nodes,
            num_neg_samples=data.test_edge_index.size(1)
        )
    pos_edge_labels = torch.ones(data.test_edge_index.size(1),dtype=torch.float32)
    neg_edge_labels = torch.zeros(test_neg_edge_index.size(1),dtype=torch.float32)
    test_edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
    data.test_neg_edge_index = test_neg_edge_index
    data.edge_labels = test_edge_labels
    # save_train_test_split(logger,args,data.train_indices,data.test_indices)

    return data

def inductive_split_node_balanced(logger,args,data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    create three graph:
    train: data.x[data.train_mask] data.train_edge_index
    val: the same
    test: the same

    :param logger:
    :param args:
    :param data:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """

    num_nodes = data.num_nodes
    labels = data.y  
    num_classes = labels.max().item() + 1
    edge_index = data.edge_index

    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(num_classes):
        class_indices = (labels == i).nonzero(as_tuple=True)[0]  
        class_indices = class_indices[torch.randperm(class_indices.size(0))] 

        num_train = int(train_ratio * len(class_indices))
        num_val = int(val_ratio * len(class_indices))

        train_indices.append(class_indices[:num_train])
        val_indices.append(class_indices[num_train:num_train + num_val])
        test_indices.append(class_indices[num_train + num_val:])

    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices)
    test_indices = torch.cat(test_indices)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_indices] = 1
    data.val_mask[val_indices] = 1
    data.test_mask[test_indices] = 1

    data.train_indices =data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    data.val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()
    
    train_edge_index = data.edge_index.clone()

    train_new_index = 0
    train_dict = {}
    for node in range(num_nodes):
        if data.train_mask[node]:
            train_dict[node] = train_new_index
            train_new_index += 1

    train_edge_mask = data.train_mask[train_edge_index[0]] & data.train_mask[train_edge_index[1]]
    data.train_edge_index = train_edge_index[:, train_edge_mask]
    for edge in range(data.train_edge_index.size(1)):
        data.train_edge_index[0][edge] = train_dict[data.train_edge_index[0][edge].item()]
        data.train_edge_index[1][edge] = train_dict[data.train_edge_index[1][edge].item()]

    return data
    
def transductive_split_edge(logger,args,data, train_ratio=0.8, val_ratio=0, test_ratio=0.2):
    """
    Splits the nodes of a graph into training, validation, and test sets inductively.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters.
        
        data:
            
            The input data object to be processed.
        
        train_ratio (float, optional):
            
            Proportion of nodes to include in the training set. Defaults to `0.6`.
        
        val_ratio (float, optional):
            
            Proportion of nodes to include in the validation set. Defaults to `0.2`.
        
        test_ratio (float, optional):
            
            Proportion of nodes to include in the test set. Defaults to `0.2`.
    
    Returns:
        torch_geometric.data.Data:
            The processed data object.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
    data.val_mask = torch.ones(num_nodes, dtype=torch.bool)
    data.test_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    data.train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    data.val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()
    
    num_train_edges = int(train_ratio * num_edges)
    num_val_edges = int(val_ratio * num_edges)
    num_test_edges = int(test_ratio * num_edges)
    
    shuffle_num = torch.randperm(num_edges)
    data.train_edge_index = data.edge_index[:, shuffle_num[:num_train_edges]]
    data.val_edge_index = data.edge_index[:, shuffle_num[num_train_edges:num_train_edges + num_val_edges]]
    data.test_edge_index = data.edge_index[:, shuffle_num[num_train_edges + num_val_edges:]]
    
    return data

def transductive_split_node(logger, args, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split the dataset into training, validation, and test sets in a transductive learning setting.
    
    Args:
        logger: Logger object for logging information.
        args: Configuration parameters.
        data: The input data object.
        train_ratio: Ratio of nodes for training.
        val_ratio: Ratio of nodes for validation.
        test_ratio: Ratio of nodes for testing.
    
    Returns:
        The processed data object with train, validation, and test masks.
    """
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    # Calculate split sizes
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Set masks
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:train_size + val_size + test_size]] = True
    
    # Set masks on data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Store indices for easier access
    data.train_indices = indices[:train_size].tolist()
    data.val_indices = indices[train_size:train_size + val_size].tolist()
    data.test_indices = indices[train_size + val_size:train_size + val_size + test_size].tolist()
    
    # For edge prediction tasks, also split edges
    if hasattr(data, 'edge_index'):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Randomly permute edges
        perm = torch.randperm(num_edges)
        
        # Calculate split sizes for edges
        train_edge_size = int(num_edges * train_ratio)
        val_edge_size = int(num_edges * val_ratio)
        test_edge_size = int(num_edges * test_ratio)
        
        # Split edges
        data.train_edge_index = edge_index[:, perm[:train_edge_size]]
        data.val_edge_index = edge_index[:, perm[train_edge_size:train_edge_size + val_edge_size]]
        data.test_edge_index = edge_index[:, perm[train_edge_size + val_edge_size:train_edge_size + val_edge_size + test_edge_size]]
    
    logger.info(f"Transductive split completed: Train nodes: {train_size}, Val nodes: {val_size}, Test nodes: {test_size}")
    
    return data

def transductive_split_node_balanced(logger, args, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Splits the nodes of a graph into training, validation, and test sets in a balanced manner.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters.
        
        data:
            
            The input data object to be processed.
        
        train_ratio (float, optional):
            
            Proportion of nodes to include in the training set. Defaults to `0.6`.
        
        val_ratio (float, optional):
            
            Proportion of nodes to include in the validation set. Defaults to `0.2`.
        
        test_ratio (float, optional):
            
            Proportion of nodes to include in the test set. Defaults to `0.2`.
    
    Returns:
        torch_geometric.data.Data:
            The processed data object.
    """
    num_nodes = data.num_nodes
    labels = data.y  
    num_classes = labels.max().item() + 1

    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(num_classes):
        class_indices = (labels == i).nonzero(as_tuple=True)[0]  
        class_indices = class_indices[torch.randperm(class_indices.size(0))]  

        num_train = int(train_ratio * len(class_indices))
        num_val = int(val_ratio * len(class_indices))

        train_indices.append(class_indices[:num_train])
        val_indices.append(class_indices[num_train:num_train + num_val])
        test_indices.append(class_indices[num_train + num_val:])

    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices)
    test_indices = torch.cat(test_indices)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    data.train_indices = train_indices.tolist()
    data.val_indices = val_indices.tolist()
    data.test_indices = test_indices.tolist()

    data.train_edge_index = data.edge_index.clone()
    data.val_edge_index = data.edge_index.clone()
    data.test_edge_index = data.edge_index.clone()
    # save_train_test_split(logger, args, data.train_indices, data.test_indices)

    return data


def c2n_to_n2c(args, community_to_node):
    """
    Converts a community-to-node mapping to a node-to-community mapping.
    
    This function transforms a dictionary that maps each community to its list of nodes into a
    dictionary that maps each individual node to its corresponding community. It processes
    a specified number of shards as defined in the `args` dictionary.
    
    Args:
        args (dict):
        
            A dictionary containing configuration parameters.
        
            - 'num_shards' (int): The number of shards to process.
        
        community_to_node (dict):
        
            A dictionary mapping each community to a list of nodes belonging to that community.
    
    Returns:
        dict:
        
            A dictionary mapping each node to its corresponding community.
    """
    node_list = []
    for i in range(args['num_shards']):
        node_list.extend(list(community_to_node.values())[i])
    node_to_community = {}

    for comm, nodes in dict(community_to_node).items():
        for node in nodes:
            # Map node id back to original graph
            # node_to_community[node_list[node]] = comm
            node_to_community[node] = comm

    return node_to_community





def save_train_test_split(logger,args,train_indices, test_indices):
    """
    Saves the train-test split data to a file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters.
        
        train_indices (list):
            
            List of training node indices.
        
        test_indices (list):
            
            List of testing node indices.
    
    Returns:
        None:
            The function saves the train-test split data to the specified file.
    """
    train_test_split_file = config.train_test_split_file
    os.makedirs(os.path.dirname(train_test_split_file), exist_ok=True)
    logger.info("save_train_test_split:{}".format(train_test_split_file) )
    pickle.dump((train_indices, test_indices), open(train_test_split_file, 'wb'))

def save_data(logger,data, filename):
    """
    Saves data to a specified file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        data:
            
            The data object to be saved.
        
        filename (str):
            
            Path to the file where data will be saved.
    
    Returns:
        None:
            The function saves the data to the specified file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    logger.info("save_data {}".format(filename))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def save_train_data(logger,data,filename):
    """
    Saves training data to a specified file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        data:
            
            The training data object to be saved.
        
        filename (str):
            
            Path to the file where training data will be saved.
    
    Returns:
        None:
            The function saves the training data to the specified file.
    """
    logger.info("save_train_data {}".format(filename))
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def save_train_graph(logger,gragh_data,filename):
    """
    Saves training graph data to a specified file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        gragh_data:
            
            The training graph data to be saved.
        
        filename (str):
            
            Path to the file where training graph data will be saved.
    
    Returns:
        None:
            The function saves the training graph data to the specified file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(gragh_data, file)

def save_unlearned_data(logger,data, suffix):
    """
    Saves unlearned data to a specified file with a suffix.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        data:
            
            The unlearned data to be saved.
        
        suffix (str):
            
            Suffix to append to the filename.
    
    Returns:
        None:
            The function saves the unlearned data to the specified file.
    """
    logger.info('saving unlearned data {}'.format('_'.join((config.unlearned_file, suffix))))
    pickle.dump(data, open('_'.join((config.unlearned_file, suffix)), 'wb'))


def load_unlearned_data(logger, suffix):
    """
    Loads unlearned data from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        suffix (str):
            
            Suffix to append to the unlearned file name.
    
    Returns:
        Any:
            The loaded unlearned data.
    """
    file_path = '_'.join((config.unlearned_file, suffix))
    logger.info('loading unlearned data from %s' % file_path)
    return pickle.load(open(file_path, 'rb'))

def save_community_data(logger,community_to_node,filename, suffix=''):
    """
    Saves community data to a specified file with an optional suffix.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        community_to_node:
            
            The community-to-node mapping data to be saved.
        
        filename (str):
            
            Path to the file where community data will be saved.
        
        suffix (str, optional):
            
            Suffix to append to the filename. Defaults to ''.
    
    Returns:
        None:
            The function saves the community data to the specified file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    logger.info('save_community_data {}'.format(filename + suffix))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(community_to_node, open(filename + suffix, 'wb'))

def save_shard_data(logger,shard_data, filename):
    """
    Saves shard data to a specified file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        shard_data:
            
            The shard data to be saved.
        
        filename (str):
            
            Path to the file where shard data will be saved.
    
    Returns:
        None:
            The function saves the shard data to the specified file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(shard_data, open(filename, 'wb'))

def load_shard_data(logger):
    """
    Loads shard data from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
    
    Returns:
        Any:
            The loaded shard data.
    """
    logger.info("load_shard_data {}".format(config.shard_file))
    return pickle.load(open(config.shard_file, 'rb'))


def load_train_graph(logger):
    """
    Loads the training graph from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
    
    Returns:
        Any:
            The loaded training graph data.
    """
    logger.info("load_train_graph {}".format(config.train_graph_file))
    return pickle.load(open(config.train_graph_file, 'rb'))

def save_embeddings(embeddings,filename):
    """
    Saves embeddings to a specified file.
    
    Args:
        embeddings:
            
            The embeddings data to be saved.
        
        filename (str):
            
            Path to the file where embeddings will be saved.
    
    Returns:
        None:
            The function saves the embeddings to the specified file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(embeddings, open(filename, 'wb'))

def load_community_data(logger,filename = config.load_community_data,suffix=''):
    """
    Loads community data from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        filename (str, optional):
            
            Path to the community data file. Defaults to `config.load_community_data`.
        
        suffix (str, optional):
            
            Suffix to append to the filename. Defaults to an empty string.
    
    Returns:
        Any:
            The loaded community data.
    """
    # logger.info("load_community_data {}".format(filename+suffix))
    return pickle.load(open(filename + suffix, 'rb'))

def load_saved_data(logger,filename = config.train_data_file):
    """
    Loads saved training data from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        filename (str, optional):
            
            Path to the saved training data file. Defaults to `config.train_data_file`.
    
    Returns:
        Any:
            The loaded training data.
    """
    logger.info('load_saved_data {}'.format(filename))
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def load_embeddings(logger,filename):
    """
    Loads embeddings from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        filename (str):
            
            Path to the embeddings file.
    
    Returns:
        Any:
            The loaded embeddings.
    """
    logger.info('load_embeddings {}'.format(filename))
    with open(filename, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

def load_train_test_split(logger, filename = config.train_test_split_file):
    """
    Loads the train-test split data from a file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        filename (str, optional):
            
            Path to the train-test split file. Defaults to `config.train_test_split_file`.
    
    Returns:
        dict:
            The loaded train-test split data.
    """
    logger.info("load_train_test_split {}".format(filename))
    return  pickle.load(open(filename, 'rb'))

def is_data_exists(filename):
    """
    Checks if a data file exists.
    
    This function verifies the existence of a file at the specified path.

    Args:
        filename (str):
        
            The path to the file to be checked.
    
    Returns:
        bool:
            `True` if the file exists, `False` otherwise.
    """
    return os.path.exists(filename)

def _extract_embedding_method(partition_method):
    return partition_method.split('_')[0]

def save_target_model(logger,args,run, model, shard, suffix=''):
    """
    Saves the target model based on experiment type and parameters.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters, including 'exp' and 'num_unlearned_nodes'.
        
        run:
            
            Identifier for the current run.
        
        model:
            
            The model object to save.
        
        shard:
            
            Shard identifier.
        
        suffix (str, optional):
            
            Suffix to append to the model file name. Defaults to ''.
    
    Returns:
        None:
            The function saves the model.
    """
    if not os.path.exists(config.MODEL_PATH + args['dataset_name']):
        os.mkdir(config.MODEL_PATH + args['dataset_name'])
    if args["exp"] in ["node_edge_unlearning", "attack_unlearning"]:
        model_path = '_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes']))) + suffix
        model.save_model(model_path)
        logger.info("save_target_model {}".format(model_path))
    else:
        model.save_model(config.target_model_file + '_' + str(shard) + '_' + str(run))
        logger.info("save_target_model {}".format(config.target_model_file + '_' + str(shard) + '_' + str(run)))
        # model.save_model(self.target_model_file + '_' + str(shard))

def load_target_model(logger,args, run, model, shard, suffix=''):
    """
    Loads the target model based on experiment type and parameters.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters, including 'exp' and 'num_unlearned_nodes'.
        
        run:
            
            Identifier for the current run.
        
        model:
            
            The model object to load the weights into.
        
        shard:
            
            Shard identifier.
        
        suffix (str, optional):
            
            Suffix to append to the model file name. Defaults to ''.
    
    Returns:
        None
    """
    if args["exp"] == "node_edge_unlearning":
        model.load_model(
            '_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes']))))
        # logger.info("load_target_model {}".format('_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes'])))))
    elif args["exp"] == "attack_unlearning":
        if(suffix == ''):
            model_path = '_'.join((config.target_model_file, str(shard), str(run)))
        else:
            model_path = '_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes']))) + suffix
        device = torch.device('cpu')
        model.load_model(model_path)
        model.device=device
        # logger.info("load_target_model {}".format(model_path))
    else:
        # model.load_model(self.target_model_file + '_' + str(shard) + '_' + str(run))
        model.load_model(config.target_model_file + '_' + str(shard) + '_' + str(run))
        # logger.info("load_target_model {}".format(config.target_model_file + '_' + str(shard) + '_' + str(0)))

def save_posteriors(logger,args, posteriors, run, suffix=''):
    """
    Saves posterior data to a specified file with an optional suffix.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters, including 'dataset_name' and 'target_model_name'.
        
        posteriors:
            
            The posterior data to be saved.
        
        run:
            
            Identifier for the current run.
        
        suffix (str, optional):
            
            Suffix to append to the filename. Defaults to ''.
    
    Returns:
        None:
            The function saves the posterior data to the specified file.
    """
    posteriors_path = config.ANALYSIS_PATH + 'posteriors/' + args['dataset_name'] + '/' + config.target_model_name
    os.makedirs(os.path.dirname(posteriors_path), exist_ok=True)
    logger.info("save_posteriors {}".format(posteriors_path))
    torch.save(posteriors, posteriors_path + '_' + str(run) + suffix)
    # breakpoint()

def save_optimal_weight(logger,args,weight,run):
    """
    Saves optimal weight data to a specified file.
    
    Args:
        logger:
            
            Logger object for logging information.
        
        args (dict):
            
            Configuration parameters, including 'dataset_name' and 'optimal_weight_name'.
        
        weight:
            
            The optimal weight data to be saved.
        
        run:
            
            Identifier for the current run.
    
    Returns:
        None:
            The function saves the optimal weight data to the specified file.
    """
    optimal_weight_path = config.ANALYSIS_PATH + 'optimal/' + args['dataset_name'] + '/' + config.optimal_weight_name
    os.makedirs(os.path.dirname(optimal_weight_path), exist_ok=True)
    torch.save(weight, optimal_weight_path + '_' + str(run))


def build_shard_data(data_full, train_shard_indices, val_indices=None, test_indices=None):
    """
    Constructs a shard of the dataset by selecting specified training, validation, and test nodes.
    
    This function creates a subset of the full dataset (`data_full`) by selecting nodes based on the provided
    `train_shard_indices`, `val_indices`, and `test_indices`. If `val_indices` or `test_indices` are not
    provided, they are derived from the corresponding masks in `data_full`. The function ensures that the
    resulting shard includes all necessary edges between the selected nodes.
    
    Args:
        data_full (torch_geometric.data.Data):
        
            The complete dataset containing all nodes, edges, labels, and masks for validation and testing.
        
        train_shard_indices (list or numpy.ndarray):
        
            A list or array of node indices designated for training within the shard.
        
        val_indices (list or numpy.ndarray, optional):
        
            A list or array of node indices designated for validation within the shard. If `None`, indices are
            derived from `data_full.val_mask`.
        
        test_indices (list or numpy.ndarray, optional):
        
            A list or array of node indices designated for testing within the shard. If `None`, indices are
            derived from `data_full.test_mask`.
    
    Returns:
        torch_geometric.data.Data:
        
            A new `Data` object representing the shard, containing:
            
                - `x` (torch.Tensor):
                    
                    The feature matrix for the selected nodes.
                
                - `edge_index` (torch.Tensor):
                    
                    The edge indices connecting the selected nodes.
                
                - `y` (torch.Tensor):
                    
                    The labels for the selected nodes.
                
                - `train_mask` (torch.Tensor):
                    
                    A boolean mask indicating which nodes in the shard are designated for training.
                
                - `val_mask` (torch.Tensor):
                    
                    A boolean mask indicating which nodes in the shard are designated for validation.
                
                - `test_mask` (torch.Tensor):
                    
                    A boolean mask indicating which nodes in the shard are designated for testing.
                
                - `edge_index_train` (torch.Tensor):
                    
                    The edge indices used specifically for training within the shard.
    """
    # Avoid computing val/test indices for multiple times when building multiple shards
    val_indices = torch.nonzero(data_full.val_mask).squeeze(1).numpy() if val_indices is None else val_indices
    test_indices = torch.nonzero(data_full.test_mask).squeeze(1).numpy() if test_indices is None else test_indices

    shard_indices = np.union1d(np.union1d(train_shard_indices, val_indices), 
                               test_indices)
    x = data_full.x[shard_indices]
    y = data_full.y[shard_indices]
    edge_index, edge_index_train = filter_edge_index_2(data_full, shard_indices)

    data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
    data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
    data.val_mask = torch.from_numpy(np.isin(shard_indices, val_indices))
    data.test_mask = torch.from_numpy(np.isin(shard_indices, test_indices))
    data.edge_index_train = torch.from_numpy(edge_index_train)

    return data

def graph_cls_process(graph_data,train_ratio=0.7,val_ratio=0.1,test_ratio=0.2):
        """
        Combines multiple graphs into one large graph and splits it into train, validation, and test sets.
        
        This function merges individual graphs by concatenating their node features, edges, and labels.
        It assigns unique graph IDs to each node to track their original graph. The combined graph is then
        divided into training, validation, and test subsets based on the provided ratios. Masks and indices
        are created to identify nodes in each subset.
        
        Args:
            graph_data (list of torch_geometric.data.Data):
            
                A list of individual graph `Data` objects to be combined.
            
            train_ratio (float, optional):
            
                Proportion of graphs to include in the training set. Defaults to `0.7`.
            
            val_ratio (float, optional):
            
                Proportion of graphs to include in the validation set. Defaults to `0.1`.
            
            test_ratio (float, optional):
            
                Proportion of graphs to include in the test set. Defaults to `0.2`.
        
        Returns:
            torch_geometric.data.Data:
            
                The combined and split graph data object.
        """
        all_x = []            
        all_edge_index = []    
        all_y = []            
        all_graph_id = []     
        node_offset = 0        

    
        
        for i, data in enumerate(graph_data):
            # print(data)
       
            all_x.append(data.x)
            all_y.append(data.y)
            

            edge_index = data.edge_index + node_offset
            all_edge_index.append(edge_index)
            

            all_graph_id.append(torch.full((data.num_nodes,), i, dtype=torch.long))
            

            node_offset += data.num_nodes

        big_x = torch.cat(all_x, dim=0)
        big_edge_index = torch.cat(all_edge_index, dim=1)
        big_y = torch.cat(all_y, dim=0)
        big_graph_id = torch.cat(all_graph_id, dim=0)

    
        big_graph = Data(x=big_x, edge_index=big_edge_index, y=big_y, graph_id=big_graph_id)
        data = big_graph
        graph_ids = torch.unique(data.graph_id)
        num_train_ids = int(len(graph_ids) * train_ratio)
        shuffle_num = torch.randperm(len(graph_ids))
        train_ids = graph_ids[shuffle_num][:num_train_ids]
        val_ids = graph_ids[shuffle_num][num_train_ids:num_train_ids + int(len(graph_ids) * val_ratio)]
        test_ids = graph_ids[shuffle_num][num_train_ids + int(len(graph_ids) * val_ratio):]
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        train_label_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        val_label_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        test_label_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for gid in train_ids:
            mask = data.graph_id == gid
            indice = mask.nonzero(as_tuple=True)[0]
            train_indices.append(indice)
            train_mask[mask] = True
            train_label_mask[gid] = True
        for gid in val_ids:
            mask = data.graph_id == gid
            indice = mask.nonzero(as_tuple=True)[0]
            val_indices.append(indice)
            val_mask[mask] = True
            val_label_mask[gid] = True
        for gid in test_ids:
            mask = data.graph_id == gid
            indice = mask.nonzero(as_tuple=True)[0]
            test_indices.append(indice)
            test_mask[mask] = True
            test_label_mask[gid] = True
            
        data.train_ids = train_ids
        data.val_ids = val_ids
        data.test_ids = test_ids
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_indices = train_indices
        data.val_indices = val_indices
        data.test_indices = test_indices
        data.train_label_mask = train_label_mask
        data.val_label_mask = val_label_mask
        data.test_label_mask = test_label_mask
        
        return data
        


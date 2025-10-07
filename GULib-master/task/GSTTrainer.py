import copy
import numpy as np
from tqdm import tqdm
from task import BaseTrainer
from config import root_path
from sklearn.metrics import f1_score, accuracy_score,recall_score
import time
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj, add_self_loops, contains_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader
# from unlearning.unlearning_methods.GST.gst_based import *
from utils import *
from utils.utils import *
from torch_geometric.data import Data

class GSTTrainer(BaseTrainer):
    """
    GSTTrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for applying the GST (Graph Scattering Transform) unlearning method.

    This class extends the `BaseTrainer` to implement specific training and evaluation routines required for the GST unlearning methodology. 
    It includes methods for training the GNN model, evaluating its accuracy and loss, generating GST embeddings, and performing 
    unlearning operations to remove the influence of specific nodes or edges from the trained model.

    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, 
                    training hyperparameters, unlearning settings, and other relevant settings.

        logger (logging.Logger): Logger object used to log training progress, metrics, 
                                 and other important information.

        model (torch.nn.Module): The neural network model that will be trained and evaluated.

        data (torch_geometric.data.Data): The dataset containing edge and node information 
                                          for training, validation, and testing.

        device (torch.device): The computation device (CPU or GPU) on which the model 
                               and data are loaded for training and evaluation.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the GSTTrainer with the provided configuration, logger, model, and data.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications, 
                        training hyperparameters, unlearning settings, and other relevant settings.
                        
            logger (logging.Logger): Logger object used to log training progress, metrics, 
                                     and other important information.
                        
            model (torch.nn.Module): The neural network model that will be trained and evaluated.
                        
            data (torch_geometric.data.Data): The dataset containing edge and node information 
                                              for training, validation, and testing.
        """
        super().__init__(args, logger, model, data)

    def train(self,model, optimizer, loader, device):
        """
        For GIN.
        Trains the GNN model for one epoch using the GST unlearning method.

        This method performs a single epoch of training, including forward propagation, loss computation, 
        backpropagation, and optimizer updates.

        Args:
            model (torch.nn.Module): The GNN model to be trained.
            
            optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
            
            loader (DataLoader): DataLoader for the training dataset.
            
            device (torch.device): The device on which the training is performed (CPU or GPU).

        Returns:
            float: The average training loss for the epoch.
        """
        model.train()
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y.view(-1))
            loss.backward()
            total_loss += loss.item() * self.num_graphs(data)
            optimizer.step()
        return total_loss / len(loader.dataset)
    
    def eval_acc(self, model, loader,device):
        """
        For GIN.
        Evaluates the accuracy of the GNN model on a given dataset.

        This method computes the accuracy by comparing the model's predictions with the true labels.

        Args:
            model (torch.nn.Module): The GNN model to be evaluated.
            
            loader (DataLoader): DataLoader for the evaluation dataset.
            
            device (torch.device): The device on which the evaluation is performed (CPU or GPU).

        Returns:
            float: The accuracy of the model on the dataset.
        """
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        return correct / len(loader.dataset)


    def eval_loss(self,model, loader,device):
        """
        For GIN.
        Evaluates the loss of the GNN model on a given dataset.

        This method computes the average negative log-likelihood loss over the dataset.

        Args:
            model (torch.nn.Module): The GNN model to be evaluated.
            
            loader (DataLoader): DataLoader for the evaluation dataset.
            
            device (torch.device): The device on which the evaluation is performed (CPU or GPU).

        Returns:
            float: The average loss of the model on the dataset.
        """
        model.eval()
        loss = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data)
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        return loss / len(loader.dataset)


    def num_graphs(self,data):
        """
        Determines the number of graphs in a batch of data.

        Args:
            data (torch_geometric.data.Data): A batch of graph data.

        Returns:
            int: The number of graphs in the batch.
        """
        if hasattr(data, 'num_graphs'):
            return data.num_graphs
        else:
            return data.x.size(0)

        
    def get_GST_emb(self,data, scattering, device,indices = None,  nonlin=True,is_train=False):
        """
        Generates GST embeddings for specified nodes in the dataset.

        This method applies the Graph Scattering Transform to the node features and returns the embeddings 
        along with labels. It can selectively return embeddings for specific node indices and optionally 
        transform labels for training.

        Args:
            data (torch_geometric.data.Data): The dataset containing node and edge information.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).
            
            indices (list or torch.Tensor, optional): Specific node indices to generate embeddings for. 
                                                     If `None`, embeddings for all nodes are returned. Defaults to `None`.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.
            
            is_train (bool, optional): If `True`, transforms labels for training purposes. Defaults to `False`.

        Returns:
            tuple:
                torch.Tensor: GST embeddings for the specified nodes.
                
                torch.Tensor: Corresponding labels for the nodes.
        """
        data = data.to(device)
        X = scattering(data, nonlin, use_batch=False,return_node = True)
        y = data.y.view(-1)
        if is_train:
            y = F.one_hot(y) * 2 - 1
            y = y.float()
        if indices is not None:
            return X.to(device)[indices], y.to(device)[indices]
        else:
            return X.to(device), y.to(device)

    def get_GST_graph_emb(self,datalist, scattering, device, train_split=False, nonlin=True, batch=-1):
        """
        Generates GST embeddings for a list of graphs.

        This method processes a list of graph data objects, applies the Graph Scattering Transform to compute embeddings, 
        and returns the concatenated embeddings and labels. It supports batch processing for efficiency.

        Args:
            datalist (list of torch_geometric.data.Data): A list of graph data objects.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).
            
            train_split (bool, optional): If `True`, transforms labels for training purposes. Defaults to `False`.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.
            
            batch (int, optional): Batch size for processing graphs. If `> 0`, uses DataLoader for batching; 
                                   otherwise, processes graphs individually. Defaults to `-1`.

        Returns:
            tuple:
                torch.Tensor: Concatenated GST embeddings for all graphs.
                
                torch.Tensor: Concatenated labels for all graphs.
        """
        X = []
        y = []
        if batch > 0:        
            if 'adj' in datalist[0]:
                tmp_loader = DenseLoader(datalist, batch, shuffle=False)
            else:
                tmp_loader = DataLoader(datalist, batch, shuffle=False)
            for data in tmp_loader:
                data = data.to(device)
                embed = scattering(data, nonlin, use_batch=True)
                X.append(embed)
                y.append(data.y.view(-1))
        else:
            for data in datalist:
                data = data.to(device)
                embed = scattering(data, nonlin, use_batch=False)
                X.append(embed)
                y.append(data.y.view(-1))
        X = torch.cat(X, dim=0)
        y = torch.cat(y)
        if train_split:
            y = F.one_hot(y) * 2 - 1
            y = y.float()
        return X.to(device), y.to(device)

    def train_GST(self,logger,args, data, scattering, device,unlearning_nodes,nonmember_id,nonlin=True):
        """
        Trains a classifier using GST (Graph Scattering Transform) embeddings in preparation for the GST unlearning method.

        This method generates GST embeddings for training, validation, and testing datasets, trains a 
        classifier using these embeddings, and evaluates the model's performance. It also computes 
        soft labels for non-member and unlearning nodes to assess the unlearning effectiveness.

        Args:
            logger (logging.Logger): Logger object used to log training progress and metrics.
            
            args (dict): Configuration parameters, including hyperparameters for GST and unlearning.
            
            data (torch_geometric.data.Data): The dataset containing edge and node information.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).
            
            unlearning_nodes (list or torch.Tensor): List of node indices targeted for unlearning.
            
            nonmember_id (list or torch.Tensor): List of node indices not present in the training set.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.

        Returns:
            tuple:
                torch.Tensor: Trained classifier weights.
                
                list: List of durations for data processing and classifier training.
                
                float: Validation accuracy.
                
                float: Test accuracy.
                
                torch.Tensor: Soft labels for non-member nodes.
                
                torch.Tensor: Soft labels for unlearning nodes.
        """
        durations = []
        t_start = time.perf_counter()
        # generate GST embeddings, use batch for computing
        X_train, y_train = self.get_GST_emb(data, scattering, device, data.train_indices,nonlin=nonlin,is_train = True)
        X_val, y_val = self.get_GST_emb(data, scattering, device, data.val_indices,nonlin=nonlin)
        X_test, y_test = self.get_GST_emb(data, scattering, device,data.test_indices, nonlin=nonlin)
        X_all, y_all = self.get_GST_emb(data, scattering, device, nonlin=nonlin)

        t_end = time.perf_counter()
        duration = t_end - t_start
        durations.append(duration)
        print(f'Train Size: {X_train.size(0)}, Val Size: {X_val.size(1)}, Test Size: {X_test.size(0)}, Feature Size: {X_test.size(1)}, Class Size: {int(y_test.max())+1}')
        print(f'GST Data Processing Time: {duration:.3f} s')
        
        # train classifier
        t_start = time.perf_counter()
        b_std = args["std"]
        b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
        w = ovr_lr_optimize(X_train, y_train, args["lam"], None, b=b, num_steps=args["num_epochs"], verbose=False, opt_choice=args["optimizer"], lr=args["opt_lr"], wd=args["opt_decay"])
        t_end = time.perf_counter()
        duration = t_end - t_start
        durations.append(duration)
        
        val_acc, test_acc = 0, 0
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
        logger.info(
            "F1 Score: {:.4f} | GST Training Time: {:.3f} s".format(
                test_acc[0], duration
            )
        )
        
        softlabel_original0 = F.softmax(X_all[nonmember_id].mm(w),dim = 1)
        softlabel_original1 = F.softmax(X_all[unlearning_nodes].mm(w),dim = 1)
        
        return w, durations, val_acc[0], test_acc[0],softlabel_original0,softlabel_original1

    def train_GST_graph(self,logger,args, train_list, test_list,scattering, device):
        """
        Trains the model on graph-level data.

        This method generates GST embeddings for training and testing graph lists, trains a classifier 
        using these embeddings, and evaluates the model's performance on test data.

        Args:
            logger (logging.Logger): Logger object used to log training progress and metrics.
            
            args (dict): Configuration parameters, including hyperparameters for GST and unlearning.
            
            train_list (list of torch_geometric.data.Data): List of training graph data objects.
            
            test_list (list of torch_geometric.data.Data): List of testing graph data objects.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).

        Returns:
            tuple:
                torch.Tensor: Trained classifier weights.
                
                list: List of durations for data processing and classifier training.
                
                float: Validation accuracy (if applicable).
                
                float: Test accuracy.
        """
        durations = []
        t_start = time.perf_counter()
        # generate GST embeddings, use batch for computing
        X_train, y_train = self.get_GST_graph_emb(train_list, scattering, device, train_split=True,batch=128)
        if test_list:
            X_test, y_test = self.get_GST_graph_emb(test_list, scattering, device,batch=128)
        
        t_end = time.perf_counter()
        duration = t_end - t_start
        durations.append(duration)

        self.logger.info(f'GST Data Processing Time: {duration:.3f} s')
        
        # train classifier
        t_start = time.perf_counter()
        b_std = args["std"]
        b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
        w = ovr_lr_optimize(X_train, y_train, args["lam"], None, b=b, num_steps=args["num_epochs"], verbose=False, opt_choice=args["optimizer"], lr=args["opt_lr"], wd=args["opt_decay"])
        t_end = time.perf_counter()
        duration = t_end - t_start
        durations.append(duration)
        
        val_acc, test_acc = 0, 0
        if test_list:
            test_acc,_,f1 = ovr_lr_eval(w, X_test, y_test)
            self.logger.info('Test f1 = {:.4f}'.format(f1) )
        self.logger.info('GST Training Time: {} s'.format(duration))
        
        return w, durations, val_acc, test_acc

    def Unlearn_GST_graph(self,logger,args, scattering, train_list, device, w_approx, budget, val_list=None, test_list=None, nonlin=True, gamma=1/4):
        """
        Performs unlearning on graph-level data.

        This method removes the influence of specific graphs from the trained model by updating the classifier weights.
        It ensures that the model forgets the targeted graphs while maintaining performance on validation and test sets.

        Args:
            logger (logging.Logger): Logger object used to log unlearning progress and metrics.
            
            args (dict): Configuration parameters, including hyperparameters for GST and unlearning.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            train_list (list of torch_geometric.data.Data): List of training graph data objects.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).
            
            w_approx (torch.Tensor): Approximate classifier weights before unlearning.
            
            budget (float): Budget parameter controlling the extent of unlearning.
            
            val_list (list of torch_geometric.data.Data, optional): List of validation graph data objects. Defaults to `None`.
            
            test_list (list of torch_geometric.data.Data, optional): List of testing graph data objects. Defaults to `None`.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.
            
            gamma (float, optional): Scaling factor used in gradient normalization. Defaults to `1/4`.

        Returns:
            tuple:
                float: Total training time for the unlearning process.
                
                int: Number of times the model was retrained during unlearning.
                
                float: Test accuracy after unlearning.
                
                float: Approximated gradient norm.
                
                float: Real gradient norm (placeholder for future implementation).
                
                float: Worst-case gradient norm (placeholder for future implementation).
                list: Updated removal queue after unlearning steps.
        """
        # F for Scattering Transform
        f = np.sqrt(args["L"])
        
        # grad_norm_approx = torch.zeros(args["num_unlearned_nodes"]).float() # Data dependent grad res norm
        grad_norm_approx = 0
        grad_norm_real = 0
        grad_norm_worst = 0
        # grad_norm_real = torch.zeros(args["num_unlearned_nodes"]).float() # true grad res norm
        # grad_norm_worst = torch.zeros(args["num_unlearned_nodes"]).float() # worst case grad res norm
        removal_time = 0
        # removal_time = torch.zeros(args["num_unlearned_nodes"]).float() # record the time of each removal
        acc_removal = torch.zeros((2, args["num_unlearned_nodes"])).float() # record the acc after removal, 0 for val and 1 for test
        num_retrain = 0
        b_std = args["std"]
        
        # if removal_queue is None:
        #     # Remove one node for a graph, generate removal graph_id in advance.
        #     graph_removal_queue = torch.randperm(len(train_list))
        #     removal_queue = []
        # else:
        #     # will use the existing removal_queue
        #     graph_removal_queue = None
            
        # beta in paper
        grad_norm_approx_sum = 0
        training_time = 0
        X_train_old, y_train = self.get_GST_graph_emb(train_list, scattering, device, train_split=True,batch=128)
        if test_list:
            X_test, y_test = self.get_GST_graph_emb(test_list, scattering, device,batch=128)
        num_removes = int(0.5*len(train_list))
        graph_removal_queue = torch.randperm(len(train_list))[:num_removes]
        for i in graph_removal_queue:
            X_train_new = X_train_old.clone().detach()
            data = train_list[i]
            num_nodes_to_remove = int(0.01 * data.num_nodes)  # 计算5%的节点数
            remove_indices = torch.randperm(data.num_nodes)[:num_nodes_to_remove]  # 随机选择节点索引
            train_list[i] = remove_node_from_graph(train_list[i],remove_indices = remove_indices)
        
            t_start = time.perf_counter()
            # generate train embeddings AFTER removal, only for the affect graph
            new_graph_emb, _ = self.get_GST_graph_emb([train_list[i]], scattering, device, train_split=True,batch=-1)
            
            X_train_new[remove_indices,:] = new_graph_emb.view(-1)

            K = get_K_matrix(X_train_new).to(device)
            spec_norm = sqrt_spectral_norm(K)



            # update classifier for each class separately
            for k in range(y_train.size(1)):
                H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train[:, k], args["lam"])
                # grad_i is the difference
                grad_old = lr_grad(w_approx[:, k], X_train_old, y_train[:,k], args["lam"])
                grad_new = lr_grad(w_approx[:, k], X_train_new, y_train[:,k], args["lam"])
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_train_new.mv(Delta)
                # update w here. If beta exceed the budget, w_approx will be retrained
                w_approx[:, k] += Delta
                
                # data dependent norm
                grad_norm_approx += (Delta.norm() * Delta_p.norm() * spec_norm * gamma * f).cpu()
                
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train_new.size(1), y_train.size(1)).float().to(device)
                    w_approx = ovr_lr_optimize(X_train_new, y_train, args["lam"], None, b=b, num_steps=args["unlearning_epochs"], verbose=False,
                                            opt_choice=args["optimizer"], lr=args["opt_lr"], wd=args["opt_decay"])
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx
        
        # record acc each round
        acc_test = ovr_lr_eval(w_approx, X_test, y_test)
        
        removal_time = time.perf_counter() - t_start
        training_time += time.perf_counter() - t_start
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        
        # logger.info('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
        logger.info('Test acc = %.4f' % (acc_test[0]))
        return training_time, num_retrain, acc_test[0], grad_norm_approx, grad_norm_real, grad_norm_worst
        
        
        
    def Unlearn_GST(self,logger,args, scattering, data, device, w_approx, budget,unlearning_nodes,nonmember_id, graph_removal_queue=None,
                    removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
        """
        Performs unlearning on graph-level data using the GST (Graph Scattering Transform) method.

        This method iteratively removes specified nodes from the training graph data, updates the classifier weights 
        to forget the influence of the removed nodes, and evaluates the model's performance after each removal. 
        It ensures that the cumulative gradient norm associated with the unlearning process does not exceed the 
        predefined budget by retraining the model when necessary. Additionally, it computes soft labels for 
        non-member and unlearning nodes to assess the effectiveness of the unlearning process.

        Args:
            logger (logging.Logger): Logger object used to log unlearning progress and metrics.
            
            args (dict): Configuration parameters, including hyperparameters for GST and unlearning. Expected keys include:
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            data (torch_geometric.data.Data): The dataset containing edge and node information, along with train, 
                                            validation, and test indices.
            
            device (torch.device): The computation device (CPU or GPU) on which the model and data are loaded.
            
            w_approx (torch.Tensor): Approximate classifier weights before unlearning. Shape should match the number 
                                    of features and classes (e.g., [num_features, num_classes]).
            
            budget (float): Budget parameter controlling the maximum allowable cumulative gradient norm for unlearning.
            
            unlearning_nodes (list or torch.Tensor): List of node indices targeted for unlearning.
            
            nonmember_id (list or torch.Tensor): List of node indices not present in the training set, used for 
                                                evaluating the impact of unlearning.
            
            graph_removal_queue (torch.Tensor, optional): Predefined queue of graph indices to remove. If `None`, 
                                                        a random queue is generated. Defaults to `None`.
            
            removal_queue (list, optional): Queue of node indices to remove. If `None`, it will be generated based 
                                            on the `graph_removal_queue`. Defaults to `None`.
            
            val_list (list of torch_geometric.data.Data, optional): List of validation graph data objects. 
                                                                Used to evaluate validation accuracy post-removal. 
                                                                Defaults to `None`.
            
            test_list (list of torch_geometric.data.Data, optional): List of testing graph data objects. 
                                                                Used to evaluate test accuracy post-removal. 
                                                                Defaults to `None`.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.
            
            gamma (float, optional): Scaling factor used in gradient normalization. Defaults to `1/4`.

        Returns:
            tuple:
                float: Total training time for the unlearning process (in seconds).
                
                int: Number of times the model was retrained during unlearning.
                
                float: Test accuracy after the unlearning process.
                
                float: Approximated gradient norm associated with the unlearning steps.
                
                float: Real gradient norm (currently a placeholder and remains `0`).
                
                float: Worst-case gradient norm (currently a placeholder and remains `0`).
                
                list: Updated removal queue after unlearning steps, containing tuples of (graph_id, node_id).
                
                torch.Tensor: Soft labels for unlearning nodes after unlearning.
                
                torch.Tensor: Soft labels for non-member nodes after unlearning.
        """
        
        # F for Scattering Transform
        f = np.sqrt(args["L"])
        
        # grad_norm_approx = torch.zeros(args["num_unlearned_nodes"]).float() # Data dependent grad res norm
        grad_norm_approx = 0
        grad_norm_real = 0
        grad_norm_worst = 0
        # grad_norm_real = torch.zeros(args["num_unlearned_nodes"]).float() # true grad res norm
        # grad_norm_worst = torch.zeros(args["num_unlearned_nodes"]).float() # worst case grad res norm
        removal_time = 0
        # removal_time = torch.zeros(args["num_unlearned_nodes"]).float() # record the time of each removal
        acc_removal = torch.zeros((2, args["num_unlearned_nodes"])).float() # record the acc after removal, 0 for val and 1 for test
        num_retrain = 0
        b_std = args["std"]
        
        # if removal_queue is None:
        #     # Remove one node for a graph, generate removal graph_id in advance.
        #     graph_removal_queue = torch.randperm(len(train_list))
        #     removal_queue = []
        # else:
        #     # will use the existing removal_queue
        #     graph_removal_queue = None
            
        # beta in paper
        grad_norm_approx_sum = 0
        training_time = 0
        X_train_old, y_train = self.get_GST_emb(data, scattering, device, data.train_indices,nonlin=nonlin,is_train=True)  # y_train will not change during unlearning process for now
        X_val, y_val = self.get_GST_emb(data, scattering, device, data.val_indices,nonlin=nonlin)
        X_test, y_test = self.get_GST_emb(data, scattering, device, data.test_indices,nonlin=nonlin)
        X_all, y_all = self.get_GST_emb(data, scattering, device, nonlin=nonlin)
        
        # start the removal process
        X_train_new = X_train_old.clone().detach()
        # for i in tqdm(range(args["num_unlearned_nodes"]),desc="Unlearning"):
        #     # we have generated the order of graphs to remove
            #     remove one node based on removal_queue
        data = remove_node_from_graph(data, node_id=removal_queue, removal_queue=removal_queue)  
        
        
        
        t_start = time.perf_counter()
        # generate train embeddings AFTER removal, only for the affect graph
        new_graph_emb, _ = self.get_GST_emb(data, scattering, device, nonlin=nonlin,is_train=True)
        
        X_train_new = new_graph_emb[data.train_indices]

        K = get_K_matrix(X_train_new).to(device)
        spec_norm = sqrt_spectral_norm(K)



        # update classifier for each class separately
        for k in range(y_train.size(1)):
            H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train[:, k], args["lam"])
            # grad_i is the difference
            grad_old = lr_grad(w_approx[:, k], X_train_old, y_train[:,k], args["lam"])
            grad_new = lr_grad(w_approx[:, k], X_train_new, y_train[:,k], args["lam"])
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_train_new.mv(Delta)
            # update w here. If beta exceed the budget, w_approx will be retrained
            w_approx[:, k] += Delta
            
            # data dependent norm
            grad_norm_approx += (Delta.norm() * Delta_p.norm() * spec_norm * gamma * f).cpu()
            
            # decide after all classes
            if grad_norm_approx_sum + grad_norm_approx > budget:
                # retrain the model
                grad_norm_approx_sum = 0
                b = b_std * torch.randn(X_train_new.size(1), y_train.size(1)).float().to(device)
                w_approx = ovr_lr_optimize(X_train_new, y_train, args["lam"], None, b=b, num_steps=args["unlearning_epochs"], verbose=False,
                                        opt_choice=args["optimizer"], lr=args["opt_lr"], wd=args["opt_decay"])
                num_retrain += 1
            else:
                grad_norm_approx_sum += grad_norm_approx
        
        # record acc each round
        acc_val = ovr_lr_eval(w_approx, X_val, y_val)
        acc_test = ovr_lr_eval(w_approx, X_test, y_test)
        
        removal_time = time.perf_counter() - t_start
        training_time += time.perf_counter() - t_start
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        
        # logger.info('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
        logger.info('Val acc = %.4f, Test acc = %.4f' % (acc_val[0], acc_test[0]))
        softlabel_new0 = F.softmax(X_all[nonmember_id].mm(w_approx),dim = 1)
        softlabel_new1 = F.softmax(X_all[unlearning_nodes].mm(w_approx),dim = 1)

        
        return training_time, num_retrain, acc_test[0], grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue,softlabel_new1, softlabel_new0

    def Unlearn_GST_guo(self,logger,args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                        removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
        """
        Performs unlearning on graph-level data using the GST method based on Guo's approach.

        This method iteratively removes graphs from the training set, updates the classifier weights 
        to forget the removed graphs, and evaluates the model's performance after each removal. It 
        ensures that the unlearning budget is not exceeded by retraining the model when necessary.

        Args:
            logger (logging.Logger): Logger object used to log unlearning progress and metrics.
            
            args (dict): Configuration parameters, including hyperparameters for GST and unlearning.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            train_list (list of torch_geometric.data.Data): List of training graph data objects.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).
            
            w_approx (torch.Tensor): Approximate classifier weights before unlearning.
            
            budget (float): Budget parameter controlling the extent of unlearning.
            
            graph_removal_queue (torch.Tensor, optional): Predefined queue of graph indices to remove. 
                                                         If `None`, a random queue is generated. Defaults to `None`.
            
            removal_queue (list, optional): Queue of node indices to remove. If `None`, it will be generated. Defaults to `None`.
            
            val_list (list of torch_geometric.data.Data, optional): List of validation graph data objects. Defaults to `None`.
            
            test_list (list of torch_geometric.data.Data, optional): List of testing graph data objects. Defaults to `None`.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.
            
            gamma (float, optional): Scaling factor used in gradient normalization. Defaults to `1/4`.

        Returns:
            tuple:
                torch.Tensor: Tensor recording the removal time for each unlearning step.
                
                int: Number of times the model was retrained during unlearning.
                
                torch.Tensor: Tensor recording the accuracy after each removal (validation and test).
                
                torch.Tensor: Approximated gradient norm.
                
                torch.Tensor: Real gradient norm.
                
                torch.Tensor: Worst-case gradient norm.
                
                list: Updated removal queue after unlearning steps.
        """
        
        # F for Scattering Transform
        f = np.sqrt(args.L)
        
        grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
        grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
        grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
        
        removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
        acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
        num_retrain = 0
        b_std = args.std
        
        grad_norm_approx_sum = 0
        
        X_train_old, y_train_old = self.get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
        if val_list and test_list:
            X_val, y_val = self.get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
            X_test, y_test = self.get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        
        # start the removal process
        for i in range(args.num_removes):
            # Randomly choose which graph to remove at each round
            remove_idx = np.random.randint(len(train_list))
            X_train_new = X_train_old.clone().detach()[torch.arange(len(train_list)) != remove_idx,:]
            y_train_new = y_train_old.clone().detach()[torch.arange(len(train_list)) != remove_idx,:]
            train_list.pop(remove_idx)
            
            
            t_start = time.perf_counter()
            K = get_K_matrix(X_train_new).to(device)
            spec_norm = sqrt_spectral_norm(K)
            
            # update classifier for each class separately
            for k in range(y_train_new.size(1)):
                H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train_new[:, k], args.lam)
                
                # grad_i is the difference
                grad_old = lr_grad(w_approx[:, k], X_train_old, y_train_old[:,k], args.lam)
                grad_new = lr_grad(w_approx[:, k], X_train_new, y_train_new[:,k], args.lam)
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_train_new.mv(Delta)
                # update w here. If beta exceed the budget, w_approx will be retrained
                w_approx[:, k] += Delta
                
                # data dependent norm
                grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma * f).cpu()
                
            # decide after all classes
            if grad_norm_approx_sum + grad_norm_approx[i] > budget:
                # retrain the model
                grad_norm_approx_sum = 0
                b = b_std * torch.randn(X_train_new.size(1), y_train_new.size(1)).float().to(device)
                w_approx = ovr_lr_optimize(X_train_new, y_train_new, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                        opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                num_retrain += 1
            else:
                grad_norm_approx_sum += grad_norm_approx[i]
            
            
            removal_time[i] = time.perf_counter() - t_start
            # record acc each round
            acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
            acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
            
            
            # Remember to replace X_old with X_new
            X_train_old = X_train_new.clone().detach()
            y_train_old = y_train_new.clone().detach()
            
            if (i+1) % args.rm_disp_step == 0:
                logger.info('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
                logger.info('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
        
        return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue



    def Retrain_GST(self,args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                    removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
        """
        Retrains the model after performing unlearning steps using the GST method.

        This method updates the classifier weights based on the modified training data after unlearning, 
        ensuring that the model retains performance on the remaining data while forgetting the unlearned parts.

        Args:
            args (dict): Configuration parameters, including hyperparameters for GST and unlearning.
            
            scattering (object): The Graph Scattering Transform object used to compute embeddings.
            
            train_list (list of torch_geometric.data.Data): List of training graph data objects.
            
            device (torch.device): The device on which computations are performed (CPU or GPU).
            
            w_approx (torch.Tensor): Approximate classifier weights before retraining.
            
            budget (float): Budget parameter controlling the extent of unlearning.
            
            graph_removal_queue (torch.Tensor, optional): Predefined queue of graph indices to remove. 
                                                         If `None`, a random queue is generated. Defaults to `None`.
            
            removal_queue (list, optional): Queue of node indices to remove. If `None`, it will be generated. Defaults to `None`.
            
            val_list (list of torch_geometric.data.Data, optional): List of validation graph data objects. Defaults to `None`.
            
            test_list (list of torch_geometric.data.Data, optional): List of testing graph data objects. Defaults to `None`.
            
            nonlin (bool, optional): Whether to apply a non-linear transformation during scattering. Defaults to `True`.
            
            gamma (float, optional): Scaling factor used in gradient normalization. Defaults to `1/4`.

        Returns:
            tuple:
                torch.Tensor: Tensor recording the removal time for each unlearning step.
                
                int: Number of times the model was retrained during unlearning.
                
                torch.Tensor: Tensor recording the accuracy after each removal (validation and test).
                
                torch.Tensor: Approximated gradient norm.
                
                torch.Tensor: Real gradient norm.
                
                torch.Tensor: Worst-case gradient norm.
                
                list: Updated removal queue after unlearning steps.
        """
        
        # F for Scattering Transform
        f = np.sqrt(args.L)
        
        grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
        grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
        grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
        
        removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
        acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
        num_retrain = 0
        b_std = args.std
        
        if removal_queue is None:
            # Remove one node for a graph, generate removal graph_id in advance.
            graph_removal_queue = torch.randperm(len(train_list))
            removal_queue = []
        else:
            # will use the existing removal_queue
            graph_removal_queue = None
            
        grad_norm_approx_sum = 0
        
        X_train_old, y_train = self.get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
        if val_list and test_list:
            X_val, y_val = self.get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
            X_test, y_test = self.get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        
        # start the removal process
        for i in range(args.num_removes):
            # we have generated the order of graphs to remove
            if graph_removal_queue is not None:
                # remove one node from graph_removal_queue[i % len(train_list)]
                train_list, removal_queue = remove_node_from_graph(train_list, graph_id=graph_removal_queue[i%len(train_list)], removal_queue=removal_queue)
            else:
                # remove one node based on removal_queue
                train_list = remove_node_from_graph(train_list, graph_id=removal_queue[i][0], node_id=removal_queue[i][1])
            
            X_train_new = X_train_old.clone().detach()
            
            t_start = time.perf_counter()
            # generate train embeddings AFTER removal, only for the affect graph
            new_graph_emb, _ = self.get_GST_emb([train_list[removal_queue[i][0]]], scattering, device, train_split=True, nonlin=nonlin, batch=-1)
            
            X_train_new[removal_queue[i][0], :] = new_graph_emb.view(-1)
            
            b = 0
            w_approx = ovr_lr_optimize(X_train_new, y_train, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                    opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            
            
            removal_time[i] = time.perf_counter() - t_start
            # record acc each round
            if val_list and test_list:
                acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
                acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
            
            
            # Remember to replace X_old with X_new
            X_train_old = X_train_new.clone().detach()
            
            if (i+1) % args.rm_disp_step == 0:
                print('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
                print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
        
        return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue
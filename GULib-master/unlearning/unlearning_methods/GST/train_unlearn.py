from tqdm import tqdm


import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam

from torch_geometric.utils import to_dense_adj, add_self_loops, contains_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

import numpy as np
from unlearning.unlearning_methods.GST.gst_based import *
from utils import *
from utils.utils import *

# import ipdb

def train(model, optimizer, loader, device):
    """
    For GIN.
    """
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader,device):
    """
    For GIN.
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader,device):
    """
    For GIN.
    """
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)

    
def get_GST_emb(data, scattering, device,indices = None,  nonlin=True,is_train=False):
    """
    Input:
        datalist: a list of Data objects.
        scattering: Scattering Transform object.
        batch: if batch > 0, meaning we want to compute the embedding for all graphs, we should use loader; otherwise we should just Data
    """
    data = data.to(device)
    X = scattering(data, nonlin, use_batch=False)
    y = data.y.view(-1)
    if is_train:
        y = F.one_hot(y) * 2 - 1
        y = y.float()
    if indices is not None:
        return X.to(device)[indices], y.to(device)[indices]
    else:
        return X.to(device), y.to(device)


def train_GST(logger,args, data, scattering, device,unlearning_nodes,nonmember_id,nonlin=True):
    """
    train_list, val_list, test_list are not dataloaders and should not be confused with the one used in GIN.
    """
    durations = []
    t_start = time.perf_counter()
    # generate GST embeddings, use batch for computing
    X_train, y_train = get_GST_emb(data, scattering, device, data.train_indices,nonlin=nonlin,is_train = True)
    X_val, y_val = get_GST_emb(data, scattering, device, data.val_indices,nonlin=nonlin)
    X_test, y_test = get_GST_emb(data, scattering, device,data.test_indices, nonlin=nonlin)
    X_all, y_all = get_GST_emb(data, scattering, device, nonlin=nonlin)

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


def Unlearn_GST(logger,args, scattering, data, device, w_approx, budget,unlearning_nodes,nonmember_id, graph_removal_queue=None,
                removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
    """
    
    # F for Scattering Transform
    f = np.sqrt(args["L"])
    
    grad_norm_approx = torch.zeros(args["num_unlearned_nodes"]).float() # Data dependent grad res norm
    grad_norm_real = torch.zeros(args["num_unlearned_nodes"]).float() # true grad res norm
    grad_norm_worst = torch.zeros(args["num_unlearned_nodes"]).float() # worst case grad res norm
    
    removal_time = torch.zeros(args["num_unlearned_nodes"]).float() # record the time of each removal
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
    X_train_old, y_train = get_GST_emb(data, scattering, device, data.train_indices,nonlin=nonlin,is_train=True)  # y_train will not change during unlearning process for now
    X_val, y_val = get_GST_emb(data, scattering, device, data.val_indices,nonlin=nonlin)
    X_test, y_test = get_GST_emb(data, scattering, device, data.test_indices,nonlin=nonlin)
    X_all, y_all = get_GST_emb(data, scattering, device, nonlin=nonlin)
    
    # start the removal process
    X_train_new = X_train_old.clone().detach()
    for i in tqdm(range(args["num_unlearned_nodes"]),desc="Unlearning"):
        # we have generated the order of graphs to remove
            # remove one node based on removal_queue
        data = remove_node_from_graph(data, node_id=removal_queue[i])
        
        
        
        t_start = time.perf_counter()
        # generate train embeddings AFTER removal, only for the affect graph
        new_graph_emb, _ = get_GST_emb(data, scattering, device, nonlin=nonlin,is_train=True)
        
        X_train_new = new_graph_emb[data.train_indices]


        ###############???????????????????##################
        K = get_K_matrix(X_train_new).to(device)
        spec_norm = sqrt_spectral_norm(K)
        ###############???????????????????###################



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
            grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma * f).cpu()
        
        # decide after all classes
        if grad_norm_approx_sum + grad_norm_approx[i] > budget:
            # retrain the model
            grad_norm_approx_sum = 0
            b = b_std * torch.randn(X_train_new.size(1), y_train.size(1)).float().to(device)
            w_approx = ovr_lr_optimize(X_train_new, y_train, args["lam"], None, b=b, num_steps=args["unlearning_epochs"], verbose=False,
                                       opt_choice=args["optimizer"], lr=args["opt_lr"], wd=args["opt_decay"])
            num_retrain += 1
        else:
            grad_norm_approx_sum += grad_norm_approx[i]
        
        # record acc each round
        acc_val = ovr_lr_eval(w_approx, X_val, y_val)
        acc_test = ovr_lr_eval(w_approx, X_test, y_test)
        
        removal_time[i] = time.perf_counter() - t_start
        training_time += time.perf_counter() - t_start
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        
        logger.info('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
        logger.info('Val acc = %.4f, Test acc = %.4f' % (acc_val[0], acc_test[0]))
        softlabel_new0 = F.softmax(X_all[nonmember_id].mm(w_approx),dim = 1)
        softlabel_new1 = F.softmax(X_all[unlearning_nodes].mm(w_approx),dim = 1)

    
    return training_time/args["num_unlearned_nodes"], num_retrain, acc_test[0], grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue,softlabel_new1, softlabel_new0

def Unlearn_GST_guo(logger,args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                    removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
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
    
    X_train_old, y_train_old = get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        X_test, y_test = get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
    
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



def Retrain_GST(args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
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
    
    X_train_old, y_train = get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        X_test, y_test = get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
    
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
        new_graph_emb, _ = get_GST_emb([train_list[removal_queue[i][0]]], scattering, device, train_split=True, nonlin=nonlin, batch=-1)
        
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
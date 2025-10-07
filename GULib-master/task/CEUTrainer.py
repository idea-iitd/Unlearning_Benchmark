from task.BaseTrainer import BaseTrainer
import torch
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import classification_report
class CEUTrainer(BaseTrainer):
    """
    CEUTrainer class for training and evaluating GNN models in preparation for the Certified Unlearning (CEU) method.

    This class extends the BaseTrainer to implement specific training and evaluation routines 
    required for the CEU methodology. It includes methods for training node-level and edge-level 
    tasks, evaluating model performance, and handling model persistence.

    Class Attributes:
        args (dict): Configuration parameters, including model type, dataset specifications, 
                     training hyperparameters, unlearning settings, and other relevant settings.

        logger (logging.Logger): Logger object used to log training progress, metrics, 
                                 and other important information.

        model (torch.nn.Module): The neural network model that will be trained.

        data (dict): A dictionary containing datasets for training, validation, and testing.
                     Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.

        device (torch.device): The computation device (CPU or GPU) on which the model 
                               and data are loaded for training.
    """
    def __init__(self, args, logger, model, data):
        """
        Initializes the CEUTrainer with the provided configuration, logger, model, and data.

        Args:
            args (dict): Configuration parameters, including model type, dataset specifications, 
                        training hyperparameters, unlearning settings, and other relevant settings.
                        
            logger (logging.Logger): Logger object used to log training progress, metrics, 
                                     and other important information.
                        
            model (torch.nn.Module): The neural network model that will be trained.
                        
            data (dict): A dictionary containing datasets for training, validation, and testing.
                         Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.
        """
        super().__init__(args, logger, model, data)
    
    def CEU_train(self,data,eval=True, verbose=True,return_epoch=False):
        """
        Initiates the training process based on the downstream task.

        This method determines the appropriate training routine (node-level or edge-level) 
        based on the specified downstream task and delegates the training process accordingly.

        Args:
            data (dict): A dictionary containing datasets for training, validation, and testing.
                        Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.
                
            eval (bool, optional): Whether to evaluate the model after training. Defaults to True.
                
            verbose (bool, optional): Whether to display training progress. Defaults to True.
                
            return_epoch (bool, optional): Whether to return the number of epochs trained. 
                                        Defaults to False.
        
        Returns:
            torch.nn.Module or tuple: The trained model. If `return_epoch` is True, returns a tuple 
                                    containing the trained model and the number of epochs.
    """
        # if self.args["downstream_task"]=="node":
        return self.CEU_train_node(data,eval,verbose,return_epoch)
        # elif self.args["downstream_task"]=="edge":
        #     return self.CEU_train_edge(data,eval,verbose,return_epoch)
        
    def CEU_train_node(self,data,eval=True, verbose=True,return_epoch=False):
        """
        Trains the model for node-level tasks.

        This method handles the training loop for node classification tasks, including loss 
        computation, backpropagation, optimizer steps, evaluation, and model saving based 
        on validation loss improvements and early stopping criteria.

        Args:
            data (dict): A dictionary containing datasets for training, validation, and testing.
                         Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.
                
            eval (bool, optional): Whether to evaluate the model after training. Defaults to True.
                
            verbose (bool, optional): Whether to display training progress. Defaults to True.
                
            return_epoch (bool, optional): Whether to return the number of epochs trained. 
                                           Defaults to False.
        
        Returns:
            torch.nn.Module or tuple: The trained model. If `return_epoch` is True, returns a tuple 
                                      containing the trained model and the number of epochs.
        """
        if verbose:
            t0 = time.time()
            print(f'Start to train a {self.args["model_name"]} model...')

        model = self.model.to(self.device)
        train_loader = DataLoader(data['train_set'], batch_size=self.args["train_batch"], shuffle=True)
        valid_loader = DataLoader(data['valid_set'], batch_size=self.args["test_batch"])
        edge_index = torch.tensor(data['edges'], device=self.device).t()
        num_epochs = self.args['num_epochs']
        lr = self.args["unlearn_lr"]
        l2 = self.args["l2"]
        patience = self.args["patience"]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        best_valid_loss = 999999.
        best_epoch = 0
        trail_count = 0

        for e in range(1, num_epochs + 1):

            train_losses = []

            model.train()
            iterator = tqdm(train_loader, f'  Epoch {e}') if verbose else train_loader
            for nodes, labels in iterator:
                model.zero_grad()

                nodes = nodes.to(self.device)
                labels = labels.to(self.device)

                y_hat = model(nodes, edge_index)
                loss = model.loss(y_hat, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.cpu().item())

            train_loss = np.mean(train_losses)

            valid_losses = []

            model.eval()
            with torch.no_grad():
                for nodes, labels in valid_loader:
                    nodes = nodes.to(self.device)
                    labels = labels.to(self.device)

                    y_hat = model(nodes, edge_index)
                    loss = model.loss(y_hat, labels)

                    valid_losses.append(loss.cpu().item())

            valid_loss = np.mean(valid_losses)

            if verbose:
                print(f'  Epoch {e}, training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}.')

            if self.args["early_stop"]:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    trail_count = 0
                    best_epoch = e
                    path = os.path.join('./data/CEU/checkpoint', 'tmp')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(model.state_dict(), path + f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt')
                else:
                    trail_count += 1
                    if trail_count > patience:
                        if verbose:
                            print(
                                f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                        break
            else:
                path = os.path.join('./data/CEU/checkpoint','tmp')
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(),path+f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt')
        path = os.path.join('./data/CEU/checkpoint', 'tmp')
        model.load_state_dict(torch.load(path + f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt'))

        if eval:
            self.CEU_evaluate()

        # os.remove(os.path.join('./checkpoint', 'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))

        if verbose:
            print(f'{self.args["base_model"]} model training finished. Duration:', int(time.time() - t0))

        if return_epoch:
            return model, num_epochs
        else:
            return model

    def CEU_train_edge(self,data,eval=True, verbose=True,return_epoch=False):
        """
        Trains the model for edge-level tasks.

        This method handles the training loop for edge classification tasks, including loss 
        computation, backpropagation, optimizer steps, evaluation, and model saving based 
        on validation loss improvements and early stopping criteria.

        Args:
            data (dict): A dictionary containing datasets for training, validation, and testing.
                         Expected keys include 'train_set', 'valid_set', 'test_set', and 'edges'.
                
            eval (bool, optional): Whether to evaluate the model after training. Defaults to True.
                
            verbose (bool, optional): Whether to display training progress. Defaults to True.
                
            return_epoch (bool, optional): Whether to return the number of epochs trained. 
                                           Defaults to False.
        
        Returns:
            torch.nn.Module or tuple: The trained model. If `return_epoch` is True, returns a tuple 
                                      containing the trained model and the number of epochs.
        """
        if verbose:
            t0 = time.time()
            print(f'Start to train a {self.args["model_name"]} model...')

        model = self.model.to(self.device)
        train_loader = DataLoader(data['train_set'], batch_size=self.args["train_batch"], shuffle=True)
        valid_loader = DataLoader(data['valid_set'], batch_size=self.args["test_batch"])
        edge_index = torch.tensor(data['edges'], device=self.device).t()
        
        num_epochs = self.args['num_epochs']
        lr = self.args["unlearn_lr"]
        l2 = self.args["l2"]
        patience = self.args["patience"]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        best_valid_loss = 999999.
        best_epoch = 0
        trail_count = 0

        for e in range(1, num_epochs + 1):

            train_losses = []

            model.train()
            iterator = tqdm(train_loader, f'  Epoch {e}') if verbose else train_loader
            for nodes, labels in iterator:
                model.zero_grad()

                nodes = nodes.to(self.device)
                labels = labels.to(self.device)

                y_hat = model(nodes, edge_index)
                
                loss = model.loss(y_hat, labels)
                
                loss.backward()
                optimizer.step()

                train_losses.append(loss.cpu().item())

            train_loss = np.mean(train_losses)

            valid_losses = []

            model.eval()
            with torch.no_grad():
                for nodes, labels in valid_loader:
                    nodes = nodes.to(self.device)
                    labels = labels.to(self.device)

                    y_hat = model(nodes, edge_index)
                    loss = model.loss(y_hat, labels)

                    valid_losses.append(loss.cpu().item())

            valid_loss = np.mean(valid_losses)

            if verbose:
                print(f'  Epoch {e}, training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}.')

            if self.args["early_stop"]:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    trail_count = 0
                    best_epoch = e
                    path = os.path.join('./data/CEU/checkpoint', 'tmp')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(model.state_dict(), path + f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt')
                else:
                    trail_count += 1
                    if trail_count > patience:
                        if verbose:
                            print(
                                f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                        break
            else:
                path = os.path.join('./data/CEU/checkpoint','tmp')
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(),path+f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt')
        path = os.path.join('./data/CEU/checkpoint', 'tmp')
        model.load_state_dict(torch.load(path + f'/{self.args["base_model"]}_{self.args["dataset_name"]}_{self.args["cuda"]}_best.pt'))

        if eval:
            self.CEU_evaluate()

        # os.remove(os.path.join('./checkpoint', 'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))

        if verbose:
            print(f'{self.args["base_model"]} model training finished. Duration:', int(time.time() - t0))

        if return_epoch:
            return model, num_epochs
        else:
            return model



    @torch.no_grad()
    def CEU_test(self,model, test_loader, edge_index):
        """
        Tests the model on the test set and computes classification metrics.

        This method evaluates the model's performance on the test set by computing predictions 
        and calculating metrics such as F1 score. It also computes the average test loss.

        Args:
            model (torch.nn.Module): The trained model to be tested.

            test_loader (DataLoader): DataLoader for the test dataset.

            edge_index (torch.LongTensor): Tensor containing edge indices for the graph.
        
        Returns:
            tuple: A tuple containing the classification report as a dictionary and the average test loss.
        """
        y_preds = []
        y_trues = []
        test_loss = []
        model.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes, labels = nodes.to(self.device), labels.to(self.device)
                y_hat = model(nodes, edge_index)
                test_loss.append(model.loss(y_hat, labels).cpu().item())
                y_pred = torch.argmax(y_hat, dim=1)
                y_preds.extend(y_pred.cpu().tolist())
                y_trues.extend(labels.cpu().tolist())
        # del model
        # torch.cuda.empty_cache()
        res = classification_report(y_trues, y_preds, digits=4, output_dict=True)
        return res, np.mean(test_loss)

    @torch.no_grad()
    def CEU_evaluate(self):
        """
        Evaluates the model's performance on the test set and prints the classification report.

        This method computes predictions on the test set and generates a classification report, 
        including metrics like precision, recall, and F1 score.
        """
        test_loader = DataLoader(self.data['test_set'], batch_size=self.args['test_batch'], shuffle=False)
        edge_index = torch.tensor(self.data['edges'], device=self.device).t()
        y_preds = []
        y_true = []

        self.model.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes = nodes.to(self.device)
                labels = labels.to(self.device)

                y_hat = self.model(nodes, edge_index)
                y_pred = torch.argmax(y_hat, dim=1)

                y_preds.extend(y_pred.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

        results = classification_report(y_true, y_preds, digits=4)
        print('  Result:')
        print(results)
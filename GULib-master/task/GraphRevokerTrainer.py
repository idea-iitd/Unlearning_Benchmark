from task.BaseTrainer import BaseTrainer
import torch.nn.functional as F
import torch
class GraphRevokerTrainer(BaseTrainer):
    """
    GraphRevokerTrainer class for training and evaluating Graph Neural Networks (GNNs) in preparation for applying the GraphRevoker unlearning method.

    This class extends the `BaseTrainer` to implement specific training and evaluation routines 
    required for various graph unlearning methodologies. It includes methods for generating 
    posteriors, performing inference, and preparing data for training and evaluation. 

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
        Initializes the GraphRevokerTrainer with the provided configuration, logger, model, and data.

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

    def posterior(self,return_features=False,mia=False):
        """
        Generates posterior probabilities for nodes or edges based on the trained model.

        This method computes the posterior probabilities by performing a forward pass through the model.
        It can optionally return node features and support member inference attacks (MIA) evaluation.

        Args:
            return_features (bool, optional): If set to `True`, the method returns both 
                                              node embeddings and additional features. Defaults to `False`.
                
            mia (bool, optional): If set to `True`, the method computes posteriors suitable for 
                                   evaluating membership inference attacks. Defaults to `False`.
    
        Returns:
            torch.Tensor or tuple:
                - If `mia` is `True`: Returns the log-softmax probabilities for the specified test mask.
                - If `return_features` is `True`: Returns a tuple containing node embeddings and additional features.
                - Otherwise: Returns node embeddings corresponding to the test mask.
        """
        # self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()
        if mia:
            if self.args["base_model"] == "SIGN":
                posteriors = F.log_softmax(self.model(self.data.xs))
            else:
                posteriors = F.log_softmax(self.model(self.data.x,self.data.edge_index))
            for _, mask in self.data('test_mask'):
                posteriors = F.log_softmax(posteriors[mask.cpu()], dim=-1)

            return posteriors.detach()
        else:
            z, f = self._inference()

        if return_features:
        #     return z[self.data_full.test_mask], f[self.data_full.test_mask]
        # return z[self.data_full.test_mask, :]
            return z[self.data.test_mask], f[self.data.test_mask]
        return z[self.data.test_mask, :]
    
    @torch.no_grad()
    def _inference(self, no_test_edges=False):
        """
        Performs inference to obtain node embeddings and features.

        This method executes a forward pass through the model to compute the log-softmax probabilities 
        and extract node features. It ensures that the model is in evaluation mode and processes the 
        data on the appropriate device.

        Args:
            no_test_edges (bool, optional): If set to `True`, excludes test edges during inference. 
                                            Defaults to `False`.
        
        Returns:
            tuple:
                - torch.Tensor: Log-softmax probabilities for each node.
                - torch.Tensor: Extracted node features.
        """


        self.model.eval()
        self.model = self.model.to(self.device)
    
        self.data = self.data.to(self.device) 
        
        z, feat = self.model(self.data.x, self.data.edge_index, return_feature=True)

        return F.log_softmax(z,dim=1), feat


    def prepare_data(self, input_data):
        """
        Prepares and splits the input data for training and evaluation.

        This method clones the input data, separates training edges, and initializes data loaders 
        based on the configuration. It ensures that the training edges are appropriately assigned 
        for the unlearning process.

        Args:
            input_data (torch_geometric.data.Data): The original dataset containing all edges and nodes.
        
        Returns:
            None
        """
        data_full = input_data.clone()
        data = input_data.clone()
        
        data.edge_index = data.edge_index_train
        
        data.edge_index_train = None
        data_full.edge_index_train = None

        self.data.edge_index = input_data.edge_index_train
        self.data.train_edge_index = input_data.edge_index_train
        self.data.edge_index_train = None
        self.data_full = data_full

        
        if self.args['is_use_train_batch']:
            self.gen_train_loader()
        if self.args['is_use_test_batch']:
            self.gen_test_loader()
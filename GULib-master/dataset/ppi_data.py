class ppi_data:
    def __init__(self,all_data) :
        """
        A class for handling and processing Protein-Protein Interaction (PPI) dataset.

        This class stores and processes data related to PPI tasks, including training and
        testing data, features, and labels.

        Class Attributes:

        all_data (Any): The raw input data used for training and testing.
        
        train_out (Optional[Tensor]): The training data features, if available.
        
        train_y (Optional[Tensor]): The training labels, if available.
        
        test_out (Optional[Tensor]): The testing data features, if available.
        
        test_y (Optional[Tensor]): The testing labels, if available.
        
        num_features (int): The number of features in the dataset (default: 50).
        
        num_classes (int): The number of classes in the dataset (default: 121).
        """
        self.all_data = all_data
        self.train_out = None
        self.train_y = None
        self.test_out = None
        self.test_y = None
        self.num_features = 50
        self.num_classes = 121

import numpy as np
import networkx as nx
import time
import torch
import random
from utils import dataset_utils
from utils import utils
from utils.dataset_utils import *
# from memory_profiler import profile
BLUE_COLOR = "\033[34m"
RESET_COLOR = "\033[0m"

class Shard_based_pipeline:
    """
    Base class for implementing a shard-based pipeline. This class defines the essential 
    structure and methods required for shard-based operations such as partitioning, training, 
    and unlearning.

    Class Attributes:
        args (dict): A dictionary containing configuration arguments for the pipeline.

        logger (Logger): A logger object for logging information during pipeline execution.

        data (Dataset): A dataset object that provides the data for the pipeline.

        model_zoo (ModelZoo): A model zoo that provides models and related functionality.

        run (int): The current run index.

        num_shards (int): The number of shards (partitions) in the pipeline.

        poison_f1 (np.ndarray): Array to store the poison F1 score for each run.

        average_f1 (np.ndarray): Array to store the average F1 score for each run.

        average_auc (np.ndarray): Array to store the average AUC score for each run.

        average_acc (np.ndarray): Array to store the average accuracy score for graph classification.

        avg_partition_time (np.ndarray): Array to store the average partition time for each run.

        avg_training_time (np.ndarray): Array to store the average training time for each run.

        avg_unlearning_time (np.ndarray): Array to store the average unlearning time for each run.
    """
    def __init__(self,args,logger,model_zoo):
        """
        Initializes the Shard_based_pipeline with the provided arguments, logger, and model zoo.

        Args:

            args (dict): A dictionary containing the configuration parameters. It must include keys like "num_runs" and "num_shards".
            logger (Logger): A logger object used to log runtime information.
            model_zoo (ModelZoo): An object that provides access to models and datasets.
        """
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.run = 0
        self.num_shards = self.args["num_shards"]
        self.poison_f1 = np.zeros(self.args["num_runs"])
        self.average_f1 = np.zeros(self.args["num_runs"])
        self.average_auc = np.zeros(self.args["num_runs"])
        self.average_acc = np.zeros(self.args["num_runs"])
        self.avg_partition_time = np.zeros(self.args["num_runs"])
        self.avg_training_time = np.zeros(self.args["num_runs"])
        self.avg_unlearning_time = np.zeros(self.args["num_runs"])
        pass
    
    # # @profile
    # def run_exp_mem(self):
    #     """
    #     Executes the experimental pipeline while profiling memory usage.

    #     During each run, this method:

    #     1. Seeds the random number generator for reproducibility.
    #     2. Executes the partitioning step.
    #     3. Trains the shard-based models.
    #     4. Performs the unlearning step.

    #     """
    #     for self.run in range(self.args["num_runs"]):
    #         self.exp_partition()
    #         self.exp_train()
    #         self.exp_unlearn()
    #         self.logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024/1024}MB")
    #         self.logger.info(f"Max Cached: {torch.cuda.max_memory_reserved()/1024/1024}MB")
            

    def run_exp_mem(self):
        """
        Executes the experimental pipeline while profiling memory usage.

        During each run, this method:

        1. Seeds the random number generator for reproducibility.
        2. Executes the partitioning step.
        3. Trains the shard-based models.
        4. Performs the unlearning step.

        """

        import time, tracemalloc, numpy as np, torch
        
        funcs = [
            ("exp_partition", self.exp_partition),
            ("exp_train", self.exp_train),
            ("exp_unlearn_request",self.generate_requests),
            ("exp_unlearn", self.exp_unlearn),
        ]

        results = {name: {"times": [], "gpu": [], "py": []} for name, _ in funcs}
        use_cuda = torch.cuda.is_available()

        for run_idx in range(self.args["num_runs"]):
            self.run = run_idx
            for name, func in funcs:
                if use_cuda:
                    torch.cuda.reset_peak_memory_stats()
                tracemalloc.start()
                t0 = time.perf_counter()
                func()
                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                py_peak = peak / 1024 / 1024
                gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if use_cuda else 0
                results[name]["times"].append(elapsed)
                results[name]["py"].append(py_peak)
                results[name]["gpu"].append(gpu_peak)

        with open("efficiency_stats.txt", "a") as f:
            f.write("============== Efficiency Statistics ============\n")
            f.write(f"Runs: {self.args['num_runs']}, Dataset: {self.args['dataset_name']},Technique: {self.args['unlearning_methods']}\n")
            for name, vals in results.items():
                t, g, p = map(np.array, (vals["times"], vals["gpu"], vals["py"]))
                f.write(f"{name}:\n")
                f.write(f"  Time (s): mean={t.mean():.6f}, std={t.std():.6f}\n")
                f.write(f"  GPU Peak (MB): mean={g.mean():.2f}, std={g.std():.2f}\n")
                f.write(f"  Python Heap Peak (MB): mean={p.mean():.2f}, std={p.std():.2f}\n\n")
            f.write(f"Crucial Unlearning time (s): mean={np.mean(self.avg_unlearning_time):.6f}, std={np.std(self.avg_unlearning_time):.6f}\n")
    
    def run_exp(self):
        """
        Executes the experimental pipeline for multiple runs.

        During each run, this method:

        1. Executes the partitioning step.
        2. Trains the shard-based models.
        3. Performs the unlearning step.

        At the end of all runs, logs performance metrics including:

        - Poison F1 Score
        - Unlearn F1 Score
        - Unlearn Accuracy for Graph Classification
        - Average AUC Score
        - Average Partition Time
        - Average Training Time
        - Average Unlearning Time

        """
        for self.run in range(self.args["num_runs"]):
            
            self.exp_partition()
            self.exp_train()
            self.generate_requests()
            self.exp_unlearn()

        # ---- Store AUC and GOLD AUC results in file ----
        # breakpoint()
        with open("/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/MIA_stats.txt", "a") as f:
            f.write(
                "{} Average MIA {} Score: {:.4f} ± {:.4f}\n".format(
                    self.args["dataset_name"],
                    self.args["unlearning_methods"],
                    np.mean(self.average_auc),
                    np.std(self.average_auc)
                )
            )
            
        self.logger.info(
            "{}Performance Metrics:\n"
            " - Poison F1 Score: {:.4f} ± {:.4f}\n"
            " - Unlearn F1 Score: {:.4f} ± {:.4f}\n"
            " - Unlearn Acc for Graph Cls: {:.4f} ± {:.4f}\n"
            " - Average AUC Score: {:.4f} ± {:.4f}\n"
            " - Average Partition Time: {:.4f} ± {:.4f} seconds\n"
            " - Average Training Time: {:.4f} ± {:.4f} seconds\n"
            " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
                BLUE_COLOR,
                np.mean(self.poison_f1), np.std(self.poison_f1),
                np.mean(self.average_f1), np.std(self.average_f1),
                np.mean(self.average_acc), np.std(self.average_auc),
                np.mean(self.average_auc), np.std(self.average_auc),
                np.mean(self.avg_partition_time), np.std(self.avg_partition_time),
                np.mean(self.avg_training_time), np.std(self.avg_training_time),
                np.mean(self.avg_unlearning_time), np.std(self.avg_unlearning_time),
                RESET_COLOR
            )
        )
        self.logger.info(self.avg_partition_time)
        self.logger.info(self.avg_unlearning_time)
            
    def exp_partition(self):
        """
        Basic Data Partition -> *num_shard* shards
        
        Applied for GraphEraser, GraphEevoker
        
        GUIDE with different method.
        """
        self.args["exp"] = "partition"
        self.gen_train_graph()
        self.graph_partition()
        self.generate_shard_data()
        pass
    
    def exp_train(self):
        """
        Basic Unlearn Training
        
        Get original model
        """
        self.args["exp"] = "unlearning"
        self.load_data()
        self.determine_target_model()
        self.train_shard_model()
        self.aggregate_shard_model()
        
        pass
    def exp_unlearn(self):
        """
        Evaluate Unlearning and Attack_Unlearning
        """
        self.args["exp"] = "attack_unlearning"
        self.unlearn()
        if self.args["attack"]:
            self.attack_unlearning()
            
            
    def gen_train_graph(self):
        pass

    def graph_partition(self):
        pass
        
    def generate_shard_data(self):
        pass
            
    def load_data(self):
        pass
        
    def determine_target_model(self):
        pass

    def train_shard_model(self):
        pass
    
    def aggregate_shard_model(self):
        pass
    
    def generate_requests(self):
        pass
    
    def unlearn(self):
        pass
    
    def attack_unlearning(self):
        pass
    
    
# def seed_everything(seed_value):
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     os.environ['PYTHONHASHSEED'] = str(seed_value)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = True
import os
import random
# import optuna
import numpy as np
import torch
from model.model_zoo import model_zoo
from dataset.original_dataset import original_dataset
from parameter_parser import parameter_parser
from unlearning.unlearning_methods.CEU.ceu import ceu
from utils.logger import create_logger
from task.node_classification import NodeClassifier
from unlearning.unlearning_methods.GNNDelete.gnndelete import gnndelete
from utils.dataset_utils import process_data,save_data
from attack.Attack_methods.GraphEraser_MIA import GraphEraser_Attack
from attack.Attack_methods.GUIDE_MIA import GUIDE_MIA
from unlearning.unlearning_methods.GraphEraser.grapheraser import grapheraser
from unlearning.unlearning_methods.GUIDE.guide import guide
from unlearning.unlearning_methods.GIF.gif import gif
from unlearning.unlearning_methods.CGU.cgu import cgu
from unlearning.unlearning_methods.GST.gst_based import gst
from unlearning.unlearning_methods.SGU import sgu
from unlearning.unlearning_methods.Projector.projector import projector
from unlearning_manager import UnlearningManager
from config import unlearning_path
import sys 
import os
# import copy
# import optuna


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
# def objective(trial):
#     para1 = trial.suggest_float('para1', 0.5, 1.5)
#     para2 = trial.suggest_float('para2', 0.0001, 0.01)
#     para3 = trial.suggest_int('para3', 20, 500)
#     para4 = trial.suggest_int('para4', 5, 50)
#     para5 = trial.suggest_int('para5', 2, 50)
#     args["para1"] = para1
#     args["para2"] = para2
#     args["para3"] = para3
#     args["para4"] = para4
#     args["para5"] = para5
#     model_zoo_copy = copy.deepcopy(model_zoo)
#     SGU_instance = sgu(args,logger,model_zoo_copy)
#     # SGU_instance.run = np.random.randint(0,5) 
#     SGU_instance.run_exp()
#     return SGU_instance.best 

#     # return 10*abs(SGU_instance.final_auc-0.5)

# def run_optuna(args,logger):
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=100)
#     best_params = study.best_params
#     args["para1"] = best_params["para1"]
#     args["para2"] = best_params["parag2"]
#     args["para3"] = best_params["para3"]
#     args["para4"] = best_params["para4"]
#     args["para5"] = best_params["para5"]
#     args["parameter_task"] = "normal"
#     model_zoo_copy = copy.deepcopy(model_zoo)
#     SGU_instance = sgu(args,logger,model_zoo_copy)
#     SGU_instance.run_exp()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    sys.path.append(base_dir)

    args = parameter_parser()
    
    logger = create_logger(args)
    seed_everything(2024)
    torch.cuda.set_device(args['cuda'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
 
    #dataset
    original_data = original_dataset(args,logger)
    print("data done")
    # breakpoint()
    data,dataset = original_data.load_data()
    print("done done 2")
    # 使用 assert 直接检查 args 中的参数
    data = process_data(logger,data,args)
    print("data processsed")
    #model
    model_zoo = model_zoo(args,data)
    model = model_zoo.model
    print("model done")
    if args["base_model"] not in ["GST","Projector"]:
        logger.log_model_info(model)   
    
    
    manager = UnlearningManager(args, original_data, data, logger, model_zoo, dataset)
    GU_method = manager.get_method()
    if args["cal_mem"]:
        # import time
        # args["num_runs"] = 1
        # # Reset peak stats
        # torch.cuda.reset_peak_memory_stats(device)
        # start_time = time.time()
        GU_method.run_exp_mem()
        # end_time = time.time()
        # total_time = end_time - start_time
        # peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        # peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)   # MB
        # Save to file
        # out_file = "running_stats.txt"
        # with open(out_file, "a") as f:
        #     f.write(f"Run on dataset={args['dataset_name']}, method={args['unlearning_methods']}, base model={args['base_model']}\n")
        #     f.write(f"Time Taken: {total_time:.2f} sec\n")
        #     f.write(f"Peak Allocated Memory: {peak_allocated:.2f} MB\n")
        #     f.write(f"Peak Reserved (Cached) Memory: {peak_reserved:.2f} MB\n")
        #     f.write("-" * 40 + "\n")

    else:
        GU_method.run_exp()
 


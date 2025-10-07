

import os
import torch
from opengu.unlearning_manager import UnlearningManager
from opengu.utils.logger import create_logger
from opengu.model.model_zoo import model_zoo
from opengu.dataset.original_dataset import original_dataset
from opengu.parameter_parser import parameter_parser
from opengu.utils.dataset_utils import process_data

args = parameter_parser()
logger = create_logger(args)
torch.cuda.set_device(args['cuda'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])

#dataset
original_data = original_dataset(args,logger)
data,dataset = original_data.load_data()
data = process_data(logger,data,args)

#model
model_zoo = model_zoo(args,data)
model = model_zoo.model
if args["base_model"] not in ["GST","Projector"]:
    logger.log_model_info(model)

#example for GraphEraser method
args["unlearning_methods"] = "GraphEraser"

manager = UnlearningManager(args, original_data, data, logger, model_zoo, dataset)
GU_method = manager.get_method()
GU_method.run_exp()
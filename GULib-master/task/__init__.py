from task.BaseTrainer import BaseTrainer
from task.GUIDETrainer import GUIDETrainer
from task.GNNDeleteTrainer import GNNDeleteTrainer
from task.GIFTrainer import GIFTrainer
from task.SGUTrainer import SGUTrainer
from task.GSTTrainer import GSTTrainer
from task.MEGUTrainer import MEGUTrainer
from task.GUKDTrainer import GUKDTrainer
from task.D2DGNTrainer import D2DGNTrainer
from task.IDEATrainer import IDEATrainer
from task.GraphRevokerTrainer import GraphRevokerTrainer
from task.CEUTrainer import CEUTrainer
from task.edge_prediction import EdgePredictor
from task.node_classification import NodeClassifier
from task.UtUTrainer import UtUTrainer
from task.GraphRevokerTrainer import GraphRevokerTrainer
trainer_mapping = {
    'BaseTrainer': BaseTrainer,
    'GUIDETrainer': GUIDETrainer,
    'GNNDeleteTrainer': GNNDeleteTrainer,
    'GIFTrainer': GIFTrainer,
    'SGUTrainer': SGUTrainer,
    'GSTTrainer': GSTTrainer,
    'MEGUTrainer': MEGUTrainer,
    'GUKDTrainer':GUKDTrainer,
    'D2DGNTrainer':D2DGNTrainer,
    'IDEATrainer':IDEATrainer,
    'GraphRevokerTrainer':GraphRevokerTrainer,
    'CEUTrainer':CEUTrainer
}


def get_trainer(args, logger, model, data):
    return trainer_mapping[args["unlearn_trainer"]](args, logger, model, data)
    

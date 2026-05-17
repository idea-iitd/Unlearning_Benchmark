from task.BaseTrainer import BaseTrainer
from task.GUIDETrainer import GUIDETrainer
from task.GNNDeleteTrainer import GNNDeleteTrainer
from task.GIFTrainer import GIFTrainer
from task.GSTTrainer import GSTTrainer
from task.MEGUTrainer import MEGUTrainer
from task.IDEATrainer import IDEATrainer
from task.CognacTrainer import CognacTrainer
from task.ETRTrainer import ETRTrainer
from task.edge_prediction import EdgePredictor
from task.node_classification import NodeClassifier

trainer_mapping = {
    'BaseTrainer': BaseTrainer,
    'GUIDETrainer': GUIDETrainer,
    'GNNDeleteTrainer': GNNDeleteTrainer,
    'GIFTrainer': GIFTrainer,
    'GSTTrainer': GSTTrainer,
    'MEGUTrainer': MEGUTrainer,
    'IDEATrainer': IDEATrainer,
    'CognacTrainer': CognacTrainer,
    'ETRTrainer': ETRTrainer,
}


def get_trainer(args, logger, model, data):
    return trainer_mapping[args["unlearn_trainer"]](args, logger, model, data)
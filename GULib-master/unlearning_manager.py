import numpy as np
import copy
from unlearning.unlearning_methods.GraphEraser.grapheraser import grapheraser
from unlearning.unlearning_methods.GUIDE.guide import guide
from unlearning.unlearning_methods.GIF.gif import gif
from unlearning.unlearning_methods.CGU.cgu import cgu
from unlearning.unlearning_methods.Cognac.cognac import cognac
from unlearning.unlearning_methods.GST.gst_based import gst
from unlearning.unlearning_methods.Projector.projector import projector
from unlearning.unlearning_methods.GNNDelete.gnndelete import gnndelete
from unlearning.unlearning_methods.MEGU.megu import megu
from unlearning.unlearning_methods.MEGU.gold import gold
from unlearning.unlearning_methods.IDEA.idea import idea
from unlearning.unlearning_methods.ScaleGUN.scalegun import scalegun
from unlearning.unlearning_methods.ETR.etr import etr
from utils.dataset_utils import process_data, save_data
from attack.Attack_methods.GraphEraser_MIA import GraphEraser_Attack
from attack.Attack_methods.GUIDE_MIA import GUIDE_MIA

method_map = {
    "GraphEraser": grapheraser,
    "GNNDelete": gnndelete,
    "CGU": cgu,
    "COGNAC": cognac,
    "GIF": gif,
    "GUIDE": guide,
    "GST": gst,
    "Projector": projector,
    "MEGU": megu,
    "GOLD": gold,
    "GraphRevoker": grapheraser,
    "IDEA": idea,
    "ScaleGUN": scalegun,
    "ETR": etr,
}


class UnlearningManager:
    def __init__(self, args, original_data, data, logger, model_zoo, dataset=None):
        self.args = args
        self.original_data = original_data
        self.data = data
        self.logger = logger
        self.model_zoo = model_zoo
        self.dataset = dataset

    def get_method(self):
        return method_map[self.args["unlearning_methods"]](self.args, self.logger, self.model_zoo)
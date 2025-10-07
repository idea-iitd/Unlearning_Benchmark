import numpy as np
import copy
from unlearning.unlearning_methods.CEU.ceu import ceu
from unlearning.unlearning_methods.GraphEraser.grapheraser import grapheraser
from unlearning.unlearning_methods.GUIDE.guide import guide
from unlearning.unlearning_methods.GIF.gif import gif
from unlearning.unlearning_methods.CGU.cgu import cgu
from unlearning.unlearning_methods.SGU.sgu import sgu
from unlearning.unlearning_methods.GST.gst_based import gst
from unlearning.unlearning_methods.Projector.projector import projector
from unlearning.unlearning_methods.GNNDelete.gnndelete import gnndelete
from unlearning.unlearning_methods.MEGU.megu import megu
from unlearning.unlearning_methods.UTU.utu import utu
from unlearning.unlearning_methods.GraphRevoker.graphrevoker import graphrevoker
from unlearning.unlearning_methods.GUKD.gukd import gukd
from unlearning.unlearning_methods.D2DGN.d2dgn import d2dgn
from unlearning.unlearning_methods.IDEA.idea import idea
from unlearning.unlearning_methods.ScaleGUN.scalegun import scalegun
from utils.dataset_utils import process_data,save_data
from attack.Attack_methods.GraphEraser_MIA import GraphEraser_Attack
from attack.Attack_methods.GUIDE_MIA import GUIDE_MIA
from attack.MIA_attack import GCNShadowModel
from attack.MIA_attack import train_shadow_model
from attack.MIA_attack import generate_shadow_model_output
from attack.MIA_attack import train_attack_model
# from memory_profiler import profile
# import optuna

# 方法名称与对应类的映射
method_map = {
    "GraphEraser": grapheraser,
    "GNNDelete": gnndelete,
    "CGU": cgu,
    "SGU": sgu,
    "GIF": gif,
    "GUIDE": guide,
    "GST": gst,
    "Projector": projector,
    "MEGU": megu,
    "GraphRevoker": grapheraser,
    "UTU": utu,
    "GUKD": gukd,
    "D2DGN": d2dgn,
    "IDEA": idea,
    "CEU": ceu,
    "ScaleGUN": scalegun,
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








from task import BaseTrainer


class ScaleGUNTrainer(BaseTrainer):
    def __init__(self, args, logger, model, data):
        super().__init__(args, logger, model, data)
import sys
import os
import logging
import pprint
import time
from config import root_path
class Logger(object):

    def __init__(self, filename):
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)

        # show on
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()

    def log_model_info(self, model):
        # 获取模型的总参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # 记录模型的总参数数量和可训练参数数量
            self.info("Model Summary:")
            self.info("Total Parameters: {:,}".format(total_params))
            self.info("Trainable Parameters: {:,}".format(trainable_params))

            # 使用pprint格式化输出模型的每一层结构
            model_structure = pprint.pformat(model)
            self.info("Model Architecture:\n{}".format(model_structure))

            # 逐层输出模型的参数信息
            for name, param in model.named_parameters():
                self.info("Layer: {} | Size: {} | Requires Grad: {}".format(
                    name, param.size(), param.requires_grad))

            # 输出模型的训练状态
            self.info("Model is in training mode: {}".format(model.training))
            # 输出模型所在的设备信息
            self.info("Model is on device: {}".format(next(model.parameters()).device))

def create_logger(args):
    timestamp = time.time()
    current_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(timestamp))
    model_name = args['base_model']
    data_name = args['dataset_name']
    method_name = args['unlearning_methods']
    run_id = f"{timestamp:.8f}"
    log_dir = os.path.join(root_path , "log", method_name,data_name, model_name)
    logger_name = os.path.join(log_dir, current_time_str + ".log")
    logger = Logger(logger_name)

    logger.info(f"my pid: {os.getpid()}")
    formatted_args = pprint.pformat(args)
    logger.info(formatted_args)

    return logger


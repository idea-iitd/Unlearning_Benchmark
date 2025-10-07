import torch
import os
from config import root_path
import yaml
class abstract_model(torch.nn.Module):
    def __init__(self):
        super(abstract_model,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def save_model(self, save_path):
        # self.logger.info('saving models')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)

    def load_config(self):
        # Have to change this below line because root path was changed
        f = open(  'GULib-master/model/properties/'+ self.args["base_model"] +'.yaml','r')
        config_str = f.read()
        config = yaml.load(config_str,Loader=yaml.FullLoader)
        # self.lr = config['lr']
        # self.decay = config['decay']
        return Config(config) 

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class RandomizedClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, requires_grad=False):
        super(RandomizedClassifier, self).__init__()

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, x):

        x = self.cls(x)
        return x




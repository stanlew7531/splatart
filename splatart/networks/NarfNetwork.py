import torch
import numpy as np
import yaml
from typing import Optional

class NarfNetwork(torch.nn.Module):
    def __init__(self, config_file: Optional[str]=None, config_str:Optional[str]=None):
        super().__init__()
        if config_file is None and config_str is None:
            raise ValueError("Must provide either a config file or a config string")
        
        self.load_config(config_str if config_file is None else open(config_file, "r").read())

    def load_config(self, config_str):
        self.config = yaml.load(config_str, Loader=yaml.FullLoader)
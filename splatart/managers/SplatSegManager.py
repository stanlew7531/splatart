import json
from pathlib import Path
import os
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional

import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManager
from splatart.utils.lie_utils import SE3, SE3Exp

import gsplat

from nerfstudio.utils.eval_utils import eval_setup

class SplatSegManager():
    def __init__(self, num_parts:int, manager: SplatManager):
        self.gauss_params = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_parts = num_parts
        self.part_scores = torch.zeros((num_parts, 1)).to(self.device) # will be updated when loading from the manager
        self.softmax_op = torch.nn.Softmax(dim=0)
        self.part_trans = torch.zeros((2, num_parts, 3)).to(self.device)
        self.part_quats = torch.zeros((2, num_parts, 4)).to(self.device)
        self.neighbor_edges = None
        self.load_from_manager(manager)

    def load_from_manager(self, manager: SplatManager, neighbors: torch.Tensor = None):
        self.manager = manager
        self.device = manager.device
        self.model = manager.model
        self.sh_degree = manager.sh_degree
        self.rasterize_mode = manager.rasterize_mode
        self.dataparser_tf_matrix = manager.dataparser_tf_matrix
        self.dataparser_scale = manager.dataparser_scale
        self.neighbor_edges = neighbors

        

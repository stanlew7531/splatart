import torch
import os
import json
import cv2 as cv
import numpy as np

class SplatTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json:str):
        self.config = json.load(open(dataset_json, "r"))
        self.base_dir = os.path.dirname(dataset_json)
        self.fl_x = self.config["fl_x"]
        self.fl_y = self.config["fl_y"]
        self.cx = self.config["cx"]
        self.cy = self.config["cy"]
        self.width = int(self.config["w"])
        self.height = int(self.config["h"])
        self.dataframes = self.config["frames"]
        # self.gt_part_world_poses = self.config["gt_part_world_poses"]
        # self.gt_configurations = self.config["configurations"]
        self.image_cache = {}
        self.semantics_cache = {}

    def __len__(self):
        return len(self.dataframes)
    
    def get_intrinsic(self):
        return torch.Tensor([[self.fl_x, 0, self.cx], [0, self.fl_y, self.cy], [0, 0, 1]])
    
    def __getitem__(self, idx):
        frame = self.dataframes[idx]
        rgb_data = None
        semantics_data = None

        if idx in self.image_cache.keys():
            rgb_data = self.image_cache[idx]
        else:
            rgb_path = os.path.join(self.base_dir, frame["file_path"])
            rgb_data = cv.imread(rgb_path,cv.IMREAD_UNCHANGED)
            # reverse only the RGB channels
            rgb_data[:,:,:3] = rgb_data[:,:,-2::-1].copy() # have to copy to eliminate negative strides
            rgb_data = torch.from_numpy(rgb_data)
            self.image_cache[idx] = rgb_data

        if idx in self.semantics_cache.keys():
            semantics_data = self.semantics_cache[idx]
        else:
            semantics_path = os.path.join(self.base_dir, frame["semantics_path"])
            semantics_data = torch.from_numpy(cv.imread(semantics_path, cv.IMREAD_UNCHANGED).astype(np.int32))
            self.semantics_cache[idx] = semantics_data

        camera_tf = torch.Tensor(frame["transform_matrix"])

        to_return = {
            "transform_matrix": camera_tf,
            "rgb": rgb_data,
            "semantics": semantics_data
        }

        return idx, to_return
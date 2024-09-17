import yaml
import numpy as np
import os
import open3d as o3d

class CloudManager():
    def __init__(self, config_file:str):
        # get the directory of the config file
        self.config_file = config_file
        self.config_dir = os.path.dirname(self.config_file)

        # set the default points and rgb directories 
        self.points_dir = os.path.join(self.config_dir, "points")
        self.rgbs_dir = os.path.join(self.config_dir, "rgb")

        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.clouds = {}
        self.parts = []
        self.load_clouds()

    def get_times(self):
        return self.clouds.keys()
    
    def get_parts(self):
        return self.parts
    
    def get_cloud_numpy(self, time:str, part:str):
        cloud_data = self.clouds[time][part]
        points = cloud_data["points"]
        rgbs = cloud_data["rgbs"]
        return points, rgbs
    
    def get_cloud_o3d(self, time:str, part:str):
        cloud_data = self.clouds[time][part]
        points = cloud_data["points"]
        rgbs = cloud_data["rgbs"]
        cloud = o3d.t.geometry.PointCloud()
        cloud.point.positions = o3d.core.Tensor(points)
        cloud.point.colors = o3d.core.Tensor(rgbs)
        return cloud

    def load_cloud(self, time:str, part:str):
        part_config = self.config["times"][time][part]
        fname_prefix = part_config["fname_prefix"]
        points_fname = os.path.join(self.points_dir, f"{fname_prefix}.npy")
        rgbs_fname = os.path.join(self.rgbs_dir, f"{fname_prefix}.npy")

        points = np.load(points_fname)
        rgbs = np.load(rgbs_fname)

        transform = None
        if("transform" in part_config.keys()):
            transform = np.array(part_config["transform"])

        return points, rgbs, transform
    
    def set_config_transform(self, time:str, part:str, transform:np.ndarray):
        # turn the transform from numpy to a basic python list
        self.config["times"][time][part]["transform"] = transform.tolist()

    def get_config_transform(self, time:str, part:str):
        return np.array(self.config["times"][time][part]["transform"])

    def save_config(self, config_file:str=None):
        if config_file is None:
            config_file = self.config_file
        with open(config_file, "w") as f:
            yaml.dump(self.config, f)

    def load_clouds(self):
        # if there are override directories in the config, use those instead
        if "points_dir" in self.config.keys():
            self.points_dir = self.config["points_dir"]
        if "rgbs_dir" in self.config.keys():
            self.rgbs_dir = self.config["rgbs_dir"]

        # get the list of part keys in the config
        parts = []
        for time in self.config["times"].keys():
            for part in self.config["times"][time].keys():
                if part not in parts:
                    parts.append(part)
        self.parts = parts

        # load the clouds from disk
        for time in self.config["times"].keys():
            time_data = self.config["times"][time]
            self.clouds[time] = {}
            for part in parts:
                if part in time_data.keys():
                    points, rgbs, transform = self.load_cloud(time, part)
                    self.clouds[time][part] = {"points":points, "rgbs":rgbs, "transform":transform}

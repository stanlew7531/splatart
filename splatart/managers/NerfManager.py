import json
from pathlib import Path
import os
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional

from splatart.utils.lie_utils import SE3, SE3Exp

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.exporter.exporter_utils import generate_point_cloud

class NerfManager(object):
    # constructor
    def __init__(self, config_file:str, test_mode:str = "inference"):
        initial_dir = os.getcwd()
        # read the config data json
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.models = {}
        self.datasets = {}

        # loop through the model configs in the config file
        for model_config in self.config["models"]:
            to_add = {}
            base_path = model_config["base_path"]
            model_config_path = Path(base_path + model_config["config_path"])
            os.chdir(model_config["base_path"])
            setup_results = eval_setup(model_config_path, test_mode = test_mode)
            trainer_config, pipeline, ckpt_path = setup_results[0], setup_results[1], setup_results[2]
            os.chdir(initial_dir)
            dataparser_tf_path = model_config["dataparser_tf_path"]
            with open(dataparser_tf_path, "r") as f:
                dataparser_tf = json.load(f)
                dataparser_tf_matrix = torch.eye(4)
                dataparser_tf_matrix[0:3, :] = torch.tensor(dataparser_tf["transform"])
                dataparser_tf_scale = torch.tensor(dataparser_tf["scale"])
            to_add["dataparser_tf_matrix"] = dataparser_tf_matrix
            to_add["dataparser_tf_scale"] = dataparser_tf_scale
            to_add["trainer_config"] = trainer_config
            to_add["pipeline"] = pipeline
            to_add["ckpt_path"] = ckpt_path
            self.models[model_config["name"]] = to_add

        # get the dataset information
        for dataset_config in self.config["datasets"]:
            to_add = {}
            transforms_json_path = dataset_config["transforms_json_path"]
            with open(transforms_json_path, "r") as f:
                transforms_json = json.load(f)
            cx = torch.Tensor([transforms_json["cx"]])
            cy = torch.Tensor([transforms_json["cy"]])
            fx = torch.Tensor([transforms_json["fl_x"]])
            fy = torch.Tensor([transforms_json["fl_y"]])
            width = transforms_json["w"]
            height = transforms_json["h"]
            to_add["cx"] = cx
            to_add["cy"] = cy
            to_add["fx"] = fx
            to_add["fy"] = fy
            to_add["width"] = width
            to_add["height"] = height
            to_add["base_path"] = dataset_config["base_path"]
            frames = transforms_json["frames"]
            poses = []
            meta_names = []
            meta_label_names = []
            scenes = []
            for frame in frames:
                tf_matrix = frame["transform_matrix"]
                pose = torch.tensor(tf_matrix).reshape(4,4)
                poses.append(pose)
                meta_names.append(frame["file_path"])
                if "time" in frame.keys():
                    scenes.append(torch.tensor(frame["time"]))
                else:
                    scenes.append(torch.tensor(0))
                if("labels_path" in frame.keys()):
                    meta_label_names.append(frame["labels_path"])

            poses = torch.stack(poses, dim=0)
            scenes = torch.stack(scenes, dim=0)
            to_add["train_poses"] = poses
            to_add["meta_names"] = meta_names
            to_add["meta_label_names"] = meta_label_names
            to_add["scenes"] = scenes
            self.datasets[dataset_config["name"]] = to_add

    def apply_dataparser_tf_to_points(self, model_name:str, points:torch.Tensor):
        model_info = self.models[model_name]
        dataparser_tf_matrix = model_info["dataparser_tf_matrix"]
        dataparser_tf_scale = model_info["dataparser_tf_scale"]
        # make sure the points are homogenous
        if(points.shape[-1] == 3):
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim = -1)
        # apply the transform
        points = torch.matmul(dataparser_tf_matrix.to(points.device, points.dtype), points.transpose(-1, -2)).transpose(-1, -2)
        points[..., :3] *= dataparser_tf_scale
        return points[..., :3]
    
    def convert_pose_to_model_space(self, model_name:str, pose:torch.Tensor):
        model_info = self.models[model_name]
        dataparser_tf_matrix = model_info["dataparser_tf_matrix"]
        dataparser_tf_scale = model_info["dataparser_tf_scale"]
        # transfom the matrix to the model's space
        pose_model = torch.matmul(dataparser_tf_matrix, pose)
        pose_model[..., :3, 3] *= dataparser_tf_scale
        return pose_model

    def render_model_at_pose(self, name:str,\
                             pose:torch.Tensor,\
                             time: Optional[torch.Tensor] = None,\
                             dataset_name:Optional[str] = None,\
                             output_semantics = False,\
                             output_numpy = True,\
                             output_device = "cpu",\
                            ):
        dataset_info = self.datasets[name if dataset_name is None else dataset_name]
        model_info = self.models[name]
        # get the camera intrinsics
        cx = dataset_info["cx"]#torch.tensor(640).to(dataset_info["cx"])#
        cy = dataset_info["cy"]#torch.tensor(480).to(dataset_info["cx"])#
        fx = dataset_info["fx"]
        fy = dataset_info["fy"]
        width  = 640# torch.tensor(1280).to(dataset_info["width"])
        height = 480#torch.tensor(960).to(dataset_info["height"])
        # get the dataparser transforms
        dataparser_tf_matrix = model_info["dataparser_tf_matrix"]
        dataparser_tf_scale = model_info["dataparser_tf_scale"]
        # transfom the matrix to the model's space
        pose_model = torch.matmul(dataparser_tf_matrix, pose)
        pose_model[..., :3, 3] *= dataparser_tf_scale
        # create the camera object
        cams = Cameras(camera_to_worlds=pose_model[..., :3, :].to(model_info["pipeline"].device), cx=cx, cy=cy, fx=fx, fy=fy, height=height, width=width)
        if(time is not None):
            assert pose.shape[0] == time.shape[0], "pose and time must have the same batch size"
            cams.times = time.to(model_info["pipeline"].device)
        # render the model
        imgs = []
        segmentations = []
        ray_bundles = []
        depths = []
        num_cams = pose_model.shape[0]
        for cam_idx in tqdm(range(num_cams), leave=False):
            # if(cams.times is None):
            #     cams.times = torch.zeros((pose_model.shape[0],1), device=cams.device)
            # ray_bundle = cams.generate_rays(camera_indices = cam_idx)
            
            # outputs = model_info["pipeline"].model.get_outputs_for_camera_ray_bundle(ray_bundle)
            outputs = model_info["pipeline"].model.get_outputs_for_camera(cams)
            print(outputs.keys())
            
            depth = outputs['depth'].to(output_device)
            # print(depth.unique())
            img = outputs['rgb'].to(output_device)
            # turn img into 4 channel
            img = torch.cat([img, torch.ones_like(img[..., :1])], dim = -1)
            # if depth is greater than a threhsold, set the alpha channel to 0
            print(img.shape)
            # print(depth.shape)
            img[..., 3] = (depth[..., 0] < 4).float()
            # print(img.shape)
            # raise Exception("stopping to print")
            if(output_semantics):
                seg = outputs['semantics'].to(output_device)
            else:
                seg = None

            if(output_numpy):
                depth = depth.numpy()
                img = img.numpy()
                seg = seg.numpy() if seg is not None else None

            depths.append(depth)
            imgs.append(img)
            segmentations.append(seg)

            # ray_bundles.append(ray_bundle)
        # stack the images
        if(output_numpy):
            imgs = np.stack(imgs, axis=0)
            depths = np.stack(depths, axis=0)
            segmentations = np.stack(segmentations, axis=0)
        else:
            imgs = torch.stack(imgs, axis=0)
            depths = torch.stack(depths, axis=0)
            if(segmentations[0] is not None):
                segmentations = torch.stack(segmentations, axis=0)

        cam_intrinsic_mat = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return imgs, ray_bundles, depths, cam_intrinsic_mat, segmentations
    
    def get_point_cloud(self, name:str, cloud_time:Optional[int] = 0):
        model_info = self.models[name]
        pipeline = model_info["pipeline"]
        # pipeline.datamanager.setup_train()
        return generate_point_cloud(pipeline, cloud_time = cloud_time)

    def get_ray_samples(self, name:str, ray_bundle:RayBundle):
        model = self.models[name]["pipeline"].model
        ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns = model.density_fns)
        if(model.field.spatial_distortion is not None):
            positions = ray_samples.frustums.get_positions()
            positions = model.field.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), model.field.aabb)
        return positions
    
    def get_field(self, name:str):
        return self.models[name]["pipeline"].model.field
    
    def set_field(self, name:str, field):
        self.models[name]["pipeline"].model.field = field

    def write_pipeline_state(self, name:str, output_ckpt:str):
        torch.save({"pipeline": self.models[name]["pipeline"].module.state_dict() 
                    if hasattr(self.models[name]["pipeline"], "module") 
                    else self.models[name]["pipeline"].state_dict()},
                    output_ckpt)
        
    def get_device_from_models(self):
        return self.models[list(self.models.keys())[0]]["pipeline"].device
    
    def get_campose_labels(self, dataset_name, idx):
        dataset_info = self.datasets[dataset_name]
        return dataset_info["train_poses"][idx]

if __name__ == "__main__":
    # take in the config file as arg 1
    import sys
    config_file = sys.argv[1]
    manager = NerfManager(config_file)
    print(manager.models)
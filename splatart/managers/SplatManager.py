import json
from pathlib import Path
import os
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional

import pytorch3d.transforms as p3dt

from splatart.utils.lie_utils import SE3, SE3Exp

import gsplat

from nerfstudio.utils.eval_utils import eval_setup

# shamelessly stolen from splatfacto.py in the nerfstudio repo
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

def load_model(model_dir:str, cfg_yml:str, ns_base_path:str, test_mode:str = "inference"):
    # the eval_setup needs the cwd to match the directory where the outputs folder is
    # so we need cache the present cwd so we can change back later
    initial_dir = os.getcwd()
    model_config_path = Path(os.path.join(model_dir, cfg_yml))
    print(f"loading model config from: {model_config_path}")
    os.chdir(ns_base_path)
    setup_results = eval_setup(model_config_path, test_mode = test_mode)
    trainer_config, pipeline, ckpt_path = setup_results[0], setup_results[1], setup_results[2]
    print(f"success!")
    os.chdir(initial_dir)
    return trainer_config, pipeline, ckpt_path


def load_managers_nerfstudio(input_model_dirs:list[str],\
                num_classes:int,\
                config_yml:str = "config.yml",\
                dataparser_tf:str = "dataparser_transforms.json",\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/"):
    
    splat_managers = []
    i = 0
    for model_dir in input_model_dirs:
        trainer_config, pipeline, ckpt_path = load_model(model_dir, config_yml, ns_base_path)

        splat_gaussian_params = pipeline.model.gauss_params
        splat_sh_degree = pipeline.model.config.sh_degree
        splat_rasterize_mode = pipeline.model.config.rasterize_mode

        dataparser_tf_path = Path(os.path.join(model_dir, dataparser_tf))
        dataparser_json = json.load(open(dataparser_tf_path, "r"))
        dataparser_tf_matrix = torch.Tensor(dataparser_json["transform"])
        # append a row to make it 4x4
        dataparser_tf_matrix = torch.cat([dataparser_tf_matrix, torch.Tensor([[0, 0, 0, 1]])], dim=0)
        dataparser_scale = dataparser_json["scale"]
        manager = SplatManager(splat_gaussian_params, splat_sh_degree, splat_rasterize_mode, num_classes, dataparser_tf_matrix, dataparser_scale)
        splat_managers.append(manager)
        
    return splat_managers

def load_managers_nerfstudio_single(input_model_dirs:list[str],\
                num_classes:int,\
                config_yml:str = "config.yml",\
                dataparser_tf:str = "dataparser_transforms.json",\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/"):
    
    splat_managers = []
    i = 0
    for model_dir in input_model_dirs:
        trainer_config, pipeline, ckpt_path = load_model(model_dir, config_yml, ns_base_path)

        splat_gaussian_params = pipeline.model.gauss_params
        splat_sh_degree = pipeline.model.config.sh_degree
        splat_rasterize_mode = pipeline.model.config.rasterize_mode

        dataparser_tf_path = Path(os.path.join(model_dir, dataparser_tf))
        dataparser_json = json.load(open(dataparser_tf_path, "r"))
        dataparser_tf_matrix = torch.Tensor(dataparser_json["transform"])
        # append a row to make it 4x4
        dataparser_tf_matrix = torch.cat([dataparser_tf_matrix, torch.Tensor([[0, 0, 0, 1]])], dim=0)
        dataparser_scale = dataparser_json["scale"]
        manager = SplatManagerSingle(splat_gaussian_params, splat_sh_degree, splat_rasterize_mode, num_classes, dataparser_tf_matrix, dataparser_scale)
        splat_managers.append(manager)
        
    return splat_managers

def load_managers(input_model_dirs:list[str],\
                managers_subdir:str = "outputs",\
                manager_prefix:str = "splat_manager"):
    
    # load the existing splat managers
    splat_managers = []
    for i in range(len(input_model_dirs)):
        manager_path = os.path.join(managers_subdir, f"{manager_prefix}_{i}.pth")
        manager = torch.load(manager_path)
        splat_managers.append(manager)
    return splat_managers

def means_quats_to_mat(means, quats):
    to_return = torch.zeros((means.shape[0], 4, 4)).to(means)
    to_return[:, :3, :3] = p3dt.quaternion_to_matrix(quats)
    to_return[:, :3, 3] = means
    to_return[:, 3, 3] = 1
    return to_return

def mat_to_means_quats(mat):
    means = mat[:, :3, 3]
    quats = p3dt.matrix_to_quaternion(mat[:, :3, :3])
    return means, quats

class SplatManagerSingle():
    # constructor
    def __init__(self, gauss_params, sh_degree, rasterize_mode, num_parts, dataparser_tf_matrix, dataparser_scale, *args, **kwargs):
        self.object_gaussian_params = gauss_params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sh_degree = sh_degree
        self.rasterize_mode = rasterize_mode
        self.num_parts = num_parts
        self.add_parts_features()
        self.dataparser_tf_matrix = dataparser_tf_matrix.to(self.device)
        self.dataparser_scale = dataparser_scale

    def add_parts_features(self):
        self.object_gaussian_params["features_semantics"] = \
            torch.zeros(self.object_gaussian_params["means"].shape[0], self.num_parts).to(self.device)
        
    def render_gauss_params_at_campose(self, cam_pose, cam_intrinsic, width, height, gauss_params, is_semantics = False):
        # if cam_pose doesnt have a batch dimension, add one
        if len(cam_pose.shape) == 2:
            cam_pose = cam_pose.unsqueeze(0)

        # transfom the matrix to the model's space
        pose_model = torch.matmul(self.dataparser_tf_matrix, cam_pose)
        pose_model[..., :3, 3] *= self.dataparser_scale
        
        viewmat = get_viewmat(pose_model)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        features_dc = gauss_params["features_dc"]
        features_rest = gauss_params["features_rest"]
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

        # render the rgb data
        if(not is_semantics):
            render, alpha, info = gsplat.rasterization(
                means = gauss_params["means"].cuda(),
                quats = gauss_params["quats"].cuda(),
                scales = torch.exp(gauss_params["scales"]).cuda(),
                opacities=torch.sigmoid(gauss_params["opacities"]).squeeze(-1).cuda(),
                colors = colors.cuda(),
                viewmats=viewmat.cuda(), #pose_model[None, :, :].cuda(), #[B, 4, 4]
                Ks = cam_intrinsic[None, :, :].cuda() if len(cam_intrinsic.shape) == 2 else cam_intrinsic.cuda(), #[B, 3, 3]
                width=width,
                height=height,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=self.sh_degree,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.rasterize_mode,
            )
        else:
            render, alpha, info = gsplat.rasterization(
                means = gauss_params["means"].cuda(),
                quats = gauss_params["quats"].cuda(),
                scales = torch.exp(gauss_params["scales"]).cuda(),
                opacities=torch.sigmoid(gauss_params["opacities"]).squeeze(-1).cuda(),
                colors = gauss_params["features_semantics"].cuda(),
                viewmats=viewmat.cuda(),
                Ks = cam_intrinsic[None, :, :].cuda() if len(cam_intrinsic.shape) == 2 else cam_intrinsic.cuda(), #[B, 3, 3]
                width=width,
                height=height,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=None,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.rasterize_mode,
            )

        return render, alpha, info
    
    
    def render_at_campose(self, cam_pose, cam_intrinsic, width, height, is_semantics = False):
        render_gauss_params = self.object_gaussian_params
        return self.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, render_gauss_params, is_semantics)

class SplatManager(torch.nn.Module):
    # constructor
    def __init__(self, gauss_params, sh_degree, rasterize_mode, num_parts, dataparser_tf_matrix, dataparser_scale, default_semantics = 0, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # create a tf_quats and tf_trans which we use to move the part or object during rendering
        # to a different position from where it was originall trained
        # base tf_quats should represent no rotations, in wxyz (scalar first) ordering
        tf_quats = torch.zeros(1,4) # 1, 4
        tf_quats[:, 0] = 1.0
        tf_trans = torch.zeros(1,3) # 1, 3
        gauss_params["tf_quats"] = tf_quats
        gauss_params["tf_trans"] = tf_trans
        self.object_gaussian_params = gauss_params

        self.sh_degree = sh_degree
        self.rasterize_mode = rasterize_mode
        self.parts_gauss_params = {}
        self.num_parts = num_parts
        self.split_parts(num_parts)
        self.dataparser_tf_matrix = dataparser_tf_matrix
        self.dataparser_scale = dataparser_scale

    def split_parts(self, num_parts):
        if(num_parts > 1):
            # get the semantics features
            obj_semantics = self.object_gaussian_params["features_semantics"] # N, num_parts
            # get the part for each N (max of the semantics)
            part_indices = torch.argmax(obj_semantics, dim=1) # N
        else:
            part_indices = torch.zeros(self.object_gaussian_params["means"].shape[0]).long() # N

        # split out each part and store in parts_gauss_params
        for i in range(num_parts):
            # create a tf_quats and tf_trans which we use to move the part or object during rendering
            # to a different position from where it was originall trained
            # base tf_quats should represent no rotations, in wxyz (scalar first) ordering
            tf_quats = torch.zeros(1, 4) # 1, 4
            tf_quats[:, 0] = 1.0
            tf_trans = torch.zeros(1,3) # N, 3

            part_gauss_params = torch.nn.ParameterDict(
                {
                    "means": self.object_gaussian_params["means"][part_indices == i],
                    "scales": self.object_gaussian_params["scales"][part_indices == i],
                    "quats": self.object_gaussian_params["quats"][part_indices == i],
                    "features_dc": self.object_gaussian_params["features_dc"][part_indices == i],
                    "features_rest": self.object_gaussian_params["features_rest"][part_indices == i],
                    "opacities": self.object_gaussian_params["opacities"][part_indices == i],
                    "features_semantics": self.object_gaussian_params["features_semantics"][part_indices == i]\
                        if "features_semantics" in self.object_gaussian_params.keys()\
                        else None,
                    "tf_quats": tf_quats,
                    "tf_trans": tf_trans,
                }
            )
            self.parts_gauss_params[i] = part_gauss_params

    def set_part(self, part_key, part_gauss_params_dict):
        self.parts_gauss_params[part_key] = torch.nn.ParameterDict(part_gauss_params_dict)


    def get_neighbor_edges_distances_deltas(self, radius=0.02):
        radius_scaled = radius * self.dataparser_scale
        n_splats = self.object_gaussian_params["means"].shape[0]
        means = self.object_gaussian_params["means"]
        quats = self.object_gaussian_params["quats"]
        tfs = means_quats_to_mat(means, quats)
        for i in tqdm(range(n_splats)):
            cur_mean = means[i]
            cur_tf = tfs[i]
            tfs_cur_frame = torch.linalg.inv(cur_tf) @ tfs
            deltas = tfs_cur_frame[:, :3, 3]
            distances = torch.norm(means - cur_mean, dim=1)
            neighbors = torch.nonzero(distances < radius_scaled)
            for neighbor in neighbors:
                if(neighbor != i):
                    yield (int(i), int(neighbor)), distances[neighbor], deltas[neighbor]

    def get_render_gauss_params(self, gauss_params, apply_transforms=True):
        # turn the quats and means into 4x4 matrices
        rots_base = p3dt.quaternion_to_matrix(gauss_params["quats"]) # 1, 3, 3
        trans_base = gauss_params["means"] # 1, 3

        # combine the rotations and translations into 4x4 matrices
        transforms_base = torch.eye(4, device=rots_base.device, dtype=rots_base.dtype).repeat(rots_base.shape[0], 1, 1)
        transforms_base[:, :3, :3] = rots_base
        transforms_base[:, :3, 3] = trans_base
        if apply_transforms == True:
            # turn our transforms into 4x4 matrices also
            rots_tf = p3dt.quaternion_to_matrix(gauss_params["tf_quats"].to(rots_base.device)) # N, 3, 3
            trans_tf = gauss_params["tf_trans"].to(rots_base.device) # N, 3

            transforms_tf = torch.eye(4, device=rots_base.device, dtype=rots_base.dtype).repeat(rots_base.shape[0], 1, 1)
            # print(f"transforms tf shape: {transforms_tf.shape}, rots_tf shape: {rots_tf.shape}")
            transforms_tf[:, :3, :3] = rots_tf
            transforms_tf[:, :3, 3] = trans_tf

            render_transforms = torch.matmul(transforms_tf, transforms_base)
        else:
            render_transforms = transforms_base

        render_means = render_transforms[:, :3, 3]
        render_rots = render_transforms[:, :3, :3]
        render_quats = p3dt.matrix_to_quaternion(render_rots)

        return render_means, render_quats
    
    def combine_gauss_params_prerender(self, gauss_params_list:list[dict], apply_transforms = True):
        combined_gauss_params = {}
        keys_naive_cat = ["scales", "opacities", "features_dc", "features_rest", "features_semantics"]
        
        for gauss_params in gauss_params_list:
            # some render parameters can be naively concatenated across different managers
            # the ones involving the part/object transforms cannot however
            # we must apply the transforms to the means and quats before concatenating so that everything
            # is in the same render space and all the batch etc. dimensions line up
            for key in keys_naive_cat:
                if key in gauss_params.keys():
                    if key not in combined_gauss_params.keys():
                        combined_gauss_params[key] = gauss_params[key]
                    else:
                        combined_gauss_params[key] = torch.cat([combined_gauss_params[key], gauss_params[key]], dim=0)
            render_means, render_quats = self.get_render_gauss_params(gauss_params, apply_transforms=apply_transforms)
            if "means" not in combined_gauss_params.keys():
                combined_gauss_params["means"] = render_means
            else:
                combined_gauss_params["means"] = torch.cat([combined_gauss_params["means"], render_means], dim=0)
            if "quats" not in combined_gauss_params.keys():
                combined_gauss_params["quats"] = render_quats
            else:
                combined_gauss_params["quats"] = torch.cat([combined_gauss_params["quats"], render_quats], dim=0)
        
        return combined_gauss_params
            


    def render_gauss_params_at_campose(self, cam_pose, cam_intrinsic, width, height, gauss_params, include_semantics = False, apply_transforms = True):
        # if cam_pose doesnt have a batch dimension, add one
        if len(cam_pose.shape) == 2:
            cam_pose = cam_pose.unsqueeze(0)

        # transfom the matrix to the model's space
        pose_model = torch.matmul(self.dataparser_tf_matrix, cam_pose)
        pose_model[..., :3, 3] *= self.dataparser_scale
        
        viewmat = get_viewmat(pose_model)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        # check if we got a list of gauss params that we need to combine
        # we run the same function regardless for simplicity, but we must lisftif if we only have one
        if(type(gauss_params) == list):
            gauss_params = self.combine_gauss_params_prerender(gauss_params, apply_transforms = apply_transforms)
        else:
            gauss_params = self.combine_gauss_params_prerender([gauss_params], apply_transforms = apply_transforms)

        features_dc = gauss_params["features_dc"]
        features_rest = gauss_params["features_rest"]
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

        # render the rgb data
        render, alpha, info = gsplat.rasterization(
            means = gauss_params["means"].cuda(),
            quats = gauss_params["quats"].cuda(),
            scales = torch.exp(gauss_params["scales"]).cuda(),
            opacities=torch.sigmoid(gauss_params["opacities"]).squeeze(-1).cuda(),
            colors = colors.cuda(),
            viewmats=viewmat.cuda(), #pose_model[None, :, :].cuda(), #[B, 4, 4]
            Ks = cam_intrinsic[None, :, :].cuda() if len(cam_intrinsic.shape) == 2 else cam_intrinsic.cuda(), #[B, 3, 3]
            width=width,
            height=height,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=self.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.rasterize_mode,
        )

        # TODO: render the semantics too if requested

        return render, alpha, info
    
    def get_part_tf_gauss_params(self, part_idx):
        part_gauss_params = self.parts_gauss_params[part_idx]

        # turn our transforms into 4x4 matrices also
        rots_tf = p3dt.quaternion_to_matrix(part_gauss_params["tf_quats"]) # 1, 3, 3
        trans_tf = part_gauss_params["tf_trans"] # 1, 3

        return rots_tf, trans_tf
    
    def render_parts_at_campose(self, cam_pose, cam_intrinsic, width, height, part_idxs, apply_transforms=True):
        parts_gauss_params = []
        for part_idx in part_idxs:
            parts_gauss_params.append(self.parts_gauss_params[part_idx])
        return self.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, parts_gauss_params, apply_transforms = apply_transforms)
    
    def render_given_parts_at_campose(self, cam_pose, cam_intrinsic, width, height, parts_gauss_params, apply_transforms=True):
        return self.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, parts_gauss_params, apply_transforms = apply_transforms)

    def render_object_at_campose(self, cam_pose, cam_intrinsic, width, height, apply_transforms=True):
        return self.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, self.object_gaussian_params, apply_transforms = apply_transforms)

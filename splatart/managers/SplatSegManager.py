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


class SplatSegManager():
    def __init__(self, num_parts:int, manager: SplatManager):
        self.gauss_params = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_parts = num_parts
        # self.part_scores = None # will be updated when loading from the manager
        self.softmax_op = torch.nn.Softmax(dim=0)
        # create starting part trans and quats with no transform
        self.part_trans = ((torch.randn((2, num_parts, 3)) - 0.5) * 0.001).to(self.device) # these should be zero centered and small mean
        # self.part_trans = torch.zeros((2, num_parts, 3)).to(self.device) # these should be zeros
        self.part_eulers = torch.zeros((2, num_parts, 3)).to(self.device)

        # self.part_quats = torch.zeros((2, num_parts, 4)).to(self.device) # wxyz order
        # add the 1 to make it a no-op
        # self.part_quats[:, :, 0] = 1.0
        
        self.load_from_manager(manager)

        self.part_trans.requires_grad = True
        self.part_eulers.requires_grad = True
        # self.part_quats.requires_grad = True
        self.part_scores.requires_grad = True

    def means_quats_to_mat(self, means, quats):
        to_return = torch.zeros((means.shape[0], 4, 4)).to(means)
        to_return[:, :3, :3] = p3dt.quaternion_to_matrix(quats)
        to_return[:, :3, 3] = means
        to_return[:, 3, 3] = 1
        return to_return
    
    def mat_to_means_quats(self, mat):
        means_to_return = mat[:,:3, 3]
        rot_mats = mat[:, :3, :3]
        quats_to_return = p3dt.matrix_to_quaternion(rot_mats)
        return means_to_return, quats_to_return

    def load_from_manager(self, manager: SplatManager, neighbors: torch.Tensor = None):
        self.manager = manager
        self.sh_degree = manager.sh_degree
        self.rasterize_mode = manager.rasterize_mode
        self.dataparser_tf_matrix = manager.dataparser_tf_matrix
        self.dataparser_scale = manager.dataparser_scale
        self.gauss_params = manager.object_gaussian_params
        self.num_splats = self.gauss_params["means"].shape[0]
        self.part_scores = torch.randn(self.num_parts, self.num_splats).to(self.device) * 100

    def render_gauss_params_at_campose(self, cam_pose, cam_intrinsic, width, height, gauss_params):
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

        return render, alpha, info

    def render_at_campose(self, cam_pose, cam_intrinsic, width, height, scene=0):
        assert (scene == 0 or scene == 1), "only support 2 scenes currently (0 and 1 idx)"

        scene_part_trans = self.part_trans[scene] # (num_parts, 3)
        scene_part_rots = p3dt.euler_angles_to_matrix(self.part_eulers[scene], "XYZ")
        scene_part_quats = p3dt.matrix_to_quaternion(scene_part_rots)
        # scene_part_quats = self.part_quats[scene] # (num_parts, 4)

        part_probabilities = torch.nn.functional.softmax(self.part_scores, dim=0) 
        # part_probabilities = self.softmax_op(self.part_scores) # (num_parts, num_splats)

        per_splat_part_trans = torch.matmul(part_probabilities.T, scene_part_trans) # (num_splats, num_parts) @ (num_parts, 3) -> (num_splats, 3)
        per_splat_part_quats = torch.matmul(part_probabilities.T, scene_part_quats) # (num_splats, num_parts) @ (num_parts, 4) -> (num_splats, 4)

        base_means = self.gauss_params["means"] # (num_splats,3)
        base_quats = self.gauss_params["quats"] # (num_splats,4) these are in wxyz form

        base_tfs = self.means_quats_to_mat(base_means, base_quats)
        per_splat_part_tfs = self.means_quats_to_mat(per_splat_part_trans, per_splat_part_quats)

        render_splat_tfs = torch.linalg.inv(per_splat_part_tfs) @ base_tfs
        render_splat_means, render_splat_quats = self.mat_to_means_quats(render_splat_tfs)

        render_gauss_params = {
            "means": self.gauss_params["means"],#render_splat_means,
            "quats": self.gauss_params["quats"],#render_splat_quats,
            "scales": self.gauss_params["scales"],
            "opacities": self.gauss_params["opacities"],
            "features_dc": self.gauss_params["features_dc"],
            "features_rest": self.gauss_params["features_rest"]
        }

        return self.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, render_gauss_params)





        
        
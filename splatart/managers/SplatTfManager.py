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


class SplatTFManager():
    def __init__(self, src_manager: SplatManager, dst_manager: SplatManager):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_from_managers(src_manager, dst_manager)

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
    
    def get_neighbor_edges_distances_deltas(self, gauss_params, scene=0, radius=0.02):
        radius_scaled = radius * self.dataparser_scale[scene]
        n_splats = gauss_params["means"].shape[0]
        means = gauss_params["means"]
        quats = gauss_params["quats"]
        tfs = self.means_quats_to_mat(means, quats)

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

    def load_from_managers(self, src_manager: SplatManager, dst_manager: SplatManager):
        self.manager = [src_manager, dst_manager]
        self.sh_degree = [src_manager.sh_degree, dst_manager.sh_degree]
        self.rasterize_mode = [src_manager.rasterize_mode, dst_manager.rasterize_mode]
        self.dataparser_tf_matrix = [src_manager.dataparser_tf_matrix, dst_manager.dataparser_tf_matrix]
        self.dataparser_scale = [src_manager.dataparser_scale, dst_manager.dataparser_scale]
        self.src_gauss_params = src_manager.object_gaussian_params
        self.dst_gauss_params = dst_manager.object_gaussian_params

        self.num_splats = [self.src_gauss_params["means"].shape[0], self.dst_gauss_params["means"].shape[0]]
        self.src_tf_trans = torch.zeros((self.num_splats[0], 3)).to(self.device)
        self.src_tf_quats = torch.zeros((self.num_splats[0], 4)).to(self.device)
        self.dst_tf_trans = torch.zeros((self.num_splats[1], 3)).to(self.device)
        self.dst_tf_quats = torch.zeros((self.num_splats[1], 4)).to(self.device)
        self.src_tf_quats[:, 0] = 1.0
        self.dst_tf_quats[:, 0] = 1.0
        self.src_tf_trans.requires_grad = True
        self.src_tf_quats.requires_grad = True
        self.dst_tf_trans.requires_grad = True
        self.dst_tf_quats.requires_grad = True

        # get the neighbor edges and distances
        self.src_neighbor_edges, self.src_neighbor_distances, self.src_neighbor_deltas = [], [], []
        self.dst_neighbor_edges, self.dst_neighbor_distances, self.dst_neighbor_deltas = [], [], []
        with torch.no_grad():
            for edge, distance, delta in self.get_neighbor_edges_distances_deltas(self.src_gauss_params,0):
                self.src_neighbor_edges.append(edge)
                self.src_neighbor_distances.append(distance)
                self.src_neighbor_deltas.append(delta)
            self.src_neighbor_edges = torch.Tensor(self.src_neighbor_edges).to(self.src_gauss_params["means"].device)
            self.src_neighbor_distances = torch.Tensor(self.src_neighbor_distances).to(self.src_gauss_params["means"].device)
            self.src_neighbor_deltas = torch.stack(self.src_neighbor_deltas, dim=0).to(self.src_gauss_params["means"].device)

            for edge, distance, delta in self.get_neighbor_edges_distances_deltas(self.dst_gauss_params,1):
                self.dst_neighbor_edges.append(edge)
                self.dst_neighbor_distances.append(distance)
                self.dst_neighbor_deltas.append(delta)
            self.dst_neighbor_edges = torch.Tensor(self.dst_neighbor_edges).to(self.src_gauss_params["means"].device)
            self.dst_neighbor_distances = torch.Tensor(self.dst_neighbor_distances).to(self.src_gauss_params["means"].device)
            self.dst_neighbor_deltas = torch.stack(self.dst_neighbor_deltas, dim=0).to(self.src_gauss_params["means"].device)

        print(f"src edges shape: {self.src_neighbor_edges.shape}")
        print(f"dst edges shape: {self.dst_neighbor_edges.shape}")
        print(f"src deltas shape: {self.src_neighbor_deltas.shape}")
        print(f"dst deltas shape: {self.dst_neighbor_deltas.shape}")
        print(f"src distances shape: {self.src_neighbor_distances.shape}")
        print(f"dst distances shape: {self.dst_neighbor_distances.shape}")

    def render_gauss_params_at_campose(self, cam_pose, cam_intrinsic, width, height, gauss_params, scene=0):
        # if cam_pose doesnt have a batch dimension, add one
        if len(cam_pose.shape) == 2:
            cam_pose = cam_pose.unsqueeze(0)

        # transfom the matrix to the model's space
        pose_model = torch.matmul(self.dataparser_tf_matrix[scene], cam_pose)
        pose_model[..., :3, 3] *= self.dataparser_scale[scene]
        
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
            sh_degree=self.sh_degree[scene],
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.rasterize_mode[scene],
        )

        return render, alpha, info
    
    def get_transformed_src_gauss_parameters(self):
        tfs_means = self.src_tf_trans
        tfs_quats = self.src_tf_quats
        tf_mats = self.means_quats_to_mat(tfs_means, tfs_quats)
        
        base_means = self.src_gauss_params["means"]
        base_quats = self.src_gauss_params["quats"]
        base_mats = self.means_quats_to_mat(base_means, base_quats)

        transformed_mats = torch.matmul(tf_mats, base_mats)
        transformed_trans, transformed_quats = self.mat_to_means_quats(transformed_mats)

        return transformed_trans, transformed_quats
    
    def get_transformed_dst_gauss_parameters(self):
        tfs_means = self.dst_tf_trans
        tfs_quats = self.dst_tf_quats
        tfs_mats = self.means_quats_to_mat(tfs_means, tfs_quats)
        inv_tfs_mats = torch.linalg.inv(tfs_mats)

        base_means = self.dst_gauss_params["means"]
        base_quats = self.dst_gauss_params["quats"]
        base_mats = self.means_quats_to_mat(base_means, base_quats)

        transformed_mats = torch.matmul(inv_tfs_mats, base_mats)
        transformed_trans, transformed_quats = self.mat_to_means_quats(transformed_mats)
        return transformed_trans, transformed_quats

    def get_scene_gauss_parameters(self, scene = 0, source = 2):
        assert (scene == 0 or scene == 1), "only support 2 scenes currently (0 and 1 idx)"
        assert (source == 0 or source == 1 or source == 2), "source must be 0 (src), 1 (dst) or 2 (both)" 

        # if scene is 0, we apply inverse tfs to the dst scene means/quats to move them from scene 1 to 0
        if(scene == 0):
            transformed_trans, transformed_quats = self.get_transformed_dst_gauss_parameters()
            render_trans = torch.concat((self.src_gauss_params["means"], transformed_trans))
            render_quats = torch.concat((self.src_gauss_params["quats"], transformed_quats))
        # if scene is 1, we apply the tfs to the src scene means/quats to move them from scene 0 to 1
        elif(scene == 1):
            transformed_trans, transformed_quats = self.get_transformed_src_gauss_parameters()
            render_trans = torch.concat((transformed_trans, self.dst_gauss_params["means"]))
            render_quats = torch.concat((transformed_quats, self.dst_gauss_params["quats"]))

        if(source == 2):    
            render_gauss_params = {
                "means": render_trans,
                "quats": render_quats,
                "scales":       torch.concat((self.src_gauss_params["scales"], self.dst_gauss_params["scales"])),
                "opacities":    torch.concat((self.src_gauss_params["opacities"], self.dst_gauss_params["opacities"])),
                "features_dc":      torch.concat((self.src_gauss_params["features_dc"], self.dst_gauss_params["features_dc"])),
                "features_rest":    torch.concat((self.src_gauss_params["features_rest"], self.dst_gauss_params["features_rest"]))
            }
        elif(source == 0):
            render_gauss_params = {
                "means": self.src_gauss_params["means"] if scene == 0 else transformed_trans,
                "quats": self.src_gauss_params["quats"] if scene == 0 else transformed_quats,
                "scales":       self.src_gauss_params["scales"],
                "opacities":    self.src_gauss_params["opacities"],
                "features_dc":  self.src_gauss_params["features_dc"],
                "features_rest": self.src_gauss_params["features_rest"]
            }
        elif(source == 1):
            render_gauss_params = {
                "means": self.dst_gauss_params["means"] if scene == 1 else transformed_trans,
                "quats": self.dst_gauss_params["quats"] if scene == 1 else transformed_quats,
                "scales":       self.dst_gauss_params["scales"],
                "opacities":    self.dst_gauss_params["opacities"],
                "features_dc":  self.dst_gauss_params["features_dc"],
                "features_rest": self.dst_gauss_params["features_rest"]
            }

        return render_gauss_params

    def render_at_campose(self, cam_pose, cam_intrinsic, width, height, scene=0, source=2):
        render_gauss_params = self.get_scene_gauss_parameters(scene, source)
        return self.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, render_gauss_params, scene)





        
        
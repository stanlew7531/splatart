import sys
import os
import open3d as o3d
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_msssim import SSIM
import pickle as pkl
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from plyfile import PlyElement, PlyData
from pytorch3d.loss import chamfer_distance

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import HDBSCAN
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.networks.PoseEstimator import PoseEstimator


def load_base_model(input_model_path:str)->SplatManagerSingle:
    return torch.load(input_model_path)

def load_part_gaussians(part_splats_path:str):
    return torch.load(part_splats_path)

def get_canonical_part_gaussians(canonical_scene_id:int=0, part_gaussians=None, dataparser_matrices = None, dataparser_scales=None):
    dataparser_scale = dataparser_scales[canonical_scene_id]
    dataparser_matrix = dataparser_matrices[canonical_scene_id]
    to_return = part_gaussians[canonical_scene_id]
    # scale the means by the dataparser scale
    for i in range(len(to_return)):
        means = to_return[i]["means"]
        # if we dont have any means (because background) skip this math
        if means.shape[0] != 0:
            # apply the inverse of the dataparser matrix
            means_homo = torch.cat([means, torch.ones(means.shape[0], 1).to(means.device)], dim=1)
            means = torch.linalg.inv(dataparser_matrix) @ means_homo.T
            # remove the homogenous coordinate
            means = means.T[..., :3]
            # scale the points
            means /= dataparser_scale
        to_return[i]["means"] = means

    return to_return

def load_pose_estimator(pose_estimates_path:str)->PoseEstimator:
    return torch.load(pose_estimates_path)

def load_articulation_estimates(articulation_estimates_path:str):
    # get joint for joint metrics.
    with open(articulation_estimates_path, 'rb') as f:
        config_vector = pkl.load(f)
    return config_vector

# adapted from https://github.com/nerfstudio-project/gsplat/issues/234#issuecomment-2197277211
def gauss_params_to_ply(gauss_params, output_fname):

    xyz = gauss_params["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gauss_params["features_dc"].detach().contiguous().cpu().numpy()
    f_rest = gauss_params["features_rest"].transpose(1, 2).flatten(start_dim=1).detach().contiguous().cpu().numpy()
    opacities = gauss_params["opacities"].detach().cpu().numpy()
    scale = gauss_params["scales"].detach().cpu().numpy()
    rotation = gauss_params["quats"].detach().cpu().numpy()


    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(gauss_params["features_dc"].shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(gauss_params["features_rest"].shape[1]*gauss_params["features_rest"].shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gauss_params["scales"].shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gauss_params["quats"].shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    if(not output_fname.endswith('.ply')):
        output_fname += '.ply'
    PlyData([el]).write(output_fname)

def get_chamfer_distance_metric(recon_ply, gt_ply, sample_pts=10000):
    num_means = len(recon_ply.points)
    if num_means < sample_pts:
        sample_pts = num_means
    print(f"sampling {sample_pts} points from each point cloud")
    recon_cloud = recon_ply.farthest_point_down_sample(num_samples=sample_pts)
    gt_cloud = gt_ply.sample_points_uniformly(number_of_points=sample_pts)

    # as tensors
    splat_T = torch.tensor(np.array(recon_cloud.points)).float().unsqueeze(0)
    gt_T = torch.tensor(np.array(gt_cloud.points)).float().unsqueeze(0)

    # average bidirectional chamfer distance to match original paper
    a, _ = chamfer_distance(splat_T, gt_T)
    b, _ = chamfer_distance(gt_T, splat_T)
    chamfer = (a + b) / 2

    return chamfer


def compute_metrics(object_name:str,\
                    manager_paths,\
                    transforms_jsons,\
                    part_splats_path,\
                    part_splats_normalized_path,\
                    pose_estimates_path,\
                    articuation_estimates_path,\
                    output_dir,\
                    canonical_scene_id=0,\
                    root_part_id=1):
    torch.set_grad_enabled(False)
    # get the data from our reconstruction
    print(f"loading our reconstruction...")
    splat_managers = [load_base_model(path) for path in manager_paths]
    dataparser_scales = [splatmanager.dataparser_scale for splatmanager in splat_managers]
    dataparser_tf_matrices = [splatmanager.dataparser_tf_matrix for splatmanager in splat_managers]
    transforms_jsons_data = [json.load(open(path)) for path in transforms_jsons]
    gt_configuration_values = []
    gt_part_pose_values = []
    for scene_idx in range(len(transforms_jsons_data)):
        gt_configuration_values.append(transforms_jsons_data[scene_idx]["configurations"][str(scene_idx)])
        gt_part_pose_values.append(transforms_jsons_data[scene_idx]["gt_part_world_poses"][str(scene_idx)])

    n_parts = splat_managers[0].num_parts
    recon_part_splats = load_part_gaussians(part_splats_path)
    recon_part_splats_normalized = load_part_gaussians(part_splats_normalized_path)
    pose_estimates = load_pose_estimator(pose_estimates_path)
    articulation_estimates = load_articulation_estimates(articuation_estimates_path)

    # get our part splats out and convert them to ply's
    print(f"converting our reconstruction to ply...")
    recon_canonical_part_gaussians = get_canonical_part_gaussians(canonical_scene_id,\
                                                                recon_part_splats,\
                                                                dataparser_tf_matrices,\
                                                                dataparser_scales) # n_parts, (gauss_params)
    
    print(f"len of articulations: {len(articulation_estimates)}")
    print(f"len of config vector: {len(gt_configuration_values[0])}")
    print(f"len of config vector1: {len(gt_configuration_values[1])}")
    for articulation_idx in range(len(articulation_estimates)):
        # get the joint param at scene 0,1 and subtract
        gt_config_0 = gt_configuration_values[0][articulation_idx]
        gt_config_1 = gt_configuration_values[1][articulation_idx]
        predicted_joint = articulation_estimates[articulation_idx]["predicted_joint"]
        gt_part_motion = np.abs(gt_config_0 - gt_config_1)
        joint_params = predicted_joint.joint_params.cpu().detach().numpy()

        if(type(predicted_joint) == PrismaticJoint):
                # have to scale prismatic joint params by the dataparser scale
                joint_params /= dataparser_scales[canonical_scene_id]
        if(type(predicted_joint) == RevoluteJoint):
            # turn the joint params into degrees
            joint_params = joint_params * 180.0 / np.pi
            gt_part_motion = np.abs(gt_part_motion * 180.0 / np.pi)

        pred_part_motion = np.abs(joint_params[0] - joint_params[1])

        print(f"gt part motion: {gt_part_motion}")
        print(f"predicted part motion: {pred_part_motion}")
        print(f"param delta for idx {articulation_idx}: {np.abs(pred_part_motion - gt_part_motion)}")

    


if __name__=="__main__":
    print("Computing metrics!")

    parser = argparse.ArgumentParser(description="Computes metrics vs. ground truth for pretrained splats, articulations, and poses")

    parser.add_argument('--object_name', 
                        type=str,
                        help='name to use in lookup for GT sapien data',
                        default="")
    
    parser.add_argument('--manager_paths', 
                        type=str,
                        help='pths of the original scenes',
                        default="")
    
    parser.add_argument('--transforms_jsons', 
                        type=str,
                        help='transforms.jsons of the original scenes with gt poses and configurations',
                        default="")

    parser.add_argument('--part_splats', 
                        type=str,
                        help='pth of the learned canonical parts',
                        default="")
    
    parser.add_argument('--part_splats_normalized', 
                        type=str,
                        help='pth of the learned canonical parts where root has been set to identity',
                        default="")
    
    parser.add_argument('--pose_estimates',
                        type=str,
                        help="pth of the learned part poses",
                        default="")
    
    parser.add_argument('--articulation_estimates',
                        type=str,
                        help="pickle of the learned articulations",
                        default="")
    
    parser.add_argument('--root_part_id',
                        type=int,
                        help="id of the static part",
                        default=1)
    
    parser.add_argument('--output_dir',
                        type=str,
                        help="output directory for metrics",
                        default="")
    
    args = parser.parse_args()
    manager_paths = args.manager_paths.split(",")
    transforms_jsons = args.transforms_jsons.split(",")
    compute_metrics(args.object_name, manager_paths, transforms_jsons, args.part_splats, args.part_splats_normalized, args.pose_estimates, args.articulation_estimates, args.output_dir, root_part_id=args.root_part_id)
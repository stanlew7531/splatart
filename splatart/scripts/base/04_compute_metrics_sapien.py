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


# Get canonical parts from obj files
# Read .ply file
paris_to_sapien = {
    'blade': '103706',
    'laptop': '10211',
    'foldchair': '102255',
    'oven': '101917',
    'fridge': '10905',
    'scissor': '11100',
    'stapler': '103111',
    'USB': '100109',
    'washer': '103776',
    'storage': '45135'
}

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

# function gets the underlying sapien mesh data for the parts/objects we care about for later comparison
def load_sapien_data(object_name:str, sapien_root_dir="/media/stanlew/Data/paris_dataset/dataset/data/sapien"):
    sapien_dir = os.path.join(sapien_root_dir, object_name, paris_to_sapien[object_name], "textured_objs")

    start_dir = os.path.join(sapien_dir, "start")
    end_dir = os.path.join(sapien_dir, "end")
    trans_json_fname = os.path.join(sapien_dir,"trans.json")
    trans_json_data = json.load(open(trans_json_fname, 'r'))

    # correct for convention differences between sapien and reconstruction
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -np.pi/2,))

    start_whole_ply_fname = os.path.join(start_dir, "start_rotate.ply")
    start_whole_ply = o3d.io.read_triangle_mesh(start_whole_ply_fname)
    start_whole_ply.rotate(R, center=(0, 0, 0))
    start_static_ply_fname = os.path.join(start_dir, "start_static_rotate.ply")
    start_static_ply = o3d.io.read_triangle_mesh(start_static_ply_fname)
    start_static_ply.rotate(R, center=(0, 0, 0))
    start_dynamic_ply_fname = os.path.join(start_dir, "start_dynamic_rotate.ply")
    start_dynamic_ply = o3d.io.read_triangle_mesh(start_dynamic_ply_fname)
    start_dynamic_ply.rotate(R, center=(0, 0, 0))

    end_whole_ply_fname = os.path.join(end_dir, "end_rotate.ply")
    end_whole_ply = o3d.io.read_triangle_mesh(end_whole_ply_fname)
    end_whole_ply.rotate(R, center=(0, 0, 0))
    end_static_ply_fname = os.path.join(end_dir, "end_static_rotate.ply")
    end_static_ply = o3d.io.read_triangle_mesh(end_static_ply_fname)
    end_static_ply.rotate(R, center=(0, 0, 0))
    end_dynamic_ply_fname = os.path.join(end_dir, "end_dynamic_rotate.ply")
    end_dynamic_ply = o3d.io.read_triangle_mesh(end_dynamic_ply_fname)
    end_dynamic_ply.rotate(R, center=(0, 0, 0))

    plys_to_return = {
        "start":{"whole": start_whole_ply, "static": start_static_ply, "dynamic": start_dynamic_ply},
        "end":{"whole": end_whole_ply, "static": end_static_ply, "dynamic": end_dynamic_ply}
    }

    # get the joint parameter information from the json data
    joint_type = trans_json_data["input"]["motion"]["type"]
    joint_param_ends = trans_json_data["input"]["motion"][joint_type]
    joint_param_ends = np.array(joint_param_ends)

    joint_axis_origin = trans_json_data["trans_info"]["axis"]["o"]
    joint_axis_direction = trans_json_data["trans_info"]["axis"]["d"]

    joint_axis_origin = np.array(joint_axis_origin)
    joint_axis_direction = np.array(joint_axis_direction)

    # rotate the joint axis dir to match the convention change
    joint_axis_direction = R @ joint_axis_direction

    return plys_to_return, joint_param_ends, joint_axis_origin, joint_axis_direction

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
                    part_splats_path,\
                    pose_estimates_path,\
                    articuation_estimates_path,\
                    output_dir,\
                    canonical_scene_id=0,\
                    static_recon_part_id=2,\
                    dynamic_recon_part_id=3,\
                    sapien_root_dir="/media/stanlew/Data/paris_dataset/dataset/data/sapien"):
    torch.set_grad_enabled(False)
    # get the data from our reconstruction
    print(f"loading our reconstruction...")
    splat_managers = [load_base_model(path) for path in manager_paths]
    dataparser_scales = [splatmanager.dataparser_scale for splatmanager in splat_managers]
    dataparser_tf_matrices = [splatmanager.dataparser_tf_matrix for splatmanager in splat_managers]
    print(f"dataparser tf matric start: {dataparser_tf_matrices[0]}")
    n_parts = splat_managers[0].num_parts
    recon_part_splats = load_part_gaussians(part_splats_path)
    pose_estimates = load_pose_estimator(pose_estimates_path)
    articulation_estimates = load_articulation_estimates(articuation_estimates_path)

    # get the object data from sapien dataset
    print(f"loading sapien data...")
    sapien_plys, sapien_joint_params, sapien_axis_origin, sapien_axis_direction = load_sapien_data(object_name, sapien_root_dir)

    # get our part splats out and convert them to ply's
    print(f"converting our reconstruction to ply...")
    recon_canonical_part_gaussians = get_canonical_part_gaussians(canonical_scene_id,\
                                                                recon_part_splats,\
                                                                dataparser_tf_matrices,\
                                                                dataparser_scales) # n_parts, (gauss_params)
    os.makedirs("tmp", exist_ok=True)

    # bit of a hack - we save the plys to disk so that we can then load them back
    # and compute the metrics on them
    recon_plys = []
    for i in tqdm(range(n_parts)):
        gauss_params_to_ply(recon_canonical_part_gaussians[i], f"tmp/part_{i}.ply")
        recon_plys.append(o3d.io.read_point_cloud(f"tmp/part_{i}.ply"))

    whole_recon_ply = o3d.io.read_point_cloud(f"tmp/part_{static_recon_part_id}.ply")
    whole_recon_ply.points.extend(recon_plys[dynamic_recon_part_id].points)
    o3d.io.write_point_cloud(f"tmp/whole_recon.ply", whole_recon_ply)

    # uncomment if we want to inspect the GT plys in blender or whatever next to the recon
    # o3d.io.write_triangle_mesh(f"tmp/gt_part_start_static.ply", sapien_plys["start"]["static"])
    # o3d.io.write_triangle_mesh(f"tmp/gt_part_start_dynamic.ply", sapien_plys["start"]["dynamic"])

    cd_s = get_chamfer_distance_metric(recon_plys[static_recon_part_id], sapien_plys["start"]["static"])
    print(f"Chamfer distance for static part: {cd_s}")
    cd_d = get_chamfer_distance_metric(recon_plys[dynamic_recon_part_id], sapien_plys["start"]["dynamic"])
    print(f"Chamfer distance for dynamic part: {cd_d}")
    cd_w = get_chamfer_distance_metric(whole_recon_ply, sapien_plys["start"]["whole"])
    print(f"Chamfer distance for whole body: {cd_w}")

    # part motion and joint metrics
    gt_part_motion = np.abs(sapien_joint_params[0] - sapien_joint_params[1])

    # get our estimated articulation
    for joint in articulation_estimates:
        src_part = joint["src_part"]
        tgt_part = joint["tgt_part"]
        print(f"src part: {src_part}, tgt part: {tgt_part}")
        if(src_part == static_recon_part_id and tgt_part == dynamic_recon_part_id):
            # check if we have a prismatic or revolute joint
            predicted_joint = joint["predicted_joint"]
            joint_params = predicted_joint.joint_params.cpu().detach().numpy()
            joint_pre_tf = predicted_joint.get_6dof_as_mat(predicted_joint.pre_tf)
            joint_axis = predicted_joint.joint_axis

            # check the type of predicted_joint
            if(type(predicted_joint) == PrismaticJoint):
                # have to scale prismatic joint params by the dataparser scale
                joint_params /= dataparser_scales[canonical_scene_id]
            if(type(predicted_joint) == RevoluteJoint):
                joint_pre_tf = predicted_joint.get_6dof_as_mat(predicted_joint.post_tf)
                # turn the joint params into degrees
                joint_params = joint_params * 180.0 / np.pi
            pred_part_motion = np.abs(joint_params[0] - joint_params[1])
            
            # transform the pre_tf by the dataparser matrix
            joint_dp_tf = torch.linalg.inv(dataparser_tf_matrices[canonical_scene_id]) @ joint_pre_tf
            # have to scale the joint_pre_tf by the dataparser scale
            joint_dp_tf /= dataparser_scales[canonical_scene_id]
            est_joint_origin = joint_dp_tf[:3, 3]
            # get the joint axis in world coordinates
            joint_axis = joint_dp_tf[:3, :3] @ joint_axis
            joint_axis = joint_axis / torch.norm(joint_axis) # normalize
    
    print(f"sapien axis origin: {sapien_axis_origin}")
    print(f"joint axis origin: {est_joint_origin}")
    print(f"sapien joint axis dir:{sapien_axis_direction}")
    print(f"joint axis dir: {joint_axis}")
    print(f"sapien part motion: {gt_part_motion}")
    print(f"predicted part motion: {pred_part_motion}")


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

    parser.add_argument('--part_splats', 
                        type=str,
                        help='pth of the learned canonical parts',
                        default="")
    
    parser.add_argument('--pose_estimates',
                        type=str,
                        help="pth of the learned part poses",
                        default="")
    
    parser.add_argument('--articulation_estimates',
                        type=str,
                        help="pickle of the learned articulations",
                        default="")
    
    parser.add_argument('--static_part_id',
                        type=int,
                        help="id of the static part",
                        default=2)
    
    parser.add_argument('--dyn_part_id',
                        type=int,
                        help="id of the dynamic part",
                        default=3)
    
    parser.add_argument('--output_dir',
                        type=str,
                        help="output directory for metrics",
                        default="")
    
    args = parser.parse_args()
    manager_paths = args.manager_paths.split(",")
    compute_metrics(args.object_name, manager_paths, args.part_splats, args.pose_estimates, args.articulation_estimates, args.output_dir, static_recon_part_id=args.static_part_id, dynamic_recon_part_id=args.dyn_part_id)
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

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats, apply_mat_tf_to_gauss_params
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint, tree_create_tfs_from_config
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.networks.PoseEstimator import PoseEstimator
from splatart.utils.helpers import gauss_params_to_ply, combine_guass_params

def load_articulation_estimates(articulation_estimates_path:str):
    # get joint for joint metrics.
    with open(articulation_estimates_path, 'rb') as f:
        config_vector = pkl.load(f)
    return config_vector

def load_base_model(input_model_path:str)->SplatManagerSingle:
    return torch.load(input_model_path)

def load_part_gaussians(part_splats_path:str):
    return torch.load(part_splats_path)

def load_pose_estimator(pose_estimates_path:str)->PoseEstimator:
    return torch.load(pose_estimates_path)

def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth, use_semantics=True)

def render_images(object_name:str,\
                    manager_paths,\
                    dataset_paths,\
                    part_splats_path,\
                    pose_estimates_path,\
                    articuation_estimates_path,\
                    output_dir,\
                    canonical_scene_id=0,\
                    root_part_id=1):
    torch.set_grad_enabled(False)
    splat_managers = [load_base_model(path) for path in manager_paths]
    dataparser_scales = [splatmanager.dataparser_scale for splatmanager in splat_managers]
    dataparser_tf_matrices = [splatmanager.dataparser_tf_matrix for splatmanager in splat_managers]
    print(f"dataparser tf matric start: {dataparser_tf_matrices[0]}")
    n_parts = splat_managers[0].num_parts
    recon_part_splats = load_part_gaussians(part_splats_path)
    pose_estimates = load_pose_estimator(pose_estimates_path)
    articulation_estimates = load_articulation_estimates(articuation_estimates_path)
    
    splat_datasets = [get_dataset(path) for path in dataset_paths]
    dataset = splat_datasets[0]
    _, inspection_data = dataset[30]
    gt_cam_pose = inspection_data["transform_matrix"].to(device="cuda:0")

    # static camera for figure generation
    # gt_cam_pose = torch.Tensor([[ 6.6417e-01,  4.8176e-01, -5.7165e-01, -2.2294e+00],
    #     [-7.4758e-01,  4.2801e-01, -5.0787e-01, -1.9807e+00],
    #     [-5.8196e-17,  7.6467e-01,  6.4443e-01,  2.5133e+00],
    #     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).to(device="cuda:0")

    cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx],\
                                [0, dataset.fl_y, dataset.cy],\
                                [0, 0, 1]])
    
    batch_size = 1
    width = dataset.width
    height = dataset.height

    # get the set of each parts gaussians for transformation

    # the gaussian parameters for each part, in its own frame
    part_frames, scene_part_frame_gauss_params = pose_estimates.get_part_splat_frames(recon_part_splats)
    canonical_part_frames = part_frames[canonical_scene_id]
    canonical_part_frame_gauss_params = scene_part_frame_gauss_params[canonical_scene_id]
    # get the joint between the static and dynamic part
    # joint = articulation_estimates[0]["predicted_joint"]
    joint_params_start = torch.tensor([articulation_estimates[i]["predicted_joint"].joint_params[0] for i in range(len(articulation_estimates))])
    joint_params_end = torch.tensor([articulation_estimates[i]["predicted_joint"].joint_params[1] for i in range(len(articulation_estimates))])
    
    n_steps = 30
    joint_deltas = joint_params_end - joint_params_start
    joint_steps = joint_deltas / (n_steps - 1)
    joint_params = joint_params_start + joint_steps * torch.arange(n_steps).unsqueeze(1).to(joint_params_start.device)
    joint_deltas_2 = joint_params_start - joint_params_end
    joint_steps_2 = joint_deltas_2 / (n_steps - 1)
    joint_params_2 = joint_params_end + joint_steps_2 * torch.arange(n_steps).unsqueeze(1).to(joint_params_end.device)

    joint_params = torch.cat([joint_params, joint_params_2], dim=0)

    root_part_id = 1

    # get the initial pose for the static recon part
    # print(f"for static part {static_recon_part_id}, the initial pose is {canonical_part_frames[static_recon_part_id]}")

    i = 0
    root_part_frame = canonical_part_frames[root_part_id].clone()
    # repeat the root part frame for each part
    root_part_frame = root_part_frame.unsqueeze(0).expand(n_parts, -1, -1)
    starting_part_poses = canonical_part_frames.clone()
    print(f"art values: {joint_params}")
    print(f"articulation estimate: {articulation_estimates}")
    for value in joint_params:
        print(f"articulation value: {value}")
        # get the estimated transforms for each part idx
        part_poses = tree_create_tfs_from_config(articulation_estimates, root_part_id, value, n_parts, part_poses=starting_part_poses) # clone is just to prevent pass by ref/val issues
        # apply the part poses to each canonical part
        render_gauss_params = []
        for part_idx in range(1,n_parts): # 1 because ignore background
            transformed_part_params = apply_mat_tf_to_gauss_params(part_poses[part_idx], canonical_part_frame_gauss_params[part_idx].copy())
            render_gauss_params.append(transformed_part_params)
            gauss_params_to_ply(transformed_part_params, os.path.join(output_dir, f"render_part_{part_idx}_gauss_params_{i}.ply"))

        # combine the gauss params into a single object
        render_gauss_params = combine_guass_params(render_gauss_params)
        # render the gauss params at gt_camera_pose
        render_image_results = splat_managers[0].render_gauss_params_at_campose(\
                    gt_cam_pose, cam_intrinsic.expand(batch_size, -1, -1), width, height, render_gauss_params, is_semantics=False)
        
        render = render_image_results[0]
        image = render[0].detach().cpu().numpy()
        out_fname = os.path.join(output_dir, f"rendered_image_{i:02d}.png")
        os.makedirs(output_dir, exist_ok=True)
        cv.imwrite(out_fname, image[:,:,::-1] * 255.0)
        i += 1
        print(f"rendered image {i} to {out_fname}")
                    



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
    
    parser.add_argument('--root_part_id',
                        type=int,
                        help="id of the static part",
                        default=1)
    
    parser.add_argument('--output_dir',
                        type=str,
                        help="output directory for metrics",
                        default="")
    
    parser.add_argument('--splat_model_datasets',
                        type=str,
                        help="list of datasets used to train the splats",
                        default="")
    
    args = parser.parse_args()
    manager_paths = args.manager_paths.split(",")
    dataset_paths = args.splat_model_datasets.split(",")
    render_images(args.object_name, manager_paths, dataset_paths, args.part_splats, args.pose_estimates, args.articulation_estimates, args.output_dir, root_part_id=args.root_part_id )
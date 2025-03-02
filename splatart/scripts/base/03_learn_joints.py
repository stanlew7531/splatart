from functools import *
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from typing import Optional
import argparse
import pytorch3d.transforms as p3dt
import json

import torch

from splatart.gui.CloudVis import CloudVisApp

from splatart.utils.lie_utils import SE3, SE3Exp, SO3Exp
from splatart.managers.JointManager import JointManager
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint, SingleAxisJoint
from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats, means_eulers_to_mat
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.utils.helpers import convert_cam_opengl_opencv
from splatart.networks.PoseEstimator import PoseEstimator

import pickle

def load_pose_estimator(pose_estimates_path:str)->PoseEstimator:
    return torch.load(pose_estimates_path)

def load_part_gaussians(part_splats_path:str):
    return torch.load(part_splats_path)

def get_scene_poses(scene_id:int=0, pose_estimates:PoseEstimator=None):
    part_means = pose_estimates.part_means[scene_id]
    part_eulers = pose_estimates.part_eulers[scene_id]

    return means_eulers_to_mat(part_means, part_eulers)

def get_part_poses(part_id:int=0, pose_estimates:PoseEstimator=None):
    part_means = pose_estimates.part_means[...,part_id, :]
    part_eulers = pose_estimates.part_eulers[...,part_id,:]
    part_mat = means_eulers_to_mat(part_means, part_eulers).to("cuda:0")
    return part_mat

def get_canonical_part_gaussians(canonical_scene_id:int=0, part_gaussians=None):
    return part_gaussians[canonical_scene_id]

# want to do a breadth first search to explore the configuration vector
def tree_explore_config_vector(configuration_vector, root_part_idx, visited=[], to_return=[]):
    visited.append(root_part_idx)
    frontier = []
    for entry in configuration_vector:
        if entry["src_part"] == root_part_idx and entry["tgt_part"] not in visited:
            print(f"adding from: {entry['src_part']} to {entry['tgt_part']}")
            to_return.append(entry)
            frontier.append(entry["tgt_part"])
            visited.append(entry["tgt_part"])
    for tgt_entry in frontier:
        tree_explore_config_vector(configuration_vector, tgt_entry, visited, to_return)

def learn_joints(part_splats_path:str, pose_estimates_path:str, output_dir:str, canonical_scene_id:int=0, err_thresh:float=0.1):
    pose_estimates = load_pose_estimator(pose_estimates_path)
    object_part_gaussians = load_part_gaussians(part_splats_path)
    num_scenes = len(object_part_gaussians)
    num_parts = len(object_part_gaussians[canonical_scene_id])

    print(f"got {len(object_part_gaussians)} scenes")
    print(f"got {num_parts} parts")

    canonical_part_gaussians = get_canonical_part_gaussians(canonical_scene_id, object_part_gaussians)
    configuration_vector = []
    # create prismatic and revolute joint estimates between each part
    for src_part_idx in range(num_parts):
        for dst_part_idx in range(num_parts):
            # dont need to check part against itself
            if src_part_idx == dst_part_idx:
                continue

            # get the src and dst part poses 
            src_part_poses = get_part_poses(src_part_idx, pose_estimates) # (num_scenes, 4, 4)
            # print(f"src part poses shape: {src_part_poses.shape}")
            dst_part_poses = get_part_poses(dst_part_idx, pose_estimates) # (num_scenes, 4, 4)
            # print(f"dst part poses shape: {src_part_poses.shape}")

            # get the canonical part gaussians
            src_part_canon_means = canonical_part_gaussians[src_part_idx]["means"] # (num_means, 3)
            dst_part_canon_means = canonical_part_gaussians[dst_part_idx]["means"] # (num_means, 3)

            # homogenize the src and dst part means
            src_part_canon_means = torch.cat([\
                                    src_part_canon_means,\
                                    torch.ones(src_part_canon_means.shape[0], 1)\
                                        .to(src_part_canon_means.device)\
                                    ], dim=1).unsqueeze(-1) # (num_means, 4, 1)
            dst_part_canon_means = torch.cat([\
                                    dst_part_canon_means,\
                                    torch.ones(dst_part_canon_means.shape[0], 1)\
                                        .to(dst_part_canon_means.device)\
                                    ], dim=1).unsqueeze(-1) # (num_means, 4, 1)
            
            # if either part has no means (i.e. it is the background), skip it
            if src_part_canon_means.shape[0] == 0:
                print(f"skipping src part {src_part_idx} because it has no means")
                continue
            if dst_part_canon_means.shape[0] == 0:
                print(f"skipping dst part {dst_part_idx} because it has no means")
                continue
            
            # get the tfs between each scenes src and dst part poses
            pose_tfs = torch.inverse(src_part_poses[...]) @ dst_part_poses[...] # (num_scenes, 4, 4)
            # get the set of means we want to compare against for the dst part
            dst_means_for_compare = pose_tfs.unsqueeze(1) @ dst_part_canon_means # num_scenes, num_means, 4, 1


            # estimate the prismatic joint
            prismatic_estimate = PrismaticJoint(num_scenes).to("cuda:0")
            prismatic_estimate.get_initial_estimate(pose_tfs.detach())
            prismatic_tf_estimates = prismatic_estimate.get_transform(prismatic_estimate.joint_params) # N, 4, 4
            prismatic_means_estimates = prismatic_tf_estimates.unsqueeze(1) @ dst_part_canon_means # N, num_means, 4, 1
            # estimate the revolute joint
            revolute_estimate = RevoluteJoint(num_scenes).to("cuda:0")
            revolute_estimate.get_initial_estimate(pose_tfs.detach())
            revolute_tf_estimates = revolute_estimate.get_transform(revolute_estimate.joint_params) # N, 4, 4
            revolute_means_estimates = revolute_tf_estimates.unsqueeze(1) @ dst_part_canon_means # N, num_means, 4, 1

            dst_means_for_compare = dst_means_for_compare.squeeze()[:,:,:3] # strip the homogenous coordinate (N, 3)
            prismatic_means_estimates = prismatic_means_estimates.squeeze()[:,:,:3] # strip the homogenous coordinate (N, 3)
            revolute_means_estimates = revolute_means_estimates.squeeze()[:,:,:3] # strip the homogenous coordinate (N, 3)

            prismatic_err = torch.linalg.norm(prismatic_means_estimates - dst_means_for_compare, dim=-1).mean()
            revolute_errr = torch.linalg.norm(revolute_means_estimates - dst_means_for_compare, dim=-1).mean() 

            print(f"prismatic error: {prismatic_err}, revolute error: {revolute_errr} with src: {src_part_idx}, tgt: {dst_part_idx}")
            to_ret_dict = {"src_part": src_part_idx, "tgt_part": dst_part_idx, "predicted_prismatic": prismatic_estimate, "predicted_revolute": revolute_estimate}
            print(f"prismatic components: {prismatic_estimate.get_gt_parameters()}, revolute components: {revolute_estimate.get_gt_parameters()}")
            if prismatic_err < err_thresh and prismatic_err < revolute_errr:
                to_ret_dict["predicted_joint"] = prismatic_estimate
                configuration_vector.append(to_ret_dict)
                print(f"predicted joint params: {prismatic_estimate.joint_params}")
            elif revolute_errr < err_thresh and revolute_errr < prismatic_err:
                to_ret_dict["predicted_joint"] = revolute_estimate
                configuration_vector.append(to_ret_dict)
                print(f"predicted joint params: {revolute_estimate.joint_params}")

    treeified_joints = []
    tree_explore_config_vector(configuration_vector, 1, [], treeified_joints)

    # pickle the joint matrix for later processing
    output_pickle_path = f"{output_dir}/configuration_vector.pkl"
    with open(output_pickle_path, "wb") as f:
        pickle.dump(treeified_joints, f)


    print("done")

if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Given the set of pre-learned part poses and canonical part splats, infer the joint parameters")

    parser.add_argument('--part_splats', 
                        type=str,
                        help='pth of the learned canonical parts',
                        default="")
    
    parser.add_argument('--pose_estimates',
                        type=str,
                        help="pth of the learned part poses",
                        default="")
    
    parser.add_argument('--output_dir',
                        type=str,
                        help="directory to save the learned joints to",
                        default="")
    
    args = parser.parse_args()
    
    learn_joints(args.part_splats, args.pose_estimates, args.output_dir)
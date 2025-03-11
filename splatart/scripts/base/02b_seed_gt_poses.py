import sys
import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_msssim import SSIM
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import copy

import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import HDBSCAN
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManagerSingle, load_managers_nerfstudio_single
from splatart.datasets.splat_train_dataset import SplatTrainDataset

def load_base_model(input_model_path:str)->SplatManagerSingle:
    return torch.load(input_model_path)

def load_og_model(input_model_path:str, num_parts:int):
    return load_managers_nerfstudio_single([input_model_path], num_parts)[0]

def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth, use_semantics=True)

def unpack_gaussian_params(object_gaussian_params, num_parts):
    to_return = []
    part_labels = torch.argmax(object_gaussian_params["features_semantics"], dim=1)
    for part_idx in range(num_parts):
        to_append = {
            "means": object_gaussian_params["means"][part_labels == part_idx],
            "quats": object_gaussian_params["quats"][part_labels == part_idx],
            "features_semantics": object_gaussian_params["features_semantics"][part_labels == part_idx],
            "features_dc": object_gaussian_params["features_dc"][part_labels == part_idx],
            "features_rest": object_gaussian_params["features_rest"][part_labels == part_idx],
            "opacities": object_gaussian_params["opacities"][part_labels == part_idx],
            "scales": object_gaussian_params["scales"][part_labels == part_idx],
        }
        to_append["means"].requires_grad = True
        to_append["quats"].requires_grad = True
        to_return.append(to_append)

    return to_return

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # red
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name="ICP Result", width=800, height=600)


def make_gt_estimate(manager_paths, og_manager_paths, dataset_paths):
    splat_managers = [load_base_model(path) for path in manager_paths]
    num_scenes = len(splat_managers)
    num_parts = splat_managers[0].num_parts
    og_managers = [load_og_model(path, num_parts) for path in og_manager_paths]
    splat_datasets = [get_dataset(path) for path in dataset_paths]
    # of size [num_scenes, num_parts, [gaussian_params]]
    object_parts_gauss_params = [unpack_gaussian_params(manager.object_gaussian_params, manager.num_parts) for manager in splat_managers]

    # we want to get the means of the parts in each scene - thats what we will perform icp on
    num_parts = len(object_parts_gauss_params[0])

    final_transforms = [(np.eye(4).tolist(), np.eye(4).tolist())]# set bg tf to identity
    print(f"num parts: {num_parts}")
    # skip part 0 because it is the BG
    for part_idx in range(1, num_parts):
        src_gt_part_pose = torch.tensor(splat_datasets[0].gt_poses[part_idx - 1])
        dst_gt_part_pose = torch.tensor(splat_datasets[1].gt_poses[part_idx - 1])

        src_means = object_parts_gauss_params[0][part_idx]["means"].detach().cpu().numpy()
        dst_means = object_parts_gauss_params[1][part_idx]["means"].detach().cpu().numpy()

        dataparser_tf_src = og_managers[0].dataparser_tf_matrix
        dataparser_tf_dst = og_managers[1].dataparser_tf_matrix

        dataparser_scale_src = og_managers[0].dataparser_scale
        dataparser_scale_dst = og_managers[1].dataparser_scale

        # transform gt part poses into the nerfstudio space
        src_gt_part_pose = torch.matmul(dataparser_tf_src, src_gt_part_pose.to(dataparser_tf_src))
        src_gt_part_pose[..., :3, 3] *= dataparser_scale_src

        dst_gt_part_pose = torch.matmul(dataparser_tf_dst, dst_gt_part_pose.to(dataparser_tf_dst))
        dst_gt_part_pose[..., :3, 3] *= dataparser_scale_dst

        final_transforms.append((src_gt_part_pose.cpu().numpy().tolist(), dst_gt_part_pose.cpu().numpy().tolist()))
    
    # save the transforms to a file
    output_path = os.path.join(os.path.dirname(manager_paths[0]), "gt_transforms.json")
    with open(output_path, "w") as f:
        json.dump(final_transforms, f)
    print(f"Saved transforms to {output_path}")

if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Given a list of splats already trained on RGB, learn segmentations for the parts")

    parser.add_argument('--splat_tf_manager_pths', 
                        type=str,
                        help='segmentation splats to learn on',
                        default="")
    
    parser.add_argument('--original_splat_manager_pth', 
                        type=str,
                        help='original splats (to get dataparser tf etc. from)',
                        default="")
    
    parser.add_argument('--splat_model_datasets',
                        type=str,
                        help="list of datasets used to train the splats",
                        default="")
    
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    manager_paths = args.splat_tf_manager_pths.split(",")
    og_manager_paths = args.original_splat_manager_pth.split(",")
    dataset_paths = args.splat_model_datasets.split(",")
    make_gt_estimate(manager_paths, og_manager_paths, dataset_paths)
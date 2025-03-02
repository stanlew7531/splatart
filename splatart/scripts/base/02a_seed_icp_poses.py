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

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats
from splatart.datasets.splat_train_dataset import SplatTrainDataset

def load_base_model(input_model_path:str)->SplatManagerSingle:
    return torch.load(input_model_path)

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

def make_icp_estimate(manager_paths, dataset_paths):
    splat_managers = [load_base_model(path) for path in manager_paths]
    splat_datasets = [get_dataset(path) for path in dataset_paths]
    # of size [num_scenes, num_parts, [gaussian_params]]
    object_parts_gauss_params = [unpack_gaussian_params(manager.object_gaussian_params, manager.num_parts) for manager in splat_managers]

    # we want to get the means of the parts in each scene - thats what we will perform icp on
    num_parts = len(object_parts_gauss_params[0])

    final_transforms = []

    for part_idx in range(num_parts):
        src_means = object_parts_gauss_params[0][part_idx]["means"].detach().cpu().numpy()
        dst_means = object_parts_gauss_params[1][part_idx]["means"].detach().cpu().numpy()
        src_centroid = np.mean(src_means, axis=0)
        dst_centroid = np.mean(dst_means, axis=0)

        # move the means to the origin
        src_means -= src_centroid
        dst_means -= dst_centroid

        # make source and destination point clouds
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_means)
        dst_pcd = o3d.geometry.PointCloud()
        dst_pcd.points = o3d.utility.Vector3dVector(dst_means)

        # compute the normals
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        dst_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        initial_transformation = np.eye(4)
        initial_transformation[:3, :3] = np.eye(3)

        random_spins = 1000

        cur_best_registration = None
        cur_best_acc = float("inf")
        for i in tqdm(range(random_spins)):
            random_rotation = R.from_euler('xyz', np.random.rand(3) * 2 * np.pi)
            initial_transformation[:3, :3] = random_rotation.as_matrix()

            reg_p2p = o3d.pipelines.registration.registration_icp(
                src_pcd, dst_pcd, 0.01,
                initial_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            if reg_p2p.inlier_rmse < cur_best_acc:
                cur_best_registration = reg_p2p
                cur_best_acc = reg_p2p.inlier_rmse

        draw_registration_result(src_pcd, dst_pcd, cur_best_registration.transformation)

        full_src_tf = np.eye(4)
        full_src_tf[:3, :3] = cur_best_registration.transformation[:3, :3]
        full_src_tf[:3, 3] = src_centroid

        full_dst_tf = np.eye(4)
        full_dst_tf[:3, :3] = np.eye(3)
        full_dst_tf[:3, 3] = dst_centroid

        final_transforms.append((full_src_tf.tolist(), full_dst_tf.tolist()))
    
    # save the transforms to a file
    output_path = os.path.join(os.path.dirname(manager_paths[0]), "icp_transforms.json")
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
    
    parser.add_argument('--splat_model_datasets',
                        type=str,
                        help="list of datasets used to train the splats",
                        default="")
    
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    manager_paths = args.splat_tf_manager_pths.split(",")
    dataset_paths = args.splat_model_datasets.split(",")
    make_icp_estimate(manager_paths, dataset_paths)
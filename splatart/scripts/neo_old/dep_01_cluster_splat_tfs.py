import sys
import os
import argparse
import torch
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

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import HDBSCAN
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManager, load_managers_nerfstudio
from splatart.datasets.splat_train_dataset import SplatTrainDataset

def load_base_model(input_model_path:str):
    return load_managers_nerfstudio([input_model_path], 0)[0]

def load_tf_model(input_model_path:str):
    manager = torch.load(input_model_path)
    return manager

def means_quats_to_mat(means, quats):
    to_return = torch.zeros((means.shape[0], 4, 4)).to(means)
    to_return[:, :3, :3] = p3dt.quaternion_to_matrix(quats)
    to_return[:, :3, 3] = means
    to_return[:, 3, 3] = 1
    return to_return

def cluster_models(base_model_pth:str, transformed_model_dir:str, other_model_dataset:str, n_samples:int = 1000, n_parts:int = 2, dist_thresh = 0.02):
    base_splat_manager = load_base_model(base_model_pth)
    dataset = SplatTrainDataset(other_model_dataset)
    dataset_len = dataset.__len__()
    width = dataset.width
    height = dataset.height
    print(f"Loaded dataset with {dataset_len} entries of height {height} and width {width}")
    cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx], [0, dataset.fl_y, dataset.cy], [0, 0, 1]])

    transformed_model_pth = os.path.join(transformed_model_dir, "splat_manager_transformed.pth")
    tf_splat_manager = load_tf_model(transformed_model_pth)

    base_means = base_splat_manager.object_gaussian_params["means"]
    base_quats = base_splat_manager.object_gaussian_params["quats"]

    tf_means = tf_splat_manager.object_gaussian_params["means"]
    tf_quats = tf_splat_manager.object_gaussian_params["quats"]

    base_tfs = means_quats_to_mat(base_means, base_quats)
    tf_tfs = means_quats_to_mat(tf_means, tf_quats)

    base_gauss_params = base_splat_manager.object_gaussian_params
    tf_gauss_params = tf_splat_manager.object_gaussian_params

    # randomly sample n_samples indices from the base splat
    sampled_idxs = torch.randint(0, base_means.shape[0], (n_samples,))
    
    sampled_base_tfs = base_tfs[sampled_idxs]
    sampled_tf_tfs = tf_tfs[sampled_idxs]

    max_close_points_idx = 0
    max_close_points_num = 0
    max_close_points = None
    base_part_gauss_params = None
    tf_part_gauss_params = None
    for i in tqdm(range(n_samples)):
        base_tf = sampled_base_tfs[i]
        tf_tf = sampled_tf_tfs[i]

        base_tfs_in_frame = torch.linalg.inv(base_tf) @ base_tfs
        tf_tfs_in_frame = torch.linalg.inv(tf_tf) @ tf_tfs

        base_points = base_tfs_in_frame[:, :3, 3]
        tf_points = tf_tfs_in_frame[:, :3, 3]

        # get the deltas between base and tf points
        deltas = tf_points - base_points

        # cluster using HDBSCAN
        hdb = HDBSCAN().fit(deltas.cpu().detach().numpy())
        print(f"for sample {i}, found {hdb.labels_.max()} clusters")

    #     # get the distances between base and tf points
    #     distances = torch.norm(base_points - tf_points, dim=-1)
    #     # get the points within the distance threshold
    #     close_points = torch.nonzero(distances < dist_thresh)
    #     # print(f"close points: {close_points}")
    #     # print(f"for sample {i}, found {close_points.shape[0]} close points")
    #     if close_points.shape[0] > max_close_points_num:
    #         max_close_points_num = close_points.shape[0]
    #         max_close_points_idx = i
    #         max_close_points = close_points.squeeze()
    #         base_part_gauss_params = {
    #                 "means": base_gauss_params["means"][max_close_points],
    #                 "scales": base_gauss_params["scales"][max_close_points],
    #                 "quats": base_gauss_params["quats"][max_close_points],
    #                 "features_dc": base_gauss_params["features_dc"][max_close_points],
    #                 "features_rest": base_gauss_params["features_rest"][max_close_points],
    #                 "opacities": base_gauss_params["opacities"][max_close_points],
    #                 "features_semantics": base_gauss_params["features_semantics"][max_close_points]\
    #                     if "features_semantics" in base_gauss_params.keys()\
    #                     else None,
    #                 "tf_quats": base_gauss_params["tf_quats"],
    #                 "tf_trans": base_gauss_params["tf_trans"]
    #             }
    #         tf_part_gauss_params = {
    #             "means": tf_gauss_params["means"][max_close_points],
    #             "scales": tf_gauss_params["scales"][max_close_points],
    #             "quats": tf_gauss_params["quats"][max_close_points],
    #             "features_dc": tf_gauss_params["features_dc"][max_close_points],
    #             "features_rest": tf_gauss_params["features_rest"][max_close_points],
    #             "opacities": tf_gauss_params["opacities"][max_close_points],
    #             "features_semantics": tf_gauss_params["features_semantics"][max_close_points]\
    #                 if "features_semantics" in tf_gauss_params.keys()\
    #                 else None,
    #             "tf_quats": tf_gauss_params["tf_quats"],
    #             "tf_trans": tf_gauss_params["tf_trans"]
    #         }
    # print(f"close points: {close_points}")
    # print(f"max close points: {max_close_points} at index {max_close_points_idx}")

    # render_data = dataset[0][1]
    # cam_pose = render_data["transform_matrix"].unsqueeze(0)

    # # render the base and tf gauss params
    # base_render_rgb, base_render_alpha, base_info = base_splat_manager.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, base_part_gauss_params)
    # tf_render_rgb, tf_render_alpha, tf_info = base_splat_manager.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, tf_part_gauss_params)

    # # save the rendered images
    # base_render_rgb = base_render_rgb.squeeze().cpu().detach().numpy() * 255
    # tf_render_rgb = tf_render_rgb.squeeze().cpu().detach().numpy() * 255

    # gt_base_img = render_data["rgb"].cpu().numpy()
    # cv.imwrite("cluster_base_render_rgb.png", base_render_rgb)
    # cv.imwrite("cluster_tf_render_rgb.png", tf_render_rgb)
    # cv.imwrite("cluster_gt_base_img.png", gt_base_img)    


if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--input_model_pth', 
                        type=str,
                        help='pre-trained splat for source of clusters',
                        default="")
    
    parser.add_argument('--transformed_model_dir', 
                        type=str,
                        help='pre-trained, transformed splat to cluster against',
                        default="")
    
    parser.add_argument('--other_model_dataset',
                        type=str,
                        help="directory of the other scene's dataset",
                        default="")
    
    args = parser.parse_args()
    cluster_models(args.input_model_pth, args.transformed_model_dir, args.other_model_dataset)
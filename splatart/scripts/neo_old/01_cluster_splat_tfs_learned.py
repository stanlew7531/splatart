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


class PartsSeperator(torch.nn.Module):
    def __init__(self, num_parts, num_splats):
        super(PartsSeperator, self).__init__()
        self.num_parts = num_parts
        self.num_splats = num_splats
        self.part_means = torch.nn.Parameter(torch.zeros((num_parts, 3, 2)))
        self.part_eulers = torch.nn.Parameter(torch.zeros((num_parts, 3, 2)))
        self.part_scores = torch.nn.Parameter(torch.rand((num_parts, num_splats)))

    def forward_and_loss(self, src_splat_tfs, dst_splat_tfs):
        part_probabilities = torch.nn.functional.softmax(self.part_scores, dim=0).to(src_splat_tfs.device)

        src_part_frames = torch.zeros((self.num_parts, 4, 4)).to(src_splat_tfs)
        dst_part_frames = torch.zeros((self.num_parts, 4, 4)).to(dst_splat_tfs)

        src_part_frames[:, :3, :3] = p3dt.euler_angles_to_matrix(self.part_eulers[:,:,0], "XYZ")
        src_part_frames[:, :3, 3] = self.part_means[:,:,0]
        src_part_frames[:, 3, 3] = 1

        dst_part_frames[:, :3, :3] = p3dt.euler_angles_to_matrix(self.part_eulers[:,:,1], "XYZ") 
        dst_part_frames[:, :3, 3] = self.part_means[:,:,1]
        dst_part_frames[:, 3, 3] = 1

        loss_tot = 0.0

        for i in range(self.num_parts):
            src_part_frame = src_part_frames[i]
            dst_part_frame = dst_part_frames[i]

            src_splat_tfs_part_frame = torch.linalg.inv(src_part_frame) @ src_splat_tfs
            dst_splat_tfs_part_frame = torch.linalg.inv(dst_part_frame) @ dst_splat_tfs

            src_splat_means = src_splat_tfs_part_frame[:, :3, 3]
            dst_splat_means = dst_splat_tfs_part_frame[:, :3, 3]

            per_splat_errors = torch.norm(src_splat_means - dst_splat_means, dim=-1)
            weighted_errors = per_splat_errors * part_probabilities[i]
            loss_tot += torch.sum(weighted_errors)

        return loss_tot
    
    def get_part_assignments(self):
        part_probabilities = torch.nn.functional.softmax(self.part_scores, dim=0)
        return torch.argmax(part_probabilities, dim=0)
    
    def get_part_trans_quats(self):
        src_part_rots = p3dt.euler_angles_to_matrix(self.part_eulers[:,:,0], "XYZ")
        dst_part_rots = p3dt.euler_angles_to_matrix(self.part_eulers[:,:,1], "XYZ")
        src_part_quats = p3dt.matrix_to_quaternion(src_part_rots)
        dst_part_quats = p3dt.matrix_to_quaternion(dst_part_rots)

        src_part_trans = self.part_means[:,:,0]
        dst_part_trans = self.part_means[:,:,1]

        return src_part_trans, src_part_quats, dst_part_trans, dst_part_quats

def cluster_models(base_model_pth:str, transformed_model_dir:str, other_model_dataset:str, n_parts:int = 2):

    writer = SummaryWriter(transformed_model_dir)

    base_splat_manager = load_base_model(base_model_pth)
    dataset = SplatTrainDataset(other_model_dataset)
    dataset_len = dataset.__len__()
    width = dataset.width
    height = dataset.height
    print(f"Loaded dataset with {dataset_len} entries of height {height} and width {width}")
    cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx], [0, dataset.fl_y, dataset.cy], [0, 0, 1]])
    render_data = dataset[0][1]
    cam_pose = render_data["transform_matrix"].unsqueeze(0)

    transformed_model_pth = os.path.join(transformed_model_dir, "splat_manager_transformed.pth")
    tf_splat_manager = load_tf_model(transformed_model_pth)

    base_means = base_splat_manager.object_gaussian_params["means"]
    base_quats = base_splat_manager.object_gaussian_params["quats"]

    tf_means = tf_splat_manager.object_gaussian_params["means"]
    tf_quats = tf_splat_manager.object_gaussian_params["quats"]
    # no grad to avoid auto grad complaining about running through the graph twice
    with torch.no_grad():
        base_tfs = means_quats_to_mat(base_means, base_quats)
        tf_tfs = means_quats_to_mat(tf_means, tf_quats)

    part_learner = PartsSeperator(n_parts, base_means.shape[0])

    n_epochs = 10000

    optimizer = torch.optim.Adam(part_learner.parameters(), lr=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = part_learner.forward_and_loss(base_tfs, tf_tfs)
        writer.add_scalar("clustering loss", loss, epoch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch} loss: {loss}")

    part_assignments = part_learner.get_part_assignments()
    src_part_trans, src_part_quats, dst_part_trans, dst_part_quats = part_learner.get_part_trans_quats()

    # empty the splat part gaussian params before repopulating
    base_splat_manager.num_parts = n_parts
    tf_splat_manager.num_parts = n_parts

    # base_splat_manager.parts_gauss_params = {}
    # tf_splat_manager.parts_gauss_params = {}

    for part_idx in range(n_parts):
        object_gaussian_params = base_splat_manager.object_gaussian_params
        dst_object_gaussian_params = tf_splat_manager.object_gaussian_params

        
        dst_part_gaussian_params = {part_gaussian_params = {
                    "means": object_gaussian_params["means"][part_assignments == part_idx],
                    "scales": object_gaussian_params["scales"][part_assignments == part_idx],
                    "quats": object_gaussian_params["quats"][part_assignments == part_idx],
                    "features_dc": object_gaussian_params["features_dc"][part_assignments == part_idx],
                    "features_rest": object_gaussian_params["features_rest"][part_assignments == part_idx],
                    "opacities": object_gaussian_params["opacities"][part_assignments == part_idx],
                    "features_semantics": object_gaussian_params["features_semantics"][part_assignments == part_idx]\
                        if "features_semantics" in object_gaussian_params.keys()\
                        else None,
                    "tf_quats": src_part_quats[part_idx, :].to(object_gaussian_params["quats"]),
                    "tf_trans": src_part_trans[part_idx, :].to(object_gaussian_params["means"]),
                }
                    "means": dst_object_gaussian_params["means"][part_assignments == part_idx],
                    "scales": dst_object_gaussian_params["scales"][part_assignments == part_idx],
                    "quats": dst_object_gaussian_params["quats"][part_assignments == part_idx],
                    "features_dc": dst_object_gaussian_params["features_dc"][part_assignments == part_idx],
                    "features_rest": dst_object_gaussian_params["features_rest"][part_assignments == part_idx],
                    "opacities": dst_object_gaussian_params["opacities"][part_assignments == part_idx],
                    "features_semantics": dst_object_gaussian_params["features_semantics"][part_assignments == part_idx]\
                        if "features_semantics" in dst_object_gaussian_params.keys()\
                        else None,
                    "tf_quats": dst_part_quats[part_idx, :].to(dst_object_gaussian_params["quats"]),
                    "tf_trans": dst_part_trans[part_idx, :].to(dst_object_gaussian_params["means"]),
                }
        
        base_render_rgb, _, _ = base_splat_manager.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, part_gaussian_params)
        writer.add_images(f"part_{part_idx}", base_render_rgb, epoch, dataformats="NHWC")

        dst_render_rgb, _, _ = tf_splat_manager.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, dst_part_gaussian_params)
        writer.add_images(f"part_{part_idx}_dst", dst_render_rgb, epoch, dataformats="NHWC")

        # add the gauss params to the splat managers part gaussian params
        base_splat_manager.set_part(part_idx, part_gaussian_params)
        tf_splat_manager.set_part(part_idx, dst_part_gaussian_params)


    print(f"saving base splat manager gauss params: {base_splat_manager}")
    # save the resulting splat managers
    manager_path = os.path.join(transformed_model_dir, f"init_src_splat_manager.pth")
    torch.save(base_splat_manager, manager_path)
    print(f"saving tf splat manager gauss params: {tf_splat_manager}")
    manager_path = os.path.join(transformed_model_dir, f"init_dst_splat_manager.pth")
    torch.save(tf_splat_manager, manager_path)

    
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
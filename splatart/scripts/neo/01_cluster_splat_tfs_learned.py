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

def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth)


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

def cluster_models(splat_tf_manager_pth:str, src_model_dataset:str, dst_model_dataset:str, n_parts:int = 2):
    splat_tf_manager = torch.load(splat_tf_manager_pth)

    # get the directory which contains splat_tf_manager_pth
    exp_dir = Path(splat_tf_manager_pth).parent

    writer = SummaryWriter(exp_dir)

    src_scene_dataset = get_dataset(src_model_dataset)
    # tgt_scene_dataset = get_dataset(dst_model_dataset)
    dataset_len = src_scene_dataset.__len__()
    width = src_scene_dataset.width
    height = src_scene_dataset.height
    print(f"Loaded dataset with {dataset_len} entries of height {height} and width {width}")
    cam_intrinsic = torch.Tensor([[src_scene_dataset.fl_x, 0, src_scene_dataset.cx], [0, src_scene_dataset.fl_y, src_scene_dataset.cy], [0, 0, 1]])
    render_data = src_scene_dataset[0][1]
    cam_pose = render_data["transform_matrix"].unsqueeze(0)

    part_learner = PartsSeperator(n_parts, splat_tf_manager.num_splats[0] + splat_tf_manager.num_splats[1])

    # so autograd doesnt complain about running through the graph twice....
    with torch.no_grad():
        src_splat_means = splat_tf_manager.src_gauss_params["means"]
        src_splat_quats = splat_tf_manager.src_gauss_params["quats"]
        src_splat_tf_trans = splat_tf_manager.src_tf_trans
        src_splat_tf_quats = splat_tf_manager.src_tf_quats
        # print(f"means shape: {src_splat_means.shape} quats shape: {src_splat_quats.shape}")
        # print(f"splat tf trans shape: {src_splat_tf_trans.shape}, splat tf quats shape: {src_splat_tf_quats.shape}")
        src_splat_frames = means_quats_to_mat(src_splat_means, src_splat_quats)
        src_splat_tfs = means_quats_to_mat(src_splat_tf_trans, src_splat_tf_quats)

        print(f"src splat frames shape: {src_splat_frames.shape}")
        print(f"src splat tfs shape: {src_splat_tfs.shape}")

        dst_splat_means = splat_tf_manager.dst_gauss_params["means"]
        dst_splat_quats = splat_tf_manager.dst_gauss_params["quats"]
        dst_splat_tf_trans = splat_tf_manager.dst_tf_trans
        dst_splat_tf_quats = splat_tf_manager.dst_tf_quats
        dst_splat_frames = means_quats_to_mat(dst_splat_means, dst_splat_quats)
        dst_splat_tfs = means_quats_to_mat(dst_splat_tf_trans, dst_splat_tf_quats)

        src_splat_tfs_temp = torch.concat((src_splat_frames, torch.linalg.inv(dst_splat_tfs) @ dst_splat_frames))
        dst_splat_tfs = torch.concat((src_splat_tfs @ src_splat_frames, dst_splat_frames))
        src_splat_tfs = src_splat_tfs_temp # lazy and dont want to make the renaming refactor dance

    n_epochs = 10000

    optimizer = torch.optim.Adam(part_learner.parameters(), lr=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = part_learner.forward_and_loss(src_splat_tfs, dst_splat_tfs)
        writer.add_scalar("clustering loss", loss, epoch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch} loss: {loss}")

    part_assignments = part_learner.get_part_assignments()
    src_scene_gauss_params = splat_tf_manager.get_scene_gauss_parameters(scene=0)
    for part_idx in range(n_parts):
        part_gaussian_params = {
                    "means": src_scene_gauss_params["means"][part_assignments == part_idx],
                    "scales": src_scene_gauss_params["scales"][part_assignments == part_idx],
                    "quats": src_scene_gauss_params["quats"][part_assignments == part_idx],
                    "features_dc": src_scene_gauss_params["features_dc"][part_assignments == part_idx],
                    "features_rest": src_scene_gauss_params["features_rest"][part_assignments == part_idx],
                    "opacities": src_scene_gauss_params["opacities"][part_assignments == part_idx]
                }
        render_results = splat_tf_manager.render_gauss_params_at_campose(cam_pose, cam_intrinsic, width, height, part_gaussian_params, scene=0)
        writer.add_images(f"part {part_idx} render", render_results[0][0, ...,:3], epoch, dataformats="HWC")
        
    
if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--splat_tf_manager_pth', 
                        type=str,
                        help='pre-trained splat for source of clusters',
                        default="")
    
    parser.add_argument('--src_model_dataset',
                        type=str,
                        help="directory of the base scene's dataset",
                        default="")
    
    parser.add_argument('--dst_model_dataset',
                        type=str,
                        help="directory of the other scene's dataset",
                        default="")
    
    parser.add_argument('--n_parts',
                        type=int,
                        help="number of parts to separate into",
                        default=2)
    
    args = parser.parse_args()
    cluster_models(args.splat_tf_manager_pth, args.src_model_dataset, args.dst_model_dataset, args.n_parts)
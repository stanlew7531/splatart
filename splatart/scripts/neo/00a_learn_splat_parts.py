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
import tqdm
from datetime import datetime

from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManager, load_managers_nerfstudio
from splatart.datasets.splat_train_dataset import SplatTrainDataset

def load_tf_model(input_model_path:str):
    manager = torch.load(input_model_path)
    return manager


def means_quats_to_tfmat(means, quats):
    to_return = torch.zeros((means.shape[0], 4, 4)).to(means)
    to_return[:, :3, :3] = p3dt.quaternion_to_matrix(quats)
    to_return[:, :3, 3] = means
    to_return[:, 3, 3] = 1
    return to_return

def tfmat_to_means_quats(tfmat):
    return tfmat[:, :3, 3], p3dt.matrix_to_quaternion(tfmat[:, :3, :3])

def quat_to_euler(quat, seq="XYZ"):
    return p3dt.matrix_to_euler_angles(p3dt.quaternion_to_matrix(quat), seq)

def euler_to_quat(euler, seq="XYZ"):
    return p3dt.matrix_to_quaternion(p3dt.euler_angles_to_matrix(euler, seq))

def dist_reg_term(means, neighbor_edges, expected_distances):
    src_idxs = neighbor_edges[:, 0].long()
    dst_idxs = neighbor_edges[:, 1].long()
    src_means = means[src_idxs]
    dst_means = means[dst_idxs]
    dists = torch.norm(src_means - dst_means, dim=-1)
    return torch.nn.functional.mse_loss(dists, expected_distances)

def delta_reg_term(tfs, neighbor_edges, expected_deltas):
    src_tfs = tfs[neighbor_edges[:, 0].long()]
    dst_tfs = tfs[neighbor_edges[:, 1].long()]

    # move the dst_tfs into the src_tfs space
    compare_tfs = torch.linalg.inv(src_tfs) @ dst_tfs
    compare_deltas = compare_tfs[:, :3, 3]

    print(f"compare deltas shape: {compare_deltas.shape}")

    return torch.nn.functional.mse_loss(compare_deltas, expected_deltas)

def finetune_tfs(src_model_path:str, dst_model_path:str, src_model_dataset:str, other_model_dataset:str, output_dir:str):
    os.makedirs(output_dir, exist_ok=True)

    iterations = 100000
    batch_size = 10

    src_splat_manager = load_tf_model(src_model_path)
    dst_splat_manager = load_tf_model(dst_model_path)

    src_part_gauss_params = src_splat_manager.parts_gauss_params
    dst_part_gauss_params = dst_splat_manager.parts_gauss_params

    num_parts = len(src_part_gauss_params.keys())

    assert num_parts == len(dst_part_gauss_params.keys()), "Number of parts in source and destination splats must match"

    # get the dataset for the destination scene
    src_dataset = SplatTrainDataset(src_model_dataset)
    dst_dataset = SplatTrainDataset(other_model_dataset)
    src_dataset_len = src_dataset.__len__()
    dst_dataset_len = dst_dataset.__len__()
    width = src_dataset.width
    height = src_dataset.height
    print(f"Loaded datasets with {src_dataset_len} a,d {dst_dataset_len} entries of height {height} and width {width}")
    cam_intrinsic = torch.Tensor([[src_dataset.fl_x, 0, src_dataset.cx], [0, src_dataset.fl_y, src_dataset.cy], [0, 0, 1]])

    # build up the part params for later rendering
    base_parts_gauss_params = []
    src_parts_tf_quats = []
    src_parts_tf_trans = []
    dst_parts_tf_quats = []
    dst_parts_tf_trans = []
    for i in range(num_parts):
        part_to_add = {}
        part_to_add["means"] = src_splat_manager.parts_gauss_params[i]["means"]
        part_to_add["quats"] = src_splat_manager.parts_gauss_params[i]["quats"]
        part_to_add["features_dc"] = src_splat_manager.parts_gauss_params[i]["features_dc"]
        part_to_add["features_rest"] = src_splat_manager.parts_gauss_params[i]["features_rest"]
        part_to_add["opacities"] = src_splat_manager.parts_gauss_params[i]["opacities"]
        base_parts_gauss_params.append(part_to_add)
        
        src_parts_tf_trans.append(src_splat_manager.parts_gauss_params[i]["tf_trans"])
        src_parts_tf_quats.append(src_splat_manager.parts_gauss_params[i]["tf_quats"])
        dst_parts_tf_trans.append(dst_splat_manager.parts_gauss_params[i]["tf_trans"])
        dst_parts_tf_quats.append(dst_splat_manager.parts_gauss_params[i]["tf_quats"])

    
    
    # setup the optimization loop
    # we want to optimize all of the part's params across both tf states
    optimization_params = [] 
    for i in range(num_parts):
        optimization_params.append(base_parts_gauss_params[i]["means"])
        optimization_params.append(base_parts_gauss_params[i]["quats"])
        optimization_params.append(base_parts_gauss_params[i]["features_dc"])
        optimization_params.append(base_parts_gauss_params[i]["features_rest"])
        optimization_params.append(base_parts_gauss_params[i]["opacities"])

        optimization_params.append(src_parts_tf_quats[i])
        optimization_params.append(src_parts_tf_trans[i])
        optimization_params.append(dst_parts_tf_quats[i])
        optimization_params.append(dst_parts_tf_trans[i])

    optimizer = torch.optim.Adam(optimization_params, lr = 0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    
    train_dataloader_src = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader_dst = DataLoader(dst_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = iterations // src_dataset_len
    print(f"Training for {n_epochs} epochs with {src_dataset_len // batch_size} batches per epoch")

    # tensorboard output and SSIM definition
    writer = SummaryWriter(output_dir)
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)

    # main optimization loop
    for epoch in tqdm.tqdm(range(n_epochs)):
        batches_per_epoch = src_dataset_len // batch_size
        loss_total = 0
        ssim_total = 0
        l1_total = 0
        accumulation_loss_total = 0

        for src_data, dst_data in tqdm.tqdm(zip(train_dataloader_src, train_dataloader_dst), leave=False):
            src_i = src_data[0]
            src_entry = src_data[1]
            dst_i = dst_data[0]
            dst_entry = dst_data[1]

            batched_cam_intrinsic = cam_intrinsic.unsqueeze(0).expand(src_i.shape[0], -1, -1)
            optimizer.zero_grad()
            src_cam_pose = src_entry["transform_matrix"]
            dst_cam_pose = dst_entry["transform_matrix"]
            gt_rgba_src = src_entry["rgb"].cuda().to(torch.float32) / 255.0
            gt_rgba_dst = dst_entry["rgb"].cuda().to(torch.float32) / 255.0

            src_render_parts_gauss_params = []
            for part_idx in range(num_parts):
                to_add = src_part_gauss_params[part_idx]
                to_add["tf_trans"] = src_parts_tf_trans[part_idx]
                to_add["tf_quats"] = src_parts_tf_quats[part_idx]
                src_render_parts_gauss_params.append(to_add)         

            src_render_rgb, src_render_alpha, _ = src_splat_manager.render_given_parts_at_campose(\
                src_cam_pose, batched_cam_intrinsic, width, height, src_render_parts_gauss_params, apply_transforms=True)
            
            dst_render_parts_gauss_params = []
            for part_idx in range(num_parts):
                to_add = src_part_gauss_params[part_idx]
                to_add["tf_trans"] = dst_parts_tf_trans[part_idx]
                to_add["tf_quats"] = dst_parts_tf_quats[part_idx]
                dst_render_parts_gauss_params.append(to_add)
            
            dst_render_rgb, dst_render_alpha, _ = dst_splat_manager.render_given_parts_at_campose(\
                dst_cam_pose, batched_cam_intrinsic, width, height, dst_render_parts_gauss_params, apply_transforms=True)

            # combine the render and alpha to get the final 4 channel image
            src_render_rgba = torch.cat([src_render_rgb, src_render_alpha], dim=-1)
            dst_render_rgba = torch.cat([dst_render_rgb, dst_render_alpha], dim=-1)

            # put the GT and rendered images against the same background
            white_bg = torch.ones_like(gt_rgba_src)

            # composite the rendered rgbas onto the white background
            src_render_whitebg = src_render_rgba * src_render_alpha + white_bg * (1 - src_render_alpha)
            dst_render_whitebg = dst_render_rgba * dst_render_alpha + white_bg * (1 - dst_render_alpha)

            src_gt_alpha = gt_rgba_src[..., 3].unsqueeze(-1)
            dst_gt_alpha = gt_rgba_dst[..., 3].unsqueeze(-1)

            src_gt_whitebg = gt_rgba_src * src_gt_alpha + white_bg * (1 - src_gt_alpha)
            dst_gt_whitebg = gt_rgba_dst * dst_gt_alpha + white_bg * (1 - dst_gt_alpha)

            src_render_whitebg = src_render_whitebg[..., :3]
            dst_render_whitebg = dst_render_whitebg[..., :3]

            src_gt_whitebg = src_gt_whitebg[..., :3]
            dst_gt_whitebg = dst_gt_whitebg[..., :3]

            # compute l1 loss between render and gt_rgb
            # loss_mse = torch.nn.functional.mse_loss(render_whitebg, gt_whitebg)
            loss_l1_src = torch.nn.functional.l1_loss(src_render_whitebg, src_gt_whitebg)
            loss_l1_dst = torch.nn.functional.l1_loss(dst_render_whitebg, dst_gt_whitebg)

            simloss_src = 1 - ssim_fn(src_gt_whitebg.permute(0, 3, 1, 2), src_render_whitebg.permute(0, 3, 1, 2))
            simloss_dst = 1 - ssim_fn(dst_gt_whitebg.permute(0, 3, 1, 2), dst_render_whitebg.permute(0, 3, 1, 2))

            acc_loss_src = torch.abs(src_gt_alpha - src_render_alpha).mean()
            acc_loss_dst = torch.abs(dst_gt_alpha - dst_render_alpha).mean()

            ssim_lambda = 0.2
            accumulation_loss_mult = 1e-7

            src_loss = (1 - ssim_lambda) * loss_l1_src\
                + ssim_lambda * simloss_src\
                + (acc_loss_src * accumulation_loss_mult)
            
            dst_loss = (1 - ssim_lambda) * loss_l1_dst\
                + ssim_lambda * simloss_dst\
                + (acc_loss_dst * accumulation_loss_mult)
            
            loss = src_loss + dst_loss

            # if i contains 0
            if(torch.any(src_i == 0)):
                # get the first gt and render image
                tb_src_gt_img = src_gt_whitebg[0].cpu().detach().numpy()[..., :3]
                tb_src_render_img = src_render_whitebg[0].cpu().detach().numpy()[..., :3]
                tb_dst_gt_img = dst_gt_whitebg[0].cpu().detach().numpy()[..., :3]
                tb_dst_render_img = dst_render_whitebg[0].cpu().detach().numpy()[..., :3]
                # combine the images into a batch
                tb_images = np.stack([tb_src_gt_img, tb_src_render_img, tb_dst_gt_img, tb_dst_render_img], axis=0)
                # write the image to tensorboard
                writer.add_images("gt vs renders fine tune", tb_images, epoch, dataformats="NHWC")

                for part_idx in range(num_parts):
                    # print(src_render_parts_gauss_params[part_idx])
                    viz_render, _, _ = src_splat_manager.render_gauss_params_at_campose(\
                        src_cam_pose, batched_cam_intrinsic, width, height, src_render_parts_gauss_params[part_idx])
                    writer.add_images(f"part_{part_idx}_finetune", viz_render, epoch, dataformats="NHWC")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()

        writer.add_scalar("loss_finetune", loss_total, epoch)
        # writer.add_scalar("ssim", ssim_total, epoch)
        # writer.add_scalar("l1", l1_total, epoch)
        # writer.add_scalar("acc_avg", accumulation_loss_total, epoch)


if __name__ == "__main__":
    print("Learning tfs from base splat to other scene...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--src_model_pth', 
                        type=str,
                        help='pre-trained splat with base scene parts and tfs',
                        default="")
    
    parser.add_argument('--dst_model_pth', 
                        type=str,
                        help='pre-trained splat with dest scene parts and tfs',
                        default="")
    
    parser.add_argument('--src_model_dataset',
                        type=str,
                        help="directory of the base scene's dataset",
                        default="")
    
    parser.add_argument('--other_model_dataset',
                        type=str,
                        help="directory of the other scene's dataset",
                        default="")
    
    parser.add_argument('--exp_dir',
                        type=str,
                        help='directory to save the results',
                        default="./results/neo_splatart/")

    args = parser.parse_args()
    # run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    finetune_tfs(args.src_model_pth, args.dst_model_pth, args.src_model_dataset, args.other_model_dataset, args.exp_dir)
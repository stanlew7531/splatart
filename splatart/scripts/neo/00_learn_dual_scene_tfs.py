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
import tinycudann as tcnn
from datetime import datetime

from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManager, load_managers_nerfstudio
from splatart.managers.SplatTfManager import SplatTFManager
from splatart.datasets.splat_train_dataset import SplatTrainDataset

def load_base_model(input_model_path:str):
    return load_managers_nerfstudio([input_model_path], 0)[0]

def load_tf_model(input_model_path:str):
    manager = torch.load(input_model_path)
    return manager

def create_tf_splat_manager(input_rigid_model_src, input_rigid_model_dst):
    src_rigid_splat = load_base_model(input_rigid_model_src)
    dst_rigid_splat = load_base_model(input_rigid_model_dst)
    return SplatTFManager(src_rigid_splat, dst_rigid_splat)


def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth)

def get_optimization_params(splat_manager:SplatTFManager):
    optimization_params = [] 

    optimization_params.append(splat_manager.src_tf_trans)
    optimization_params.append(splat_manager.src_tf_quats)
    optimization_params.append(splat_manager.dst_tf_trans)
    optimization_params.append(splat_manager.dst_tf_quats)

    return optimization_params

def get_optimizer(optimization_params, lr=0.001):
    return torch.optim.Adam(optimization_params, lr = lr)

def preprocess_images(render_image_results, gt_rgba):
    render_rgb = render_image_results[0]
    render_alpha = render_image_results[1]
    render_rgba = torch.cat([render_rgb, render_alpha], dim=-1)

    gt_rgb = gt_rgba[..., :3]
    gt_alpha = gt_rgba[..., 3].unsqueeze(-1)

    # put the GT and rendered images against the same background
    white_bg = torch.ones_like(gt_rgba)

    # composite rendered onto white background
    render_whitebg = render_rgba * render_alpha + white_bg * (1 - render_alpha)
    gt_whitebg = gt_rgba * gt_alpha + white_bg * (1 - gt_alpha)
    render_whitebg = render_whitebg[..., :3] # only want rgb for the white bg versions
    gt_whitebg = gt_whitebg[..., :3]

    return render_rgb, render_alpha, gt_rgb, gt_alpha, render_whitebg, gt_whitebg

def dist_reg_term(means, neighbor_edges, expected_distances):
    src_idxs = neighbor_edges[:, 0].long()
    dst_idxs = neighbor_edges[:, 1].long()
    src_means = means[src_idxs]
    dst_means = means[dst_idxs]
    dists = torch.norm(src_means - dst_means, dim=-1)
    return torch.nn.functional.mse_loss(dists, expected_distances)

def compute_neighbor_losses(splat_tf_manager):
    src_means, _ = splat_tf_manager.get_transformed_src_gauss_parameters()
    src_neighbor_edges = splat_tf_manager.src_neighbor_edges
    src_neighbor_distances = splat_tf_manager.src_neighbor_distances

    dst_means, _ = splat_tf_manager.get_transformed_dst_gauss_parameters()
    dst_neighbor_edges = splat_tf_manager.dst_neighbor_edges
    dst_neighbor_distances = splat_tf_manager.dst_neighbor_distances

    src_dist_reg = dist_reg_term(src_means, src_neighbor_edges, src_neighbor_distances)
    dst_dist_reg = dist_reg_term(dst_means, dst_neighbor_edges, dst_neighbor_distances)

    losses = {
        "src_dist_reg": src_dist_reg,\
        "dst_dist_reg": dst_dist_reg,\
        }
    
    return losses
    
def compute_image_losses(render_whitebg, gt_whitebg, gt_alpha, render_alpha, ssim_fn):
    
    loss_l1 = torch.nn.functional.l1_loss(render_whitebg, gt_whitebg)
    ssim_loss = 1 - ssim_fn(gt_whitebg.permute(0,3,1,2), render_whitebg.permute(0,3,1,2))
    acc_loss = torch.abs(gt_alpha - render_alpha).mean()

    losses = {
        "l1": loss_l1,\
        "ssim": ssim_loss,\
        "acc": acc_loss,\
    }

    return losses

def combine_losses(batch_losses:dict, loss_lambdas:dict, loss_tots:dict):
    loss_tot = 0

    def add_loss_to_dict(loss_key, loss_val, loss_dict):
        if(not loss_key in loss_dict):
            loss_dict[loss_key] = loss_val
        else:
            loss_dict[loss_key] += loss_val

    for loss_key in batch_losses.keys():
        add_loss_to_dict(loss_key, batch_losses[loss_key], loss_tots)
        loss_tot += loss_lambdas[loss_key] * (batch_losses[loss_key])

    return loss_tot

def learn_splat_tfs(input_model_src:str, input_model_dst:str, src_model_dataset:str, tgt_model_dataset:str, output_dir:str):

    run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(output_dir, run_datetime)
    writer = SummaryWriter(exp_dir)

    splat_tf_manager = create_tf_splat_manager(input_model_src, input_model_dst)
    src_scene_dataset = get_dataset(src_model_dataset)
    tgt_scene_dataset = get_dataset(tgt_model_dataset)

    width = src_scene_dataset.width
    height = tgt_scene_dataset.height

    iterations = 100000
    batch_size = 10

    src_dataset_len = src_scene_dataset.__len__()
    tgt_dataset_len = tgt_scene_dataset.__len__()

    n_epochs = iterations // src_dataset_len

    train_dataloader_src = DataLoader(src_scene_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader_dst = DataLoader(tgt_scene_dataset, batch_size=batch_size, shuffle=False)

    cam_intrinsic = torch.Tensor([[src_scene_dataset.fl_x, 0, src_scene_dataset.cx],\
                                [0, src_scene_dataset.fl_y, src_scene_dataset.cy],\
                                [0, 0, 1]])
    

    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)
    
    lr = 0.0005
    optimization_params = get_optimization_params(splat_tf_manager)
    optimizer = get_optimizer(optimization_params, lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    loss_lambdas = {
        "l1": 0.8,
        "ssim": 0.2,
        "acc": 1e-7,
        "src_dist_reg": 100,
        "dst_dist_reg": 100,
    }

    for epoch in tqdm.tqdm(range(n_epochs)):
        batches_per_epoch = src_dataset_len // batch_size
        loss_total = 0
        src_loss_totals = {}
        src_only_loss_totals = {}
        dst_loss_totals = {}
        dst_only_loss_totals = {}
        neighbor_loss_totals = {}

        for src_data, dst_data in tqdm.tqdm(zip(train_dataloader_src, train_dataloader_dst), leave=False):
            # inidices and data of the source and dest dataloaders
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

            src_render_result = splat_tf_manager.render_at_campose(src_cam_pose, batched_cam_intrinsic, width, height, scene=0, source = 2)
            dst_render_result = splat_tf_manager.render_at_campose(dst_cam_pose, batched_cam_intrinsic, width, height, scene=1, source = 2)

            src_only_render_result = splat_tf_manager.render_at_campose(dst_cam_pose, batched_cam_intrinsic, width, height, scene=1, source = 0)
            dst_only_render_result = splat_tf_manager.render_at_campose(src_cam_pose, batched_cam_intrinsic, width, height, scene=0, source = 1)

            # images of both source and target splats combined
            src_render_rgb, src_render_alpha, src_gt_rgb, src_gt_alpha, src_render_whitebg, src_gt_whitebg = \
                preprocess_images(src_render_result, gt_rgba_src)
            dst_render_rgb, dst_render_alpha, dst_gt_rgb, dst_gt_alpha, dst_render_whitebg, dst_gt_whitebg = \
                preprocess_images(dst_render_result, gt_rgba_dst)
            
            # images with only the transformed source and transformed target splats
            src_only_render_rgb, src_only_render_alpha, src_only_gt_rgb, src_only_gt_alpha, src_only_render_whitebg, src_only_gt_whitebg = \
                preprocess_images(src_only_render_result, gt_rgba_dst)
            dst_only_render_rgb, dst_only_render_alpha, dst_only_gt_rgb, dst_only_gt_alpha, dst_only_render_whitebg, dst_only_gt_whitebg = \
                preprocess_images(dst_only_render_result, gt_rgba_src)
            
            if(torch.any(src_i == 0) and (epoch % 10 ==  0)):
                writer.add_images(f"src render", src_render_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"dst render", dst_render_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"src GT", src_gt_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"dst GT", dst_gt_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"src only render", src_only_render_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"dst only render", dst_only_render_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"src only GT", src_only_gt_whitebg[...,:3], epoch, dataformats="NHWC")
                writer.add_images(f"dst only GT", dst_only_gt_whitebg[...,:3], epoch, dataformats="NHWC")

            src_losses = compute_image_losses(src_render_whitebg, src_gt_whitebg, src_gt_alpha, src_render_alpha, ssim_fn)
            dst_losses = compute_image_losses(dst_render_whitebg, dst_gt_whitebg, dst_gt_alpha, dst_render_alpha, ssim_fn)
            src_only_losses = compute_image_losses(src_only_render_whitebg, dst_gt_whitebg, dst_gt_alpha, src_only_render_alpha, ssim_fn)
            dst_only_losses = compute_image_losses(dst_only_render_whitebg, src_gt_whitebg, src_gt_alpha, dst_only_render_alpha, ssim_fn)

            neighbor_losses = combine_losses(compute_neighbor_losses(splat_tf_manager), loss_lambdas, neighbor_loss_totals)
            src_image_loss = combine_losses(src_losses, loss_lambdas, src_loss_totals)
            dst_image_loss = combine_losses(dst_losses, loss_lambdas, dst_loss_totals)
            src_only_image_loss = 0.1 * combine_losses(src_only_losses, loss_lambdas, src_only_loss_totals)
            dst_only_image_loss = 0.1 * combine_losses(dst_only_losses, loss_lambdas, dst_only_loss_totals)

            loss = src_image_loss + dst_image_loss + neighbor_losses + src_only_image_loss + dst_only_image_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()

        writer.add_scalar("00_loss total", loss_total, epoch)
        writer.add_scalar("01_src ssim", src_loss_totals["ssim"], epoch)
        writer.add_scalar("01_dst ssim", dst_loss_totals["ssim"], epoch)
        writer.add_scalar("02_src l1", src_loss_totals["l1"], epoch)
        writer.add_scalar("02_dst l1", dst_loss_totals["l1"], epoch)
        writer.add_scalar("03_src acc", src_loss_totals["acc"], epoch)
        writer.add_scalar("03_dst acc", dst_loss_totals["acc"], epoch)
        writer.add_scalar("04_src dist reg", neighbor_loss_totals["src_dist_reg"], epoch)
        writer.add_scalar("04_dst dist reg", neighbor_loss_totals["dst_dist_reg"], epoch)
        writer.add_scalar("05_src_only ssim", src_only_loss_totals["ssim"], epoch)
        writer.add_scalar("05_dst_only ssim", dst_only_loss_totals["ssim"], epoch)

        if(epoch % 50 == 0):
            manager_path = os.path.join(exp_dir, f"dual_splat_manager_tfs_e{epoch}.pth")
            torch.save(splat_tf_manager, manager_path)

    manager_path = os.path.join(exp_dir, f"dual_splat_manager_tfs.pth")
    torch.save(splat_tf_manager, manager_path)



if __name__ == "__main__":
    print("Learning tfs from base splat to other scene...")

    parser = argparse.ArgumentParser(description="Given a pre-trained non-articulated initial splat, learn to render the target scene and the part seperation")

    parser.add_argument('--src_model_pth', 
                        type=str,
                        help='pre-trained splat with base scene parts and tfs',
                        default="")
    
    parser.add_argument('--dst_model_pth', 
                        type=str,
                        help='pre-trained splat with other scene parts and tfs',
                        default="")
    
    parser.add_argument('--src_model_dataset',
                        type=str,
                        help="directory of the base scene's dataset",
                        default="")
    
    parser.add_argument('--dst_model_dataset',
                        type=str,
                        help="directory of the other scene's dataset",
                        default="")
    
    parser.add_argument('--exp_dir',
                        type=str,
                        help='directory to save the results',
                        default="./results/neo_splatart/")

    args = parser.parse_args()

    learn_splat_tfs(args.src_model_pth, args.dst_model_pth, args.src_model_dataset, args.dst_model_dataset, args.exp_dir)
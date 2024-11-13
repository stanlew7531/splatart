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

def load_base_model(input_model_path:str):
    return load_managers_nerfstudio([input_model_path], 0)[0]

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

def learn_tfs(input_model:str, other_model_dataset:str, output_dir:str):
    os.makedirs(output_dir, exist_ok=True)

    iterations = 100000
    batch_size = 10

    splat_manager = load_base_model(input_model)

    # get the neighbor edges and distances
    neighbor_edges, neighbor_distances, neighbor_deltas = [], [], []

    with torch.no_grad():
        for edge, distance, delta in splat_manager.get_neighbor_edges_distances_deltas(radius=0.06):
            neighbor_edges.append(edge)
            neighbor_distances.append(distance)
            neighbor_deltas.append(delta)
        neighbor_edges = torch.Tensor(neighbor_edges).to(splat_manager.object_gaussian_params["means"].device)
        neighbor_distances = torch.Tensor(neighbor_distances).to(splat_manager.object_gaussian_params["means"].device)
        neighbor_deltas = torch.stack(neighbor_deltas, dim=0).to(splat_manager.object_gaussian_params["means"].device)
        print(f"edges shape: {neighbor_edges.shape}")
        print(f"distances shape: {neighbor_distances.shape}")
        print(f"deltas shape: {neighbor_deltas.shape}")

    # get the dataset for the destination scene
    dataset = SplatTrainDataset(other_model_dataset)
    dataset_len = dataset.__len__()
    width = dataset.width
    height = dataset.height
    print(f"Loaded dataset with {dataset_len} entries of height {height} and width {width}")
    cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx], [0, dataset.fl_y, dataset.cy], [0, 0, 1]])
    
    # setup the optimization loop
    optimization_params = [splat_manager.object_gaussian_params["means"], splat_manager.object_gaussian_params["quats"]]
    optimizer = torch.optim.Adam(optimization_params, lr = 0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_epochs = iterations // dataset_len
    print(f"Training for {n_epochs} epochs with {dataset_len // batch_size} batches per epoch")

    # tensorboard output and SSIM definition
    writer = SummaryWriter(output_dir)
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)

    # main optimization loop
    for epoch in tqdm.tqdm(range(n_epochs)):
        batches_per_epoch = dataset_len // batch_size
        loss_total = 0
        ssim_total = 0
        l1_total = 0
        accumulation_loss_total = 0

        for i, entry in tqdm.tqdm(train_dataloader, leave=False):
            batched_cam_intrinsic = cam_intrinsic.unsqueeze(0).expand(i.shape[0], -1, -1)
            optimizer.zero_grad()
            gt_cam_pose = entry["transform_matrix"]
            gt_rgba = entry["rgb"].cuda().to(torch.float32) / 255.0

            render_rgb, render_alpha, info = splat_manager.render_object_at_campose(gt_cam_pose, batched_cam_intrinsic, width, height, apply_transforms=False)

            # combine the render and alpha to get the final 4 channel image
            render_rgba = torch.cat([render_rgb, render_alpha], dim=-1)

            # put the GT and rendered images against the same background
            white_bg = torch.ones_like(gt_rgba)

            # composite the render_rgba onto the white background
            render_whitebg = render_rgba * render_alpha + white_bg * (1 - render_alpha)
            gt_alpha = gt_rgba[..., 3].unsqueeze(-1)
            gt_whitebg = gt_rgba * gt_alpha + white_bg * (1 - gt_alpha)
            render_whitebg = render_whitebg[..., :3]
            gt_whitebg = gt_whitebg[..., :3]

            # compute l1 loss between render and gt_rgb
            # loss_mse = torch.nn.functional.mse_loss(render_whitebg, gt_whitebg)
            loss_l1 = torch.nn.functional.l1_loss(render_whitebg, gt_whitebg)
            simloss = 1 - ssim_fn(gt_whitebg.permute(0, 3, 1, 2), render_whitebg.permute(0, 3, 1, 2))

            acc_loss = torch.abs(gt_alpha - render_alpha).mean()

            ssim_lambda = 0.2
            accumulation_loss_mult = 1e-7
            dist_reg_mult = 100


            reg_distances_loss = dist_reg_term(splat_manager.object_gaussian_params["means"], neighbor_edges, neighbor_distances)
            # reg_deltas_loss = delta_reg_term(\
            #     means_quats_to_tfmat(splat_manager.object_gaussian_params["means"], splat_manager.object_gaussian_params["quats"]),\
            #     neighbor_edges,\
            #     neighbor_deltas)

            # print(f"distances delta: {distances_delta}")

            loss = (1 - ssim_lambda) * loss_l1\
                + ssim_lambda * simloss\
                + (acc_loss * accumulation_loss_mult)\
                + (reg_distances_loss * dist_reg_mult)

            # if i contains 0
            if(torch.any(i == 0)):
                # get the first gt and render image
                tb_gt_img = gt_whitebg[0].cpu().detach().numpy()[..., :3]
                tb_render_img = render_whitebg[0].cpu().detach().numpy()[..., :3]
                # combine the images into a batch
                tb_images = np.stack([tb_gt_img, tb_render_img], axis=0)
                # write the image to tensorboard
                writer.add_images("gt vs render", tb_images, epoch, dataformats="NHWC")

                # only need to do this once per epoch
                writer.add_scalar("dist_reg", reg_distances_loss, epoch)
                # writer.add_scalar("delta_reg", reg_deltas_loss, epoch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
            ssim_total += simloss.item()
            l1_total += loss_l1.item()
            accumulation_loss_total += acc_loss.item()

        writer.add_scalar("loss", loss_total, epoch)
        writer.add_scalar("ssim", ssim_total, epoch)
        writer.add_scalar("l1", l1_total, epoch)
        writer.add_scalar("acc_avg", accumulation_loss_total, epoch)


    # save the learned manager back to disk
    manager_path = os.path.join(output_dir, f"splat_manager_transformed.pth")
    neighbors_path = os.path.join(output_dir, f"neighbor_edges.pth")
    distances_path = os.path.join(output_dir, f"neighbor_distances.pth")
    torch.save(splat_manager, manager_path)
    torch.save(neighbor_edges, neighbors_path)
    torch.save(neighbor_distances, distances_path)
    print(f"Saved splat manager to {manager_path}")
    print(f"Saved neighbor edges to {neighbors_path}")
    print(f"Saved neighbor distances to {distances_path}")
            

if __name__ == "__main__":
    print("Learning tfs from base splat to other scene...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--input_model_pth', 
                        type=str,
                        help='pre-trained splat to learn TFs for',
                        default="")
    
    parser.add_argument('--other_model_dataset',
                        type=str,
                        help="directory of the other scene's dataset",
                        default="")
    
    parser.add_argument('--output_dir',
                        type=str,
                        help='output directory to save the results',
                        default="./results/neo_splatart/")

    args = parser.parse_args()
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    learn_tfs(args.input_model_pth, args.other_model_dataset, os.path.join(args.output_dir, run_datetime))
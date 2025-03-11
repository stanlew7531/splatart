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

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import HDBSCAN
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats, apply_mat_tf_to_gauss_params
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.networks.PoseEstimator import PoseEstimator


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

def get_gaussian_params_centroids(object_parts_gauss_params):
    to_return = []
    for scene_i in range(len(object_parts_gauss_params)):
        part_means = []
        for part_i in range(len(object_parts_gauss_params[scene_i])):
            part_means.append(object_parts_gauss_params[scene_i][part_i]["means"].mean(dim=0))
        to_return.append(torch.stack(part_means, dim=0))
    return to_return

def get_opt_params_from_obj_part_gauss_params(object_parts_gauss_params):
    to_return = []
    for scene_i in range(len(object_parts_gauss_params)):
        for part_i in range(len(object_parts_gauss_params[scene_i])):
            object_parts_gauss_params[scene_i][part_i]["means"].requires_grad = True
            object_parts_gauss_params[scene_i][part_i]["quats"].requires_grad = True
            object_parts_gauss_params[scene_i][part_i]["scales"].requires_grad = True
            object_parts_gauss_params[scene_i][part_i]["features_dc"].requires_grad = True
            object_parts_gauss_params[scene_i][part_i]["features_rest"].requires_grad = True
            object_parts_gauss_params[scene_i][part_i]["opacities"].requires_grad = True
            to_return.append(object_parts_gauss_params[scene_i][part_i]["means"])
            to_return.append(object_parts_gauss_params[scene_i][part_i]["quats"])
            to_return.append(object_parts_gauss_params[scene_i][part_i]["scales"])
            to_return.append(object_parts_gauss_params[scene_i][part_i]["features_dc"])
            to_return.append(object_parts_gauss_params[scene_i][part_i]["features_rest"])
            to_return.append(object_parts_gauss_params[scene_i][part_i]["opacities"])
    return to_return

def learn_poses(manager_paths: list[str], dataset_paths: list[str], icp_json=None):
    with torch.no_grad():
        batch_size = 5
        splat_managers = [load_base_model(path) for path in manager_paths]
        splat_datasets = [get_dataset(path) for path in dataset_paths]
        splat_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in splat_datasets]
        object_parts_gauss_params = [unpack_gaussian_params(manager.object_gaussian_params, manager.num_parts) for manager in splat_managers]
        part_centroids = get_gaussian_params_centroids(object_parts_gauss_params)
        
        # set the output dir as the directory of the first manager path
        output_dir = os.path.dirname(manager_paths[0])

        n_managers = len(splat_managers)
        num_parts = splat_managers[0].num_parts
        # make sure all the parts for all managers match
        for i in range(1, n_managers):
            other_manager_parts_num =  splat_managers[i].num_parts
            assert num_parts == other_manager_parts_num,\
                f"Manager {i} has {other_manager_parts_num} parts, but manager 0 has {num_parts} parts"


        pose_estimator = PoseEstimator(num_parts, n_managers)
        if icp_json is not None:
            pose_estimator.preset_tfs_from_icp(icp_json)
        else:
            pose_estimator.preset_centroids(part_centroids)
        

    # create the writer
    exp_dir = output_dir
    writer = SummaryWriter(exp_dir)


    # create optimization params
    splat_opt_params = get_opt_params_from_obj_part_gauss_params(object_parts_gauss_params)
    optimization_params_trans = [pose_estimator.part_means]
    optimization_params_rot = [pose_estimator.part_eulers]

    # splat_opt_params[6].register_hook(lambda grad: print(f"got to splat grad {grad}!"))

    # create the optimizer
    # optimizer_joint = torch.optim.Adam(
    #     [
    #         {"params": splat_opt_params, "lr": 1e-3},
    #         {"params": optimization_params_trans, "lr": 1e-2},
    #         {"params": optimization_params_rot, "lr": 1e-2}
    #     ])
    
    # optimizer_joint = torch.optim.Adam(
    #     [
    #         {"params": splat_opt_params, "lr": 1e-4},
    #         {"params": optimization_params_trans, "lr": 1e-3},
    #         {"params": optimization_params_rot, "lr": 1e-3}
        # ])
    optimizer_splats = torch.optim.Adam(splat_opt_params, lr=1e-4)
    optimizer_trans = torch.optim.Adam(optimization_params_trans, lr=1e-3)
    optimizer_rot = torch.optim.Adam(optimization_params_rot, lr=1e-3)

    # create the scheduler
    # scheduler = lr_scheduler.StepLR(optimizer_joint, step_size=100, gamma=0.5)


    n_epochs = 150

    for epoch in tqdm(range(n_epochs)):
        print(f"Epoch {epoch}")
        total_loss = 0
        for i in range(n_managers): # train for every scene
            print(f"Learning poses for manager {i}...")
            total_scene_loss = 0
            cur_manager = splat_managers[i]
            cur_dataset = splat_datasets[i]
            cur_dataloader = splat_dataloaders[i]

            cam_intrinsic = torch.Tensor([[cur_dataset.fl_x, 0, cur_dataset.cx],\
                                [0, cur_dataset.fl_y, cur_dataset.cy],\
                                [0, 0, 1]])
            width = cur_dataset.width
            height = cur_dataset.height
            iteration = 0
            for batch_idx, batch_data in tqdm(cur_dataloader, leave=False):
                # get the data
                gt_rgb = batch_data["rgb"].to(cur_manager.device).to(torch.float32) / 255.0
                semantics_gt = batch_data["semantics"].to(torch.int64).to(cur_manager.device)
                render_poses = batch_data["transform_matrix"].to(cur_manager.device)

                loss_batch, losses_scenes, render_rgb_scenes, _ = pose_estimator.forward_and_loss(splat_managers, object_parts_gauss_params, render_poses, gt_rgb, semantics_gt, cam_intrinsic, width, height, render_scene=i)

                # accumulate the loss
                total_scene_loss += loss_batch.item()
                total_loss += loss_batch.item()

                # backward pass
                loss_batch.backward()
                
                # update/step
                # optimizer_joint.step()
                optimizer_trans.step()
                optimizer_rot.step()
                optimizer_splats.step()
                # scheduler.step()

                # zero out for next pass
                # optimizer_joint.zero_grad()
                optimizer_trans.zero_grad()
                optimizer_rot.zero_grad()
                optimizer_splats.zero_grad()

                writer.add_scalar(f"batch_loss_{i}", loss_batch.item(), iteration)
                writer.add_image
                iteration += batch_size

                if(0 in batch_idx):
                    # turn render and gt images into the format tensorboard wants
                    # need bchw not bhwc
                    viz_render_rgb = [render_rgb.permute(0, 3, 1, 2) for render_rgb in render_rgb_scenes]
                    viz_gt_rgb = gt_rgb[...,:3].permute(0, 3, 1, 2)
                    viz_render_rgb.append(viz_gt_rgb)
                    combined_rgb = torchvision.utils.make_grid(torch.cat(viz_render_rgb, dim=0), nrow=batch_size)
                    writer.add_image(f"rgb_{i}_scene", combined_rgb, epoch)
            
            if(epoch % 25 == 0):
                # save the current state of the pose estimator
                save_fname = os.path.join(exp_dir, f"pose_estimator_{epoch}.pth")
                print(f"Saving learned pose estimator to {save_fname}...")
                torch.save(pose_estimator, save_fname)
                # save the current state of the splat manager
                save_fname = os.path.join(exp_dir, f"part_gauss_params_{epoch}.pth")
                print(f"Saving learned part gauss params to {save_fname}...")
                torch.save(object_parts_gauss_params, save_fname)
                
            writer.add_scalar(f"scene_{i} loss", total_scene_loss, epoch)
        writer.add_scalar("total loss", total_loss, epoch)

    # save the pose estimator to disk
    save_fname = os.path.join(exp_dir, f"pose_estimator.pth")
    print(f"Saving learned pose estimator to {save_fname}...")
    torch.save(pose_estimator, save_fname)

    # save the resulting part gauss params to disk
    save_fname = os.path.join(exp_dir, f"part_gauss_params.pth")
    print(f"Saving learned part gauss params to {save_fname}...")
    torch.save(object_parts_gauss_params, save_fname)


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
    
    parser.add_argument('--icp_seed',
                        type=str,
                        help="json containing the icp results to initialize the optimization",
                        default="")
    
    args = parser.parse_args()
    manager_paths = args.splat_tf_manager_pths.split(",")
    dataset_paths = args.splat_model_datasets.split(",")

    if(args.icp_seed != ""):
        print(f"using icp seed: {args.icp_seed}")
        with open(args.icp_seed, "r") as f:
            icp_data = json.load(f)
    else:
        icp_data = None

    learn_poses(manager_paths, dataset_paths, icp_json=icp_data)
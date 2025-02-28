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

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats
from splatart.datasets.splat_train_dataset import SplatTrainDataset

class PoseEstimator():
    def __init__(self, num_parts, num_managers):
        self.num_parts = num_parts
        self.num_scenes = num_managers
        self.part_means = torch.nn.Parameter(torch.zeros((num_managers, num_parts, 3)))
        self.part_eulers = torch.nn.Parameter(torch.zeros((num_managers, num_parts, 3)))
        self.SSIM = SSIM(data_range=1.0, size_average=True, channel=3)

    def preset_centroids(self, centroids):
        for idx in range(self.num_scenes):
            self.part_means[idx] = centroids[idx]

    def get_part_splat_frames(self, object_parts_gauss_params:list):
        # get the current estimate of each parts frame
        part_frames = torch.zeros((self.num_scenes, self.num_parts, 4, 4)).to(object_parts_gauss_params[0][0]["means"])
        part_frames[..., :3, :3] = p3dt.euler_angles_to_matrix(self.part_eulers[...], "XYZ")
        part_frames[..., :3, 3] = self.part_means[...]
        part_frames[..., 3, 3] = 1

        # take each scene/part gauss param input and put all splats into the predicted part frame
        # (num_managers, (num_parts, gauss_params dict))
        scene_part_frame_gauss_params = []
        for scene_i in range(self.num_scenes):
            part_frame_gauss_params = []
            for part_i in range(self.num_parts):
                means = object_parts_gauss_params[scene_i][part_i]["means"]
                # means.register_hook(lambda grad: print("got to means grad: {grad}"))

                quats = object_parts_gauss_params[scene_i][part_i]["quats"]
                # quats.register_hook(lambda grad: print("got to quats grad: {grad}"))
                mat = means_quats_to_mat(means, quats)
                
                mat = torch.linalg.inv(part_frames[scene_i][part_i]) @ mat
                gauss_params = {
                    "features_semantics": object_parts_gauss_params[scene_i][part_i]["features_semantics"],
                    "features_dc": object_parts_gauss_params[scene_i][part_i]["features_dc"],
                    "features_rest": object_parts_gauss_params[scene_i][part_i]["features_rest"],
                    "opacities": object_parts_gauss_params[scene_i][part_i]["opacities"],
                    "scales": object_parts_gauss_params[scene_i][part_i]["scales"],
                    "mat": mat
                }
                part_frame_gauss_params.append(gauss_params)
            scene_part_frame_gauss_params.append(part_frame_gauss_params)

        return part_frames, scene_part_frame_gauss_params
    
    def generate_scene_gauss_params(self, part_frames, object_parts_gauss_params, render_scene):
        # convert each part frame into the frame of the render_scene
        # and assign it back to the object gauss params as needed
        dst_part_frames = part_frames[render_scene] # of shape (num_parts, 4, 4)

        scene_part_frame_gauss_params = []
        for scene_i in range(self.num_scenes):
            part_frame_gauss_params = []
            for part_i in range(self.num_parts):
                mat = object_parts_gauss_params[scene_i][part_i]["mat"]
                mat = dst_part_frames[part_i] @ mat
                means, quats = mat_to_means_quats(mat)
                gauss_params = {
                    "means": means,
                    "quats": quats,
                    "features_semantics": object_parts_gauss_params[scene_i][part_i]["features_semantics"],
                    "features_dc": object_parts_gauss_params[scene_i][part_i]["features_dc"],
                    "features_rest": object_parts_gauss_params[scene_i][part_i]["features_rest"],
                    "opacities": object_parts_gauss_params[scene_i][part_i]["opacities"],
                    "scales": object_parts_gauss_params[scene_i][part_i]["scales"],
                }
                part_frame_gauss_params.append(gauss_params)
            scene_part_frame_gauss_params.append(part_frame_gauss_params)

        return scene_part_frame_gauss_params
    
    # given a list of gaussian parameters, combine them for later rendering
    def combine_guass_params(self, scene_part_frame_gauss_params):
        per_scene_gauss_params = []
        for scene_i in range(0, self.num_scenes):
            object_gauss_params = {
                "means": scene_part_frame_gauss_params[scene_i][0]["means"],
                "quats": scene_part_frame_gauss_params[scene_i][0]["quats"],
                "features_semantics": scene_part_frame_gauss_params[scene_i][0]["features_semantics"],
                "features_dc": scene_part_frame_gauss_params[scene_i][0]["features_dc"],
                "features_rest": scene_part_frame_gauss_params[scene_i][0]["features_rest"],
                "opacities": scene_part_frame_gauss_params[scene_i][0]["opacities"],
                "scales": scene_part_frame_gauss_params[scene_i][0]["scales"],
            }

            for part_i in range(1, self.num_parts):
                if scene_i == 0 and part_i == 0:
                    continue
                object_gauss_params["means"] = torch.cat((object_gauss_params["means"], scene_part_frame_gauss_params[scene_i][part_i]["means"]), dim=0)
                object_gauss_params["quats"] = torch.cat((object_gauss_params["quats"], scene_part_frame_gauss_params[scene_i][part_i]["quats"]), dim=0)
                object_gauss_params["features_semantics"] = torch.cat((object_gauss_params["features_semantics"], scene_part_frame_gauss_params[scene_i][part_i]["features_semantics"]), dim=0)
                object_gauss_params["features_dc"] = torch.cat((object_gauss_params["features_dc"], scene_part_frame_gauss_params[scene_i][part_i]["features_dc"]), dim=0)
                object_gauss_params["features_rest"] = torch.cat((object_gauss_params["features_rest"], scene_part_frame_gauss_params[scene_i][part_i]["features_rest"]), dim=0)
                object_gauss_params["opacities"] = torch.cat((object_gauss_params["opacities"], scene_part_frame_gauss_params[scene_i][part_i]["opacities"]), dim=0)
                object_gauss_params["scales"] = torch.cat((object_gauss_params["scales"], scene_part_frame_gauss_params[scene_i][part_i]["scales"]), dim=0)
            
            per_scene_gauss_params.append(object_gauss_params)

        return per_scene_gauss_params
    
    def preprocess_images(self, render_image_results, gt_rgba):
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
    
    def compute_image_losses(self, render_whitebg, gt_whitebg, gt_alpha, render_alpha, ssim_fn):
        loss_l1 = torch.nn.functional.l1_loss(render_whitebg, gt_whitebg)

        ssim_loss = 1 - ssim_fn(gt_whitebg.permute(0,3,1,2), render_whitebg.permute(0,3,1,2))
        acc_loss = torch.abs(gt_alpha - render_alpha).mean()

        losses = {
            "l1": loss_l1,\
            "ssim": ssim_loss,\
            "acc": acc_loss,\
        }

        return losses
    
    def combine_losses(self, batch_losses:dict, loss_lambdas:dict):
        loss_tot = 0

        for loss_key in batch_losses.keys():
            loss_tot += loss_lambdas[loss_key] * (batch_losses[loss_key])

        return loss_tot

    def forward_and_loss(self,\
                            managers:list[SplatManagerSingle],\
                            object_parts_gauss_params,\
                            manager_tfs,\
                            rgbs_gt,\
                            segmentations_gt,\
                            cam_intrinsic,\
                            width,\
                            height):
        # managers are the SplatManagerSingle objects
        # manager tfs is of shape (num_managers, batch_size, 4, 4) and are the poses to render each scene at
        # rgbs_gt is of shape (num_managers, batch_size, height, width, 4) and are the expexted gt image to render
        # segmentations_gt is of shape (num_managers, batch_size, height, width) and are the expected segmentation to render

        # first, transform every splat to its part's frame
        part_frames, scene_part_frame_gauss_params = self.get_part_splat_frames(object_parts_gauss_params)

        scene_0_part_gauss_params = self.generate_scene_gauss_params(part_frames, scene_part_frame_gauss_params, render_scene=0)
        scene_0_gauss_params = self.combine_guass_params(scene_0_part_gauss_params) # represents each scenes gaussian splats transformed to hopefully match scense 0's geometry

        loss = 0.0
        scene_losses = []
        scene_render_rgbs = []

        loss_lambdas = {
                "l1": 0.8,
                "ssim": 0.2,
                "acc": 1e-7,
                "segmentation": 1e-1,
            }
        
        batch_size = rgbs_gt.shape[0]

        for render_scene_idx in range(0, self.num_scenes):
            render_image_results = managers[0].render_gauss_params_at_campose(\
                manager_tfs, cam_intrinsic.expand(batch_size, -1, -1), width, height, scene_0_gauss_params[render_scene_idx], is_semantics=False)
            
            semantics_pred, _, _ = managers[0].render_gauss_params_at_campose(
                manager_tfs, cam_intrinsic.expand(batch_size, -1, -1), width, height, scene_0_gauss_params[render_scene_idx], is_semantics=True)

            # compute the loss
            render_rgb, render_alpha, gt_rgb, gt_alpha, render_whitebg, gt_whitebg = \
                    self.preprocess_images(render_image_results, rgbs_gt)
            
            image_losses = self.compute_image_losses(render_whitebg, gt_whitebg, gt_alpha, render_alpha, self.SSIM)
            image_losses["segmentation"] = torch.nn.CrossEntropyLoss()(semantics_pred.reshape(-1, self.num_parts), segmentations_gt.flatten())
            scene_loss = self.combine_losses(image_losses, loss_lambdas)
            
            loss += scene_loss
            scene_render_rgbs.append(render_rgb)
            scene_losses.append(image_losses)

        return loss, scene_losses, scene_render_rgbs, gt_rgb


def load_base_model(input_model_path:str):
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

def get_opt_params_from_splats(splat_managers):
    to_return = []
    for splat_manager in splat_managers:
        splat_manager.object_gaussian_params["means"].requires_grad = True
        splat_manager.object_gaussian_params["quats"].requires_grad = True
        to_return.append(splat_manager.object_gaussian_params["means"])
        to_return.append(splat_manager.object_gaussian_params["quats"])
    return to_return

def get_opt_params_from_obj_part_gauss_params(object_parts_gauss_params):
    to_return = []
    for scene_i in range(len(object_parts_gauss_params)):
        for part_i in range(len(object_parts_gauss_params[scene_i])):
            object_parts_gauss_params[scene_i][part_i]["means"].requires_grad = True
            object_parts_gauss_params[scene_i][part_i]["quats"].requires_grad = True
            to_return.append(object_parts_gauss_params[scene_i][part_i]["means"])
            to_return.append(object_parts_gauss_params[scene_i][part_i]["quats"])
    return to_return

def learn_poses(manager_paths: list[str], dataset_paths: list[str]):
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
        print(f"Got num parts: {num_parts}")
        # make sure all the parts for all managers match
        for i in range(1, n_managers):
            other_manager_parts_num =  splat_managers[i].num_parts
            assert num_parts == other_manager_parts_num,\
                f"Manager {i} has {other_manager_parts_num} parts, but manager 0 has {num_parts} parts"


        pose_estimator = PoseEstimator(num_parts, n_managers)
        pose_estimator.preset_centroids(part_centroids)
        

    # create the writer
    exp_dir = output_dir
    writer = SummaryWriter(exp_dir)


    # create optimization params
    # splat_opt_params = get_opt_params_from_splats(splat_managers)
    splat_opt_params = get_opt_params_from_obj_part_gauss_params(object_parts_gauss_params)
    optimization_params_trans = [pose_estimator.part_means]
    optimization_params_rot = [pose_estimator.part_eulers]

    # create the optimizer
    optimizer_splats = torch.optim.Adam(splat_opt_params, lr=0.01)
    optimizer_trans = torch.optim.Adam(optimization_params_trans, lr=0.001)
    optimizer_rot = torch.optim.Adam(optimization_params_rot, lr=0.001)

    # create the scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


    n_epochs = 150

    for epoch in tqdm(range(n_epochs)):
        print(f"Epoch {epoch}")
        total_loss = 0
        for i in range(1):
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

                loss_batch, losses_scenes, render_rgb_scenes, _ = pose_estimator.forward_and_loss(splat_managers, object_parts_gauss_params, render_poses, gt_rgb, semantics_gt, cam_intrinsic, width, height)
                # accumulate the loss
                total_scene_loss += loss_batch.item()
                total_loss += loss_batch.item()

                # backward pass
                loss_batch.backward()
                # print(splat_opt_params[2].grad)
                # update/step
                # optimizer.step()
                optimizer_trans.step()
                optimizer_rot.step()
                optimizer_splats.step()
                # scheduler.step()
                # optimizer.zero_grad()
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
                    writer.add_image(f"rgb_{i}", combined_rgb, epoch)
                
            writer.add_scalar(f"scene_{i} loss", total_scene_loss, epoch)
        writer.add_scalar("total loss", total_loss, epoch)

    # save the pose estimator to disk
    save_fname = os.path.join(exp_dir, f"pose_estimator.pth")
    print(f"Saving learned pose estimator to {save_fname}...")
    torch.save(pose_estimator, save_fname)


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
    
    args = parser.parse_args()
    manager_paths = args.splat_tf_manager_pths.split(",")
    dataset_paths = args.splat_model_datasets.split(",")
    learn_poses(manager_paths, dataset_paths)
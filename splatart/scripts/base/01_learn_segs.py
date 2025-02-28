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

from splatart.managers.SplatManager import SplatManagerSingle, load_managers_nerfstudio_single
from splatart.datasets.splat_train_dataset import SplatTrainDataset

def load_base_model(input_model_path:str, num_parts:int):
    return load_managers_nerfstudio_single([input_model_path], num_parts)[0]

def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth, use_semantics=True)

def generate_tensorboard_seg_viz(predicted_semantics, gt_semantics, gt_rgb, num_parts):
    combined_rgb = torch.cat([gt_rgb, gt_rgb], dim=1)[..., :3] # [H, 2W, 3] - make sure we remove alpha channel
    # print(f"combined_rgb shape: {combined_rgb.shape}")

    combined_masks = []
    # turn predicted semantics into labels instead of logits
    predicted_semantics = torch.argmax(predicted_semantics, dim=-1, keepdim=True)
    # generate the list of masks from the gt and pred images
    for i in range(num_parts):
        gt_mask_i = gt_semantics == i # [H, W]
        pred_mask_i = predicted_semantics == i # [H, W]
        combined_masks_i = torch.cat([gt_mask_i, pred_mask_i.squeeze()], dim=1) # [H, 2W]
        combined_masks.append(combined_masks_i)
    # stack all the masks into a single tensor
    combined_segmentation = torch.stack(combined_masks, dim=0).squeeze() # [C, H, 2W]
    # switch from [H, W, C] to [C, H, W] for torchvision/tensorboard draw
    combined_rgb_draw = torch.moveaxis(combined_rgb, -1, 0)
    combined_segmentation = torchvision.utils.draw_segmentation_masks(
        image=(combined_rgb_draw * 255.0).to(dtype=torch.uint8),
        masks=combined_segmentation,
        alpha=0.75
    ).to(dtype=torch.float32) / 255.0
    # switch back to [H, W, C] if needed
    # combined_segmentation = torch.moveaxis(combined_segmentation, 0, -1)
    return combined_segmentation

def learn_segmentations(manager_paths: list[str], dataset_paths: list[str], num_parts: int, output_dir: str, exp_name: str = ""):
    batch_size = 1
    splat_managers = [load_base_model(path, num_parts) for path in manager_paths]
    splat_datasets = [get_dataset(path) for path in dataset_paths]
    splat_dataloaders = [DataLoader(dataset, batch_size=1, shuffle=True) for dataset in splat_datasets]

    n_managers = len(splat_managers)


    # create the writer
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if exp_name == "":
        exp_dir = os.path.join(output_dir, run_datetime)
    else:
        exp_dir = os.path.join(output_dir, exp_name)
    writer = SummaryWriter(exp_dir)

    for i in range(n_managers):
        print(f"Learning semantics for manager {i}...")
        # set the num parts for the manager object params
        cur_manager = splat_managers[i]
        cur_dataset = splat_datasets[i]
        cur_dataloader = splat_dataloaders[i]

        # create the optimization parameters
        optimization_params = [cur_manager.object_gaussian_params["features_semantics"]]

        # create the optimizer
        optimizer = torch.optim.Adam(optimization_params, lr=0.01)
        # create the scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # create the loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        cam_intrinsic = torch.Tensor([[cur_dataset.fl_x, 0, cur_dataset.cx],\
                                [0, cur_dataset.fl_y, cur_dataset.cy],\
                                [0, 0, 1]])
        width = cur_dataset.width
        height = cur_dataset.height

        n_epochs = 10
        for epoch in tqdm(range(n_epochs)):
            loss_total = 0
            for batch_idx, batch_data in tqdm(cur_dataloader, leave=False):
                # get the data
                gt_rgb = batch_data["rgb"].to(cur_manager.device)
                semantics_gt = batch_data["semantics"].to(torch.int64).to(cur_manager.device)
                render_poses = batch_data["transform_matrix"].to(cur_manager.device)

                # forward pass
                optimizer.zero_grad()
                batched_cam_intrinsic = cam_intrinsic.unsqueeze(0).expand(batch_size, -1, -1)
                semantics_pred, _, _ = cur_manager.render_at_campose(render_poses, batched_cam_intrinsic, width, height, is_semantics=True)

                # compute the loss
                loss = loss_fn(semantics_pred.reshape(-1, num_parts), semantics_gt.flatten())
                # accumulate the loss
                loss_total += loss.item()
                # backward pass
                loss.backward()
                # update the optimizer
                optimizer.step()
                if(0 in batch_idx):
                    # log the images
                    combined_segmentation = generate_tensorboard_seg_viz(semantics_pred[0], semantics_gt[0], gt_rgb[0], num_parts)
                    writer.add_image(f"segmentation_{i}", combined_segmentation, epoch)
            # log the loss
            writer.add_scalar(f"loss_{i}", loss_total, epoch)
            # update the scheduler
            scheduler.step()
        
        # save the updated manager
        save_fname = os.path.join(exp_dir, f"seg_learned_manager_{i}.pth")
        print(f"Saving learned manager {i} to {save_fname}...")
        torch.save(cur_manager, save_fname)

if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Given a list of splats already trained on RGB, learn segmentations for the parts")

    parser.add_argument('--splat_tf_manager_pths', 
                        type=str,
                        help='pre-trained splats to learn on',
                        default="")
    
    parser.add_argument('--splat_model_datasets',
                        type=str,
                        help="list of datasets used to train the splats",
                        default="")
    
    parser.add_argument('--n_parts',
                        type=int,
                        help="number of parts in the object/scene",
                        default=2)
    
    parser.add_argument('--output_dir',
                        type=str,
                        help="output directory for the experiment",
                        default="")
    
    parser.add_argument('--exp_name',
                        type=str,
                        help="name of the experiment",
                        default="")
    
    args = parser.parse_args()
    manager_paths = args.splat_tf_manager_pths.split(",")
    dataset_paths = args.splat_model_datasets.split(",")
    learn_segmentations(manager_paths, dataset_paths, args.n_parts, args.output_dir, args.exp_name)
import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM
import numpy as np
import json
import cv2 as cv
from pathlib import Path
import tqdm
import pytorch3d.transforms as p3dt


from scipy.spatial.transform import Rotation as R

from splatart.managers.SplatManager import SplatManager, load_managers
from splatart.datasets.splat_train_dataset import SplatTrainDataset

ssim = SSIM(data_range=1.0, size_average=True, channel=3)

def get_dataset(dataset_dir):
    transforms_json = os.path.join(dataset_dir, "transforms.json")
    dataset_json = json.load(open(transforms_json, "r"))

def l1_loss(pred, gt):
    return torch.mean(torch.abs(pred - gt))

def l2_loss(pred, gt):
    return torch.mean(torch.pow(pred - gt, 2))

def semantic_loss(pred, gt):
    # cross entropy loss
    return torch.nn.functional.cross_entropy(pred, gt)

def ssim_loss(pred, gt):
    return ssim(gt.permute(0, 3, 1, 2), pred.permute(0, 3, 1 ,2)) # permute to deal with CHW vs HWC formats

def process_models(input_model_dirs:list[str],\
                config_yml:str,\
                dataparser_tf:str,
                num_classes:int,\
                dataset_dir:str,\
                output_dir:str,\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/"):
    # load the existing splat managers
    splat_managers = []
    for i in range(len(input_model_dirs)):
        manager_path = os.path.join(output_dir, f"splat_manager_{i}.pth")
        manager = torch.load(manager_path)
        splat_managers.append(manager)

    canonical_manager = splat_managers[0] # is of type SplatManager

    n_epochs = 50
    img_write_moduo = 50
    epoch_write_module = 10
    batch_size = 50

    num_other_managers = len(splat_managers) - 1

    opt_loss = torch.compile(l2_loss)
    for other_manager_idx in range(1, num_other_managers + 1):
        
        other_manager = splat_managers[other_manager_idx] # is of type SplatManager
        print("getting the training dataset...")
        dataset = SplatTrainDataset(dataset_dir)
        cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx], [0, dataset.fl_y, dataset.cy], [0, 0, 1]])
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for part_num in tqdm.tqdm(range(2,num_classes)): # skip the first two background classes
            # creat the optimizer and parameters
            optimization_params = []
            # set quats and trans to random values for optimization initially?
            # other_manager.parts_gauss_params[part_num]["tf_quats"] = p3dt.matrix_to_quaternion(p3dt.euler_angles_to_matrix(torch.rand(3)*0.1, "XYZ").to(torch.float32).cuda())
            # other_manager.parts_gauss_params[part_num]["tf_trans"] = torch.rand(3).to(torch.float32).cuda() * 0.05
            optimization_params.append(other_manager.parts_gauss_params[part_num]["tf_quats"])
            optimization_params.append(other_manager.parts_gauss_params[part_num]["tf_trans"])
            optimizer = torch.optim.Adam(optimization_params, lr=0.005)

            for epoch in tqdm.tqdm(range(n_epochs), leave = False):
                
                loss_sum = 0

                for i, entry in tqdm.tqdm(train_dataloader, leave=False):
                    # zero out the gradients
                    optimizer.zero_grad()

                    gt_cam_pose = entry["transform_matrix"]
                    gt_rgb = entry["rgb"].cuda().to(torch.float32) / 255.0

                    # check if gt_rgb has alpha channel
                    # if so, make it black
                    if gt_rgb.shape[-1] == 4:
                        gt_rgb = gt_rgb[..., :3]

                    gt_semantics = entry["semantics"]

                    # merge the manager under training with the canonical manager for rendering
                    # other_part_gauss_params = other_manager.parts_gauss_params[part_num]
                    # # modify the other manager's tf params
                    # blade_gauss_params["tf_quats"] = torch.Tensor(R.from_euler("xyz", [0, 0, 90], degrees=True).as_quat(scalar_first=True))
                    # blade_gauss_params["tf_trans"] = torch.Tensor([0, 0, 0])
                    # render_gauss_params = [canonical_manager.object_gaussian_params, other_part_gauss_params]

                    # render canonical at the cam_pose
                    # img, _, _ = canonical_manager.render_gauss_params_at_campose(gt_cam_pose, cam_intrinsic, dataset.width, dataset.height, render_gauss_params)

                    # render just the parts at cam pose
                    batched_cam_intrinsic = cam_intrinsic.unsqueeze(0).repeat(gt_cam_pose.shape[0], 1, 1)
                    part_img, _, _ = canonical_manager.render_parts_at_campose(gt_cam_pose, batched_cam_intrinsic, dataset.width, dataset.height, [part_num])
                    part_other_img, _, _ = other_manager.render_parts_at_campose(gt_cam_pose, batched_cam_intrinsic, dataset.width, dataset.height, [part_num])

                    # get the transformed means from the canonical_manager


                    # get the losses
                    # loss_l1 = l1_loss(img, gt_rgb)
                    # loss_l2 = l2_loss(img, gt_rgb)
                    # loss_ssim = ssim_loss(img, gt_rgb)

                    # get the part only losses
                    # loss_part_l1 = l1_loss(part_img, part_other_img)
                    loss_part_l2 = opt_loss(part_img, part_other_img)#l2_loss(part_img, part_other_img)
                    # loss_part_ssim = ssim_loss(part_img, part_other_img)

                    # compute the loss and backprop
                    loss = loss_part_l2
                    loss_sum += loss.item()
                    loss.backward()
                    optimizer.step()

                    # print(f"losses: l1={loss_l1.item()}, l2={loss_l2.item()}, ssim={loss_ssim.item()}")
                    if(epoch % epoch_write_module == 0 or epoch == n_epochs - 1):
                        for idx in i:
                            if(idx % img_write_moduo == 0):
                                os.makedirs(f"{output_dir}/part{part_num}/scene{other_manager_idx}/epoch{epoch}/", exist_ok=True)
                                cv.imwrite(f"{output_dir}/part{part_num}/scene{other_manager_idx}/epoch{epoch}/rgb{idx.item()}_rendered.png", part_img.cpu().detach().numpy()[0,:,:,::-1] * 255.0)
                                cv.imwrite(f"{output_dir}/part{part_num}/scene{other_manager_idx}/epoch{epoch}/rgb{idx.item()}_gt.png", part_other_img.cpu().detach().numpy()[0,:,:,::-1] * 255.0)
                
                avg_loss = loss_sum / len(train_dataloader)
                tqdm.tqdm.write(f"avg loss: {avg_loss}")

            tf_quats_final = other_manager.parts_gauss_params[part_num]["tf_quats"]
            tf_trans_final = other_manager.parts_gauss_params[part_num]["tf_trans"]

            other_means, other_quats = other_manager.get_render_gauss_params(other_manager.parts_gauss_params[part_num])
            canonical_means, canonical_quats = canonical_manager.get_render_gauss_params(canonical_manager.parts_gauss_params[part_num])

            # compare the averages
            avg_other_means = torch.mean(other_means, dim=0)
            avg_canonical_means = torch.mean(canonical_means, dim=0)
            print(f"avg_other_means: {avg_other_means}, avg_canonical_means: {avg_canonical_means}")
            delta_means = avg_other_means - avg_canonical_means
            print(f"avg_delta: {torch.norm(delta_means)}")
                


    # save the managers back to disk
    for i, manager in enumerate(splat_managers):
        manager_path = os.path.join(output_dir, f"splat_manager_registered_{i}.pth")
        torch.save(manager, manager_path)
        print(f"Saved splat manager to {manager_path}")


if __name__ == "__main__":
    print("Getting gaussians and part seperations from pretrained splats...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--input_model_dirs', 
                        type=str,
                        help='list of directories containing the pre-trained splats we care about (comma delimited)',
                        default="")
    parser.add_argument('--num_classes',
                        type=int,
                        help='number of parts to seperate out from the splats',
                        default=4)
    parser.add_argument('--config_yml_name',
                        type=str,
                        help='name of the config file that contains the model information (relative to input_model_dirs)',
                        default="config.yml")
    parser.add_argument('--dataparser_tf_name',
                        type=str,
                        help='name of the dataparser transform file (relative to input_model_dirs)',
                        default="dataparser_transforms.json")
    parser.add_argument('--canonical_model_dataset',
                        type=str,
                        help='directory of the canonical model dataset to use as a reference',
                        default="")
    parser.add_argument('--output_dir',
                        type=str,
                        help='output directory to save the results',
                        default="./results/sapien/blade/")
    
    args = parser.parse_args()

    input_model_dirs = args.input_model_dirs.split(",")
    num_classes = args.num_classes
    output_dir = args.output_dir
    config_yml = args.config_yml_name
    dataparser_tf = args.dataparser_tf_name

    dataset_dir = args.canonical_model_dataset

    process_models(input_model_dirs, config_yml, dataparser_tf, num_classes, dataset_dir, output_dir)
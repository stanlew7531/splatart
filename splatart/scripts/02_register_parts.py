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
import pytorch3d.ops as p3dops
import torchvision


from scipy.spatial.transform import Rotation as R

from splatart.managers.SplatManager import SplatManager, load_managers
from splatart.datasets.splat_train_dataset import SplatTrainDataset

from nerfstudio.models.splatfacto import SH2RGB

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
    return 1 - ssim(gt.permute(0, 3, 1, 2), pred.permute(0, 3, 1 ,2)) # permute to deal with CHW vs HWC formats

def alpha_loss(pred_alpha, gt_alpha):
    intersection_alpha = pred_alpha * gt_alpha
    union_alpha = pred_alpha + gt_alpha - intersection_alpha
    return torch.exp(-1 * (torch.sum(intersection_alpha) / torch.sum(union_alpha)))

def get_sub_images(image, alpha, threshold=0.01, resize_shape=(256, 256)):
    alpha_row_maxs = torch.max(alpha, dim=2).values
    alpha_col_maxs = torch.max(alpha, dim=1).values
    rows_clear_thresh = (alpha_row_maxs > threshold).to(dtype=torch.int8)
    cols_clear_thresh = (alpha_col_maxs > threshold).to(dtype=torch.int8)
    num_rows = rows_clear_thresh.shape[1]
    num_cols = cols_clear_thresh.shape[1]
    alpha_row_min = torch.argmax(rows_clear_thresh, dim=1)
    # print(alpha_row_min)
    alpha_row_max = num_rows - torch.argmax(rows_clear_thresh.flip(1), dim=1)
    # print(alpha_row_max)
    alpha_col_min = torch.argmax(cols_clear_thresh, dim=1)
    alpha_col_max = num_cols - torch.argmax(cols_clear_thresh.flip(1), dim=1)

    # print(f"alpha_row_min: {alpha_row_min}, alpha_row_max: {alpha_row_max}, alpha_col_min: {alpha_col_min}, alpha_col_max: {alpha_col_max}")
    sub_images = []
    for b in range(image.shape[0]):
        sub_image = image[b, alpha_row_min[b]:alpha_row_max[b], alpha_col_min[b]:alpha_col_max[b]]
        # need c,h,w format
        sub_image = sub_image.permute(2, 0, 1)
        sub_image = torchvision.transforms.functional.resize(sub_image, resize_shape)
        # change back
        sub_image = sub_image.permute(1, 2, 0)
        sub_images.append(sub_image)

    return torch.stack(sub_images)


def process_models(input_model_dirs:list[str],\
                config_yml:str,\
                dataparser_tf:str,
                num_classes:int,\
                dataset_dir:str,\
                output_dir:str,\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/",
                parts_to_skip:list[int] = [0,1]):
    # load the existing splat managers
    splat_managers = []
    dataparser_tfs = []
    for i in range(len(input_model_dirs)):
        manager_path = os.path.join(output_dir, f"splat_manager_{i}.pth")
        dataparser_tf_path = os.path.join(input_model_dirs[i], dataparser_tf)
        dataparser_data = json.load(open(dataparser_tf_path, "r"))
        dataparser_tfs.append(dataparser_data)
        manager = torch.load(manager_path)
        splat_managers.append(manager)

    canonical_manager = splat_managers[0] # is of type SplatManager

    n_epochs = 50
    img_write_moduo = 10
    epoch_write_module = 10
    batch_size = 5

    num_managers = len(splat_managers)

    opt_loss = l1_loss#ssim_loss# torch.compile(ssim_loss)
    for other_manager_idx in range(1, num_managers):
        
        other_manager = splat_managers[other_manager_idx] # is of type SplatManager
        print("getting the training dataset...")
        dataset = SplatTrainDataset(dataset_dir)
        cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx], [0, dataset.fl_y, dataset.cy], [0, 0, 1]])
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for part_num in tqdm.tqdm(range(0,num_classes)): # skip the first two background classes
            if(part_num in parts_to_skip):
                continue
            # creat the optimizer and parameters
            optimization_params = []

            canonical_means = canonical_manager.parts_gauss_params[part_num]["means"]
            other_means = other_manager.parts_gauss_params[part_num]["means"]
            if(len(canonical_means.shape) ==2 ):
                canonical_means = canonical_means.unsqueeze(0)
                other_means = other_means.unsqueeze(0)
            canonical_means = canonical_means / dataparser_tfs[0]["scale"]
            other_means = other_means / dataparser_tfs[other_manager_idx]["scale"]
            canonical_dataparser_tf = dataparser_tfs[0]["transform"]
            other_dataparser_tf = dataparser_tfs[other_manager_idx]["transform"]


            with torch.no_grad():
                degree_angles = torch.linspace(0, 360, 15)
                rad_angles = degree_angles * (np.pi / 180)
                pre_check_angles = torch.cartesian_prod(rad_angles, rad_angles, rad_angles)
                pre_check_losses = []
                # get the first entry in the dataset
                entry = dataset[0][1]
                precheck_cam_pose = entry["transform_matrix"].unsqueeze(0)
                batched_cam_intrinsic = cam_intrinsic.unsqueeze(0).repeat(precheck_cam_pose.shape[0], 1, 1)
                print(f"precheck_cam_pose: {precheck_cam_pose}")
                precheck_part_img, precheck_part_alpha, _ = canonical_manager.render_parts_at_campose(precheck_cam_pose, batched_cam_intrinsic, dataset.width, dataset.height, [part_num])
                precheck_part_sub_imgs = get_sub_images(precheck_part_img, precheck_part_alpha)
                quat_cache = other_manager.parts_gauss_params[part_num]["tf_quats"]
                precheck_other_sub_img_cache = []
                for i in tqdm.tqdm(range(pre_check_angles.shape[0])):
                    # turn the precheck angles into a quat
                    precheck_quat = p3dt.matrix_to_quaternion(p3dt.euler_angles_to_matrix(pre_check_angles[i], "XYZ"))
                    other_manager.parts_gauss_params[part_num]["tf_quats"] = precheck_quat
                    part_other_img, part_other_alpha, _ = other_manager.render_parts_at_campose(precheck_cam_pose, batched_cam_intrinsic, dataset.width, dataset.height, [part_num])
                    part_other_sub_imgs = get_sub_images(part_other_img, part_other_alpha)
                    precheck_other_sub_img_cache.append(part_other_sub_imgs)
                    loss_rot = opt_loss(precheck_part_sub_imgs, part_other_sub_imgs)
                    pre_check_losses.append(loss_rot.item())
                # get the minimum precheck loss
                min_loss_idx = torch.argmin(torch.tensor(pre_check_losses))
                # get the associated precheck angle
                precheck_angle = pre_check_angles[min_loss_idx]
                # turn it into a quat
                precheck_quat = p3dt.matrix_to_quaternion(p3dt.euler_angles_to_matrix(precheck_angle, "XYZ"))
                cv.imwrite("best_precheck_canonical.png", precheck_part_sub_imgs.cpu().detach().numpy()[0,:,:,::-1] * 255.0)
                cv.imwrite("best_precheck_other.png", precheck_other_sub_img_cache[min_loss_idx].cpu().detach().numpy()[0,:,:,::-1] * 255.0)
                print(f"min loss: {pre_check_losses[min_loss_idx]}, precheck_angle: {precheck_angle}, precheck_quat: {precheck_quat}")

                other_manager.parts_gauss_params[part_num]["tf_quats"] = precheck_quat

            optimization_params.append(other_manager.parts_gauss_params[part_num]["tf_quats"])
            optimization_params.append(other_manager.parts_gauss_params[part_num]["tf_trans"])
            
            optimizer = torch.optim.Adam(optimization_params, lr = 0.005)
            optimizer_trans = torch.optim.Adam([other_manager.parts_gauss_params[part_num]["tf_trans"]], lr = .05)
            optimizer_rotation = torch.optim.Adam([other_manager.parts_gauss_params[part_num]["tf_quats"]], lr=0.005)

            for epoch in tqdm.tqdm(range(n_epochs), leave = False):
                
                loss_sum = 0

                for i, entry in tqdm.tqdm(train_dataloader, leave=False):
                    # zero out the gradients
                    optimizer_trans.zero_grad()
                    optimizer_rotation.zero_grad()
                    optimizer.zero_grad()

                    gt_cam_pose = entry["transform_matrix"]
                    gt_rgb = entry["rgb"].cuda().to(torch.float32) / 255.0

                    # check if gt_rgb has alpha channel
                    # if so, make it black
                    if gt_rgb.shape[-1] == 4:
                        gt_rgb = gt_rgb[..., :3]

                    # render just the parts at cam pose
                    batched_cam_intrinsic = cam_intrinsic.unsqueeze(0).repeat(gt_cam_pose.shape[0], 1, 1)
                    bigger_intrinsic = batched_cam_intrinsic.clone()
                    expansion_factor = 1.0
                    bigger_intrinsic[:, 0, 2] *= expansion_factor
                    bigger_intrinsic[:, 1, 2] *= expansion_factor
                    
                    part_img, part_alpha, _ = canonical_manager.render_parts_at_campose(gt_cam_pose, bigger_intrinsic, int(dataset.width*expansion_factor), int(dataset.height*expansion_factor), [part_num])
                    part_other_img, part_other_alpha, _ = other_manager.render_parts_at_campose(gt_cam_pose, bigger_intrinsic, int(dataset.width*expansion_factor), int(dataset.height*expansion_factor), [part_num])

                    # compute the loss and backprop
                    loss = l1_loss(part_img, part_other_img) + alpha_loss(part_alpha, part_other_alpha)
                    loss_sum += loss.item()
                    loss.backward()
                    optimizer_trans.step()
                    optimizer_rotation.step()
                    # optimizer.step()

                    # print(f"losses: l1={loss_l1.item()}, l2={loss_l2.item()}, ssim={loss_ssim.item()}")
                    if(epoch % epoch_write_module == 0 or epoch == n_epochs - 1):
                        for idx in range(len(i)):
                            src_idx = i[idx]
                            if(src_idx % img_write_moduo == 0):
                                os.makedirs(f"{output_dir}/part{part_num}/scene{other_manager_idx}/epoch{epoch}/", exist_ok=True)
                                cv.imwrite(f"{output_dir}/part{part_num}/scene{other_manager_idx}/epoch{epoch}/rgb{src_idx.item()}_canonical.png", part_img.cpu().detach().numpy()[idx,:,:,::-1] * 255.0)
                                cv.imwrite(f"{output_dir}/part{part_num}/scene{other_manager_idx}/epoch{epoch}/rgb{src_idx.item()}_other.png", part_other_img.cpu().detach().numpy()[idx,:,:,::-1] * 255.0)

                avg_loss = loss_sum / len(train_dataloader)
                tqdm.tqdm.write(f"avg loss: {avg_loss}")

            # tf_quats_final = other_manager.parts_gauss_params[part_num]["tf_quats"]
            # tf_trans_final = other_manager.parts_gauss_params[part_num]["tf_trans"]

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
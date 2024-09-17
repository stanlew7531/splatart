from functools import *
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from typing import Optional
import argparse
import cv2
import pickle
import os
import torch
from tqdm import tqdm
from splatart.gui.CloudVis import CloudVisApp
import json

import pytorch3d.transforms as p3dt

from splatart.utils.lie_utils import SE3, SE3Exp, SO3Exp
from splatart.managers.JointManager import JointManager
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint, SingleAxisJoint
from splatart.managers.SplatManager import SplatManager, load_managers
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.utils.helpers import convert_cam_opengl_opencv


def draw_pose_in_image(image, pose, intrinsic, axis_length=0.1):
    assert pose.shape == (4, 4), "Pose must be a 4x4 matrix"
    # get rvec and tvecs
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    tvec = pose[:3, 3]

    origin, _  = cv2.projectPoints(np.float32([[0, 0, 0]]), rvec, tvec, intrinsic, None)
    origin = tuple(origin.ravel().astype(int))

    axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, intrinsic, None)

    image_with_axis = cv2.line(image, origin, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)  # X-axis in red
    image_with_axis = cv2.line(image_with_axis, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)  # Y-axis in green
    image_with_axis = cv2.line(image_with_axis, origin, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)  # Z-axis in blue
    return image_with_axis

    
@torch.no_grad()
def generate_part_datasets(input_model_dirs:list[str],\
                config_yml:str,\
                dataparser_tf:str,
                num_classes:int,\
                dataset_dirs:list[str],\
                output_dir:str,\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/",
                parts_to_ignore = [0,1]):
    
    splat_managers = load_managers(input_model_dirs, output_dir, "splat_manager_registered")

    idxs_to_generate = list(range(num_classes))
    idxs_to_generate = [idx for idx in idxs_to_generate if idx not in parts_to_ignore]

    splat_datasets = [SplatTrainDataset(dataset_dir) for dataset_dir in dataset_dirs]

    for part_idx in idxs_to_generate:
        print(f"Generating dataset for part {part_idx}...")
        part_output_dir = os.path.join(output_dir, f"part_{part_idx}")
        part_transforms_json = {}
        part_transforms_json["fl_x"] = splat_datasets[0].fl_x
        part_transforms_json["fl_y"] = splat_datasets[0].fl_y
        part_transforms_json["cx"] = splat_datasets[0].cx
        part_transforms_json["cy"] = splat_datasets[0].cy
        part_transforms_json["w"] = splat_datasets[0].width
        part_transforms_json["h"] = splat_datasets[0].height
        part_transforms_json["camera_model"] = "OPENCV"
        part_transforms_json["frames"] = []

        os.makedirs(part_output_dir, exist_ok=True)
        os.makedirs(os.path.join(part_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(part_output_dir, "masks"), exist_ok=True)
        print(f"writing output to: {part_output_dir}")
        # we want to make a new set of images and masks for this part
        # which have our estimated transform applied to them
        dataset_idxs = list(range(len(splat_datasets)))
        for dataset_idx in dataset_idxs:
            print(f"Processing dataset {dataset_idx}...")
            dataset = splat_datasets[dataset_idx]
            for i in tqdm(range(len(dataset))):
                entry = dataset[i][1]
                gt_cam_pose = entry["transform_matrix"]
                original_image = entry["rgb"] # H, W, 4
                original_segmentation = entry["semantics"] # H, W

                cam_intrinsic = dataset.get_intrinsic()
                width = dataset.width
                height = dataset.height
                # render out at the gt pose for inspection
                splat_manager = splat_managers[dataset_idx]
                learned_quat = splat_manager.parts_gauss_params[part_idx]["tf_quats"]
                learned_trans = splat_manager.parts_gauss_params[part_idx]["tf_trans"]
                # print(f"learned quat: {learned_quat}, learned trans:{learned_trans}")
                transforms_tf = torch.eye(4)
                transforms_tf[:3, 3] = learned_trans
                transforms_tf[:3, :3] = p3dt.quaternion_to_matrix(learned_quat)
                # print(f"transforms_tf: {transforms_tf}")
                transformed_camera = torch.inverse(transforms_tf) @ gt_cam_pose
                new_camera = transforms_tf @ gt_cam_pose

                cam_part_img, cam_part_alpha, _ = splat_manager.render_parts_at_campose(gt_cam_pose, cam_intrinsic, width, height, [part_idx], apply_transforms=False)
                cam_part_alpha[cam_part_alpha > 1] = 1.0 # clamp alpha to [0, 1]
                # apply the rendered alpha to the original image
                new_train_img = original_image.clone()
                new_train_img[..., -1] = cam_part_alpha[0, ..., 0] * 255
                # also make sure our part idx from segmentation has full alpha
                # wherever original segmentation matches the part_idx, make sure the alpha is 255
                new_train_img[..., 3] = torch.where(original_segmentation == part_idx, torch.tensor(255, dtype=original_image.dtype, device=original_image.device), new_train_img[..., 3])
                
                
                # raise Exception("ASdf")
                # create the training mask
                # we train on all pixels, except for those which have a nonzero alpha value in the rendering
                # and also have a segmentation mask value other than the part we are rendering
                rendering_alpha_mask = cam_part_alpha[0, ..., 0] > 0
                train_mask = torch.ones_like(rendering_alpha_mask)
                alpha_masked_segmentation = original_segmentation * rendering_alpha_mask.to(original_segmentation) # mask out the segmentation
                os.makedirs(os.path.join(part_output_dir, "tests"), exist_ok=True)
                alpha_masked_segmentation[alpha_masked_segmentation == part_idx] = 0 # mask out the part we are rendering
                # reverse the mask
                train_mask[alpha_masked_segmentation > 0] = 0

                # append the alpha to the original image
                cam_part_img = torch.cat([cam_part_img, cam_part_alpha], dim=-1)

                # deal with bgra vs rgba
                new_train_img = new_train_img.cpu().numpy()
                new_train_img[..., :3] = new_train_img[..., -2::-1] # reverse the RGB channels
                cam_part_img = cam_part_img.cpu().numpy()
                cam_part_img[..., :3] = cam_part_img[..., -2::-1]
                cv2.imwrite(os.path.join(part_output_dir, "images", f"dataset_{dataset_idx}_entry_{i}.png"), new_train_img)
                cv2.imwrite(os.path.join(part_output_dir, "masks", f"dataset_{dataset_idx}_entry_{i}.png"), train_mask.cpu().numpy() * 255.0)
                
                frames_entry = {}
                frames_entry["file_path"] = os.path.join("images", f"dataset_{dataset_idx}_entry_{i}.png")
                frames_entry["transform_matrix"] = new_camera.cpu().numpy().tolist() #transformed_camera.cpu().numpy().tolist()
                frames_entry["mask_path"] = os.path.join("masks", f"dataset_{dataset_idx}_entry_{i}.png")
                part_transforms_json["frames"].append(frames_entry)


                # part_img, _, _ = splat_manager.render_parts_at_campose(gt_cam_pose, cam_intrinsic, width, height, [part_idx], apply_transforms=False)
                # learned_tf_part_img, _, _ = splat_manager.render_parts_at_campose(gt_cam_pose, cam_intrinsic, width, height, [part_idx], apply_transforms=True)
                

                # cv2.imwrite(os.path.join(part_output_dir, f"dataset_{dataset_idx}_entry_{i}_part_{part_idx}.png"), part_img[0].cpu().numpy() * 255.0)
                # cv2.imwrite(os.path.join(part_output_dir, f"dataset_{dataset_idx}_entry_{i}_learned_tf_part_{part_idx}.png"), learned_tf_part_img[0].cpu().numpy() * 255.0)
                # cv2.imwrite(os.path.join(part_output_dir, f"dataset_{dataset_idx}_entry_{i}_moved_cam_part_{part_idx}.png"), moved_cam_part_img[0].cpu().numpy() * 255.0)
        json.dump(part_transforms_json, open(os.path.join(part_output_dir, "transforms.json"), "w"), indent=4)
        
if __name__ == "__main__":
    print("Getting gaussians and part seperations from pretrained splats...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--input_model_dirs', 
                        type=str,
                        help='list of directories containing the pre-trained and parts-registered splats we care about (comma delimited)',
                        default="")
    parser.add_argument('--num_classes',
                        type=int,
                        help='number of parts to estimate joints of',
                        default=4)
    parser.add_argument('--config_yml_name',
                        type=str,
                        help='name of the config file that contains the model information (relative to input_model_dirs)',
                        default="config.yml")
    parser.add_argument('--dataparser_tf_name',
                        type=str,
                        help='name of the dataparser transform file (relative to input_model_dirs)',
                        default="dataparser_transforms.json")
    parser.add_argument('--input_datasets',
                        type=str,
                        help='list of dataset transform.jsons originally used to train the splats (comma delimited)',
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

    dataset_dirs = args.input_datasets.split(",")
    print(f"input datasets: {dataset_dirs}")

    generate_part_datasets(input_model_dirs, config_yml, dataparser_tf, num_classes, dataset_dirs, output_dir)
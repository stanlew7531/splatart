import sys
import os
import argparse
import torch
import numpy as np
import json
import cv2 as cv
from pathlib import Path

from scipy.spatial.transform import Rotation as R

import splatart.utils.constants as constants
from splatart.managers.SplatManager import SplatManager, load_managers

def process_models(input_model_dirs:list[str],\
                config_yml:str,\
                dataparser_tf:str,
                num_classes:int,\
                output_dir:str,\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/"):
    os.makedirs(output_dir, exist_ok=True)
    splat_managers = []
    # splat_managers = load_managers(input_model_dirs, num_classes, config_yml, dataparser_tf, ns_base_path)

    for i in range(len(input_model_dirs)):
        manager_path = os.path.join(output_dir, f"splat_manager_{i}.pth")
        manager = torch.load(manager_path)
        splat_managers.append(manager)
    
    print(splat_managers)

    i = 0
    for manager in splat_managers:
        
        example_cam_pose = constants.example_cam_pose
        fx = fy = constants.example_focal
        w = constants.example_width
        h = constants.example_height

        # render the full object for inspection
        rgb, alpha, info = manager.render_object_at_campose(example_cam_pose, torch.Tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]]), w, h)

        rgb = rgb.cpu().detach().numpy()[0]
        cv.imwrite(f"{output_dir}/rgb{i}.png", rgb * 255.0)

        # render each part for inspection
        for partidx in [0,1,2,3]:
            manager.parts_gauss_params[partidx]["tf_trans"] = torch.Tensor([0, 0, 0])
            manager.parts_gauss_params[partidx]["tf_quats"] = torch.Tensor(R.from_euler("xyz", [0, 0, 90], degrees=True).as_quat(scalar_first=True))
            rgb, alpha, info = manager.render_parts_at_campose(example_cam_pose, torch.Tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]]), w, h, [partidx])
            rgb = rgb.cpu().detach().numpy()[0]
            cv.imwrite(f"{output_dir}/rgb{i}_part{partidx}.png", rgb * 255)
        i+=1


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

    process_models(input_model_dirs, config_yml, dataparser_tf, num_classes, output_dir)
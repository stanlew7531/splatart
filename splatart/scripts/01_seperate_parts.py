import sys
import os
import argparse
import torch
import numpy as np
import json
import cv2 as cv
from pathlib import Path

from scipy.spatial.transform import Rotation as R

from splatart.managers.SplatManager import SplatManager, load_managers_nerfstudio

def process_models(input_model_dirs:list[str],\
                config_yml:str,\
                dataparser_tf:str,
                num_classes:int,\
                output_dir:str,\
                ns_base_path:str = "/home/vishalchandra/Desktop/nerfstudio_dev/"):
    os.makedirs(output_dir, exist_ok=True)
    splat_managers = []
    i = 0
    splat_managers = load_managers_nerfstudio(input_model_dirs, num_classes, config_yml, dataparser_tf, ns_base_path)

    for i, manager in enumerate(splat_managers):
        manager_path = os.path.join(output_dir, f"splat_manager_{i}.pth")
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
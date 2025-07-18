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

from splatart.utils.lie_utils import SE3, SE3Exp, SO3Exp
from splatart.managers.JointManager import JointManager
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint, SingleAxisJoint
from splatart.managers.SplatManager import SplatManager, load_managers_nerfstudio
from splatart.managers.ArticulatedSplatManager import ArticulatedSplatManager
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.utils.helpers import convert_cam_opengl_opencv

    
@torch.no_grad()
def rerender_configuration(input_model_dirs:list[str],\
                config_yml:str,\
                config_mins:list[float],\
                config_maxs:list[float],\
                num_steps:int,\
                parts:list[int],\
                dataparser_tf:str,
                dataset_dir:str,\
                output_dir:str,\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/",
                configuration_vector_pkl:str = "configuration_vector.pkl"):
    
    art_splat_manager = ArticulatedSplatManager(input_model_dirs, os.path.join(output_dir, configuration_vector_pkl))
    print(art_splat_manager.configuration_vector)

    # get a dataset so that we can render at train poses
    dataset = SplatTrainDataset(dataset_dir)
    _, inspection_data = dataset[0]
    gt_cam_pose = inspection_data["transform_matrix"]
    # convert the gt cam pose from OpenGL to OpenCV format
    gt_cam_pose_ocv = convert_cam_opengl_opencv(gt_cam_pose)
    gt_img = inspection_data["rgb"]
    gt_semantics = inspection_data["semantics"]
    cam_intrinsic = dataset.get_intrinsic()
    for alpha in tqdm(torch.linspace(0, 1.0, num_steps)):
        configs = [[config_mins[i] + alpha * (config_maxs[i] - config_mins[i])] for i in range(len(config_mins))]
        configs = torch.Tensor(configs)
        img, _, _ = art_splat_manager.render_parts_at_campose_with_configuration(gt_cam_pose, cam_intrinsic, dataset.width, dataset.height, parts, configs)
        cv2.imwrite(os.path.join(output_dir, f"rendered_image_{alpha}.png"), img[0].cpu().numpy()[:,:,::-1] * 255.0)

if __name__ == "__main__":
    print("Getting gaussians and part seperations from pretrained splats...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--input_model_dirs', 
                        type=str,
                        help='list of directories containing the pre-trained and parts-registered splats we care about (comma delimited)',
                        default="")
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
    parser.add_argument('--dataset_pose_idx',
                        type=int,
                        help='index of dataset entry to get the render pose from',
                        default=0)
    parser.add_argument('--config_mins',
                        type=str,
                        help='min configuration values for the joints(command separated)',
                        default="0.0")
    parser.add_argument('--config_maxs',
                        type=str,
                        help='min configuration values for the joints(command separated)',
                        default="0.5")
    parser.add_argument('--num_steps',
                        type=int,
                        help='number of steps to take between min and max configuration values',
                        default=10)
    parser.add_argument('--parts',
                        type=str,
                        help='list of parts to render',
                        default="2,3,4,5,6")
    parser.add_argument('--output_dir',
                        type=str,
                        help='output directory to save the results',
                        default="./results/sapien/blade/")
    
    args = parser.parse_args()

    input_model_dirs = args.input_model_dirs.split(",")
    output_dir = args.output_dir
    config_yml = args.config_yml_name
    dataparser_tf = args.dataparser_tf_name

    dataset_dir = args.canonical_model_dataset

    config_mins = [float(min) for min in args.config_mins.split(",")]
    config_maxs = [float(max) for max in args.config_maxs.split(",")]
    parts = [int(part) for part in args.parts.split(",")]
    num_steps = args.num_steps

    rerender_configuration(input_model_dirs, config_yml, config_mins, config_maxs, num_steps, parts, dataparser_tf, dataset_dir, output_dir)
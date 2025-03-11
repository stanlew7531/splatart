import sys
import os
import open3d as o3d
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_msssim import SSIM
import pickle as pkl
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from plyfile import PlyElement, PlyData
from pytorch3d.loss import chamfer_distance

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import HDBSCAN
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats, apply_mat_tf_to_gauss_params
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.networks.PoseEstimator import PoseEstimator

def load_articulation_estimates(articulation_estimates_path:str):
    # get joint for joint metrics.
    with open(articulation_estimates_path, 'rb') as f:
        config_vector = pkl.load(f)
    return config_vector

def load_base_model(input_model_path:str)->SplatManagerSingle:
    return torch.load(input_model_path)

def load_part_gaussians(part_splats_path:str):
    return torch.load(part_splats_path)

def load_pose_estimator(pose_estimates_path:str)->PoseEstimator:
    return torch.load(pose_estimates_path)

def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth, use_semantics=True)

def combine_guass_params(gauss_params_list):
        combined_gauss_params = []
        object_gauss_params = {
            "means": gauss_params_list[0]["means"],
            "quats": gauss_params_list[0]["quats"],
            "features_semantics": gauss_params_list[0]["features_semantics"],
            "features_dc": gauss_params_list[0]["features_dc"],
            "features_rest": gauss_params_list[0]["features_rest"],
            "opacities": gauss_params_list[0]["opacities"],
            "scales": gauss_params_list[0]["scales"],
        }

        for part_i in range(1, len(gauss_params_list)):
            if part_i == 0:
                continue
            object_gauss_params["means"] = torch.cat((object_gauss_params["means"], gauss_params_list[part_i]["means"]), dim=0)
            object_gauss_params["quats"] = torch.cat((object_gauss_params["quats"], gauss_params_list[part_i]["quats"]), dim=0)
            object_gauss_params["features_semantics"] = torch.cat((object_gauss_params["features_semantics"], gauss_params_list[part_i]["features_semantics"]), dim=0)
            object_gauss_params["features_dc"] = torch.cat((object_gauss_params["features_dc"], gauss_params_list[part_i]["features_dc"]), dim=0)
            object_gauss_params["features_rest"] = torch.cat((object_gauss_params["features_rest"], gauss_params_list[part_i]["features_rest"]), dim=0)
            object_gauss_params["opacities"] = torch.cat((object_gauss_params["opacities"], gauss_params_list[part_i]["opacities"]), dim=0)
            object_gauss_params["scales"] = torch.cat((object_gauss_params["scales"], gauss_params_list[part_i]["scales"]), dim=0)

        return object_gauss_params

def render_images(object_name:str,\
                    manager_paths,\
                    dataset_paths,\
                    part_splats_path,\
                    pose_estimates_path,\
                    articuation_estimates_path,\
                    output_dir,\
                    canonical_scene_id=0,\
                    static_recon_part_id=1,\
                    dynamic_recon_part_id=2,\
                    sapien_root_dir="/media/stanlew/Data/paris_dataset/dataset/data/sapien"):
    torch.set_grad_enabled(False)
    splat_managers = [load_base_model(path) for path in manager_paths]
    dataparser_scales = [splatmanager.dataparser_scale for splatmanager in splat_managers]
    dataparser_tf_matrices = [splatmanager.dataparser_tf_matrix for splatmanager in splat_managers]
    print(f"dataparser tf matric start: {dataparser_tf_matrices[0]}")
    n_parts = splat_managers[0].num_parts
    recon_part_splats = load_part_gaussians(part_splats_path)
    pose_estimates = load_pose_estimator(pose_estimates_path)
    articulation_estimates = load_articulation_estimates(articuation_estimates_path)
    
    splat_datasets = [get_dataset(path) for path in dataset_paths]
    dataset = splat_datasets[0]
    _, inspection_data = dataset[0]
    gt_cam_pose = inspection_data["transform_matrix"].to(device="cuda:0")

    cam_intrinsic = torch.Tensor([[dataset.fl_x, 0, dataset.cx],\
                                [0, dataset.fl_y, dataset.cy],\
                                [0, 0, 1]])
    
    batch_size = 1
    width = dataset.width
    height = dataset.height

    # get the set of each parts gaussians for transformation

    part_frames, scene_part_frame_gauss_params = pose_estimates.get_part_splat_frames(recon_part_splats)
    recon_part_splats = recon_part_splats[canonical_scene_id]

    # get the joint between the static and dynamic part
    joint = articulation_estimates[0]["predicted_joint"]
    # get min and max articulations
    joint_params = joint.joint_params
    min_value = joint_params.min()
    max_value = joint_params.max() * -1
    print(f"joint: {joint}")
    n_steps = 9
    values = torch.linspace(min_value, max_value, n_steps).unsqueeze(-1).to(joint_params.device)
    part_tfs = joint.get_transform(values)

    src_part_gauss = recon_part_splats[static_recon_part_id]
    tgt_part_gauss = recon_part_splats[dynamic_recon_part_id]

    # put parts back into part frame
    src_part_gauss = apply_mat_tf_to_gauss_params(part_frames[0, static_recon_part_id], src_part_gauss)
    # tgt_part_gauss = apply_mat_tf_to_gauss_params(part_frames[0, dynamic_recon_part_id], tgt_part_gauss)
    
    for i in range(n_steps):
        part_tf = part_tfs[i]

        new_tgt_part_gauss = tgt_part_gauss.copy()
        new_tgt_part_gauss = apply_mat_tf_to_gauss_params(part_tf, new_tgt_part_gauss)
        render_splats = combine_guass_params([src_part_gauss, new_tgt_part_gauss])

        render_image_results = splat_managers[0].render_gauss_params_at_campose(\
                    gt_cam_pose, cam_intrinsic.expand(batch_size, -1, -1), width, height, render_splats, is_semantics=False)
                    
            
        render = render_image_results[0]
        image = render[0].detach().cpu().numpy()
        out_fname = os.path.join(output_dir, f"rendered_image_{i}.png")
        os.makedirs(output_dir, exist_ok=True)
        cv.imwrite(out_fname, image[:,:,::-1] * 255.0)



if __name__=="__main__":
    print("Computing metrics!")

    parser = argparse.ArgumentParser(description="Computes metrics vs. ground truth for pretrained splats, articulations, and poses")

    parser.add_argument('--object_name', 
                        type=str,
                        help='name to use in lookup for GT sapien data',
                        default="")
    
    parser.add_argument('--manager_paths', 
                        type=str,
                        help='pths of the original scenes',
                        default="")

    parser.add_argument('--part_splats', 
                        type=str,
                        help='pth of the learned canonical parts',
                        default="")
    
    parser.add_argument('--pose_estimates',
                        type=str,
                        help="pth of the learned part poses",
                        default="")
    
    parser.add_argument('--articulation_estimates',
                        type=str,
                        help="pickle of the learned articulations",
                        default="")
    
    parser.add_argument('--static_part_id',
                        type=int,
                        help="id of the static part",
                        default=2)
    
    parser.add_argument('--dyn_part_id',
                        type=int,
                        help="id of the dynamic part",
                        default=3)
    
    parser.add_argument('--output_dir',
                        type=str,
                        help="output directory for metrics",
                        default="")
    
    parser.add_argument('--splat_model_datasets',
                        type=str,
                        help="list of datasets used to train the splats",
                        default="")
    
    args = parser.parse_args()
    manager_paths = args.manager_paths.split(",")
    dataset_paths = args.splat_model_datasets.split(",")
    render_images(args.object_name, manager_paths, dataset_paths, args.part_splats, args.pose_estimates, args.articulation_estimates, args.output_dir, static_recon_part_id=args.static_part_id, dynamic_recon_part_id=args.dyn_part_id, )
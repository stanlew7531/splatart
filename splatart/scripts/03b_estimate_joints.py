from functools import *
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from typing import Optional
import argparse
import cv2
import pickle

import torch

from splatart.gui.CloudVis import CloudVisApp

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

def draw_axis_in_image(image, intrinsic, joint:SingleAxisJoint):
    joint_origin = SE3Exp(joint.get_gt_paramaters()["pre_tf"])
    joint_parameters = joint.get_gt_paramaters()["joint_params"]
    joint_axis = joint.joint_axis
    print(f"origin: {joint_origin}, axis: {joint_axis}, params: {joint_parameters}")

    # return draw_pose_in_image(image, joint_origin, intrinsic)

# want to do a breadth first search to explore the configuration vector
def tree_explore_config_vector(configuration_vector, root_part_idx, visited=[], to_return=[]):
    visited.append(root_part_idx)
    frontier = []
    for entry in configuration_vector:
        if entry["src_part"] == root_part_idx and entry["tgt_part"] not in visited:
            print(f"adding from: {entry['src_part']} to {entry['tgt_part']}")
            to_return.append(entry)
            frontier.append(entry["tgt_part"])
            visited.append(entry["tgt_part"])
    for tgt_entry in frontier:
        tree_explore_config_vector(configuration_vector, tgt_entry, visited, to_return)


def process_joints(input_model_dirs:list[str],\
                config_yml:str,\
                dataparser_tf:str,
                num_classes:int,\
                dataset_dir:str,\
                output_dir:str,\
                ns_base_path:str = "/home/stanlew/src/nerfstudio_splatart/",
                parts_to_ignore = [0,2],
                err_thresh = 0.1):
    
    splat_managers = load_managers(input_model_dirs, output_dir, "splat_manager_registered")

    part_idxs = list(range(num_classes))
    for part_idx in parts_to_ignore:
        part_idxs.remove(part_idx)

    # get a dataset so that we can render at train poses
    dataset = SplatTrainDataset(dataset_dir)
    _, inspection_data = dataset[0] #30]
    gt_cam_pose = inspection_data["transform_matrix"]
    # convert the gt cam pose from OpenGL to OpenCV format
    gt_cam_pose_ocv = convert_cam_opengl_opencv(gt_cam_pose)
    gt_img = inspection_data["rgb"]
    gt_semantics = inspection_data["semantics"]
    cam_intrinsic = dataset.get_intrinsic()

    # print(f"inspection data: {inspection_data}")

    pose_matrix = []
    means_for_chamfer = []
    with torch.no_grad():
        for part_idx in part_idxs:
            print(f"Processing part {part_idx}...")
            # loop through each splat manager
            splat_idx = 0
            part_poses = []
            means = splat_managers[0].parts_gauss_params[part_idx]['means']
            means_for_chamfer.append(means)

            for cur_splat_manager in splat_managers:
                # render the current part with the current splat manager, and get our estimated trasform
                part_img, _, _ = cur_splat_manager.render_parts_at_campose(gt_cam_pose, cam_intrinsic, dataset.width, dataset.height, [part_idx], apply_transforms=False)
                part_img_tst, _, _ = cur_splat_manager.render_parts_at_campose(gt_cam_pose, cam_intrinsic, dataset.width, dataset.height, [part_idx], apply_transforms=True)
                
                # read the dataparser tf file
                
                # get the estimated part pose
                part_rot, part_trans = cur_splat_manager.get_part_tf_gauss_params(part_idx)
                part_tf = torch.eye(4)
                # tmp_tf = torch.eye(4)
                part_tf[:3, :3] = part_rot
                # part_tf[:3, :3] = part_rot
                part_tf[:3, 3] = part_trans

                # print(f"tmp tf: {tmp_tf}, \npart tf: {part_tf}\n\n{part_tf @ tmp_tf} \n\n {tmp_tf @ part_tf}")
                # part_tf = part_tf @ tmp_tf
                # need to put the part_tf into the camera frame
                cam_pose_ocv = torch.inverse(convert_cam_opengl_opencv(part_tf @ gt_cam_pose))
                # print(f"cam pose ocv: {cam_pose_ocv}")
                part_poses.append(cam_pose_ocv)
                # part_tf = torch.inverse(gt_cam_pose_ocv @ part_tf)
                # print(f"got part tf: {part_tf}")
                # render the pose onto the image
                image_with_pose = draw_pose_in_image(part_img.detach().cpu().numpy()[0], cam_pose_ocv.detach().numpy(), cam_intrinsic.detach().numpy())
                image_with_pose_tst = draw_pose_in_image(part_img_tst.detach().cpu().numpy()[0], cam_pose_ocv.detach().numpy(), cam_intrinsic.detach().numpy())
                cv2.imwrite(f"pose_part_{part_idx}_{splat_idx}.png", (image_with_pose * 255.0) [:,:, ::-1])
                cv2.imwrite(f"pose_part_{part_idx}_{splat_idx}_test.png", (image_with_pose_tst * 255.0) [:,:, ::-1])
                splat_idx += 1
                # raise Exception("asdf")
                # print(f"part rot: {part_rot}, part trans: {part_trans}")
            part_poses = torch.stack(part_poses)
            pose_matrix.append(part_poses)
        pose_matrix = torch.stack(pose_matrix)

    configuration_vector = []
    # match between each part pose and get an initial rotation and prismatic join estimate
    for part_idx_src in range(pose_matrix.shape[0]):
        for part_idx_tgt in range(pose_matrix.shape[0]):
            if part_idx_src == part_idx_tgt: # dont match itself
                continue

            src_poses_ocv = pose_matrix[part_idx_src] # N, 4, 4
            tgt_poses_ocv = pose_matrix[part_idx_tgt] # N, 4, 4

            base_means = means_for_chamfer[part_idx_tgt].to(src_poses_ocv.device) # N, 3
            # random_points = torch.randn(base_means.shape[0], 3).to(base_means.device)
            # base_means = random_points / torch.linalg.norm(random_points, dim=-1, keepdim=True) * 0.1 # N, 3 # sphere of radius 0.1

            base_means_homogenized = torch.cat([base_means, torch.ones((base_means.shape[0], 1), device=base_means.device)], dim=1).unsqueeze(-1) # N, 4, 1

            # get the actual pose tfs from src/tgt poses
            pose_tfs = torch.inverse(src_poses_ocv) @ tgt_poses_ocv# get the transform between the parts # N, 4, 4

            base_means_for_compare = pose_tfs.unsqueeze(1) @ base_means_homogenized # N, num_means, 4, 1
            # print(f"base means for compare: {base_means_for_compare.shape}")

            # points = pose_tfs[:, 0:3, 3] # need just points for the initial estimates
            prismatic_estimate = PrismaticJoint(len(splat_managers))
            prismatic_estimate.get_initial_estimate(pose_tfs)
            prismatic_tf_estimates = prismatic_estimate.get_transform(prismatic_estimate.joint_params) # N, 4, 4
            prismatic_means_estimates = prismatic_tf_estimates.unsqueeze(1) @ base_means_homogenized # N, num_means, 4, 1

            revolute_estimate = RevoluteJoint(len(splat_managers))
            revolute_estimate.get_initial_estimate(pose_tfs)
            revolute_tf_estimates = revolute_estimate.get_transform(revolute_estimate.joint_params) # N, 4, 4
            revolute_means_estimates = revolute_tf_estimates.unsqueeze(1) @ base_means_homogenized # N, num_means, 4, 1

            # print(f"prismatic means estimates: {prismatic_means_estimates.shape}, revolute means estimates: {revolute_means_estimates.shape}")

            base_means_for_compare = base_means_for_compare.squeeze()[:,:,:3] # strip the homogenous coordinate
            prismatic_means_estimates = prismatic_means_estimates.squeeze()[:,:,:3] # strip the homogenous coordinate
            revolute_means_estimates = revolute_means_estimates.squeeze()[:,:,:3] # strip the homogenous coordinate

            prismatic_err = torch.linalg.norm(prismatic_means_estimates - base_means_for_compare, dim=-1).mean()
            revolute_errr = torch.linalg.norm(revolute_means_estimates - base_means_for_compare, dim=-1).mean() 
            print(f"prismatic error: {prismatic_err}, revolute error: {revolute_errr} with src: {part_idx_src}, tgt: {part_idx_tgt}")
            to_ret_dict = {"src_part": part_idx_src, "tgt_part": part_idx_tgt, "predicted_prismatic": prismatic_estimate, "predicted_revolute": revolute_estimate}
            print(f"prismatic components: {prismatic_estimate.get_gt_parameters()}, revolute components: {revolute_estimate.get_gt_parameters()}")
            if prismatic_err < err_thresh and prismatic_err < revolute_errr:
                to_ret_dict["predicted_joint"] = prismatic_estimate
                configuration_vector.append(to_ret_dict)
                print(f"predicted joint params: {prismatic_estimate.joint_params}")
            elif revolute_errr < err_thresh and revolute_errr < prismatic_err:
                to_ret_dict["predicted_joint"] = revolute_estimate
                configuration_vector.append(to_ret_dict)
                print(f"predicted joint params: {revolute_estimate.joint_params}")
            # print(f"prismatic error: {prismatic_err}, revolute error: {revolute_errr}")

            # configuration_vector.append({"src_part": part_idx_src, "tgt_part": part_idx_tgt, "predicted_prismatic": prismatic_estimate, "predicted_revolute": revolute_estimate})

    treeified_joints = []
    tree_explore_config_vector(configuration_vector, 0, [], treeified_joints)

    # pickle the joint matrix for later processing
    output_pickle_path = f"{output_dir}/configuration_vector.pkl"
    with open(output_pickle_path, "wb") as f:
        pickle.dump(treeified_joints, f)

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

    process_joints(input_model_dirs, config_yml, dataparser_tf, num_classes, dataset_dir, output_dir)
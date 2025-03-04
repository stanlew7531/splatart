from splatart.networks.PoseEstimator import PoseEstimator
from splatart.managers.SplatManager import SplatManagerSingle
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def ADD_metric(splatManager:SplatManagerSingle, poseEstimator:PoseEstimator, gaussParams:list, transforms:dict, scene_id:int):
    """
    Computes the ADD metric for a given prediction and ground truth.
    """

    # dataparser and means
    dp_scale, dp_tf_matrix = splatManager.dataparser_scale, splatManager.dataparser_tf_matrix.cpu()
    gauss_means = [gaussParams[scene_id][i]['means'].detach().cpu() for i in range(poseEstimator.num_parts)]

    # estimated transforms
    trans_estim = poseEstimator.part_means[scene_id, :, :].detach().cpu()
    rot_estim = poseEstimator.part_eulers[scene_id, :, :].detach().cpu()

    # ground truth transforms
    gt_transforms = transforms['gt_part_world_poses'][str(scene_id)]

    add_values = {}
    for i in range(poseEstimator.num_parts):
        part_name = list(gt_transforms.keys())[i]

        gt_transform = np.array(gt_transforms[part_name])
        gt_transform = torch.from_numpy(gt_transform).float()

        # convert translation and euler angles to 4x4 rotation matrix
        pred_transform = torch.eye(4)
        pred_transform[:3, :3] = torch.from_numpy(R.from_euler('xyz', rot_estim[i]).as_matrix())
        pred_transform[3, :3] = trans_estim[i]
        pred_transform = pred_transform.float()

        # invert dataparser transform
        pred_transform = torch.linalg.inv(dp_tf_matrix) @ pred_transform
        pred_transform = pred_transform / dp_scale

        # apply to gaussian means
        pred_gauss_mean = gauss_means[i]
        pred_gauss_mean = torch.cat((pred_gauss_mean, torch.ones(pred_gauss_mean.shape[0], 1)), dim=1)
        pred_gauss_mean = pred_transform @ pred_gauss_mean.T
        pred_gauss_mean = pred_gauss_mean.T[:, :3]

        gt_gauss_mean = torch.from_numpy(np.array(gt_transform[:3, 3])).float()
        gt_gauss_mean = gt_gauss_mean.unsqueeze(0)
        gt_gauss_mean = torch.cat((gt_gauss_mean, torch.ones(gt_gauss_mean.shape[0], 1)), dim=1)
        gt_gauss_mean = torch.linalg.inv(gt_transform) @ gt_gauss_mean.T
        gt_gauss_mean = gt_gauss_mean.T[:, :3]

        # compute ADD metric
        add = torch.norm(pred_gauss_mean - gt_gauss_mean, dim=1).mean()
        add_values[part_name] = add.item()

    return add_values

manager_path = '/home/vishalchandra/Desktop/splatart/splatart/arm_accuracy_pths/seg_learned_manager_0.pth'
pose_estimator_path = '/home/vishalchandra/Desktop/splatart/splatart/arm_accuracy_pths/pose_estimator.pth'
gauss_param_path = '/home/vishalchandra/Desktop/splatart/splatart/arm_accuracy_pths/part_gauss_params.pth'
transforms_path = '/home/vishalchandra/Desktop/splatart/splatart/arm_accuracy_pths/transforms.json'

splatManager = torch.load(manager_path)
poseEstimator = torch.load(pose_estimator_path)
gaussParams = torch.load(gauss_param_path)
transforms = json.load(open(transforms_path, 'r'))

add_values = ADD_metric(splatManager, poseEstimator, gaussParams, transforms, 0)
print(add_values)
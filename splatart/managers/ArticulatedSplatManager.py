from functools import *
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from typing import Optional
import argparse
import cv2
import pickle
import pytorch3d.transforms as p3dt
import torch

from splatart.gui.CloudVis import CloudVisApp

from splatart.utils.lie_utils import SE3, SE3Exp, SO3Exp
from splatart.managers.JointManager import JointManager
from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint, SingleAxisJoint
from splatart.managers.SplatManager import SplatManager, load_managers_nerfstudio, load_managers, load_model
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.utils.helpers import convert_cam_opengl_opencv


class ArticulatedSplatManager(torch.nn.Module):
    def __init__(self, input_splat_managers: list[str], configuration_vector_pkl: str, root_idx = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if(len(input_splat_managers) == 1):
            self.splat_manager = torch.load(input_splat_managers[0])
            self.splat_managers_cache = [self.splat_manager]
        else:
            self.splat_managers_cache = load_managers(input_splat_managers)
            self.splat_manager = self.splat_managers_cache[0]
            self.splat_manager.num_parts = len(self.splat_managers_cache)
            for i in range(1, self.splat_manager.num_parts):
                self.splat_manager.parts_gauss_params[i] = self.splat_managers_cache[i].object_gaussian_params

        with open(configuration_vector_pkl, "rb") as f:
            configuration_vector = pickle.load(f)
        self.configuration_vector = configuration_vector

        self.root_idx = root_idx

    def update_part_poses_with_configuration(self, configuration_params, joint_config_values, part_idx = 0, part_poses = None, visited_parts = []):
        print(f"visited parts a: {visited_parts}")
        part_poses = torch.eye(4).unsqueeze(0).repeat(len(self.splat_manager.parts_gauss_params),1,1) if part_poses is None else part_poses

        # get the configuration entries which have a src part matching the current part idx
        # for those that dont match, hold onto them for the next recursion call
        new_config_entries = []
        new_config_values = []
        joint_entries = []
        joint_values = []
        for idx in range(len(configuration_params)):
            print(f"checking {configuration_params[idx]}")
            if configuration_params[idx]["src_part"] == part_idx:
                joint_entries.append(configuration_params[idx])
                joint_values.append(joint_config_values[idx])
            else:
                new_config_entries.append(configuration_params[idx])
                new_config_values.append(joint_config_values[idx])

        # update the part poses based on the joint entries
        # since we treeified the configuration entries earlier, we can naively go through the list
        for idx in range(len(joint_entries)):
            joint_entry = joint_entries[idx]
            joint_value = joint_values[idx]
            print(f"joint_entry: {joint_entry}, joint_value: {joint_value}")
            src_part_idx = joint_entry["src_part"]
            tgt_part_idx = joint_entry["tgt_part"]
            print(f"visited partsd: {visited_parts}")
            #if(tgt_part_idx not in visited_parts):
            visited_parts.append(tgt_part_idx)
            pred_rev = joint_entry["predicted_revolute"]
            pred_prism = joint_entry["predicted_prismatic"]
            pred_joint = joint_entry["predicted_joint"]
            predicted_jnt_tf = pred_joint.get_transform(joint_value)
            print(f"predicted_tf: {predicted_jnt_tf}")
            part_poses[tgt_part_idx] = part_poses[src_part_idx] @ predicted_jnt_tf

            # recurse for the next part
            #part_poses = self.update_part_poses_with_configuration(new_config_entries, new_config_values, tgt_part_idx, part_poses, visited_parts)

        return part_poses
        

    def render_parts_at_campose_with_configuration(self, cam_pose, cam_intrinsic, width, height, part_idxs, configuration_values, base_part_idx = 2):
        part_poses = self.update_part_poses_with_configuration(self.configuration_vector, configuration_values, part_idx = base_part_idx, part_poses=None, visited_parts=[])
        print("for config values: ", configuration_values)
        print(part_poses)
        part_poses = torch.inverse(part_poses)
        # turn the part poses into tf_quats and tf_trans
        tf_trans = part_poses[:,:3,3]
        tf_rots = part_poses[:,:3,:3]
        tf_quats = p3dt.matrix_to_quaternion(tf_rots)
        print(tf_trans)
        print(tf_quats)

        # print(f"num_parts: {self.splat_manager.num_parts}, part keys: {self.splat_manager.parts_gauss_params.keys()}")

        # get the parts render gauss params
        # render_gauss_params = [self.splat_managers_cache[i].object_gaussian_params for i in range(len(self.splat_managers_cache))]
        # combined_params = self.splat_managers_cache[0].combine_gauss_params_prerender(render_gauss_params, apply_transforms=True)

        for idx in part_idxs:
            self.splat_manager.parts_gauss_params[idx]["tf_quats"] = tf_quats[idx]
            self.splat_manager.parts_gauss_params[idx]["tf_trans"] = tf_trans[idx]

        # render the object
        return self.splat_manager.render_parts_at_campose(cam_pose, cam_intrinsic, width, height, part_idxs)


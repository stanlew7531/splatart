import json
from pathlib import Path
import os
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional
import cv2 as cv

class UrdfSamplerManager(Model):

    def __init__(self, dataset_dir: str, generating_urdf: str):
        self.dataset_dir = dataset_dir
        self.urdf_file = generating_urdf

        joint_states_file = os.path.join(self.dataset_dir, "joint_states.json")
        part_poses_file = os.path.join(self.dataset_dir, "part_poses.json")
        transforms_file = os.path.join(self.dataset_dir, "transforms.json")

        self.transforms_json = json.load(open(transforms_file, 'r'))
        self.joint_states_json = json.load(open(joint_states_file, 'r'))
        self.part_poses_json = json.load(open(part_poses_file, 'r'))

    def get_parts_list(self, obj_id:str):
        return self.part_poses_json[obj_id].keys()
    
    def get_transforms_json_meta(self):
        # get the list of keys from the transforms json and remove frames
        meta = self.transforms_json.copy()
        meta.pop("frames")
        return meta

    def masks_for_parts(self, base_rgb, base_seg, part_segs, part_id_list, bg_color = [255, 255, 255, 0]):
        # make a mask which contains only the invisible areas of the part
        # we will mask out everything on the base image except for the part
        # but we want to make sure we don't train on the invisible sections
        # in other words, we train on visible sections of the part, and mask out the background
        invisible_part_mask = np.zeros_like(base_seg)

        for part_seg_i in part_segs:
            invisible_part_mask[part_seg_i > 0] = 255

        for part_id in part_id_list:
            invisible_part_mask[base_seg == int(part_id)] = 0

        visible_part_mask = np.zeros_like(base_seg)
        for part_id in part_id_list:
            visible_part_mask[base_seg == int(part_id)] = 255

        training_image = base_rgb.copy() # copy the base image
        # if the training image doesnt have an alpha channel, add it
        
        if training_image.shape[2] == 3:
            training_image = cv.cvtColor(training_image, cv.COLOR_BGR2BGRA)
        
        # the alpha channel is important because small objects in relation
        # to the background cause nerf training to converge to the background color everywhere
        # if its alpha though, the randomized background color will prevent this
        train_img_mask = np.zeros_like(base_seg)
        for part_seg_i in part_segs:
            train_img_mask[part_seg_i > 0] = 1
        training_image[train_img_mask == 0] = bg_color # mask out the background

        # we also want to impose a ratio on trainable background vs part pixels
        # to alleviate this problem
        expansion_ratio = 3.5
        # get the bounding box for the part
        part_bbox = cv.boundingRect(train_img_mask)
        # expand the bounding box by the ratio
        part_bbox = (part_bbox[0] - int(part_bbox[2] * expansion_ratio/2), part_bbox[1] - int(part_bbox[3] * expansion_ratio/2), int(part_bbox[2] * expansion_ratio), int(part_bbox[3] * expansion_ratio))

        training_segmentation = np.zeros_like(base_seg) # start with no trainable pixels
        # draw the part_bbox on the training segmentation
        cv.rectangle(training_segmentation, (part_bbox[0], part_bbox[1]), (part_bbox[0] + part_bbox[2], part_bbox[1] + part_bbox[3]), 255, -1)
        
        
        # remove the invisible part pixels
        training_segmentation = training_segmentation - invisible_part_mask

        return training_image, training_segmentation
        
    def create_dataset_for_part(self, obj_id:str, part_id_list_str:str, bg_color = [255, 255, 255, 0]):
        # get the set of global transforms
        frames = self.transforms_json["frames"]

        part_id_list = part_id_list_str.split(",")


        # iterate through each transform to produce the part's image and transform matrix
        for frame in frames:
            # if(int(frame["pose_idx"]) != 1):
            #     print(f"Skipping frame with pose_idx {frame['pose_idx']}")
            #     continue
            base_rgb_fname = frame["file_path"]
            base_seg_fname = frame["labels_path"]
            part_seg_fnames = [\
                base_seg_fname.replace("segmentations", "segmentations/parts")\
                    .replace(".", f"_part_{part_id_list[i]}.")\
                    for i in range(len(part_id_list))]

            base_rgb_fname = os.path.join(self.dataset_dir, base_rgb_fname) 
            base_seg_fname = os.path.join(self.dataset_dir, base_seg_fname) 
            part_seg_fnames = [os.path.join(self.dataset_dir, part_seg_fname) for part_seg_fname in part_seg_fnames]

            # load the original rgb image
            base_rgb = cv.imread(base_rgb_fname)
            # load the segmentations
            base_seg = cv.imread(base_seg_fname, cv.IMREAD_UNCHANGED) # only the visible parts
            part_segs = [cv.imread(part_seg_fname, cv.IMREAD_UNCHANGED) for part_seg_fname in part_seg_fnames] # full part segmentation ignoring occlusions

            training_image, training_segmentation = self.masks_for_parts(base_rgb, base_seg, part_segs, part_id_list, bg_color)

            # get the global transform for this frame
            transform = frame["transform_matrix"]

            # get the part's pose in global frame
            scene_key = str(frame["pose_idx"])
            part_pose_data = self.part_poses_json[obj_id][part_id_list[0]][scene_key]
            part_pos = part_pose_data["pos"] # x,y,z transformation
            part_rot = part_pose_data["orn"] # quaternion in [x,y,z,w] format

            # convert the quaternion to a rotation matrix
            part_rot = R.from_quat(part_rot).as_matrix()
            # create the full SE3 matrix
            part_transform = np.eye(4)
            part_transform[:3,:3] = part_rot
            part_transform[:3,3] = part_pos

            # we want the camera pose in relation to the part transform, instead of the global transform
            # so we invert the part transform and multiply it by the global transform
            # print(part_transform)
            camera_transform = np.linalg.inv(part_transform) @ np.array(transform)

            # yield out the training image, segmentation, and camera transform
            yield training_image, training_segmentation, camera_transform


if __name__ == "__main__":
    test_dataset_dir = "/home/stanlew/src/URDFSampler/outputs_mycobot"
    urdf_file = "/home/stanlew/src/cobot_ws/src/mycobot_ros/mycobot_description/urdf/280jn/mycobot_urdf_gripper_cam.urdf"

    urdf_sampler_manager = UrdfSamplerManager(test_dataset_dir, urdf_file)

    for train_img, train_seg, cam_tf in urdf_sampler_manager.create_dataset_for_part("0", "0"):
        print(cam_tf)
        cv.imshow("train_img", train_img)
        cv.imshow("train_seg", train_seg)
        cv.waitKey(0)
import torch
import numpy as np

def convert_cam_opengl_opencv(camera_pose):
    # check if pose is a numpy array - if so, convert to torch then convert back
    ret_numpy = False
    if isinstance(camera_pose, np.ndarray):
        camera_pose = torch.tensor(camera_pose).float()
        ret_numpy = True
    # create a 180 degree rotation about the x axis
    x_rot = torch.eye(4).to(camera_pose.device)
    x_rot[1, 1] = -1
    x_rot[2, 2] = -1
    # the gl pose is the apriltag pose with the x rotation applied
    gl_pose = camera_pose @ torch.matmul(x_rot, torch.eye(4, device = camera_pose.device))    
    return gl_pose.numpy() if ret_numpy else gl_pose
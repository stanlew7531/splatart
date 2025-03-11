import torch
import numpy as np
from plyfile import PlyElement, PlyData

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

# adapted from https://github.com/nerfstudio-project/gsplat/issues/234#issuecomment-2197277211
def gauss_params_to_ply(gauss_params, output_fname):

    xyz = gauss_params["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gauss_params["features_dc"].detach().contiguous().cpu().numpy()
    f_rest = gauss_params["features_rest"].transpose(1, 2).flatten(start_dim=1).detach().contiguous().cpu().numpy()
    opacities = gauss_params["opacities"].detach().cpu().numpy()
    scale = gauss_params["scales"].detach().cpu().numpy()
    rotation = gauss_params["quats"].detach().cpu().numpy()


    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(gauss_params["features_dc"].shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(gauss_params["features_rest"].shape[1]*gauss_params["features_rest"].shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gauss_params["scales"].shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gauss_params["quats"].shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    if(not output_fname.endswith('.ply')):
        output_fname += '.ply'
    PlyData([el]).write(output_fname)

def means_to_ply(means, output_fname):

    xyz = means[:, :3].detach().cpu().numpy() # strip homogenous point if it exists
    normals = np.zeros_like(xyz)

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    if(not output_fname.endswith('.ply')):
        output_fname += '.ply'
    PlyData([el]).write(output_fname)

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
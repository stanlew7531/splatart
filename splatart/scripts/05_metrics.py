import numpy as np
import open3d as o3d
import json
import torch
import torch.nn.functional as F
from plyfile import PlyElement, PlyData
import numpy as np
from pytorch3d.loss import chamfer_distance
import pickle as pkl
import pytorch3d.transforms as p3dt
from yourdfpy import URDF
from splatart.networks.NarfJoint import PrismaticJoint

# PARAMS
transforms_path = "/home/vishalchandra/Desktop/nerfstudio_dev/outputs/exp_sapien_blade/semantic-splatfacto/config_0/dataparser_transforms.json"
splatcloud_path = "/home/vishalchandra/Desktop/nerfstudio_dev/exports/splat/splat.ply"
sapien_root = '/home/vishalchandra/Desktop/sapien_dataset/'
manager_paths = [
    '/home/vishalchandra/Desktop/splatart/results/sapien_exp/blade/splat_manager_0.pth',
    '/home/vishalchandra/Desktop/splatart/results/sapien_exp/blade/splat_manager_1.pth'
]
splatcloud_path = "/home/vishalchandra/Desktop/nerfstudio_dev/exports/splat/splat.ply"
gtmesh_path = "/home/vishalchandra/Desktop/splatart_data/narf_sapien_data/v5/blade/0/blade_0.ply"
config_vector_path = '/home/vishalchandra/Desktop/splatart/results/sapien_exp/blade/configuration_vector.pkl'
narf_data_root = "/home/vishalchandra/Desktop/splatart_data/narf_sapien_data/v5/blade"
movable = 1
# Part correspondences between recon and canonical
# NEEDS TO BE DEFINE PER EVAL EXAMPLE
canonical_corresp = {
    0: 1,
    1: 0
}



# read point cloud
splatcloud = o3d.io.read_point_cloud(splatcloud_path)

# Get and invert transforms
transforms = json.load(open(transforms_path, 'r'))
inv_scale = 1.0 / transforms['scale']

linear = np.array(transforms['transform'])[0:3, 0:3]
T = np.array(transforms['transform'])[0:3, 3]
inv = np.linalg.inv(linear)

inv_hom = np.eye(4)
inv_hom[0:3, 0:3] = inv
inv_hom[0:3, 3] = -inv @ T


# Get canonical parts from obj files
# Read .ply file
paris_to_sapien = {
    'blade': '103706',
    'laptop': '10211',
    'foldchair': '102255',
    'oven': '101917',
    'fridge': '10905',
    'scissor': '11100',
    'stapler': '103111',
    'USB': '100109',
    'washer': '103776',
    'storage': '45135'
}

name = 'blade'
sapien_dir = sapien_root + paris_to_sapien[name] + '/'
description = json.load(open(sapien_dir + 'result.json'))[0]
parts = description['children']

parts_files = [part['objs'] for part in parts]

canonical_part_objs = [
    [o3d.io.read_triangle_mesh(sapien_dir + 'textured_objs/' + obj + '.obj') for obj in part]
    for part in parts_files
]

# add all triangle meshes to a single mesh but keep them separate
canonical_parts = []
for part in canonical_part_objs:
    mesh = o3d.geometry.TriangleMesh()
    for obj in part:
        mesh += obj
    canonical_parts.append(mesh)

# correct for convention differences between sapien and reconstruction
R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, -np.pi/2, 0))
for part in canonical_parts:
    part.rotate(R, center=(0, 0, 0))


# Extract Reconstructed Parts from Gauss Params
managers = [torch.load(manager_path) for manager_path in manager_paths]

# adapted from https://github.com/nerfstudio-project/gsplat/issues/234#issuecomment-2197277211
def part_to_ply(part, part_num):

    xyz = part["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = part["features_dc"].detach().contiguous().cpu().numpy()
    f_rest = part["features_rest"].transpose(1, 2).flatten(start_dim=1).detach().contiguous().cpu().numpy()
    opacities = part["opacities"].detach().cpu().numpy()
    scale = part["scales"].detach().cpu().numpy()
    rotation = part["quats"].detach().cpu().numpy()


    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(part.features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(part.features_rest.shape[1]*part.features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(part.scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(part.quats.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write('part{}.ply'.format(part_num))

for i in range(2, managers[0].num_parts):
    part = managers[0].parts_gauss_params[i]
    part_to_ply(part, i-2)


recon_part_clouds = [
    o3d.io.read_point_cloud('part{}.ply'.format(i)) for i in range(2)
]

# apply transforms
for part_cloud in recon_part_clouds:
    part_cloud.transform(inv_hom)
    part_cloud.scale(inv_scale, center=(0, 0, 0))

canonical_part_clouds = [
    canonical_parts[canonical_corresp[i]].sample_points_uniformly(len(recon_part_clouds[i].points))
    for i in range(2)
]


# register moving parts
# here, we know it's part 1
reg_p2p = o3d.pipelines.registration.registration_icp(
    recon_part_clouds[movable], canonical_part_clouds[movable], 0.02, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

part_chamfers = []
for i in range(2):

    recon_cloud = recon_part_clouds[i]
    canonical_cloud = canonical_part_clouds[i]

    if i == movable:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            recon_part_clouds[movable], canonical_part_clouds[movable], 0.02, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        recon_cloud.transform(reg_p2p.transformation)
        
    recon_T = torch.tensor(np.array(recon_cloud.points)).float().unsqueeze(0)
    canonical_T = torch.tensor(np.array(canonical_cloud.points)).float().unsqueeze(0)

    a, _ = chamfer_distance(recon_T, canonical_T)
    b, _ = chamfer_distance(canonical_T, recon_T)
    part_chamfers.append((a + b)/2)

# whole object chamfer
splatcloud = o3d.io.read_point_cloud(splatcloud_path)
splatcloud.transform(inv_hom)
splatcloud.scale(inv_scale, center=(0, 0, 0))
R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
splatcloud.rotate(R, center=(0, 0, 0))

gtmesh = o3d.io.read_triangle_mesh(gtmesh_path)
gtcloud = gtmesh.sample_points_uniformly(number_of_points=len(splatcloud.points))

splat_T = torch.tensor(np.array(splatcloud.points)).float().unsqueeze(0)
gt_T = torch.tensor(np.array(gtcloud.points)).float().unsqueeze(0)

a, _ = chamfer_distance(splat_T, gt_T)
b, _ = chamfer_distance(gt_T, splat_T)
chamfer_whole = (a + b) / 2

# get joint for joint metrics.
with open(config_vector_path, 'rb') as f:
    config_vector = pkl.load(f)

joint = config_vector[0]['predicted_joint']
joint_params = joint.get_gt_parameters()


# get pre transform matrix
pre_tf = joint_params['pre_tf'].detach()
initial_tf_rot = p3dt.euler_angles_to_matrix(pre_tf[:3], "XYZ") # (4, 4)
initial_tf = torch.eye(4)
initial_tf[:3, :3] = initial_tf_rot[:3, :3]
initial_tf[:3, 3] = pre_tf[3:6]

joint_axis_hom = torch.ones(4, 1)
joint_axis_hom[:3, 0] = joint_params['joint_axis'].detach()
pred_axis = initial_tf @ joint_axis_hom


## so far, there have been 3 spaces (2 transforms) that we have worked with
## (1) sapien space where we got gt part meshes, (2) splat space where we got splatcloud, (3) ground truth space where we got gtcloud
## (1) --> (2) was XYZ = (pi/2, -pi/2, 0), (2) --> (3) was inv_hom, inv_scale, XYZ = (0, 0, pi/2)
## axis is in (2) space, gt axis is in (3) space


# inv_hom, inv_scale doesn't matter here
pred_axis = torch.tensor(inv_hom, dtype=torch.float32) @ pred_axis
pred_axis = (pred_axis / pred_axis[3, 0])[:3, 0]
pred_axis = pred_axis / np.linalg.norm(pred_axis)
pred_axis_o = pre_tf[3:].detach()

# now, get gt axis and apply gt base link transform to it.
robot = URDF.load(sapien_dir + 'mobility.urdf')

base_joint, joint = None, None
for j in robot.joint_map.values():
    if j.type == 'fixed':
        base_joint = j
    else:
        joint = j

gt_pre_tf = base_joint.origin
gt_axis = joint.axis

gt_axis = torch.tensor(gt_pre_tf)[:3, :3] @ torch.tensor(gt_axis)
gt_axis_o = torch.tensor(gt_pre_tf[0:3, 3] + joint.origin[0:3, 3])

# axis comparison: angle arror and pos error
# angular difference between two vectors
pred_axis = pred_axis.double()
pred_axis_o = pred_axis_o.double()

cos_theta = torch.dot(pred_axis, gt_axis) / (torch.norm(pred_axis) * torch.norm(gt_axis))
ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))

# positonal difference between two axis lines
w = gt_axis_o - pred_axis_o
cross = torch.cross(pred_axis, gt_axis)
if (cross == torch.zeros(3)).sum().item() == 3:
    pos_err = torch.tensor(0)
else:
    pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)


# part motion comparison
pred_motion = joint_params['joint_params'][1, 0].item()

gt_param_0 = list(json.load(open(narf_data_root + '/0/transforms.json'))['configurations'].values())[0]
gt_param_1 = list(json.load(open(narf_data_root + '/1/transforms.json'))['configurations'].values())[0]
gt_motion = [final - init for init,final in zip(gt_param_0, gt_param_1) if init != final][0]

# either translation or geodesic distance
F.normalize(pred_axis, p=2, dim=0)
F.normalize(gt_axis, p=2, dim=0)

if isinstance(config_vector[0]['predicted_joint'], PrismaticJoint):
    err = torch.sqrt((pred_motion * pred_axis - gt_motion * gt_axis)**2).sum()
else:
    pred_R = p3dt.axis_angle_to_matrix(pred_motion * pred_axis)
    gt_R = p3dt.axis_angle_to_matrix(gt_motion * gt_axis)

    pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
    R_diff = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R_diff) - 1.0) * 0.5, min=-1., max=1.)
    angle = torch.rad2deg(torch.arccos(cos_angle))

    err = angle

print('CD-m: {}, CD-s: {}, CD-w: {}, ang_err: {}, pos_err: {}, part motion err: {}'.format(
    part_chamfers[movable], part_chamfers[int(not movable)], chamfer_whole, ang_err, pos_err, err)
)
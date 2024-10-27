import numpy as np
import open3d as o3d
import json
from pytorch3d.loss import chamfer_distance
import torch

# Read .ply file
transforms_path = "/home/vishalchandra/Desktop/nerfstudio_dev/outputs/exp_sapien_blade/semantic-splatfacto/config_0/dataparser_transforms.json"
splatcloud_path = "/home/vishalchandra/Desktop/nerfstudio_dev/exports/splat/splat.ply"
gtmesh_path = "/home/vishalchandra/Desktop/splatart_data/narf_sapien_data/v5/blade/0/blade_0.ply"


splatcloud = o3d.io.read_point_cloud(splatcloud_path)
gtmesh = o3d.io.read_triangle_mesh(gtmesh_path)
gtcloud = gtmesh.sample_points_uniformly(number_of_points=len(splatcloud.points))

# Get and invert transforms
transforms = json.load(open(transforms_path, 'r'))
inv_scale = 1.0 / transforms['scale']

linear = np.array(transforms['transform'])[0:3, 0:3]
T = np.array(transforms['transform'])[0:3, 3]
inv = np.linalg.inv(linear)

inv_hom = np.eye(4)
inv_hom[0:3, 0:3] = inv
inv_hom[0:3, 3] = -inv @ T

# apply transforms
splatcloud.transform(inv_hom)
splatcloud.scale(inv_scale, center=(0, 0, 0))

# correct for convention difference
R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
splatcloud.rotate(R, center=(0, 0, 0))


# convert to torch tensors
splat_T = torch.tensor(np.array(splatcloud.points)).float().unsqueeze(0)
gt_T = torch.tensor(np.array(gtcloud.points)).float().unsqueeze(0)

# calculate chamfer distance
a, _ = chamfer_distance(splat_T, gt_T)
b, _ = chamfer_distance(gt_T, splat_T)
print((a + b) / 2)


# show the point clouds
o3d.visualization.draw_geometries([splatcloud, gtcloud])

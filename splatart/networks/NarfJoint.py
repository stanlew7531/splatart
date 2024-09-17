import torch
import numpy as np
import yaml
from typing import Optional
import warnings
import pytorch3d.transforms as p3dt
# from splatart.utils.lie_utils import SE3, SE3Exp, SO3Exp

class NarfJoint(torch.nn.Module):
    def __init__(self, num_scenes:int):
        super().__init__()
        self.num_scenes = num_scenes

    def get_gt_parameters(self):
        raise NotImplementedError

    def get_transform(self, scene_paramters:torch.Tensor):
        raise NotImplementedError
    
    def get_scene_transform(self, scene_idx:torch.Tensor):
        raise NotImplementedError
    
    def set_num_scenes(self, num_scenes:int):
        raise NotImplementedError
    
    def configuration_estimation(self, points:torch.Tensor):
        raise NotImplementedError
    
    def get_initial_estimate(self, points:torch.Tensor):
        raise NotImplementedError
    

    
class SingleAxisJoint(NarfJoint):
    def __init__(self, num_scenes:int):
        super().__init__(num_scenes)
        self.pre_tf = torch.nn.Parameter(torch.Tensor([0,0,0,0,0,0]))
        self.joint_params = torch.nn.Parameter(torch.zeros(num_scenes, 1))
        self.joint_axis = torch.nn.Parameter(torch.Tensor([0,0,1])) # default to z axis

    def set_num_scenes(self, num_scenes: int):
        if(self.num_scenes == num_scenes):
            # same number of scenes so no change
            return
        elif(self.num_scenes < num_scenes):
            # create a new tensor with the new number of scenes, but copy the existing values
            new_joint_params = torch.zeros(num_scenes, 1)
            new_joint_params[:self.num_scenes] = self.joint_params
            self.joint_params = torch.nn.Parameter(new_joint_params)
        else:
            # create a warning that we are reducing the number of scenes
            warnings.warn("Reduction in scene count! This will result in loss of data!")
            # copy only the first num_scenes values, the rest will be lost
            new_joint_params = torch.zeros(num_scenes, 1)
            new_joint_params = self.joint_params[:num_scenes]
            self.joint_params = new_joint_params
            self.num_scenes = num_scenes


# Joint formulations modeled after paper:
# "A Probabilistic Framework for Learning Kinematic Models of Articulated Objects"
# by J. Sturm, C. Stachniss, and W. Burgard (JAIR 2011)

class RevoluteJoint(SingleAxisJoint):
    def __init__(self, num_scenes: int):
        super().__init__(num_scenes)
        self.post_tf = torch.nn.Parameter(torch.Tensor([0,0,0,0,0,0]))

    def get_gt_parameters(self):
        return {"pre_tf": self.pre_tf, "post_tf": self.post_tf, "joint_params": self.joint_params}

    # scene_parameters: torch.Tensor of shape (B, )
    def get_transform(self, scene_paramters: torch.Tensor):
        initial_tf_rot = p3dt.euler_angles_to_matrix(self.pre_tf[:3], "XYZ") # (4, 4)
        initial_tf = torch.eye(4).to(scene_paramters.device)
        initial_tf[:3, :3] = initial_tf_rot[:3, :3]
        initial_tf[:3, 3] = self.pre_tf[3:6]

        # make sure its normalized first - amount of rotation is taken as the magnitude of the vector
        axis_angle = self.joint_axis / torch.norm(self.joint_axis) # (3, )
        axis_angle = axis_angle.repeat(scene_paramters.shape[0], 1)
        axis_angle = axis_angle * scene_paramters # (B, 3)
        rotation_tf = torch.eye(4).to(scene_paramters.device).repeat(scene_paramters.shape[0], 1, 1) # (B, 4, 4)
        rotation_tf[:, :3, :3] = p3dt.axis_angle_to_matrix(axis_angle) # (B, 3, 3)

        final_tf_rot = p3dt.euler_angles_to_matrix(self.post_tf[:3], "XYZ") # (4, 4)
        final_tf = torch.eye(4).to(scene_paramters.device)
        final_tf[:3, :3] = final_tf_rot[:3, :3]
        final_tf[:3, 3] = self.post_tf[3:6]        

        # compose the transforms
        scene_tf = initial_tf @ rotation_tf @ final_tf

        return scene_tf                

    
    # scene_idx: torch.Tensor of shape (B,)
    def get_scene_transform(self, scene_idx:torch.Tensor):
        # get the joint angles form each scene_idx
        joint_angles = self.joint_params[scene_idx]

        return self.get_transform(joint_angles)
    

    def configuration_estimation(self, poses:torch.Tensor):

        self.num_scenes = poses.shape[0]

        optimizer = torch.optim.Adam( [self.pre_tf, self.joint_axis, self.post_tf, self.joint_params], lr=0.001)
        poses_quats = p3dt.matrix_to_quaternion(poses[:, :3, :3])
        poses_trans = poses[:, :3, 3]

        for epoch in range(1000):
            optimizer.zero_grad()
            joint_params = torch.zeros_like(self.joint_params)
            joint_params[1:] = self.joint_params[1:] # keep the rest of the params
            transforms = self.get_transform(joint_params)

            transforms_quats = p3dt.matrix_to_quaternion(transforms[:, :3, :3])
            transforms_trans = transforms[:, :3, 3]
            loss = (5 * torch.mean(torch.norm(poses_quats - transforms_quats, dim=1)) + 
                    torch.mean(torch.norm(poses_trans - transforms_trans, dim=1)))
            loss.backward()
            optimizer.step()
            # print(f"Epoch: {epoch}, pretf: {self.pre_tf}, joint axis: {self.joint_axis}, params: {self.joint_params}, Loss: {loss.item()}")
    
    def get_initial_estimate(self, poses:torch.Tensor):
        self.configuration_estimation(poses)
        

class PrismaticJoint(SingleAxisJoint):
    def __init__(self, num_scenes:int):
        super().__init__(num_scenes)

    def get_gt_parameters(self):
        return {"pre_tf": self.pre_tf, "joint_axis": self.joint_axis, "joint_params": self.joint_params}

    # scene_parameters: torch.Tensor of shape (B,)
    def get_transform(self, scene_paramters: torch.Tensor):
        initial_tf_rot = p3dt.euler_angles_to_matrix(self.pre_tf[:3], "XYZ") # (4, 4)
        initial_tf = torch.eye(4).to(scene_paramters.device)
        initial_tf[:3, :3] = initial_tf_rot[:3, :3]
        initial_tf[:3, 3] = self.pre_tf[3:6]

        # make sure its normalized first - amount of rotation is taken as the magnitude of the vector
        axis_translate = self.joint_axis / torch.norm(self.joint_axis) # (3, )
        axis_translate = axis_translate.repeat(scene_paramters.shape[0], 1)
        axis_translate = axis_translate * scene_paramters # (B, 3)

        translation_tf = torch.eye(4).to(scene_paramters.device).repeat(scene_paramters.shape[0], 1, 1) # (B, 4, 4)
        translation_tf[:, :3, 3] = axis_translate # (B, 3)

        # compose the transforms
        scene_tf = initial_tf @ translation_tf

        return scene_tf

    # scene_idx: torch.Tensor of shape (B,)
    def get_scene_transform(self, scene_idx:torch.Tensor):
        # get the joint displacements using the scene_idx's
        joint_displacements = self.joint_params[scene_idx]

        return self.get_transform(joint_displacements)
    
    def configuration_estimation(self, poses:torch.Tensor):
        self.num_scenes = poses.shape[0]

        optimizer = torch.optim.Adam( [self.pre_tf, self.joint_axis, self.joint_params], lr=0.001)
        poses_quats = p3dt.matrix_to_quaternion(poses[:, :3, :3])
        poses_trans = poses[:, :3, 3]

        for epoch in range(1000):
            optimizer.zero_grad()
            joint_params = torch.zeros_like(self.joint_params)
            joint_params[1:] = self.joint_params[1:] # keep the rest of the params
            transforms = self.get_transform(joint_params)

            transforms_quats = p3dt.matrix_to_quaternion(transforms[:, :3, :3])
            transforms_trans = transforms[:, :3, 3]
            loss = (5 * torch.mean(torch.norm(poses_quats - transforms_quats, dim=1)) + 
                    torch.mean(torch.norm(poses_trans - transforms_trans, dim=1)))
            loss.backward()
            optimizer.step()
            # print(f"Epoch: {epoch}, pretf: {self.pre_tf}, 
    
    def get_initial_estimate(self, poses:torch.Tensor):
        self.configuration_estimation(poses)

if __name__ == "__main__":
    import scipy.spatial.transform.rotation as R
    from splatart.gui.CloudVis import CloudVisApp
    from functools import *

    # initial and final tfs for the revoute and primatic joints
    initial_tf = torch.Tensor([0,0,0,1,0,0])# x translation of 1
    post_tf = torch.Tensor([0,0,0,0,1,0]) # y translation 1

    rev_joint = RevoluteJoint(3)
    rev_joint.pre_tf = initial_tf
    rev_joint.post_tf = post_tf

    prs_joint = PrismaticJoint(3)
    prs_joint.pre_tf = initial_tf
    prs_joint.joint_axis = torch.Tensor([0,1,1])

    # get the revolute tf at angle 0, 90, 180
    n_scenes = 3
    
    rev_joint.joint_params = torch.zeros(n_scenes) #torch.Tensor([0, np.pi/2, np.pi])
    rev_joint.joint_params[0] = 0
    rev_joint.joint_params[1] = np.pi/2
    rev_joint.joint_params[2] = np.pi
    rev_transforms = rev_joint.get_scene_transform(torch.Tensor([0, 1, 2]).to(torch.long))

    prs_joint.joint_params = torch.zeros(n_scenes) #torch.Tensor([0, 1, 2])
    prs_joint.joint_params[0] = 0
    prs_joint.joint_params[1] = 1
    prs_joint.joint_params[2] = 2
    prs_transforms = prs_joint.get_scene_transform(torch.Tensor([0, 1, 2]).to(torch.long))

    n_points = 1000

    # create a random point cloud for each scene
    clouds = torch.rand((n_scenes * 2, n_points, 3))

    # make the clouds homogenous
    clouds = torch.cat((clouds, torch.ones((n_scenes * 2, n_points, 1))), dim=-1)

    # visualize the clouds at each joint angle using open3d
    import open3d as o3d
    import open3d.visualization.gui as gui

    clouds_o3d = []
    colors_seq = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    
    for scene in range(n_scenes):
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(clouds[scene*2,:,:3].numpy())
        cloud_o3d.paint_uniform_color(colors_seq[scene])
        cloud_o3d.transform(rev_transforms[scene].numpy())
        clouds_o3d.append(cloud_o3d)

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(clouds[scene*2+1,:,:3].numpy())
        cloud_o3d.paint_uniform_color(colors_seq[scene])
        cloud_o3d.transform(prs_transforms[scene].numpy())
        clouds_o3d.append(cloud_o3d)

    gui.Application.instance.initialize()
    app = CloudVisApp()

    def rev_slider_cb_fn(slider_name, scene_idx, float_val):
        rev_joint.joint_params[scene_idx] = float_val
        # get the new transforms
        rev_transforms = rev_joint.get_scene_transform(torch.Tensor([scene_idx]).to(torch.long))
        # get the existing cloud 
        cloud = clouds_o3d[scene_idx*2]
        # update the cloud
        cloud.points = o3d.utility.Vector3dVector(clouds[scene_idx*2,:,:3].numpy())
        cloud.transform(rev_transforms[0].numpy())
        print(f"updated transform: {rev_transforms[0]}")
        app.update_cloud(cloud, f"cloud_{scene_idx}")
        print(f"slider name: {slider_name}")
        print(f"scene_idx: {scene_idx}")
        print(f"slider value: {float_val}")

    def prs_slider_cb_fn(slider_name, scene_idx, float_val):
        prs_joint.joint_params[scene_idx] = float_val
        # get the new transforms
        prs_transforms = prs_joint.get_scene_transform(torch.Tensor([scene_idx]).to(torch.long))
        # get the existing cloud
        cloud = clouds_o3d[scene_idx * 2 + 1]
        # update the cloud
        cloud.points = o3d.utility.Vector3dVector(clouds[scene_idx*2+1,:,:3].numpy())
        cloud.transform(prs_transforms[0].numpy())
        print(f"updated transform: {prs_transforms[0]}")
        app.update_cloud(cloud, f"cloud_{scene_idx}_prs")
        print(f"slider name: {slider_name}")
        print(f"scene_idx: {scene_idx}")
        print(f"slider value: {float_val}")

    for scene in range(n_scenes):
        app.add_cloud(clouds_o3d[scene*2], f"cloud_{scene}")
        name = f"cloud_{scene}_angle"
        app.add_slider(name, 0, 2*np.pi, rev_joint.joint_params[scene], partial(rev_slider_cb_fn, name, scene))


        app.add_cloud(clouds_o3d[scene*2+1], f"cloud_{scene}_prs")
        name = f"cloud_{scene}_trans"
        app.add_slider(name, -5, 5, prs_joint.joint_params[scene], partial(prs_slider_cb_fn, name, scene))

    gui.Application.instance.run()
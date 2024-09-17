import torch
import numpy as np
import yaml
from typing import Optional
import warnings
import pytorch3d.transforms as p3dt
from splatart.utils.lie_utils import SE3, SE3Exp, SO3Exp

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
        rotation_center = SE3Exp(self.pre_tf) # (4, 4)
        rigid_tf = SE3Exp(self.post_tf) # (4, 4)

        # create a rotation around the z axis for each joint angle
        joint_axis = self.joint_axis.repeat(scene_paramters.shape[0], 1)
        rotation_params = joint_axis[..., :] * scene_paramters[..., None]
        tf_params = torch.cat((rotation_params, torch.zeros(scene_paramters.shape[0], 3)), dim=1)
        joint_transform = SE3Exp(tf_params)

        return rotation_center @ joint_transform @ rigid_tf
    
    # scene_idx: torch.Tensor of shape (B,)
    def get_scene_transform(self, scene_idx:torch.Tensor):

        # get the joint angles form each scene_idx
        joint_angles = self.joint_params[scene_idx]

        return self.get_transform(joint_angles)
    
    def configuration_estimation(self, points:torch.Tensor):
        # points is a tensor of shape (B, 3) in the world frame
        # from these points, we make an estimation of the joint configuration

        # this is an implementation of eq 21 in section 2.5.3 from Sturm et al.
        pre_tf = SE3Exp(self.pre_tf)


        # express the points in the pre_tf frame
        points_homogenous = torch.cat((points, torch.ones(points.shape[0], 1)), dim=1) # (B, 4)
        # print(f"points_homogenous: {points_homogenous}")
        points_pre_tf = torch.matmul(torch.linalg.inv(pre_tf), points_homogenous.T).T
        
        # the rotation axis is always the z axis in the pre_tf frame
        # so we just need the x and y components of the points to get the inverse angle via atan2
        angles = torch.atan2(points_pre_tf[:, 1], points_pre_tf[:, 0])

        return angles

    def initial_estimate_points(self, points:torch.Tensor):
        # points is a tensor of shape (B, 3), with B >= 3
        # from these points, we create an initial estimate for the joint
        # center of rotation, rotation axis, and post transformation
        # this process matches the one described in section 2.5.3 of Sturm et al.

        # if B is > 3, we randomly select 3 points
        if(points.shape[0] > 3):
            points = points[torch.randperm(points.shape[0])[:3]]

        # estimate the plane spanned by the points
        # get the normal of the plane
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]

        # check that p1,2,3 are not colinear
        # if they are, we'll set some estimates then throw a warning
        alpha_test = (p3[0] - p2[0]) / p1[0]
        if(torch.allclose(p1*alpha_test + p2 - p3, torch.zeros(3))):
            warnings.warn("Points are colinear, using default estimates")
            default_center = (p1 + p2 + p3) / 3
            self.pre_tf = torch.nn.Parameter(torch.Tensor([0,0,0,default_center[0],default_center[1],default_center[2]]))
            self.post_tf = torch.nn.Parameter(torch.Tensor([0,0,0,0,0,0]))
            self.joint_params = torch.nn.Parameter(torch.zeros(points.shape[3]))
            return

        # v1 and v2 are spanning vectors for the plane (although not necessarily orthonormal)
        v1 = p2 - p1
        v1 = v1 / torch.norm(v1)
        v2 = p3 - p2
        v2 = v2 / torch.norm(v2)

        # cross product of v1 and v2 gives the normal of the plane
        # which is parallel to the axis of rotation
        normal = torch.cross(v2, v1)
        normal = normal / torch.norm(normal)

        # get the line segment midpoints
        m1 = (p1 + p2) / 2
        m2 = (p2 + p3) / 2
        # get the perpendicular vectors for each line segment
        perp1 = torch.cross(v1, normal)
        perp1 = perp1 / torch.norm(perp1)
        perp2 = torch.cross(v2, normal)
        perp2 = perp2 / torch.norm(perp2)
        # we now need to find the point where the two perpendiculars intersect
        # we want m1 + t1 * perp1 = m2 + t2 * perp2
        A = torch.stack((perp1, -perp2), dim=1)
        B = (m2 - m1).unsqueeze(-1)
        # we use lstsq because want to maybe allow for overconstrainted estimation later
        solution = torch.linalg.lstsq(A, B).solution
        t1 = solution[0, 0]
        t2 = solution[1, 0]
        center_1 = m1 + t1 * perp1
        center_2 = m2 + t2 * perp2
        
        # make sure the predicted centers aren't too far apart
        # if they are, we did something wrong and we need to throw an error
        if(torch.norm(center_1 - center_2) > 0.01):
            raise ValueError("Centers are too far apart, something went wrong")
        
        # make our pre_tf translation component match center_1
        # we then make the x axis point to p1, and the z axis be parallel to the normal
        pre_tf = torch.eye(4)
        pre_tf[:3, 3] = center_1

        # find the rotation matrix to align the x axis with the vector from center_1 to p1
        x_axis = p1 - center_1
        x_axis = x_axis / torch.norm(x_axis)
        z_axis = normal
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        R = torch.stack((x_axis, y_axis, z_axis), dim=1)
        pre_tf[:3, :3] = R
        
        # make our post_tf translation component match p1 in the pre_tf frame
        # this should realistically just be a translation on the x axis
        post_tf = torch.eye(4)
        p1_homo = torch.cat((p1, torch.ones(1)))
        p1_homo_pre_frame = torch.matmul(torch.linalg.inv(pre_tf), p1_homo)

        post_tf[:3, 3] = p1_homo_pre_frame[:3]

        self.pre_tf =  torch.nn.Parameter(SE3.Log(pre_tf))
        self.post_tf = torch.nn.Parameter(SE3.Log(post_tf))

        # next we need to estimate the joint configuration for each scene
        # and set the joint parameters
        angles = self.configuration_estimation(points)
        # print(f"got angles: {angles} for configuration estimation")
        self.joint_params = torch.nn.Parameter(angles)

    def estimate_fk_parameters(self, pose_transforms: torch.Tensor):
        """
        Estimates the pre-rotation transform (c), post-rotation transform (r), and
        the rotation angles about the z-axis (q) from a set of pose transformations.

        Args:
        pose_transforms (torch.Tensor): A tensor of shape (B, 4, 4) representing B transformation matrices.

        Returns:
        c (torch.Tensor): The pre-rotation transformation matrix.
        r (torch.Tensor): The post-rotation transformation matrix.
        q (torch.Tensor): The estimated rotation angles about the z-axis for each pose.
        """
        B = pose_transforms.shape[0]
        
        # Initialize c and r as identity transformations
        c = torch.eye(4, dtype=torch.float32)
        r = torch.eye(4, dtype=torch.float32)
        
        # Rotation angles q for each pose
        q = torch.zeros(B, dtype=torch.float32)
        
        # Iterate through each pose to estimate q and update c and r
        for i in range(B):
            pose = pose_transforms[i]

            # Decompose the rotation matrix from the current pose
            R_pose = pose[:3, :3]  # Rotation part
            t_pose = pose[:3, 3]   # Translation part
            
            # Estimate the rotation angle q from the z-axis rotation
            q[i] = torch.atan2(R_pose[1, 0], R_pose[0, 0])  # Extract angle from rotz
            
            # Construct the rotz(q) matrix for the current pose
            cos_q = torch.cos(q[i])
            sin_q = torch.sin(q[i])
            rotz_q = torch.tensor([[cos_q, -sin_q, 0],
                                [sin_q, cos_q, 0],
                                [0, 0, 1]], dtype=torch.float32)
            
            # Estimate c and r using the decomposed parts of the pose
            # Solve for c and r iteratively
            R_c = R_pose @ rotz_q.T
            R_r = rotz_q.T @ R_pose
            
            # Update c and r rotation parts
            c[:3, :3] = R_c
            r[:3, :3] = R_r
            
            # Update c and r translation parts
            c[:3, 3] = t_pose  # Assuming c includes the majority of the translation
            r[:3, 3] = torch.zeros(3)  # Assuming r has no translation component
        
        return c, r, q

    def rotz(self, q: torch.Tensor):
        """
        Computes the rotation matrix around the z-axis for each angle q.

        Args:
        q (torch.Tensor): A tensor of shape (N,) representing the rotation angles.

        Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) representing the rotation matrices.
        """
        cos_q = torch.cos(q)
        sin_q = torch.sin(q)
        
        # Initialize an identity matrix of shape (N, 4, 4)
        rotz_matrices = torch.eye(4).repeat(q.shape[0], 1, 1)
        
        # Fill in the z-rotation components in the 2D plane
        rotz_matrices[:, 0, 0] = cos_q
        rotz_matrices[:, 0, 1] = -sin_q
        rotz_matrices[:, 1, 0] = sin_q
        rotz_matrices[:, 1, 1] = cos_q
        
        return rotz_matrices
    
    def roty(self, q: torch.Tensor):
        """
        Computes the rotation matrix around the y-axis for each angle q.

        Args:
        q (torch.Tensor): A tensor of shape (N,) representing the rotation angles.

        Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) representing the rotation matrices.
        """
        cos_q = torch.cos(q)
        sin_q = torch.sin(q)
        
        # Initialize an identity matrix of shape (N, 4, 4)
        roty_matrices = torch.eye(4).repeat(q.shape[0], 1, 1)
        
        # Fill in the y-rotation components in the 2D plane
        roty_matrices[:, 0, 0] = cos_q
        roty_matrices[:, 0, 2] = sin_q
        roty_matrices[:, 2, 0] = -sin_q
        roty_matrices[:, 2, 2] = cos_q
        
        return roty_matrices

    def forward_kinematics(self, t: torch.Tensor, axis: torch.Tensor, q: torch.Tensor):
        """
        Computes the joint's transformation matrix for each q using the forward kinematics model.

        Args:
        c (torch.Tensor): A tensor of shape (4, 4) representing the pre-rotation transformation.
        r (torch.Tensor): A tensor of shape (4, 4) representing the post-rotation transformation.
        q (torch.Tensor): A tensor of shape (N,) representing the rotation angles around the z-axis.

        Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) representing the transformation matrices for each q.
        """
        # Compute the rotation matrices around the z-axis for each q
        rotz_matrices = self.roty(q)
        
        # Compute the transformation matrices for each q
        transforms = torch.einsum('ij,bjk->bik', c, rotz_matrices)  # c @ rotz(q) for each q
        transforms = torch.einsum('bij,jk->bik', transforms, r)     # (c @ rotz(q)) @ r
        
        return transforms
    
    def initial_estimate_gd(self, poses: torch.Tensor):
        self.num_scenes = poses.shape[0]
        q_rest = torch.nn.Parameter(torch.zeros(self.num_scenes - 1))
        r = torch.nn.Parameter(torch.eye(4))
        c = torch.nn.Parameter(torch.eye(4))
        # Define the optimizer
        optimizer = torch.optim.Adam( [q_rest, r, c], lr=0.001)
        poses_quats = p3dt.matrix_to_quaternion(poses[:, :3, :3])
        poses_trans = poses[:, :3, 3]

        for epoch in range(10000):
            optimizer.zero_grad()
            # add zeros before q rest to force first joint angle to be zero
            q = torch.zeros(1)
            q = torch.cat((q, q_rest)) 
            transforms = self.forward_kinematics(c, r, q)
            transforms_quats = p3dt.matrix_to_quaternion(transforms[:, :3, :3])
            transforms_trans = transforms[:, :3, 3]
            loss = (10 * torch.mean(torch.norm(poses_quats - transforms_quats, dim=1)) + 
                    torch.mean(torch.norm(poses_trans - transforms_trans, dim=1)))
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, q: {q}, c: {c}, r: {r}, Loss: {loss.item()}")
        return c, r, q

    def get_initial_estimate(self, poses:torch.Tensor):
        # c, r, q = self.estimate_fk_parameters(poses)
        c, r, q = self.initial_estimate_gd(poses)
        self.joint_params = torch.nn.Parameter(q)
        self.pre_tf = torch.nn.Parameter(SE3.Log(c))
        self.post_tf = torch.nn.Parameter(SE3.Log(r))
        

class PrismaticJoint(SingleAxisJoint):
    def __init__(self, num_scenes:int):
        super().__init__(num_scenes)

    def get_gt_parameters(self):
        return {"pre_tf": self.pre_tf, "joint_axis": self.joint_axis, "joint_params": self.joint_params}

    # scene_parameters: torch.Tensor of shape (B,)
    def get_transform(self, scene_paramters: torch.Tensor):
        initial_tf = SE3Exp(self.pre_tf) # (4, 4)

        # assume each joint is a translation along self.joint_axis
        trans_joint = self.joint_axis.repeat(scene_paramters.shape[0], 1) * scene_paramters[..., None]

        tf_params = torch.cat((torch.zeros(scene_paramters.shape[0], 3), trans_joint), dim=1)
        joint_transform = SE3Exp(tf_params)

        # compose the transforms
        scene_tf = initial_tf @ joint_transform
        return scene_tf

    # scene_idx: torch.Tensor of shape (B,)
    def get_scene_transform(self, scene_idx:torch.Tensor):
        # get the joint displacements using the scene_idx's
        joint_displacements = self.joint_params[scene_idx]

        return self.get_transform(joint_displacements)
    
    def configuration_estimation(self, points:torch.Tensor):
        # get the points in the pre_tf frame
        pre_tf = SE3Exp(self.pre_tf)
        points_homogenous = torch.cat((points, torch.ones(points.shape[0], 1)), dim=1)
        points_pre_tf = torch.matmul(torch.linalg.inv(pre_tf), points_homogenous.T).T # (B, 4)

        # configuration estimate is projection of the points onto the joint axis
        displacements = torch.matmul(points_pre_tf[:,:3], self.joint_axis) 
        return displacements
    
    def get_initial_estimate(self, points:torch.Tensor):
        # points is a tensor of shape (B, 3), with B >= 3
        # from these points, we create an initial estimate for the joint
        # pre transformation and prismatic axis
        # this process matches the one described in section 2.5.2 of Sturm et al.

        # if B is > 2, we randomly select 2 points
        if(points.shape[0] > 2):
            points = points[torch.randperm(points.shape[0])[:2]]

        # estimate the plane spanned by the points
        # get the normal of the plane
        p1 = points[0]
        p2 = points[1]
        # print(f"p1: {p1}, p2: {p2}")
        self.pre_tf = torch.nn.Parameter(torch.Tensor([0,0,0,p1[0],p1[1],p1[2]])) # pre_tf is the translation to the first point

        # the prismatic axis becomes the vector from p1 to p2
        axis = (p2 - p1)
        axis = axis / torch.norm(axis)
        # print(f"axis: {axis}")
        self.joint_axis = torch.nn.Parameter(axis) # (3, )

        # next we need to estimate the joint configuration
        self.joint_params = torch.nn.Parameter(self.configuration_estimation(points))
        # print(f"found pre tf: {self.pre_tf}, joint axis: {self.joint_axis}, joint params: {self.joint_params}")

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
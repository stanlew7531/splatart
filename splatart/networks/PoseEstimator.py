import torch
from pytorch_msssim import SSIM
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pytorch3d.transforms as p3dt
import cv2 as cv
import os
import numpy as np
from splatart.managers.SplatManager import SplatManagerSingle, means_quats_to_mat, mat_to_means_quats, means_eulers_to_mat


class PoseEstimator():
    def __init__(self, num_parts, num_managers):
        self.num_parts = num_parts
        self.num_scenes = num_managers
        self.part_means = torch.nn.Parameter(torch.zeros((num_managers, num_parts, 3)))
        self.part_eulers = torch.nn.Parameter(torch.zeros((num_managers, num_parts, 3)))
        self.SSIM = SSIM(data_range=1.0, size_average=True, channel=3)
        self.opt_centroids = None

    def preset_centroids(self, centroids):
        for idx in range(self.num_scenes):
            self.part_means[idx] = centroids[idx]

    def preset_tfs_from_icp(self, icp_json):
        print(f"presetting tfs from icp json")
        for part_idx in range(self.num_parts):
            part_tfs = icp_json[part_idx]
            for manager_idx in range(self.num_scenes):
                part_tf = np.array(part_tfs[manager_idx])
                trans = part_tf[0:3, 3]
                rot = part_tf[0:3, 0:3]
                # turn the rot to euler angles
                rot = torch.from_numpy(rot)
                eulers = p3dt.matrix_to_euler_angles(rot, "XYZ")
                self.part_means[manager_idx][part_idx] = torch.from_numpy(trans)
                self.part_eulers[manager_idx][part_idx] = eulers

    ######################
    # returns a tensor part_frames which contains the homogenous matrix form of the estimated part frames
    # also returns scene_part_frame_gauss_params which is each part from object_parts_gauss_params, but in that predicted frame
    def get_part_splat_frames(self, object_parts_gauss_params:list):
        # get the current estimate of each parts frame
        part_frames = torch.zeros((self.num_scenes, self.num_parts, 4, 4)).to(object_parts_gauss_params[0][0]["means"])
        part_frames[..., :3, :3] = p3dt.euler_angles_to_matrix(self.part_eulers[...], "XYZ")
        part_frames[..., :3, 3] = self.part_means[...]
        part_frames[..., 3, 3] = 1

        # take each scene/part gauss param input and put all splats into the predicted part frame
        # (num_managers, (num_parts, gauss_params dict))
        scene_part_frame_gauss_params = []
        for scene_i in range(self.num_scenes):
            part_frame_gauss_params = []
            for part_i in range(self.num_parts):
                means = object_parts_gauss_params[scene_i][part_i]["means"]
                # means.register_hook(lambda grad: print("got to means grad: {grad}"))

                quats = object_parts_gauss_params[scene_i][part_i]["quats"]
                # quats.register_hook(lambda grad: print("got to quats grad: {grad}"))
                mat = means_quats_to_mat(means, quats)
                mat = torch.linalg.inv(part_frames[scene_i][part_i]) @ mat

                new_means, new_quats = mat_to_means_quats(mat)
                
                gauss_params = {
                    "means": new_means,
                    "quats": new_quats,
                    "features_semantics": object_parts_gauss_params[scene_i][part_i]["features_semantics"],
                    "features_dc": object_parts_gauss_params[scene_i][part_i]["features_dc"],
                    "features_rest": object_parts_gauss_params[scene_i][part_i]["features_rest"],
                    "opacities": object_parts_gauss_params[scene_i][part_i]["opacities"],
                    "scales": object_parts_gauss_params[scene_i][part_i]["scales"],
                    "mat": mat
                }
                part_frame_gauss_params.append(gauss_params)
            scene_part_frame_gauss_params.append(part_frame_gauss_params)

        return part_frames, scene_part_frame_gauss_params
    
    def generate_scene_gauss_params(self, part_frames, object_parts_gauss_params, render_scene):
        # convert each part frame into the frame of the render_scene
        # and assign it back to the object gauss params as needed
        dst_part_frames = part_frames[render_scene] # of shape (num_parts, 4, 4)

        scene_part_frame_gauss_params = []
        for scene_i in range(self.num_scenes):
            part_frame_gauss_params = []
            for part_i in range(self.num_parts):
                mat = object_parts_gauss_params[scene_i][part_i]["mat"]
                mat = dst_part_frames[part_i] @ mat
                means, quats = mat_to_means_quats(mat)
                gauss_params = {
                    "means": means,
                    "quats": quats,
                    "features_semantics": object_parts_gauss_params[scene_i][part_i]["features_semantics"],
                    "features_dc": object_parts_gauss_params[scene_i][part_i]["features_dc"],
                    "features_rest": object_parts_gauss_params[scene_i][part_i]["features_rest"],
                    "opacities": object_parts_gauss_params[scene_i][part_i]["opacities"],
                    "scales": object_parts_gauss_params[scene_i][part_i]["scales"],
                }
                part_frame_gauss_params.append(gauss_params)
            scene_part_frame_gauss_params.append(part_frame_gauss_params)

        return scene_part_frame_gauss_params
    
    # given a list of gaussian parameters, combine them for later rendering
    def combine_guass_params(self, scene_part_frame_gauss_params):
        per_scene_gauss_params = []
        for scene_i in range(0, self.num_scenes):
            object_gauss_params = {
                "means": scene_part_frame_gauss_params[scene_i][0]["means"],
                "quats": scene_part_frame_gauss_params[scene_i][0]["quats"],
                "features_semantics": scene_part_frame_gauss_params[scene_i][0]["features_semantics"],
                "features_dc": scene_part_frame_gauss_params[scene_i][0]["features_dc"],
                "features_rest": scene_part_frame_gauss_params[scene_i][0]["features_rest"],
                "opacities": scene_part_frame_gauss_params[scene_i][0]["opacities"],
                "scales": scene_part_frame_gauss_params[scene_i][0]["scales"],
            }

            for part_i in range(1, self.num_parts):
                if scene_i == 0 and part_i == 0:
                    continue
                object_gauss_params["means"] = torch.cat((object_gauss_params["means"], scene_part_frame_gauss_params[scene_i][part_i]["means"]), dim=0)
                object_gauss_params["quats"] = torch.cat((object_gauss_params["quats"], scene_part_frame_gauss_params[scene_i][part_i]["quats"]), dim=0)
                object_gauss_params["features_semantics"] = torch.cat((object_gauss_params["features_semantics"], scene_part_frame_gauss_params[scene_i][part_i]["features_semantics"]), dim=0)
                object_gauss_params["features_dc"] = torch.cat((object_gauss_params["features_dc"], scene_part_frame_gauss_params[scene_i][part_i]["features_dc"]), dim=0)
                object_gauss_params["features_rest"] = torch.cat((object_gauss_params["features_rest"], scene_part_frame_gauss_params[scene_i][part_i]["features_rest"]), dim=0)
                object_gauss_params["opacities"] = torch.cat((object_gauss_params["opacities"], scene_part_frame_gauss_params[scene_i][part_i]["opacities"]), dim=0)
                object_gauss_params["scales"] = torch.cat((object_gauss_params["scales"], scene_part_frame_gauss_params[scene_i][part_i]["scales"]), dim=0)
            
            per_scene_gauss_params.append(object_gauss_params)

        return per_scene_gauss_params
    
    def preprocess_images(self, render_image_results, gt_rgba):
        render_rgb = render_image_results[0]
        render_alpha = render_image_results[1]
        render_rgba = torch.cat([render_rgb, render_alpha], dim=-1)

        gt_rgb = gt_rgba[..., :3]
        gt_alpha = gt_rgba[..., 3].unsqueeze(-1)

        # put the GT and rendered images against the same background
        white_bg = torch.ones_like(gt_rgba)

        # composite rendered onto white background
        render_whitebg = render_rgba * render_alpha + white_bg * (1 - render_alpha)
        gt_whitebg = gt_rgba * gt_alpha + white_bg * (1 - gt_alpha)
        render_whitebg = render_whitebg[..., :3] # only want rgb for the white bg versions
        gt_whitebg = gt_whitebg[..., :3]

        return render_rgb, render_alpha, gt_rgb, gt_alpha, render_whitebg, gt_whitebg
    
    def compute_image_losses(self, render_whitebg, gt_whitebg, gt_alpha, render_alpha, ssim_fn):
        loss_l1 = torch.nn.functional.l1_loss(render_whitebg, gt_whitebg)

        ssim_loss = 1 - ssim_fn(gt_whitebg.permute(0,3,1,2), render_whitebg.permute(0,3,1,2))
        acc_loss = torch.abs(gt_alpha - render_alpha).mean()

        losses = {
            "l1": loss_l1,\
            "ssim": ssim_loss,\
            "acc": acc_loss,\
        }

        return losses
    
    def combine_losses(self, batch_losses:dict, loss_lambdas:dict):
        loss_tot = 0

        for loss_key in batch_losses.keys():
            loss_tot += loss_lambdas[loss_key] * (batch_losses[loss_key])

        return loss_tot

    def forward_and_loss(self,\
                            managers:list[SplatManagerSingle],\
                            object_parts_gauss_params,\
                            manager_tfs,\
                            rgbs_gt,\
                            segmentations_gt,\
                            cam_intrinsic,\
                            width,\
                            height,\
                            render_scene=0,
                            writer:SummaryWriter=None):
        # managers are the SplatManagerSingle objects
        # manager tfs is of shape (num_managers, batch_size, 4, 4) and are the poses to render each scene at
        # rgbs_gt is of shape (num_managers, batch_size, height, width, 4) and are the expexted gt image to render
        # segmentations_gt is of shape (num_managers, batch_size, height, width) and are the expected segmentation to render

        # first, transform every splat to its part's frame
        part_frames, scene_part_frame_gauss_params = self.get_part_splat_frames(object_parts_gauss_params)

        render_scene_part_gauss_params = self.generate_scene_gauss_params(part_frames, scene_part_frame_gauss_params, render_scene=render_scene)
        render_scene_gauss_params = self.combine_guass_params(render_scene_part_gauss_params) # represents each scenes gaussian splats transformed to hopefully match scense 0's geometry

        loss = 0.0
        scene_losses = []
        scene_render_rgbs = []

        loss_lambdas = {
                "l1": 0.8,
                "ssim": 0.2,
                "acc": 1e-7,
                "segmentation": 1e-1,
            }
        
        batch_size = rgbs_gt.shape[0]

        for render_scene_idx in range(0, self.num_scenes):
            render_image_results = managers[0].render_gauss_params_at_campose(\
                manager_tfs, cam_intrinsic.expand(batch_size, -1, -1), width, height, render_scene_gauss_params[render_scene_idx], is_semantics=False)
            
            semantics_pred, _, _ = managers[0].render_gauss_params_at_campose(
                manager_tfs, cam_intrinsic.expand(batch_size, -1, -1), width, height, render_scene_gauss_params[render_scene_idx], is_semantics=True)
            
            # compute the loss
            render_rgb, render_alpha, gt_rgb, gt_alpha, render_whitebg, gt_whitebg = \
                    self.preprocess_images(render_image_results, rgbs_gt)
            
            image_losses = self.compute_image_losses(render_whitebg, gt_whitebg, gt_alpha, render_alpha, self.SSIM)
            image_losses["segmentation"] = torch.nn.CrossEntropyLoss()(semantics_pred.reshape(-1, self.num_parts), segmentations_gt.flatten())
            scene_loss = self.combine_losses(image_losses, loss_lambdas)
            
            loss += scene_loss
            scene_render_rgbs.append(render_rgb)
            scene_losses.append(image_losses)

        return loss, scene_losses, scene_render_rgbs, gt_rgb
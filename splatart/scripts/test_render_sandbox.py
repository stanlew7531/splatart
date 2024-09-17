from splatart.managers.NerfManager import NerfManager
import torch
import cv2 as cv

config = "splatart/configs/blade_test_manager.json"

manager = NerfManager(config)

example_cam_pose = torch.Tensor([
                [
                    0.10764255182033479,
                    -0.618228138426726,
                    -0.778592993736157,
                    -2.3357789812084704
                ],
                [
                    -0.9941896604962287,
                    -0.06693657867470884,
                    -0.08429954565546588,
                    -0.2528986369663969
                ],
                [
                    1.4888164241419157e-16,
                    0.7831433223189412,
                    -0.6218412471903506,
                    -1.8655237415710524
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])
fx = fy = 761.1827392578125
w = 640
h = 480

imgs, ray_bundles, depths, cam_intrinsic_mat, segmentations = manager.render_model_at_pose("blade", example_cam_pose.unsqueeze(0))

print(cam_intrinsic_mat)

cv.imwrite("test_blade.png", imgs[0])
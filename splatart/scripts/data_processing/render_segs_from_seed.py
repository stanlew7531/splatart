import sys
import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_msssim import SSIM
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import HDBSCAN
import pytorch3d.transforms as p3dt

from splatart.managers.SplatManager import SplatManagerSingle, load_managers, means_quats_to_mat, mat_to_means_quats
from splatart.datasets.splat_train_dataset import SplatTrainDataset
from splatart.networks.PoseEstimator import PoseEstimator

def load_base_model(input_model_path:str)->SplatManagerSingle:
    return torch.load(input_model_path)

def get_dataset(dataset_pth:str):
    return SplatTrainDataset(dataset_pth, use_semantics=True)

def unpack_gaussian_params(object_gaussian_params, num_parts):
    to_return = []
    part_labels = torch.argmax(object_gaussian_params["features_semantics"], dim=1)
    for part_idx in range(num_parts):
        to_append = {
            "means": object_gaussian_params["means"][part_labels == part_idx],
            "quats": object_gaussian_params["quats"][part_labels == part_idx],
            "features_semantics": object_gaussian_params["features_semantics"][part_labels == part_idx],
            "features_dc": object_gaussian_params["features_dc"][part_labels == part_idx],
            "features_rest": object_gaussian_params["features_rest"][part_labels == part_idx],
            "opacities": object_gaussian_params["opacities"][part_labels == part_idx],
            "scales": object_gaussian_params["scales"][part_labels == part_idx],
        }
        to_append["means"].requires_grad = True
        to_append["quats"].requires_grad = True
        to_return.append(to_append)

    return to_return


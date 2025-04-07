import sys
import os
import argparse
import torch
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as p3dt

from splatart.networks.PointTransformer.model import PointTransformerV3

def load_models(input_model_paths:list[str]):
    to_return = []
    for model_path in input_model_paths:
        print(f"loading pth from: {model_path}")
        to_return.append(torch.load(model_path))
    return to_return

def make_pt3_dict_from_splat(splat_manager, batch_size=2):
    gauss_params = splat_manager.object_gaussian_params
    means = gauss_params["means"]
    quats = gauss_params["quats"]
    semantics_features = gauss_params["features_semantics"]
    color_features = gauss_params["features_dc"]
    scale_features = gauss_params["scales"]

    train_dict = {}
    dummy_normals = torch.zeros_like(color_features)
    train_dict["feat"] = torch.concat([color_features, dummy_normals], dim=-1)
    print(f'features shape: {train_dict["feat"].shape}')
    train_dict["coord"] = means
    train_dict["grid_size"] = 0.01
    n_splats = means.shape[0]
    n_batches = n_splats // batch_size + 1
    batch_labels = torch.arange(n_batches).repeat_interleave(batch_size, dim=0)[:n_splats]
    train_dict["batch"] = batch_labels.to(means.device)

    return train_dict

def run_pt3_model(input_dict, model):
    model.to(input_dict["coord"].device)
    return model(input_dict)

def make_test_dictionary(manager_paths, output_dir):
    # make the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load each of the input splats from nerfstudio
    print(f"Loading splat managers from {manager_paths}")
    managers_list = load_models(manager_paths)

    print(f"Making point net dicts from splat managers...")
    train_dicts = [make_pt3_dict_from_splat(manager) for manager in managers_list]

    print(f"Making the PT3 model...")
    # note that default configuration for PTv3 matches what the paper used
    # so we will use that (lets us load the pretrained models if desired)
    model = PointTransformerV3(enable_flash=False)

    print(f"Running the model...")
    for i, train_dict in enumerate(train_dicts):
        print(f"Running model on splat {i}...")
        output = run_pt3_model(train_dict, model)
        train_dict["output"] = output
        print(f"output keys: {output.keys()}")


if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Given a list of splats already trained on RGB, learn segmentations for the parts")

    parser.add_argument('--splat_manager_pths', 
                        type=str,
                        help='pre-trained splats to learn on',
                        default="results/icra_workshop/sapien_35059/seg_learned_manager_0.pth;results/icra_workshop/sapien_35059/seg_learned_manager_1.pth")
    
    parser.add_argument('--output_dir', 
                        type=str,
                        help='folder to save the output',
                        default="outputs/icra_sapien_35059/splatart_neo/")

    args = parser.parse_args()
    manager_paths = args.splat_manager_pths.split(";")
    make_test_dictionary(manager_paths, args.output_dir)

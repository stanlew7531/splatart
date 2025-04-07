import os
import sys
import numpy as np
import argparse
import cv2 as cv
import json

def get_num_parts(transforms_json):
    # tld of transforms.json so we can load the data
    transfoms_json_dir = os.path.dirname(transforms_json)

    with open(transforms_json, "r") as f:
        transforms = json.load(f)
    
    data_frames = transforms["frames"]

    max_idx = 0
    for frame in data_frames:
        segmentation_fname = frame["semantics_path"]
        segmentation_data = cv.imread(os.path.join(transfoms_json_dir, segmentation_fname), cv.IMREAD_UNCHANGED)
        max_part = np.max(segmentation_data)
        if max_part > max_idx:
            max_idx = max_part
    
    return max_idx

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--transforms_json', 
                        type=str,
                        help="location of the generated sapien dataset's transforms.json file",
                        default="/home/stanlew/Data/icra_sapien_data/v0/sapien_35059/0/transforms.json")

    args = parser.parse_args()

    max_part = get_num_parts(args.transforms_json)
    print(max_part + 1)
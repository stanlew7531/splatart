import argparse
import json
import os
import sys
import cv2 as cv
import numpy as np
from tqdm import tqdm

def fix_segmentations(transform_json_fnames, num_parts=2, parts_to_combine=[]):
    for json_fname in transform_json_fnames:
        json_data = json.load(open(json_fname, "r"))
        json_dir = os.path.dirname(json_fname)
        data_frames = json_data["frames"]
        unique_ids = None
        for frame in data_frames:
            segmentation_mask_file = frame["semantics_path"]
            segmentation_data = cv.imread(os.path.join(json_dir,segmentation_mask_file), cv.IMREAD_UNCHANGED)
            unique_segmentation_ids = np.unique(segmentation_data)
            if unique_ids is None:
                unique_ids = unique_segmentation_ids
            else:
                unique_ids = np.union1d(unique_ids, unique_segmentation_ids)
            unique_ids = np.unique(unique_ids)

        print(f"Unique ids in segmentation masks: {unique_ids}")
        print(f"Got {len(unique_ids)} unique ids in segmentation masks, expected at most {num_parts}")

        # Now we have all the unique ids, we need to remap them to 0, 1, 2, ...
        for frame in data_frames:
            segmentation_mask_file = frame["semantics_path"]
            segmentation_data = cv.imread(os.path.join(json_dir,segmentation_mask_file), cv.IMREAD_UNCHANGED)
            os.makedirs(os.path.join(json_dir,"old",segmentation_mask_file), exist_ok=True)
            cv.imwrite(os.path.join(json_dir,"old",segmentation_mask_file), segmentation_data)
            for i, unique_id in enumerate(unique_ids):
                segmentation_data[segmentation_data == unique_id] = i
            
            print(f"parst to combine: {parts_to_combine}")
            for combination in parts_to_combine:
                base_id = int(combination[0])
                print(f"got base id: {base_id}")
                for i in range(1, len(combination)):
                    part_id = int(combination[i])
                    print(f"checking for part id: {part_id}")
                    segmentation_data[segmentation_data == part_id] = base_id

            unique_segmentation_ids = np.unique(segmentation_data)
            assert len(unique_segmentation_ids) <= num_parts, f"Segmentation mask {segmentation_mask_file} has {len(unique_segmentation_ids)} unique ids, expected at most {num_parts}"
            print(f"new unique segment ids: {np.unique(segmentation_data)}")
            # Save the new segmentation mask
            cv.imwrite(os.path.join(json_dir,segmentation_mask_file), segmentation_data)


if __name__ == "__main__":
    print("Clustering the parts between base splat and transformed scene...")

    parser = argparse.ArgumentParser(description="Fix the segmentation masks for the parts to be more reasonable numbers")
    
    parser.add_argument('--splat_model_datasets',
                        type=str,
                        help="list of datasets used to train the splats",
                        default="")
    
    parser.add_argument('--n_parts',
                        type=int,
                        help="number of parts in the object/scene",
                        default=2)
    
    parser.add_argument('--parts_to_combine',
                        type=str,
                        help="list of parts to combine",
                        default="")
    
    args = parser.parse_args()
    dataset_paths = args.splat_model_datasets.split(",")
    combine_parts = args.parts_to_combine.split(";")
    combinations = []
    for line in combine_parts:
        parts = line.split(",")
        if len(parts) > 1:
            combinations.append(parts)
    fix_segmentations(dataset_paths, args.n_parts, combinations)
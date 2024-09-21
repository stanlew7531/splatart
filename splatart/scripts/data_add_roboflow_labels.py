import cv2
import numpy as np
import os
import json
import glob
import argparse

def load_transforms_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def create_mask(image_shape, mask_entries, class_offset = 1):
    mask_img = np.zeros(image_shape, dtype=np.uint8)
    for class_id, polygons in mask_entries.items():
        for polygon in polygons:
            polygon_img_shape = polygon * np.array([image_shape[1], image_shape[0]]) # turn into pixel coordinates vs normalized
            cv2.fillPoly(mask_img, [polygon_img_shape.astype(np.int32)], class_id + class_offset)
    return mask_img

def load_yolo_data(yolo_labels_dir, dataset_type='train', ns_transforms_fname="transforms.json"):
    images_dir = os.path.join(yolo_labels_dir, dataset_type, 'images')
    labels_dir = os.path.join(yolo_labels_dir, dataset_type, 'labels')

    mask_data = {}
    for image_path in glob.glob(os.path.join(images_dir, '*.jpg')):
        print(f"Processing yolo img: {image_path}")
        image_name = os.path.basename(image_path)
        original_image = cv2.imread(image_path)
        original_image_size = original_image.shape[:2]
        original_image_name = image_name.split(".")[0].replace("_jpg", ".jpg")
        label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))

        # read in the label file and determine the mask locations
        with(open(label_path)) as f:
            lines = f.readlines()
        mask_entries = {}
        for line in lines:
            # process each line
            entries = line.split(" ")
            class_id = int(entries[0])
            polygon_pts = np.array(entries[1:], dtype=np.float32).reshape(-1, 2)
            if class_id not in mask_entries:
                mask_entries[class_id] = [polygon_pts]
            else:
                mask_entries[class_id].append(polygon_pts)

        mask_data[original_image_name] = mask_entries
    return mask_data

def merge_r3d_data_masks(input_r3d_data_dir, mask_data, ns_transforms_fname="transforms.json", masks_dir="masks"):
    transforms_json_data = load_transforms_json(os.path.join(input_r3d_data_dir, ns_transforms_fname))
    frame_data = transforms_json_data["frames"]
    mask_img_dir = os.path.join(input_r3d_data_dir, masks_dir)
    os.makedirs(mask_img_dir, exist_ok=True)
    new_frames = []
    for frame in frame_data:
        print(f"Processing R3D frame: {frame['file_path']}")
        image_file = frame["file_path"]
        image_name = os.path.basename(image_file)
        if not image_name in mask_data:
            print(f"Warning: {image_name} not found in mask data.")
            continue
        mask_entries = mask_data[image_name]
    
        # create mask
        original_image = cv2.imread(os.path.join(input_r3d_data_dir,image_file))
        origina_image_shape = original_image.shape[:2]
        mask_img = create_mask(origina_image_shape, mask_entries)

        # save mask
        mask_img_fname = os.path.join(mask_img_dir, image_name.replace('.jpg', '.png'))
        cv2.imwrite(mask_img_fname, mask_img)
        # get the mask img path relative to input_r3d_data
        frame["semantics_path"] = os.path.join(masks_dir, image_name.replace('.jpg', '.png'))

        new_frames.append(frame)
    filtered_json_data = transforms_json_data
    filtered_json_data["frames"] = new_frames

    output_ns_transforms_fname = "masked_" + ns_transforms_fname
    output_masked_ns_transforms_fname = "masked_filtered_" + ns_transforms_fname
    with(open(os.path.join(input_r3d_data_dir, output_ns_transforms_fname), 'w')) as f:
        json.dump(transforms_json_data, f, indent=4)
    with(open(os.path.join(input_r3d_data_dir, output_masked_ns_transforms_fname), 'w')) as f:
        json.dump(filtered_json_data, f, indent=4)

def process_data(input_r3d_data_dir, input_labels_dir, output_dir):
    mask_data = load_yolo_data(input_labels_dir)
    merge_r3d_data_masks(input_r3d_data_dir, mask_data)
        

if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description='Add Roboflow labels to images.')
    parser.add_argument('--input_r3d_data_dir', type=str, required=True, help='Path to record 3d dataset')
    parser.add_argument('--input_labels_dir', type=str, required=True, help='Path to the YOLO labels directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()
    process_data(args.input_r3d_data_dir, args.input_labels_dir, args.output_dir)


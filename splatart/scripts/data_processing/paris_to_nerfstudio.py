import argparse
import json
import os

def load_paris_data(paris_input_folder):
    camera_test_json = os.path.join(paris_input_folder, "camera_test.json")
    camera_train_json = os.path.join(paris_input_folder, "camera_train.json")
    camera_val_json = os.path.join(paris_input_folder, "camera_val.json")

    camera_test_json = json.load(open(camera_test_json, "r"))
    camera_train_json = json.load(open(camera_train_json, "r"))
    camera_val_json = json.load(open(camera_val_json, "r"))

    return camera_test_json, camera_train_json, camera_val_json

def translate_dataset(paris_input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train_images"), exist_ok=True)

    camera_test_json, camera_train_json, camera_val_json = load_paris_data(paris_input_folder)
    train_images_folder = os.path.join(paris_input_folder, "train")
    test_images_folder = os.path.join(paris_input_folder, "test")
    val_images_folder = os.path.join(paris_input_folder, "val")

    train_camera_K = camera_train_json["K"]
    fx = train_camera_K[0][0]
    fy = train_camera_K[1][1]
    cx = train_camera_K[0][2]
    cy = train_camera_K[1][2]

    # get all other keys except for K (these are the file names)
    train_camera_keys = list(camera_train_json.keys())
    train_camera_keys.remove("K")
    train_camera_keys = sorted(train_camera_keys)

    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    print(f"train_camera_keys: {train_camera_keys}")

    nerfstudio_json = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "frames": []}

    w = None
    h = None
    frames = []
    for key in train_camera_keys:
        frame_to_add = {}
        frame_to_add["file_path"] = os.path.join("train_images", f"{key}.png")
        frame_to_add["transform_matrix"] = camera_train_json[key]
        # copy the image from the paris dataset to the output folder
        os.system(f"cp {os.path.join(train_images_folder, key + '.png')} {os.path.join(output_folder, 'train_images', key + '.png')}")
        frames.append(frame_to_add)
        # if we dont have the width and height, get it from the image
        if w is None or h is None:
            from PIL import Image
            im = Image.open(os.path.join(train_images_folder, key + '.png'))
            w, h = im.size

    nerfstudio_json["w"] = w
    nerfstudio_json["h"] = h
    nerfstudio_json["frames"] = frames

    with open(os.path.join(output_folder, "transforms_train.json"), "w") as f:
        json.dump(nerfstudio_json, f, indent=4)


if __name__ == "__main__":
    print("Learning tfs from base splat to other scene...")

    parser = argparse.ArgumentParser(description="Given a pre-trained non-articulated initial splat, learn to render the target scene and the part seperation")

    parser.add_argument('--paris_input_folder', 
                        type=str,
                        help='location of the paris data folder',
                        default="")
    
    parser.add_argument('--output_folder', 
                        type=str,
                        help='output dir to save the results',
                        default="")

    args = parser.parse_args()

    translate_dataset(args.paris_input_folder, args.output_folder)
import sapien.core as sapien
import os
import sys
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageColor
import cv2 as cv
import json
from tqdm import tqdm

# see https://sapien.ucsd.edu/docs/2.2/tutorial/rendering/index.html
# for more information on how to render Sapien objects

def setup_sapien():
    engine = sapien.Engine()
    # enable ray tracing for better images
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_viewer_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(256)
    renderer = sapien.SapienRenderer(offscreen_only=False)
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.)

    scene.set_ambient_light([0.2, 0.2, 0.2])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    scene.add_directional_light([1, 1, 0.5], [0.5, 0.5, 0.5])
    scene.add_directional_light([-1, 1, 0.5], [0.5, 0.5, 0.5])

    scene.add_point_light([4.0762, 1.0055, 5.9039], [1, 1, 1])
    scene.add_point_light([3.4154, 4.6753, 6.5981], [1, 1, 1])
    scene.add_point_light([0.80797, -7.77557, 4.78247], [1, 1, 1])
    scene.add_point_light([-4.96121, 1.9155, 9.01307], [1, 1, 1])

    return engine, renderer, scene

def get_urdf_path(input_sapien:str, obj_id:str):
    # get the urdf path
    urdf_path = os.path.join(input_sapien, obj_id, "mobility.urdf")
    return urdf_path

def load_urdf(scene, urdf_path:str):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    object = loader.load(urdf_path)
    object.set_qf(object.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
    assert object, "Failed to load object from urdf"
    return object, loader

def create_camera(scene):
    near, far = 0.1, 100
    width, height = 640, 480
    
    camera = scene.add_camera(\
        name = "camera", \
        width = width, \
        height = height, \
        fovy = np.deg2rad(35),\
        near = near, \
        far = far, \
            )
    
    return camera

def render_image(scene, camera):
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color") #camera.get_float_texture('Color') # [H, W, 4]
    actor_labels = camera.get_picture("Segmentation")[...,1] #camera.get_visual_actor_segmentation()[..., 1] # [H, W]
    set_background_color(rgba, actor_labels, [1, 1, 1, 0]) # set the background color to transparent white (makes nerfstudio happy)
    return rgba, actor_labels

def generate_training_mask(mask_img, expansion_ratio = 1.125):
    # get the bounding box for the part
    part_bbox = cv.boundingRect(mask_img)
    # expand the bounding box by the ratio
    part_bbox = (part_bbox[0] - int(part_bbox[2] * expansion_ratio/2), part_bbox[1] - int(part_bbox[3] * expansion_ratio/2), int(part_bbox[2] * expansion_ratio), int(part_bbox[3] * expansion_ratio))

    training_segmentation = np.zeros_like(mask_img) # start with no trainable pixels
    # draw the part_bbox on the training segmentation
    cv.rectangle(training_segmentation, (part_bbox[0], part_bbox[1]), (part_bbox[0] + part_bbox[2], part_bbox[1] + part_bbox[3]), 255, -1)
    
    return training_segmentation

def save_images(rgba, actor_labels, scene_idx, sample_idx, output_dir):
    rgba_pil = Image.fromarray((rgba * 255).clip(0, 255).astype("uint8"))

    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
    
    # note: only for the oven
    # actor_labels[actor_labels == 3] = 4 # tmp
    # print(f"actor labels min max: {actor_labels.min()} {actor_labels.max()}")
    # actor_labels[actor_labels == 10] = 3 # set all labels greater than 3 to 2 (the base part)
    # actor_labels[actor_labels > 3] = 2 # set all labels greater than 3 to 0 (the base part)
    # actor_labels[actor_labels > 3] = 3 # set all labels greater than 3 to the moved part (for stapler)
    labels_pil = Image.fromarray(color_palette[actor_labels]) # human readable version of the segmentation
    # actor_labels becomes the part-index labels for use in other parts of the pipeline
    object_mask = actor_labels.copy().astype(np.uint8)
    object_mask[object_mask != 0] = 1
    train_mask_img = generate_training_mask(object_mask)

    # create the output directory
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "segmentations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_colored"), exist_ok=True)

    rgba_fname = os.path.join("images", f"pose_{scene_idx}_{sample_idx}.png")
    segmentation_fname = os.path.join("segmentations", f"seg_{scene_idx}_{sample_idx}.png")
    labels_fname = os.path.join("seg_colored", f"hseg_{scene_idx}_{sample_idx}.png")
    train_mask_output_fname = os.path.join("segmentations", f"{scene_idx}_{sample_idx}_train_mask.png")
    

    rgba_pil.save(os.path.join(output_dir, rgba_fname))
    # make sure actor labels is of type uint16
    actor_labels = actor_labels.astype(np.uint16)
    cv.imwrite(os.path.join(output_dir, segmentation_fname), actor_labels)
    labels_pil.save(os.path.join(output_dir, labels_fname))
    cv.imwrite(os.path.join(output_dir, train_mask_output_fname), train_mask_img)

    return rgba_fname, segmentation_fname, labels_fname, train_mask_output_fname

def get_spherical_pose(radius, theta, phi):
    initial_pose = np.eye(4)
    initial_pose[3,3] = 0.5 # slightly above the object

    # radius transform - move along the +x axis
    tf_radius = np.eye(4)
    tf_radius[0, 3] = radius
    # elevation transform - rotate about the y axis
    tf_theta = np.eye(4)
    rot_matrix = R.from_euler('y', theta, degrees=True).as_matrix()
    tf_theta[:3, :3] = rot_matrix
    # azimuth transform - rotate about the z axis
    tf_phi = np.eye(4)
    rot_matrix = R.from_euler('z', phi, degrees=True).as_matrix()
    tf_phi[:3, :3] = rot_matrix
    # compose the matrices
    final_pose = tf_radius @ tf_theta @ tf_phi @ initial_pose
    return final_pose

def set_object_configuration(obj, qpos):
    obj.set_qpos(qpos)

def get_object_part_poses(obj):
    links = obj.get_links()
    pose_dict = {}
    for link in links:
        pose_dict[link.get_name()] = link.get_pose().to_transformation_matrix().tolist()
    return pose_dict

def get_object_config_minmax(obj):
    qlimits = obj.get_qlimits()
    mins = qlimits[:, 0]
    maxs = qlimits[:, 1]
    return mins, maxs

def set_camera_pose(camera, cam_t, cam_q):
    # camera.set_pose(sapien.Pose.from_transformation_matrix(pose))
    camera.set_pose(sapien.Pose(cam_t, cam_q))

def set_background_color(rgba_img, segmentation_img, color):
    rgba_img[segmentation_img == 0] = color
    return rgba_img

def generate_dataset(urdf_path:str,\
                    obj_name:str,\
                    num_scenes:int,\
                    scene_samples:int,\
                    sample_radius:float,\
                    output_dir:str,\
                    configurations_in=None):
    
    print(f"Generating dataset for {obj_name} with urdf {urdf_path}")
    
    # initial setup of renderer etc.\
    output_dir = os.path.join(output_dir, obj_name)
    os.makedirs(output_dir, exist_ok=True)
    sap_engine, sap_renderer, sap_scene = setup_sapien()
    # urdf_path = get_urdf_path(input_sapien, obj_id)
    obj, sap_loader = load_urdf(sap_scene, urdf_path)
    sap_cam = create_camera(sap_scene)
    cam_intrinsics = sap_cam.get_intrinsic_matrix()
    
    
    config_mins, config_maxs = get_object_config_minmax(obj)
    default_min = -3
    default_max = 3
    # repalce any infs with default values
    config_mins[np.isinf(config_mins)] = default_min
    config_maxs[np.isinf(config_maxs)] = default_max
    print(f"Config mins: {config_mins}, config maxs: {config_maxs}")
    
    # loop over scenes
    for i in tqdm(range(num_scenes), desc = "Generating scenes"):
        transforms_json_data = {"fl_x": float(cam_intrinsics[0, 0]),\
                    "fl_y": float(cam_intrinsics[1, 1]),\
                    "cx": float(cam_intrinsics[0, 2]),\
                    "cy": float(cam_intrinsics[1, 2]),\
                    "w": float(cam_intrinsics[0, 2] * 2),\
                    "h": float(cam_intrinsics[1, 2] * 2),\
                    "camera_model": "OPENCV",\
                    "frames": [],\
                    "configurations": {},
                    "gt_part_world_poses": {}}
        
        # scene_configuration = np.random.uniform(config_mins[0], config_maxs[0])

        # get the middle between the min and max config values
        config_mids = (config_mins + config_maxs) / 2
        # randomly draw from a norma distribution centered on config_mids
        # and with a variance of 0.25 * (config_maxs - config_mins)
        # if(i > 0):
        scene_configuration = np.random.normal(config_mids, 0.25 * (config_maxs - config_mins))
        # else:
        #     scene_configuration = obj.get_qpos()

        # only set the first joint for now
        if(configurations_in is None):
            cur_scene_configuration = obj.get_qpos()
            cur_scene_configuration = scene_configuration # leave the default config and set the first joint to a random spot
            #scene_configuration = cur_scene_configuration
        else:
            scene_configuration = obj.get_qpos()
            for idx in range(len(configurations_in[i])):
                scene_configuration[idx] = configurations_in[i][idx]
        
        transforms_json_data["configurations"][i] = scene_configuration.tolist()
        print(f"current configuration: {obj.get_qpos()}, config in: {scene_configuration}")
        set_object_configuration(obj, scene_configuration)
        sap_scene.step()
        print(f"after setting configuration: {obj.get_qpos()}")
        transforms_json_data["gt_part_world_poses"][i] = get_object_part_poses(obj)
        # print(f"Generating data for scene: {i} and configuration {scene_configuration}")
        # loop over samples
        for j in tqdm(range(scene_samples), desc="Generating samples", leave=False):
            
            set_object_configuration(obj, scene_configuration)
            tf_json_to_add = {}
            # theta_phi_sample = np.random.uniform(-180, 180, 2) # sample theta and phi
            theta = np.random.uniform(45, -45)
            phi = np.random.uniform(-180, 180)
            # theta, phi = theta_phi_sample
            # tqdm.write(f"sample:{j}, theta:{theta}, phi:{phi}", end="\r")
            camera_position = np.linalg.inv(get_spherical_pose(sample_radius, theta, phi))
            camera_trans = camera_position[:3, 3]
            camera_rot = camera_position[:3, :3]
            # turn camera_rot to quaternion
            camera_rot = R.from_matrix(camera_rot)
            camera_rot = camera_rot.as_quat()
            # turn from xyzw to wxyz
            camera_rot = np.array([camera_rot[3], camera_rot[0], camera_rot[1], camera_rot[2]])
            set_camera_pose(sap_cam, camera_trans, camera_rot)
            rgba_render, actor_labels_render = render_image(sap_scene, sap_cam)
            rgba_fname, seg_fname, labels_fname, train_mask_fname = save_images(rgba_render, actor_labels_render, i, j, os.path.join(output_dir, str(i)))
            # need to get the transform between the SAPIEN world space and the NeRF world space
            # SAPIEN: x forward, z up
            # NeRF: z forward, y up
            # + 90 deg rotation about x, then +90 deg rotation about z
            rot_matrix = R.from_euler('z', -90, degrees=True).as_matrix() @ R.from_euler('x', 90, degrees=True).as_matrix()
            cam_tf = np.eye(4)
            cam_tf[:3, :3] = rot_matrix
            camera_position = camera_position @ cam_tf
            tf_json_to_add["file_path"] = rgba_fname
            tf_json_to_add["semantics_path"] = seg_fname
            tf_json_to_add["transform_matrix"] = (camera_position).tolist()
            tf_json_to_add["pose_idx"] = i
            tf_json_to_add["time"] = i
            # tf_json_to_add["mask_path"] = train_mask_fname
            transforms_json_data["frames"].append(tf_json_to_add)

        # save the json file
        with open(os.path.join(output_dir, str(i), "transforms.json"), "w") as f:
            json.dump(transforms_json_data, f, indent=4)

if __name__=="__main__":
    print("Generating data for sapien dataset...")

    parser = argparse.ArgumentParser(description="Given an entry in the Sapien object set, generate a NARF dataset (images, masks, etc.)")

    parser.add_argument('--urdf_path',
                        type=str,
                        required=True,
                        help='filename of the urdf file to use')
    parser.add_argument('--obj_name',
                        type=str,
                        help='name of the object to use for the dataset',
                        default="blade")
    parser.add_argument('--num_scenes',
                        type=int,
                        help='number of scenes to sample from the configuration space',
                        default=3)
    parser.add_argument('--scene_samples',
                        type=int,
                        help='number of samples to take in each scene',
                        default=100)
    parser.add_argument('--sample_radius',
                        type=float,
                        help='distance from the object to samples from',
                        default=3.0)
    parser.add_argument('--configuration',
                        type=str,
                        required=False,
                        help='comma delimited list describing the configuration of the object, | delimited across scenes',
                        default=None)
    parser.add_argument('--output_dir',
                        type=str,
                        help='output directory to save the datasets',
                        default="/media/stanlew/Data/narf_urdf_data/v1/")
    
    args = parser.parse_args()

    configurations = None
    if args.configuration is not None:
        configurations = [list(map(float, scene.split(","))) for scene in args.configuration.split(":")]

    print(f"Generating for Configurations: {configurations}")
    
    generate_dataset(args.urdf_path,\
                    args.obj_name,\
                    args.num_scenes,\
                    args.scene_samples,\
                    args.sample_radius,\
                    args.output_dir,\
                    configurations_in=configurations)  # pass in the configurations if they exist
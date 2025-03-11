declare urdf_file="/home/stanlew/src/urdf_files_dataset/urdf_files/robotics-toolbox/fetch_description/robots/fetch.urdf"

python splatart/scripts/00_sapien_datagen_urdf.py \
 --urdf_path $urdf_file \
 --obj_name fetch\
 --num_scenes 2 --scene_samples 300\
 --sample_radius 3.57 \
 --output_dir /media/stanlew/Data/narf_urdf_data/v1

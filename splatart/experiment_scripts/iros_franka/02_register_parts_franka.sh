declare -a objects=("franka_seg") 
declare -a num_classes=(12)
declare splats_base_dir="results/iros_splatart/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a times=(0 1 2 3)
current_dir="$(pwd)"

python splatart/scripts/base/02_learn_poses.py \
    --splat_tf_manager_pths ${splats_base_dir}/${objects[$i]}/seg_learned_manager_0.pth,${splats_base_dir}/${objects[$i]}/seg_learned_manager_1.pth \
    --splat_model_datasets ${data_base_dir}/${objects[0]}/transforms_0.json,${data_base_dir}/${objects[0]}/transforms_1.json,${data_base_dir}/${objects[0]}/transforms_2.json
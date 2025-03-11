declare splats_base_dir="results/iros_splatart_robots" #/home/stanlew/src/nerfstudio_splatart/outputs/"
declare data_base_dir="/media/stanlew/Data/narf_urdf_data/v1"

declare -a objects=("panda")
declare -a num_classes=(12)
declare -a parts_to_combine=("1")

declare -a objects=("fanuc")
declare -a num_classes=(12)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        python splatart/scripts/base/02a_seed_icp_poses.py \
            --splat_tf_manager_pths ${splats_base_dir}/${objects[$i]}/seg_learned_manager_0.pth,${splats_base_dir}/${objects[$i]}/seg_learned_manager_1.pth \
            --splat_model_datasets ${data_base_dir}/${objects[$i]}/0/transforms.json,${data_base_dir}/${objects[$i]}/1/transforms.json
    done
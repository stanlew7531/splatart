declare splats_base_dir="results/iros_splatart_sapien" #/home/stanlew/src/nerfstudio_splatart/outputs/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a num_classes=(3 3 3 3 3 3 3 3 3 3)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        python splatart/scripts/base/02_learn_poses.py \
            --splat_tf_manager_pths ${splats_base_dir}/${objects[$i]}/seg_learned_manager_0.pth,${splats_base_dir}/${objects[$i]}/seg_learned_manager_1.pth \
            --splat_model_datasets ${data_base_dir}/master/${objects[$i]}/0/transforms.json,${data_base_dir}/master/${objects[$i]}/1/transforms.json
    done
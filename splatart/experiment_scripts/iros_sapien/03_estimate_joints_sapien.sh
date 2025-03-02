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
        python splatart/scripts/base/03_learn_joints.py \
            --part_splats ${splats_base_dir}/${objects[$i]}/part_gauss_params.pth \
            --pose_estimates ${splats_base_dir}/${objects[$i]}/pose_estimator.pth \
            --output_dir ${splats_base_dir}/${objects[$i]}/
    done
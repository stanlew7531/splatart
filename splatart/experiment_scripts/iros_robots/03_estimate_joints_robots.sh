declare splats_base_dir="results/iros_splatart_robots" #/home/stanlew/src/nerfstudio_splatart/outputs/"
declare data_base_dir="/media/stanlew/Data/narf_urdf_data/v1"

declare -a objects=("fanuc")
declare -a num_classes=(12)

declare -a objects=("panda")
declare -a num_classes=(12)
declare -a parts_to_combine=("1")



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
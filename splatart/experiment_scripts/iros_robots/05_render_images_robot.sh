declare splats_base_dir="results/iros_splatart_robots" #/home/stanlew/src/nerfstudio_splatart/outputs/"
declare data_base_dir="/media/stanlew/Data/narf_urdf_data/v1"


declare -a objects=("fanuc")
declare -a num_classes=(11)

declare -a objects=("panda")
declare -a num_classes=(12)

declare -a parts_to_combine=("1")

declare -a root_part_ids=(1 )

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        python splatart/scripts/base/05_render_images_robots.py \
            --object_name ${objects[$i]} \
            --manager_paths ${splats_base_dir}/${objects[$i]}/seg_learned_manager_0.pth,${splats_base_dir}/${objects[$i]}/seg_learned_manager_1.pth \
            --part_splats ${splats_base_dir}/${objects[$i]}/part_gauss_params.pth \
            --pose_estimates ${splats_base_dir}/${objects[$i]}/pose_estimator.pth \
            --articulation_estimates ${splats_base_dir}/${objects[$i]}/configuration_vector.pkl \
            --output_dir ${splats_base_dir}/${objects[$i]} \
            --root_part_id ${root_part_ids[$i]} \
            --splat_model_datasets ${data_base_dir}/${objects[$i]}/0/transforms.json,${data_base_dir}/${objects[$i]}/1/transforms.json
    done
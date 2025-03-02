declare splats_base_dir="results/iros_splatart_sapien" #/home/stanlew/src/nerfstudio_splatart/outputs/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a num_classes=(3 3 3 3 3 3 3 3 3 3)

declare -a static_obj_ids=(1 1 1 1 1 1 1 1 1 1 1)
declare -a dyn_part_ids=(2 2 2 2 2 2 2 2 2 2 2)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        python splatart/scripts/base/05_render_images.py \
            --object_name ${objects[$i]} \
            --manager_paths ${splats_base_dir}/${objects[$i]}/seg_learned_manager_0.pth,${splats_base_dir}/${objects[$i]}/seg_learned_manager_1.pth \
            --part_splats ${splats_base_dir}/${objects[$i]}/part_gauss_params.pth \
            --pose_estimates ${splats_base_dir}/${objects[$i]}/pose_estimator.pth \
            --articulation_estimates ${splats_base_dir}/${objects[$i]}/configuration_vector.pkl \
            --output_dir ${splats_base_dir}/${objects[$i]} \
            --static_part_id ${static_obj_ids[$i]} \
            --dyn_part_id ${dyn_part_ids[$i]} \
            --splat_model_datasets ${data_base_dir}/master/${objects[$i]}/0/transforms.json,${data_base_dir}/master/${objects[$i]}/1/transforms.json
    done
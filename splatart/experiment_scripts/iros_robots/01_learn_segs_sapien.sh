declare splats_base_dir="/home/stanlew/src/nerfstudio_splatart/outputs/" #iros_${objects[0]}/splatfacto/"
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
        python splatart/scripts/base/01a_preprocess_segs.py \
            --splat_model_datasets ${data_base_dir}/${objects[$i]}/0/transforms.json,${data_base_dir}/${objects[$i]}/1/transforms.json \
            --n_parts ${num_classes[$i]} \
            --parts_to_combine ${parts_to_combine[$i]}
        python splatart/scripts/base/01_learn_segs.py \
            --splat_tf_manager_pths ${splats_base_dir}/iros_robots_${objects[$i]}/splatfacto/config_0,${splats_base_dir}/iros_robots_${objects[$i]}/splatfacto/config_1 \
            --splat_model_datasets ${data_base_dir}/${objects[$i]}/0/transforms.json,${data_base_dir}/${objects[$i]}/1/transforms.json \
            --n_parts ${num_classes[$i]} --output_dir results/iros_splatart_robots/ --exp_name ${objects[$i]}
    done
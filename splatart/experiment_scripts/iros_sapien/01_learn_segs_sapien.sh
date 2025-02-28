declare splats_base_dir="/home/stanlew/src/nerfstudio_splatart/outputs/" #iros_${objects[0]}/splatfacto/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a num_classes=(4 4 4 4 11 4 5 6 4 7)

declare -a objects=("stapler")
declare -a num_classes=(5)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        python splatart/scripts/base/01_learn_segs.py \
            --splat_tf_manager_pths ${splats_base_dir}/iros_sapien_${objects[$i]}/splatfacto/config_0,${splats_base_dir}/iros_sapien_${objects[$i]}/splatfacto/config_1 \
            --splat_model_datasets ${data_base_dir}/master/${objects[$i]}/0/transforms.json,${data_base_dir}/master/${objects[$i]}/1/transforms.json \
            --n_parts ${num_classes[$i]} --output_dir results/iros_splatart_sapien/ --exp_name ${objects[$i]}
    done
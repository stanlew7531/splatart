declare splats_base_dir="/home/stanlew/src/nerfstudio_splatart/outputs/" #iros_${objects[0]}/splatfacto/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a num_classes=(3 3 3 3 10 3 4 5 3 6)
declare -a parts_to_combine=("1" "1" "1" "1" "1,3,4,5,6,7,8,9" "1" "2,3" "1" "1" "2,3,4,5")


declare -a objects=("oven")
declare -a num_classes=(3)
declare -a parts_to_combine=("1,3,4,5,6,7,8,9")

declare -a objects=("washer" "stapler")
declare -a num_classes=(3 3)
declare -a parts_to_combine=("1,3,4,5" "2,3")




declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        python splatart/scripts/base/01a_preprocess_segs.py \
            --splat_model_datasets ${data_base_dir}/master/${objects[$i]}/0/transforms.json,${data_base_dir}/master/${objects[$i]}/1/transforms.json \
            --n_parts ${num_classes[$i]} \
            --parts_to_combine ${parts_to_combine[$i]}
        python splatart/scripts/base/01_learn_segs.py \
            --splat_tf_manager_pths ${splats_base_dir}/iros_sapien_${objects[$i]}/splatfacto/config_0,${splats_base_dir}/iros_sapien_${objects[$i]}/splatfacto/config_1 \
            --splat_model_datasets ${data_base_dir}/master/${objects[$i]}/0/transforms.json,${data_base_dir}/master/${objects[$i]}/1/transforms.json \
            --n_parts ${num_classes[$i]} --output_dir results/iros_splatart_sapien/ --exp_name ${objects[$i]}
    done
declare splats_base_dir="/home/stanlew/Repos/splatart/outputs"
declare data_base_dir="/home/stanlew/Data/icra_sapien_data/v0"

declare -a objects=(\
"sapien_35059" \
"sapien_38516" \
"sapien_40147" \
"sapien_40417" \
"sapien_40453" \
"sapien_41003" \
"sapien_41004" \
"sapien_41083" \
"sapien_41085" \
"sapien_41086" \
"sapien_41452" \
"sapien_41510" \
"sapien_41529" \
"sapien_44781" \
"sapien_44817" \
"sapien_44826" \
"sapien_44853" \
"sapien_44962" \
"sapien_45001" \
"sapien_45007" \
)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        num_parts=$(python splatart/scripts/00c_sapien_data_getClasses.py --transforms_json ${data_base_dir}/${objects[$i]}/0/transforms.json)
        echo "${num_parts}"
        python splatart/scripts/base/01_learn_segs.py \
            --splat_tf_manager_pths ${splats_base_dir}/icra_${objects[$i]}/splatfacto/config_0,${splats_base_dir}/icra_${objects[$i]}/splatfacto/config_1,${splats_base_dir}/icra_${objects[$i]}/splatfacto/config_2,${splats_base_dir}/icra_${objects[$i]}/splatfacto/config_3,${splats_base_dir}/icra_${objects[$i]}/splatfacto/config_4 \
            --splat_model_datasets ${data_base_dir}/${objects[$i]}/0/transforms.json,${data_base_dir}/${objects[$i]}/1/transforms.json,${data_base_dir}/${objects[$i]}/2/transforms.json,${data_base_dir}/${objects[$i]}/3/transforms.json,${data_base_dir}/${objects[$i]}/4/transforms.json \
            --n_parts ${num_parts} --output_dir results/icra_workshop/ --exp_name ${objects[$i]}
    done
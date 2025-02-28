declare -a objects=("franka_seg") 
declare -a num_classes=(12)
declare splats_base_dir="/home/stanlew/src/nerfstudio_splatart/outputs/iros_${objects[0]}/splatfacto/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a times=(0 1 2 3)

python splatart/scripts/base/01_learn_segs.py \
 --splat_tf_manager_pths ${splats_base_dir}/config_0,${splats_base_dir}/config_1,${splats_base_dir}/config_2,${splats_base_dir}/config_3\
 --splat_model_datasets ${data_base_dir}/${objects[0]}/transforms_0.json,${data_base_dir}/${objects[0]}/transforms_1.json,${data_base_dir}/${objects[0]}/transforms_2.json,${data_base_dir}/${objects[0]}/transforms_3.json\
  --n_parts ${num_classes} --output_dir results/iros_splatart/ --exp_name ${objects[$i]}


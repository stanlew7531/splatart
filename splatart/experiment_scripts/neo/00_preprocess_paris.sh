declare -a paris_base_folder=("/media/stanlew/Data/paris_dataset/dataset/load/")
declare -a output_base_folder=("/media/stanlew/Data/splatart_neo_data/")
declare -a preprocess_script=("splatart/scripts/data_processing/paris_to_nerfstudio.py")

python $preprocess_script --paris_input_folder ${paris_base_folder}/realscan/real_fridge/start --output_folder ${output_base_folder}/realscan/fridge/start
python $preprocess_script --paris_input_folder ${paris_base_folder}/realscan/real_fridge/end   --output_folder ${output_base_folder}/realscan/fridge/end
python $preprocess_script --paris_input_folder ${paris_base_folder}/realscan/real_storage/start --output_folder ${output_base_folder}/realscan/storage/start
python $preprocess_script --paris_input_folder ${paris_base_folder}/realscan/real_storage/end   --output_folder ${output_base_folder}/realscan/storage/end

declare -a sapien_objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a sapien_object_ids=("103706" "102255" "10905" "10211" "101917" "11100" "103111" "45135" "100109" "103776")

for i in "${!sapien_objects[@]}"
    do
        python $preprocess_script --paris_input_folder ${paris_base_folder}/sapien/${sapien_objects[$i]}/${sapien_object_ids[$i]}/start --output_folder ${output_base_folder}/sapien/${sapien_objects[$i]}/start
        python $preprocess_script --paris_input_folder ${paris_base_folder}/sapien/${sapien_objects[$i]}/${sapien_object_ids[$i]}/end   --output_folder ${output_base_folder}/sapien/${sapien_objects[$i]}/end
    done
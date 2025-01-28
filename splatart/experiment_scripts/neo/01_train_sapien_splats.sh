declare -a paris_base_folder=("/media/stanlew/Data/paris_dataset/dataset/load/")
declare -a output_base_folder=("/media/stanlew/Data/splatart_neo_data/")
declare -a preprocess_script=("splatart/scripts/data_processing/paris_to_nerfstudio.py")

declare -a sapien_objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a sapien_object_ids=("103706" "102255" "10905" "10211" "101917" "11100" "103111" "45135" "100109" "103776")

for i in "${!sapien_objects[@]}"
    do
        cd ~/src/nerfstudio_splatart
        ns-train splatfacto --data ${output_base_folder}/sapien/${sapien_objects[$i]}/start/transforms_train.json --experiment_name splatart_neo --timestamp sapien_${sapien_objects[$i]}_start --vis=viewer+tensorboard --pipeline.model.random_scale 0.1 --pipeline.model.camera_optimizer.mode off --pipeline.model.continue_cull_post_densification=False --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.use_scale_regularization True --viewer.quit-on-train-completion True
        ns-train splatfacto --data ${output_base_folder}/sapien/${sapien_objects[$i]}/end/transforms_train.json --experiment_name splatart_neo --timestamp sapien_${sapien_objects[$i]}_end --vis=viewer+tensorboard --pipeline.model.random_scale 0.1 --pipeline.model.camera_optimizer.mode off --pipeline.model.continue_cull_post_densification=False --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.use_scale_regularization True --viewer.quit-on-train-completion True
    done
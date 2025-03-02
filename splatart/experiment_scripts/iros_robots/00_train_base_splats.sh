declare -a objects=("panda")
declare -a num_classes=(12)

declare -a objects=("fanuc")
declare -a num_classes=(12)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        
        echo "${num_classes[$i]}"
        for j in "${times[@]}"
            do
                echo "$j"
                cd ~/src/nerfstudio_splatart
                ns-train splatfacto\
                 --data /media/stanlew/Data/narf_urdf_data/v1/${objects[$i]}/$j/transforms.json \
                 --experiment_name iros_robots_${objects[$i]} --timestamp config_$j --vis=viewer+tensorboard \
                 --max-num-iterations 20000 --viewer.quit-on-train-completion True \
                 --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3\
                 --pipeline.model.random_scale 0.5 --pipeline.model.camera_optimizer.mode off \
                 --pipeline.model.continue_cull_post_densification False --pipeline.model.use_scale_regularization True
            done
            cd $current_dir
    done
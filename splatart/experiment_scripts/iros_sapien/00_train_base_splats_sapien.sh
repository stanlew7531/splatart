declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a num_classes=(3 3 3 3 10 3 4 5 3 6)

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
                 --data /home/stanlew/Desktop/narf_iros_data/master/${objects[$i]}/$j/transforms.json \
                 --experiment_name iros_sapien_${objects[$i]} --timestamp config_$j --vis=viewer+tensorboard \
                 --max-num-iterations 20000 --viewer.quit-on-train-completion True \
                 --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3\
                 --pipeline.model.random_scale 0.5 --pipeline.model.camera_optimizer.mode off \
                 --pipeline.model.continue_cull_post_densification False --pipeline.model.use_scale_regularization True\
                 &

                #  --pipeline.model.random_scale 0.5 --pipeline.model.camera_optimizer.mode off\
                #  --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3\
                #  --pipeline.model.continue_cull_post_densification False --pipeline.model.use_scale_regularization True\
            done
            cd $current_dir
    done
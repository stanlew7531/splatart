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
        for j in "${times[@]}"
            do
                echo "$j"
                cd ~/src/nerfstudio_splatart
                ns-train splatfacto --data /home/stanlew/Desktop/narf_iros_data/master/${objects[$i]}/$j/transforms.json --experiment_name iros_sapien_${objects[$i]} --timestamp config_$j --vis=viewer+tensorboard --pipeline.model.random_scale 0.1 --pipeline.model.camera_optimizer.mode off --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3 --max-num-iterations 20000 --viewer.quit-on-train-completion True
            done
            cd $current_dir
    done

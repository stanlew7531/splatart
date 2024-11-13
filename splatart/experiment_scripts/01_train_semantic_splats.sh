#declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
#declare -a num_classes = (4 4 4 4 11 4 5 6 4 7)

# declare -a objects=("blade" "foldchair" "fridge")
# declare -a num_classes=(4 4 4)

# declare -a objects=("laptop" "oven" "scissor")
# declare -a num_classes=(4 11 4)

# declare -a objects=("stapler" "storage" "USB")
# declare -a num_classes=(5 6 4)


# declare -a objects=("blade" "washer" "stapler" "oven")
# declare -a num_classes=(4 7 5 11)

declare -a objects=("USB")
declare -a num_classes=(4)

declare -a times=(0 1)


for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        for j in "${times[@]}"
            do
                echo "$j"
                cd ~/src/nerfstudio_splatart
                ns-train semantic-splatfacto --data /media/stanlew/Data/narf_sapien_data/v5/${objects[$i]}/$j/transforms.json --experiment_name exp_sapien_${objects[$i]} --timestamp config_$j --vis=viewer+tensorboard --pipeline.model.num_classes ${num_classes[$i]} --pipeline.model.random_scale 0.1 --pipeline.model.camera_optimizer.mode off --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3 --max-num-iterations 20000 --viewer.quit-on-train-completion True
            done
    done

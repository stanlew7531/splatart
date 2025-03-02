declare -a objects=("franka_seg") 
declare -a num_classes=(12)
declare base_dir="/home/stanlew/Desktop/narf_iros_data/"

declare -a times=(0 1 2 3)

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
                ns-train splatfacto-big --data ${base_dir}/${objects[$i]}/transforms_$j.json --experiment_name iros_${objects[$i]} --timestamp config_$j --vis=viewer+tensorboard --pipeline.model.camera_optimizer.mode off --max-num-iterations 20000 --viewer.quit-on-train-completion True --pipeline.model.background_color random --pipeline.model.random_scale 0.1 --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3 --pipeline.model.background_color random --pipeline.model.cull_screen_size 0.5 --pipeline.model.cull_scale_thresh 1.3
            done
    done

cd $current_dir

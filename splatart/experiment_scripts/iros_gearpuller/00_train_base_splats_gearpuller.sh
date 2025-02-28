declare -a objects=("gearpuller") 
declare -a num_classes=(9)
declare base_dir="/home/stanlew/Desktop/narf_iros_data/master"

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
                ns-train splatfacto-big --data ${base_dir}/${objects[$i]}/$j/transforms.json --experiment_name iros_${objects[$i]} --timestamp config_$j --vis=viewer+tensorboard --pipeline.model.camera_optimizer.mode off --max-num-iterations 20000 --viewer.quit-on-train-completion True --pipeline.model.background_color white
            done
    done

cd $current_dir

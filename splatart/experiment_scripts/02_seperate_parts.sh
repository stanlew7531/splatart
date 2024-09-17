# declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
# declare -a num_classes = (4 4 4 4 11 4 5 6 4 7)

# declare -a objects=("blade" "foldchair" "fridge" "laptop" "scissor" "stapler" "storage" "USB" "washer")
# declare -a num_classes=(4 4 4 4 4 5 6 4 7)

# declare -a objects=("blade" "foldchair" "fridge")
# declare -a num_classes=(4 4 4)

# declare -a objects=("scissor" "stapler" "storage")
# declare -a num_classes=(4 5 6)

# declare -a objects=("laptop" "oven" "USB" "washer")
# declare -a num_classes=(4 11 4 7)

# declare -a objects=("blade" "fridge" "laptop" "scissor" "storage" "USB" "oven")
# declare -a num_classes=(4 4 4 4 6 4 4)


declare -a objects=("oven" "storage")
declare -a num_classes=(11 6)

declare -a times=(0 1)

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        for j in "${times[@]}"
            do
                echo "$j"
                python splatart/scripts/01_seperate_parts.py --input_model_dirs /home/stanlew/src/nerfstudio_splatart/outputs/exp_sapien_${objects[$i]}/semantic-splatfacto/config_0,/home/stanlew/src/nerfstudio_splatart/outputs/exp_sapien_${objects[$i]}/semantic-splatfacto/config_1 --num_classes ${num_classes[$i]} --output_dir ./results/sapien_exp/${objects[$i]}
            done
    done

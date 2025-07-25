# declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
# declare -a num_classes=(4 4 4 4 4 4 4 6 4 7)

# declare -a objects=("blade" "foldchair" "fridge")
# declare -a num_classes=(4 4 4)

# declare -a objects=("scissor" "stapler" "storage")
# declare -a num_classes=(4 5 6)

# declare -a objects=("laptop" "oven" "USB" "washer")
# declare -a num_classes=(4 11 4 7)

# declare -a objects=("blade" "fridge" "laptop" "scissor" "storage" "USB" "oven")
# declare -a num_classes=(4 4 4 4 6 4 4)

declare -a objects=("USB")
declare -a num_classes=(4)


# declare -a objects=("blade" "foldchair" "fridge" "laptop" "scissor" "stapler" "USB" "washer" "oven" "storage")
# declare -a num_classes=(4 4 4 4 4 5 4 7 11 6)

# declare -a objects=("USB")
# declare -a num_classes=(4)

declare -a times=(0 1)

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        echo "${num_classes[$i]}"
        python splatart/scripts/03b_estimate_joints.py --input_model_dirs /home/stanlew/src/nerfstudio_splatart/outputs/exp_sapien_${objects[$i]}/semantic-splatfacto/config_0,/home/stanlew/src/nerfstudio_splatart/outputs/exp_sapien_${objects[$i]}/semantic-splatfacto/config_1 --num_classes ${num_classes[$i]} --canonical_model_dataset /media/stanlew/Data/narf_sapien_data/v5/${objects[$i]}/0/transforms.json --output_dir ./results/sapien_exp/${objects[$i]}
    done

declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
# declare -a objects=("fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a times=(0)

for i in "${objects[@]}"
    do
        echo "$i"
        for j in "${times[@]}"
            do
                python splatart/scripts/00c_sapien_data_getClasses.py --transforms_json /media/stanlew/Data/narf_sapien_data/v3/$i/$j/transforms.json    
            done
    done

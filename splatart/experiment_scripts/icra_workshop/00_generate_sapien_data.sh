declare -a sapien_ids=(\
"35059" \
"38516" \
"40147" \
"40417" \
"40453" \
"41003" \
"41004" \
"41083" \
"41085" \
"41086" \
"41452" \
"41510" \
"41529" \
"44781" \
"44817" \
"44826" \
"44853" \
"44962" \
"45001" \
"45007" \
)

declare num_scenes=5

for i in "${!sapien_ids[@]}"
    do
        echo "writing to /home/stanlew/Data/icra_sapien_data/v0/sapien_${sapien_ids[$i]}/"
        python splatart/scripts/00_sapien_datagen.py \
        --obj_id "${sapien_ids[$i]}" --obj_name sapien_${sapien_ids[$i]} \
        --num_scenes $num_scenes --scene_samples 100 \
        --output_dir /home/stanlew/Data/icra_sapien_data/v0/
    done


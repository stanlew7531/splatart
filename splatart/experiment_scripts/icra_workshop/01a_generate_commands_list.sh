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
declare commands_fname="splatart/experiment_scripts/icra_workshop/01a_commands.txt"
rm $commands_fname
touch $commands_fname

for i in "${!sapien_ids[@]}"
    do
        for j in $(seq 0 $(( $num_scenes - 1 )))
            do
                echo "(cd ~/Repos/splatart; ns-train splatfacto --data ~/Data/icra_sapien_data/v0/sapien_${sapien_ids[$i]}/$j/transforms.json --experiment_name icra_sapien_${sapien_ids[$i]} --timestamp config_$j --vis=viewer+tensorboard --pipeline.model.camera_optimizer.mode off --max-num-iterations 10000 --viewer.quit-on-train-completion True )" >> $commands_fname
            done
    done
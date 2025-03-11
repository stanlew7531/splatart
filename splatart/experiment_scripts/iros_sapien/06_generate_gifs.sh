declare splats_base_dir="results/iros_splatart_sapien" #/home/stanlew/src/nerfstudio_splatart/outputs/"
declare data_base_dir="/home/stanlew/Desktop/narf_iros_data"

declare -a objects=("blade" "foldchair" "fridge" "laptop" "oven" "scissor" "stapler" "storage" "USB" "washer")
declare -a num_classes=(3 3 3 3 3 3 3 3 3 3)

declare -a static_obj_ids=(1 1 1 1 1 1 1 1 1 1 1)
declare -a dyn_part_ids=(2 2 2 2 2 2 2 2 2 2 2)

declare -a times=(0 1)
current_dir="$(pwd)"

for i in "${!objects[@]}"
    do
        echo "$i"
        echo "${objects[$i]}"
        convert -delay 10 -loop 0 ./results/iros_splatart_sapien/${objects[$i]}/rendered_image_*.png ./results/iros_splatart_sapien/${objects[$i]}/video.gif
        cp ./results/iros_splatart_sapien/${objects[$i]}/video.gif ~/Desktop/narf_iros_data/video_${objects[$i]}.gif
    done
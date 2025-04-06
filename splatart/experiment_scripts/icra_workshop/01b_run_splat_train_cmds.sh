declare commands_fname="splatart/experiment_scripts/icra_workshop/01a_commands.txt"

cat $commands_fname | xargs -I CMD --max-procs 5 bash -c CMD
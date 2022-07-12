# -*- coding: utf-8 -*-
# Bash script to automate the procedure of evaluating an experiment.
# The script is taking as granted that the experiments are saved under "base_path" as "base_path/exp$EXP_NUM".
# First, -for each split- get the training loss from tensorboard as csv file. Then, compute the fscore (txt file)
# for the current experiment. Finally, based ONLY on the training loss choose the best epoch (and model).
base_path="/data/project/rw/video_summarization/PGL-SUM/Summaries/PGL-SUM"
expr_name=$1 #exp1, exp2, exp3, exp4
dataset=$2 #tvsum, summe
data_file=$3 # h5 file name
set_id=$4

exp_path="$base_path/${expr_name}_try${set_id}"; echo $exp_path
path="$exp_path/$dataset/logs/$data_file"
python evaluation/exportTensorFlowLog.py $path $path
results_path="$exp_path/$dataset/results/$data_file"
python evaluation/compute_fscores.py --path $results_path --expr_name $expr_name --data_file $data_file --dataset $dataset --set_id $set_id
python evaluation/choose_best_epoch.py $exp_path $dataset $data_file
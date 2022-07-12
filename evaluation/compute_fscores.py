# -*- coding: utf-8 -*-
from os import listdir
import json
import numpy as np
import h5py
from evaluation_metrics import evaluate_summary, coverage_count
from generate_summary import generate_summary
import argparse
import pandas as pd 

# arguments to run the script
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    default='../PGL-SUM/Summaries/PGL-SUM/exp1/summe/results/vip_summe_inorder_length_25_9000',
                    help="Path to the json files with the scores of the frames for each epoch")
parser.add_argument("--dataset", type=str, default='summe', choices=['summe', 'tvsum'], help="Dataset to be used")
parser.add_argument("--eval", type=str, default="max", help="Eval method to be used for f_score reduction (max or avg)")
parser.add_argument("--expr_name", type=str)
parser.add_argument("--data_file", type=str)
parser.add_argument("--set_id", type=str, default=None)

args = vars(parser.parse_args())
path = args["path"]
dataset = args["dataset"]
expr_name = args["expr_name"]
data_file = args["data_file"]
set_id = args["set_id"]


if dataset == "tvsum":
    eval_method = "avg"
elif dataset == "summe":
    eval_method = "max"
else:
    raise NotImplementedError()

dir_path = [
            '/data/project/rw/video_summarization/dataset/exp1_Order',
            '/data/project/rw/video_summarization/dataset/exp2_ConcatRatio_and_Type',
            '/data/project/rw/video_summarization/dataset/exp3_VideoLength',
            '/data/project/rw/video_summarization/dataset/exp4_Importance_focus',
            '/data/project/rw/video_summarization/dataset/exp5_focus_length',
            '/data/project/rw/video_summarization/dataset/exp6_Diversity',
            '/data/project/rw/video_summarization/dataset/exp7_Importance_focus'
        ]

if expr_name == "exp1":
    base_dir = dir_path[0]
elif expr_name == "exp2":
    base_dir = dir_path[1]
elif expr_name == "exp3":
    base_dir = dir_path[2]
elif expr_name == "exp4":
    base_dir = dir_path[3]
elif expr_name == "exp5":
    base_dir = dir_path[4]
elif expr_name == "exp6":
    base_dir = f'{dir_path[5]}/exp6_Diversity_try{set_id}'
elif expr_name == "exp7":
    base_dir = f'{dir_path[6]}/exp7_try{set_id}'
else:
    raise NotImplementedError("only implemented exp1, exp2, exp3, exp4, exp5, exp6, exp7")

results = [f for f in listdir(path) if f.endswith(".json")]
results.sort(key=lambda video: int(video[6:-5]))
dataset_path = f'{base_dir}/{data_file}.h5'
# json_dataset_path = f'{base_dir}/{data_file}.json'
# with open(json_dataset_path, 'r') as f:
    # json_data_file = json.load(f)
# all_length_ratio =[items['length_ratio'] for key, items in json_data_file.items()]

print(f'loading h5 file... : {dataset_path}')
f_score_epochs = []
df = pd.DataFrame()

max_fscore = 0. #!
max_fscore_epoch = 0 #!

for epoch in results:                       # for each epoch ...
    all_scores = []
    with open(path + '/' + epoch) as f:     # read the json file ...
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:             # for each video inside that json file ...
            scores = np.asarray(data[video_name])  # read the importance scores from frames
            all_scores.append(scores) #! 이게 model output

    all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
    all_sum_ratio, all_video_boundary = [], []
    all_video_name = []
    with h5py.File(dataset_path, 'r') as hdf:
        for video_name in keys:
            video_index = video_name[6:]

            user_summary = np.array(hdf.get('video_' + video_index + '/user_summary'))
            sb = np.array(hdf.get('video_' + video_index + '/change_points'))
            n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
            positions = np.array(hdf.get('video_' + video_index + '/picks'))
            video_boundary = np.array(hdf.get('video_' + video_index + '/video_boundary')) # (4,)
            sum_ratio = np.array(hdf.get('video_' + video_index + '/sum_ratio')) # (n_users, )
            # length_ratio = json_data_file[video_name]['length_ratio']
        
            all_video_name.append(video_name)
            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)
            all_video_boundary.append(video_boundary)
            all_sum_ratio.append(sum_ratio) # (n_videos, n_users, )
            # all_length_ratio.append(length_ratio)

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions, all_sum_ratio)
    # print(f'L#68 all summaries shape? (this is in `compute_fscores.py`) : {all_summaries.shape}\t hopefully (n_videos, n_users, ___)')

    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index, video_name in enumerate(all_video_name):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        video_boundary = all_video_boundary[video_index]
        # length_ratio = all_length_ratio[video_index]
        #! new parts... 
        f_scores = []
        for user_id in range(user_summary.shape[0]):
            f_score = evaluate_summary(summary[user_id], user_summary[user_id])
            f_scores.append(f_score)
            # if epoch == results[-1]:
            #     coverage = coverage_count(video_name, user_id, summary[user_id], user_summary[user_id], video_boundary, sum_ratio[user_id])
            #     df = df.append(coverage, ignore_index=True)
        if eval_method == 'max':
            all_f_scores.append(max(f_scores))
        else:
            all_f_scores.append(sum(f_scores)/len(f_scores))
    
    f_score_epochs.append(np.mean(all_f_scores))
    current_f_score = np.mean(all_f_scores)
    # print(f"epoch {epoch}\tf_score: {current_f_score}")

    if max_fscore < current_f_score:
        max_fscore = current_f_score
        max_fscore_epoch = epoch
        best_all_video_name, best_summ, best_user_summ, best_all_video_boundary, best_all_sum_ratio = all_video_name, all_summaries, all_user_summary, all_video_boundary, all_sum_ratio
"""
coverage count based on base model
"""
for video_index, video_name in enumerate(best_all_video_name):
    summary = best_summ[video_index]
    user_summary = best_user_summ[video_index]
    video_boundary = best_all_video_boundary[video_index]
    sum_ratio = best_all_sum_ratio[video_index]
    # length_ratio = all_length_ratio[video_index]
    #! new parts... 
    for user_id in range(user_summary.shape[0]):
        coverage = coverage_count(video_name, user_id, summary[user_id], user_summary[user_id], video_boundary, sum_ratio[user_id])
        df = df.append(coverage, ignore_index=True)

n_videos = len(video_boundary)
# print(f"\nn_videos : {n_videos}")
list_for_sort = ['video_id', 'user_id', 'sum_ratio']
for i in range(1, n_videos+1):
    tmp_list = [f'v{i}_frames', f'v{i}_pred_frames', f'v{i}_gt_frames', f'v{i}_n_overlap', f'v{i}_overlap_ratio', f'v{i}_pred_sum_ratio', f'v{i}_gt_sum_ratio']
    list_for_sort.extend(tmp_list)

df = df[list_for_sort]
df.to_csv(f"{path}/best_epoch_results.csv", index=False)

# Save the importance scores in txt format.
with open(path + '/f_scores.txt', 'w') as outfile:
    json.dump(f_score_epochs, outfile)

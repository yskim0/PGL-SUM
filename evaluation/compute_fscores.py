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

args = vars(parser.parse_args())
path = args["path"]
dataset = args["dataset"]
expr_name = args["expr_name"]
data_file = args["data_file"]


if dataset == "tvsum":
    eval_method = "avg"
elif dataset == "summe":
    eval_method = "max"
else:
    raise NotImplementedError()

dir_path = [
            '/data/project/rw/video_summarization/dataset/exp1_Order',
            '/data/project/rw/video_summarization/dataset/exp2_ConcatRatio_and_Type',
            '/data/project/rw/video_summarization/dataset/exp3_VideoLength'
        ]
if expr_name == "exp1":
    base_dir = dir_path[0]
elif expr_name == "exp2":
    base_dir = dir_path[1]
elif expr_name == "exp3":
    base_dir = dir_path[2]
else:
    raise NotImplementedError("only implemented exp1, exp2, exp3")

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
                coverage = coverage_count(video_name, user_id, summary[user_id], user_summary[user_id], video_boundary, sum_ratio[user_id])
                df = df.append(coverage, ignore_index=True)
                f_scores.append(f_score)
        if eval_method == 'max':
            all_f_scores.append(max(f_scores))
        else:
            all_f_scores.append(sum(f_scores)/len(f_scores))
    f_score_epochs.append(np.mean(all_f_scores))
    print("f_score: ", np.mean(all_f_scores))

df = df[['video_id', 'user_id', 'sum_ratio', 'v1_frames', 'v1_pred_frames', 'v1_gt_frames', 'v1_n_overlap', 'v1_overlap_ratio', 'v1_pred_sum_ratio', 'v1_gt_sum_ratio', 
'v2_frames', 'v2_pred_frames', 'v2_gt_frames', 'v2_n_overlap', 'v2_overlap_ratio', 'v2_pred_sum_ratio', 'v2_gt_sum_ratio', 
'v3_frames', 'v3_pred_frames', 'v3_gt_frames', 'v3_n_overlap', 'v3_overlap_ratio', 'v3_pred_sum_ratio', 'v3_gt_sum_ratio', 
'v4_frames', 'v4_pred_frames', 'v4_gt_frames', 'v4_n_overlap', 'v4_overlap_ratio', 'v4_pred_sum_ratio', 'v4_gt_sum_ratio']]

df.to_csv(f"{path}/last_epoch_results.csv", index=False)

# Save the importance scores in txt format.
with open(path + '/f_scores.txt', 'w') as outfile:
    json.dump(f_score_epochs, outfile)

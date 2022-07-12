# -*- coding: utf-8 -*-
import torch
import numpy as np
import csv
import json
import sys
import tqdm
import os, glob
from ..model.data_loader import get_loader
from ..model.configs import get_config
from ..model.solver import Solver
from evaluation_metrics import evaluate_summary, coverage_count
from generate_summary import generate_summary
import pandas as pd

''' without args
exp_path = "/data/project/rw/video_summarization/PGL-SUM/Summaries/PGL-SUM/exp1"
dataset = "summe"
data_file = "vip_summe_inorder_length_25_9000"
'''
exp_path = sys.argv[1]
dataset = sys.argv[2]
data_file = sys.argv[3]
df = pd.DataFrame()

def lookup_best_weights_file(data_path):
    """
    best_weights_file -> 
    Summaries/PGL-SUM/exp6_try1/summe/
    models/vip_summe_inorder_length_25_9000/best_score_model.pkl
    """
    weights_filename = os.path.join(data_path, '/models/best_score_model.pkl')
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return 

    # Get the first weights filename in the dir
    weights_filename = weights_filename[0]
    return weights_filename

test_config = get_config(mode='test')
test_loader = get_loader(test_config.mode, test_config.video_type, test_config.expr, test_config.data_file, test_config.set_id)
print(test_config)


weights_filename = lookup_best_weights_file(os.path.join(exp_path, dataset))
print("Loading model:", weights_filename)
best_model = torch.load(weights_filename)
best_model.eval()

eval_method = 'max' if dataset=='summe' else 'avg'
all_f_scores = []

for data in tqdm(test_loader, desc='Evaluate', ncols=80, leave=False):
    frame_features = data['frame_features'].view(-1, test_config.input_size).to(test_config.device)
    with torch.no_grad():
        scores, attn_weights = best_model(frame_features)  # [1, seq_len]
        scores = scores.squeeze(0).cpu().numpy().tolist()
        attn_weights = attn_weights.cpu().numpy()
    
    video_name = data['video_name']
    user_summary = data['user_summary']
    sb = data['change_points']
    n_frames = data['n_frames']
    positions = data['picks']
    video_boundary = data['video_boundary']
    sum_ratio = data['sum_ratio']

    machine_summary = generate_summary(sb, scores, n_frames, positions, sum_ratio)
    print(f'solver L#187 machine_summary shape? : {machine_summary.shape}\t hopefully (n_users,)')
    
    user_f_scores = []
    for user_id in range(user_summary.shape[0]):
        f_score = evaluate_summary(machine_summary[user_id], user_summary[user_id])
        user_f_scores.append(f_score)
        coverage = coverage_count(video_name, user_id, machine_summary[user_id], user_summary[user_id], video_boundary, sum_ratio[user_id])
        df = df.append(coverage, ignore_index=True)
        
    if eval_method == 'max':
        all_f_scores.append(max(user_f_scores))
    else:
        all_f_scores.append(sum(user_f_scores)/len(user_f_scores))

n_videos = len(video_boundary)
list_for_sort = ['video_id', 'user_id', 'sum_ratio']
for i in range(1, n_videos+1):
    tmp_list = [f'v{i}_frames', f'v{i}_pred_frames', f'v{i}_gt_frames', f'v{i}_n_overlap', f'v{i}_overlap_ratio', f'v{i}_pred_sum_ratio', f'v{i}_gt_sum_ratio']
    list_for_sort.extend(tmp_list)

df = df[list_for_sort]
df.to_csv(f"{exp_path}/{dataset}/results/{data_file}/best_epoch_results.csv", index=False)

current_avg_fscore = np.mean(all_f_scores)
print(f'Best model f-score : {current_avg_fscore}')
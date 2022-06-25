# -*- coding: utf-8 -*-
import torch
import numpy as np
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary, coverage_count, calc_coverage_ratio
from layers.summarizer import PGL_SUM
from os import listdir
from os.path import isfile, join
import h5py
import json
import argparse
import pandas as pd


def inference(model, data_path, keys, eval_method):
    """ Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
        the dataset located in `data_path'.

        :param nn.Module model: Pretrained model to be inferenced.
        :param str data_path: File path for the dataset in use.
        :param list keys: Containing the test video keys of the used data split.
        :param str eval_method: The evaluation method in use {SumMe: max, TVSum: avg}
    """
    model.eval()
    video_fscores = []
    all_video_coverage = []
    df = pd.DataFrame()
    for i, video_id in enumerate(keys):
        with h5py.File(data_path, "r") as hdf:
            # Input features for inference
            frame_features = torch.Tensor(np.array(hdf[f"{video_id}/features"])).view(-1, 1024)
            # Input need for evaluation
            user_summary = np.array(hdf[f"{video_id}/user_summary"])
            sb = np.array(hdf[f"{video_id}/change_points"])
            n_frames = np.array(hdf[f"{video_id}/n_frames"])
            positions = np.array(hdf[f"{video_id}/picks"])
            video_boundary = np.array(hdf[f"{video_id}/video_boundary"]) # (4,)
            sum_ratio = np.array(hdf[f"{video_id}/sum_ratio"]) # (n_users, )
        
        # video_coverage = np.zeros(user_summary.shape[0], dtype=list)
        with torch.no_grad():
            scores, _ = model(frame_features)  # [1, seq_len]
            scores = scores.squeeze(0).cpu().numpy().tolist()
            for user_id in range(user_summary.shape[0]):
                summary = generate_summary([sb], [scores], [n_frames], [positions], sum_ratio[user_id])[0]
                f_score = evaluate_summary(summary, user_summary[user_id])
                coverage = coverage_count(video_id, user_id, summary, user_summary[user_id], video_boundary, sum_ratio[user_id])
                df = df.append(coverage, ignore_index=True)
                video_fscores.append(f_score)

                # coverage = calc_coverage_ratio(summary, user_summary[user_id], video_boundary) # for ratio
                # video_coverage[user_id] = coverage # for ratio
            # all_video_coverage.append(video_coverage) # for ratio
    # all_video_coverage = np.array(all_video_coverage) # for ratio

    # print(f"video coverage shape : {all_video_coverage.shape}, it should be (50, 15) for summe")
    print(f"Trained for split: {split_id} achieved an F-score of {np.mean(video_fscores):.2f}%")

    df = df[['vid', 'uid', 'sum_ratio', 's1_n_pred', 's1_n_gt', 's1_n_overlap', 's2_n_pred', 's2_n_gt', 's2_n_overlap', 
    's3_n_pred', 's3_n_gt', 's3_n_overlap', 's4_n_pred', 's4_n_gt', 's4_n_overlap']]
    df.to_csv(f"results_{split_id}.csv", index=False)


if __name__ == "__main__":
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used. Supported: {SumMe, TVSum}")
    parser.add_argument("--table", type=str, default='4', help="Table to be reproduced. Supported: {3, 4}")

    args = vars(parser.parse_args())
    dataset = args["dataset"]
    table = args["table"]

    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    for split_id in range(5):
        # Model data
        model_path = f"../PGL-SUM/inference/pretrained_models/table{table}_models/{dataset}/split{split_id}"
        model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]
        
        # Dataset path
        dataset_path = "/data/project/rw/video_summarization/dataset/vip_dataset_diversity/vip_tvsum_diversity.h5"
        h5 = h5py.File(dataset_path, 'r')
        test_keys = list(h5.keys())

        # Create model with paper reported configuration
        trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute")
        trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))
        inference(trained_model, dataset_path, test_keys, eval_metric)

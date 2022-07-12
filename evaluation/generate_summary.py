# -*- coding: utf-8 -*-
import numpy as np
from knapsack_implementation import knapSack


def generate_summary(shot_bound, scores, nframes, positions, sum_ratio):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    # user_summaries = []
    # user_summaries = np.zeros((len(all_scores), len(all_sum_ratio[0])), dtype=object) #!유저 수 all_sum_ratio.shape[1]이 맞나?
    # user_summaries = []
    summaries = []
    print(f'user_summaries.shape : {user_summaries.shape}, hopefully (n_videos,n_users,)')

    # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
    frame_scores = np.zeros(nframes, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != nframes:
        positions = np.concatenate([positions, [nframes]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(frame_init_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    # final_max_length = int((final_shot[1] + 1) * 0.15)
    for n_user in range(sum_ratio.shape[0]):
        final_max_length = int((final_shot[1] + 1) * sum_ratio[n_user])

        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1
        # user_summaries[video_index][n_user] = summary
        # user_summary_per_video.append(summary)
        summaries.append(summary)
        
    # user_summaries.append(user_summary_per_video)

    return summaries

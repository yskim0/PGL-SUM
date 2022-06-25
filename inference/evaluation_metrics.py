# -*- coding: utf-8 -*-
import numpy as np

def evaluate_summary(predicted_summary, user_summary):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), len(user_summary))
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    G[:len(user_summary)] = user_summary
    overlapped = S & G

    # Compute precision, recall, f-score
    # for safe division
    if sum(S) == 0:
        precision = 0
    else:
        precision = sum(overlapped)/sum(S)
    recall = sum(overlapped)/sum(G)
    if precision+recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall * 100 / (precision + recall)
    return f_score

def coverage_count(vid, uid, predicted_summary, user_summary, video_boundary, sum_ratio):
    """ #TODO
    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), len(user_summary))
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    G[:len(user_summary)] = user_summary

    G_split = np.split(G, video_boundary + 1)[:-1] # last element is empty
    S_split = np.split(S, video_boundary + 1)[:-1] # last element is empty 

    # for dataframe
    raw_data = {}
    raw_data['vid'] = vid
    raw_data['uid'] = int(uid)
    raw_data['sum_ratio'] = sum_ratio

    for i, g_seg in enumerate(G_split):
        s_seg = S_split[i]
        overlapped = s_seg & g_seg
        raw_data[f's{i+1}_n_pred'] = int(sum(s_seg))
        raw_data[f's{i+1}_n_gt'] = int(sum(g_seg))
        raw_data[f's{i+1}_n_overlap'] = int(sum(overlapped))
    
    return raw_data


def calc_coverage_ratio(predicted_summary, user_summary, video_boundary):
    """ 아직 필요 x
    #! 생각해보니 gt가 0개일 때 0개를 골랐는지 아님 다른 걸 골랐는지 보는 것도 중요해서 ratio로 하면 안되겠는데
    #! 왜냐면 ratio=0일 때 이를 판단하기 어렵
    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), len(user_summary))
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    G[:len(user_summary)] = user_summary

    G_split = np.split(G, video_boundary + 1)[:-1] # last element is empty
    S_split = np.split(S, video_boundary + 1)[:-1] # last element is empty 
    coverage_scores = np.zeros(len(G_split), dtype = float)
    for i, g_seg in enumerate(G_split):
        s_seg = S_split[i]
        overlapped = s_seg & g_seg
        # for safe division
        if sum(g_seg) == 0:
            covered = 0
        else:
            covered = sum(overlapped) / sum(g_seg)
        coverage_scores[i] = float(covered)
    return coverage_scores

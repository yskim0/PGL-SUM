# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from functools import partial
# import horovod.torch as hvd


class VideoData(Dataset):
    def __init__(self, mode, video_type, expr_type, h5file_name, set_id=None):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int h5file_name: The name of the Dataset being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.dir_path = [
            '/data/project/rw/video_summarization/dataset/exp1_Order/',
            '/data/project/rw/video_summarization/dataset/exp2_ConcatRatio_and_Type/',
            '/data/project/rw/video_summarization/dataset/exp3_VideoLength/',
            '/data/project/rw/video_summarization/dataset/exp4_Importance_focus/',
            '/data/project/rw/video_summarization/dataset/exp5_focus_length/',
            '/data/project/rw/video_summarization/dataset/exp6_Diversity',
            '/data/project/rw/video_summarization/dataset/exp7_Importance_focus'
        ]
        self.splits_filename = f'/data/project/rw/video_summarization/dataset/{self.name}_splits.json'
        # self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if expr_type == "exp1":
            base_dir = self.dir_path[0]
        elif expr_type == "exp2":
            base_dir = self.dir_path[1]
        elif expr_type == "exp3":
            base_dir = self.dir_path[2]
        elif expr_type == "exp4":
            base_dir = self.dir_path[3]
        elif expr_type == "exp5":
            base_dir = self.dir_path[4]
        elif expr_type == "exp6":
            base_dir = f'{self.dir_path[5]}/exp6_Diversity_try{set_id}/'
        elif expr_type == "exp7":
            base_dir = f'{self.dir_path[6]}/exp7_try{set_id}/'
        else:
            raise NotImplementedError("only implemented exp1~6")
        
        self.filename = base_dir + h5file_name + str('.h5')
        print(f'loading dataset h5 file... : {self.filename}')
        self.video_data = h5py.File(self.filename, 'r')

        with open(self.splits_filename) as f:
            self.data = json.loads(f.read())

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        d['frame_features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))
        d['change_points'] = np.array(self.video_data[video_name + '/change_points'])
        d['n_frames'] = np.array(self.video_data[video_name + '/n_frames'])
        d['n_frame_per_seg'] = np.array(self.video_data[video_name + '/n_frame_per_seg'])
        d['picks'] = np.array(self.video_data[video_name + '/picks'])
        d['user_summary'] = np.array(self.video_data[video_name + '/user_summary'])
        d['sum_ratio'] = np.array(self.video_data[video_name + '/sum_ratio'])
        d['video_boundary'] = np.array(self.video_data[video_name + '/video_boundary'])

        # if self.mode == 'test':
        #     return frame_features, video_name
        # else:
        #     return frame_features, gtscore
        return d


def get_loader(mode, video_type, expr_name, data_file, set_id=None):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, expr_name, data_file, set_id)
        # return DataLoader(vd, shuffle=True, collate_fn=partial(my_collate, mode=mode))
        return DataLoader(vd, shuffle=True)
    
    else:
        vd = VideoData(mode, video_type, expr_name, data_file, set_id)
        # return DataLoader(vd, shuffle=True, collate_fn=partial(my_collate, mode=mode))
        return vd

# def my_collate(batch, mode):
#     if mode == 'test':
#         for frame_features, video_name in batch:
#             features = frame_features.numpy()
#             videos = video_name
#         return torch.Tensor(features), videos

#     elif mode == 'train':
#         features, gtscores = [], []
#         for frame_features, gtscore in batch:
#             features.append(frame_features.numpy())
#             gtscores.append(gtscore.numpy())
#         return torch.Tensor(features), torch.Tensor(gtscores)

if __name__ == '__main__':
    pass

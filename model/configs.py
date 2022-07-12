# -*- coding: utf-8 -*-
import argparse
import torch
from pathlib import Path
import pprint



def str2bool(v):
    """ Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.log_dir, self.score_dir, self.save_dir = None, None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='TVSum'):
        """ Function that sets as class attributes the necessary directories for logging important training information.

        :param str video_type: The Dataset being used, SumMe or TVSum.
        """
        if self.set_id is None:
            save_dir = Path(f'../PGL-SUM/Summaries/PGL-SUM/{self.expr}')
        else:
            save_dir = Path(f'../PGL-SUM/Summaries/PGL-SUM/{self.expr}_try{self.set_id}')
        self.score_dir = save_dir.joinpath(video_type, 'results/' + str(self.data_file))
        self.log_dir = save_dir.joinpath(video_type, 'logs/' + str(self.data_file))
        self.save_dir = save_dir.joinpath(video_type, 'models/' + str(self.data_file))

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """ Get configurations as attributes of class
        1. Parse configurations with argparse.
        2. Create Config class initialized with parsed kwargs.
        3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train', help='Mode for the configuration [train | test]')
    parser.add_argument('--verbose', type=str2bool, default='false', help='Print or not training messages')
    parser.add_argument('--video_type', type=str, default='summe', choices=['summe', 'tvsum'], help='Dataset to be used')
    
    # About Experiments (YS)
    parser.add_argument('--data_file', type=str, default='vip_summe_inorder_length_25_9000', help='Dataset(.h5 file)')
    parser.add_argument('--expr', type=str, default='exp1', choices=['exp1','exp2','exp3', 'exp4', 'exp5', 'exp6', 'exp7'], help='Experiments type (exp1:order, exp2:concat ratio & type, exp3:video len.')
    parser.add_argument('--set_id', type=str, default=None)

    # Model
    parser.add_argument('--input_size', type=int, default=1024, help='Feature size expected in the input')
    parser.add_argument('--seed', type=int, default=12345, help='Chosen seed for generating random numbers')
    parser.add_argument('--fusion', type=str, default="add", help="Type of feature fusion")
    parser.add_argument('--n_segments', type=int, default=4, help='Number of segments to split the video')
    parser.add_argument('--pos_enc', type=str, default="absolute", help="Type of pos encoding [absolute|relative|None]")
    parser.add_argument('--heads', type=int, default=8, help="Number of global heads for the attention module")

    # Train
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Size of each batch in training')
    parser.add_argument('--clip', type=float, default=5.0, help='Max norm of the gradients')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate used for the modules')
    parser.add_argument('--l2_req', type=float, default=1e-5, help='Regularization factor')
    # parser.add_argument('--split_index', type=int, default=0, help='Data split to be used [0-4]')
    parser.add_argument('--init_type', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--init_gain', type=float, default=None, help='Scaling factor for the initialization methods')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()

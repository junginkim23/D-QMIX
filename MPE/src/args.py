import os
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--map_name', default='simple_crypto', type=str)
    parser.add_argument('--env', default='mpe_env', type=str)
    parser.add_argument('--algorithm', default='d_qmix_mpe', type=str)
    parser.add_argument('--t_max', default=5000000, type=int)
    parser.add_argument('--anneal_time', default=50000, type=int)
    parser.add_argument('--seed', default=1011, type=int)
    
    # dqmix 
    parser.add_argument('--multi_step', default=4, type=int)
    parser.add_argument('--num_cur_step', default=8, type=int)
    parser.add_argument('--mac', default='basic_mac_dqmix', type=str)
    parser.add_argument('--attention', default=True, type=str2bool)
    parser.add_argument('--use_global', default=True, type=str2bool)

    args = parser.parse_args()

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
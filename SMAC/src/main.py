import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES']= 'GPU-fc510ea9-4379-ecd0-c6bb-2ff12531d4e3'
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th


from utils.logging import get_logger
import utils.config_util as cu

import yaml
from run import run
from args import get_args


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = cu.config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"] # 1 

    # run the framework
    run(_run, config, _log)


if __name__ == '__main__':
    args = get_args()

    if args.algorithm == 'd_qmix':
        args.local_results_path = os.path.join(os.path.dirname(__file__), "results", "{}/seed_{}/{}/num_cur_step+{}/multi_step+{}/attention_{}+global_{}/coef_{}".format(args.map_name, args.seed, args.algorithm, args.num_cur_step, args.multi_step, args.attention, args.use_global, args.coef))
    else:
        args.local_results_path = os.path.join(os.path.dirname(__file__), "results", "{}/seed_{}/{}".format(args.map_name, args.seed, args.algorithm))

    config_dict = cu.config_copy(cu.get_config(args))

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(args.local_results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(['src/main.py', 'with', f'env_args.map_name={args.map_name}'])

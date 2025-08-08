import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
import sys
sys.path.append('/workspace/jungin/STUDY/D_QMIX_ORIGIN/src/envs/multiagent_particle_envs/multiagent')

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
    
    if config['env'] != 'mpe_env':
        config['env_args']['seed'] = config["seed"] # 1
     

    # ADD NEW
    if config['env'] == 'mpe_env':
        assert config['scenario_name'] in ['simple_tag', 'simple_world', 'simple_adversary', 'simple_crypto']

        config['target_update_interval'] = 800
        if config['scenario_name'] in ['simple_tag', 'simple_world']:
            config['res_lambda'] = 0.05
        elif config['scenario_name'] in ['simple_adversary']:
            config['res_lambda'] = 0.5
        elif config['scenario_name'] in ['simple_crypto']:
            config['res_lambda'] = 0.01

    # run the framework
    run(_run, config, _log)


if __name__ == '__main__':
    args = get_args()

    if args.algorithm == 'd_qmix' or args.algorithm == 'd_qmix_mpe':
        args.local_results_path = os.path.join(os.path.dirname(__file__), "results", "{}/tb_logs_seed_{}_num_cur_step+{}_multi_step+{}_attention_{}+global_{}".format(args.map_name, args.seed, args.num_cur_step, args.multi_step, args.attention, args.use_global))
    else:
        args.local_results_path = os.path.join(os.path.dirname(__file__), "results", "{}/tb_logs_seed_{}".format(args.map_name, args.seed))

    # ADD NEW
    if args.env == 'mpe_env': 
        config_dict = cu.config_copy(cu.get_config_mpe(args))
    else: 
        config_dict = cu.config_copy(cu.get_config(args))

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.") 
    file_obs_path = os.path.join(args.local_results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    if args.env != 'mpe_env':
        ex.run_commandline(['src/main.py', 'with', f'env_args.map_name={args.map_name}'])
    else:
        ex.run_commandline(['src/main.py', 'with', f'map_name={args.map_name}'])

import yaml
import collections
from copy import deepcopy
import os 
from os.path import dirname

def config_copy(config):
    if isinstance(config, dict):

        return {k: config_copy(v) for k, v in config.items()}

    elif isinstance(config, list):

        return [config_copy(v) for v in config]

    else:

        return deepcopy(config)


def recursive_dict_update(d, u):
    for k, v in u.items():

        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)

        else:
            d[k] = v

    return d


def get_config(args): #   / (algorithm, minigame, masking_ratio, anneal_time, t_max, seed, momentum)

    with open(os.path.join(dirname(os.path.dirname(__file__)), "config", "{}.yaml".format('default')), "r") as f:
        try:
            default_config = yaml.load(f, Loader=yaml.SafeLoader)

        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    with open(os.path.join(dirname(os.path.dirname(__file__)), "config", 'envs', "{}.yaml".format('sc2_beta')), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format('sc2', exc)

        env_config = config_dict

    with open(os.path.join(dirname(os.path.dirname(__file__)), "config", 'algs', "{}.yaml".format(args.algorithm)), "r") as f:
        try:
            config_dict_ = yaml.load(f, Loader=yaml.SafeLoader)

        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format('sc2', exc)

        alg_config = config_dict_

    final_config_dict = recursive_dict_update(default_config, env_config)
    final_config_dict = recursive_dict_update(final_config_dict, alg_config)

    final_config_dict['env_args']['map_name'] = args.map_name
    final_config_dict['coef'] = args.coef
    final_config_dict['t_max'] = args.t_max
    final_config_dict['epsilon_anneal_time'] = args.anneal_time
    final_config_dict['seed'] = args.seed
    final_config_dict['num_cur_step'] = args.num_cur_step
    final_config_dict['multi_step'] = args.multi_step
    final_config_dict['mac'] = args.mac
    final_config_dict['local_results_path'] = args.local_results_path
    final_config_dict['attention'] = args.attention
    final_config_dict['use_global'] = args.use_global

    return final_config_dict

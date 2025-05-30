import os
from confuse import Configuration
from typing import Optional
import torch
import logging
from importlib import resources

DEFAULT_CONFIG_DIR_PATH = os.path.join(resources.files('bilevel_optimisation'), 'config_data', 'default')
TYPE_DICT = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}

def load_app_config(app_name: str, custom_config_dir: str, configuring_module: str) -> Configuration:
    path_to_package_custom_config = locate_custom_config_in_package(custom_config_dir)
    if path_to_package_custom_config:
        custom_config_dir_path = path_to_package_custom_config
    else:
        custom_config_dir_path = custom_config_dir

    config = Configuration(app_name)
    for dir_path in [DEFAULT_CONFIG_DIR_PATH, custom_config_dir_path]:
        for file in os.listdir(dir_path):
            logging.info('[{:s}] load configs from {:s} '
                         'in {:s}'.format(configuring_module.upper(), file, dir_path))
            config.set_file(os.path.join(dir_path, file))

    return config

def parse_datatype(config: Configuration) -> Optional[torch.dtype]:
    type_str = config['data']['type'].get()
    dtype = None
    if type_str in TYPE_DICT.keys():
        dtype = TYPE_DICT[type_str]
    return dtype

def locate_custom_config_in_package(subdir_name: str) -> Optional[str]:
    package_root = resources.files('bilevel_optimisation')
    custom_config_dir = os.path.join(package_root, 'config_data', 'custom')
    custom_config_subdir_list = [d for d in os.listdir(custom_config_dir)
                                 if os.path.isdir(os.path.join(custom_config_dir, d))]
    ret_val = None
    if subdir_name in custom_config_subdir_list:
        ret_val = os.path.join(custom_config_dir, subdir_name)

    return ret_val

import os
from confuse import Configuration
from typing import Optional
import torch
import logging
from importlib import resources

TYPE_DICT = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}

def load_app_config(app_name: str, path_to_config_super_dir: str, custom_config_dir: str, configuring_module: str) -> Configuration:
    default_config_dir_path = os.path.join(path_to_config_super_dir, 'default')
    custom_config_dir_path = os.path.join(path_to_config_super_dir, custom_config_dir)
    config = Configuration(app_name)
    for dir_path in [default_config_dir_path, custom_config_dir_path]:
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

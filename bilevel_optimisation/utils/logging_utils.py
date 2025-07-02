import os
import logging
import torch

from bilevel_optimisation.data.LogLevel import LogLevel

def setup_logger(data_dir_path: str = '.', log_level_str: str = 'info') -> None:
    log_level = LogLevel.from_string(log_level_str)
    logging.basicConfig(level=log_level,
                        format='[{asctime}.{msecs:0<3.0f}][{levelname}] {message}',
                        style='{', datefmt='%Y-%m-%dT%H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(data_dir_path, 'application.log'))])

def log_trainable_params_stats(model: torch.nn.Module, logging_module: str) -> None:
    num_params_total = sum([p.numel() for p in model.parameters()])
    num_params_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])

    logging.info('[{:s}] parameter stats'.format(logging_module.upper()))
    logging.info('[{:s}]   > total number of parameters: {:d}'.format(logging_module.upper(), num_params_total))
    logging.info('[{:s}]   > trainable parameters: {:d}'.format(logging_module.upper(), num_params_trainable))
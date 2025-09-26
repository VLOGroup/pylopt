import os
from confuse import Configuration
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

def dump_configs(config: Configuration, path_to_data_dir: str) -> None:
    config_data = config.dump(full=True)
    config_file_path = os.path.join(path_to_data_dir, 'config.yaml')
    with open(config_file_path, 'w') as file:
        file.write(str(config_data))

def dump_bilevel_training_settings(upper_settings: Dict[str, Any], lower_settings: Dict[str, Any],
                                   backward_settings: Dict[str, Any],  path_to_data_dir: str) -> None:
    settings_dict = {'upper_problem': upper_settings, 'lower_problem': lower_settings, 'backward': backward_settings}
    settings_file_path = os.path.join(path_to_data_dir, 'bilevel_training_settings.yaml')
    with open(settings_file_path, 'w') as file:
        yaml.dump(settings_dict, file, sort_keys=False)

def create_experiment_dir(config: Configuration) -> str:
    experiments_root_dir = config['data']['experiments']['root_dir'].get()
    if not experiments_root_dir:
        package_root_path = Path(__file__).resolve().parents[2]
        experiments_root_dir = os.path.join(package_root_path, 'data', 'evaluation')
        os.makedirs(experiments_root_dir, exist_ok=True)

    experiment_list = sorted(os.listdir(experiments_root_dir))
    if experiment_list:
        experiment_id = str(int(experiment_list[-1]) + 1).zfill(5)
    else:
        experiment_id = str(0).zfill(5)
    path_to_eval_dir = os.path.join(experiments_root_dir, experiment_id)
    os.makedirs(path_to_eval_dir, exist_ok=True)

    return path_to_eval_dir

def get_repo_root_path(start: Path = Path(__file__).resolve(), marker: str='pyproject.toml') -> Optional[Path]:
    repo_root_path = None
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            repo_root_path = parent
            break
    
    return repo_root_path
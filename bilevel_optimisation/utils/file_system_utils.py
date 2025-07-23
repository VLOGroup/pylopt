import os
from confuse import Configuration
from pathlib import Path

def dump_config_file(config: Configuration, path_to_data_dir: str):
    config_data = config.dump(full=True)
    config_file_path = os.path.join(path_to_data_dir, 'config.dump')

    with open(config_file_path, 'w') as file:
        file.write(str(config_data))

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

    dump_config_file(config, path_to_eval_dir)
    return path_to_eval_dir
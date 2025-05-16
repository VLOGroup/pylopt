import os
from confuse import Configuration
from pathlib import Path
import torch

from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts

def dump_config_file(config: Configuration, path_to_data_dir: str):
    config_data = config.dump(full=True)
    config_file_path = os.path.join(path_to_data_dir, 'config.dump')

    with open(config_file_path, 'w') as file:
        file.write(str(config_data))

def create_evaluation_dir(config: Configuration) -> str:
    package_root_path = Path(__file__).resolve().parents[2]
    export_path = os.path.join(package_root_path, 'data', 'evaluation')
    os.makedirs(export_path, exist_ok=True)

    experiment_list = sorted(os.listdir(export_path))
    if experiment_list:
        experiment_id = str(int(experiment_list[-1]) + 1).zfill(5)
    else:
        experiment_id = str(0).zfill(5)
    path_to_eval_dir = os.path.join(export_path, experiment_id)
    os.makedirs(path_to_eval_dir, exist_ok=True)

    dump_config_file(config, path_to_eval_dir)

    return path_to_eval_dir

def save_foe_model(model: FieldsOfExperts, path_to_data_dir: str, model_dir_name: str='models') -> None:
    """
    Function which saves a FoE model to file. This is implemented as follows:
        > Filters, filter weights and state dict of potential are stored separately
        > The method state_dict() of the module representing the potential is overloaded - next
            the usual data, also the architecture of the model is stored. This allows to
            load model.

    :param model: Object of class FieldsOfExperts
    :param path_to_data_dir: Path to super directory of directory where model files shall be stored
    :param model_dir_name: Subdirectory of path_to_data_dir where model files will be stored
    :return: /
    """
    path_to_model_dir = os.path.join(path_to_data_dir, model_dir_name)
    if not os.path.exists(path_to_model_dir):
        os.makedirs(path_to_model_dir, exist_ok=True)

    filters_file_name = 'filters.pt'
    filter_weights_file_name = 'filter_weights.pt'
    potential_file_name = 'potential.pt'

    torch.save(model.get_filters(), os.path.join(path_to_model_dir, filters_file_name))
    torch.save(model.get_filter_weights(), os.path.join(path_to_model_dir, filter_weights_file_name))
    potential = model.get_potential()
    torch.save(potential.state_dict(), os.path.join(path_to_model_dir, potential_file_name))



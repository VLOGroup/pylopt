import os
import torch

from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.potential.GaussianMixture import GaussianMixture
from bilevel_optimisation.projection.ParameterProjections import zero_mean_projection
from bilevel_optimisation.utils.FileSystemUtils import save_foe_model, load_foe_model

def create_dummy_regulariser() -> FieldsOfExperts:
    filter_dim = 7
    num_filters = filter_dim ** 2 - 1
    filters_spec = ParamSpec(value=2 * torch.rand(filter_dim ** 2 - 1, 1, filter_dim, filter_dim) - 1, trainable=True,
                             projection=zero_mean_projection, parameters={'padding_mode': 'reflect'})
    filter_weights_spec = ParamSpec(value=torch.ones(num_filters), trainable=True)

    num_components = 123
    log_weights_spec = ParamSpec(value=2 * torch.rand(num_filters, num_components) - 1, trainable=True)
    potential = GaussianMixture(num_components=num_components, box_lower=-1.0, box_upper=1.0,
                                log_weights_spec=log_weights_spec, num_gmms=num_filters)
    return FieldsOfExperts(potential, filters_spec, filter_weights_spec)

def main():
    regulariser = create_dummy_regulariser()

    # save_model
    path_to_model_dir = os.path.join('./data', 'pretrained_models')
    save_foe_model(regulariser, path_to_model_dir, model_dir_name='dummy_models')
    print('[SAVE] saved model to {:s}'.format(path_to_model_dir))

    # load_model
    regulariser_loaded = load_foe_model(path_to_data_dir=path_to_model_dir, model_dir_name='dummy_models')
    print('[LOAD] loaded model from {:s}'.format(path_to_model_dir))

if __name__ == '__main__':
    main()
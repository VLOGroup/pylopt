import os
import numpy as np
import torch
from confuse import Configuration
from scipy.fftpack import idct
from typing import Iterator
from importlib import resources
from itertools import chain

from bilevel_optimisation import solver
from bilevel_optimisation import losses
from bilevel_optimisation import energy
from bilevel_optimisation.bilevel.Bilevel import Bilevel
from bilevel_optimisation.data.ParamSpec import ParamSpec
# from bilevel_optimisation.energy import OptimisationEnergy, UnrollingEnergy
from bilevel_optimisation.energy.Energy import Energy
from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.filters.Filters import ImageFilter
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.potential import GaussianMixture, StudentT, Potential, LinearSplinePotential

def get_model_data_dir_path(config: Configuration) -> str:
    models_root_dir = config['data']['models']['root_dir'].get()
    model_data_dir = os.path.join(resources.files('bilevel_optimisation'), 'model_data')
    if models_root_dir:
        model_data_dir = models_root_dir
    return model_data_dir

def set_up_image_filter(config: Configuration) -> ImageFilter:
    filter_data = None

    trainable = config['regulariser']['image_filter']['trainable'].get()
    padding_mode = config['regulariser']['image_filter']['padding_mode'].get()
    filter_file = config['regulariser']['image_filter']['initialisation']['file'].get()
    filter_multiplier = config['regulariser']['image_filter']['initialisation']['multiplier'].get()

    if filter_file:
        model_data_dir_path = get_model_data_dir_path(config)
        filter_data = torch.load(os.path.join(model_data_dir_path, filter_file))

        initialisation_dict = filter_data['initialisation_dict']
        filter_dim = initialisation_dict['filter_dim']
        padding_mode = initialisation_dict['padding_mode']

        dummy_filter_data = torch.ones(filter_dim ** 2 - 1, 1, filter_dim, filter_dim)
        dummy_filter_spec = ParamSpec(dummy_filter_data, trainable=trainable, parameters={'padding_mode': padding_mode})
        image_filter = ImageFilter(dummy_filter_spec)

        state_dict = filter_data['state_dict']
        image_filter.load_state_dict(state_dict)

        with torch.no_grad():
            image_filter.filter_tensor.copy_(image_filter.filter_tensor * filter_multiplier)
    else:
        filter_dim = config['regulariser']['image_filter']['initialisation']['parameters']['filter_dim'].get()
        filter_name = config['regulariser']['image_filter']['initialisation']['parameters']['name'].get()

        if filter_name == 'dct':
            can_basis = np.reshape(np.eye(filter_dim ** 2, dtype=np.float64), (filter_dim ** 2, filter_dim, filter_dim))
            dct_basis = idct(idct(can_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
            dct_basis = dct_basis[1:].reshape(-1, 1, filter_dim, filter_dim)
            filter_data = torch.tensor(dct_basis)

        if filter_name == 'rand':
            filter_data = 2 * torch.rand(filter_dim ** 2 - 1, 1, filter_dim, filter_dim) - 1

        if filter_name == 'randn':
            filter_data = torch.randn(filter_dim ** 2 - 1, 1, filter_dim, filter_dim)

        filter_data = filter_data * filter_multiplier
        filter_spec = ParamSpec(filter_data, trainable=trainable,
                                parameters={'padding_mode': padding_mode})
        image_filter = ImageFilter(filter_spec)
    return image_filter

def set_up_gaussian_mixture(config: Configuration, num_filters: int) -> GaussianMixture:
    potential_file = config['regulariser']['potential']['parameters']['gaussian_mixture']['initialisation']['file'].get()
    trainable = config['regulariser']['potential']['parameters']['gaussian_mixture']['trainable'].get()

    if potential_file:

        # TODO
        #   test me !!!
        print('to be tested ...')

        model_data_dir_path = get_model_data_dir_path(config)
        model_data = torch.load(os.path.join(model_data_dir_path, potential_file))
        initialisation_dict = model_data['initialisation_dict']
        num_gmms = initialisation_dict['num_potentials'].item()
        num_components = initialisation_dict['num_components'].item()

        dummy_log_weights = torch.ones(num_gmms, num_components)
        dummy_log_weights_spec = ParamSpec(dummy_log_weights, trainable=trainable)

        potential = GaussianMixture(num_components=num_components,
                                    box_lower=initialisation_dict['box_lower'].item(),
                                    box_upper=initialisation_dict['box_upper'].item(),
                                    log_weights_spec=dummy_log_weights_spec, num_potentials=num_gmms)

        state_dict = model_data['state_dict']
        potential.load_state_dict(state_dict)
    else:
        num_components = (
            config['regulariser']['potential']['parameters']['gaussian_mixture']['num_components'].get())
        box_lower = config['regulariser']['potential']['parameters']['gaussian_mixture']['box_lower'].get()
        box_upper = config['regulariser']['potential']['parameters']['gaussian_mixture']['box_upper'].get()

        initialisation_type = (
            config['regulariser']['potential']['parameters']['gaussian_mixture']['initialisation']['name'].get())
        if initialisation_type == 'uniform':
            log_weights = torch.ones(num_filters, num_components)

        if initialisation_type == 'rand':
            log_weights = 2 * torch.rand(num_filters, num_components) - 1

        log_weights_spec = ParamSpec(log_weights, trainable=trainable)
        potential = GaussianMixture(num_components=num_components,
                                    box_lower=box_lower, box_upper=box_upper,
                                    log_weights_spec=log_weights_spec, num_potentials=num_filters)

    return potential

def set_up_linear_spline_potential(config: Configuration, num_filters: int) -> LinearSplinePotential:
    potential_file = config['regulariser']['potential']['parameters']['linear_spline']['initialisation']['file'].get()
    trainable = config['regulariser']['potential']['parameters']['linear_spline']['trainable'].get()

    if potential_file:
        model_data_dir_path = get_model_data_dir_path(config)
        model_data = torch.load(os.path.join(model_data_dir_path, potential_file))

        initialisation_dict = model_data['initialisation_dict']
        num_potentials = initialisation_dict['num_potentials']
        num_nodes = initialisation_dict['num_nodes']

        dummy_params = torch.ones(num_potentials, num_nodes)
        dummy_param_spec = ParamSpec(dummy_params, trainable=trainable)

        potential = LinearSplinePotential(num_nodes=num_nodes,
                                          box_lower=initialisation_dict['box_lower'],
                                          box_upper=initialisation_dict['box_upper'],
                                          param_spec=dummy_param_spec)

        state_dict = model_data['state_dict']
        potential.load_state_dict(state_dict)
    else:
        nodal_values = None
        multiplier = (
            config['regulariser']['potential']['parameters']['linear_spline']['initialisation']['multiplier'].get())

        num_nodes = (
            config['regulariser']['potential']['parameters']['linear_spline']['num_nodes'].get())
        box_lower = config['regulariser']['potential']['parameters']['linear_spline']['box_lower'].get()
        box_upper = config['regulariser']['potential']['parameters']['linear_spline']['box_upper'].get()

        if config['regulariser']['potential']['parameters']['linear_spline']['initialisation'][
            'name'].get() == 'uniform':
            nodal_values = torch.ones(num_filters, num_nodes)
            nodal_values[0] = 0
            nodal_values[-1] = 0

        if config['regulariser']['potential']['parameters']['linear_spline']['initialisation'][
            'name'].get() == 'student_t':
            nodal_values = 1 / (1 + torch.linspace(-1, 1, num_nodes) ** 2)
            nodal_values = nodal_values.repeat(num_filters, 1)
            nodal_values[0] = 0
            nodal_values[-1] = 0

        nodal_values_spec = ParamSpec(multiplier * nodal_values, trainable=trainable)
        potential = LinearSplinePotential(num_nodes=num_nodes, box_lower=box_lower, box_upper=box_upper,
                                          param_spec=nodal_values_spec)

    return potential

def set_up_student_t_potential(config: Configuration, num_filters: int) -> StudentT:
    potential_file = config['regulariser']['potential']['parameters']['student_t']['initialisation']['file'].get()
    trainable = config['regulariser']['potential']['parameters']['student_t']['trainable'].get()

    if potential_file:
        model_data_dir_path = get_model_data_dir_path(config)
        model_data = torch.load(os.path.join(model_data_dir_path, potential_file))

        initialisation_dict = model_data['initialisation_dict']
        num_potentials = initialisation_dict['num_potentials'].item()

        dummy_weights = torch.ones(num_potentials)
        dummy_weights_spec = ParamSpec(dummy_weights, trainable=trainable)

        potential = StudentT(num_potentials=num_potentials, weights_spec=dummy_weights_spec)
        state_dict = model_data['state_dict']
        potential.load_state_dict(state_dict)
    else:
        weights = None
        weight_multiplier = (
            config['regulariser']['potential']['parameters']['student_t']['initialisation']['multiplier'].get())

        if config['regulariser']['potential']['parameters']['student_t']['initialisation']['name'].get() == 'uniform':
            weights = weight_multiplier * torch.ones(num_filters)

        weights_spec = ParamSpec(weights, trainable=trainable)
        potential = StudentT(num_potentials=num_filters, weights_spec=weights_spec)

    return potential

def set_up_potential(config: Configuration, num_filters: int) -> Potential:
    potential_name = config['regulariser']['potential']['name'].get()
    if potential_name == 'GaussianMixture':
        potential = set_up_gaussian_mixture(config, num_filters)

    if potential_name == 'StudentT':
        potential = set_up_student_t_potential(config, num_filters)

    if potential_name == 'LinearSpline':
        potential = set_up_linear_spline_potential(config, num_filters)
    return potential

def set_up_regulariser(config: Configuration) -> torch.nn.Module:
    image_filter = set_up_image_filter(config)
    num_filters = image_filter.get_num_filters()
    potential = set_up_potential(config, num_filters)
    return FieldsOfExperts(potential, image_filter)

def load_backward_mode(config: Configuration):
    energy_type = config['inner_energy']['type'].get()
    if energy_type == OptimisationEnergy.__name__:
        return 'differentiation'
    elif energy_type == UnrollingEnergy.__name__:
        return 'unrolling'
    else:
        raise ValueError('Unknown energy type - cannot assign differentiation mode.')

def load_bilevel_optimiser(parameters: Iterator[torch.nn.Parameter], config: Configuration) -> torch.optim.Optimizer:
    optimiser_name = config['bilevel']['optimiser']['name'].get()
    optimiser_cls = load_optimiser_class(optimiser_name, projected=True)

    if config['bilevel']['optimiser']['parameters']['param_groups'].get():
        param_groups = []
        for group_params, group_configs \
                in zip(parameters, config['bilevel']['optimiser']['parameters']['param_groups'].get()):
            hyper_params = {k: v for k, v in group_configs.items() if k != 'params'}
            param_groups.append({'params': group_params, **hyper_params})

        optimiser_ = optimiser_cls(param_groups)
    else:
        optimiser_params = config['bilevel']['optimiser']['parameters'].get()

        optimiser_ = optimiser_cls(parameters, **optimiser_params)

    return optimiser_

def set_up_bilevel_problem(filter_parameters: Iterator[torch.nn.Parameter],
                           potential_parameters: Iterator[torch.nn.Parameter],
                           config: Configuration) -> Bilevel:

    parameters = chain(filter_parameters, potential_parameters)
    optimiser_ = load_bilevel_optimiser(parameters, config)

    solver_name = config['bilevel']['solver']['name'].get()
    solver_params = config['bilevel']['solver']['parameters'].get()
    solver_cls = getattr(solver, solver_name)

    backward_mode = load_backward_mode(config)

    return Bilevel(optimiser_, build_solver_factory(SolverSpec(solver_class=solver_cls, solver_params=solver_params)),
                   backward_mode=backward_mode)

def set_up_measurement_model(data: torch.Tensor, config: Configuration) -> MeasurementModel:
    noise_level = config['measurement_model']['noise_level'].get()
    forward_operator = config['measurement_model']['forward_operator'].get()
    operator_cls = getattr(torch.nn, forward_operator)
    return MeasurementModel(data, operator_cls(), noise_level)

def set_up_outer_loss(data: torch.Tensor, config: Configuration):
    loss_name = config['bilevel']['loss'].get()
    loss_cls = getattr(losses, loss_name)
    return loss_cls(data)

# def load_energy_class(config: Configuration) -> type[InnerEnergy]:
#     energy_type = config['inner_energy']['type'].get()
#     if hasattr(energy, energy_type):
#         energy_cls = getattr(energy, energy_type)
#     else:
#         raise ValueError('Cannot find energy {:s}'.format(energy_type))
#     return energy_cls

# def optimiser_is_compatible(energy_cls: type[InnerEnergy], config: Configuration) -> bool:
#     ret_val = True
#     if energy_cls.__name__ == UnrollingEnergy.__name__:
#         optimiser_name = config['inner_energy']['optimiser']['name'].get()
#         optimiser_cls = load_optimiser_class(optimiser_name)
#         ret_val = optimiser_cls.__name__ in UNROLLING_TYPE_OPTIMISER
#
#     return ret_val

def set_up_inner_energy(measurement_model, regulariser, config: Configuration) -> Energy:
    # energy_cls = load_energy_class(config)
    #
    # if not optimiser_is_compatible(energy_cls, config):
    #     raise ValueError('Chosen optimiser and chosen energy are not compatible')

    # load regularisation parameter
    lam = config['inner_energy']['lam'].get()
    return Energy(measurement_model, regulariser, lam)


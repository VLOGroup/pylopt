import os
import numpy as np
import torch
from confuse import Configuration
from scipy.fftpack import idct
from typing import Iterator
from importlib import resources

from bilevel_optimisation import optimiser
from bilevel_optimisation import solver
from bilevel_optimisation import losses
from bilevel_optimisation import energy
from bilevel_optimisation.bilevel.Bilevel import Bilevel
from bilevel_optimisation.data.OptimiserSpec import OptimiserSpec
from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.data.SolverSpec import SolverSpec
from bilevel_optimisation.energy.InnerEnergy import InnerEnergy
from bilevel_optimisation.factories.BuildFactory import build_solver_factory
from bilevel_optimisation.factories.BuildFactory import build_optimiser_factory, build_prox_map_factory
from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.filters.Filters import ImageFilter
from bilevel_optimisation.optimiser import FixedIterationsStopping
from bilevel_optimisation.optimiser.ProjectedOptimiser import create_projected_optimiser
from bilevel_optimisation.optimiser import NAG_TYPE_OPTIMISER
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.potential import GaussianMixture, StudentT, Potential

def get_model_data_dir_path(config: Configuration) -> str:
    models_root_dir = config['data']['models']['root_dir'].get()
    model_data_dir = os.path.join(resources.files('bilevel_optimisation'), 'model_data')
    if models_root_dir:
        model_data_dir = models_root_dir
    return model_data_dir

def set_up_image_filter(config: Configuration) -> ImageFilter:
    trainable = config['regulariser']['image_filter']['trainable'].get()
    padding_mode = config['regulariser']['image_filter']['padding_mode'].get()
    filter_file = config['regulariser']['image_filter']['initialisation']['file'].get()
    filter_multiplier = config['regulariser']['image_filter']['initialisation']['multiplier'].get()

    if filter_file:
        model_data_dir_path = get_model_data_dir_path(config)
        filter_data = torch.load(os.path.join(model_data_dir_path, filter_file))
    else:
        filter_dim = config['regulariser']['filters']['initialisation']['parameters']['filter_dim'].get()
        filter_name = config['regulariser']['filters']['initialisation']['parameters']['filter_name'].get()

        if filter_name == 'dct':
            can_basis = np.reshape(np.eye(filter_dim ** 2, dtype=np.float64), (filter_dim ** 2, filter_dim, filter_dim))
            dct_basis = idct(idct(can_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
            dct_basis = dct_basis[1:].reshape(-1, 1, filter_dim, filter_dim)
            filter_data = torch.tensor(dct_basis)

        if filter_name == 'rand':
            filter_data = 2 * torch.rand(filter_dim ** 2 - 1, 1, filter_dim, filter_dim) - 1

        if filter_name == 'randn':
            filter_data = torch.randn(filter_dim ** 2 - 1, 1, filter_dim, filter_dim)

    filter_spec = ParamSpec(filter_data * filter_multiplier, trainable=trainable,
                            parameters={'padding_mode': padding_mode})
    return ImageFilter(filter_spec)

def set_up_gaussian_mixture(config: Configuration, num_filters: int) -> GaussianMixture:
    potential_file = config['regulariser']['potential']['parameters']['gaussian_mixture']['parameters'].get()
    trainable = (
        config['regulariser']['parameters']['gaussian_mixture']['initialisation']['parameters']['trainable'].get())

    if potential_file:
        model_data = torch.load(potential_file)
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
        initialisation_type = (
            config['regulariser']['potential']['parameters']['gaussian_mixture']['initialisation']['name'].get())
        num_components = (
            config['regulariser']['potential']['parameters']['gaussian_mixture']['num_components'].get())
        box_lower = config['regulariser']['potential']['parameters']['gaussian_mixture']['box_lower'].get()
        box_upper = config['regulariser']['potential']['parameters']['gaussian_mixture']['box_upper'].get()

        if initialisation_type == 'uniform':
            log_weights = torch.ones(num_filters, num_components)

        if initialisation_type == 'rand':
            log_weights = 2 * torch.rand(num_filters, num_components) - 1

        log_weights_spec = ParamSpec(log_weights, trainable=trainable)
        potential = GaussianMixture(num_components=num_components,
                                    box_lower=box_lower, box_upper=box_upper,
                                    log_weights_spec=log_weights_spec, num_potentials=num_filters)

    return potential

def set_up_student_t_potential(config: Configuration, num_filters: int) -> StudentT:
    potential_file = config['regulariser']['potential']['parameters']['student_t']['initialisation']['file'].get()
    trainable = (
        config['regulariser']['potential']['parameters']['student_t']['trainable'].get())

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

    return potential

def set_up_regulariser(config: Configuration) -> torch.nn.Module:
    image_filter = set_up_image_filter(config)
    num_filters = image_filter.get_num_filters()
    potential = set_up_potential(config, num_filters)
    return FieldsOfExperts(potential, image_filter)

def load_optimiser_class(optimiser_name: str, projected: bool=False) -> type[torch.optim.Optimizer]:
    # TODO
    #   > Optimisers from torch.optim do not apply projections, prox-maps, etc. by default.
    #   > The implementations of NAGOptimiser, AlternatingNAGOptimiser on the other hand do.
    #   > Wrapping optimisers from torch.optim appropriately one can ensure that
    #       projections, etc. are applied.
    #   > For the sake of consistency apply projections, prox-maps also for NAGOptimiser, AlternatingNAGOptimiser
    #       by wrapping the corresponding classes.
    if hasattr(optimiser, optimiser_name):
        optimiser_cls = getattr(optimiser, optimiser_name)
    elif hasattr(torch.optim, optimiser_name):
        optimiser_cls = getattr(torch.optim, optimiser_name)
        if projected:
            optimiser_cls = create_projected_optimiser(optimiser_cls)
    else:
        raise ValueError('Cannot find optimiser {:s}'.format(optimiser_name))

    return optimiser_cls

def set_up_bilevel_problem(parameters: Iterator[torch.nn.Parameter], config: Configuration) -> Bilevel:
    optimiser_name = config['bilevel']['optimiser']['name'].get()
    optimiser_params = config['bilevel']['optimiser']['parameters'].get()
    optimiser_cls = load_optimiser_class(optimiser_name, projected=True)
    optimiser_ = optimiser_cls(parameters, **optimiser_params)

    solver_name = config['bilevel']['solver']['name'].get()
    solver_params = config['bilevel']['solver']['parameters'].get()
    solver_cls = getattr(solver, solver_name)

    return Bilevel(optimiser_, build_solver_factory(SolverSpec(solver_class=solver_cls,
                                                               solver_params=solver_params)))

def set_up_measurement_model(data: torch.Tensor, config: Configuration) -> MeasurementModel:
    noise_level = config['measurement_model']['noise_level'].get()
    forward_operator = config['measurement_model']['forward_operator'].get()
    operator_cls = getattr(torch.nn, forward_operator)
    return MeasurementModel(data, operator_cls(), noise_level)

def set_up_outer_loss(data: torch.Tensor, config: Configuration):
    loss_name = config['bilevel']['loss'].get()
    loss_cls = getattr(losses, loss_name)
    return loss_cls(data)

def set_up_inner_energy(measurement_model, regulariser, config: Configuration) -> InnerEnergy:
    optimiser_name = config['inner_energy']['optimiser']['name'].get()
    optimiser_params = config['inner_energy']['optimiser']['parameters'].get()
    optimiser_cls = load_optimiser_class(optimiser_name)

    stopping_name = config['inner_energy']['optimiser']['stopping']['name'].get()
    stopping_parameters = config['inner_energy']['optimiser']['stopping']['parameters'].get()

    if hasattr(optimiser, stopping_name):
        stopping_cls = getattr(optimiser, stopping_name)
    else:
        stopping_cls = FixedIterationsStopping(max_num_iterations=1000)

    prox_map_factory = None
    if config['inner_energy']['optimiser']['use_prox'].get() and optimiser_name in NAG_TYPE_OPTIMISER:
        noise_level = config['measurement_model']['noise_level'].get()
        prox_map = lambda x, y, tau: ((tau / noise_level) * y + x) / (1 + (tau / noise_level))
        prox_map_factory = build_prox_map_factory(prox_map)

    optimiser_spec = OptimiserSpec(optimiser_class=optimiser_cls, optimiser_params=optimiser_params,
                                   stopping_class=stopping_cls, stopping_params=stopping_parameters,
                                   prox_map_factory=prox_map_factory)
    optimiser_factory = build_optimiser_factory(optimiser_spec)

    energy_type = config['inner_energy']['type'].get()
    if hasattr(energy, energy_type):
        energy_cls = getattr(energy, energy_type)
    else:
        raise ValueError('Cannot find energy {:s}'.format(energy_type))

    lam = config['inner_energy']['lam'].get()
    return energy_cls(measurement_model, regulariser, lam, optimiser_factory)


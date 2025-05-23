import os
import numpy as np
import torch
from confuse import Configuration
from pathlib import Path
from scipy.fftpack import idct
from typing import Iterator

from bilevel_optimisation import optimiser
from bilevel_optimisation import solver
from bilevel_optimisation import losses
from bilevel_optimisation import energy
from bilevel_optimisation.optimiser import FixedIterationsStopping
from bilevel_optimisation.optimiser.ProjectedOptimiser import create_projected_optimiser
from bilevel_optimisation.bilevel.Bilevel import Bilevel
from bilevel_optimisation.data.OptimiserSpec import OptimiserSpec
from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.data.SolverSpec import SolverSpec
from bilevel_optimisation.energy.InnerEnergy import InnerEnergy
from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.factories.BuildFactory import build_solver_factory
from bilevel_optimisation.factories.BuildFactory import build_optimiser_factory, build_prox_map_factory
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.potential import GaussianMixture, StudentT
from bilevel_optimisation.projection.ParameterProjections import (zero_mean_projection,
                                                                  unit_simplex_projection)

def load_filters_spec(config: Configuration) -> ParamSpec:
    filters = None

    trainable = config['regulariser']['filters']['trainable'].get()
    padding_mode = config['regulariser']['filters']['padding_mode'].get()
    filters_file = config['regulariser']['filters']['initialisation']['file'].get()
    filters_multiplier = config['regulariser']['filters']['initialisation']['multiplier'].get()

    if filters_file:
        package_root_path = Path(__file__).resolve().parents[2]
        filters = torch.load(os.path.join(package_root_path, filters_file))
    else:
        filters_params = config['regulariser']['filters']['initialisation']['parameters'].get()
        filter_dim = filters_params['filter_dim']

        if filters_params['name'] == 'dct':
            can_basis = np.reshape(np.eye(filter_dim ** 2, dtype=np.float64), (filter_dim ** 2, filter_dim, filter_dim))
            dct_basis = idct(idct(can_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
            dct_basis = dct_basis[1:].reshape(-1, 1, filter_dim, filter_dim)
            filters = torch.tensor(dct_basis)

        if filters_params['name'] == 'rand':
            filters = 2 * torch.rand(filter_dim ** 2 - 1, 1, filter_dim, filter_dim) - 1

        if filters_params['name'] == 'randn':
            filters = torch.randn(filter_dim ** 2 - 1, 1, filter_dim, filter_dim)

    filters = filters * filters_multiplier

    return ParamSpec(filters, trainable=trainable, projection=zero_mean_projection,
                     parameters={'padding_mode': padding_mode})

def load_filter_weights_spec(config: Configuration, num_filters: int) -> ParamSpec:
    filter_weights = None

    trainable = config['regulariser']['filter_weights']['trainable'].get()
    filter_weights_file = config['regulariser']['filter_weights']['initialisation']['file'].get()
    filter_weights_multiplier = config['regulariser']['filter_weights']['initialisation']['multiplier'].get()
    if filter_weights_file:
        package_root_path = Path(__file__).resolve().parents[2]
        filter_weights = torch.load(os.path.join(package_root_path, filter_weights_file))
    else:
        filter_weights_params = config['regulariser']['filter_weights']['initialisation']['parameters'].get()

        if filter_weights_params['name'] == 'uniform':
            filter_weights = torch.ones(num_filters)

    filter_weights = filter_weights * filter_weights_multiplier

    return ParamSpec(filter_weights, trainable=trainable, projection=lambda z: torch.clamp(z, min=0.00001))

def load_gmm_weights_spec(config: Configuration) -> ParamSpec:
    weights = None
    potential_params = config['regulariser']['potential']['parameters'].get()
    num_components = potential_params['num_components']
    trainable = potential_params['weights']['trainable']
    weights_file = potential_params['weights']['initialisation']['file']
    if weights_file:
        package_root_path = Path(__file__).resolve().parents[2]
        weights = torch.load(os.path.join(package_root_path, weights_file))
    else:
        weights_params = potential_params['weights']['initialisation']['parameters']

        # TODO
        #   > clean me up!!

        # box_lower = potential_params['box_lower']
        # box_upper = potential_params['box_upper']
        # centers = torch.linspace(box_lower, box_upper, num_components)
        #
        # sig = 2 * (box_upper - box_lower) / (num_components - 1)
        # sig_sq = sig ** 2
        #
        # weights = np.sqrt(2 * np.pi * sig_sq) / (torch.pi * (1 + centers ** 2))
        # weights = unit_simplex_projection(weights)

                # # debug
                # from matplotlib import pyplot as plt
                # fig = plt.figure()
                # ax = fig.add_subplot(1, 1, 1)
                # t = np.linspace(-3, 3, 333)
                # ax.plot(t, 1 / (np.pi * (1 + t ** 2)))
                # y =  np.exp((-0.5 / sig_sq) * (t.reshape(-1, 1) - centers.detach().cpu().numpy().reshape(1, -1))**2) / np.sqrt(2 * np.pi * sig_sq)
                #
                # ax.plot(t, 1 / (np.pi * (1 + t ** 2)), color='green')
                # ax.plot(t, np.sum(weights.detach().cpu().numpy() * y, axis=-1), color='orange')
                # plt.show()

        #
        # if weights_params['name'] == 'uniform':
        #     weights = torch.ones(num_components)
        #     weights *= weights_params['multiplier']

    return ParamSpec(weights, trainable=trainable, projection=unit_simplex_projection)

def set_up_regulariser(config: Configuration) -> torch.nn.Module:
    filters_spec = load_filters_spec(config)
    num_filters = filters_spec.value.shape[0]
    filter_weights_spec = load_filter_weights_spec(config, num_filters=num_filters)

    potential_name = config['regulariser']['potential']['name'].get()
    if potential_name == 'GaussianMixture':
        potential_file = config['regulariser']['potential']['file'].get()
        gmm_params = config['regulariser']['potential']['parameters']['gmm'].get()
        trainable = gmm_params['trainable']
        if potential_file:
            package_root_path = Path(__file__).resolve().parents[2]
            model_data_file = os.path.join(package_root_path, potential_file)
            model_data = torch.load(model_data_file)

            # dummy weight initialisation
            num_components = model_data['_arch']['num_components']
            arch = model_data['_arch']
            arch['weights_spec'] = ParamSpec(-torch.ones(num_components), trainable=trainable)

            potential_ = GaussianMixture(**arch)
            potential_.load_state_dict(model_data)
        else:
            num_components = gmm_params['num_components']
            box_lower = gmm_params['box_lower']
            box_upper = gmm_params['box_upper']
            trainable = trainable
            log_weights_spec = ParamSpec(value=2 * torch.rand(num_filters, num_components) - 1, trainable=trainable)
            potential_ = GaussianMixture(num_components=num_components, box_lower=box_lower, box_upper=box_upper,
                                         log_weights_spec=log_weights_spec, num_filters=num_filters)
    elif potential_name == 'StudentT':
        potential_ = StudentT()
    else:
        raise ValueError('There is no potential titled {:s}'.format(potential_name))

    return FieldsOfExperts(potential_, filters_spec, filter_weights_spec)

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

    prox_map = lambda x, y, tau: (tau * y + x) / (1 + tau)
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


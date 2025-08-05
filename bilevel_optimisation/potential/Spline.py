import torch
from typing import Any, Dict, Tuple
from confuse import Configuration

from bilevel_optimisation.data.Constants import EPSILON
from bilevel_optimisation.potential.Potential import Potential
from bilevel_optimisation.solver.solve_tridiagonal import solve_tridiagonal

class NaturalCubicSplineAutogradFunction(torch.autograd.Function):

    generate_vmap_rule = True

    @staticmethod
    def bucketise(x: torch.Tensor, box_lower: float, box_upper: float, num_nodes: int) -> torch.Tensor:
        x_scaled = (x - box_lower) / (box_upper - box_lower)
        return torch.clamp((x_scaled * (num_nodes - 1)).ceil().long() - 1, min=0, max=num_nodes - 2)

    @staticmethod
    def forward(nodes: torch.Tensor, nodal_values: torch.Tensor,
                coeffs_1st_order: torch.Tensor, coeffs_2nd_order: torch.Tensor, coeffs_3rd_order: torch.Tensor,
                box_lower: float, box_upper: float, num_nodes: int, x: torch.Tensor) -> Tuple:
        index_tensor = NaturalCubicSplineAutogradFunction.bucketise(x, box_lower, box_upper, num_nodes)
        bs, f, w, h = x.shape
        idx_flat = index_tensor.view(bs, f, w, h).permute(1, 0, 2, 3).reshape(f, -1)

        a = nodal_values.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
        b = coeffs_1st_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
        c = coeffs_2nd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
        d = coeffs_3rd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)

        y = x - nodes[index_tensor]
        return a + y * (b + y * (c + y * d)), y, b, c, d

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, outputs: Tuple):
        _, y, b, c, d = outputs
        ctx.save_for_backward(y, b, c, d)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: Any) -> Any:
        y, b, c, d = ctx.saved_tensors
        return *[None] * 8, b + y * (c + y * d)

@Potential.register_subclass('spline')
class NaturalCubicSpline(Potential):

    def __init__(self, num_marginals: int, config: Configuration):
        super().__init__(num_marginals)

        initialisation_mode = config['potential']['spline']['initialisation']['mode'].get()
        multiplier = config['potential']['spline']['initialisation']['multiplier'].get()
        trainable = config['potential']['spline']['trainable'].get()

        model_path = config['potential']['spline']['initialisation']['file_path'].get()
        if not model_path:
            self.num_nodes = config['potential']['spline']['num_nodes'].get()
            self.box_lower = config['potential']['spline']['box_lower'].get()
            self.box_upper = config['potential']['spline']['box_upper'].get()

            self.register_buffer('nodes', torch.linspace(self.box_lower, self.box_upper, self.num_nodes))
            self.register_buffer('coeffs_1st_order', torch.zeros(self.num_marginals, self.num_nodes - 1))
            self.register_buffer('coeffs_2nd_order', torch.zeros(self.num_marginals, self.num_nodes - 1))
            self.register_buffer('coeffs_3rd_order', torch.zeros(self.num_marginals, self.num_nodes - 1))

            if initialisation_mode == 'rand':
                nodal_vals = min(multiplier, 1.0) * torch.rand(num_marginals, self.num_nodes)
            elif initialisation_mode == 'uniform':
                nodal_vals = min(multiplier, 1.0 - EPSILON) * torch.ones(num_marginals, self.num_nodes)
            elif initialisation_mode == 'student_t':
                nodal_vals = torch.log(1 + torch.stack([self.nodes for _ in range(0, num_marginals)], dim=0) ** 2)
            else:
                raise ValueError('Unknown initialisation method')
            self.nodal_values = torch.nn.Parameter(data=nodal_vals, requires_grad=trainable)
        else:
            pass

        self.step_size = self.nodes[1] - self.nodes[0]
        self._fit()

    def _fit(self) -> None:
        device = next(self.parameters()).device

        diag = torch.ones(self.num_nodes, device=device)
        diag[1:-1] = 4
        diag_super = torch.ones(self.num_nodes - 1, device=device)
        diag_super[0] = 0
        diag_sub = torch.ones(self.num_nodes - 1, device=device)
        diag_sub[-1] = 0

        rhs = torch.zeros_like(self.nodal_values)
        rhs[:, 1: self.num_nodes - 1] = 3 * (self.nodal_values[:, 0: self.num_nodes - 2] -
                                             2 * self.nodal_values[:, 1: self.num_nodes - 1] +
                                             self.nodal_values[:, 2::]) / (self.step_size ** 2)

        self.coeffs_2nd_order = solve_tridiagonal(diag, diag_super, diag_sub, rhs)
        self.coeffs_1st_order = ((self.nodal_values[:, 1::] - self.nodal_values[:, 0: -1]) / self.step_size -
                                 self.step_size * (2 * self.coeffs_2nd_order[:, 0: -1] +
                                                   self.coeffs_2nd_order[:, 1::]) / 3)
        self.coeffs_3rd_order = (self.coeffs_2nd_order[:, 1::] - self.coeffs_2nd_order[:, 0: -1]) / (3 * self.step_size)

    def get_parameters(self) -> torch.Tensor:
        return self.nodal_values.data

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:

        pot, _, _, _, _ = NaturalCubicSplineAutogradFunction.apply(self.nodes, self.nodal_values,
                                                                   self.coeffs_1st_order,
                                                                   self.coeffs_2nd_order, self.coeffs_3rd_order,
                                                                   self.box_lower, self.box_upper, self.num_nodes, x)
        return torch.sum(pot)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, kwargs)
        return state

    def initialisation_dict(self) -> Dict[str, Any]:
        return {'num_marginals': self.num_marginals, 'num_nodes': self.num_nodes,
                'bow_lower': self.box_lower, 'box_upper': self.box_upper}

    def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
        potential_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = potential_data['initialisation_dict']
        self.num_marginals = initialisation_dict['num_marginals']
        self.num_nodes = initialisation_dict['num_nodes']
        self.box_lower = initialisation_dict['box_lower']
        self.box_upper = initialisation_dict['box_upper']

        state_dict = potential_data['state_dict']
        self.load_state_dict(state_dict)

# ### TODO
#   > CLEAN ME UP
#
# def sub_function_0(x: torch.Tensor) -> torch.Tensor:
#     z = x + 0.5
#     return (11 + z * (12 + z * (-6 + z * (-12 + 6 * z)))) / 24
#
# def sub_function_1(x: torch.Tensor) -> torch.Tensor:
#     z = 1.5 - x
#     return (1 + z * (4 + z * (6 + z * (4 - 4 * z)))) / 24
#
# def sub_function_2(x: torch.Tensor) -> torch.Tensor:
#     z = (2.5 - x) * (2.5 - x)
#     return (z * z) / 24
#
# def sub_function_3(x: torch.Tensor) -> torch.Tensor:
#     return torch.zeros_like(x)
#
# def base_spline_func(x: torch.Tensor) -> torch.Tensor:
#     x_abs = torch.abs(x)
#     ret_val = sub_function_0(x_abs)
#     sub_functions = [sub_function_1, sub_function_2, sub_function_3]
#     thresholds = [0.5, 1.5, 2.5]
#
#     for th, func in zip(thresholds, sub_functions):
#         ret_val = torch.where(x_abs >= th, func(x_abs), ret_val)
#
#     return ret_val
#
# class QuarticSpline(Potential):
#
#     def __init__(self, num_marginals: int, config: Configuration) -> None:
#         super().__init__(num_marginals)
#
#         initialisation_mode = config['potential']['spline']['initialisation']['mode'].get()
#         multiplier = config['potential']['spline']['initialisation']['multiplier'].get()
#         trainable = config['potential']['spline']['trainable'].get()
#
#         model_path = config['potential']['spline']['initialisation']['file_path'].get()
#         if not model_path:
#             self.num_nodes = config['potential']['spline']['num_nodes'].get()
#             self.box_lower = config['potential']['spline']['box_lower'].get()
#             self.box_upper = config['potential']['spline']['box_upper'].get()
#
#             if initialisation_mode == 'rand':
#                 weights = torch.log(multiplier * torch.rand(num_marginals, self.num_nodes))
#             elif initialisation_mode == 'uniform':
#                 weights = torch.log(multiplier * torch.ones(num_marginals, self.num_nodes))
#             else:
#                 raise ValueError('Unknown initialisation method')
#             self.weight_tensor = torch.nn.Parameter(data=weights, requires_grad=trainable)
#         else:
#             dummy_data = torch.ones(num_marginals, 1)
#             self.weight_tensor = torch.nn.Parameter(data=dummy_data, requires_grad=trainable)
#             self._load_from_file(model_path)
#
#             with torch.no_grad():
#                 self.weight_tensor.add_(torch.log(torch.tensor(multiplier)))
#
#         self.nodes = torch.nn.Parameter(torch.linspace(self.box_lower, self.box_upper,
#                                                        self.num_nodes).reshape(1, 1, -1, 1, 1), requires_grad=False)
#         self.scale = (self.box_upper - self.box_lower) / (self.num_nodes - 1)
#
#     def get_parameters(self) -> torch.Tensor:
#         return self.weight_tensor.data
#
#     def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
#         x_scaled = (x.unsqueeze(dim=2) - self.nodes) / self.scale
#         #
#         # import torch.profiler
#         # with torch.profiler.profile() as prof:
#         #     y = base_spline_func(x_scaled)
#         # print(prof.key_averages().table())
#
#         y = torch.einsum('bfnhw, fn->bfhw', base_spline_func(x_scaled), torch.exp(self.weight_tensor))
#         y = torch.log(y + EPSILON)
#         return torch.sum(y)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         pass
#
#     def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
#         state = super().state_dict(*args, kwargs)
#         return state
#
#     def initialisation_dict(self) -> Dict[str, Any]:
#         return {'num_marginals': self.num_marginals, 'num_nodes': self.num_nodes, 'bow_lower': self.box_lower,
#                 'box_upper': self.box_upper}
#
#     def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
#         result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
#         return result
#
#     def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
#         potential_data = torch.load(path_to_model, map_location=device)
#
#         initialisation_dict = potential_data['initialisation_dict']
#         self.num_marginals = initialisation_dict['num_marginals']
#         self.num_nodes = initialisation_dict['num_nodes']
#         self.box_lower = initialisation_dict['box_lower']
#         self.box_upper = initialisation_dict['box_upper']
#
#         state_dict = potential_data['state_dict']
#         self.load_state_dict(state_dict)
#
#     def save(self, path_to_model_dir: str, model_name: str='spline') -> str:
#         path_to_model = os.path.join(path_to_model_dir, '{:s}.pt'.format(os.path.splitext(model_name)[0]))
#         potential_dict = {'initialisation_dict': self.initialisation_dict(),
#                           'state_dict': self.state_dict()}
#
#         torch.save(potential_dict, path_to_model)
#         return path_to_model



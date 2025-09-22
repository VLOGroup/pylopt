import torch
from typing import Any, Dict, Optional, Tuple, Self
from confuse import Configuration

from pylopt.potential.Potential import Potential
from quartic_bspline_extension.functions import QuarticBSplineFunction

EPSILON = 1e-7

class QuarticBSpline(Potential):

    def __init__(self, 
                 num_marginals: int,
                 box_lower: float=-1.0,
                 box_upper: float=1.0,
                 num_centers: int=87,
                 initialisation_mode: str='student_t', 
                 multiplier: float=1.0,
                 trainable: bool=True) -> None:
        super().__init__(num_marginals)

        self.box_lower = box_lower
        self.box_upper = box_upper
        self.num_centers = num_centers
        self.scale = (self.box_upper - self.box_lower) / (self.num_centers - 1)
        self.register_buffer('centers', torch.linspace(self.box_lower, self.box_upper, self.num_centers))

        weight_data = self._init_weight_tensor(num_marginals, self.centers, initialisation_mode, multiplier)
        self.weight_tensor = torch.nn.Parameter(data=weight_data, requires_grad=trainable)

    @staticmethod
    def _init_weight_tensor(num_marginals: int, centers: torch.Tensor, initialisation_mode: str, multiplier: float) -> torch.Tensor:
        if initialisation_mode == 'rand':
            weight_data = torch.log(multiplier * torch.rand(num_marginals, len(centers)))
        elif initialisation_mode == 'uniform':
            weight_data = torch.log(multiplier * torch.ones(num_marginals, len(centers)))
        elif initialisation_mode == 'student_t':
            weight_data = torch.log(multiplier * torch.log(1 + torch.stack([centers 
                                                                for _ in range(0, num_marginals)], dim=0) ** 2))
        else:
            raise ValueError('Unknown initialisation method')
        return weight_data
        
    def initialisation_dict(self) -> Dict[str, Any]:
        return {'num_marginals': self.num_marginals, 
                'box_lower': self.box_lower,
                'box_upper': self.box_upper,
                'num_centers': self.num_centers}

    def get_parameters(self) -> torch.Tensor:
        return self.weight_tensor.data

    def forward(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        y, _ = QuarticBSplineFunction.apply(x, torch.exp(self.weight_tensor), self.centers, self.scale)
        if reduce:
            return torch.sum(y)
        else:
            return y

    @classmethod
    def from_file(cls, path_to_model: str, device: torch.device=torch.device('cpu')) -> Self:
        potential_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = potential_data['initialisation_dict']
        state_dict = potential_data['state_dict']

        num_marginals = initialisation_dict.get('num_marginals', 48)
        box_lower = initialisation_dict.get('box_lower', -1.0)
        box_upper = initialisation_dict.get('box_upper', 1.0)
        num_centers = initialisation_dict.get('num_centers', 87)
        potential = cls(num_marginals=num_marginals, box_lower=box_lower, box_upper=box_upper, num_centers=num_centers)
        potential.load_state_dict(state_dict, strict=True)
        return potential

    @classmethod
    def from_config(cls, config: Configuration) -> Self:
        num_marginals = config['potential']['quartic_bspline']['num_marginals'].get()
        box_lower = config['potential']['quartic_bspline']['box_lower'].get()
        box_upper = config['potential']['quartic_bspline']['box_upper'].get()
        num_centers = config['potential']['quartic_bspline']['num_centers'].get()
        initialisation_mode = config['potential']['quartic_bspline']['initialisation']['mode'].get()
        multiplier = config['potential']['quartic_bspline']['initialisation']['multiplier'].get()
        trainable = config['potential']['quartic_bspline']['trainable'].get()
        
        return cls(num_marginals=num_marginals, 
                   box_lower=box_lower, 
                   box_upper=box_upper, 
                   num_centers=num_centers,
                   initialisation_mode=initialisation_mode,
                   multiplier=multiplier,
                   trainable=trainable)

# ### NATURAL CUBIC SPLINE

            # def bucketise(x: torch.Tensor, box_lower: float, box_upper: float, num_nodes: int) -> torch.Tensor:
            #     x_scaled = (x - box_lower) / (box_upper - box_lower)
            #     return torch.clamp((x_scaled * (num_nodes - 1)).ceil().long() - 1, min=0, max=num_nodes - 2)

            # class NaturalCubicSplineAutogradFunction(torch.autograd.Function):
            #
            #     generate_vmap_rule = True
            #
            #     @staticmethod
            #     def first_order_spline_derivative(x: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
            #                                       d: torch.Tensor) -> torch.Tensor:
            #         return b + x * (2 * c + 3 * d * x)
            #
            #     @staticmethod
            #     def second_order_spline_derivative(x: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            #         return 2 * c + 6 * d * x
            #
            #     @staticmethod
            #     def forward(nodes: torch.Tensor, nodal_values: torch.Tensor,
            #                 coeffs_1st_order: torch.Tensor, coeffs_2nd_order: torch.Tensor, coeffs_3rd_order: torch.Tensor,
            #                 box_lower: float, box_upper: float, num_nodes: int, x: torch.Tensor,
            #                 grad_outputs: Optional[torch.Tensor]=None, retain_graph: bool=False, order: int=0) -> Tuple:
            #         index_tensor = bucketise(x, box_lower, box_upper, num_nodes)
            #         y = x - nodes[index_tensor]
            #
            #         bs, f, w, h = x.shape
            #         idx_flat = index_tensor.view(bs, f, w, h).permute(1, 0, 2, 3).reshape(f, -1)
            #
            #         context = torch.enable_grad() if order == 2 else nullcontext()
            #         with context:
            #             a = nodal_values.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
            #             b = coeffs_1st_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
            #             c = coeffs_2nd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
            #             d = coeffs_3rd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
            #             dspline_dx = NaturalCubicSplineAutogradFunction.first_order_spline_derivative(y, b, c, d)
            #
            #         if order == 0:
            #             return a + y * (b + y * (c + y * d)), None, y, b, c, d
            #         elif order == 1:
            #             grad_outputs = torch.ones_like(y) if grad_outputs is None else grad_outputs
            #             return (grad_outputs * NaturalCubicSplineAutogradFunction.first_order_spline_derivative(y, b, c, d),
            #                     None, y, b, c, d)
            #         elif order == 2:
            #             grad_outputs = torch.ones_like(y) if grad_outputs is None else grad_outputs
            #             d2s_mixed = torch.autograd.grad(inputs=nodal_values, outputs=dspline_dx,
            #                                             grad_outputs=grad_outputs, retain_graph=retain_graph)
            #             return NaturalCubicSplineAutogradFunction.second_order_spline_derivative(y, c, d), d2s_mixed, y, b, c, d
            #         else:
            #             raise ValueError('Invalid value {:d}. Expected one of 0, 1, 2.'.format(order))
            #
            #     @staticmethod
            #     def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, outputs: Tuple):
            #         *_, order = inputs
            #         *_, y, b, c, d = outputs
            #
            #         ctx.save_for_backward(y, b, c, d)
            #         ctx.order = order
            #
            #     @staticmethod
            #     def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: Any) -> Any:
            #         y, b, c, d = ctx.saved_tensors
            #         order = ctx.order
            #
            #         if order == 0:
            #             # NOTE
            #             # ----
            #             #   > use the first item of the tuple grad_outputs here - since it corresponds to the function value
            #             #       returned by the forward function.
            #             #   > same for second derivative.
            #             grad = grad_outputs[0] * NaturalCubicSplineAutogradFunction.first_order_spline_derivative(y, b, c, d)
            #         elif order == 1:
            #             grad = grad_outputs[0] * NaturalCubicSplineAutogradFunction.second_order_spline_derivative(y, c, d)
            #         else:
            #             raise ValueError('Invalid value {:d}. Expected one of 0, 1.'.format(order))
            #
            #         return *[None] * 8, grad, None, None, None

                    # @Potential.register_subclass('spline')
                    # class NaturalCubicSpline(Potential):

                    #     def __init__(self, num_marginals: int, config: Configuration):
                    #         super().__init__(num_marginals)

                    #         initialisation_mode = config['potential']['spline']['initialisation']['mode'].get()
                    #         multiplier = config['potential']['spline']['initialisation']['multiplier'].get()
                    #         trainable = config['potential']['spline']['trainable'].get()

                    #         model_path = config['potential']['spline']['initialisation']['file_path'].get()
                    #         if not model_path:
                    #             self.num_nodes = config['potential']['spline']['num_nodes'].get()
                    #             self.box_lower = config['potential']['spline']['box_lower'].get()
                    #             self.box_upper = config['potential']['spline']['box_upper'].get()

                    #             self.register_buffer('nodes', torch.linspace(self.box_lower, self.box_upper, self.num_nodes))
                    #             self.register_buffer('coeffs_1st_order', torch.zeros(self.num_marginals, self.num_nodes - 1))
                    #             self.register_buffer('coeffs_2nd_order', torch.zeros(self.num_marginals, self.num_nodes - 1))
                    #             self.register_buffer('coeffs_3rd_order', torch.zeros(self.num_marginals, self.num_nodes - 1))

                    #             if initialisation_mode == 'rand':
                    #                 nodal_vals = min(multiplier, 1.0) * torch.rand(num_marginals, self.num_nodes)
                    #             elif initialisation_mode == 'uniform':
                    #                 nodal_vals = min(multiplier, 1.0 - EPSILON) * torch.ones(num_marginals, self.num_nodes)
                    #             elif initialisation_mode == 'student_t':
                    #                 nodal_vals = multiplier * torch.log(1 + torch.stack([self.nodes
                    #                                                                      for _ in range(0, num_marginals)], dim=0) ** 2)
                    #             else:
                    #                 raise ValueError('Unknown initialisation method')
                    #             self.nodal_values = torch.nn.Parameter(data=nodal_vals, requires_grad=trainable)
                    #         else:
                    #             # TODO: implement me!
                    #             pass

                    #         # NOTE
                    #         #   > After each parameter update, the spline coefficients need to be recomputed. This is handled by means
                    #         #       of the flag self.recompute_coefficients, which is set to True if coefficients need to be recomputed.
                    #         #   > At each backward call, the function self._coefficient_recomputation_hook() is called, which
                    #         #       sets self.recompute_coefficients = True. Since we deal with gradient based updates only, this
                    #         #       approach is sufficient.
                    #         #   > **IMPORTANT** The hook, calling the coefficient update method, is registered only when module is set
                    #         #       to be trainable.
                    #         self.recompute_coefficients = False
                    #         if trainable:
                    #             self.nodal_values.register_hook(self._coefficient_recomputation_hook)

                    #         self.step_size = self.nodes[1] - self.nodes[0]
                    #         self._fit()

                    #     def _coefficient_recomputation_hook(self, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                    #         self.recompute_coefficients = True

                    #         return grad

                    #     def _fit(self) -> None:
                    #         device = next(self.parameters()).device

                    #         diag = torch.ones(self.num_nodes, device=device)
                    #         diag[1:-1] = 4
                    #         diag_super = torch.ones(self.num_nodes - 1, device=device)
                    #         diag_super[0] = 0
                    #         diag_sub = torch.ones(self.num_nodes - 1, device=device)
                    #         diag_sub[-1] = 0

                    #         rhs = torch.zeros_like(self.nodal_values)
                    #         rhs[:, 1: self.num_nodes - 1] = 3 * (self.nodal_values[:, 0: self.num_nodes - 2] -
                    #                                              2 * self.nodal_values[:, 1: self.num_nodes - 1] +
                    #                                              self.nodal_values[:, 2::]) / (self.step_size ** 2)

                    #         self.coeffs_2nd_order = solve_tridiagonal(diag, diag_super, diag_sub, rhs)
                    #         self.coeffs_1st_order = ((self.nodal_values[:, 1::] - self.nodal_values[:, 0: -1]) / self.step_size -
                    #                                  self.step_size * (2 * self.coeffs_2nd_order[:, 0: -1] +
                    #                                                    self.coeffs_2nd_order[:, 1::]) / 3)
                    #         self.coeffs_3rd_order = (self.coeffs_2nd_order[:, 1::] - self.coeffs_2nd_order[:, 0: -1]) / (3 * self.step_size)

                    #     def get_parameters(self) -> torch.Tensor:
                    #         return self.nodal_values.data

                    #     def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
                    #         # if self.recompute_coefficients:
                    #         #     self._fit()
                    #         #     self.recompute_coefficients = False

                    #         index_tensor = bucketise(x, self.box_lower, self.box_upper, self.num_nodes)
                    #         y = x - self.nodes[index_tensor]

                    #         bs, f, w, h = x.shape
                    #         idx_flat = index_tensor.view(bs, f, w, h).permute(1, 0, 2, 3).reshape(f, -1)

                    #         a = self.nodal_values.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
                    #         b = self.coeffs_1st_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
                    #         c = self.coeffs_2nd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
                    #         d = self.coeffs_3rd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)

                    #         neg_log_pot = a + y * (b + y * (c + y * d))

                    #         # neg_log_pot = NaturalCubicSplineAutogradFunction.apply(self.nodes, self.nodal_values,
                    #         #                                                        self.coeffs_1st_order,
                    #         #                                                        self.coeffs_2nd_order, self.coeffs_3rd_order,
                    #         #                                                        self.box_lower, self.box_upper, self.num_nodes, x)[0]
                    #         if reduce:
                    #             return torch.sum(neg_log_pot)
                    #         else:
                    #             return neg_log_pot

                    #     def first_derivative(self, x):
                    #         index_tensor = bucketise(x, self.box_lower, self.box_upper, self.num_nodes)
                    #         y = x - self.nodes[index_tensor]

                    #         bs, f, w, h = x.shape
                    #         idx_flat = index_tensor.view(bs, f, w, h).permute(1, 0, 2, 3).reshape(f, -1)

                    #         a = self.nodal_values.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
                    #         b = self.coeffs_1st_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
                    #         c = self.coeffs_2nd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)
                    #         d = self.coeffs_3rd_order.gather(1, idx_flat).view(f, bs, w, h).permute(1, 0, 2, 3)

                    #         return b + y * (c + y * d)



                    #                 # def first_derivative(self, x: torch.Tensor, grad_outputs: Optional[torch.Tensor]=None) -> torch.Tensor:
                    #                 #     return NaturalCubicSplineAutogradFunction.apply(self.nodes, self.nodal_values,
                    #                 #                                                     self.coeffs_1st_order,
                    #                 #                                                     self.coeffs_2nd_order, self.coeffs_3rd_order,
                    #                 #                                                     self.box_lower, self.box_upper, self.num_nodes,
                    #                 #                                                     x, grad_outputs, False, 1)[0]
                    #                 #
                    #                 # def second_derivative(self, x: torch.Tensor, grad_outputs: Optional[torch.Tensor]=None,
                    #                 #                       mixed: bool=False, retain_graph: bool=True) -> torch.Tensor:
                    #                 #     if not mixed:
                    #                 #         return NaturalCubicSplineAutogradFunction.apply(self.nodes, self.nodal_values,
                    #                 #                                                         self.coeffs_1st_order,
                    #                 #                                                         self.coeffs_2nd_order, self.coeffs_3rd_order,
                    #                 #                                                         self.box_lower, self.box_upper, self.num_nodes,
                    #                 #                                                         x, grad_outputs, retain_graph, 2)[0]
                    #                 #     else:
                    #                 #         return NaturalCubicSplineAutogradFunction.apply(self.nodes, self.nodal_values,
                    #                 #                                                         self.coeffs_1st_order,
                    #                 #                                                         self.coeffs_2nd_order, self.coeffs_3rd_order,
                    #                 #                                                         self.box_lower, self.box_upper, self.num_nodes,
                    #                 #                                                         x, grad_outputs, retain_graph, 2)[1]

                    #     def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
                    #         state = super().state_dict(*args, kwargs)
                    #         return state

                    #     def initialisation_dict(self) -> Dict[str, Any]:
                    #         return {'num_marginals': self.num_marginals, 'num_nodes': self.num_nodes,
                    #                 'bow_lower': self.box_lower, 'box_upper': self.box_upper}

                    #     def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
                    #         potential_data = torch.load(path_to_model, map_location=device)

                    #         initialisation_dict = potential_data['initialisation_dict']
                    #         self.num_marginals = initialisation_dict['num_marginals']
                    #         self.num_nodes = initialisation_dict['num_nodes']
                    #         self.box_lower = initialisation_dict['box_lower']
                    #         self.box_upper = initialisation_dict['box_upper']

                    #         state_dict = potential_data['state_dict']
                    #         self.load_state_dict(state_dict)


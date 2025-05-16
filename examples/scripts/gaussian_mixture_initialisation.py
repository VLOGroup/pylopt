import matplotlib.pyplot as plt
import torch

from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.gaussian_mixture_model.GaussianMixtureModel import GaussianMixtureModel
from bilevel_optimisation.projection.ParameterProjections import unit_simplex_projection

def main():

    num_components = 125
    box_lower = - 123
    box_upper = 2
    dummy_weights_spec = ParamSpec(torch.ones(num_components), trainable=False, projection=None)
    gmm = GaussianMixtureModel(num_components, box_lower, box_upper, dummy_weights_spec)

    centers = gmm.centers
    variances = gmm.variances

    w = torch.sqrt(2 * torch.pi * variances) # torch.exp(-2 * torch.abs(centers.reshape(1, -1))) * torch.sqrt(2 * torch.pi * variances)
    # w_proj = unit_simplex_projection(w.squeeze())
    w_proj = torch.nn.functional.softmax(w)

    import numpy as np
    alpha = 1000
    w = (0.1 * np.sqrt(alpha)) / (1 + alpha * centers ** 2)

    print(torch.sum(w))


    gmm._weights.data.copy_(w.squeeze())
    #
    t = torch.linspace(-2, 2, 71)
    y = gmm(t)


    yy = gmm(centers)

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    # ax_1.plot(t.detach().cpu().numpy(), y.detach().cpu().numpy())
    # ax_1.plot(centers.detach().cpu().numpy(), w.detach().cpu().numpy(), color='cyan')
    # ax_1.plot(centers.detach().cpu().numpy(), (1 / (1 + centers ** 2)).detach().cpu().numpy(), color='magenta')
    # ax_1.plot(centers.detach().cpu().numpy(), torch.exp(-torch.abs(centers)).detach().cpu().numpy(), color='green')

    component_list = []

    for j in range(0, num_components):
        mu = centers[j].detach().cpu().numpy()
        var = variances[j].detach().cpu().numpy()
        scale = (1 / np.sqrt(2 * np.pi * var))
        y = scale * (1 / (1 + centers[j] ** 2)) * np.exp((-0.5 / var) * (t - mu) ** 2)

        # ax_1.plot(t, y / scale)

        component_list.append(y / scale)
    #
    # mu_1 = centers[1].detach().cpu().numpy()
    # var_1 = variances[1].detach().cpu().numpy()
    # scale_1 = (1 / np.sqrt(2 * np.pi * var_1))
    # y_1 = scale_1 * np.exp((-0.5 / var_1) * (t - mu_1) ** 2)
    #
    # mu_2 = centers[2].detach().cpu().numpy()
    # var_2 = variances[2].detach().cpu().numpy()
    # scale_2 = (1 / np.sqrt(2 * np.pi * var_2))
    # y_2 = scale_2 * np.exp((-0.5 / var_2) * (t - mu_2) ** 2)

    comp = torch.stack(component_list, dim=0)
    approx = torch.sum(comp, dim=0)
    approx_proj = torch.nn.functional.softmax(approx)
    ax_1.plot(t, approx.detach().cpu().numpy())
    ax_1.plot(t, 40 * approx_proj.detach().cpu().numpy(), color='cyan')
    student_t = 1 / (1 + t ** 2)
    ax_1.plot(t, student_t.detach().cpu().numpy(), color='magenta')

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.plot(t.detach().cpu().numpy(), -torch.log(approx).detach().cpu().numpy())
    ax_2.plot(t.detach().cpu().numpy(), torch.abs(t).detach().cpu().numpy(), color='green')
    ax_2.plot(t.detach().cpu().numpy(), torch.log(1 + t ** 2).detach().cpu().numpy(), color='magenta')

    plt.show()




if __name__ == '__main__':
    main()
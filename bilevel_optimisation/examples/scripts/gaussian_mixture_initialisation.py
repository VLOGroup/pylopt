import matplotlib.pyplot as plt
import torch
import numpy as np

from bilevel_optimisation.gaussian_mixture_model.GaussianMixtureModel import GaussianMixtureModel
from bilevel_optimisation.potential.StudentT import StudentT

def plot_single_gaussian_components(gmm: GaussianMixtureModel, num_component_plots: int = 10):
    num_components = gmm.get_number_of_components()

    steps = 151
    t = torch.linspace(-3, 3, steps=steps)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for j in range(0, np.minimum(num_component_plots, num_components)):
        y = gmm.forward_single_component(t, j)
        ax.plot(t.numpy(), y.detach().numpy(), color='blue', label='component {:d}'.format(j))

    fig.legend()
    plt.show()
    plt.close(fig)

def plot_neg_log_mixture(gmm: GaussianMixtureModel):
    potential_student_t = StudentT()

    steps = 151
    t = torch.linspace(-3, 3, steps=steps)
    y = gmm.forward(t)
    y_neg_log = gmm.forward_negative_log(t)

    fig = plt.figure()

    ax_1 = fig.add_subplot(1, 2, 1)

    ax_1.plot(t.numpy(), y.detach().numpy(), label='gmm', color='blue')
    ax_1.plot(t.numpy(), potential_student_t(t).detach().numpy(), label='student t', color='orange')
    ax_1.legend()

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.plot(t.numpy(), y_neg_log.detach().numpy() - torch.min(y_neg_log).detach().numpy(), label='neg log gmm', color='blue')
    ax_2.plot(t.numpy(), -torch.log(potential_student_t(t)).detach().numpy(),
              label='neg log student t', color='orange')
    ax_2.plot(t.numpy(), torch.torch.abs(t).detach().numpy(), label='absolute value', color='magenta')
    ax_2.legend()

    plt.show()
    plt.close(fig)

def main():

    num_components = 125
    box_lower = -3
    box_upper = 3
    weights = 2 * torch.ones(num_components) - 1
    centers = torch.linspace(start=box_lower, end=box_upper, steps=num_components)
    variance = torch.tensor((2 * (box_upper - box_lower) / (num_components - 1)) ** 2)
    gmm = GaussianMixtureModel(weights=torch.nn.functional.softmax(weights, dim=0),
                                         centers=centers, variance=variance,
                                         box_lower=box_lower, box_upper=box_upper)

    plot_single_gaussian_components(gmm, num_component_plots=10)
    plot_neg_log_mixture(gmm)

if __name__ == '__main__':
    main()
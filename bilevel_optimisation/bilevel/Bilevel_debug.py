import glob
import torch
import torch.nn.functional as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import random
import time

from bilevel_optimisation.solver.CGSolver import CGSolver


def psnr(u, g):
    return 20 * np.log10(1.0 / np.sqrt(np.mean((u - g)**2)))

def bilevel_learn(inner_energy, maxit=1000):
    sigma = 0.1

    test_img_clean = ski.io.imread('/home/florianthaler/Documents/data/image_data/some_images/watercastle.jpg') / 255.0
    test_img_clean = test_img_clean.mean(-1).astype(np.float32)
    test_img_clean = torch.from_numpy(test_img_clean).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

    test_img_noisy = test_img_clean + torch.randn_like(test_img_clean) * sigma

    thetas = inner_energy._regulariser.filter_weights.data.clone()
    filters = inner_energy._regulariser.filters.data.clone()

    thetas_old = inner_energy._regulariser.filter_weights.data.clone()
    filters_old = inner_energy._regulariser.filters.data.clone()

    filters_list = []
    thetas_list = []

    solver = CGSolver(max_num_iterations=500, rel_tol=1e-5)

    with torch.no_grad():
        Lip = 10000000
        for it in range(1):

            thetas_i = thetas + 0.71 * (thetas - thetas_old)
            filters_i = filters + 0.71 * (filters - filters_old)

            # clean = get_sample(images).cuda()
            # noisy = clean + torch.randn_like(clean) * sigma

            # debug
            clean = torch.load(
                '/home/florianthaler/Documents/research/stochastic_bilevel_optimisation/data/data_batches/batch_clean_{:d}.pt'.format(
                    0)).to(device=torch.device('cuda:0'), dtype=torch.float32)
            noisy = torch.load(
                '/home/florianthaler/Documents/research/stochastic_bilevel_optimisation/data/data_batches/batch_noisy_{:d}.pt'.format(
                    0)).to(device=torch.device('cuda:0'), dtype=torch.float32)

            # 1. solve for \nabla_u E(u)=0
            # denoised = foe_apgd(noisy, thetas_i, filters_i, verbose=0, maxit=1000)

            inner_energy._regulariser.filters.data.copy_(filters_i)
            inner_energy._regulariser.filter_weights.data.copy_(thetas_i)
            denoised = inner_energy.argmin(noisy)

            # 2. solve for the Lagrange multipliers
            lin_operator = lambda z: inner_energy.hvp_state(denoised, z)
            lagrange_multiplier_result = solver.solve(lin_operator, clean - denoised)
            lagrange = lagrange_multiplier_result.solution

            # 3. compute gradient update for filetrs, thetas

            grad_filters, grad_thetas = inner_energy.hvp_mixed(denoised, lagrange)
            loss = 0.5 * torch.sum((clean - denoised) ** 2)

            thetas_old = thetas.clone()
            filters_old = filters.clone()

            for bt in range(10):

                step_size = 1 / Lip  # constant ...

                thetas = thetas_i - step_size * grad_thetas
                thetas = torch.clamp(thetas, min=0.0)

                filters = filters_i - step_size * grad_filters
                filters -= torch.mean(filters, axis=(2, 3), keepdims=True)

                inner_energy._regulariser.filters.data.copy_(filters)
                inner_energy._regulariser.filter_weights.data.copy_(thetas)
                denoised = inner_energy.argmin(noisy)

                loss_new = 0.5 * torch.sum((clean - denoised) ** 2)

                quad = loss + torch.sum(grad_thetas * (thetas - thetas_i)) + \
                       torch.sum(grad_filters * (filters - filters_i)) + \
                       Lip / 2.0 * torch.sum((thetas - thetas_i) ** 2) + \
                       Lip / 2.0 * torch.sum((filters - filters_i) ** 2)

                if loss_new <= quad:
                    Lip = Lip / 2.0
                    break
                else:
                    Lip = Lip * 2.0

            # filters_list.append(filters.detach().clone())
            # thetas_list.append(thetas.detach().clone())

            # test_img_denoised = inner_energy.argmin_debug(test_img_noisy, thetas, filters)
            test_img_denoised = inner_energy.argmin(test_img_noisy)

            psnr_ = psnr(test_img_denoised.cpu().numpy(), test_img_clean.cpu().numpy())
            print("iter = ", it,
                  ", Lip = ", "{:3.3f}".format(Lip),
                  ", Loss = ", "{:3.3f}".format(loss.cpu().numpy()),
                  ", step_size = ", "{:3.7f}".format(step_size),
                  ", psnr = ", "{:3.3f}".format(psnr_),
                  ", norm_grad_filter_0 = ",
                  "{:3.3f}".format(torch.linalg.norm(grad_filters[0, :, :, :]).detach().cpu().item()),
                  ", norm_grad_filter_1 = ",
                  "{:3.3f}".format(torch.linalg.norm(grad_filters[1, :, :, :]).detach().cpu().item()),
                  ", norm_grad_filter_2 = ",
                  "{:3.3f}".format(torch.linalg.norm(grad_filters[2, :, :, :]).detach().cpu().item()),
                  end="\n")

        # fig = plt.figure()
        # ax_filter_norms = fig.add_subplot(1, 2, 1)
        # filter_norm_list = [torch.sum(f ** 2, dim=(-2, -1)).squeeze().detach().cpu().numpy() for f in filters_list]
        # ax_filter_norms.plot(np.arange(0, len(filter_norm_list)), filter_norm_list)
        #
        # ax_thetas = fig.add_subplot(1, 2, 2)
        # thetas_list_ = [theta.detach().cpu().numpy() for theta in thetas_list]
        # ax_thetas.plot(np.arange(0, len(thetas_list)), thetas_list_)
        #
        # plt.show()

    return loss, thetas, filters, denoised

# if __name__ == '__main__':
#     np.random.seed(123)
#     torch.manual_seed(123)
#     random.seed(123)
#     from scipy.fftpack import idct
#
#     K = 48
#     Nf = 7
#
#     # ### dct initialisation
#     # np_basis = np.reshape(np.eye(Nf ** 2, dtype=np.float32), (Nf**2, Nf, Nf))
#     # np_basis = idct(idct(np_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
#     # np_basis = np_basis[1:].reshape(-1, 1, Nf, Nf)
#     # filters = torch.from_numpy(np_basis[:K])*25
#     # filters = filters.cuda()
#
#     ### pretrained filters
#     # pretrained_filters_path = '/home/florianthaler/Documents/data/models/foe_models/foe_filters_7x7_chen-ranftl-pock_2014.pt'
#     # filters = torch.load(pretrained_filters_path)
#     # filters = filters.cuda()
#     # filters = filters * 255
#
#     # ### uniform initialisation
#     # filters = # 2 * torch.rand(K, 1, Nf, Nf).cuda() - 1  # torch.randn(K,1,Nf,Nf).cuda()/100
#     filters = torch.load(
#         '/home/florianthaler/Documents/research/stochastic_bilevel_optimisation/data/'
#         'pretrained_models/foe_filters_7x7_chen-ranftl-pock_2014.pt').to(device=torch.device('cuda:0'),
#                                                                          dtype=torch.float64)
#     filters = filters * 255
#     filters = filters.to(device=torch.device('cuda:0'), dtype=torch.float32)
#
#     # pretrained_filter_weights_path = '/home/florianthaler/Documents/data/models/foe_models/foe_filter-weights_7x7_chen-ranftl-pock_2014_normalised.pt'
#     # thetas = torch.load(pretrained_filter_weights_path)
#     # thetas = thetas.cuda()
#
#     images = load_images_from_directory('/home/florianthaler/Documents/data/image_data/BSDS300/images/train')
#
#
#     # thetas = torch.load(
#     #     '/home/florianthaler/Documents/research/stochastic_bilevel_optimisation/data/pretrained_models/random_weights.pt').to(device=torch.device('cuda:0'), dtype=torch.float64)
#     # thetas = thetas.to(device=torch.device('cuda:0'), dtype=torch.float64)
#     thetas = 1e-5 * torch.ones(48)
#     thetas = thetas.to(device=torch.device('cuda:0'), dtype=torch.float32)
#
#     thetas, filters, denoised = bilevel_learn(thetas, filters, maxit=1)


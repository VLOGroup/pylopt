import torch

def solve_tridiagonal(diag: torch.Tensor, diag_super: torch.Tensor, diag_sub: torch.Tensor,
                      rhs: torch.Tensor) -> torch.Tensor:
    """
    This function solves a tridiagonal linear system given a single system matrix, and multiple right hand sides.
    For the solution, the Thomas algorithm is applied - which corresponds basically to the application of
    Gaussian elimination while exploiting the special structure of the system matrix

    NOTE
    ----
        > Regularity of the system matrix is not checked. It is assumed that the system matrix
            is indeed regular.
        > The input tensors diag, rhs will NOTE (!) be altered - altering these would lower memory consumption though.

    :param diag: PyTorch tensor representing the (main) diagonal of the system matrix
    :param diag_super: PyTorch tensor representing the super diagonal of the system matrix
    :param diag_sub: PyTorch tensor representing the sub diagonal of the system matrix
    :param rhs: Two-dimensional PyTorch tensor, where dimension 0 is assumed to be the batch dimension; i.e. if
        the system has n unknowns, and it shall be solved for m right hand sides, the tensor rhs is assumed to
        be of shape [m, n].
    :return: PyTorch tensor representing the solution tensor. It has the same shape as the right hand side tensor
        rhs.
    """
    diag_ = torch.clone(diag)
    rhs_ = torch.clone(rhs)

    n = rhs_.shape[1]
    for i in range(1, n):
        tmp = diag_sub[i - 1] / diag_[i - 1]
        diag_[i] = diag_[i] - tmp * diag_super[i - 1]
        rhs_[:, i] = rhs_[:, i] - tmp * rhs_[:, i - 1]

    x = torch.empty_like(rhs)
    x[:, n - 1] = rhs_[:, -1] / diag_[-1]
    for i in range(n - 2, -1, -1):
        x[:, i] = (rhs_[:, i] - diag_super[i] * x[:, i + 1]) / diag_[i]
    return x
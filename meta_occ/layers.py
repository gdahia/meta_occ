import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn


def svdd_qp(dim):
    kernel_sqrt = cp.Parameter((dim, dim))
    kernel_diag = cp.Parameter(dim)

    alpha = cp.Variable(dim)
    expr = cp.sum_squares(kernel_sqrt @ alpha) - kernel_diag.T @ alpha

    objective = cp.Minimize(expr)
    constraints = [cp.sum(alpha) == 1, alpha >= 0]
    problem = cp.Problem(objective, constraints)

    return problem, (kernel_sqrt, kernel_diag), (alpha,)


class SVDDLayer(nn.Module):
    def __init__(self, shot, dim=1, eps=1e-6):
        super(SVDDLayer, self).__init__()
        self._dim = dim
        self._eps = eps

        problem, parameters, variables = svdd_qp(shot)
        self._qp_layer = CvxpyLayer(problem, parameters=parameters, variables=variables)

    def forward(self, inputs):
        # TODO: eps?

        kernel_matrices = torch.bmm(inputs, inputs.transpose(1, 2))
        kernel_matrices_sqrt = torch.cholesky(kernel_matrices)
        # TODO: is cholesky required or can we use the more natural approach with matmul?
        kernel_matrices_diags = torch.diagonal(kernel_matrices, dim1=-2, dim2=-1)
        (alphas,) = self._qp_layer(
            kernel_matrices_sqrt,
            kernel_matrices_diags,
            solver_args={"solve_method": "ECOS"},
        )

        alphas = alphas.unsqueeze(-1)
        centers = torch.sum(alphas * inputs, dim=self._dim)
        # `keepdim=True` here to avoid unsqueezing in `CentersDistance`, which
        # could be used for vanilla protonet?

        return centers


class PrototypeLayer(nn.Module):
    def __init__(self, dim=1):
        super(PrototypeLayer, self).__init__()
        self._dim = dim
        # TODO: does this work for vanilla prototypical networks?

    def forward(self, inputs):
        prototypes = torch.mean(inputs, dim=self._dim)
        return prototypes


class CentersDistance(nn.Module):
    def __init__(self, dim=-1):
        super(CentersDistance, self).__init__()
        self._dim = dim

    def forward(self, inputs, centers):
        logits = -torch.sum((centers.unsqueeze(1) - inputs) ** 2, dim=self._dim)
        # if `keepdim=True` up there, remove unsqueeze here
        return logits

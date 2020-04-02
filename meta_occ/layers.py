import torch
from torch import nn

from qpth.qp import QPFunction


class SVDDLayer(nn.Module):
    def __init__(self, shot, dim=1, eps=1e-6):
        super(SVDDLayer, self).__init__()
        self._dim = dim
        self._eps = eps

    def forward(self, inputs):
        shot = inputs.shape[1]

        kernel_matrices = torch.bmm(inputs, inputs.transpose(1, 2))
        kernel_matrices += self._eps * torch.eye(shot)
        kernel_diags = torch.diagonal(kernel_matrices, dim1=-2, dim2=-1)
        Q = 2 * kernel_matrices
        p = -kernel_diags
        A = torch.ones(1, shot)
        b = torch.ones(1)
        G = -torch.eye(shot)
        h = torch.zeros(shot)
        alphas = QPFunction(verbose=False)(
            Q,
            p,
            G.detach(),
            h.detach(),
            A.detach(),
            b.detach(),
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
        logits = -torch.sum((centers.unsqueeze(1) - inputs)**2, dim=self._dim)
        # if `keepdim=True` up there, remove unsqueeze here
        return logits

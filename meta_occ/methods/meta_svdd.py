import torch
from torch import nn

from qpth.qp import QPFunction


def batch_svdd(inputs: torch.Tensor, eps: float) -> torch.Tensor:
    shot = inputs.shape[1]

    kernel_matrices = torch.bmm(inputs, inputs.transpose(1, 2))
    kernel_matrices += eps * torch.eye(shot)
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
    centers = torch.sum(alphas * inputs, dim=1)

    return centers


class MetaSVDD(nn.Module):
    def __init__(self, embedding_net: nn.Module, eps: float = 1e-6) -> None:
        super(MetaSVDD, self).__init__()
        self._embedding_net = embedding_net
        self._eps = eps
        self._loss = nn.BCEWithLogitsLoss()

    def forward(self, support_inputs, query_inputs):
        support_embeddings = self._embedding_net(support_inputs)
        centers = batch_svdd(support_embeddings, self._eps)

        query_embeddings = self._embedding_net(query_inputs)
        logits = -torch.sum(
            (centers.unsqueeze(1) - query_embeddings)**2, dim=-1)
        return logits

    def loss(self, support_inputs: torch.Tensor, query_inputs: torch.Tensor,
             query_labels: torch.Tensor) -> torch.Tensor:
        logits = self(support_inputs, query_inputs)
        return self._loss(logits, query_labels)

    def infer(self, support_inputs, query_inputs):
        logits = self(support_inputs, query_inputs)
        probs = 1.0 + torch.tanh(logits)
        return probs

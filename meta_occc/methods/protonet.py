import torch
from torch import nn


class OneClassPrototypicalNet(nn.Module):
    def __init__(self, embedding_net: nn.Module) -> None:
        super(OneClassPrototypicalNet, self).__init__()
        self._embedding_net = embedding_net
        self._loss = nn.BCEWithLogitsLoss()

    def forward(self, support_inputs, query_inputs):
        support_embeddings = self._embedding_net(support_inputs)
        prototypes = torch.mean(support_embeddings, dim=1)

        query_embeddings = self._embedding_net(query_inputs)
        logits = -torch.sum(
            (prototypes.unsqueeze(1) - query_embeddings)**2, dim=-1)
        return logits

    def loss(self, support_inputs: torch.Tensor, query_inputs: torch.Tensor,
             query_labels: torch.Tensor) -> torch.Tensor:
        logits = self(support_inputs, query_inputs)
        return self._loss(logits, query_labels)

    def infer(self, support_inputs, query_inputs):
        logits = self(support_inputs, query_inputs)
        probs = 1.0 + torch.tanh(logits)
        return probs

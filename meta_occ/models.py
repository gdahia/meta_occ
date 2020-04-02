import torch
from torch import nn

from meta_occ.layers import CentersDistance


def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2),
        nn.ReLU(),
    )


class EmbeddingNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_size: int = 64) -> None:
        super(EmbeddingNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels),
        )

    def forward(self, inputs):
        reshaped = inputs.reshape(-1, *inputs.shape[2:])
        embeddings = self.net(reshaped)
        outputs = embeddings.view(*inputs.shape[:2], -1)
        return outputs


class MetaOCCModel(nn.Module):
    def __init__(self, embedding_net, occ_layer):
        super(MetaOCCModel, self).__init__()
        self._embedding_net = embedding_net
        self._net = nn.Sequential(embedding_net, occ_layer)
        self._to_logits = CentersDistance()

    def forward(self, support_inputs, query_inputs):
        centers = self._net(support_inputs)
        query_embeddings = self._embedding_net(query_inputs)
        logits = self._to_logits(query_embeddings, centers)
        return logits

    def infer(self, support_inputs, query_inputs):
        logits = self(support_inputs, query_inputs)
        probs = 1.0 + torch.tanh(logits)
        return probs

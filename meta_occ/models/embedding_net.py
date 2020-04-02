from torch import nn


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

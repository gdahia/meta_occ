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
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            conv3x3(in_channels, out_channels),
            conv3x3(out_channels, out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.maxpool = nn.MaxPool2d(stride)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.layer(inputs) + self.projection(inputs)
        output = self.maxpool(output)
        output = self.relu(output)
        return output


class ResNet12(nn.Module):
    """TODO"""
    def __init__(self, in_channels: int, avgpool: bool = False) -> None:
        super(ResNet12, self).__init__()
        self.net = nn.Sequential(
            ResidualBlock(in_channels, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),
        )

    def forward(self, inputs):
        reshaped = inputs.reshape(-1, *inputs.shape[2:])
        embeddings = self.net(reshaped)
        outputs = embeddings.view(*inputs.shape[:2], -1)
        return outputs

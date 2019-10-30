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
    """The neural network architecture commonly used in meta-learning research.

    This is the neural network architecture originally proposed in "Matching
    Networks for One Shot Learning" (Vinyals et al., 2016) and that is commonly
    used in meta-learning research ("Model-agnostic meta-learning for fast
    adaptation of deep networks", Finn et al., 2017; "Prototypical networks for
    few-shot learning", Snell et al., 2017; "Optimization as a model for
    few-shot learning", Ravi & Larochelle, 2017). It has four convolutional
    blocks, each comprised of a convolutional layer with a 3x3 kernel,
    `hidden_size` filters, and "same" padding, batch normalization, ReLU and
    max-pooling.

    Parameters
    ----------
    in_channels:
        Number of channels in the input data.
    out_channels:
        Number of channels in the output.
    hidden_size:
        Number of channels in the hidden layers. (Default: `64`)
    """
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
        """Compute the embedding for the given inputs.

        Parameters
        ----------
        inputs:
            A tensor of shape `batch_dims + (channels, height, width)`, where
            `batch_dims` are extra batch dimensions, with the inputs to embed.

        Returns
        -------
        outputs:
            A tensor of shape `batch_dims + (out_channels,)` with the
            embeddings for the inputs.
        """
        reshaped = inputs.reshape(-1, *inputs.shape[2:])
        embeddings = self.net(reshaped)
        outputs = embeddings.view(*inputs.shape[:2], -1)
        return outputs

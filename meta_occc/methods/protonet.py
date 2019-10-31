import torch
from torch import nn


class OneClassPrototypicalNet(nn.Module):
    """One-class prototypical network module.

    Given support and query examples, this module first computes embeddings for
    them using the `embedding_net`. Then, prototypes are computed with the
    examples in the support set with `get_prototypes`, and the probabilities
    for the examples in the query set are computed from their distance to the
    prototypes.

    Parameters
    ----------
    embedding_net:
        `nn.Module` with which to generate the embeddings of the examples in
        the support set.
    """
    def __init__(self, embedding_net: nn.Module) -> None:
        super(OneClassPrototypicalNet, self).__init__()
        self._embedding_net = embedding_net
        self._loss = nn.BCEWithLogitsLoss()

    def forward(self, support_inputs, query_inputs):
        """Compute logits for the query set with the support's prototypes.

        The `embedding_net` is used to compute embeddings for both support and
        query set examples. Then, prototypes for each support set are computed
        as the average of their vectors. Last, the logit for each query set
        example is computed as minus its square distance to the prototype.

        In more details, the support set examples are given in a tensor of
        shape `(batch_size, shot, channels, height, width)`. `batch_size` is
        the number of episodes that should be processed in this forward; each
        episode is, apart from batch normalization, independent of the other
        episodes. `shot` is the size of the support set. Each episode contains
        a single support set with all its elements belonging to the same class.
        The query set, in a tensor of shape `(batch_size, query_size, channels,
        height, width)`, are similarly indexed: the `batch_size` dimension
        indexes the set's episode, and the `query_size` dimension indexes the
        episode's examples. The resulting logits are of shape `(batch_size,
        query_size)`, with entry `logits[i, j]` indicating the unnormalized
        probability that the `j`-th example in the query set should be
        classified with the same label as the examples in the `i`-th support
        set.

        Parameters
        ----------
        support_inputs:
            A tensor of shape `(batch_size, shot, channels, height, width)`
            with the support set inputs.
        query_inputs:
            A tensor of shape `(batch_size, query_size, channels, height,
            width)` with the query set inputs.

        Returns
        -------
        logits:
            A tensor of shape `(batch_size, query_size)` containing the logits
            for the one-class prototypical network.
        """
        support_embeddings = self._embedding_net(support_inputs)
        prototypes = torch.mean(support_embeddings, dim=1)

        query_embeddings = self._embedding_net(query_inputs)
        logits = -torch.sum(
            (prototypes.unsqueeze(1) - query_embeddings)**2, dim=-1)
        return logits

    def loss(self, support_inputs: torch.Tensor, query_inputs: torch.Tensor,
             query_labels: torch.Tensor) -> torch.Tensor:
        """Compute the binary cross-entropy one-class prototypical loss.

        After obtaining the logit for each example in the query set using the
        prototypes computed from the corresponding support set, the loss is
        computed as the binary cross-entropy between the normalized
        probabilities and the provided class labels.

        Parameters
        ----------
        support_inputs:
            A tensor of shape `(batch_size, shot, channels, height, width)`
            with the support set inputs.
        query_inputs:
            A tensor of shape `(batch_size, query_size, channels, height,
            width)` with the query set inputs.
        query_labels:
            A tensor of shape `(batch_size, query_size)` where `query_labels[i,
            j]` is `1` if `query_inputs[i, j]` is from the same class of
            `support_inputs[i]`, and `0` otherwise.

        Returns
        -------
            The value of the binary cross-entropy one-class prototypical loss.
        """
        logits = self(support_inputs, query_inputs)
        return self._loss(logits, query_labels)

    def infer(self, support_inputs, query_inputs):
        """Compute the one-class classification probability for the query inputs.

        The probability of the inputs in the query set is computed with regards
        to the support set inputs using the one-class prototypical network.
        Since the output of the negative squared distance is always a
        non-positive number, the output of the `tanh` is also a non-positive
        number in the range `[-1, 0]`. To get valid probabilities, then, we add
        `1.0` to this output.

        Specifically, the output is a tensor of shape `(batch_size,
        query_size)`, where `probs[i, j]` indicates the probability that the
        `j`-th element of the `i`-th query set should be classified with the
        same label as the corresponding support set.

        Parameters
        ----------
        support_inputs:
            A tensor of shape `(batch_size, shot, channels, height, width)`
            with the support set inputs.
        query_inputs:
            A tensor of shape `(batch_size, query_size, channels, height,
            width)` with the query set inputs.

        Returns
        -------
        probs:
            A tensor of shape `(batch_size, query_size)` containing the
            probabilities that the query set inputs should be classified with
            the same label as the support set inputs.
        """
        logits = self(support_inputs, query_inputs)
        probs = 1.0 + torch.tanh(logits)
        return probs

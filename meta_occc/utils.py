from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from torchmeta.datasets.helpers import (
    omniglot,
    miniimagenet,
    tieredimagenet,
    cifar_fs,
)
from torchmeta.utils.data import BatchMetaDataLoader


# TODO: create test for `to_one_class_batch`
def to_one_class_batch(batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                       shot: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO"""
    support_inputs, support_labels = batch['train']
    query_inputs, query_labels = batch['test']

    support_inputs = support_inputs[:, :shot]
    support_labels = support_labels[:, 0]

    # TODO: accuracy-wise sampling: get half of the query size from the same \\
    # class, divide the remaining examples among the other classes

    query_labels = query_labels.eq(
        support_labels.unsqueeze(-1).expand_as(query_labels)).long()

    return support_inputs, query_inputs, query_labels


def evaluate(model,
             loader: Iterable[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
             total_episodes: int,
             shot: int,
             device: Optional[str] = None) -> Tuple[float, float]:
    """TODO"""
    model.train(False)
    accs: List[float] = []
    for val_batch in loader:
        (support_inputs, query_inputs,
         query_labels) = to_one_class_batch(val_batch, shot)

        if device:
            support_inputs = support_inputs.to(device=device)
            query_inputs = query_inputs.to(device=device)
            query_labels = query_labels.to(device=device)

        probs = model.infer(support_inputs, query_inputs)
        preds = (probs >= 0.5).long()
        correct = preds.eq(query_labels)
        batch_accs = torch.mean(correct.float(), dim=1).detach().cpu().numpy()

        episodes_so_far = len(accs)
        if episodes_so_far + len(batch_accs) < total_episodes:
            accs.extend(batch_accs)
        else:
            rem = total_episodes - episodes_so_far
            accs.extend(batch_accs[:rem])
            break
    model.train(True)

    mean = 100 * np.mean(accs)
    std = 100 * np.std(accs)
    ci95 = 1.96 * std / np.sqrt(len(accs))

    return mean, ci95


def get_dataset(dataset_id: str,
                folder: str,
                shot: int,
                query_size: int,
                batch_size: int,
                shuffle: bool,
                train: bool = False,
                val: bool = False,
                test: bool = False
                ) -> Iterable[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """TODO"""
    datasets = {
        'omniglot': omniglot,
        'miniimagenet': miniimagenet,
        'tieredimagenet': tieredimagenet,
        'cifar_fs': cifar_fs
    }
    if dataset_id not in datasets:
        raise KeyError(
            f'"{dataset_id}" not recognized. Options are "omniglot",'
            '"miniimagenet", "tieredimagenet", and "cifar_fs"')

    dataset = datasets[dataset_id](folder,
                                   shot,
                                   2,
                                   shuffle=shuffle,
                                   meta_train=train,
                                   meta_val=val,
                                   meta_test=test,
                                   test_shots=query_size // 2,
                                   download=True)
    loader = BatchMetaDataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)

    return loader

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score

from torchmeta.datasets.helpers import (
    omniglot,
    miniimagenet,
    tieredimagenet,
    cifar_fs,
)
from torchmeta.utils.data import BatchMetaDataLoader, CombinationMetaDataset


def to_one_class_batch(batch, shot):
    support_inputs, support_labels = batch['train']
    query_inputs, query_labels = batch['test']

    if len(support_inputs.shape) == 4:
        support_inputs = support_inputs.unsqueeze(0)
        support_labels = support_labels.unsqueeze(0)

        query_inputs = query_inputs.unsqueeze(0)
        query_labels = query_labels.unsqueeze(0)

    support_inputs = support_inputs[:, :shot]
    support_labels = support_labels[:, 0]

    query_labels = query_labels.eq(
        support_labels.unsqueeze(-1).expand_as(query_labels)).long()

    return support_inputs, query_inputs, query_labels


def evaluate(model, loader, total_episodes, shot, device=None):
    accs = []
    while len(accs) < total_episodes:
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
            batch_accs = torch.mean(correct.float(),
                                    dim=1).detach().cpu().numpy()

            episodes_so_far = len(accs)
            if episodes_so_far + len(batch_accs) < total_episodes:
                accs.extend(batch_accs)
            else:
                rem = total_episodes - episodes_so_far
                accs.extend(batch_accs[:rem])
                break

    mean = np.mean(accs)
    std = np.std(accs)
    ci95 = 1.96 * std / np.sqrt(len(accs))

    return mean, ci95


def get_dataset(dataset_id: str,
                folder: str,
                shot: int,
                query_size: int,
                shuffle: bool,
                train: bool = False,
                val: bool = False,
                test: bool = False) -> CombinationMetaDataset:
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
    return dataset


def get_dataset_loader(dataset_id,
                       folder,
                       shot,
                       query_size,
                       batch_size,
                       shuffle,
                       train=False,
                       val=False,
                       test=False):
    dataset = get_dataset(dataset_id, folder, shot, query_size, shuffle, train,
                          val, test)
    loader = BatchMetaDataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)

    return loader


def collate_task(task):
    return default_collate([task[idx] for idx in range(len(task))])


def auc(model, dataset, episodes_per_class, shot, device):
    auc_means = []
    auc_stds = []
    n_classes = len(dataset.dataset)
    classes = range(n_classes)

    for i in classes:
        aucs = []
        for _ in range(episodes_per_class):
            probs = []
            labels = []
            for j in classes:
                if i != j:
                    batch = dataset[i, j]
                    batch['train'] = collate_task(batch['train'])
                    batch['test'] = collate_task(batch['test'])
                    (support_inputs, query_inputs,
                     query_labels) = to_one_class_batch(batch, shot)

                    if device:
                        support_inputs = support_inputs.to(device=device)
                        query_inputs = query_inputs.to(device=device)

                    batch_probs = model.infer(
                        support_inputs, query_inputs).detach().cpu().numpy()
                    probs.extend(batch_probs[0])
                    labels.extend(query_labels[0].numpy())

            auc = roc_auc_score(labels, probs)
            aucs.append(auc)

        mean = np.mean(aucs)
        std = np.std(aucs)

        auc_means.append(mean)
        auc_stds.append(std)

    return auc_means, auc_stds

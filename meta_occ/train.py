import numpy as np
import torch

from meta_occ.models import MetaOCCModel, EmbeddingNet
from meta_occ.layers import PrototypeLayer, SVDDLayer
from meta_occ import utils


def train(args):
    train_loader = utils.get_dataset_loader(args.dataset,
                                            args.dataset_folder,
                                            args.shot,
                                            args.query_size,
                                            args.batch_size,
                                            True,
                                            train=True)
    val_loader = utils.get_dataset_loader(args.dataset,
                                          args.dataset_folder,
                                          args.shot,
                                          args.query_size,
                                          args.batch_size,
                                          True,
                                          val=True)

    channels = 1 if args.dataset == 'omniglot' else 3
    if args.method == 'meta_svdd':
        layer = SVDDLayer(args.shot)
    elif args.method == 'protonet':
        layer = PrototypeLayer()
    else:
        raise KeyError(f'Unsupported method "{args.method}.'
                       'Options are "meta_svdd" and "protonet".')

    model = MetaOCCModel(EmbeddingNet(channels, 64), layer)
    model.to(device=args.device)
    loss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 1
    best_lb = 0.
    best_mean = 0.
    faults = 0
    while faults < args.patience:
        for batch in train_loader:
            (support_inputs, query_inputs,
             query_labels) = utils.to_one_class_batch(batch, args.shot)

            support_inputs = support_inputs.to(device=args.device)
            query_inputs = query_inputs.to(device=args.device)
            query_labels = query_labels.to(device=args.device)

            loss_val = loss(model(support_inputs, query_inputs),
                            query_labels.float())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f'Step {step}, loss = {loss_val.item()}')

            if step % 100 == 0:
                model.train(False)
                mean, ci95 = utils.evaluate(model, val_loader,
                                            args.val_episodes, args.shot,
                                            args.device)
                model.train(True)
                print(f'Accuracy = {100*mean:.2f} ± {100*ci95:.2f}%')

                lb = mean - ci95
                if lb > best_lb or (np.isclose(lb, best_lb)
                                    and mean > best_mean):
                    print('New best')
                    best_lb = lb
                    best_mean = mean
                    faults = 0
                    torch.save(model.state_dict(), args.save_path)
                else:
                    faults += 1
                    print(f'{faults} fault{"s" if faults > 1 else ""}')
                    if faults >= args.patience:
                        ci95 = best_mean - best_lb
                        print('Training finished.'
                              f' Best = {100*best_mean:.2f} ± {100*ci95:.2f}%')
                        break

            step += 1


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder',
                        type=str,
                        help='Path to the dataset folder.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='omniglot',
        choices=('omniglot', 'miniimagenet', 'tieredimagenet', 'cifar_fs'),
        help='Dataset in which to train the model (Default: "omniglot").')
    parser.add_argument(
        '--method',
        type=str,
        default='meta_svdd',
        choices=('protonet', 'meta_svdd'),
        help='Meta one-class classification method (Default: "meta_svdd").')
    parser.add_argument(
        '--shot',
        type=int,
        default=5,
        help='Number of examples in the support set (Default: 5).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Number of episodes in a batch (Default: 16).')
    parser.add_argument(
        '--query_size',
        type=int,
        default=10,
        help='Number of examples in each query set (Default: 10).')
    parser.add_argument('--val_episodes',
                        type=int,
                        default=500,
                        help='Number of validation episodes (Default: 500).')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=5e-4,
                        help='Learning rate (Default: 5e-4)')
    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help='Early stopping patience (Default: 10).')
    parser.add_argument(
        '--save_path',
        type=str,
        default='model.pth',
        help='Path in which to save model (Default: "model.pth").')

    args = parser.parse_args()
    args.device = torch.device('cpu')

    return args


if __name__ == '__main__':
    train(parse_args())

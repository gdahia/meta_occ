import torch

from meta_occc.models import EmbeddingNet, ResNet12
from meta_occc.protonet import OneClassPrototypicalNet
from meta_occc import utils


def train(args):
    train_loader = utils.get_dataset(args.dataset,
                                     args.dataset_folder,
                                     args.shot,
                                     args.query_size,
                                     args.batch_size,
                                     True,
                                     train=True)
    val_loader = utils.get_dataset(args.dataset,
                                   args.dataset_folder,
                                   args.shot,
                                   args.query_size,
                                   args.batch_size,
                                   True,
                                   val=True)

    channels = 1 if args.dataset == 'omniglot' else 3
    if args.model == 'embedding':
        model = OneClassPrototypicalNet(EmbeddingNet(channels, 64))
    elif args.model == 'resnet12':
        model = OneClassPrototypicalNet(ResNet12(channels))
    else:
        raise KeyError(f'Unsupported model "{args.model}.'
                       'Options are "embedding" and "resnet12".')
    model.to(device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 1
    while True:
        for batch in train_loader:
            (support_inputs, query_inputs,
             query_labels) = utils.to_one_class_batch(batch, args.shot)

            support_inputs = support_inputs.to(device=args.device)
            query_inputs = query_inputs.to(device=args.device)
            query_labels = query_labels.to(device=args.device)

            loss = model.loss(support_inputs, query_inputs,
                              query_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f'Step {step}, loss = {loss.item()}')

            if step % 100 == 0:
                mean, ci95 = utils.evaluate(model, val_loader,
                                            args.val_episodes, args.shot,
                                            args.device)
                print(f'Accuracy = {mean:.2f} ± {ci95:.2f}%')

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
        choices=('omniglot', 'miniimagenet', 'tieredimagenet'),
        help='Dataset in which to train the model (Default: "omniglot").')
    parser.add_argument('--model',
                        type=str,
                        default='embedding',
                        choices=('embedding', 'resnet12'),
                        help='Model architecture (Default: "embedding").')
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
                        default=100,
                        help='Number of validation episodes (Default: 100).')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=5e-4,
                        help='Learning rate (Default: 5e-4)')
    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device(
        'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    return args


if __name__ == '__main__':
    train(parse_args())

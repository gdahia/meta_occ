import torch

from meta_occc.models import EmbeddingNet, ResNet12
from meta_occc.methods import OneClassPrototypicalNet, MetaSVDD
from meta_occc import utils


def evaluate(args):
    test_loader = utils.get_dataset(args.dataset,
                                    args.dataset_folder,
                                    args.shot,
                                    args.query_size,
                                    args.batch_size,
                                    True,
                                    train=args.split == 'train',
                                    val=args.split == 'val',
                                    test=args.split == 'test')

    channels = 1 if args.dataset == 'omniglot' else 3
    if args.model == 'embedding':
        model = EmbeddingNet(channels, 64)
    elif args.model == 'resnet12':
        model = ResNet12(channels)
    else:
        raise KeyError(f'Unsupported model "{args.model}.'
                       'Options are "embedding" and "resnet12".')
    if args.method == 'meta_svdd':
        model = MetaSVDD(model)
    elif args.method == 'protonet':
        model = OneClassPrototypicalNet(model)
    else:
        raise KeyError(f'Unsupported method "{args.method}.'
                       'Options are "meta_svdd" and "protonet".')
    model.load_state_dict(torch.load(args.model_path))
    model.to(device=args.device)

    mean, ci95 = utils.evaluate(model, test_loader, args.episodes, args.shot,
                                args.device)
    print(f'Test accuracy = {mean:.2f} ± {ci95:.2f}%')


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
    parser.add_argument('--model',
                        type=str,
                        default='embedding',
                        choices=('embedding', 'resnet12'),
                        help='Model architecture (Default: "embedding").')
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
    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'val', 'test'),
                        help='Dataset split to evaluate (Default: "test").')
    parser.add_argument('--episodes',
                        type=int,
                        default=10_000,
                        help='Number of evaluation episodes (Default: 10000).')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to saved trained model.')
    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device(
        'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    return args


if __name__ == '__main__':
    evaluate(parse_args())

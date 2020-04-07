from statistics import median_low
import torch

from meta_occ.models import MetaOCCModel, EmbeddingNet
from meta_occ.layers import PrototypeLayer, SVDDLayer
from meta_occ import utils


def evaluate(args):
    channels = 1 if args.dataset == 'omniglot' else 3
    model = EmbeddingNet(channels, 64)

    if args.method == 'meta_svdd':
        layer = SVDDLayer(args.shot)
    elif args.method == 'protonet':
        layer = PrototypeLayer()
    else:
        raise KeyError(f'Unsupported method "{args.method}.'
                       'Options are "meta_svdd" and "protonet".')
    model = MetaOCCModel(EmbeddingNet(channels, 64), layer)
    model.load_state_dict(torch.load(args.model_path,
                                     map_location=args.device))
    model.to(device=args.device)

    if args.metric == 'acc':
        test_loader = utils.get_dataset_loader(args.dataset,
                                               args.dataset_folder,
                                               args.shot,
                                               args.query_size,
                                               args.batch_size,
                                               True,
                                               train=args.split == 'train',
                                               val=args.split == 'val',
                                               test=args.split == 'test')

        model.eval()
        mean, ci95 = utils.evaluate(model, test_loader, args.episodes,
                                    args.shot, args.device)
        print(f'{args.split} accuracy = {100*mean:.2f} ± {100*ci95:.2f}%')
    else:
        dataset = utils.get_dataset(args.dataset,
                                    args.dataset_folder,
                                    args.shot,
                                    args.query_size,
                                    True,
                                    train=args.split == 'train',
                                    val=args.split == 'val',
                                    test=args.split == 'test')

        model.eval()
        means, stds = utils.auc(model,
                                dataset,
                                args.episodes,
                                args.shot,
                                device=args.device)
        for i, (mean, std) in enumerate(zip(means, stds), 1):
            print(f'{args.split} AUC for class {i} = '
                  f'{100*mean:.2f} ± {100*std:.2f}%')

        min_mean, min_mean_std = min(zip(means, stds))
        median_mean, median_mean_std = median_low(zip(means, stds))
        max_mean, max_mean_std = max(zip(means, stds))
        print()
        print(f'Minimum {args.split} class mean AUC '
              f'= {100*min_mean:.2f} ± {100*min_mean_std:.2f}%')
        print(f'Median {args.split} class mean AUC '
              f'= {100*median_mean:.2f} ± {100*median_mean_std:.2f}%')
        print(f'Maximum {args.split} class mean AUC '
              f'= {100*max_mean:.2f} ± {100*max_mean_std:.2f}%')


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
    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'val', 'test'),
                        help='Dataset split to evaluate (Default: "test").')
    parser.add_argument('--episodes',
                        type=int,
                        help='Number of evaluation episodes (Default:'
                        ' 10,000 for accuracy, 10 for AUC).')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to saved trained model.')
    parser.add_argument('--metric',
                        type=str,
                        default='acc',
                        choices=('acc', 'auc'),
                        help='Evaluation metric (Default: "acc").')

    args = parser.parse_args()
    args.device = torch.device('cpu')
    if not args.episodes:
        args.episodes = 10 if args.metric == 'auc' else 10_000

    return args


if __name__ == '__main__':
    evaluate(parse_args())

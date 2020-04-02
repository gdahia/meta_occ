import numpy as np

from meta_occ.pca_ocsvm import PCAOneClassSVM
from meta_occ import utils


def grid_search(args):
    loader = utils.get_dataset_loader(args.dataset,
                                      args.dataset_folder,
                                      args.shot,
                                      args.query_size,
                                      args.batch_size,
                                      True,
                                      train=args.split == 'train',
                                      val=args.split == 'val',
                                      test=args.split == 'test')

    nu_search_space = [0.1, 0.01]
    gamma_search_space = np.logspace(-10, -1, num=10, base=2)
    best_lb = 0.
    best_mean = 0.
    best_nu = None
    best_gamma = None
    for nu in nu_search_space:
        for gamma in gamma_search_space:
            if args.method == 'pca_ocsvm':
                model = PCAOneClassSVM(nu, gamma)
            else:
                raise KeyError()

            mean, ci95 = utils.evaluate(model, loader, args.episodes,
                                        args.shot, args.device)
            lb = mean - ci95
            if lb > best_lb or (np.isclose(lb, best_lb) and mean > best_mean):
                best_lb = lb
                best_mean = mean
                best_nu = nu
                best_gamma = gamma

    ci95 = best_mean - best_lb
    print(
        f'best {args.split} accuracy = {100*best_mean:.2f} ± {100*ci95:.2f}%')

    print(f'best_nu={best_nu}')
    print(f'best_gamma={best_gamma}')


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
        default='pca_ocsvm',
        choices=('pca_ocsvm', ),
        help='Meta one-class classification method (Default: "pca_ocsvm").')
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
    parser.add_argument(
        '--episodes',
        type=int,
        default=10_000,
        help='Number of validation episodes (Default: 10,000).')
    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'val', 'test'),
                        help='Dataset split to evaluate (Default: "test").')
    args = parser.parse_args()
    args.device = 'cpu'

    return args


if __name__ == '__main__':
    grid_search(parse_args())

# Meta Learning for Few-Shot One-class Classification

Official PyTorch implementation of "Meta Learning for Few-Shot One-class
Classification" by _Gabriel Dahia, Maurício Pamplona Segundo_, 2020.

## Installation

The code is tested in Ubuntu 16.04 with Python 3.6.9. To install the
dependencies, we recommend setting up a virtual environment with `venv`:

```bash
python3 -m venv env
```

After that, enter the virtual environment with

```bash
source env/bin/activate
```

and install the dependencies in the `requirements.txt` file with:

```bash
python3 -m pip install -r requirements.txt
```

## Experiments

### Training a model

To reproduce the experiments in the paper with "Meta SVDD" and "One-class
Prototypical Network", first we must train a model for each. To do that,
inside the virtual enviroment, run

```bash
python3 -m meta_occ.train data/ --dataset $dataset --method $method
```

and replace `$dataset` with either `omniglot` or `cifar_fs`, and `$method` with
either `meta_svdd` or `protonet`. The trained model will be saved in a file
called `model.pth`.

The full list of possible arguments for `meta_occ.train` is

```bash
usage: train.py [-h]
                [--dataset {omniglot,miniimagenet,tieredimagenet,cifar_fs}]
                [--method {protonet,meta_svdd}] [--shot SHOT]
                [--batch_size BATCH_SIZE] [--query_size QUERY_SIZE]
                [--val_episodes VAL_EPISODES] [--learning_rate LEARNING_RATE]
                [--patience PATIENCE] [--save_path SAVE_PATH] dataset_folder

positional arguments:
  dataset_folder        Path to the dataset folder.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {omniglot,miniimagenet,tieredimagenet,cifar_fs}
                        Dataset in which to train the model (Default:
                        "omniglot").
  --method {protonet,meta_svdd}
                        Meta one-class classification method (Default:
                        "meta_svdd").
  --shot SHOT           Number of examples in the support set (Default: 5).
  --batch_size BATCH_SIZE
                        Number of episodes in a batch (Default: 16).
  --query_size QUERY_SIZE
                        Number of examples in each query set (Default: 10).
  --val_episodes VAL_EPISODES
                        Number of validation episodes (Default: 500).
  --learning_rate LEARNING_RATE
                        Learning rate (Default: 5e-4)
  --patience PATIENCE   Early stopping patience (Default: 10).
  --save_path SAVE_PATH
                        Path in which to save model (Default: "model.pth").
```

### AUC experiment

To reproduce the results in the last column of table 1 in the paper, run:

```bash
python3 -m meta_occ.evaluate data/ --dataset $dataset --method meta_svdd --model_path $model --metric auc
```

and replace `$dataset` with either `omniglot` or `cifar_fs`, and `$model_path`
with the path to a `meta_svdd` model file trained in the appropriate dataset.

#### Deep SVDD few-shot experiment

To reproduce the results for the Deep SVDD few-shot experiment (i.e. the second
to last column in table 1 in the paper), read the README.md file in the
`few-shot_deep-svdd` folder and follow its instructions.

### Accuracy experiment

To reproduce the results in the last two columns of table 2 in the paper, run:

```bash
python3 -m meta_occ.evaluate data/ --dataset $dataset --method $method --model_path $model --metric acc
```

and replace `$dataset` with either `omniglot` or `cifar_fs`, `$method` with
either `meta_svdd` or `protonet`, and `$model_path` with the path to a model
file trained in the appropriate dataset for the given method.

The full list of possible arguments for `meta_occ.evaluate` is

```bash
usage: evaluate.py [-h]
                   [--dataset {omniglot,miniimagenet,tieredimagenet,cifar_fs}]
                   [--method {protonet,meta_svdd}] [--shot SHOT]
                   [--batch_size BATCH_SIZE] [--query_size QUERY_SIZE]
                   [--split {train,val,test}] [--episodes EPISODES]
                   --model_path MODEL_PATH [--metric {acc,auc}] [--use_cuda]
                   dataset_folder

positional arguments:
  dataset_folder        Path to the dataset folder.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {omniglot,miniimagenet,tieredimagenet,cifar_fs}
                        Dataset in which to train the model (Default:
                        "omniglot").
  --method {protonet,meta_svdd}
                        Meta one-class classification method (Default:
                        "meta_svdd").
  --shot SHOT           Number of examples in the support set (Default: 5).
  --batch_size BATCH_SIZE
                        Number of episodes in a batch (Default: 16).
  --query_size QUERY_SIZE
                        Number of examples in each query set (Default: 10).
  --split {train,val,test}
                        Dataset split to evaluate (Default: "test").
  --episodes EPISODES   Number of evaluation episodes (Default: 10,000 for
                        accuracy, 10 for AUC).
  --model_path MODEL_PATH
                        Path to saved trained model.
  --metric {acc,auc}    Evaluation metric (Default: "acc").
```

#### Shallow baseline

To reproduce the results in the first column of table 2 in the paper (_i.e._
the results for the One-class SVM with PCA), run

```bash
python3 -m meta_occ.baseline_grid_search data/ --dataset $dataset
```

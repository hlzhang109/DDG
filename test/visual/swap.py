# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import copy
import uuid
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import yaml
from azureml.core import Run
run = Run.get_context()
import datasets
import hparams_registry
import algorithms_gen as algorithms
import numpy.random as random
from lib import misc
from scripts.save_images import write_2images
from lib.fast_data_loader import InfiniteDataLoader

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str, default='/home/v-yifanzhang/datasets')
parser.add_argument('--dataset', type=str, default="RotatedMNIST")
parser.add_argument('--gen_dir', type=str, default="models/mnist.pkl", help="if not empty, the generator of DEDF will be loaded")
parser.add_argument('--algorithm', type=str, default="DDG")
parser.add_argument('--hparams', type=str,
    help='JSON-serialized hparams dict')
parser.add_argument('--hparams_seed', type=int, default=0,
    help='Seed for random hparams (0 means "default hparams")')
parser.add_argument('--trial_seed', type=int, default=0,
    help='Trial number (used for seeding split_dataset and '
    'random_hparams).')
parser.add_argument('--seed', type=int, default=0,
    help='Seed for everything else')
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
args = parser.parse_args()

# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
start_step = 0
algorithm_dict = None

if args.hparams_seed == 0:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
else:
    hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
        misc.seed_hash(args.hparams_seed, args.trial_seed))
if args.hparams:
    hparams.update(json.loads(args.hparams))
hparams['batch_size'] = 2
print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args.dataset in vars(datasets):
    dataset = vars(datasets)[args.dataset](args.data_dir,
        args.test_envs, hparams)
else:
    raise NotImplementedError

in_splits = []
out_splits = []
uda_splits = []
for env_i, env in enumerate(dataset):

    out, in_ = misc.split_dataset(env,
        int(len(env)*args.holdout_fraction),
        misc.seed_hash(args.trial_seed, env_i))
    in_splits.append((in_, None))
    out_splits.append((out, None))
train_loaders = [InfiniteDataLoader(
    dataset=env,
    weights=env_weights,
    batch_size=hparams['batch_size'],
    num_workers=dataset.N_WORKERS)
    for i, (env, env_weights) in enumerate(in_splits)
    if i not in args.test_envs]

algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
    len(dataset) - len(args.test_envs), hparams)

if algorithm_dict is not None:
    algorithm.load_state_dict(algorithm_dict)

algorithm.to(device)
pretext_model = torch.load(args.gen_dir)['model_dict']
alg_dict = algorithm.state_dict()
ignored_keys = []
state_dict = {k: v for k, v in pretext_model.items() if k in alg_dict.keys() and ('id_featurizer' in k or 'gen' in k)}
alg_dict.update(state_dict)
algorithm.load_state_dict(alg_dict)

train_minibatches_iterator = zip(*train_loaders)

for step in range(start_step, 10):
    minibatches_device =  [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
    minibatches_device_neg = [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
    images_a = torch.cat([x for x, y, pos in minibatches_device])
    images_b = torch.cat([x for x, y, pos in minibatches_device_neg])
    perm = torch.randperm(len(images_b)).tolist()
    image_outputs = algorithm.sample(images_a, images_b[perm])
    write_2images(image_outputs, hparams['batch_size']*len(train_loaders), "test/visual/results", 'train_%08d' % (step + 1), run)
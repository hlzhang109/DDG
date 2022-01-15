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
from torchvision import transforms
run = Run.get_context()
import datasets
import hparams_registry
import algorithms_gen as algorithms
import numpy.random as random
from lib import misc
from scripts.save_images import write_2images
from lib.fast_data_loader import InfiniteDataLoader
from lib.misc import Augmix

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str, default='/home/v-yifanzhang/datasets')
parser.add_argument('--dataset', type=str, default="PACS")
parser.add_argument('--gen_dir', type=str, default="models/mnist.pkl", help="if not empty, the generator of DEDF will be loaded")
parser.add_argument('--algorithm', type=str, default="DDG_AugMix")
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

train_minibatches_iterator = zip(*train_loaders)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean, std)])
TO_pil = transforms.ToPILImage()

def sample(x_a, x_b, pretrain_model=None):
    device = "cuda" if x_a.is_cuda else "cpu" 
    x_as, x_bs, x_a_aug, x_b_aug, x_a_aug1, x_b_aug1 = [], [], [], [], [], []
    for image_a, image_b in zip(x_a, x_b):
        x_b_, x_ab1, x_ab2= Augmix(TO_pil(image_b.cpu()), preprocess, no_jsd=False)
        x_a_, x_ba1, x_ba2= Augmix(TO_pil(image_a.cpu()), preprocess, no_jsd=False)
        x_a_aug.append(x_ba1.to(device).unsqueeze(0)); x_a_aug1.append(x_ba2.to(device).unsqueeze(0))
        x_b_aug.append(x_ab1.to(device).unsqueeze(0)); x_b_aug1.append(x_ab2.to(device).unsqueeze(0))
        x_as.append(x_a_.to(device).unsqueeze(0)); x_bs.append(x_b_.to(device).unsqueeze(0))
    x_a_aug, x_a_aug1=torch.cat(x_a_aug), torch.cat(x_a_aug1)
    x_b_aug, x_b_aug1 = torch.cat(x_b_aug), torch.cat(x_b_aug1)
    x_as, x_bs = torch.cat(x_as), torch.cat(x_bs)
    return x_as, x_a_aug, x_a_aug1, x_bs, x_b_aug, x_b_aug1

for step in range(start_step, 5):
    minibatches_device =  [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
    minibatches_device_neg = [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
    images_a = torch.cat([x for x, y, pos in minibatches_device])
    images_b = torch.cat([x for x, y, pos in minibatches_device_neg])
    perm = torch.randperm(len(images_b)).tolist()
    image_outputs = sample(images_a, images_b[perm])
    write_2images(image_outputs, hparams['batch_size']*len(train_loaders), "test/visual/results/aug_mix", 'Augmix_%08d' % (step + 1), run)
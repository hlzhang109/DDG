# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
import lib.augmentations as augmentations

import numpy as np
import torch
import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def sample_tuple_of_minibatches(minibatches, device):
    disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])
    perm = torch.randperm(len(minibatches)).tolist()
    tuples = []
    labels = np.array([minibatches[i][1] for i in range(len(minibatches))])
    
    for i in range(len(minibatches)):

        x, y, d = minibatches[i][0], minibatches[i][1], disc_labels[i]
        x_n, y_n, d_n = minibatches[perm[i]][0], minibatches[perm[i]][1], disc_labels[perm[i]]
        while y_n == y:
            i = perm[i]
            x_n, y_n = minibatches[perm[i]][0], minibatches[perm[i]][1], disc_labels[perm[i]]
        
        pos_ind = np.argwhere(labels == y); pos_n_ind = np.where(labels == y_n)
        x_p, x_np = minibatches[pos_ind[0]][0], minibatches[pos_n_ind[0]][0]

        tuples.append((x, y, d, x_p), (x_n, y_n, d_n, x_np))

    return tuples

def plot_confusion(matrix):
    pass

def accuracy(network, loader, weights, device, args=None, step=None, is_ddg=False):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        if is_ddg:
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset : weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)
                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
        else:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)
                #if step % 50 == 0 and args.dataset != 'WILDSCamelyon':
                #    pass
                #    confusion = confusion_matrix(p.gt(0).cpu().data, y.cpu().data)
                #    with open(gen_dir + '/confusion_{}_{}_d{}/confusion{}.npy'.format(args.algorithm, args.dataset, step), 'wb') as f:
                #       np.save(f, confusion)

                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset : weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)
                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
                
    network.train()

    return correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

augmentations.IMAGE_SIZE = 224
def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    mixture_width = 3
    mixture_depth = -1
    aug_severity = 1
    ws = np.float32(
        np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

def Augmix(x, preprocess, no_jsd):
    if no_jsd:
      return aug(x, preprocess)
    else:
      return preprocess(x), aug(x, preprocess), aug(x, preprocess)
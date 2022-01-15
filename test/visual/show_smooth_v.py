# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import random
import numpy as np
from torch.autograd import Variable
import torch
import torch.utils.data
import yaml
from azureml.core import Run
run = Run.get_context()
import datasets
import hparams_registry
import algorithms_gen as algorithms
import numpy.random as random
from lib import misc
import imageio
from PIL import Image
from scripts.save_images import write_2images
from lib.fast_data_loader import InfiniteDataLoader

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str, default='/home/v-yifanzhang/datasets')
parser.add_argument('--dataset', type=str, default="RotatedMNIST")
parser.add_argument('--gen_dir', type=str, default="models/mnist_gen.pkl", help="if not empty, the generator of DEDF will be loaded")
parser.add_argument('--algorithm', type=str, default="DDG")
parser.add_argument('--hparams', type=str,
    help='JSON-serialized hparams dict')
parser.add_argument('--hparams_seed', type=int, default=0,
    help='Seed for random hparams (0 means "default hparams")')
parser.add_argument('--trial_seed', type=int, default=0,
    help='Trial number (used for seeding split_dataset and '
    'random_hparams).')
parser.add_argument('--seed', type=int, default=15,
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
hparams['batch_size'] = 4
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

minibatches_device =  [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
minibatches_device_neg = [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
def recover(inp):
    """Imshow for Tensor."""
    if len(inp.shape) > 2:
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
    else:
        inp = inp.numpy()
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp
def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward
im = {}
bg_img, _, _ = minibatches_device_neg[0]
gray = to_gray(False)
bg_ori = bg_img
bg_img = gray(bg_img)
bg_img = Variable(bg_img.cuda())
ff = []
gif = []
encode = algorithm.gen.encode # encode function
id_encode = algorithm.id_featurizer # encode function
decode = algorithm.gen.decode # decode function
with torch.no_grad():
    for data in minibatches_device:
        id_img, _, _ = data
        minibatches_device =  [(x.to(device), y, pos) for x,y,pos in next(train_minibatches_iterator)]
        id_img[:2] = minibatches_device[0][0][:2]
        id_img = Variable(id_img.cuda())
        n, c, h, w = id_img.size()
        # Start testing
        s = encode(bg_img)
        f, _ = id_encode(id_img) 
        for count in range(hparams['batch_size']):
            input1 = recover(id_img[count].squeeze().data.cpu())
            im[count] = input1
            gif.append(input1)
            for i in range(11):
                s_tmp = s[count,:,:,:] if len(s.shape)==4 else s[count,:]
                tmp_f = 0.1*i*f[count] + (1-0.1*i)*f[1-count]
                tmp_f = tmp_f.view(1, -1)
                s_tmp = torch.cat([s_tmp.unsqueeze(0), s_tmp.unsqueeze(0)])
                tmp_f = torch.cat([tmp_f,tmp_f])
                outputs = decode(s_tmp, tmp_f)[0]
                tmp = recover(outputs[0].data.cpu())
                im[count] = np.concatenate((im[count], tmp), axis=1)
                gif.append(tmp)
        break

# save long image
pic = np.concatenate( (im[0], im[1],im[2], im[3]) , axis=0)
pic = Image.fromarray(pic.astype('uint8'))
pic.save('smooth_.jpg')

# save gif
imageio.mimsave('./smooth.gif', gif)
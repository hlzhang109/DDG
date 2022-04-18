# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from scripts.save_images import write_2images
import copy
import numpy as np
import random

import networks
from lib.misc import random_pairs_of_minibatches, Augmix

ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'DDG'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.device = device

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, device)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, device)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance, device='cpu'):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP_(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian, device='cpu'):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    variation Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # variation network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized variation
        return self.network_c(self.randomize(self.network_f(x), "variation"))

    def forward_s(self, x):
        # learning variation network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="variation", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "variation":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn variation
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class DDG(ERM):
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(DDG, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, device)
        self.hparams = hparams
        self.class_balance=hparams['class_balanced']
        self.iteration = 0
        self.id_featurizer = self.featurizer
        self.dis_id = self.classifier
        self.gen = networks.AdaINGen(1, self.id_featurizer.n_outputs, hparams) if not hparams['is_mnist'] else networks.VAEGen()
        self.dis_img = networks.MsImageDis(hparams=hparams) 
        self.recon_xp_w = hparams['recon_xp_w']
        self.recon_xn_w = hparams['recon_xn_w']
        self.margin = hparams['margin']
        self.eta = hparams['eta']
        
        def to_gray(half=False): #simple
            def forward(x):
                x = torch.mean(x, dim=1, keepdim=True)
                if half:
                    x = x.half()
                return x
            return forward
        self.single = to_gray(False)
        self.optimizer_gen = torch.optim.Adam([p for p in list(self.gen.parameters())  if p.requires_grad], lr=self.hparams['lr_g'], betas=(0, 0.999), weight_decay=self.hparams['weight_decay_g'])
        if self.hparams['stage'] == 0:
            # Setup the optimizers
            self.optimizer_dis_img = torch.optim.Adam(
                self.dis_img.parameters(),
                lr=self.hparams["lr_d"],
                weight_decay=self.hparams['weight_decay'])
            step = hparams['steps']*0.6
            print(step)
            self.dis_scheduler = lr_scheduler.MultiStepLR(self.optimizer_dis_img, milestones=[step, step+step//2, step+step//2+step//4],
                                        gamma=0.1)
            self.gen_scheduler = lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=[step, step+step//2, step+step//2+step//4],
                                        gamma=0.1)

        self.id_criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.CrossEntropyLoss()
        

    def recon_criterion(self, input, target, reduction=True):
            diff = input - target.detach()
            B,C,H,W = input.shape
            if reduction == False:
                return torch.mean(torch.abs(diff[:]).view(B,-1),dim=-1)
            return torch.mean(torch.abs(diff[:]))
    
    def train_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
            print('there has bn')

    def forward(self, x_a, x_b, xp_a, xp_b):
        '''
            inpus:
                x_a, x_b: image from dataloader a,b
                xp_a, xp_b: positive pair of x_a, x_b
        '''
        s_a = self.gen.encode(self.single(x_a))# v for x_a
        s_b = self.gen.encode(self.single(x_b))# v for x_b
        s_pa = self.gen.encode(self.single(xp_a))# v for x_a
        s_pb = self.gen.encode(self.single(xp_b))# v for x_b
        f_a, x_fa = self.id_featurizer(x_a, self.hparams['stage']) # f_a: detached s for x_a, x_fa: s for x_a
        p_a = self.dis_id(x_fa)             # semantics classification result for x_a
        f_b, x_fb = self.id_featurizer(x_b, self.hparams['stage'])
        p_b = self.dis_id(x_fb)
        fp_a, xp_fa = self.id_featurizer(xp_a, self.hparams['stage'])
        pp_a = self.dis_id(xp_fa)
        fp_b, xp_fb = self.id_featurizer(xp_b, self.hparams['stage'])
        pp_b = self.dis_id(xp_fb)
        if self.hparams['stage'] == 0:
            # cross-variation generation
            x_ba = self.gen.decode(s_b, f_a) # x_ba: generated from semantics of a and variation of b
            x_ab = self.gen.decode(s_a, f_b)
            x_a_recon = self.gen.decode(s_a, f_a) # generate from semantics and variation of a
            x_b_recon = self.gen.decode(s_b, f_b) 
        else:
            x_ba = None
            x_ab = None
            x_a_recon = None
            x_b_recon = None

        x_a_recon_p = self.gen.decode(s_a, fp_a)
        x_b_recon_p = self.gen.decode(s_b, fp_b)

        return x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p    

    def dis_update(self, x_ab, x_ba, x_a, x_b, hparams):
        '''
            inpus:
                x_ab: generated from semantics of b and variation of a (fake)
                x_ba: generated from semantics of a and variation of b (fake)
                x_a, x_b: real image
        '''
        self.optimizer_dis_img.zero_grad()
        self.loss_dis_a, reg_a = self.dis_img.calc_dis_loss(self.dis_img, x_ba.detach(), x_a)
        self.loss_dis_b, reg_b = self.dis_img.calc_dis_loss(self.dis_img, x_ab.detach(), x_b)
        self.loss_dis_total = hparams['gan_w'] * self.loss_dis_a + hparams['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward() # discriminators are trained here
        self.optimizer_dis_img.step()     

    def gen_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b,  l_a, l_b, hparams):
        '''
            inputs:
                x_ab: generated from semantics of b and variation of a
                x_ba: generated from semantics of a and variation of b
                s_a, s_b: variation factors for x_a, x_b
                f_a, f_b: detached semantic factors for x_a, x_b
                p_a, p_b: semantics prediction results for x_a, x_b
                pp_a, pp_b: semantics prediction results for the positive pair of x_a, x_b
                x_a_recon, x_b_recon: reconstruction of x_a, x_b
                x_a_recon_p, x_b_recon_p: reconstruction of the positive pair of x_a, x_b
                x_a, x_b,  l_a, l_b: images and semantics labels
                hparams: parameters
        '''
        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()

        #################################

        # auto-encoder image reconstruction
        self.recon_a2a, self.recon_b2b = self.recon_criterion(x_a_recon_p, x_a, reduction=False), self.recon_criterion(x_b_recon_p, x_b, reduction=False)
        self.loss_gen_recon_p =  torch.mean(torch.max(self.recon_a2a-self.margin, torch.zeros_like(self.recon_a2a)))+ torch.mean(torch.max(self.recon_b2b-self.margin, torch.zeros_like(self.recon_b2b)))

        # Emprical Loss
        if not hparams['is_mnist']:
            _, x_fa_recon = self.id_featurizer(x_ab); p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ba); p_b_recon = self.dis_id(x_fb_recon)
        else:
            _, x_fa_recon = self.id_featurizer(x_ba); p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ab); p_b_recon = self.dis_id(x_fb_recon)            
        self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b) +  self.id_criterion(pp_a, l_a) + self.id_criterion(pp_b, l_b)
        self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b)

        self.step(torch.mean(self.recon_a2a))
        # total loss
        self.loss_gen_total = self.loss_id +\
                self.recon_xp_w * self.loss_gen_recon_p +\
                hparams['recon_id_w'] * self.loss_gen_recon_id 

        self.loss_gen_total.backward()
        self.optimizer_gen.step()
        self.optimizer.step()

    def gan_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b,  l_a, l_b, hparams):
        '''
            Train the GAN
            inputs:
                x_ab: generated from semantics of b and variation of a
                x_ba: generated from semantics of a and variation of b
                s_a, s_b: variation factors for x_a, x_b
                f_a, f_b: detached semantic factors for x_a, x_b
                p_a, p_b: semantics prediction results for x_a, x_b
                pp_a, pp_b: semantics prediction results for the positive pair of x_a, x_b
                x_a_recon, x_b_recon: reconstruction of x_a, x_b
                x_a_recon_p, x_b_recon_p: reconstruction of the positive pair of x_a, x_b
                x_a, x_b,  l_a, l_b: images and semantics labels
                hparams: parameters
        '''
        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()
 
        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)
        f_a, f_b = f_a.detach(), f_b.detach()

        rand_num = random.uniform(0,1)
        #################################
        # encode structure
        if 0.5>=rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen.enc_content(self.single(x_ba_copy))
        else:
            # copy the encoder
            self.enc_content_copy = copy.deepcopy(self.gen.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy(self.single(x_ab))
            s_b_recon = self.enc_content_copy(self.single(x_ba))

        #################################
        # encode appearance
        self.id_copy = copy.deepcopy(self.id_featurizer)
        self.dis_id_copy = copy.deepcopy(self.dis_id)
        self.id_copy.eval()
        self.dis_id_copy.eval()

        # encode again (encoder is fixed, input is tuned)
        f_a_recon, _ = self.id_copy(x_ba);
        f_b_recon, _ = self.id_copy(x_ab); 

        # auto-encoder image reconstruction
        self.loss_gen_recon_x =  self.recon_criterion(x_a_recon, x_a)+self.recon_criterion(x_b_recon, x_b)

        # Emprical Loss

        x_aba, x_bab = self.gen.decode(s_a_recon, f_a_recon), self.gen.decode(s_b_recon, f_b_recon)  if hparams['recon_x_cyc_w'] > 0 else None
        self.loss_gen_cycrecon_x = self.recon_criterion(x_aba, x_a) + self.recon_criterion(x_bab, x_b) if hparams['recon_x_cyc_w'] > 0 else torch.tensor(0)

        # GAN loss
        self.loss_gen_adv = self.dis_img.calc_gen_loss(self.dis_img, x_ba) + self.dis_img.calc_gen_loss(self.dis_img, x_ab)

        self.step()
        if self.iteration > hparams['steps'] * hparams['warm_iter_r']:
            hparams['recon_x_cyc_w'] += hparams['warm_scale']
            hparams['recon_x_cyc_w'] = min(hparams['recon_x_cyc_w'], hparams['max_cyc_w'])

        # total loss
        self.loss_gen_total = hparams['gan_w'] * self.loss_gen_adv + \
                              hparams['recon_x_w'] * self.loss_gen_recon_x + \
                              hparams['recon_x_cyc_w'] * self.loss_gen_cycrecon_x

        self.loss_gen_total.backward()
        self.optimizer_gen.step()
        self.optimizer.step()


    def update(self, minibatches, minibatches_neg, pretrain_model=None, unlabeled=None, iteration=0):
        images_a = torch.cat([x for x, y, pos in minibatches])
        labels_a = torch.cat([y for x, y, pos in minibatches])
        pos_a = torch.cat([pos for x, y, pos in minibatches])
        images_b = torch.cat([x for x, y, pos in minibatches_neg])
        labels_b = torch.cat([y for x, y, pos in minibatches_neg])
        pos_b = torch.cat([pos for x, y, pos in minibatches_neg])
        

        if self.hparams['stage'] == 1 and pretrain_model is not None:
            # swap semantic factors
            s_a = pretrain_model.gen.encode(self.single(images_a))# v for x_a
            s_b = pretrain_model.gen.encode(self.single(images_b))# v for x_b
            f_a, x_fa = pretrain_model.id_featurizer(images_a) # f_a: detached s for x_a, x_fa: s for x_a
            f_b, x_fb = pretrain_model.id_featurizer(images_b)
            # cross-variation generation
            x_ba = pretrain_model.gen.decode(s_b, f_a) # x_ba: generated from semantics of a and variation of b
            x_ab = pretrain_model.gen.decode(s_a, f_b)
            _, _, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = self.forward(images_a, images_b, pos_a, pos_b)
        else:
            x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = self.forward(images_a, images_b, pos_a, pos_b)

        if self.iteration % 500 == 0:
            write_2images(images_a, 4, postfix='image_a')
            write_2images(pos_a, 4, postfix='pos_a')
            write_2images(x_a_recon_p, 4, postfix='x_a_recon_p')
            write_2images(x_ab, 4, postfix='x_ab')
            write_2images(images_b, 4, postfix='x_b')

        if self.hparams['stage'] == 0:
            self.dis_update(x_ab.clone(), x_ba.clone(), images_a, images_b, self.hparams)
            self.gan_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
            self.gen_scheduler.step()
            self.dis_scheduler.step()
            return {'loss_total': self.loss_gen_total.item(), 
                    'loss_gan': self.loss_gen_adv.item(),
                    'loss_recon_x': self.loss_gen_recon_x.item(),
                    'loss_x_cyc': self.loss_gen_cycrecon_x.item()}
        else:
            self.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
            return {
                    'loss_cls': self.loss_id.item(),
                    'loss_gen_recon_id': self.loss_gen_recon_id.item(), 
                    'recon_xp_w': self.recon_xp_w,
                    'loss_recon_p': self.loss_gen_recon_p.item()}
                    
    def sample(self, x_a, x_b, pretrain_model=None):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ab1, x_aba, x_bab = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            model = pretrain_model if pretrain_model is not None else self
            s_a = model.gen.encode( self.single(x_a[i].unsqueeze(0)) )
            s_b = model.gen.encode( self.single(x_b[i].unsqueeze(0)) )
            f_a, _ = model.id_featurizer(x_a[i].unsqueeze(0))
            f_b, _ = model.id_featurizer(x_b[i].unsqueeze(0))
            x_a_recon.append(model.gen.decode(s_a, f_a))
            x_b_recon.append(model.gen.decode(s_b, f_b))
            x_ba = model.gen.decode(s_b, f_a)
            x_ab = model.gen.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ab1.append(x_ab)

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ab1 = torch.cat(x_ba1), torch.cat(x_ab1)
        self.train()

        return x_a, x_ba1, x_b, x_ab1

    def predict(self, x):
        return self.dis_id(self.id_featurizer(x)[-1])

    def step(self, recon_p=None):
        self.iteration += 1
        if recon_p is None:
            return
        self.recon_xp_w = min(max(self.recon_xp_w + self.eta * (recon_p.item() - self.margin), 0), 1)

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch._six import inf
from typing import List, Type, Union, Iterable

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

CLUSTER_METRICS = {
    rand_score: "rand",
    adjusted_rand_score: "arand",
    normalized_mutual_info_score: "nmi",
    adjusted_mutual_info_score: "anmi",
    fowlkes_mallows_score: "fm",
    completeness_score: "comp",
    homogeneity_score: "homo",
    v_measure_score: "vm"
}

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_task_grad_norm(
        grads: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    # if isinstance(parameters, torch.Tensor):
    #     parameters = [parameters]
    # grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if len(source.shape) <= 1:
        source = source.unsqueeze(0)
    if len(target.shape) <= 1:
        target = target.unsqueeze(0)
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss


def contrastive_mmd(features, labels, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    assert len(features) == len(labels)
    if torch.unique(labels).nelement() <= 1: return 0.
    
    unique_labels = torch.unique(labels).squeeze()   
    class_by_index = {}
    for label in unique_labels:
        class_by_index[label.item()] = (labels == label).nonzero().squeeze()
    
    loss, count = 0., 1
    for s_label in unique_labels:
        for t_label in unique_labels:
            if s_label.item() == t_label.item(): continue
            loss += mmd(features[class_by_index[s_label.item()]], features[class_by_index[t_label.item()]])
            count += 1
    
    return -loss / (count - 1)


def calculate_classification_metric(actual_labels, predict_labels):
    result = {}
    result['acc'] = accuracy_score(actual_labels, predict_labels)
    p, r, f, _ = precision_recall_fscore_support(actual_labels, predict_labels, average='micro')
    result['prec_micro'], result['rec_micro'], result['f1_micro'] = p, r, f
    p, r, f, _ = precision_recall_fscore_support(actual_labels, predict_labels, average='macro')
    result['prec_macro'], result['rec_macro'], result['f1_macro'] = p, r, f

    return result


def calculate_cluster_metric(actual_labels, predict_labels):
    num_class = max(actual_labels.max(), predict_labels.max()) + 1
    weight_M = np.zeros((num_class, num_class))
    for i in range(len(actual_labels)):
        weight_M[predict_labels[i], actual_labels[i],] += 1
    ind = linear_sum_assignment(weight_M.max() - weight_M)
    best_map = {ind[0][i]: ind[1][i] for i in range(num_class)}
    best_map_labels = [best_map[x] for x in predict_labels]
    result = calculate_classification_metric(actual_labels, best_map_labels)
    result.update({name: metric(actual_labels, predict_labels) for metric, name in CLUSTER_METRICS.items()})

    return result


class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self, device, total_num_inner_loop_steps, 
                 init_learning_rate=1e-5, clf_lr_multiplier=10., 
                 use_learnable_lr=True, lr_of_lr=1e-3,
                 max_grad_norm=1.):
        super(LSLRGradientDescentLearningRule, self).__init__()
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.clf_lr_multiplier = clf_lr_multiplier
        self.use_learnable_lr = use_learnable_lr
        self.lr_of_lr = lr_of_lr
        self.max_grad_norm = max_grad_norm

    def initialize(self, names_weights_dict):
        clf_list = ['classifier', 'verbalizer']
        self.names_learning_rates_dict = nn.ParameterDict()
        for key, param in names_weights_dict:
            if any([x in key for x in clf_list]):
                self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                    data=torch.ones(self.total_num_inner_loop_steps) * self.init_learning_rate * self.clf_lr_multiplier,
                    requires_grad=self.use_learnable_lr)
            else:
                self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                    data=torch.ones(self.total_num_inner_loop_steps) * self.init_learning_rate,
                    requires_grad=self.use_learnable_lr)
    
    def update_lrs(self, loss, scaler=None):
        if self.use_learnable_lr:
            if scaler is not None:
                scaled_grads = torch.autograd.grad(scaler.scale(loss), self.names_learning_rates_dict.values())
                inv_scale = 1. / scaler.get_scale()
                grads = [p * inv_scale for p in scaled_grads]
                clip_task_grad_norm(grads, self.max_grad_norm)
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, adjust scale and zero out gradients')
                    if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                        scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                    for g in grads: g.zero_()
            else:
                grads = torch.autograd.grad(loss, self.names_learning_rates_dict.values())
                clip_task_grad_norm(grads, self.max_grad_norm)
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, zero out gradients')
                    for g in grads: g.zero_()
            
            for idx, key in enumerate(self.names_learning_rates_dict.keys()):
                self.names_learning_rates_dict[key] = nn.Parameter(self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx])
    
    def update_params(self, names_weights_dict, grads, num_step):
        return OrderedDict(
            (key, names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * grads[idx])
            for idx, key in enumerate(names_weights_dict.keys()))
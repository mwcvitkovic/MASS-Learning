import json
import math
import os
from functools import reduce
from operator import mul
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

import models
from tests import DummyWriter


def save_model_kwargs(writer, model_class_name, model_kwargs):
    pprint(model_kwargs)
    if not isinstance(writer, DummyWriter):
        path = os.path.join(writer.file_writer.get_logdir(), 'model_kwargs.json')
        with open(path, 'w') as f:
            json.dump({'model_class_name': model_class_name, 'model_kwargs': model_kwargs}, f, indent=2)


def save_model_checkpoint(writer, model, chkpt_name=None):
    if not isinstance(writer, DummyWriter):
        if chkpt_name is None:
            chkpt_name = writer.global_step
        path = os.path.join(writer.file_writer.get_logdir(), 'model_checkpoint_{}.pt'.format(chkpt_name))
        torch.save(model.state_dict(), path)


def load_model_from_checkpoint(writer, logdir, checkpoint):
    path = os.path.join(logdir, 'model_kwargs.json')
    with open(path, 'r') as f:
        decoded = json.load(f)
        model_class_name = decoded['model_class_name']
        model_kwargs = decoded['model_kwargs']
    model_class = models.__dict__[model_class_name]
    model = model_class(writer, **model_kwargs)
    state_dict = torch.load(os.path.join(logdir, 'model_checkpoint_{}.pt'.format(checkpoint)), map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def jacobian(input, output, diffable=False):
    '''
    Returns the Jacobian matrix (batch x out_size x in_size) of the function that produced the output evaluated at the input
    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        if diffable:
            J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
        else:
            J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)
    return J


class MOG(nn.Module):
    def __init__(self, n_mixture_components, rep_dim, requires_grad=False):
        """
        Stores parameters as n_mixture_components x rep_dim (x rep_dim)
        """
        super(MOG, self).__init__()
        self.n_mixture_components = n_mixture_components
        self.mixture_logits = nn.Parameter(torch.zeros(n_mixture_components), requires_grad=requires_grad)
        self.loc = nn.Parameter(torch.zeros(n_mixture_components, rep_dim), requires_grad=requires_grad)
        self.scale_tril = nn.Parameter(torch.zeros(n_mixture_components, rep_dim, rep_dim), requires_grad=requires_grad)

    def __repr__(self):
        return 'MOG, loc shape {},requires_grad={}'.format(tuple(self.loc.shape), self.loc.requires_grad)

    def log_prob(self, x, detach):
        """
        :param x: (batch x rep_dim) Tensor
        :param detach: if True, computes the log_prob using detached versions of this MOG's parameters.
           Gradients computed for this MOG's parameters won't depend on downstream processing of this function's output.
        :return: (batch) Tensor of log probabilities
        """
        if detach:
            mixture_logits = self.mixture_logits.detach()
            loc = self.loc.detach()
            scale_tril = self.scale_tril.detach()
        else:
            mixture_logits = self.mixture_logits
            loc = self.loc
            scale_tril = self.scale_tril
        return self._log_prob(x, mixture_logits, loc, scale_tril)

    @staticmethod
    def _log_prob(x, mixture_logits, loc, scale_tril):
        """
        :param x: (batch x rep_dim) Tensor
        :return: (batch) Tensor of log probabilities
        """
        component_log_probs = []
        for i in range(len(mixture_logits)):
            c_loc = loc[i, :]
            c_scale_tril = scale_tril[i, :, :]
            assert torch.sum(torch.triu(c_scale_tril.detach(), diagonal=1) != 0) == 0
            diff = x - c_loc
            M = torch.distributions.multivariate_normal._batch_mahalanobis(c_scale_tril, diff)
            half_log_det = torch.distributions.multivariate_normal._batch_diag(c_scale_tril).log().sum(-1)
            c_log_prob = -0.5 * (c_loc.shape[0] * math.log(2 * math.pi) + M) - half_log_det
            component_log_probs.append(c_log_prob)
        component_log_probs = torch.stack(component_log_probs, dim=1)
        mixture_log_probs = component_log_probs + torch.log_softmax(mixture_logits, dim=0).expand(x.shape[0],
                                                                                                  len(mixture_logits))
        return torch.logsumexp(mixture_log_probs, dim=1)


def get_MLE_of_rep_distribution(loader,
                                model,
                                max_batches,
                                n_components,
                                covariance_type,
                                return_GM=False):
    q_parameters = []
    outputs = []
    targets = []
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx > max_batches:
            break
        model.eval()
        data, target = data.to(model.device), target.to(model.device)
        output = model.net.forward(data)
        outputs.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        targets.append(target.detach())
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    for output_class in range(model.n_classes):
        print('MOG estimation class: {}'.format(output_class))
        idxs = (targets == output_class).nonzero()
        outputs_this_class = outputs[idxs, :].squeeze(1)
        mog = GaussianMixture(n_components=n_components,
                              covariance_type=covariance_type,
                              verbose=1)
        mog.fit(outputs_this_class.cpu())
        if return_GM:
            q_parameters.append(mog)
        else:
            q_parameters.append((mog.weights_, mog.means_, mog.covariances_))
    return q_parameters

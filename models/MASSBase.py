import copy

import numpy as np
import torch
import torch.optim as optim

from models.MASSVariationalDist import MASSVariationalDist
from models.ModelBase import ModelBase
from models.utils import jacobian


class MASSBase(ModelBase):
    '''
    Base class for Model trained with MASS loss
    '''

    def __init__(self,
                 writer,
                 n_classes,
                 net_name,
                 net_kwargs,
                 var_dist_init_strategy,
                 beta,
                 n_mixture_components):
        net_kwargs = copy.deepcopy(net_kwargs)
        super(MASSBase, self).__init__(writer, n_classes, True, net_name, net_kwargs)
        self.beta = beta
        self.var_dist = MASSVariationalDist(rep_dim=self.net.out_dim,
                                            n_classes=n_classes,
                                            var_dist_init_strategy=var_dist_init_strategy,
                                            n_mixture_components=n_mixture_components)

    def initialize(self, train_loader):
        self.var_dist.init_params(self, train_loader)

    def logits_from_net_output(self, output):
        raise NotImplementedError

    def loss(self, input, output, target):
        raise NotImplementedError

    def cond_ent_loss(self, output, target):
        raise NotImplementedError

    def ent_loss(self, output):
        ent_loss = 0
        if self.beta != 0.0:
            log_probs = []
            # Assumes balanced classes for now
            for q_i in self.var_dist.q:
                log_probs.append(q_i.log_prob(output, detach=False))
            log_probs = torch.stack(log_probs, dim=1) - np.log(len(self.var_dist.q))
            ent_loss = -torch.mean(torch.logsumexp(log_probs, dim=1))
        return ent_loss

    def jacobian_loss(self, input, output):
        jac_loss = 0
        if self.beta != 0.0:
            diffable = self.training
            D_f = jacobian(input, output, diffable=diffable)
            predets = torch.matmul(D_f, D_f.transpose(1, 2))
            for pd in predets:
                # Numerically stable computation of -1/2 * torch.logdet(i)
                jac_loss -= (torch.logdet(pd * pd.shape[0]) - pd.shape[0] * torch.log(
                    torch.tensor(pd.shape[0]).to(pd))) / 2
            jac_loss /= input.shape[0]
        return jac_loss

    def get_optimizer(self, optimizer_class_name, optimizer_kwargs):
        return optim.__dict__[optimizer_class_name]([
            dict(params=self.net.parameters()),
            dict(params=self.var_dist.parameters(), **optimizer_kwargs.pop('var_dist_optimizer_kwargs'))
        ],
            **optimizer_kwargs)

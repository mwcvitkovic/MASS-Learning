import copy

import torch

from models.ModelBase import ModelBase
from models.utils import MOG


class VIB(ModelBase):
    '''
    Model trained with Variational Information Bottleneck loss
    self.net is the encoder network
    '''

    def __init__(self,
                 writer,
                 n_classes,
                 net_name,
                 net_kwargs,
                 beta,
                 covariance_type=None,
                 train_var_dist_samples=None,
                 test_var_dist_samples=None,
                 n_mixture_components=None):
        net_kwargs = copy.deepcopy(net_kwargs)
        self.rep_dim = net_kwargs['out_dim']
        self.covariance_type = covariance_type
        if covariance_type == 'diag':  # We're going to need the mean and variance of a multivariate gaussian
            net_kwargs['out_dim'] *= 2
        elif covariance_type == 'full':  # We're going to need the mean and scale_tril of a multivariate gaussian
            net_kwargs['out_dim'] += int(1 / 2 * net_kwargs['out_dim'] * (net_kwargs['out_dim'] + 1))
        else:
            raise NotImplementedError
        super(VIB, self).__init__(writer, n_classes, True, net_name, net_kwargs)
        self.beta = beta
        self.train_var_dist_samples = train_var_dist_samples
        self.test_var_dist_samples = test_var_dist_samples
        self.n_mixture_components = n_mixture_components
        self.marginal = MOG(n_mixture_components, self.rep_dim)
        for p in self.marginal.parameters():
            p.requires_grad = True
        self.marginal.mixture_logits.data.zero_()
        self.marginal.loc.data.uniform_(-0.1, 0.1)
        for j in range(self.rep_dim):
            self.marginal.scale_tril.data[:, j, j] = 1
        self.decoder = torch.nn.Linear(self.rep_dim, n_classes)

    def encode(self, output):
        if self.covariance_type == 'diag':
            mean, std = torch.split(output, self.rep_dim, dim=1)
            std = torch.log(1 + torch.exp(std))
        elif self.covariance_type == 'full':
            mean, scale_tril_vec = torch.split(output, [self.rep_dim, self.net.out_dim - self.rep_dim], dim=1)
            std = torch.zeros(output.shape[0], self.rep_dim, self.rep_dim).to(output)
            std[:, torch.eye(self.rep_dim) == 1] = torch.log(1 + torch.exp(scale_tril_vec[:, :self.rep_dim]))
            std[:, torch.tril(torch.ones(self.rep_dim, self.rep_dim), -1) == 1] = scale_tril_vec[:, self.rep_dim:]
        return mean, std

    def sample_representation(self, mean, std, n_samples):
        """
        Returns MultivariateNormal samples with an extra 0th dimension of length n_samples added
        """
        eps = torch.normal(mean=torch.zeros(n_samples, *mean.shape).to(mean), std=1.0)
        if self.covariance_type == 'diag':
            return eps * std.expand_as(eps) + mean.expand_as(eps)
        elif self.covariance_type == 'full':
            return torch.matmul(std, eps.unsqueeze(-1)).squeeze(-1) + mean.expand_as(eps)

    def logits_from_net_output(self, output):
        mean, std = self.encode(output)
        if self.training:
            rep = self.sample_representation(mean, std, self.train_var_dist_samples).mean(dim=0)
        else:
            rep = self.sample_representation(mean, std, self.test_var_dist_samples).mean(dim=0)
        return torch.nn.functional.log_softmax(self.decoder(rep),
                                               dim=1)  # decoder after taking mean is okay b/c decoder is linear

    def vib_loss_kl_term(self, n_samples, rep, mean, std):
        assert rep.dim() == 2
        assert rep.shape[0] == n_samples * mean.shape[0]
        kl_term = 0
        if self.beta != 0.0:
            # Monte Carlo approximation of the KL, since there's no analytic expression for the KL of a Gaussian from an MOG
            if self.covariance_type == 'diag':
                scale_tril = torch.zeros(std.shape[0], self.rep_dim, self.rep_dim).to(rep)
                scale_tril[:, torch.eye(self.rep_dim) == 1] = std
            elif self.covariance_type == 'full':
                scale_tril = std
            enc_mvn = torch.distributions.MultivariateNormal(mean.repeat(n_samples, 1),
                                                             scale_tril=scale_tril.repeat(n_samples, 1, 1))
            enc_log_probs = enc_mvn.log_prob(rep)
            marg_log_probs = self.marginal.log_prob(rep, detach=False)
            kl_term = torch.mean(enc_log_probs - marg_log_probs)
        return kl_term

    def loss(self, input, output, target):
        mean, std = self.encode(output)
        if self.training:
            n_samples = self.train_var_dist_samples
        else:
            n_samples = self.test_var_dist_samples
        rep = self.sample_representation(mean, std, n_samples)
        rep = rep.reshape(-1, self.rep_dim)
        logits = self.decoder(rep)
        expanded_target = target.repeat(n_samples, 1).reshape(-1)
        cross_entropy_term = torch.nn.functional.cross_entropy(logits, expanded_target, reduction='mean')

        kl_term = self.beta * self.vib_loss_kl_term(n_samples, rep, mean, std)

        loss = cross_entropy_term + kl_term
        if self.training and self.writer.global_step % self.writer.train_loss_plot_interval == 0:
            self.writer.add_scalar('Train Loss/VIB cross entropy term', cross_entropy_term, self.writer.global_step)
            self.writer.add_scalar('Train Loss/VIB KL term', kl_term, self.writer.global_step)
        return loss

    def rate(self, output):
        mean, std = self.encode(output)
        if self.training:
            n_samples = self.train_var_dist_samples
        else:
            n_samples = self.test_var_dist_samples
        rep = self.sample_representation(mean, std, n_samples)
        rep_flat = rep.reshape(-1, self.rep_dim)

        # Monte Carlo approximation of the KL, since there's no analytic expression for the KL of a Gaussian from an MOG
        if self.covariance_type == 'diag':
            scale_tril = torch.zeros(std.shape[0], self.rep_dim, self.rep_dim).to(rep_flat)
            scale_tril[:, torch.eye(self.rep_dim) == 1] = std
        elif self.covariance_type == 'full':
            scale_tril = std
        enc_mvn = torch.distributions.MultivariateNormal(mean.repeat(n_samples, 1),
                                                         scale_tril=scale_tril.repeat(n_samples, 1, 1))
        enc_log_probs = enc_mvn.log_prob(rep_flat)
        marg_log_probs = self.marginal.log_prob(rep_flat, detach=False)

        return (enc_log_probs - marg_log_probs).reshape(n_samples, -1).mean(dim=0)

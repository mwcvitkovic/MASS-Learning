import torch
from torch import nn

from models.utils import MOG, get_MLE_of_rep_distribution


class MASSVariationalDist(nn.Module):
    '''
    Module encapsulating the MASS variational distribution q(x|y)
    '''

    def __init__(self,
                 rep_dim,
                 n_classes,
                 var_dist_init_strategy,
                 n_mixture_components):
        super(MASSVariationalDist, self).__init__()
        self.rep_dim = rep_dim
        self.var_dist_init_strategy = var_dist_init_strategy
        self.n_mixture_components = n_mixture_components
        self.n_classes = n_classes
        self.q = nn.ModuleList()
        for i in range(self.n_classes):
            mog = MOG(n_mixture_components, rep_dim, requires_grad=True)
            self.q.append(mog)

    def init_params(self, model, train_loader):
        if self.var_dist_init_strategy == 'zeros':
            for i, mog in enumerate(self.q):
                mog.loc.data.uniform_(-0.001, 0.001)  # Breaking symmetry
                for j in range(self.rep_dim):
                    mog.scale_tril.data[:, j, j] = 1
        elif self.var_dist_init_strategy == 'random':
            for mog in self.q:
                mog.mixture_logits.data.uniform_(-0.1, 0.1)
                mog.loc.data.uniform_(-10, 10)
                mog.scale_tril.data.uniform_(-1, 1)
                mog.scale_tril.data = torch.stack([torch.tril(st) for st in mog.scale_tril.data])
                for j in range(self.rep_dim):
                    mog.scale_tril.data[:, j, j] = 1
        elif self.var_dist_init_strategy == 'standard_basis':
            for i, mog in enumerate(self.q):
                if i < self.rep_dim:
                    mog.loc.data[0, i] = 10
                    mog.loc.data[1:, i].uniform_(-5, 5)
                else:
                    mog.loc.data.uniform_(-10, 10)
                for j in range(self.rep_dim):
                    mog.scale_tril.data[:, j, j] = 1
        elif self.var_dist_init_strategy == 'MLE_from_training_data':
            q_params = get_MLE_of_rep_distribution(loader=train_loader,
                                                   model=model,
                                                   max_batches=50,
                                                   n_components=self.n_mixture_components,
                                                   covariance_type='full')
            for i, mog in enumerate(self.q):
                mog.mixture_logits.data = torch.log(torch.Tensor(q_params[i][0]).to(mog.mixture_logits))
                mog.loc.data = torch.Tensor(q_params[i][1]).to(mog.loc)
                mog.scale_tril.data = torch.cholesky(torch.Tensor(q_params[i][2]).to(mog.scale_tril))
        else:
            raise NotImplementedError

    def rep_to_logits(self, representation):
        # Assumes balanced classes, for now
        logits = []
        for q_i in self.q:
            logits.append(q_i.log_prob(representation, detach=False))
        logits = nn.functional.log_softmax(torch.stack(logits, dim=1), dim=1)
        return logits

import torch.nn.functional as F

from experiments.ExperimentBase import ExperimentBase
from models import MASSBase


class MASSLossTerms(ExperimentBase):
    """
    Estimates the value of the terms in the MASS loss function (cross entropy term, entropy term, and jacobian term) on
    the output of any network.
    The terms are estimated by making a mixture of Gaussian MLE of the variational distribution q(f(X)|y) and
      computing the loss terms based on that.
    """

    def __init__(self, **kwargs):
        super(MASSLossTerms, self).__init__(kwargs['writer'])
        self.model = kwargs['model']
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.run_interval = kwargs.get('run_interval', None)
        self.max_batches = kwargs['max_batches']
        self.n_mixture_components = kwargs['n_mixture_components']
        self.covariance_type = kwargs['covariance_type']

    def run(self, batch_idx, epoch):
        self.model.eval()

        # Creating empty MASS model to estimate the MASS loss terms
        dummy_MASS = MASSBase(writer=None,
                              n_classes=self.model.n_classes,
                              net_name='NullNet',
                              net_kwargs={'out_dim': self.model.net.out_dim},
                              # var_dist_update=None,
                              var_dist_init_strategy='MLE_from_training_data',
                              beta=None,
                              n_mixture_components=self.n_mixture_components,
                              covariance_type=self.covariance_type)
        dummy_MASS.net = self.model.net
        dummy_MASS.to(self.model.device)
        dummy_MASS.eval()
        dummy_MASS.initialize(self.train_loader)

        for loader_name, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            cross_ent_term = 0
            ent_term = 0
            jacobian_term = 0

            # Get all the loss terms
            assert 0 < self.max_batches < len(loader)
            assert self.model.net.batch_norm == False  # Batch norm leads to bad jacobian estimation for self.model
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx > self.max_batches:
                    break
                data, target = data.to(self.model.device), target.to(self.model.device)
                data.requires_grad = True
                output = self.model.net.forward(data)
                cross_ent_term += F.nll_loss(dummy_MASS.var_dist.rep_to_logits(output), target)
                ent_term += dummy_MASS.ent_loss(output)
                jacobian_term += dummy_MASS.jacobian_loss(data, output)
            cross_ent_term /= (batch_idx - 1)
            ent_term /= (batch_idx - 1)
            jacobian_term /= (batch_idx - 1)

            self.writer.add_scalar('MASSLossTerms/{}, cross entropy term'.format(loader_name),
                                   cross_ent_term,
                                   self.writer.global_step)
            self.writer.add_scalar('MASSLossTerms/{}, entropy term'.format(loader_name),
                                   ent_term,
                                   self.writer.global_step)
            self.writer.add_scalar('MASSLossTerms/{}, Jacobian term'.format(loader_name),
                                   jacobian_term,
                                   self.writer.global_step)

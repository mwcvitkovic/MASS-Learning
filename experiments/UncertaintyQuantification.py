import numpy as np
import torch

from experiments.ExperimentBase import ExperimentBase


def entropy(probs):
    SIs = -torch.log(probs)
    SIs[SIs == np.inf] = 0
    SIs[SIs == np.nan] = 0  # 0 log 0 is defined as 0
    return torch.sum(probs * SIs, dim=1).mean().item()


class UncertaintyQuantification(ExperimentBase):
    def __init__(self, **kwargs):
        super(UncertaintyQuantification, self).__init__(kwargs['writer'])
        self.model = kwargs['model']
        self.val_loader = kwargs['val_loader']
        self.run_interval = kwargs.get('run_interval', None)

    def run(self, batch_idx, epoch):
        self.model.eval()
        NLL = torch.nn.NLLLoss()
        MSE = torch.nn.MSELoss()
        val_nll = 0
        val_brier = 0
        val_entropy = 0
        for data, target in self.val_loader:
            data, target = data.to(self.model.device), target.to(self.model.device)
            output = self.model.net(data)
            logits = self.model.logits_from_net_output(output)
            probs = torch.softmax(logits, dim=1).detach()
            val_nll += NLL(logits.detach(), target).item()
            val_entropy += entropy(probs)
            one_hot_target = torch.zeros(probs.shape).to(data)
            one_hot_target.scatter_(1, target.unsqueeze(1), 1)
            val_brier += MSE(probs, one_hot_target).item()
        val_nll /= len(self.val_loader)
        val_brier /= len(self.val_loader)
        val_entropy /= len(self.val_loader)

        self.writer.add_scalar('UncertaintyQuantification/NLL Loss',
                               val_nll,
                               self.writer.global_step)
        self.writer.add_scalar('UncertaintyQuantification/Entropy of Output Distribution',
                               val_entropy,
                               self.writer.global_step)
        self.writer.add_scalar('UncertaintyQuantification/Brier Score',
                               val_brier,
                               self.writer.global_step)

        return {'val_nll': val_nll, 'val_entropy': val_entropy, 'val_brier': val_brier}

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.mixture import GaussianMixture

from experiments.ExperimentBase import ExperimentBase
from models import VIB, MASSBase
from models.utils import get_MLE_of_rep_distribution, MOG
from utils import get_dataloaders


def batch_entropy(probs):
    SIs = -torch.log(probs)
    SIs[SIs == np.inf] = 0
    SIs[SIs == np.nan] = 0  # 0 log 0 is defined as 0
    return torch.sum(probs * SIs, dim=1)


def batch_max_var_dist_log_density(outputs, var_dist):
    """
    A higher max log density indicates the model believes the data are in-distribution
    """
    log_probs = []
    for q_i in var_dist:
        if isinstance(q_i, MOG):
            log_probs.append(q_i.log_prob(outputs, detach=True))
        elif isinstance(q_i, GaussianMixture):
            log_probs.append(torch.Tensor(q_i.score_samples(outputs.detach().cpu().numpy())))
    log_probs = torch.stack(log_probs, dim=1)
    return log_probs.max(dim=1)[0]


class OODDetection(ExperimentBase):
    def __init__(self, **kwargs):
        super(OODDetection, self).__init__(kwargs['writer'])
        self.model = kwargs['model']
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.run_interval = kwargs.get('run_interval', None)
        OOD_loader, _, _, _, _ = get_dataloaders('SVHN',
                                                 self.val_loader.batch_size,
                                                 len(self.val_loader.dataset),
                                                 0,
                                                 str(self.model.device),
                                                 normalize_inputs=True)
        self.OOD_loader = OOD_loader

    def run(self, batch_idx, epoch):
        self.model.eval()
        ent_scores = []
        max_density_scores = []
        if isinstance(self.model, (MASSBase, VIB)):
            alternate_scores = []

        # Assemble results
        q = get_MLE_of_rep_distribution(self.train_loader,
                                        self.model,
                                        max_batches=100,
                                        n_components=10,
                                        covariance_type='full',
                                        return_GM=True)
        for loader in [self.val_loader, self.OOD_loader]:
            for data, _ in loader:
                data = data.to(self.model.device)
                output = self.model.net(data)
                logits = self.model.logits_from_net_output(output)
                probs = torch.softmax(logits, dim=1)
                ent_scores.append(batch_entropy(probs).detach().cpu())
                max_density_scores.append(batch_max_var_dist_log_density(output, q).detach().cpu())
                if isinstance(self.model, MASSBase):
                    alternate_scores.append(
                        batch_max_var_dist_log_density(output, self.model.var_dist.q).detach().cpu())
                elif isinstance(self.model, VIB):
                    alternate_scores.append(-self.model.rate(output).detach().cpu())

        # The higher the score, the more the model is predicting a sample is OOD
        ent_scores = torch.cat(ent_scores).numpy()
        max_density_scores = -torch.cat(max_density_scores).numpy()
        if isinstance(self.model, (MASSBase, VIB)):
            alternate_scores = -torch.cat(alternate_scores).numpy()
            alternate_scores[np.where(np.isnan(alternate_scores))] = 1e6
        targets = torch.cat([torch.zeros(len(self.val_loader.dataset)),
                             torch.ones(len(self.OOD_loader.dataset))]).to(torch.int32).cpu().numpy()

        # Log everything
        auroc_ent = roc_auc_score(targets, ent_scores)
        auroc_density = roc_auc_score(targets, max_density_scores)
        self.writer.add_scalar('OODDetection/OOD AUROC Entropy',
                               auroc_ent,
                               self.writer.global_step)
        self.writer.add_scalar('OODDetection/OOD AUROC Max Density',
                               auroc_density,
                               self.writer.global_step)
        auroc_alternate = None
        if isinstance(self.model, (MASSBase, VIB)):
            auroc_alternate = roc_auc_score(targets, alternate_scores)
            self.writer.add_scalar('OODDetection/OOD AUROC Alternate',
                                   auroc_alternate,
                                   self.writer.global_step)

        apr_out_ent = average_precision_score(targets, ent_scores)
        apr_out_density = average_precision_score(targets, max_density_scores)
        self.writer.add_scalar('OODDetection/OOD APR Out Entropy',
                               apr_out_ent,
                               self.writer.global_step)
        self.writer.add_scalar('OODDetection/OOD APR Out Max Density',
                               apr_out_density,
                               self.writer.global_step)
        apr_out_alternate = None
        if isinstance(self.model, (MASSBase, VIB)):
            apr_out_alternate = average_precision_score(targets, alternate_scores)
            self.writer.add_scalar('OODDetection/OOD APR Out Alternate',
                                   apr_out_alternate,
                                   self.writer.global_step)

        targets = 1 - targets
        apr_in_ent = average_precision_score(targets, -ent_scores)
        apr_in_density = average_precision_score(targets, -max_density_scores)
        self.writer.add_scalar('OODDetection/OOD APR In Entropy',
                               apr_in_ent,
                               self.writer.global_step)
        self.writer.add_scalar('OODDetection/OOD APR In Max Density',
                               apr_in_density,
                               self.writer.global_step)
        apr_in_alternate = None
        if isinstance(self.model, (MASSBase, VIB)):
            apr_in_alternate = average_precision_score(targets, -alternate_scores)
            self.writer.add_scalar('OODDetection/OOD APR In Alternate',
                                   apr_in_alternate, self.writer.global_step)

        return dict(auroc_ent=auroc_ent,
                    auroc_density=auroc_density,
                    auroc_alternate=auroc_alternate,
                    apr_out_ent=apr_out_ent,
                    apr_out_density=apr_out_density,
                    apr_out_alternate=apr_out_alternate,
                    apr_in_ent=apr_in_ent,
                    apr_in_density=apr_in_density,
                    apr_in_alternate=apr_in_alternate)

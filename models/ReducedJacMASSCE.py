import torch

from models import MASSCE


class ReducedJacMASSCE(MASSCE):
    """
    Model that trains with MASSCE loss, but with the Jacobian term estimated with a subset of the batch for efficiency
    """

    def net_forward_and_loss(self, input, target):
        # Split the loss so the jacobian term is only estimated using a fraction of the batch
        # Splitting to avoid batches of size 1, which plays badly with batch norm
        jac_estimation_samples = max(self.net.out_dim // 2, input.shape[0] // self.net.out_dim)
        in_jac = input[:jac_estimation_samples]
        in_no_jac = input[jac_estimation_samples:]
        out_jac = self.net.forward(in_jac)
        out_no_jac = self.net.forward(in_no_jac)
        output = torch.cat([out_jac, out_no_jac], dim=0)

        cross_ent_loss = self.cross_ent_loss(output, target)
        ent_loss = self.beta * self.ent_loss(output)
        jac_loss = self.beta * self.jacobian_loss(in_jac, out_jac)
        loss = cross_ent_loss + ent_loss + jac_loss
        if self.training and self.writer.global_step % self.writer.train_loss_plot_interval == 0:
            self.writer.add_scalar('Train Loss/MASSCE cross entropy term', cross_ent_loss, self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE entropy term', ent_loss, self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE cross ent plus ent terms', cross_ent_loss + ent_loss,
                                   self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE entropy plus Jacobian terms', ent_loss + jac_loss,
                                   self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE Jacobian term', jac_loss, self.writer.global_step)
        return loss, output

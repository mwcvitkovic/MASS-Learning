import torch.nn.functional as F

from models.MASSBase import MASSBase


class MASSCE(MASSBase):
    '''
    Model trained with MASS loss where the -I(Z,Y) term is minimized as H(Y|Z) computed from var_dist
    '''

    def loss(self, input, output, target):
        cross_ent_loss = self.cross_ent_loss(output, target)
        ent_loss = self.beta * self.ent_loss(output)
        jac_loss = self.beta * self.jacobian_loss(input, output)
        loss = cross_ent_loss + ent_loss + jac_loss
        if self.training and self.writer.global_step % self.writer.train_loss_plot_interval == 0:
            self.writer.add_scalar('Train Loss/MASSCE cross entropy term', cross_ent_loss, self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE entropy term', ent_loss, self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE cross ent plus ent terms', cross_ent_loss + ent_loss,
                                   self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE entropy plus Jacobian terms', ent_loss + jac_loss,
                                   self.writer.global_step)
            self.writer.add_scalar('Train Loss/MASSCE Jacobian term', jac_loss, self.writer.global_step)
        return loss

    def logits_from_net_output(self, output):
        return self.var_dist.rep_to_logits(output)

    def cross_ent_loss(self, output, target):
        return F.nll_loss(self.var_dist.rep_to_logits(output), target)

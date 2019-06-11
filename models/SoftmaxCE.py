import copy

import torch
from torch import nn

from models.ModelBase import ModelBase


class SoftmaxCE(ModelBase):
    '''
    Standard feedforward model
    '''

    def __init__(self, writer, n_classes, net_name, net_kwargs):
        net_kwargs = copy.deepcopy(net_kwargs)
        input_grad = False
        super(SoftmaxCE, self).__init__(writer, n_classes, input_grad, net_name, net_kwargs)
        self.loss_fxn = nn.CrossEntropyLoss()

    def logits_from_net_output(self, output):
        return torch.nn.functional.log_softmax(output, dim=1)

    def loss(self, input, output, target):
        loss = self.loss_fxn(output, target)
        return loss

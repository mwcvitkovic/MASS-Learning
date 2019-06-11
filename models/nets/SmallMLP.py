import numpy as np
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self, writer, input_grad, in_shape, out_dim, nonlinearity, batch_norm, dropout):
        super(SmallMLP, self).__init__()
        self.writer = writer
        self.input_grad = input_grad
        self.in_dim = np.prod(in_shape)
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fcout = nn.Linear(200, out_dim)
        self.nlin = nn.functional.__dict__[nonlinearity]
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(400)
            self.bn2 = nn.BatchNorm1d(200)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
        if dropout:
            self.do1 = nn.Dropout()
            self.do2 = nn.Dropout()
        else:
            self.do1 = lambda x: x
            self.do2 = lambda x: x

    def forward(self, x):
        if self.input_grad:
            x.requires_grad = True
        x = x.view(-1, self.in_dim)
        x = self.bn1(self.fc1(x))
        self.writer.debug_info('Model Outputs/fc1 preactivations', x, self.writer.global_step)
        x = self.nlin(x)
        self.writer.debug_info('Model Outputs/fc1 activations', x, self.writer.global_step)
        x = self.do1(x)
        x = self.bn2(self.fc2(x))
        self.writer.debug_info('Model Outputs/fc2 preactivations', x, self.writer.global_step)
        x = self.nlin(x)
        self.writer.debug_info('Model Outputs/fc2 activations', x, self.writer.global_step)
        x = self.do2(x)
        x = self.fcout(x)
        self.writer.debug_info('Model Outputs/representation', x, self.writer.global_step)
        return x

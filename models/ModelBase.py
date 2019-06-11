import torch.optim as optim
from torch import nn

import models.nets as nets


class ModelBase(nn.Module):
    '''
    Base class for all test_models
    '''

    def __init__(self, writer, n_classes, input_grad, net_name, net_kwargs):
        super(ModelBase, self).__init__()
        self.writer = writer
        self.n_classes = n_classes
        self.net = nets.__dict__[net_name](writer, input_grad, **net_kwargs)

    def initialize(self, train_loader):
        """
        Does any necessary model initialization.
        Runs one time, before training starts.
        """
        pass

    def forward(self, x):
        """
        Returns logits for output classes
        """
        x = self.net.forward(x)
        return self.logits_from_net_output(x)

    def logits_from_net_output(self, output):
        """
        Returns the logits (log probabilities) for model's prediction of the class given the output of self.net.forward
        """
        raise NotImplementedError

    def pred_from_net_output(self, output):
        """
        Returns the model's prediction of the class of the input given the output of self.net.forward
        """
        logits = self.logits_from_net_output(output)
        return logits.max(1, keepdim=True)[1]

    def loss(self, input, output, target):
        """
        Returns the differentiable loss for the model.
        output argument is the output of self.net.forward
        """
        raise NotImplementedError

    def net_forward_and_loss(self, input, target):
        """
        Combines net forward and loss computation - needed for models like ReducedJacMASSCE that
        need to run their net separately on different parts of the input batch
        """
        output = self.net.forward(input)
        return self.loss(input, output, target), output

    def get_optimizer(self, optimizer_class_name, optimizer_kwargs):
        return optim.__dict__[optimizer_class_name](self.parameters(), **optimizer_kwargs)

    @property
    def device(self):
        return self.parameters().__next__().device

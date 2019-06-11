from datetime import datetime

import numpy as np
import pytz

from experiments.ExperimentBase import ExperimentBase
from models.utils import save_model_checkpoint


class ModelLossAndAccuracy(ExperimentBase):
    def __init__(self, **kwargs):
        super(ModelLossAndAccuracy, self).__init__(kwargs['writer'])
        self.model = kwargs['model']
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.run_interval = kwargs.get('run_interval', None)
        self.full_train_set_MLE = kwargs.get('full_train_set_MLE', False)
        self.n_mixture_components = kwargs.get('n_mixture_components', None)
        self.covariance_type = kwargs.get('covariance_type', None)
        self.last_acc = -np.inf
        self.best_acc = -np.inf

    def run(self, batch_idx, epoch):
        self.model.eval()
        val_loss = 0
        n_correct = 0
        for data, target in self.val_loader:
            data, target = data.to(self.model.device), target.to(self.model.device)
            output = self.model.net.forward(data)
            val_loss += self.model.loss(data, output, target).item()  # sum up mean batch losses
            pred = self.model.pred_from_net_output(output)
            n_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.val_loader)
        self.writer.add_scalar('ModelLossAndAccuracy/Validation Loss', val_loss, self.writer.global_step)
        val_accuracy = 100 * n_correct / len(self.val_loader.dataset)
        self.writer.add_scalar('ModelLossAndAccuracy/Validation Accuracy', val_accuracy,
                               self.writer.global_step)
        if val_accuracy > self.best_acc:
            save_model_checkpoint(self.writer, self.model, 'best')
            self.best_acc = val_accuracy
        self.last_acc = val_accuracy

        print(
            '[{}] Global Step {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tVal Loss: {:.6f}\t Val Accuracy: {:.6f}'.format(
                datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S (%Z)'),
                self.writer.global_step,
                epoch,
                batch_idx * data.shape[0],
                len(self.train_loader.dataset),
                100 * batch_idx / len(self.train_loader),
                val_loss,
                val_accuracy)
        )

        return {'val_accuracy': val_accuracy, 'val_loss': val_loss}

from experiments.ExperimentBase import ExperimentBase
from models.utils import save_model_checkpoint


class SaveModelParameters(ExperimentBase):
    def __init__(self, **kwargs):
        super(SaveModelParameters, self).__init__(kwargs['writer'])
        self.model = kwargs['model']
        self.run_interval = kwargs.get('run_interval', None)

    def run(self, batch_idx, epoch):
        self.model.eval()
        save_model_checkpoint(self.writer, self.model)

from experiments.ExperimentBase import ExperimentBase
from utils import log_param_values, log_var_dist_parameters


class LogModelParameters(ExperimentBase):
    def __init__(self, **kwargs):
        super(LogModelParameters, self).__init__(kwargs['writer'])
        self.model = kwargs['model']
        self.run_interval = kwargs.get('run_interval', None)
        self.log_var_dist = kwargs['log_var_dist']

    def run(self, batch_idx, epoch):
        self.model.eval()
        log_param_values(self.writer, self.model)
        if self.log_var_dist and hasattr(self.model, 'var_dist'):
            log_var_dist_parameters(self.writer, self.model.var_dist)

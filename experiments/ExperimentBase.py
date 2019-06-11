class ExperimentBase:
    """
    Base Class for all experiments
    """

    def __init__(self, writer):
        self.writer = writer

    def run(self, *args):
        raise NotImplementedError

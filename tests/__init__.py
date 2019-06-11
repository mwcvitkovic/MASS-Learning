class DummyWriter:
    def __init__(self):
        self.global_step = 0
        self.train_loss_plot_interval = 1

    def add_histogram(self, *args, **kwargs):
        pass

    def add_scalar(self, tag_name, object, iter_number, *args, **kwargs):
        if tag_name == 'Train Loss/Train Loss':
            # For testing purposes
            self.train_loss = object
            print('Train Loss step {} = {}'.format(iter_number, object))
        else:
            pass

    def debug_info(self, *args, **kwargs):
        pass

import unittest

from start_training import train_model
from tests import DummyWriter


class TestMASSCE(unittest.TestCase):
    def test_MASS_var_dist_SGD_update_memorize_minibatch(self):
        writer = DummyWriter()
        kwargs = dict(
            seed=632,
            dataset_name='CIFAR10',
            model_class_name='MASSCE',
            model_kwargs=dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=10,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False,
                ),
                var_dist_init_strategy='standard_basis',
                beta=0.001,
                n_mixture_components=2,
            ),
            normalize_inputs=True,
            batch_size=25,
            train_size=25,
            val_size=0,
            epochs=50,
            total_batches=None,
            optimizer_class_name='Adam',
            optimizer_kwargs=dict(
                lr=0.001,
                var_dist_optimizer_kwargs=dict(
                    lr=0.001
                )
            ),
            lr_scheduler_class_name='ExponentialLR',
            lr_scheduler_kwargs=dict(
                gamma=1.0
            ),
            device_id='cuda:0')
        train_model(writer, **kwargs)
        self.assertLess(writer.train_loss, 0.1)

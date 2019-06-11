import copy
import unittest

import numpy as np
import torch

from models import VIB
from start_training import train_model
from tests import DummyWriter


class TestVIB(unittest.TestCase):
    def test_VIB_memorize_minibatch(self):
        for covariance_type in ['diag', 'full']:
            writer = DummyWriter()
            kwargs = dict(
                seed=632,
                dataset_name='CIFAR10',
                model_class_name='VIB',
                model_kwargs=dict(
                    net_name='SmallMLP',
                    net_kwargs=dict(
                        out_dim=10,
                        nonlinearity='elu',
                        batch_norm=True,
                        dropout=False,
                    ),
                    covariance_type=covariance_type,
                    beta=0.001,
                    n_mixture_components=2,
                    train_var_dist_samples=9,
                    test_var_dist_samples=12,
                ),
                normalize_inputs=True,
                batch_size=25,
                train_size=25,
                val_size=0,
                epochs=100,
                total_batches=None,
                optimizer_class_name='Adam',
                optimizer_kwargs=dict(
                    lr=0.001,
                ),
                lr_scheduler_class_name='ExponentialLR',
                lr_scheduler_kwargs=dict(
                    gamma=1.0
                ),
                device_id='cuda:0')
            train_model(writer, **kwargs)
            self.assertLess(writer.train_loss, 0.1)
            self.assertGreater(writer.train_loss, 0.0)

    def test_VIB_init(self):
        for covariance_type in ['diag', 'full']:
            writer = DummyWriter()
            model_kwargs = dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=10,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False,
                    in_shape=(1, 28, 28),
                ),
                covariance_type=covariance_type,
                beta=0.001,
                n_mixture_components=2,
                train_var_dist_samples=8,
                test_var_dist_samples=12,
                n_classes=10,
            )
            model = VIB(writer, **model_kwargs)
        for m in model.marginal.scale_tril.detach():
            np.testing.assert_array_equal(0, torch.triu(m, diagonal=1))

    def test_VIB_sample_representation(self):
        torch.manual_seed(42)
        np.random.seed(42)

        n_samples, n_batch, rep_dim = (9, 256, 10)

        for covariance_type in ['diag', 'full']:
            writer = DummyWriter()
            model_kwargs = dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=rep_dim,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False,
                    in_shape=(1, 28, 28),
                ),
                covariance_type=covariance_type,
                beta=0.001,
                n_mixture_components=2,
                train_var_dist_samples=n_samples,
                test_var_dist_samples=12,
                n_classes=10,
            )
            model = VIB(writer, **model_kwargs)

            # Here every row should be close to an ascending integer
            mean = torch.Tensor(np.arange(0, n_batch))
            mean = mean.unsqueeze(1).repeat(1, rep_dim)
            if covariance_type == 'diag':
                std = torch.ones(rep_dim).repeat(n_batch, 1)
            elif covariance_type == 'full':
                std = torch.eye(rep_dim, rep_dim).repeat(n_batch, 1, 1)
            std *= 1e-4
            samples = model.sample_representation(mean, std, model.train_var_dist_samples)
            self.assertEqual(tuple(samples.shape), (n_samples, n_batch, rep_dim))
            np.testing.assert_allclose(np.arange(0, n_batch), torch.mean(samples, dim=[0, 2]), atol=1e-3)

            # And here every column should be close to an ascending integer
            mean = torch.Tensor(np.arange(0, rep_dim))
            mean = mean.repeat(n_batch, 1)
            if covariance_type == 'diag':
                std = torch.ones(rep_dim).repeat(n_batch, 1)
            elif covariance_type == 'full':
                std = torch.eye(rep_dim, rep_dim).repeat(n_batch, 1, 1)
            std *= 1e-4
            samples = model.sample_representation(mean, std, model.train_var_dist_samples)
            self.assertEqual(tuple(samples.shape), (n_samples, n_batch, rep_dim))
            np.testing.assert_allclose(np.arange(0, rep_dim), torch.mean(samples, dim=[0, 1]), atol=1e-3)

            # And here all the elements in the representation should be tightly correlated, but differ between samples
            if covariance_type == 'full':
                mean = torch.zeros(n_batch, rep_dim)
                std = torch.eye(rep_dim, rep_dim).repeat(n_batch, 1, 1) * 1e-4
                std[:, :, 0] = 1
                samples = model.sample_representation(mean, std, model.train_var_dist_samples)

                intra_rep_differences = torch.max(samples, dim=2)[0] - torch.min(samples, dim=2)[0]
                self.assertEqual(tuple(intra_rep_differences.shape), (n_samples, n_batch))
                np.testing.assert_array_less(intra_rep_differences, 1e-3)

                inter_sample_differences = torch.max(samples, dim=0)[0] - torch.min(samples, dim=0)[0]
                self.assertEqual(tuple(inter_sample_differences.shape), (n_batch, rep_dim))
                np.testing.assert_array_less(1, inter_sample_differences)

                inter_batch_differences = torch.max(samples, dim=1)[0] - torch.min(samples, dim=1)[0]
                self.assertEqual(tuple(inter_batch_differences.shape), (n_samples, rep_dim))
                np.testing.assert_array_less(1, inter_batch_differences)

    def test_VIB_encode(self):
        n_samples, n_batch, rep_dim = (9, 256, 10)
        in_shape = (3, 32, 32)

        for covariance_type in ['diag', 'full']:
            writer = DummyWriter()
            model_kwargs = dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=rep_dim,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False,
                    in_shape=in_shape,
                ),
                covariance_type=covariance_type,
                beta=0.001,
                n_mixture_components=2,
                train_var_dist_samples=n_samples,
                test_var_dist_samples=12,
                n_classes=10,
            )
            model = VIB(writer, **model_kwargs)

            data = torch.ones(n_batch, *in_shape)
            output = model.net.forward(data)
            mean, std = model.encode(output)
            self.assertEqual(tuple(mean.shape), (n_batch, rep_dim))
            if covariance_type == 'diag':
                self.assertEqual(tuple(output.shape), (n_batch, 2 * rep_dim))
                self.assertEqual(tuple(std.shape), (n_batch, rep_dim))
            elif covariance_type == 'full':
                self.assertEqual(tuple(output.shape), (n_batch, rep_dim + rep_dim * (rep_dim + 1) / 2))
                self.assertEqual(tuple(std.shape), (n_batch, rep_dim, rep_dim))

    def test_VIB_vib_loss_kl_term(self):
        n_samples, n_batch, rep_dim = (50, 256, 10)
        in_shape = (3, 32, 32)

        for covariance_type in ['diag', 'full']:
            writer = DummyWriter()
            model_kwargs = dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=rep_dim,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False,
                    in_shape=in_shape,
                ),
                covariance_type=covariance_type,
                beta=0.001,
                n_mixture_components=5,
                train_var_dist_samples=n_samples,
                test_var_dist_samples=12,
                n_classes=10,
            )
            model = VIB(writer, **model_kwargs)

            mean = torch.zeros(n_batch, rep_dim)
            if covariance_type == 'diag':
                std = torch.ones(rep_dim).repeat(n_batch, 1)
            elif covariance_type == 'full':
                std = torch.eye(rep_dim, rep_dim).repeat(n_batch, 1, 1)
            n_samples = model.train_var_dist_samples
            rep = model.sample_representation(mean, std, n_samples)
            rep = rep.reshape(-1, rep_dim)

            # Here the empirical kl should be large and positive (sample dist is far from marginal)
            np.testing.assert_array_equal(0, model.marginal.mixture_logits.detach())
            model.marginal.loc.data += 100
            np.testing.assert_array_less(model.marginal.scale_tril.detach(), 1.1)
            self.assertGreater(1e5, model.vib_loss_kl_term(n_samples, rep, mean, std))

            # Here it should be near 0 (sample dist is close to marginal)
            model.marginal.mixture_logits.data = torch.zeros(5)
            model.marginal.loc.data = torch.zeros(5, rep_dim)
            np.testing.assert_array_less(model.marginal.scale_tril.detach(), 1.1)
            self.assertAlmostEqual(0, model.vib_loss_kl_term(n_samples, rep, mean, std), delta=0.1)

            # Here it should be large and negative (samples is closer to marginal than to our manipulated mean and std)
            # (This wouldn't happen during training)
            big_mean = mean + 100
            self.assertLess(-1e5, model.vib_loss_kl_term(n_samples, rep, big_mean, std))

            # Here it should be exactly 0
            model.marginal.mixture_logits.data = torch.zeros(5)
            model.marginal.loc.data = torch.zeros(5, rep_dim)
            model.marginal.scale_tril.data = torch.eye(rep_dim).repeat(5, 1, 1)
            self.assertAlmostEqual(0, model.vib_loss_kl_term(n_samples, rep, mean, std), delta=1e-8)

    def test_VIB_marginal_update(self):
        """
        Make sure SGD is actually updating the parameters of the marginal distribution
        """
        n_samples, n_batch, rep_dim = (50, 256, 10)
        in_shape = (3, 32, 32)
        data = torch.zeros(n_batch, *in_shape).uniform_(0, 1)
        target = torch.zeros(n_batch, dtype=torch.int64).random_(0, rep_dim)
        optimizer_class_name = 'SGD'
        optimizer_kwargs = dict(
            lr=1e-3,
            momentum=0.9,
        )
        writer = DummyWriter()
        model_kwargs = dict(
            net_name='SmallMLP',
            net_kwargs=dict(
                out_dim=rep_dim,
                nonlinearity='elu',
                batch_norm=True,
                dropout=False,
                in_shape=in_shape,
            ),
            covariance_type='full',
            beta=0.001,
            n_mixture_components=5,
            train_var_dist_samples=n_samples,
            test_var_dist_samples=12,
            n_classes=10,
        )
        model_kwargs['n_classes'] = 10
        model_kwargs['net_kwargs']['in_shape'] = in_shape
        model = VIB(writer, **model_kwargs)
        model.initialize(None)

        orig_marg_params = copy.deepcopy(list(model.marginal.named_parameters()))
        optimizer = model.get_optimizer(optimizer_class_name, optimizer_kwargs)

        model.train()
        optimizer.zero_grad()
        output = model.net.forward(data)
        loss = model.loss(data, output, target)

        # Make sure gradients start not initialized
        for nparam in model.marginal.named_parameters():
            self.assertIsNone(nparam[1].grad)
        loss.backward()
        for nparam in model.marginal.named_parameters():
            self.assertIsNotNone(nparam[1].grad)

        # Make sure none of the parameters changed values
        for orig_nparam, nparam in zip(orig_marg_params, model.marginal.named_parameters()):
            self.assertEqual(orig_nparam[0], nparam[0])
            np.testing.assert_array_equal(orig_nparam[1].detach(), nparam[1].detach())

        optimizer.step()
        # Make sure marginal parameters changed values
        for orig_nparam, nparam in zip(orig_marg_params, model.marginal.named_parameters()):
            self.assertEqual(orig_nparam[0], nparam[0])
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(orig_nparam[1].detach(), nparam[1].detach())

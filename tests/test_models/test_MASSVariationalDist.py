import copy
import unittest

import numpy as np
import torch

from models import MASSCE
from tests import DummyWriter


class TestMASSVariationalDist(unittest.TestCase):
    def test_MASS_var_dist_random_init(self):
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
            var_dist_init_strategy='random',
            beta=0.9,
            n_mixture_components=10,
            n_classes=10,
        )
        model = MASSCE(writer, **model_kwargs)
        # Make sure everything started as zeros
        for p in model.var_dist.parameters():
            self.assertEqual((p != 0).sum(), 0)
            self.assertTrue(p.requires_grad)

        # Make sure initialization worked
        model.initialize(None)
        for mog in model.var_dist.q:
            self.assertEqual((mog.mixture_logits > 0.1).sum(), 0)
            self.assertEqual((mog.mixture_logits < -0.1).sum(), 0)
            self.assertEqual((mog.loc < -10).sum(), 0)
            self.assertEqual((mog.loc > 10).sum(), 0)
            self.assertEqual((mog.scale_tril < -1).sum(), 0)
            self.assertEqual((mog.scale_tril > 1).sum(), 0)

    def test_MASS_var_dist_standard_basis_init(self):
        writer = DummyWriter()
        model_kwargs = dict(
            net_name='SmallMLP',
            net_kwargs=dict(
                out_dim=8,
                nonlinearity='elu',
                batch_norm=True,
                dropout=False,
                in_shape=(1, 28, 28),
            ),
            var_dist_init_strategy='standard_basis',
            beta=0.9,
            n_mixture_components=2,
            n_classes=10,
        )
        model = MASSCE(writer, **model_kwargs)
        # Make sure everything started as zeros
        for p in model.var_dist.parameters():
            self.assertEqual((p != 0).sum(), 0)
            self.assertTrue(p.requires_grad)

        # Make sure initialization worked
        model.initialize(None)
        for i, mog in enumerate(model.var_dist.q):
            self.assertEqual((mog.mixture_logits != 0).sum(), 0)
            if i < 8:
                self.assertEqual((mog.loc > 5).sum(), 1)
                self.assertEqual((mog.loc < -5).sum(), 0)
                self.assertTrue(mog.loc[0, i] == 10)
            else:
                self.assertEqual((mog.loc < -10).sum(), 0)
                self.assertEqual((mog.loc > 10).sum(), 0)
            self.assertEqual((mog.scale_tril != 0).sum(), 8 * 2)
            self.assertTrue(mog.scale_tril[0, 0, 0] == 1)

    def test_MASS_var_dist_MLE_from_training_data_init(self):
        """
        Make a dummy training dataset where half the inputs are all zeros and are class 1, and the other half are all
            ones and are class 2.  Then overwrite the model weights so it spits out normally-distributed outputs.
            The test makes sure the variational distribution of outputs the model learns is correct.
        """
        for out_dim in [8, 10, 12]:
            data = torch.zeros(256000, 1, 28, 28)
            data[:128000, :, :, :] = 1
            target = torch.zeros(256000, dtype=torch.int64)
            target[128000:] = 1
            writer = DummyWriter()
            model_kwargs = dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=out_dim,
                    nonlinearity='tanh',
                    batch_norm=False,
                    dropout=False,
                    in_shape=(1, 28, 28),
                ),
                var_dist_init_strategy='MLE_from_training_data',
                beta=0.9,
                n_mixture_components=1,
                n_classes=2,
            )
            model = MASSCE(writer, **model_kwargs)
            # overwrite fcout so the outputs are normally distributed
            model.net.fcout.forward = lambda x: torch.normal(torch.matmul(x, torch.ones(200, out_dim)))
            for name, p in model.net.named_parameters():
                if 'weight' in name:
                    p.data[:] = 1
                elif 'bias' in name:
                    p.data[:] = 0
            # Make sure everything started as zeros
            for p in model.var_dist.parameters():
                self.assertEqual((p != 0).sum(), 0)
                self.assertTrue(p.requires_grad)

            # Make sure initialization worked
            model.initialize([(data, target)])
            for i, mog in enumerate(model.var_dist.q):
                if i == 0:
                    mean = 200
                else:
                    mean = 0
                self.assertEqual(mog.mixture_logits, 0)
                np.testing.assert_array_almost_equal(mean * np.ones((1, out_dim)),
                                                     mog.loc.detach().numpy(),
                                                     decimal=2)
                np.testing.assert_array_almost_equal(np.eye(out_dim),
                                                     mog.scale_tril.squeeze(0).detach().numpy(),
                                                     decimal=2)

    def test_var_dist_SGD_update(self):
        """
        Make sure SGD is actually updating the parameters of the variational distribution according to the var_dist_lr
        """
        for var_dist_lr in [0, 1e-3]:
            data = torch.zeros(256, 1, 28, 28).uniform_(0, 1)
            target = torch.zeros(256, dtype=torch.int64).random_(0, 10)
            writer = DummyWriter()
            optimizer_class_name = 'SGD'
            optimizer_kwargs = dict(
                lr=1e-3,
                momentum=0.9,
                var_dist_optimizer_kwargs=dict(
                    lr=var_dist_lr
                )
            )
            model_kwargs = dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=10,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False,
                ),
                var_dist_init_strategy='standard_basis',
                beta=0.1,
                n_mixture_components=3,
            )
            model_kwargs['n_classes'] = 10
            model_kwargs['net_kwargs']['in_shape'] = (1, 28, 28)
            model = MASSCE(writer, **model_kwargs)
            model.initialize(None)

            orig_var_dist_params = copy.deepcopy(list(model.var_dist.named_parameters()))
            orig_var_dist_q_ids = [id(q_i) for q_i in model.var_dist.q]
            optimizer = model.get_optimizer(optimizer_class_name, optimizer_kwargs)

            model.train()
            optimizer.zero_grad()
            output = model.net.forward(data)
            loss = model.loss(data, output, target)

            # Make sure gradients start not initialized
            for nparam in model.var_dist.named_parameters():
                self.assertIsNone(nparam[1].grad)
            loss.backward()
            for nparam in model.var_dist.named_parameters():
                self.assertIsNotNone(nparam[1].grad)

            # Make sure none of the parameters changed values
            for orig_nparam, nparam in zip(orig_var_dist_params, model.var_dist.named_parameters()):
                self.assertEqual(orig_nparam[0], nparam[0])
                np.testing.assert_array_equal(orig_nparam[1].detach(), nparam[1].detach())
            self.assertEqual(orig_var_dist_q_ids, [id(q_i) for q_i in model.var_dist.q])

            optimizer.step()
            # Make sure var_dist parameters changed values
            for orig_nparam, nparam in zip(orig_var_dist_params, model.var_dist.named_parameters()):
                self.assertEqual(orig_nparam[0], nparam[0])
                if var_dist_lr == 0:
                    np.testing.assert_array_equal(orig_nparam[1].detach(), nparam[1].detach())
                elif var_dist_lr == 1e-3:
                    with self.assertRaises(AssertionError):
                        np.testing.assert_array_equal(orig_nparam[1].detach(), nparam[1].detach())
            self.assertEqual(orig_var_dist_q_ids, [id(q_i) for q_i in model.var_dist.q])

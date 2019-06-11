import unittest

import numpy as np
import sklearn
import torch
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hpnp
from torch.autograd import gradcheck
from torch.distributions import MultivariateNormal

from models.nets import SmallMLP
from models.utils import jacobian, MOG
from tests import DummyWriter


class TestMOG(unittest.TestCase):
    @given(data=hpnp.arrays(dtype=np.dtype('float32'),
                            shape=hpnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=20),
                            elements=st.floats(-10, 10)))
    def test_MOG_log_prob_matches_pytorch(self, data):
        data = torch.Tensor(data)
        n_mixture_components = 1
        rep_dim = data.shape[1]
        mog = MOG(n_mixture_components, rep_dim)
        mog.mixture_logits.data.uniform_(0.1, 1)
        mog.loc.data.uniform_(-10, 10)
        mog.scale_tril.data.uniform_(-1, 1)
        mog.scale_tril.data = torch.tril(mog.scale_tril.data[0] + torch.eye(data.shape[1])).unsqueeze(0)

        mvn = MultivariateNormal(loc=mog.loc[0], scale_tril=mog.scale_tril[0])
        mvn_log_prob = mvn.log_prob(data)
        mvn_log_prob[mvn_log_prob == -np.inf] = np.nan
        np.testing.assert_array_equal(mvn_log_prob, mog.log_prob(data, True))
        np.testing.assert_array_equal(mvn_log_prob, mog.log_prob(data, False))

    @given(data=hpnp.arrays(dtype=np.dtype('float64'),
                            shape=(20, 11),
                            elements=st.floats(-0.5, 0.5)))
    def test_MOG_log_prob_matches_sklearn(self, data):
        np.random.seed(123)
        fit_data = 0.5 - np.random.rand(20, 11).astype('float64')
        data = torch.from_numpy(data)
        n_mixture_components = 3
        rep_dim = data.shape[1]
        for cov_type in ['full', 'diag']:
            gm = sklearn.mixture.GaussianMixture(n_components=n_mixture_components,
                                                 covariance_type=cov_type)
            gm.fit(fit_data)
            mog = MOG(n_mixture_components, rep_dim)
            mog.mixture_logits.data = torch.log(torch.from_numpy(gm.weights_))
            mog.loc.data = torch.from_numpy(gm.means_)
            if cov_type == 'full':
                mog.scale_tril.data = torch.cholesky(torch.from_numpy(gm.covariances_))
            elif cov_type == 'diag':
                mog.scale_tril.data = torch.sqrt(torch.from_numpy(np.stack([np.diag(d) for d in gm.covariances_])))

            gm_log_prob = gm.score_samples(data)
            np.testing.assert_allclose(gm_log_prob, mog.log_prob(data, True), rtol=1e-9)
            np.testing.assert_allclose(gm_log_prob, mog.log_prob(data, False), rtol=1e-9)

    def test_MOG_detach(self):
        data = torch.Tensor(np.random.rand(20, 11)).requires_grad_()
        n_mixture_components = 2
        rep_dim = data.shape[1]
        mog = MOG(n_mixture_components, rep_dim)
        for p in mog.parameters():
            p.requires_grad = True
        mog.mixture_logits.data.uniform_(0.1, 1)
        mog.loc.data.uniform_(-10, 10)
        mog.scale_tril.data.uniform_(-1, 1)
        mog.scale_tril.data[0] = torch.tril(mog.scale_tril.data[0] + torch.eye(data.shape[1])).unsqueeze(0)
        mog.scale_tril.data[1] = torch.tril(mog.scale_tril.data[1] + torch.eye(data.shape[1])).unsqueeze(0)

        for p in mog.parameters():
            self.assertIsNone(p.grad, "Grads weren't zero before testing.")
        log_prob = mog.log_prob(data, detach=True)
        log_prob.sum().backward()
        for p in mog.parameters():
            self.assertIsNone(p.grad, "Grads didn't stay zero when detached.")
        log_prob = mog.log_prob(data, detach=False)
        log_prob.sum().backward()
        for p in mog.parameters():
            self.assertNotEqual((p.grad != 0).sum().item(), 0, "Grads weren't computed when not detached.")


class TestJacobianCorrectness(unittest.TestCase):
    @given(input=hpnp.arrays(dtype=np.dtype('float64'),
                             shape=hpnp.array_shapes(min_dims=2, max_dims=2, max_side=100),
                             elements=st.floats(-100, 100)))
    def test_jacobian_util_gradcheck_simple_net(self, input):
        """
        Finite differences gradient check on jacobian
        """
        input = torch.DoubleTensor(input).requires_grad_()
        net = SmallMLP(DummyWriter(), True, tuple(input.shape[1:]), 10, 'elu', False, False).to(torch.float64)
        net.eval()
        output = net(input)
        J = jacobian(input, output)

        class TestFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                return net(i)

            @staticmethod
            def backward(ctx, grad_output):
                go = grad_output.unsqueeze(1)
                return torch.bmm(go, J).squeeze(1)

        testfunction = TestFunction.apply

        self.assertTrue(gradcheck(testfunction, [input]))

    @given(input=hpnp.arrays(dtype=np.dtype('float64'),
                             shape=(6, 784),
                             elements=st.floats(-100, 100)))
    def test_jacobian_util_gradcheck_SmallMLP(self, input):
        """
        Finite differences gradient check on jacobian
        """
        input = torch.DoubleTensor(input).requires_grad_()
        net = SmallMLP(DummyWriter(), True, (784,), 10, 'elu', False, False).to(torch.float64)
        net.eval()
        output = net(input)
        J = jacobian(input, output)

        class TestFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                return net(i)

            @staticmethod
            def backward(ctx, grad_output):
                go = grad_output.unsqueeze(1)
                return torch.bmm(go, J).squeeze(1)

        testfunction = TestFunction.apply

        self.assertTrue(gradcheck(testfunction, [input]))


class TestGradOfJacobian(unittest.TestCase):
    @given(input=hpnp.arrays(dtype=np.dtype('float32'),
                             shape=(6, 784),
                             elements=st.floats(-100, 100)))
    def test_diffable_option(self, input):
        """
        Make sure the jacobian can only be used in a loss function if diffable=True
        """
        input = torch.tensor(input).requires_grad_()
        net = SmallMLP(DummyWriter(), True, (784,), 10, 'elu', False, False)
        net.eval()
        output = net(input)

        with self.assertRaises(RuntimeError):
            J = jacobian(input, output)
            loss = J.sum()
            loss.backward()

        J = jacobian(input, output, diffable=True)
        loss = J.sum()
        loss.backward()

    @given(input=hpnp.arrays(dtype=np.dtype('float32'),
                             shape=(6, 784),
                             elements=st.floats(-100, 100)))
    def test_gradients_computed_at_right_time(self, input):
        """
        Make sure the jacobian is having gradients computed only after calling backward on it
        """
        input = torch.tensor(input).requires_grad_()
        net = SmallMLP(DummyWriter(), True, (784,), 10, 'elu', False, False)
        net.eval()
        output = net(input)

        J = jacobian(input, output, diffable=True)
        loss = J.sum()
        self.assertIsNone(input.grad)
        self.assertTrue(all(p.grad == None for p in net.parameters()), 'parameter had grad before backward')
        loss.backward()
        self.assertIsNotNone(input.grad)
        self.assertTrue(all(list(p.grad is not None for p in net.parameters())[:-1]),
                        'parameters missing grads after backward')

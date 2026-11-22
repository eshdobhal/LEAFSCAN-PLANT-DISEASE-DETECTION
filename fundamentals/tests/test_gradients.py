"""Numerical gradient checks: every hand-written backward pass must match central differences."""
import numpy as np
import pytest

from fundamentals.mlp import MLP, BatchNorm, Dense, Activation, Dropout, cross_entropy
from fundamentals.tensors import numerical_gradient, relative_error

RNG = np.random.default_rng(0)
TOL = 1e-6


def check_layer(layer, x):
    """Use L = sum(out * R) for a fixed random R, so dL/dout = R."""
    r = RNG.normal(size=layer.forward(x).shape)
    layer.forward(x)
    dx = layer.backward(r)
    num_dx = numerical_gradient(lambda xx: float((layer.forward(xx) * r).sum()), x.copy())
    assert relative_error(dx, num_dx) < TOL
    for name, p in layer.params.items():
        layer.forward(x)
        layer.backward(r)
        analytic = layer.grads[name].copy()
        numeric = numerical_gradient(lambda pp: float((layer.forward(x) * r).sum()), p)
        assert relative_error(analytic, numeric) < TOL, name


def test_dense():
    check_layer(Dense(5, 4, "xavier", RNG), RNG.normal(size=(3, 5)))


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu"])
def test_activations(name):
    x = RNG.normal(size=(4, 6))
    x[np.abs(x) < 1e-3] = 0.1  # keep ReLU away from its kink, where the derivative is undefined
    check_layer(Activation(name), x)


def test_batchnorm():
    check_layer(BatchNorm(4), RNG.normal(size=(8, 4)))


def test_dropout_with_fixed_mask():
    layer = Dropout(0.5, np.random.default_rng(1))
    x = RNG.normal(size=(4, 5))
    layer.forward(x)
    mask = layer.mask.copy()
    r = RNG.normal(size=x.shape)
    assert np.allclose(layer.backward(r), r * mask)


def test_softmax_cross_entropy():
    logits, y = RNG.normal(size=(5, 4)), np.array([0, 3, 1, 1, 2])
    _, grad = cross_entropy(logits, y)
    numeric = numerical_gradient(lambda z: cross_entropy(z, y)[0], logits.copy())
    assert relative_error(grad, numeric) < TOL


@pytest.mark.parametrize("batchnorm", [False, True])
def test_whole_network_with_l2(batchnorm):
    model = MLP([6, 5, 4, 3], activation="tanh", init="xavier", batchnorm=batchnorm, seed=2)
    x, y, l2 = RNG.normal(size=(7, 6)), np.array([0, 1, 2, 0, 1, 2, 0]), 0.01
    model.loss_and_backward(x, y, l2)
    for params, grads in model.parameters():
        for name, p in params.items():
            analytic = grads[name].copy()
            numeric = numerical_gradient(lambda pp: _loss(model, x, y, l2), p)
            if np.abs(analytic).max() < 1e-12:
                # A Dense bias feeding BatchNorm is cancelled by the mean subtraction, so its true gradient
                # is exactly 0; relative error is meaningless there, so compare absolutely.
                assert np.abs(numeric).max() < 1e-8, name
            else:
                assert relative_error(analytic, numeric) < 1e-5, name


def _loss(model, x, y, l2):
    loss, _ = cross_entropy(model.forward(x, train=True), y)
    return loss + 0.5 * l2 * sum(float((p["W"] ** 2).sum()) for p, _ in model.parameters() if "W" in p)

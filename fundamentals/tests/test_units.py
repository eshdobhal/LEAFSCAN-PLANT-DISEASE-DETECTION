"""Unit tests for tensors, neurons, optimizers and the classical learning rules."""
import numpy as np
import pytest

from fundamentals import hebbian, tensors
from fundamentals.neurons import McCullochPitts, SigmoidNeuron, sigmoid, step, tanh
from fundamentals.optim import OPTIMIZERS

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ tensors
@pytest.mark.parametrize("a,b", [((3, 1), (1, 4)), ((2, 3, 4), (4,)), ((5, 1, 3), (1, 6, 1)), ((1,), (7,))])
def test_broadcast_rule_matches_numpy(a, b):
    assert tensors.broadcast_shape(a, b) == np.broadcast_shapes(a, b)


def test_broadcast_rule_rejects_mismatch():
    with pytest.raises(ValueError):
        tensors.broadcast_shape((3, 2), (4,))


def test_matmul_is_inner_products():
    a, b = RNG.normal(size=(3, 4)), RNG.normal(size=(4, 2))
    assert np.allclose(tensors.matmul_as_inner_products(a, b), a @ b)


def test_reshape_and_normalise():
    imgs = RNG.integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
    batch = tensors.image_to_batch(imgs)
    assert batch.shape == (2, 3, 8, 8) and tensors.flatten(batch).shape == (2, 192)
    out = tensors.normalise_channels(batch, batch.mean((0, 2, 3)), batch.std((0, 2, 3)))
    assert np.allclose(out.mean((0, 2, 3)), 0, atol=1e-5)


def test_svd_error_shrinks_and_full_rank_is_exact():
    m = RNG.normal(size=(20, 12))
    errors = [tensors.svd_compress(m, k)[2] for k in (1, 4, 8, 12)]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-10


def test_numerical_gradient_of_a_known_function():
    x = RNG.normal(size=(3, 2))
    assert tensors.relative_error(2 * x, tensors.numerical_gradient(lambda z: float((z ** 2).sum()), x.copy())) < 1e-8


# ------------------------------------------------------------------ neurons
def test_mcculloch_pitts_gates():
    assert [o for _, o in McCullochPitts.truth_table(McCullochPitts.AND(), 2)] == [0, 0, 0, 1]
    assert [o for _, o in McCullochPitts.truth_table(McCullochPitts.OR(), 2)] == [0, 1, 1, 1]
    assert [o for _, o in McCullochPitts.truth_table(McCullochPitts.NOT(), 1)] == [1, 0]


def test_sigmoid_neuron_learns_and():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    neuron = SigmoidNeuron(2)
    history = neuron.fit(x, np.array([0, 0, 0, 1.0]), lr=2.0, epochs=2000)
    assert history[-1] < history[0] and np.array_equal(neuron.forward(x) > 0.5, [False, False, False, True])


def test_activation_ranges():
    z = np.linspace(-10, 10, 101)
    assert set(np.unique(step(z))) <= {0.0, 1.0}
    assert sigmoid(z).min() > 0 and sigmoid(z).max() < 1 and np.isclose(sigmoid(0), 0.5)
    assert tanh(z).min() > -1 and tanh(z).max() < 1 and np.isclose(tanh(0), 0)
    assert np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))).all()  # no overflow


# ------------------------------------------------------------------ optimizers
@pytest.mark.parametrize("name", ["sgd", "momentum", "nesterov"])
def test_optimizers_minimise_a_quadratic(name):
    """f(w) = 0.5 * w^T A w with an ill-conditioned A; the minimum is w = 0."""
    a = np.diag([1.0, 25.0])
    params = {"w": np.array([4.0, 1.0])}
    grads = {"w": np.zeros(2)}
    opt = OPTIMIZERS[name](lr=0.03)
    for _ in range(200):
        grads["w"] = a @ params["w"]
        opt.step([(params, grads)])
    assert np.linalg.norm(params["w"]) < 1e-2


def test_momentum_beats_plain_sgd_on_ill_conditioned_problem():
    a = np.diag([1.0, 25.0])

    def run(opt, steps=60):
        params, grads = {"w": np.array([4.0, 1.0])}, {"w": np.zeros(2)}
        for _ in range(steps):
            grads["w"] = a @ params["w"]
            opt.step([(params, grads)])
        return float(0.5 * params["w"] @ a @ params["w"])

    assert run(OPTIMIZERS["momentum"](lr=0.03, mu=0.8)) < run(OPTIMIZERS["sgd"](lr=0.03))


# ------------------------------------------------------------------ learning rules
def _correlated_data(n=400):
    direction = np.array([3.0, 1.0, 0.5]) / np.linalg.norm([3.0, 1.0, 0.5])
    return RNG.normal(size=(n, 1)) * 3 * direction + RNG.normal(size=(n, 3)) * 0.3


def test_oja_finds_first_principal_component_and_hebb_diverges():
    x = _correlated_data()
    x -= x.mean(0)
    w, norms = hebbian.oja(x)
    assert abs(tensors.cosine_similarity(w, hebbian.first_principal_component(x))) > 0.99
    assert abs(norms[-1] - 1) < 0.05
    _, hebb_norms = hebbian.hebbian(x, lr=0.01, epochs=1)
    assert hebb_norms[-1] > 100 * hebb_norms[0]


def test_competitive_learning_reduces_distortion():
    x = np.vstack([RNG.normal(c, 0.2, size=(50, 2)) for c in ((0, 0), (3, 3), (0, 3))])
    _, assign, distortion = hebbian.competitive(x, k=3)
    assert distortion[-1] < distortion[0] + 1e-9 and len(set(assign)) == 3


def test_error_correction_rules():
    x = np.vstack([RNG.normal(-2, 0.5, (40, 2)), RNG.normal(2, 0.5, (40, 2))])
    t = np.array([0] * 40 + [1] * 40, float)
    _, mistakes = hebbian.perceptron(x, t)
    assert mistakes[-1] == 0  # linearly separable: the perceptron converges
    _, mse = hebbian.delta_rule(x, t)
    assert mse[-1] < mse[0]


def test_rbm_learns_to_reconstruct():
    patterns = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], float)
    v = patterns[RNG.integers(0, 2, 300)]
    rbm = hebbian.RBM(6, 3)
    history = rbm.fit(v, epochs=60, lr=0.1)
    assert history[-1]["reconstruction_error"] < history[0]["reconstruction_error"]
    assert rbm.free_energy(patterns).max() < rbm.free_energy(np.array([[1, 0, 1, 0, 1, 0.0]]))[0]

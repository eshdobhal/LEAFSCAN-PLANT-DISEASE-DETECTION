"""A multilayer feed-forward network with hand-written backpropagation (brief item 4), in NumPy.

Layers: Dense, Activation, BatchNorm (normalisation), Dropout (regularisation).
Loss: softmax cross-entropy, plus optional L2 weight decay.
Every layer's backward() is verified against numerical gradients in tests/test_gradients.py.

Parameters vs hyperparameters:
- parameters are learned by backpropagation (Dense weights/biases, BatchNorm gamma/beta);
- hyperparameters are chosen by us (layer sizes, activation, initialisation, learning rate, batch size,
  L2, dropout, epochs, optimizer). RunLog records both, so every experiment is traceable.
"""
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .neurons import ACTIVATIONS


# ------------------------------------------------------------------ initialisation
def init_weights(n_in, n_out, scheme, rng):
    if scheme == "zeros":    # every neuron identical: symmetry is never broken
        return np.zeros((n_in, n_out))
    if scheme == "normal":   # small fixed scale: fine for shallow nets, vanishes in deep ones
        return rng.normal(0, 0.01, (n_in, n_out))
    if scheme == "xavier":   # Glorot: keeps variance for tanh/sigmoid
        return rng.normal(0, np.sqrt(2 / (n_in + n_out)), (n_in, n_out))
    if scheme == "he":       # He: keeps variance for ReLU (half the units are off)
        return rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
    raise ValueError(scheme)


# ------------------------------------------------------------------ layers
class Dense:
    def __init__(self, n_in, n_out, init="he", rng=None):
        rng = rng or np.random.default_rng(0)
        self.params = {"W": init_weights(n_in, n_out, init, rng), "b": np.zeros(n_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        self.grads["W"] = self.x.T @ grad_out          # dL/dW = x^T . dL/dy
        self.grads["b"] = grad_out.sum(axis=0)         # dL/db = sum over the batch
        return grad_out @ self.params["W"].T           # dL/dx, passed to the previous layer


class Activation:
    def __init__(self, name):
        self.name = name
        self.f, self.df = ACTIVATIONS[name]
        self.params, self.grads = {}, {}

    def forward(self, x, train=True):
        self.z = x
        return self.f(x)

    def backward(self, grad_out):
        return grad_out * self.df(self.z)  # chain rule, element by element


class BatchNorm:
    """Normalise each feature over the batch, then scale (gamma) and shift (beta). Running statistics
    are used at evaluation time."""

    def __init__(self, n, momentum=0.9, eps=1e-5):
        self.params = {"gamma": np.ones(n), "beta": np.zeros(n)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.running_mean, self.running_var = np.zeros(n), np.ones(n)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, train=True):
        if train:
            mean, var = x.mean(0), x.var(0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean, var = self.running_mean, self.running_var
        self.std = np.sqrt(var + self.eps)
        self.xhat = (x - mean) / self.std
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, grad_out):
        n = grad_out.shape[0]
        self.grads["gamma"] = (grad_out * self.xhat).sum(0)
        self.grads["beta"] = grad_out.sum(0)
        dxhat = grad_out * self.params["gamma"]
        return (n * dxhat - dxhat.sum(0) - self.xhat * (dxhat * self.xhat).sum(0)) / (n * self.std)


class Dropout:
    """Randomly zero a fraction p of activations during training (inverted dropout: scale by 1/(1-p))."""

    def __init__(self, p, rng=None):
        self.p, self.rng = p, rng or np.random.default_rng(0)
        self.params, self.grads = {}, {}

    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = np.ones_like(x)
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


# ------------------------------------------------------------------ loss
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)  # subtracting the max avoids overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    """Mean softmax cross-entropy, and its gradient w.r.t. the logits: (softmax - one_hot) / n."""
    p = softmax(logits)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    grad = p.copy()
    grad[np.arange(n), y] -= 1
    return float(loss), grad / n


# ------------------------------------------------------------------ the network
class MLP:
    def __init__(self, sizes, activation="relu", init="he", batchnorm=False, dropout=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Dense(n_in, n_out, init, rng))
            if i < len(sizes) - 2:  # hidden layers only
                # With BatchNorm the Dense bias is redundant: subtracting the batch mean cancels it, so
                # its gradient is exactly 0 (the gradient test checks this). BatchNorm's beta shifts instead.
                if batchnorm:
                    self.layers.append(BatchNorm(n_out))
                self.layers.append(Activation(activation))
                if dropout:
                    self.layers.append(Dropout(dropout, rng))

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """(params dict, grads dict) pairs of every layer that learns: what the optimizer updates."""
        return [(l.params, l.grads) for l in self.layers if l.params]

    def n_parameters(self):
        return int(sum(v.size for p, _ in self.parameters() for v in p.values()))

    def loss_and_backward(self, x, y, l2=0.0):
        logits = self.forward(x, train=True)
        loss, grad = cross_entropy(logits, y)
        self.backward(grad)
        if l2:  # weight decay on weights (not biases): loss += l2/2 |W|^2, dW += l2 W
            for p, g in self.parameters():
                if "W" in p:
                    loss += 0.5 * l2 * float((p["W"] ** 2).sum())
                    g["W"] += l2 * p["W"]
        return loss

    def predict_proba(self, x):
        return softmax(self.forward(x, train=False))

    def accuracy(self, x, y):
        return float((self.predict_proba(x).argmax(1) == y).mean())


# ------------------------------------------------------------------ experiment tracking
@dataclass
class RunLog:
    """Everything needed to reproduce and compare a run: hyperparameters in, metrics per epoch out."""
    name: str
    hyperparameters: dict
    n_parameters: int = 0
    history: list = field(default_factory=list)
    seconds: float = 0.0

    def log(self, **metrics):
        self.history.append({k: round(float(v), 5) for k, v in metrics.items()})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=1)


def train(model, optimizer, data, epochs, batch_size, l2=0.0, augment=None, seed=0, name="run", hyper=None):
    """Mini-batch training loop. batch_size=None means full-batch gradient descent."""
    x_tr, y_tr, x_va, y_va = data
    rng = np.random.default_rng(seed)
    log = RunLog(name, hyper or {}, model.n_parameters())
    start = time.time()
    n = len(y_tr)
    bs = batch_size or n
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            xb = augment(x_tr[idx], rng) if augment else x_tr[idx]
            losses.append(model.loss_and_backward(xb, y_tr[idx], l2))
            optimizer.step(model.parameters())
        log.log(epoch=epoch + 1, train_loss=np.mean(losses), train_acc=model.accuracy(x_tr, y_tr),
                val_acc=model.accuracy(x_va, y_va))
    log.seconds = time.time() - start
    return log

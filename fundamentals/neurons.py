"""Single neurons and activation functions (brief items 4 and 9: step vs sigmoid vs tanh).

McCulloch-Pitts (1943): binary inputs, fixed weights, a hard threshold. No learning.
Sigmoid neuron: real inputs, a smooth activation, and weights learned by gradient descent
(this is logistic regression).
"""
import numpy as np


# ------------------------------------------------------------------ activations and derivatives
def step(z):
    return (np.asarray(z) >= 0).astype(np.float64)


def step_grad(z):
    return np.zeros_like(np.asarray(z, dtype=np.float64))  # zero almost everywhere: nothing to learn from


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))  # overflow-safe


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)  # at most 0.25: gradients shrink layer after layer (vanishing gradients)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2  # at most 1 and zero-centred: usually trains faster than sigmoid


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (np.asarray(z) > 0).astype(np.float64)


ACTIVATIONS = {"step": (step, step_grad), "sigmoid": (sigmoid, sigmoid_grad),
               "tanh": (tanh, tanh_grad), "relu": (relu, relu_grad)}


# ------------------------------------------------------------------ McCulloch-Pitts neuron
class McCullochPitts:
    """Fires (1) when the weighted sum of binary inputs reaches the threshold, else 0."""

    def __init__(self, weights, threshold):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.threshold = float(threshold)

    def __call__(self, x):
        return int(np.dot(self.weights, x) >= self.threshold)

    # Classic gates, hand-wired (not learned).
    @classmethod
    def AND(cls, n=2):
        return cls(np.ones(n), threshold=n)

    @classmethod
    def OR(cls, n=2):
        return cls(np.ones(n), threshold=1)

    @classmethod
    def NOT(cls):
        return cls([-1], threshold=0)

    @staticmethod
    def truth_table(neuron, n_inputs):
        rows = []
        for i in range(2 ** n_inputs):
            x = [(i >> (n_inputs - 1 - k)) & 1 for k in range(n_inputs)]
            rows.append((tuple(x), neuron(x)))
        return rows


# ------------------------------------------------------------------ sigmoid neuron
class SigmoidNeuron:
    """y = sigmoid(w.x + b), trained with gradient descent on binary cross-entropy."""

    def __init__(self, n_inputs, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_inputs)
        self.b = 0.0

    def forward(self, x):
        return sigmoid(x @ self.w + self.b)

    def loss(self, x, y):
        p = np.clip(self.forward(x), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def gradients(self, x, y):
        """d(BCE)/dz = p - y for a sigmoid output, so dW = x^T (p - y) / n."""
        err = self.forward(x) - y
        return x.T @ err / len(y), float(err.mean())

    def fit(self, x, y, lr=0.5, epochs=200):
        history = []
        for _ in range(epochs):
            dw, db = self.gradients(x, y)
            self.w -= lr * dw
            self.b -= lr * db
            history.append(self.loss(x, y))
        return history

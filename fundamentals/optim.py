"""Optimizers (brief item 5). Each one updates params[k] in place from grads[k].

Batch GD vs SGD is a choice of batch size (see mlp.train); the update rules below differ in how they
use the gradient:
- SGD:       w -= lr * g
- Momentum:  v = mu * v - lr * g;  w += v                (a velocity that averages past gradients)
- Nesterov:  look ahead along the velocity first, then correct; the standard reformulation is
             v_new = mu * v - lr * g;  w += -mu * v + (1 + mu) * v_new
"""
import numpy as np


class SGD:
    """Plain gradient descent. With full batches this is batch GD; with mini-batches, SGD."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, parameters):
        for params, grads in parameters:
            for k in params:
                params[k] -= self.lr * grads[k]


class Momentum:
    def __init__(self, lr=0.1, mu=0.9):
        self.lr, self.mu, self.v = lr, mu, {}

    def step(self, parameters):
        for i, (params, grads) in enumerate(parameters):
            for k in params:
                v = self.v.setdefault((i, k), np.zeros_like(params[k]))
                v *= self.mu
                v -= self.lr * grads[k]
                params[k] += v


class Nesterov:
    def __init__(self, lr=0.1, mu=0.9):
        self.lr, self.mu, self.v = lr, mu, {}

    def step(self, parameters):
        for i, (params, grads) in enumerate(parameters):
            for k in params:
                v_prev = self.v.get((i, k), np.zeros_like(params[k]))
                v_new = self.mu * v_prev - self.lr * grads[k]
                params[k] += -self.mu * v_prev + (1 + self.mu) * v_new
                self.v[(i, k)] = v_new


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "nesterov": Nesterov}

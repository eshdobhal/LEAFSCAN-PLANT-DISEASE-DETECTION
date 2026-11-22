"""Classical learning rules (brief item 6), each in a few lines so they can be compared directly.

- Hebbian:            dw = lr * y * x                     ("cells that fire together wire together")
                      Unstable: |w| grows without limit.
- Oja (modified Hebb): dw = lr * y * (x - y * w)          The decay term keeps |w| = 1, and w converges to
                      the first principal component of the data (checked against SVD in the tests).
- Competitive:        only the winning prototype moves: dw_win = lr * (x - w_win)   (online k-means)
- Error-correction:   perceptron dw = lr * (t - step(w.x)) * x  and delta/LMS dw = lr * (t - w.x) * x
                      (learning is driven by the *error*, unlike the unsupervised rules above).
- Boltzmann (RBM):    an energy-based model E(v, h) = -a.v - b.h - v^T W h trained by contrastive
                      divergence (CD-1); learning lowers the energy of data and raises it elsewhere.
"""
import numpy as np

from .neurons import sigmoid, step


# ------------------------------------------------------------------ unsupervised Hebbian rules
def hebbian(x, lr=0.01, epochs=5, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, x.shape[1])
    norms = []
    for _ in range(epochs):
        for xi in x[rng.permutation(len(x))]:
            y = w @ xi
            w += lr * y * xi
            norms.append(np.linalg.norm(w))
    return w, np.array(norms)


def oja(x, lr=0.005, epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, x.shape[1])
    w /= np.linalg.norm(w)
    norms = []
    for _ in range(epochs):
        for xi in x[rng.permutation(len(x))]:
            y = w @ xi
            w += lr * y * (xi - y * w)
            norms.append(np.linalg.norm(w))
    return w, np.array(norms)


def first_principal_component(x):
    """Reference answer for Oja's rule: the top right-singular vector of the centred data."""
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    return vt[0]


def competitive(x, k=4, lr=0.05, epochs=20, seed=0):
    """Winner-take-all: prototypes move towards the inputs they win; returns prototypes, assignments and
    the mean squared distance to the nearest prototype, from before training (index 0) to the end.

    Prototypes start far apart (farthest-point seeding). With purely random starts two prototypes can
    share one cluster while another prototype never wins: the 'dead unit' problem of competitive learning.
    """
    rng = np.random.default_rng(seed)
    w = [x[rng.integers(len(x))]]
    while len(w) < k:
        d = ((x[:, None, :] - np.array(w)[None]) ** 2).sum(2).min(1)
        w.append(x[np.argmax(d)])
    w = np.array(w, dtype=np.float64)
    distortion = [float(((x[:, None, :] - w[None]) ** 2).sum(2).min(1).mean())]
    for _ in range(epochs):
        for xi in x[rng.permutation(len(x))]:
            win = np.argmin(((w - xi) ** 2).sum(1))
            w[win] += lr * (xi - w[win])
        d = ((x[:, None, :] - w[None]) ** 2).sum(2)
        distortion.append(float(d.min(1).mean()))
    return w, np.argmin(((x[:, None, :] - w[None]) ** 2).sum(2), axis=1), distortion


# ------------------------------------------------------------------ error-correction rules
def perceptron(x, t, lr=0.1, epochs=20, seed=0):
    """t in {0, 1}. Updates only on mistakes. Converges if the classes are linearly separable."""
    rng = np.random.default_rng(seed)
    xb = np.hstack([x, np.ones((len(x), 1))])  # bias as an extra input fixed at 1
    w = np.zeros(xb.shape[1])
    errors = []
    for _ in range(epochs):
        mistakes = 0
        for i in rng.permutation(len(x)):
            err = t[i] - step(w @ xb[i])
            if err:
                w += lr * err * xb[i]
                mistakes += 1
        errors.append(mistakes)
    return w, errors


def delta_rule(x, t, lr=0.01, epochs=20, seed=0):
    """Least-mean-squares (Widrow-Hoff): the error is measured before the threshold, so every example
    contributes a correction proportional to how wrong it is."""
    rng = np.random.default_rng(seed)
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    mse = []
    for _ in range(epochs):
        for i in rng.permutation(len(x)):
            w += lr * (t[i] - w @ xb[i]) * xb[i]
        mse.append(float(((xb @ w - t) ** 2).mean()))
    return w, mse


# ------------------------------------------------------------------ Boltzmann learning (RBM)
class RBM:
    """Restricted Boltzmann Machine with binary visible and hidden units."""

    def __init__(self, n_visible, n_hidden, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)  # visible biases
        self.b = np.zeros(n_hidden)   # hidden biases

    def energy(self, v, h):
        return -(v @ self.a) - (h @ self.b) - np.einsum("ni,ij,nj->n", v, self.W, h)

    def free_energy(self, v):
        """F(v) = -a.v - sum_j log(1 + exp(b_j + v.W_j)); lower = the model finds v more probable."""
        return -(v @ self.a) - np.logaddexp(0, v @ self.W + self.b).sum(1)

    def sample_h(self, v):
        p = sigmoid(v @ self.W + self.b)
        return p, (self.rng.random(p.shape) < p).astype(float)

    def sample_v(self, h):
        p = sigmoid(h @ self.W.T + self.a)
        return p, (self.rng.random(p.shape) < p).astype(float)

    def cd1(self, v0, lr=0.05):
        """Contrastive divergence: positive phase on data, negative phase after one Gibbs step."""
        ph0, h0 = self.sample_h(v0)
        pv1, _ = self.sample_v(h0)
        ph1, _ = self.sample_h(pv1)
        n = len(v0)
        self.W += lr * (v0.T @ ph0 - pv1.T @ ph1) / n
        self.a += lr * (v0 - pv1).mean(0)
        self.b += lr * (ph0 - ph1).mean(0)
        return float(((v0 - pv1) ** 2).mean())

    def fit(self, v, epochs=30, batch=32, lr=0.05):
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(v))
            errs = [self.cd1(v[order[i:i + batch]], lr) for i in range(0, len(v), batch)]
            history.append({"reconstruction_error": float(np.mean(errs)),
                            "mean_free_energy": float(self.free_energy(v).mean())})
        return history

    def reconstruct(self, v):
        ph, _ = self.sample_h(v)
        return sigmoid(ph @ self.W.T + self.a)

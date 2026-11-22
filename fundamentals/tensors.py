"""Tensor sandbox: the maths every neural network is built from, in plain NumPy.

Each function returns numbers that the figures in experiments/sandbox.py draw, so the computation can be
seen as well as read. Brief item 3: tensors, broadcasting, inner products, reshaping, SVD and numerical
gradient checking.
"""
import numpy as np


# ------------------------------------------------------------------ tensors and reshaping
def describe(t: np.ndarray) -> dict:
    """Rank (number of axes), shape, size and strides: what a tensor *is* in memory."""
    return {"rank": t.ndim, "shape": t.shape, "size": t.size, "strides_bytes": t.strides, "dtype": str(t.dtype)}


def image_to_batch(images: np.ndarray) -> np.ndarray:
    """(N, H, W, C) uint8 photos -> (N, C, H, W) float32 in [0, 1]: the layout PyTorch models expect."""
    return images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0


def flatten(batch: np.ndarray) -> np.ndarray:
    """(N, C, H, W) -> (N, C*H*W): the reshape that feeds an image into a fully-connected layer.
    Reshape never moves data; it only changes how the same numbers are indexed."""
    return batch.reshape(batch.shape[0], -1)


# ------------------------------------------------------------------ broadcasting
def normalise_channels(batch: np.ndarray, mean, std) -> np.ndarray:
    """ImageNet-style normalisation. mean/std have shape (3,); reshaped to (1, 3, 1, 1) they broadcast
    against (N, 3, H, W): NumPy stretches every size-1 axis to match, without copying memory."""
    mean = np.asarray(mean, dtype=batch.dtype).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=batch.dtype).reshape(1, -1, 1, 1)
    return (batch - mean) / std


def broadcast_shape(a_shape, b_shape):
    """The broadcasting rule written out: align shapes from the right; sizes must match or one be 1."""
    out = []
    for a, b in zip(reversed((1,) * (len(b_shape) - len(a_shape)) + tuple(a_shape)),
                    reversed((1,) * (len(a_shape) - len(b_shape)) + tuple(b_shape))):
        if a != b and 1 not in (a, b):
            raise ValueError(f"shapes {a_shape} and {b_shape} cannot broadcast ({a} vs {b})")
        out.append(max(a, b))
    return tuple(reversed(out))


# ------------------------------------------------------------------ inner products
def inner(u: np.ndarray, v: np.ndarray) -> float:
    """Inner (dot) product as an explicit sum: sum_i u_i v_i. A neuron's pre-activation is exactly this."""
    return float(sum(a * b for a, b in zip(u.ravel(), v.ravel())))


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """<u, v> / (|u| |v|): 1 = same direction, 0 = orthogonal, -1 = opposite."""
    return inner(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)


def matmul_as_inner_products(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C[i, j] = <row i of A, column j of B>: a dense layer is a batch of inner products."""
    out = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            out[i, j] = inner(a[i], b[:, j])
    return out


# ------------------------------------------------------------------ SVD
def svd_compress(matrix: np.ndarray, rank: int):
    """Best rank-k approximation (Eckart-Young): keep the k largest singular values.
    Returns the approximation, the singular values, and the relative Frobenius error."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
    error = np.linalg.norm(matrix - approx) / np.linalg.norm(matrix)
    return approx, s, float(error)


def svd_energy(singular_values: np.ndarray) -> np.ndarray:
    """Fraction of the matrix's 'energy' (sum of squared singular values) kept by the first k values."""
    energy = singular_values ** 2
    return np.cumsum(energy) / energy.sum()


# ------------------------------------------------------------------ numerical gradient checking
def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central differences: df/dx_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps). Slow but trustworthy."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        plus = f(x)
        x[i] = old - eps
        minus = f(x)
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """max |a - n| / max(|a| + |n|): below ~1e-6 means the hand-written backward pass is right."""
    return float(np.max(np.abs(analytic - numeric)) / max(1e-12, np.max(np.abs(analytic) + np.abs(numeric))))

"""Brief item 3 (and 9: step vs sigmoid vs tanh): visual outputs that explain the computations.

    python -m fundamentals.experiments.sandbox

Writes fundamentals/results/sandbox_*.png and sandbox.json.
"""
import json

import numpy as np

from fundamentals import data, tensors
from fundamentals.mlp import MLP, BatchNorm, Dense, Activation, cross_entropy
from fundamentals.neurons import ACTIVATIONS, McCullochPitts
from fundamentals.plotting import RESULTS, SERIES, plt, save


def main():
    leaves = data.load()
    x_tr, y_tr = leaves["train"]
    out = {}

    # 1. Tensors and reshaping: one photo through the layouts a network sees.
    batch = tensors.image_to_batch(x_tr[:16])
    out["tensor_layouts"] = {
        "photos (N,H,W,C) uint8": tensors.describe(x_tr[:16]),
        "batch (N,C,H,W) float32": tensors.describe(batch),
        "flattened (N, C*H*W)": tensors.describe(tensors.flatten(batch)),
    }

    # 2. Broadcasting: per-channel normalisation (1,3,1,1) against (N,3,H,W).
    mean, std = batch.mean((0, 2, 3)), batch.std((0, 2, 3))
    norm = tensors.normalise_channels(batch, mean, std)
    out["broadcasting"] = {"shapes": [list(batch.shape), [1, 3, 1, 1],
                                      list(tensors.broadcast_shape(batch.shape, (1, 3, 1, 1)))],
                           "channel_mean_before": mean.round(4).tolist(),
                           "channel_mean_after": norm.mean((0, 2, 3)).round(6).tolist(),
                           "channel_std_after": norm.std((0, 2, 3)).round(4).tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for c, colour in enumerate(("#d03b3b", "#0ca30c", "#2a78d6")):
        axes[0].hist(batch[:, c].ravel(), bins=40, alpha=0.55, color=colour, label="RGB"[c])
        axes[1].hist(norm[:, c].ravel(), bins=40, alpha=0.55, color=colour, label="RGB"[c])
    axes[0].set_title("Pixel values per channel, before")
    axes[1].set_title("After (x - mean) / std, broadcast (1,3,1,1)")
    axes[0].legend()
    save(fig, "sandbox_broadcasting")

    # 3. Inner products: cosine similarity between the average leaf of each class.
    feats = tensors.flatten(tensors.image_to_batch(x_tr)).astype(np.float64)
    feats -= feats.mean(0)
    means = np.stack([feats[y_tr == c].mean(0) for c in range(len(data.CLASSES))])
    sim = np.array([[tensors.cosine_similarity(a, b) for b in means] for a in means])
    out["cosine_similarity_of_class_means"] = sim.round(3).tolist()
    names = [c.split("___")[0][:6] + " " + c.split("___")[1].replace("_", " ")[:12] for c in data.CLASSES]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=50, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    ax.set_title("Inner products: cosine similarity of class-mean leaves")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "sandbox_inner_products")

    # 4. SVD: low-rank approximations of one leaf (each colour channel is a 24x24 matrix).
    leaf = x_tr[0].astype(np.float64) / 255
    ranks = [1, 3, 6, 12, 24]
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(12, 2.6))
    axes[0].imshow(leaf)
    axes[0].set_title("original")
    svd = {}
    for ax, k in zip(axes[1:], ranks):
        channels, errs = [], []
        for c in range(3):
            approx, s, err = tensors.svd_compress(leaf[:, :, c], k)
            channels.append(approx)
            errs.append(err)
        ax.imshow(np.clip(np.stack(channels, 2), 0, 1))
        ax.set_title(f"rank {k}\nerror {np.mean(errs):.1%}")
        svd[k] = round(float(np.mean(errs)), 4)
    for ax in axes:
        ax.axis("off")
    save(fig, "sandbox_svd_leaf")
    # SVD of the whole training matrix: how many directions hold the variance of 2,000 leaves.
    _, s_all, _ = np.linalg.svd(feats, full_matrices=False)
    energy = tensors.svd_energy(s_all)
    out["svd"] = {"leaf_relative_error_by_rank": svd,
                  "dataset_components_for_90pct_energy": int(np.searchsorted(energy, 0.90) + 1),
                  "dataset_components_for_99pct_energy": int(np.searchsorted(energy, 0.99) + 1),
                  "dataset_dimensions": int(feats.shape[1])}
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(1, len(energy) + 1), energy * 100, color=SERIES[0], lw=2)
    ax.axhline(90, color="#898781", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("number of singular vectors")
    ax.set_ylabel("energy kept (%)")
    ax.set_title(f"SVD of {feats.shape[0]} leaves x {feats.shape[1]} pixels")
    save(fig, "sandbox_svd_energy")

    # 5. Numerical gradient checking of every layer type.
    rng = np.random.default_rng(0)
    checks = {}
    for name, layer, x in [("Dense", Dense(6, 4, "xavier", rng), rng.normal(size=(5, 6))),
                           ("Sigmoid", Activation("sigmoid"), rng.normal(size=(5, 6))),
                           ("Tanh", Activation("tanh"), rng.normal(size=(5, 6))),
                           ("ReLU", Activation("relu"), rng.normal(size=(5, 6)) + 0.05),
                           ("BatchNorm", BatchNorm(6), rng.normal(size=(8, 6)))]:
        r = rng.normal(size=layer.forward(x).shape)
        layer.forward(x)
        analytic = layer.backward(r)
        numeric = tensors.numerical_gradient(lambda xx: float((layer.forward(xx) * r).sum()), x.copy())
        checks[name] = tensors.relative_error(analytic, numeric)
    logits, y = rng.normal(size=(6, 4)), rng.integers(0, 4, 6)
    checks["Softmax + cross-entropy"] = tensors.relative_error(
        cross_entropy(logits, y)[1], tensors.numerical_gradient(lambda z: cross_entropy(z, y)[0], logits.copy()))
    out["gradient_check_relative_error"] = {k: float(f"{v:.2e}") for k, v in checks.items()}
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(list(checks), list(checks.values()), color=SERIES[0], height=0.55, zorder=3)
    ax.axvline(1e-6, color="#d03b3b", ls="--", lw=1)
    ax.text(1.2e-6, -0.6, "pass threshold 1e-6", color="#d03b3b", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("relative error, analytic vs numerical gradient")
    ax.set_title("Numerical gradient check of each hand-written backward pass")
    ax.grid(axis="y", visible=False)
    save(fig, "sandbox_gradient_check")

    # 6. Activations and their derivatives (step vs sigmoid vs tanh vs ReLU).
    z = np.linspace(-5, 5, 401)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for (name, (f, df)), colour in zip(ACTIVATIONS.items(), SERIES):
        axes[0].plot(z, f(z), color=colour, lw=2, label=name)
        axes[1].plot(z, df(z), color=colour, lw=2, label=name)
    axes[0].set_title("Activation")
    axes[1].set_title("Derivative (what backpropagation multiplies by)")
    axes[1].legend()
    save(fig, "sandbox_activations")
    out["activation_max_derivative"] = {n: float(df(z).max()) for n, (_, df) in ACTIVATIONS.items()}

    # 7. McCulloch-Pitts truth tables.
    out["mcculloch_pitts"] = {g: [[list(x), o] for x, o in McCullochPitts.truth_table(n, k)]
                              for g, n, k in (("AND", McCullochPitts.AND(), 2), ("OR", McCullochPitts.OR(), 2),
                                              ("NOT", McCullochPitts.NOT(), 1))}
    out["mcculloch_pitts_note"] = "XOR is impossible for one threshold unit: it is not linearly separable."
    (RESULTS / "sandbox.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("svd", "gradient_check_relative_error")}, indent=1))


if __name__ == "__main__":
    main()

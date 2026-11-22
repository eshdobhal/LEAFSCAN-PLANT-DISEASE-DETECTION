"""Brief item 6: Hebbian, modified Hebbian (Oja), competitive and error-correction learning, plus a
Boltzmann (RBM) experiment, all on the same leaf photos so they can be compared conceptually.

    python -m fundamentals.experiments.learning_rules

Writes fundamentals/results/rules_*.png and learning_rules.json.
"""
import json

import numpy as np
from PIL import Image

from fundamentals import data, hebbian, tensors
from fundamentals.plotting import RESULTS, SERIES, plt, save

SMALL = 12  # Hebbian rules on 12 x 12 RGB = 432 inputs, so the online rules converge in seconds


def shrink(images, size=SMALL, grey=False):
    out = []
    for im in images:
        p = Image.fromarray(im).resize((size, size), Image.BILINEAR)
        out.append(np.asarray(p.convert("L") if grey else p, dtype=np.float64) / 255.0)
    return np.stack(out)


def main():
    leaves = data.load()
    x_img, y = leaves["train"]
    xv_img, yv = leaves["val_lab"]
    healthy = np.array([("healthy" in data.CLASSES[c]) for c in range(len(data.CLASSES))])
    out = {}

    x = shrink(x_img).reshape(len(x_img), -1)
    mean = x.mean(0)
    xc = x - mean

    # ---------------------------------------------------------------- Hebbian vs Oja vs PCA
    z = xc / np.linalg.norm(xc, axis=1).mean()  # average example length 1: same scale for both rules
    _, hebb_norms = hebbian.hebbian(z, lr=0.01, epochs=2)
    w_oja, oja_norms = hebbian.oja(z, lr=0.05, epochs=20)
    pc1 = hebbian.first_principal_component(xc)
    cos = abs(tensors.cosine_similarity(w_oja, pc1))
    out["hebbian"] = {"weight_norm_start": float(hebb_norms[0]), "weight_norm_after_2_epochs": float(hebb_norms[-1]),
                      "note": "plain Hebbian learning has no decay term, so |w| grows without bound"}
    out["oja"] = {"final_weight_norm": round(float(oja_norms[-1]), 4), "cosine_with_svd_pc1": round(float(cos), 4)}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].plot(hebb_norms, color=SERIES[7], lw=1.5, label="Hebbian")
    axes[0].plot(oja_norms[:len(hebb_norms)], color=SERIES[0], lw=1.5, label="Oja")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("update")
    axes[0].set_title("|w| during learning")
    axes[0].legend()

    def as_img(v):
        v = v.reshape(SMALL, SMALL, 3)
        return (v - v.min()) / (v.max() - v.min() + 1e-12)
    axes[1].imshow(as_img(np.sign(w_oja @ pc1) * w_oja))
    axes[1].set_title("Oja weight vector")
    axes[2].imshow(as_img(pc1))
    axes[2].set_title("SVD principal component 1")
    axes[3].imshow(mean.reshape(SMALL, SMALL, 3))
    axes[3].set_title("Mean leaf")
    for ax in axes[1:]:
        ax.axis("off")
    fig.suptitle(f"Oja's rule recovers the first principal component (cosine {cos:.3f})", x=0.01, ha="left")
    save(fig, "rules_hebbian_oja")

    # ---------------------------------------------------------------- competitive learning
    k = len(data.CLASSES)
    protos, assign, distortion = hebbian.competitive(x, k=k, lr=0.05, epochs=15)
    purity = sum(np.bincount(y[assign == j]).max() for j in range(k) if (assign == j).any()) / len(y)
    out["competitive"] = {"prototypes": k, "distortion_start": round(distortion[0], 4),
                          "distortion_end": round(distortion[-1], 4), "cluster_purity_vs_true_labels": round(float(purity), 4),
                          "note": "unsupervised: prototypes group leaves by colour and texture, not by disease name"}
    fig, axes = plt.subplots(1, k + 1, figsize=(14, 2.2))
    axes[0].plot(distortion, color=SERIES[0], lw=2)
    axes[0].set_title("distortion")
    for j in range(k):
        axes[j + 1].imshow(np.clip(protos[j].reshape(SMALL, SMALL, 3), 0, 1))
        axes[j + 1].set_title(f"unit {j} ({(assign == j).sum()})", fontsize=8)
        axes[j + 1].axis("off")
    fig.suptitle(f"Competitive (winner-take-all) learning: prototypes, purity {purity:.0%}", x=0.01, ha="left")
    save(fig, "rules_competitive")

    # ---------------------------------------------------------------- error-correction (healthy vs diseased)
    t = healthy[y].astype(float)
    tv = healthy[yv].astype(float)
    xv = shrink(xv_img).reshape(len(xv_img), -1) - mean
    scale = np.abs(xc).max()
    w_p, mistakes = hebbian.perceptron(xc / scale, t, lr=0.1, epochs=30)
    w_d, mse = hebbian.delta_rule(xc / scale, t, lr=0.002, epochs=30)
    xvb = np.hstack([xv / scale, np.ones((len(xv), 1))])
    acc_p = float(((xvb @ w_p >= 0) == tv).mean())
    acc_d = float(((xvb @ w_d >= 0.5) == tv).mean())
    out["error_correction"] = {"task": "healthy vs diseased (2 of 8 classes healthy)",
                               "perceptron_val_accuracy": round(acc_p, 4), "delta_rule_val_accuracy": round(acc_d, 4),
                               "perceptron_mistakes_per_epoch": mistakes, "delta_rule_mse": [round(m, 4) for m in mse],
                               "majority_baseline": round(float(max(tv.mean(), 1 - tv.mean())), 4)}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(mistakes, color=SERIES[0], lw=2)
    axes[0].set_title(f"Perceptron: mistakes per epoch (val acc {acc_p:.0%})")
    axes[1].plot(mse, color=SERIES[1], lw=2)
    axes[1].set_title(f"Delta rule: mean squared error (val acc {acc_d:.0%})")
    for ax in axes:
        ax.set_xlabel("epoch")
    save(fig, "rules_error_correction")

    # ---------------------------------------------------------------- Boltzmann learning (RBM)
    v = (shrink(x_img, 16, grey=True).reshape(len(x_img), -1) > 0.5).astype(float)  # binarised leaf silhouettes
    vv = (shrink(xv_img, 16, grey=True).reshape(len(xv_img), -1) > 0.5).astype(float)
    rbm = hebbian.RBM(256, 64, seed=0)
    history = rbm.fit(v, epochs=40, batch=32, lr=0.05)
    noise = (np.random.default_rng(0).random(vv.shape) < 0.5).astype(float)
    out["rbm"] = {"visible": 256, "hidden": 64, "history_first": history[0], "history_last": history[-1],
                  "val_free_energy_real_leaves": round(float(rbm.free_energy(vv).mean()), 2),
                  "val_free_energy_random_noise": round(float(rbm.free_energy(noise).mean()), 2),
                  "note": "lower free energy = more probable under the model; real leaves score lower than noise"}
    fig, axes = plt.subplots(2, 7, figsize=(12, 4.2))
    rec = rbm.reconstruct(vv[:6])
    axes[0, 0].plot([h["reconstruction_error"] for h in history], color=SERIES[0], lw=2)
    axes[0, 0].set_title("reconstruction error", fontsize=8)
    axes[1, 0].plot([h["mean_free_energy"] for h in history], color=SERIES[1], lw=2)
    axes[1, 0].set_title("mean free energy", fontsize=8)
    for j in range(6):
        axes[0, j + 1].imshow(vv[j].reshape(16, 16), cmap="Greys_r")
        axes[0, j + 1].set_title("input", fontsize=8)
        axes[1, j + 1].imshow(rec[j].reshape(16, 16), cmap="Greys_r")
        axes[1, j + 1].set_title("RBM reconstruction", fontsize=8)
        axes[0, j + 1].axis("off")
        axes[1, j + 1].axis("off")
    fig.suptitle("Boltzmann learning: RBM on binarised leaf thumbnails (CD-1)", x=0.01, ha="left")
    save(fig, "rules_rbm")

    (RESULTS / "learning_rules.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()

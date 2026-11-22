"""Brief items 4 and 5: an MLP baseline on leaf photos, then controlled ablations.

    python -m fundamentals.experiments.mlp_ablations

Task: 8 leaf conditions, 24 x 24 RGB PlantVillage photos, leaf-grouped splits (fundamentals/data.py).
Every ablation changes one thing against the baseline and runs 3 seeds; results are mean +/- std of
validation accuracy (the test set is used once, for the baseline only).
Writes fundamentals/results/mlp_*.json and mlp_*.png.
"""
import json

import numpy as np

from fundamentals import data
from fundamentals.mlp import MLP, train
from fundamentals.optim import OPTIMIZERS
from fundamentals.plotting import RESULTS, SERIES, plt, save

SEEDS = (0, 1, 2)
BASE = {"sizes": [1728, 256, 128, 8], "activation": "relu", "init": "he", "batchnorm": False, "dropout": 0.0,
        "optimizer": "momentum", "lr": 0.01, "batch_size": 64, "epochs": 30, "l2": 0.0, "augment": False,
        "train_per_class": 250, "standardise": True}


def prepare(leaves, cfg, seed):
    x_tr_img, y_tr = leaves["train"]
    if cfg["train_per_class"] < 250:  # smaller training sets for the overfitting studies
        rng = np.random.default_rng(100 + seed)
        keep = np.concatenate([rng.choice(np.where(y_tr == c)[0], cfg["train_per_class"], replace=False)
                               for c in range(len(data.CLASSES))])
        x_tr_img, y_tr = x_tr_img[keep], y_tr[keep]
    if cfg["standardise"]:
        x_tr, mean, std = data.features(x_tr_img)
    else:  # raw pixel values in [0, 1]
        x_tr = x_tr_img.reshape(len(x_tr_img), -1) / 255.0
        mean, std = np.zeros(x_tr.shape[1]), np.ones(x_tr.shape[1])
    x_va = (leaves["val_lab"][0].reshape(len(leaves["val_lab"][0]), -1) / 255.0 - mean) / std
    x_te = (leaves["test_lab"][0].reshape(len(leaves["test_lab"][0]), -1) / 255.0 - mean) / std
    return (x_tr, y_tr, x_va, leaves["val_lab"][1]), (x_te, leaves["test_lab"][1]), mean, std


def make_augment(mean, std):
    """Random horizontal/vertical flips, brightness change and pixel noise, done in image space."""
    def augment(xb, rng):
        imgs = (xb * std + mean).reshape(-1, 24, 24, 3)
        flip_h = rng.random(len(imgs)) < 0.5
        flip_v = rng.random(len(imgs)) < 0.5
        imgs = imgs.copy()
        imgs[flip_h] = imgs[flip_h, :, ::-1]
        imgs[flip_v] = imgs[flip_v, ::-1]
        imgs = imgs * rng.uniform(0.8, 1.2, (len(imgs), 1, 1, 1)) + rng.normal(0, 0.02, imgs.shape)
        return (imgs.reshape(len(imgs), -1) - mean) / std
    return augment


def run(leaves, cfg, seed, name):
    data_tuple, test, mean, std = prepare(leaves, cfg, seed)
    model = MLP(cfg["sizes"], cfg["activation"], cfg["init"], cfg["batchnorm"], cfg["dropout"], seed=seed)
    opt = OPTIMIZERS[cfg["optimizer"]](lr=cfg["lr"])
    log = train(model, opt, data_tuple, cfg["epochs"], cfg["batch_size"], cfg["l2"],
                make_augment(mean, std) if cfg["augment"] else None, seed=seed, name=name, hyper=cfg)
    return model, log, test


def study(leaves, title, variants):
    """variants: {label: dict of changes to BASE}. Returns summary rows and per-epoch curves."""
    rows, curves = [], {}
    for label, change in variants.items():
        cfg = {**BASE, **change}
        logs = [run(leaves, cfg, s, f"{title}/{label}")[1] for s in SEEDS]
        val = np.array([[h["val_acc"] for h in log.history] for log in logs])
        train_acc = np.array([log.history[-1]["train_acc"] for log in logs])
        rows.append({"variant": label, "changed": change,
                     "val_acc_mean": round(float(val[:, -1].mean()), 4), "val_acc_std": round(float(val[:, -1].std()), 4),
                     "train_acc_mean": round(float(train_acc.mean()), 4),
                     "generalisation_gap": round(float(train_acc.mean() - val[:, -1].mean()), 4),
                     "seconds_per_run": round(float(np.mean([l.seconds for l in logs])), 2)})
        curves[label] = val.mean(0)
        print(f"  {title:15s} {label:28s} val {rows[-1]['val_acc_mean']:.3f} +/- {rows[-1]['val_acc_std']:.3f}"
              f"  gap {rows[-1]['generalisation_gap']:+.3f}")
    return rows, curves


def plot_curves(curves, title, name):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for (label, c), colour in zip(curves.items(), SERIES):
        ax.plot(np.arange(1, len(c) + 1), c * 100, color=colour, lw=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    save(fig, name)


def main():
    leaves = data.load()

    # ---------------------------------------------------------------- baseline (items 4)
    model, log, (x_te, y_te) = run(leaves, BASE, 0, "baseline")
    probs = model.predict_proba(x_te)
    conf = np.zeros((8, 8), int)
    for t, p in zip(y_te, probs.argmax(1)):
        conf[t, p] += 1
    baseline = {"hyperparameters": BASE, "n_parameters": log.n_parameters,
                "parameters_by_layer": {f"layer{i}_{k}": list(v.shape) for i, (p, _) in enumerate(model.parameters())
                                        for k, v in p.items()},
                "history": log.history, "test_accuracy": round(float((probs.argmax(1) == y_te).mean()), 4),
                "test_confusion": conf.tolist(), "classes": data.CLASSES, "seconds": round(log.seconds, 1)}
    (RESULTS / "mlp_baseline.json").write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    log.save(RESULTS / "mlp_baseline_runlog.json")
    print(f"baseline: {log.n_parameters:,} parameters, test accuracy {baseline['test_accuracy']:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ep = [h["epoch"] for h in log.history]
    axes[0].plot(ep, [h["train_loss"] for h in log.history], color=SERIES[0], lw=2)
    axes[0].set_title("Training loss (cross-entropy)")
    axes[1].plot(ep, [h["train_acc"] * 100 for h in log.history], color=SERIES[0], lw=2, label="train")
    axes[1].plot(ep, [h["val_acc"] * 100 for h in log.history], color=SERIES[1], lw=2, label="validation")
    axes[1].set_title("Accuracy (%)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
    save(fig, "mlp_baseline_curves")

    # ---------------------------------------------------------------- ablations (item 5)
    results = {}
    studies = {
        "optimizer": {
            "Batch GD (full batch, lr 0.1)": {"optimizer": "sgd", "batch_size": None, "lr": 0.1},
            "SGD (batch 64)": {"optimizer": "sgd"},
            "Momentum 0.9": {"optimizer": "momentum"},
            "Nesterov 0.9": {"optimizer": "nesterov"},
        },
        "initialisation": {  # a deeper net, where initialisation matters most
            "zeros": {"init": "zeros", "sizes": [1728, 128, 128, 128, 8]},
            "normal(0, 0.01)": {"init": "normal", "sizes": [1728, 128, 128, 128, 8]},
            "Xavier": {"init": "xavier", "sizes": [1728, 128, 128, 128, 8]},
            "He": {"init": "he", "sizes": [1728, 128, 128, 128, 8]},
        },
        "normalisation": {
            "none (raw pixels 0-1)": {"standardise": False},
            "input standardisation": {},
            "standardisation + BatchNorm": {"batchnorm": True},
        },
        "regularisation": {  # 40 photos per class and 80 epochs: overfitting is easy to see
            "none": {"train_per_class": 40, "epochs": 80},
            "L2 1e-3": {"train_per_class": 40, "epochs": 80, "l2": 1e-3},
            "dropout 0.3": {"train_per_class": 40, "epochs": 80, "dropout": 0.3},
            "augmentation": {"train_per_class": 40, "epochs": 80, "augment": True},
            "L2 + dropout + augmentation": {"train_per_class": 40, "epochs": 80, "l2": 1e-3, "dropout": 0.3, "augment": True},
        },
        "training_set_size": {
            f"{n} per class": {"train_per_class": n} for n in (20, 50, 100, 250)
        },
        "activation": {
            "sigmoid": {"activation": "sigmoid", "init": "xavier"},
            "tanh": {"activation": "tanh", "init": "xavier"},
            "ReLU": {},
        },
    }
    for title, variants in studies.items():
        rows, curves = study(leaves, title, variants)
        results[title] = rows
        plot_curves(curves, f"Ablation: {title.replace('_', ' ')}", f"mlp_ablation_{title}")
    (RESULTS / "mlp_ablations.json").write_text(json.dumps({"base": BASE, "seeds": SEEDS, "studies": results},
                                                           indent=1), encoding="utf-8")

    # Generalisation: train vs validation accuracy as the training set grows.
    rows = results["training_set_size"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sizes = [int(r["variant"].split()[0]) for r in rows]
    ax.plot(sizes, [r["train_acc_mean"] * 100 for r in rows], "o-", color=SERIES[0], lw=2, label="train")
    ax.plot(sizes, [r["val_acc_mean"] * 100 for r in rows], "o-", color=SERIES[1], lw=2, label="validation")
    ax.set_xscale("log")
    ax.set_xlabel("training photos per class")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Overfitting shrinks as data grows")
    ax.legend()
    save(fig, "mlp_generalisation")


if __name__ == "__main__":
    main()

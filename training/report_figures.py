"""Figures for the README: training curves, model comparison, per-class field recall, confusion matrix.

    .venv-train/Scripts/python training/report_figures.py

Step 1 (GPU, once): runs the shipped ensemble (224 px + 288 px, mirror TTA) on both test sets and caches
the predictions in training/runs/ensemble/test_predictions.npz.
Step 2: draws every figure twice, light and dark, into docs/images/ (README picks one per GitHub theme).
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import CLASSES, ROOT, RUNS, LeafDataset, build_model, eval_transform, load_splits, predict_probs

OUT = ROOT / "docs" / "images"
CACHE = RUNS / "ensemble" / "test_predictions.npz"
MEMBERS = [("caformer_s18", 224), ("caformer_s18_288", 288)]

# Reference data-viz palette (categorical slots 1-3 in fixed order, sequential blue, chart chrome).
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"],
                  seq=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70"],
                 seq=["#1a1a19", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]),
}


def pretty(key):
    """'Tomato___Tomato_Yellow_Leaf_Curl_Virus' -> 'Tomato · Yellow leaf curl virus'."""
    if key == "Background_without_leaves":
        return "No leaf (background)"
    crop, cond = key.split("___")
    crop = crop.replace("_", " ").replace(",", "").replace("Pepper bell", "Bell pepper").split(" (")[0]
    cond = cond.replace("_", " ").replace("Tomato ", "").strip()
    cond = {"Cercospora leaf spot Gray leaf spot": "Gray leaf spot", "Haunglongbing (Citrus greening)": "Citrus greening",
            "Spider mites Two-spotted spider mite": "Spider mites", "Esca (Black Measles)": "Esca",
            "Leaf blight (Isariopsis Leaf Spot)": "Leaf blight"}.get(cond, cond)
    return f"{crop} · {cond[0].upper() + cond[1:].lower()}"


# ---------------------------------------------------------------- step 1: predictions (GPU)
def predict_test_sets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    out = {}
    for split in ("test_lab", "test_field"):
        member_probs = []
        for run, size in MEMBERS:
            model = build_model(pretrained=False)
            state = torch.load(RUNS / run / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
            model.to(device)
            loader = DataLoader(LeafDataset(splits[split], eval_transform(size)), batch_size=64, num_workers=4)
            probs, y = predict_probs(model, loader, device, tta=True)
            member_probs.append(probs)
            del model
            torch.cuda.empty_cache()
        conf, pred = torch.stack(member_probs).mean(0).max(1)
        out[f"{split}_true"], out[f"{split}_pred"], out[f"{split}_conf"] = y.numpy(), pred.numpy(), conf.numpy()
    np.savez_compressed(CACHE, **out)


# ---------------------------------------------------------------- step 2: figures
def style(plt, t):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": t["ink"],
        "axes.facecolor": t["surface"], "figure.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": t["ink2"],
    })


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=160, bbox_inches="tight", pad_inches=0.25)


def training_curves(plt, t, mode):
    runs = {"224 px model": "caformer_s18", "288 px model": "caformer_s18_288"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, key, title in ((axes[0], "lab_acc", "Lab validation accuracy"),
                           (axes[1], "field_acc", "Field validation accuracy")):
        for (label, run), colour in zip(runs.items(), t["series"]):
            hist = json.loads((RUNS / run / "history.json").read_text())
            x = [h["epoch"] for h in hist]
            y = [h[key] * 100 for h in hist]
            ax.plot(x, y, color=colour, lw=2, marker="o", ms=5, label=label,
                    markeredgecolor=t["surface"], markeredgewidth=1.5)
            ax.annotate(f"{y[-1]:.1f}%", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=t["ink2"], fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 9))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "training-curves", mode)
    plt.close(fig)


def model_comparison(plt, t, mode):
    runs = {"224 px": "caformer_s18", "288 px": "caformer_s18_288", "Ensemble (shipped)": "ensemble"}
    metrics = {"Accuracy": "accuracy", "Macro recall": "macro_recall", "Macro precision": "macro_precision",
               "Accuracy when ≥80% sure": "accuracy_when_confident"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.26
    x = np.arange(len(metrics))
    for i, ((label, run), colour) in enumerate(zip(runs.items(), t["series"])):
        m = json.loads((RUNS / run / "test_metrics.json").read_text())["test_field_tta"]
        vals = [m[k] * 100 for k in metrics.values()]
        bars = ax.bar(x + (i - 1) * (width + 0.02), vals, width, color=colour, label=label, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=t["ink2"])
    ax.set_xticks(x, metrics.keys())
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Real field photos (PlantDoc test, 225 photos): single models vs ensemble")
    ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 112)  # bars start at zero so lengths compare honestly; headroom for the legend
    ax.set_yticks(range(0, 101, 20))
    fig.tight_layout()
    save(fig, "model-comparison", mode)
    plt.close(fig)


def field_per_class(plt, t, mode, pred):
    y, p = pred["test_field_true"], pred["test_field_pred"]
    rows = []
    for c in np.unique(y):
        mask = y == c
        rows.append((pretty(CLASSES[c]), (p[mask] == c).mean() * 100, int((p[mask] == c).sum()), int(mask.sum())))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 8))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(names, vals, height=0.62, color=t["series"][0], zorder=3)
    for i, (_, v, hit, n) in enumerate(rows):
        ax.text(v + 1, i, f"{v:.0f}%  ({hit}/{n})", va="center", fontsize=8.5, color=t["ink2"])
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Recall per condition on real field photos (ensemble, PlantDoc test)")
    fig.tight_layout()
    save(fig, "field-recall-per-class", mode)
    plt.close(fig)


def confusion(plt, t, mode, pred):
    from matplotlib.colors import LinearSegmentedColormap
    y, p = pred["test_field_true"], pred["test_field_pred"]
    classes = sorted(set(y.tolist()) | set(p.tolist()), key=lambda c: pretty(CLASSES[c]))
    idx = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, p):
        counts[idx[a], idx[b]] += 1
    rows = counts.sum(1, keepdims=True)
    share = np.divide(counts, rows, out=np.zeros_like(counts, dtype=float), where=rows > 0)
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if counts[i, j]:
                ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=7,
                        color="#ffffff" if share[i, j] > 0.45 else t["ink2"])
    labels = [pretty(CLASSES[c]) for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(classes)), labels, fontsize=7.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True condition")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion matrix, real field photos (counts; shading = share of each true row)")
    fig.tight_layout()
    save(fig, "confusion-field", mode)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not CACHE.exists():
        predict_test_sets()
    pred = dict(np.load(CACHE))
    lab_acc = (pred["test_lab_true"] == pred["test_lab_pred"]).mean()
    field_acc = (pred["test_field_true"] == pred["test_field_pred"]).mean()
    print(f"check: lab accuracy {lab_acc:.2%}, field accuracy {field_acc:.2%}")

    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        style(plt, t)
        training_curves(plt, t, mode)
        model_comparison(plt, t, mode)
        field_per_class(plt, t, mode, pred)
        confusion(plt, t, mode, pred)
    print("figures written to", OUT)


# The guard is required on Windows: DataLoader workers start by re-importing this file.
if __name__ == "__main__":
    main()

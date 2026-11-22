"""The evaluation package (brief item 10): one command, every metric the brief names.

    python -m surveillance.evaluate_all

1. Classification benchmark: CAFormer-S18 (224, 288, ensemble) vs ResNet-50 vs GoogLeNet: accuracy, macro F1,
   macro recall, and expected calibration error (ECE) before/after temperature scaling (fitted on validation).
2. Robustness: accuracy and macro F1 of the shipped ensemble on field photos under 16 corruptions.
3. Detection (mAP), segmentation (IoU, Dice, lesion F1, severity) and instance (mask mAP) summaries.
4. Deployment: CPU latency, parameters, weight size and process memory for every component.
Writes surveillance/results/evaluation.json, eval_*.png and EVALUATION_SUMMARY.md.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "surveillance" / "results"
SRUNS = ROOT / "surveillance" / "runs"
sys.path.insert(0, str(ROOT / "training"))
from common import RUNS, LeafDataset, build_model, eval_transform, load_splits, metrics, predict_probs, run_arch  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIERS = {"CAFormer-S18 224": [("caformer_s18", 224)], "CAFormer-S18 288": [("caformer_s18_288", 288)],
               "CAFormer ensemble (shipped)": [("caformer_s18", 224), ("caformer_s18_288", 288)],
               "ResNet-50": [("resnet50", 224)], "GoogLeNet": [("googlenet", 224)]}


def load(run):
    net = build_model(pretrained=False, arch=run_arch(run))
    state = torch.load(RUNS / run / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return net.to(DEVICE).eval()


def probs_for(members, items, transform=None):
    out = []
    for run, size in members:
        tf = eval_transform(size) if transform is None else transform(size)
        loader = torch.utils.data.DataLoader(LeafDataset(items, tf), batch_size=64, num_workers=4)
        p, y = predict_probs(load(run), loader, DEVICE, tta=True)
        out.append(p)
    return torch.stack(out).mean(0), y


# ------------------------------------------------------------------ calibration
def ece(probs, y, bins=15):
    conf, pred = probs.max(1)
    edges = torch.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).float().mean().item(), conf[m].mean().item()
            total += m.float().mean().item() * abs(acc - c)
            rows.append((float((lo + hi) / 2), acc, c, int(m.sum())))
    return total, rows


def fit_temperature(probs, y):
    """Temperature scaling: one scalar T dividing the log-probabilities, fitted on validation (LBFGS on NLL)."""
    logp = probs.clamp_min(1e-9).log()
    t = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logp / t, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.detach())


def apply_t(probs, t):
    return torch.softmax(probs.clamp_min(1e-9).log() / t, 1)


# ------------------------------------------------------------------ robustness
def corruption(kind, level):
    rng = np.random.default_rng(0)

    def f(img):
        if kind == "gaussian noise":
            a = np.asarray(img, np.float32) / 255 + rng.normal(0, level, (img.size[1], img.size[0], 3))
            return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))
        if kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(level))
        if kind == "brightness":
            return ImageEnhance.Brightness(img).enhance(level)
        if kind == "contrast":
            return ImageEnhance.Contrast(img).enhance(level)
        if kind == "JPEG quality":
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=level)
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        if kind == "rotation":
            return img.rotate(level, expand=True, fillcolor=(128, 128, 128))
        if kind == "occlusion (grey square)":
            img = img.copy()
            w, h = img.size
            s = int(min(w, h) * level)
            x, y = rng.integers(0, w - s), rng.integers(0, h - s)
            img.paste((128, 128, 128), (x, y, x + s, y + s))
            return img
        raise ValueError(kind)
    return f


CORRUPTIONS = [("gaussian noise", 0.05), ("gaussian noise", 0.1), ("gaussian noise", 0.2), ("blur", 1), ("blur", 2),
               ("blur", 4), ("brightness", 0.5), ("brightness", 1.5), ("contrast", 0.5), ("contrast", 1.5),
               ("JPEG quality", 20), ("JPEG quality", 8), ("rotation", 30), ("rotation", 90),
               ("occlusion (grey square)", 0.25), ("occlusion (grey square)", 0.5)]


def robustness(items, members):
    from torchvision import transforms
    rows = []
    for kind, level in [("none", None)] + CORRUPTIONS:
        def tf(size, kind=kind, level=level):
            base = eval_transform(size)
            return base if kind == "none" else transforms.Compose([transforms.Lambda(corruption(kind, level)), base])
        p, y = probs_for(members, items, tf)
        m = metrics(y.tolist(), p.argmax(1).tolist())
        rows.append({"corruption": kind, "level": level, "accuracy": round(m["accuracy"], 4), "macro_F1": round(m["macro_f1"], 4)})
        print(f"  {kind:24s} {str(level):5s} acc {m['accuracy']:.3f}", flush=True)
    return rows


# ------------------------------------------------------------------ deployment
def process_memory_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 2**20)
    except ImportError:
        return None


def deployment():
    """CPU latency / size of every component, measured the same way (median of 10 after warm-up)."""
    rows = {}
    for name, path in (("classifier: CAFormer ensemble", RUNS / "caformer_s18" / "test_metrics.json"),):
        pass
    for f in sorted(SRUNS.glob("*/test_metrics.json")):
        m = json.loads(f.read_text())
        weights = f.parent / "best.pt"
        rows[f.parent.name] = {"cpu_latency_ms": m.get("cpu_latency_ms"), "parameters_M": m.get("parameters_M"),
                               "weights_MB_fp16": round(weights.stat().st_size / 2**20, 1) if weights.exists() else None}
    yolo_dir = SRUNS
    for d in ("tea", "plantdoc"):
        w = yolo_dir / f"{d}_yolo_train" / "weights" / "best.pt"
        if f"{d}_yolo" in rows and w.exists():
            rows[f"{d}_yolo"]["weights_MB_fp16"] = round(w.stat().st_size / 2**20, 1)
    return rows


def full_pipeline_profile():
    """End-to-end CLI pipeline on CPU: per-component latency and the process's peak memory."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from surveillance.pipeline import Surveillance
    before = process_memory_mb()
    s = Surveillance(device="cpu")
    after = process_memory_mb()
    runs = []
    for name in ("potato_early_blight.JPG", "tomato_bacterial_spot.JPG", "corn_common_rust.JPG", "tomato-mold.jpg"):
        for _ in range(3):
            report, _ = s.analyze(ROOT / "test_images" / name, name=name)
            runs.append(report.timings_ms)
    med = {k: round(float(np.median([r[k] for r in runs if k in r])), 1) for k in runs[0]}
    return {"components": s.components, "median_ms_per_component": med,
            "process_memory_MB_before_models": before, "process_memory_MB_all_models_loaded": after,
            "peak_memory_MB_after_inference": process_memory_mb()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    splits = load_splits()
    rng = np.random.default_rng(0)
    lab_test = [splits["test_lab"][i] for i in rng.choice(len(splits["test_lab"]), 2000, replace=False)]
    lab_val = [splits["val_lab"][i] for i in rng.choice(len(splits["val_lab"]), 1000, replace=False)]
    out = {"note": "lab = 2,000 random PlantVillage test photos (leaf-grouped split); field = all 225 PlantDoc test photos"}

    # 1. Classification benchmark + calibration
    bench, reliability = {}, {}
    for name, members in CLASSIFIERS.items():
        if not all((RUNS / r / "best.pt").exists() for r, _ in members):
            continue
        p_val, y_val = probs_for(members, splits["val_field"] + lab_val)
        t = fit_temperature(p_val, y_val)
        row = {"temperature": round(t, 3)}
        for split, items in (("field", splits["test_field"]), ("lab", lab_test)):
            p, y = probs_for(members, items)
            m = metrics(y.tolist(), p.argmax(1).tolist())
            e_raw, rel = ece(p, y)
            e_cal, rel_cal = ece(apply_t(p, t), y)
            row[split] = {"accuracy": round(m["accuracy"], 4), "macro_F1": round(m["macro_f1"], 4),
                          "macro_recall": round(m["macro_recall"], 4), "ECE": round(e_raw, 4),
                          "ECE_after_temperature_scaling": round(e_cal, 4)}
            if split == "field":
                reliability[name] = (rel, rel_cal)
        bench[name] = row
        print(f"{name:30s} field acc {row['field']['accuracy']:.3f} F1 {row['field']['macro_F1']:.3f} ECE {row['field']['ECE']:.3f}"
              f" -> {row['field']['ECE_after_temperature_scaling']:.3f} (T={t:.2f})", flush=True)
    out["classification"] = bench

    fig, axes = plt.subplots(1, len(reliability), figsize=(3.4 * len(reliability), 3.4), sharey=True)
    for ax, (name, (rel, rel_cal)) in zip(np.atleast_1d(axes), reliability.items()):
        ax.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--")
        for r, colour, lab in ((rel, "#2a78d6", "raw"), (rel_cal, "#eb6834", "temperature-scaled")):
            ax.plot([c for _, _, c, _ in r], [a for _, a, _, _ in r], "o-", color=colour, lw=1.8, ms=4, label=lab)
        ax.set_title(f"{name}\nECE {bench[name]['field']['ECE']:.3f} -> {bench[name]['field']['ECE_after_temperature_scaling']:.3f}",
                     fontsize=8, loc="left")
        ax.set_xlabel("confidence")
    np.atleast_1d(axes)[0].set_ylabel("accuracy (field test)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Reliability diagrams on real field photos (perfect calibration = dashed line)", x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / "eval_reliability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Robustness of the shipped ensemble on field photos
    print("robustness (shipped ensemble, field test):", flush=True)
    out["robustness"] = robustness(splits["test_field"], CLASSIFIERS["CAFormer ensemble (shipped)"])
    base = out["robustness"][0]["accuracy"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rows = out["robustness"][1:]
    labels = [f"{r['corruption']} {r['level']}" for r in rows]
    ax.barh(labels, [r["accuracy"] * 100 for r in rows], color="#2a78d6", height=0.6, zorder=3)
    ax.axvline(base * 100, color="#d03b3b", ls="--", lw=1)
    ax.text(base * 100 + 0.5, len(rows) - 0.4, f"clean {base:.0%}", color="#d03b3b", fontsize=8)
    ax.set_xlabel("accuracy on field test photos (%)")
    ax.set_title("Robustness of the shipped ensemble to image corruptions", loc="left", fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e1e0d9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "eval_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Detection / segmentation summaries (from each run's test_metrics.json)
    out["detection"] = {d.name: json.loads((d / "test_metrics.json").read_text()) for d in sorted(SRUNS.iterdir())
                        if (d / "test_metrics.json").exists() and (d.name.startswith("tea_") or d.name.startswith("plantdoc_"))}
    out["segmentation"] = {d.name: json.loads((d / "test_metrics.json").read_text()) for d in sorted(SRUNS.iterdir())
                           if (d / "test_metrics.json").exists() and d.name.startswith("leafseg_")}

    # 4. Deployment profile
    out["deployment"] = deployment()
    out["pipeline_cpu_profile"] = full_pipeline_profile()
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=1))
    write_summary(out)
    print(json.dumps(out["pipeline_cpu_profile"], indent=1))


def write_summary(out):
    lines = ["# Evaluation summary (generated by surveillance/evaluate_all.py)", "",
             "## Classification (39 classes)", "",
             "| Model | Field acc | Field macro F1 | Field ECE (raw → scaled) | Lab acc | Lab macro F1 | Lab ECE | T |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in out["classification"].items():
        f, lab = r["field"], r["lab"]
        lines.append(f"| {name} | {f['accuracy']:.2%} | {f['macro_F1']:.3f} | {f['ECE']:.3f} → {f['ECE_after_temperature_scaling']:.3f} | "
                     f"{lab['accuracy']:.2%} | {lab['macro_F1']:.3f} | {lab['ECE']:.3f} | {r['temperature']} |")
    lines += ["", "## Robustness of the shipped ensemble (field test)", "", "| Corruption | Level | Accuracy | Macro F1 |", "|---|---|---:|---:|"]
    lines += [f"| {r['corruption']} | {r['level']} | {r['accuracy']:.2%} | {r['macro_F1']:.3f} |" for r in out["robustness"]]
    lines += ["", "## Detection (COCO mAP on the untouched test split)", "", "| Run | mAP@[.5:.95] | mAP@.5 | Pest mAP | CPU ms | Params (M) |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in out["detection"].items():
        lines.append(f"| {name} | {m['mAP']:.3f} | {m['mAP50']:.3f} | {m.get('pest_mAP') if m.get('pest_mAP') is not None else '-'} | "
                     f"{m.get('cpu_latency_ms')} | {m.get('parameters_M')} |")
    lines += ["", "## Segmentation and instance analysis (LeafNet75 test)", ""]
    for name, m in out["segmentation"].items():
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    p = out["pipeline_cpu_profile"]
    lines += ["", "## Full pipeline on CPU", "", f"Components: {', '.join(p['components'])}.", "",
              "| Component | Median ms |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in p["median_ms_per_component"].items()]
    lines += ["", f"Process memory: {p['process_memory_MB_before_models']} MB before models, "
              f"{p['process_memory_MB_all_models_loaded']} MB with all models loaded, {p['peak_memory_MB_after_inference']} MB after inference."]
    (RESULTS / "EVALUATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

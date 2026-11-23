"""The evaluation package (brief item 10): one command, every metric the brief names.

    python -m surveillance.evaluate_all

1. Classification benchmark: CAFormer-S18 (224, 288, ensemble) vs ResNet-50 vs GoogLeNet: accuracy, macro F1,
   macro recall, and expected calibration error (ECE) before/after temperature scaling. Temperatures are fitted
   on validation only, per domain (field, lab), plus once on mixed validation to show a mismatched fit.
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
import zlib
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
    # clone(): probabilities come from inference_mode, whose tensors autograd refuses to use
    logp = probs.clone().float().clamp_min(1e-9).log()
    y = y.clone().long()
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) > 0 always (a raw T can overshoot below 0)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logp / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def apply_t(probs, t):
    return torch.softmax(probs.clamp_min(1e-9).log() / t, 1)


# ------------------------------------------------------------------ robustness
class Corruption:
    """One image corruption (kind, level). A class, not a closure, so DataLoader worker processes can pickle it."""

    def __init__(self, kind, level):
        self.kind, self.level = kind, level

    def __call__(self, img):
        kind, level = self.kind, self.level
        # seeded by the image itself: deterministic in any DataLoader worker process
        rng = np.random.default_rng(zlib.crc32(img.tobytes()))
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


def corruption(kind, level):
    return Corruption(kind, level)


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


def pipeline_threshold(key):
    from surveillance.pipeline import DETECT_THRESHOLD, THRESHOLDS
    return THRESHOLDS.get(key, DETECT_THRESHOLD)


@torch.inference_mode()
def pest_out_of_domain(threshold=0.40):
    """How often the tea-trained pest model reports a pest on photos that contain none (PlantDoc test, 13 other
    crops), versus how often it finds the pest on tea test photos that have one. Motivates the crop="tea" gate."""
    from torchvision.transforms import functional as F

    from surveillance import detection
    net, names = detection.load("tea", "retinanet", DEVICE)
    pest_ids = {i + 1 for i, n in enumerate(names) if "(pest)" in n}

    def has_pest(path):
        out = net([F.to_tensor(Image.open(path).convert("RGB")).to(DEVICE)])[0]
        return bool(((out["scores"] >= threshold) & torch.isin(out["labels"], torch.tensor(sorted(pest_ids), device=DEVICE))).any())

    other = sorted((ROOT / "data" / "plantdoc_det" / "images" / "test").iterdir())
    tea_dir = ROOT / "data" / "tea_pest" / "yolo"
    tea_pest = [p for p in sorted((tea_dir / "images" / "test").iterdir())
                if any(int(line.split()[0]) + 1 in pest_ids
                       for line in (tea_dir / "labels" / "test" / (p.stem + ".txt")).read_text().splitlines() if line.strip())]
    fp = sum(has_pest(p) for p in other)
    tp = sum(has_pest(p) for p in tea_pest)
    return {"threshold": threshold, "other_crops_photos": len(other), "other_crops_with_false_pest": fp,
            "false_pest_rate_other_crops": round(fp / len(other), 4), "tea_photos_with_pest": len(tea_pest),
            "tea_pest_photos_detected": tp, "pest_photo_recall_tea": round(tp / max(1, len(tea_pest)), 4)}


PROFILE_PHOTOS = ("potato_early_blight.JPG", "tomato_bacterial_spot.JPG", "corn_common_rust.JPG", "tomato-mold.jpg")

# Runs in a fresh process so its memory is the pipeline's alone (the evaluation process holds five classifiers).
_PIPELINE_CHILD = r"""
import json, sys
import numpy as np
root = sys.argv[1]
sys.path.insert(0, root)
from pathlib import Path
from surveillance.pipeline import Surveillance
s = Surveillance(device="cpu")
history = Path(root) / "demo" / "field_history.csv"
runs = []
for name in %r:
    for crop in (None, "tea"):  # crop=None times classify; crop="tea" times the pest model
        s.crop = crop
        for _ in range(3):
            report, _ = s.analyze(Path(root) / "test_images" / name, history_csv=history, name=name)
            runs.append(report.timings_ms)
keys = ["classify", "detect", "pests", "segment", "forecast"]
med = {k: round(float(np.median([r[k] for r in runs if k in r])), 1) for k in keys if any(k in r for r in runs)}
med["total (all five)"] = round(sum(med.values()), 1)
print("DONE " + json.dumps({"components": s.components, "median_ms_per_component": med}), flush=True)
sys.stdin.readline()
""" % (PROFILE_PHOTOS,)

# The Flask web app (both classifiers loaded), in its own environment, as a server would run it.
_WEBAPP_CHILD = r"""
import io, json, sys, time
from pathlib import Path
app_dir = Path(sys.argv[1]) / "Flask Deployed App"
sys.path.insert(0, str(app_dir))
from app import create_app
app = create_app()
app.config["RATE_LIMIT_PER_MINUTE"] = 0
client = app.test_client()
times = []
for name in %r:
    data = (Path(sys.argv[1]) / "test_images" / name).read_bytes()
    for _ in range(3):
        t = time.perf_counter()
        r = client.post("/api/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")
        times.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200, r.status_code
times.sort()
print("DONE " + json.dumps({"median_request_ms": round(times[len(times) // 2], 1)}), flush=True)
sys.stdin.readline()
""" % (PROFILE_PHOTOS,)


def _profile_child(python, code):
    """Run code in a new process; return its JSON result and its peak / current memory in MB (Windows: peak
    working set; elsewhere the current RSS at the end)."""
    import subprocess

    import psutil
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([str(python), "-c", code, str(ROOT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, env=env, cwd=str(ROOT))
    result = None
    for line in proc.stdout:
        if line.startswith("DONE "):
            result = json.loads(line[5:])
            break
    # A virtual-environment python.exe on Windows is a small launcher that runs the real interpreter as a child
    # process, so add up the whole process tree.
    root = psutil.Process(proc.pid)
    infos = [q.memory_info() for q in [root] + root.children(recursive=True)]
    peak = sum(getattr(i, "peak_wset", i.rss) for i in infos)
    rss = sum(i.rss for i in infos)
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait(timeout=60)
    if result is None:
        raise RuntimeError(f"profiling child failed (exit {proc.returncode})")
    result.update({"peak_memory_MB": round(peak / 2**20), "memory_MB_at_end": round(rss / 2**20)})
    return result


def full_pipeline_profile():
    """End-to-end CLI pipeline on a 4-thread CPU (all five components) and the web app, each in a fresh process."""
    out = {"pipeline": _profile_child(sys.executable, _PIPELINE_CHILD)}
    web_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if web_python.exists():
        out["web_app"] = _profile_child(web_python, _WEBAPP_CHILD)
    return out


def lab_subsets(splits):
    rng = np.random.default_rng(0)
    lab_test = [splits["test_lab"][i] for i in rng.choice(len(splits["test_lab"]), 2000, replace=False)]
    lab_val = [splits["val_lab"][i] for i in rng.choice(len(splits["val_lab"]), 1000, replace=False)]
    return lab_test, lab_val


def classification_stage(splits, lab_test, lab_val):
    """Accuracy, macro F1 and calibration of every classifier; returns (results, field reliability bins)."""
    bench, reliability = {}, {}
    for name, members in CLASSIFIERS.items():
        if not all((RUNS / r / "best.pt").exists() for r, _ in members):
            continue
        # Two temperatures, both fitted on validation only. The product sees field photos, so the one that counts
        # is fitted on field validation photos; the one fitted on mixed lab + field validation is kept to show
        # what happens when the calibration set does not match the deployment domain (lab photos dominate it).
        pv_field, yv_field = probs_for(members, splits["val_field"])
        pv_lab, yv_lab = probs_for(members, lab_val)
        t = fit_temperature(pv_field, yv_field)
        t_mixed = fit_temperature(torch.cat([pv_field, pv_lab]), torch.cat([yv_field, yv_lab]))
        t_lab = fit_temperature(pv_lab, yv_lab)
        row = {"temperature": round(t, 3), "temperature_fitted_on_mixed_validation": round(t_mixed, 3),
               "temperature_fitted_on_lab_validation": round(t_lab, 3)}
        for split, items in (("field", splits["test_field"]), ("lab", lab_test)):
            p, y = probs_for(members, items)
            m = metrics(y.tolist(), p.argmax(1).tolist())
            e_raw, rel = ece(p, y)
            e_cal, rel_cal = ece(apply_t(p, t if split == "field" else t_lab), y)
            e_mixed, _ = ece(apply_t(p, t_mixed), y)
            row[split] = {"accuracy": round(m["accuracy"], 4), "macro_F1": round(m["macro_f1"], 4),
                          "macro_recall": round(m["macro_recall"], 4), "ECE": round(e_raw, 4),
                          "ECE_after_temperature_scaling": round(e_cal, 4),
                          "ECE_with_mixed_validation_temperature": round(e_mixed, 4)}
            if split == "field":
                reliability[name] = (rel, rel_cal)
        bench[name] = row
        print(f"{name:30s} field acc {row['field']['accuracy']:.3f} F1 {row['field']['macro_F1']:.3f} ECE {row['field']['ECE']:.3f}"
              f" -> {row['field']['ECE_after_temperature_scaling']:.3f} (T={t:.2f}; mixed T={t_mixed:.2f} gives "
              f"{row['field']['ECE_with_mixed_validation_temperature']:.3f}); lab ECE {row['lab']['ECE']:.3f} -> "
              f"{row['lab']['ECE_after_temperature_scaling']:.3f} (T={t_lab:.2f})", flush=True)
    return bench, reliability


def plot_reliability(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bench, reliability = out["classification"], out["reliability_field"]
    fig, axes = plt.subplots(1, len(reliability), figsize=(3.4 * len(reliability), 3.6), sharey=True)
    for ax, (name, (rel, rel_cal)) in zip(np.atleast_1d(axes), reliability.items()):
        ax.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--")
        for r, colour, lab in ((rel, "#2a78d6", "raw"), (rel_cal, "#eb6834", "temperature-scaled (field-fitted T)")):
            ax.plot([c for _, _, c, _ in r], [a for _, a, _, _ in r], "o-", color=colour, lw=1.8, ms=4, label=lab)
        ax.set_title(f"{name}\nECE {bench[name]['field']['ECE']:.3f} -> {bench[name]['field']['ECE_after_temperature_scaling']:.3f}",
                     fontsize=8, loc="left")
        ax.set_xlabel("confidence")
    np.atleast_1d(axes)[0].set_ylabel("accuracy (field test)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Reliability diagrams on real field photos (perfect calibration = dashed line)", x=0.01, y=1.04,
                 ha="left", fontsize=10)
    fig.savefig(RESULTS / "eval_reliability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    splits = load_splits()
    lab_test, lab_val = lab_subsets(splits)
    out = {"note": "lab = 2,000 random PlantVillage test photos (leaf-grouped split); field = all 225 PlantDoc test photos"}

    # 1. Classification benchmark + calibration
    out["classification"], out["reliability_field"] = classification_stage(splits, lab_test, lab_val)
    plot_reliability(out)

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

    # 4. Out-of-domain behaviour of the pest model
    out["pest_out_of_domain"] = pest_out_of_domain()
    out["pest_out_of_domain_at_tuned_threshold"] = pest_out_of_domain(pipeline_threshold("tea_retinanet"))
    print("pest model out of domain:", out["pest_out_of_domain"], flush=True)

    # 5. Deployment profile
    out["deployment"] = deployment()
    out["pipeline_cpu_profile"] = full_pipeline_profile()
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=1))
    # 6. Confusion matrices and failure cases (surveillance/failure_cases.py)
    from surveillance import failure_cases
    failure_cases.main()
    write_summary(out)
    print(json.dumps(out["pipeline_cpu_profile"], indent=1))


def write_summary(out):
    lines = ["# Evaluation summary (generated by surveillance/evaluate_all.py)", "",
             "## Classification (39 classes)", "",
             "Temperatures are fitted on validation only: T(field) on field validation photos (used for field test),",
             "T(lab) on lab validation photos, and T(mixed) on both together, to show the effect of a mismatched calibration set.", "",
             "| Model | Field acc | Field macro F1 | Field ECE raw → T(field) | Field ECE with T(mixed) | Lab acc | Lab macro F1 | Lab ECE raw → T(lab) | T(field) / T(mixed) / T(lab) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, r in out["classification"].items():
        f, lab = r["field"], r["lab"]
        lines.append(f"| {name} | {f['accuracy']:.2%} | {f['macro_F1']:.3f} | {f['ECE']:.3f} → {f['ECE_after_temperature_scaling']:.3f} | "
                     f"{f['ECE_with_mixed_validation_temperature']:.3f} | {lab['accuracy']:.2%} | {lab['macro_F1']:.3f} | "
                     f"{lab['ECE']:.3f} → {lab['ECE_after_temperature_scaling']:.3f} | {r['temperature']} / "
                     f"{r['temperature_fitted_on_mixed_validation']} / {r['temperature_fitted_on_lab_validation']} |")
    lines += ["", "## Robustness of the shipped ensemble (field test)", "", "| Corruption | Level | Accuracy | Macro F1 |", "|---|---|---:|---:|"]
    lines += [f"| {r['corruption']} | {r['level']} | {r['accuracy']:.2%} | {r['macro_F1']:.3f} |" for r in out["robustness"]]
    lines += ["", "## Detection (COCO mAP on the untouched test split)", "", "| Run | mAP@[.5:.95] | mAP@.5 | Pest mAP | CPU ms | Params (M) |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in out["detection"].items():
        lines.append(f"| {name} | {m['mAP']:.3f} | {m['mAP50']:.3f} | {m.get('pest_mAP') if m.get('pest_mAP') is not None else '-'} | "
                     f"{m.get('cpu_latency_ms')} | {m.get('parameters_M')} |")
    po = out.get("pest_out_of_domain")
    if po:
        lines += ["", "## Pest model outside its training crop", "",
                  f"Photos of 13 other crops (PlantDoc test, no tea pests): {po['false_pest_rate_other_crops']:.0%} get a false pest "
                  f"box at score ≥ {po['threshold']} ({po['other_crops_with_false_pest']}/{po['other_crops_photos']}). "
                  f"Tea test photos that contain a pest: {po['pest_photo_recall_tea']:.0%} detected "
                  f"({po['tea_pest_photos_detected']}/{po['tea_photos_with_pest']}). Hence pest detection runs only for crop = tea."]
        pt = out.get("pest_out_of_domain_at_tuned_threshold")
        if pt:
            lines += ["", f"At the validation-tuned threshold {pt['threshold']}: {pt['false_pest_rate_other_crops']:.0%} false alarms on "
                          f"other crops ({pt['other_crops_with_false_pest']}/{pt['other_crops_photos']}), "
                          f"{pt['pest_photo_recall_tea']:.0%} of tea pest photos caught. Still far too many false alarms: the crop gate stays."]
    lines += ["", "## Segmentation and instance analysis (LeafNet75 test)", ""]
    for name, m in out["segmentation"].items():
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    p = out["pipeline_cpu_profile"]["pipeline"]
    lines += ["", "## Full pipeline on CPU (4 threads, fresh process)", "", f"Components: {', '.join(p['components'])}.", "",
              "| Component | Median ms |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in p["median_ms_per_component"].items()]
    lines += ["", f"Pipeline process: peak memory {p['peak_memory_MB']} MB, {p['memory_MB_at_end']} MB at the end."]
    w = out["pipeline_cpu_profile"].get("web_app")
    if w:
        lines += [f"Web app (Flask, both classifiers, `/api/predict`): median request {w['median_request_ms']} ms, "
                  f"peak memory {w['peak_memory_MB']} MB."]
    fc_path = RESULTS / "failure_cases.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text())
        lines += ["", "## Confusion matrices", "",
                  "Figures: `confusion_field_ensemble.png` (shipped ensemble, field test) and `confusion_mlp_baseline.png`.", "",
                  "| Classifier (field test) | Accuracy | Most confused pairs (true → predicted, photos) |", "|---|---:|---|"]
        for name, r in fc["confusion"]["classifiers_field"].items():
            pairs = "; ".join(f"{q['true']} → {q['predicted']} ({q['photos']})" for q in r["most_confused_pairs"][:3])
            lines.append(f"| {name} | {r['field_accuracy']:.2%} | {pairs} |")
        (tn, fp), (fn, tp) = fc["confusion"]["segmentation_pixels"]
        lines += ["", f"DeepLabV3 pixels on the lesion test set: lesion recall {tp / (tp + fn):.1%}, lesion precision "
                      f"{tp / (tp + fp):.1%}, background kept {tn / (tn + fp):.1%}."]
        c, sg, d = fc["failures"]["classification"], fc["failures"]["segmentation"], fc["failures"]["detection"]
        lines += ["", "## Failure cases (chosen by rule on the test sets)", "",
                  f"- Classification: {c['mistakes']} mistakes in 225 field photos, {c['mistakes_at_80pct_or_more_confidence']} of them "
                  "made at ≥ 80% confidence. Most confident mistakes:"]
        lines += [f"  - `{Path(r['photo']).name}`: true {r['true']}, predicted {r['predicted']} ({r['confidence']:.0%})" for r in c["shown"]]
        lines += ["- Segmentation, lowest lesion IoU: " + ", ".join(f"{w['image']} ({w['IoU']:.2f})" for w in sg["worst"]),
                  f"- Detection: RetinaNet finds none of the true boxes on {d['photos_with_zero_recall']} of {d['photos_scored']} "
                  "PlantDoc test photos; lowest recall: " + ", ".join(f"`{w['image']}`" for w in d["lowest_recall"]),
                  "- Tea pest model on other crops, strongest false alarms: "
                  + ", ".join(f"`{w['image']}` ({w['max_pest_score']:.0%})" for w in d["false_pest_alarms"]),
                  "", "Photo galleries: `python -m surveillance.failure_cases` writes them to `surveillance/results/local_only/` "
                  "(not committed: the photos are web-sourced, some watermarked)."]
    (RESULTS / "EVALUATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if "--classification-only" in sys.argv:  # re-run the classifier benchmark, keep every other result
        saved = json.loads((RESULTS / "evaluation.json").read_text())
        sp = load_splits()
        saved["classification"], saved["reliability_field"] = classification_stage(sp, *lab_subsets(sp))
        plot_reliability(saved)
        (RESULTS / "evaluation.json").write_text(json.dumps(saved, indent=1))
        write_summary(saved)
    elif "--pests-only" in sys.argv:  # re-measure the pest model on other crops, keep every other result
        saved = json.loads((RESULTS / "evaluation.json").read_text())
        saved["pest_out_of_domain_at_tuned_threshold"] = pest_out_of_domain(pipeline_threshold("tea_retinanet"))
        (RESULTS / "evaluation.json").write_text(json.dumps(saved, indent=1))
        write_summary(saved)
        print(saved["pest_out_of_domain_at_tuned_threshold"])
    elif "--summary-only" in sys.argv:  # rewrite EVALUATION_SUMMARY.md from the saved results
        write_summary(json.loads((RESULTS / "evaluation.json").read_text()))
    elif "--profile-only" in sys.argv:  # re-measure latency and memory, keep every other result
        saved = json.loads((RESULTS / "evaluation.json").read_text())
        saved["pipeline_cpu_profile"] = full_pipeline_profile()
        (RESULTS / "evaluation.json").write_text(json.dumps(saved, indent=1))
        write_summary(saved)
        print(json.dumps(saved["pipeline_cpu_profile"], indent=1))
    else:
        main()

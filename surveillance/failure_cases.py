"""Confusion matrices and failure cases (part of the evaluation package, brief item 10).

    python -m surveillance.failure_cases

1. Confusion matrices: every classifier on the 225 field test photos (the shipped ensemble drawn in full, the
   most-confused pairs of all five tabulated), the NumPy MLP baseline (8 classes) and DeepLab's pixel matrix.
2. Failure galleries, chosen by rule (not by hand) on the untouched test sets:
   - classification: the 12 most confident mistakes of the shipped ensemble on field photos;
   - segmentation: the 4 test photos with the lowest lesion IoU (among photos with >= 1% lesion);
   - detection: the 3 PlantDoc photos where RetinaNet finds the fewest true boxes, and 3 photos of other crops
     on which the tea pest model raises a false alarm.
Writes surveillance/results/confusion_*.png and failure_cases.json (committed), and the photo galleries
surveillance/results/local_only/failures_*.png (git-ignored: many PlantDoc and LeafNet75 photos are web-sourced,
some are watermarked stock photos, so they are regenerated locally rather than redistributed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "surveillance" / "results"
GALLERY = RESULTS / "local_only"
sys.path.insert(0, str(ROOT / "training"))
from common import CLASSES, load_splits  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def short(name):
    crop, _, cond = name.partition("___")
    crop = crop.split(",")[0].split("_(")[0].replace("_", " ")
    cond = cond.replace("_", " ").replace("Cercospora leaf spot Gray leaf spot", "gray leaf spot")
    cond = cond.replace("Spider mites Two-spotted spider mite", "spider mites")
    return (f"{crop}: {cond}" if cond else crop)[:30]


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


def confusion(y, p, n):
    m = np.zeros((n, n), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def top_pairs(m, k=8):
    off = [(m[i, j], i, j) for i in range(len(m)) for j in range(len(m)) if i != j and m[i, j] > 0]
    return [{"true": short(CLASSES[i]), "predicted": short(CLASSES[j]), "photos": int(c)} for c, i, j in sorted(off, reverse=True)[:k]]


# ------------------------------------------------------------------ 1. confusion matrices
def classifier_confusions(splits):
    from surveillance.evaluate_all import CLASSIFIERS, probs_for
    out, matrices = {}, {}
    for name, members in CLASSIFIERS.items():
        p, y = probs_for(members, splits["test_field"])
        m = confusion(y.numpy(), p.argmax(1).numpy(), len(CLASSES))
        matrices[name] = m
        out[name] = {"most_confused_pairs": top_pairs(m), "field_accuracy": round(float(np.trace(m) / m.sum()), 4)}
        print(f"confusion: {name} acc {out[name]['field_accuracy']:.3f}", flush=True)
    m = matrices["CAFormer ensemble (shipped)"]
    present = [i for i in range(len(CLASSES)) if m[i].sum() > 0]
    sub = m[np.ix_(present, present)]
    norm = sub / sub.sum(1, keepdims=True)
    p = plt()
    fig, ax = p.subplots(figsize=(13, 11.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(present)):
        for j in range(len(present)):
            if sub[i, j]:
                ax.text(j, i, str(sub[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b")
    labels = [short(CLASSES[i]) for i in present]
    ax.set_xticks(range(len(present)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(present)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (PlantDoc field test, 225 photos)")
    ax.set_title("Shipped ensemble on real field photos: counts, shaded by share of the true class", loc="left", fontsize=11)
    fig.savefig(RESULTS / "confusion_field_ensemble.png", dpi=150, bbox_inches="tight")
    p.close(fig)
    return out


def mlp_confusion():
    b = json.loads((ROOT / "fundamentals" / "results" / "mlp_baseline.json").read_text())
    m = np.array(b["test_confusion"])
    labels = [short(c) for c in b["classes"]]
    p = plt()
    fig, ax = p.subplots(figsize=(6.4, 5.6))
    ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] / m[i].sum() > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(m)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(m)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (100 test photos per class)")
    ax.set_title(f"NumPy MLP baseline, test accuracy {b['test_accuracy']:.1%}", loc="left", fontsize=10)
    fig.savefig(RESULTS / "confusion_mlp_baseline.png", dpi=150, bbox_inches="tight")
    p.close(fig)
    return {"test_accuracy": b["test_accuracy"], "most_confused_pairs": [
        {"true": labels[i], "predicted": labels[j], "photos": int(m[i, j])}
        for c, i, j in sorted(((m[i, j], i, j) for i in range(len(m)) for j in range(len(m)) if i != j), reverse=True)[:5]]}


# ------------------------------------------------------------------ 2. failure galleries
def classification_failures(splits):
    z = np.load(ROOT / "training" / "runs" / "ensemble" / "test_predictions.npz")
    y, pred, conf = z["test_field_true"], z["test_field_pred"], z["test_field_conf"]
    wrong = [i for i in np.argsort(-conf) if pred[i] != y[i]][:12]
    p = plt()
    fig, axes = p.subplots(3, 4, figsize=(12, 10))
    rows = []
    for ax, i in zip(axes.flat, wrong):
        path = ROOT / "data" / splits["test_field"][i][0]
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            ax.imshow(im)
        ax.set_title(f"true: {short(CLASSES[y[i]])}\nsaid: {short(CLASSES[pred[i]])} ({conf[i]:.0%})", fontsize=8, loc="left")
        ax.axis("off")
        rows.append({"photo": splits["test_field"][i][0], "true": CLASSES[y[i]], "predicted": CLASSES[pred[i]], "confidence": round(float(conf[i]), 3)})
    fig.suptitle(f"The 12 most confident mistakes of the shipped ensemble on field photos "
                 f"({int((pred != y).sum())} mistakes in 225)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(GALLERY / "failures_classification.png", dpi=130, bbox_inches="tight")
    p.close(fig)
    over80 = int(((pred != y) & (conf >= 0.8)).sum())
    return {"mistakes": int((pred != y).sum()), "mistakes_at_80pct_or_more_confidence": over80, "shown": rows}


def segmentation_failures():
    from surveillance import segmentation as seg
    net = seg.build_semantic("deeplab", pretrained=False)
    state = torch.load(seg.RUNS / "leafseg_deeplab" / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    net.to(DEVICE).eval()
    ds = seg.LesionDataset("test")
    scores, pix = [], np.zeros((2, 2), np.int64)
    with torch.inference_mode():
        for i in range(len(ds)):
            x, m = ds[i]
            pr = net(x[None].to(DEVICE))["out"].argmax(1)[0].cpu().numpy()
            m = m.numpy()
            for a in (0, 1):
                for b in (0, 1):
                    pix[a, b] += int(((m == a) & (pr == b)).sum())
            inter, union = ((pr == 1) & (m == 1)).sum(), ((pr == 1) | (m == 1)).sum()
            if m.mean() >= 0.01:
                scores.append((inter / max(1, union), i, pr, m))
    scores.sort(key=lambda t: t[0])
    p = plt()
    fig, axes = p.subplots(4, 3, figsize=(9, 12))
    shown = []
    for row, (iou, i, pr, m) in zip(axes, scores[:4]):
        img, _ = ds.load(i)
        img = img.resize((seg.SIZE, seg.SIZE))
        for ax, im, t in zip(row, (img, m, pr), ("photo", "true damaged tissue", f"predicted (IoU {iou:.2f})")):
            ax.imshow(im, cmap=None if t == "photo" else "Reds", vmin=0, vmax=1)
            ax.set_title(t, fontsize=9, loc="left")
            ax.axis("off")
        shown.append({"image": ds.names[i], "IoU": round(float(iou), 3)})
    fig.suptitle("Segmentation: the 4 test photos with the lowest lesion IoU (DeepLabV3)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(GALLERY / "failures_segmentation.png", dpi=130, bbox_inches="tight")
    p.close(fig)
    return {"pixel_confusion_[true][pred]_background_lesion": pix.tolist(), "worst": shown}


def _boxes(ax, boxes, colour, labels=None):
    import matplotlib.patches as patches
    for k, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=1.6))
        if labels:
            ax.text(x1, y1, labels[k], color="white", fontsize=6, backgroundcolor=colour, va="bottom")


def detection_failures():
    from torchvision.ops import box_iou

    from surveillance import detection as det
    ds = det.YoloDetection("plantdoc", "test")
    net, names = det.load("plantdoc", "retinanet", DEVICE)
    pest_net, pest_names = det.load("tea", "retinanet", DEVICE)
    pest_ids = [i + 1 for i, n in enumerate(pest_names) if "(pest)" in n]
    recalls, false_pests = [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            img, t = ds[i]
            o = net([img.to(DEVICE)])[0]
            keep = o["scores"].cpu() >= 0.40
            pb, pl = o["boxes"].cpu()[keep], o["labels"].cpu()[keep]
            if len(t["boxes"]):
                ious = box_iou(t["boxes"], pb) if len(pb) else torch.zeros(len(t["boxes"]), 0)
                hit = [(ious[g] >= 0.5).logical_and(pl == t["labels"][g]).any().item() if len(pb) else False for g in range(len(t["boxes"]))]
                recalls.append((float(np.mean(hit)), -len(t["boxes"]), i, pb, pl))
            q = pest_net([img.to(DEVICE)])[0]
            sel = (q["scores"].cpu() >= 0.40) & torch.isin(q["labels"].cpu(), torch.tensor(pest_ids))
            if sel.any():
                false_pests.append((float(q["scores"].cpu()[sel].max()), i, q["boxes"].cpu()[sel], q["labels"].cpu()[sel], q["scores"].cpu()[sel]))
    recalls.sort(key=lambda r: (r[0], r[1]))
    false_pests.sort(key=lambda r: -r[0])
    p = plt()
    fig, axes = p.subplots(2, 3, figsize=(13, 8.6))
    shown = {"lowest_recall": [], "false_pest_alarms": []}
    for ax, (rec, _, i, pb, pl) in zip(axes[0], recalls[:3]):
        img, t = ds[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        _boxes(ax, t["boxes"], "#1f9d55", [names[int(c) - 1] for c in t["labels"]])
        _boxes(ax, pb, "#2a78d6", [names[int(c) - 1] for c in pl])
        ax.set_title(f"recall {rec:.0%}: green = truth, blue = RetinaNet", fontsize=8, loc="left")
        ax.axis("off")
        shown["lowest_recall"].append({"image": ds.images[i].name, "recall_at_IoU0.5": rec})
    for ax, (s, i, pb, pl, ps) in zip(axes[1], false_pests[:3]):
        img, t = ds[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        _boxes(ax, pb, "#eb6834", [f"{pest_names[int(c) - 1].split(' (')[0]} {v:.0%}" for c, v in zip(pl, ps)])
        truth = ", ".join(sorted({names[int(c) - 1] for c in t["labels"]})) or "no boxes"
        ax.set_title(f"false pest alarm (truth: {truth[:40]})", fontsize=8, loc="left")
        ax.axis("off")
        shown["false_pest_alarms"].append({"image": ds.images[i].name, "max_pest_score": round(s, 3)})
    fig.suptitle("Detection failures on PlantDoc test photos: top row, RetinaNet's lowest recall; bottom row, "
                 "the tea pest model on other crops", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(GALLERY / "failures_detection.png", dpi=130, bbox_inches="tight")
    p.close(fig)
    shown["photos_with_zero_recall"] = sum(r[0] == 0 for r in recalls)
    shown["photos_scored"] = len(recalls)
    return shown


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    splits = load_splits()
    out = {"confusion": {"classifiers_field": classifier_confusions(splits), "mlp_baseline": mlp_confusion()},
           "failures": {"classification": classification_failures(splits)}}
    seg = segmentation_failures()
    out["confusion"]["segmentation_pixels"] = seg.pop("pixel_confusion_[true][pred]_background_lesion")
    out["failures"]["segmentation"] = seg
    out["failures"]["detection"] = detection_failures()
    (RESULTS / "failure_cases.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["failures"]["classification"].items() if k != "shown"}))
    print("written:", sorted(p.name for p in RESULTS.glob("confusion_*.png")) + sorted(p.name for p in GALLERY.glob("failures_*.png")))


if __name__ == "__main__":
    main()

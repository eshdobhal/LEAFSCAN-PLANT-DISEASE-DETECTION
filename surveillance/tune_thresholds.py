"""Choose each detector's box-score threshold on the VALIDATION split (never the test split).

    python -m surveillance.tune_thresholds

For every threshold from 0.05 to 0.90 a box counts as correct when it has the right class and IoU >= 0.5 with
an unmatched true box (greedy, highest score first). The threshold with the best F1 is kept. Writes
surveillance/results/detection_thresholds.json, which surveillance/pipeline.py reads.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from surveillance import detection as det

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "surveillance" / "results" / "detection_thresholds.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = [round(t, 2) for t in np.arange(0.05, 0.91, 0.05)]


def counts(preds, targets, thr):
    tp = fp = fn = 0
    for o, t in zip(preds, targets):
        keep = o["scores"] >= thr
        boxes, labels = o["boxes"][keep], o["labels"][keep]
        used = torch.zeros(len(t["boxes"]), dtype=torch.bool)
        ious = box_iou(boxes, t["boxes"]) if len(boxes) and len(t["boxes"]) else None
        for k in range(len(boxes)):  # boxes come sorted by score
            hit = False
            if ious is not None:
                ok = (ious[k] >= 0.5) & (t["labels"] == labels[k]) & ~used
                if ok.any():
                    used[int(torch.argmax(torch.where(ok, ious[k], torch.zeros_like(ious[k]))))] = True
                    hit = True
            tp, fp = tp + hit, fp + (not hit)
        fn += int((~used).sum())
    return tp, fp, fn


@torch.inference_mode()
def tune(data, model):
    ds = det.YoloDetection(data, "val")
    net, _ = det.load(data, model, DEVICE)
    preds, targets = [], []
    for i in range(len(ds)):
        img, t = ds[i]
        o = net([img.to(DEVICE)])[0]
        preds.append({k: v.cpu() for k, v in o.items()})
        targets.append(t)
    rows = []
    for thr in GRID:
        tp, fp, fn = counts(preds, targets, thr)
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        rows.append({"threshold": thr, "precision": round(p, 4), "recall": round(r, 4), "F1": round(2 * p * r / max(1e-9, p + r), 4)})
    best = max(rows, key=lambda r: r["F1"])
    at40 = next(r for r in rows if r["threshold"] == 0.4)
    print(f"{data}_{model}: best threshold {best['threshold']} F1 {best['F1']} (P {best['precision']}, R {best['recall']}); "
          f"at 0.40: F1 {at40['F1']} (P {at40['precision']}, R {at40['recall']})", flush=True)
    return {"best": best, "at_0.40": at40, "sweep": rows, "validation_images": len(ds)}


def main():
    out = {f"{d}_{m}": tune(d, m) for d, m in (("plantdoc", "retinanet"), ("plantdoc", "fasterrcnn"), ("tea", "retinanet"))}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

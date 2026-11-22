"""Disease and pest detection (brief items 7 and 8): RetinaNet-ResNet50 (main), Faster R-CNN-ResNet50 and YOLO.

    python -m surveillance.detection train --data plantdoc --model retinanet --epochs 24
    python -m surveillance.detection train --data tea      --model fasterrcnn --epochs 24
    python -m surveillance.detection yolo  --data plantdoc --epochs 60
    python -m surveillance.detection eval  --data plantdoc --model retinanet

Datasets (YOLO-format folders, prepared by surveillance/prepare_data.py):
- plantdoc: 29 leaf/disease classes on 13 crops, real web photos (disease detection);
- tea:      6 classes incl. 2 pests (red spider mite, tea mosquito bug), real field photos (pest detection).
All models are scored with the same COCO mAP code (torchmetrics) on the same untouched test split.
"""
import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "surveillance" / "runs"
DATASETS = {
    "plantdoc": ROOT / "data" / "plantdoc_det",
    "tea": ROOT / "data" / "tea_pest" / "yolo",
}
PEST_CLASSES = {"tea": [2, 3]}  # indices of the pest classes, for a pest-only mAP


def class_names(name):
    base = DATASETS[name]
    if (base / "classes.txt").exists():
        return [c.strip() for c in (base / "classes.txt").read_text().splitlines() if c.strip()]
    import yaml  # PlantDoc-corrected ships data.yaml
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


class YoloDetection(Dataset):
    """images/<split>/*.jpg + labels/<split>/*.txt (class cx cy w h, normalised) -> torchvision targets.
    Labels are shifted by +1 because torchvision detectors reserve 0 for the background."""

    def __init__(self, name, split, train=False):
        base = DATASETS[name]
        self.images = sorted(p for p in (base / "images" / split).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.labels = base / "labels" / split
        self.train = train

    def __len__(self):
        return len(self.images)

    def read_boxes(self, path, w, h):
        txt = self.labels / (path.stem + ".txt")
        boxes, labels = [], []
        if txt.exists():
            for line in txt.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = line.split()[:5]
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                x1, y1, x2, y2 = max(0, cx - bw / 2), max(0, cy - bh / 2), min(w, cx + bw / 2), min(h, cy + bh / 2)
                if x2 - x1 > 1 and y2 - y1 > 1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c) + 1)
        return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64))

    def __getitem__(self, i):
        with Image.open(self.images[i]) as im:
            im = im.convert("RGB")
        w, h = im.size
        boxes, labels = self.read_boxes(self.images[i], w, h)
        img = F.to_tensor(im)
        if self.train:
            if random.random() < 0.5:  # horizontal flip, boxes mirrored too
                img = img.flip(-1)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
        return img, {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))


def build(model, num_classes, pretrained=True, min_size=512, max_size=853):
    """torchvision v2 detectors with COCO weights; the class-specific head is replaced (num_classes + background)."""
    import torchvision.models.detection as det
    if model == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        net = det.retinanet_resnet50_fpn_v2(weights=det.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                            min_size=min_size, max_size=max_size)
        net.head.classification_head = RetinaNetClassificationHead(
            256, net.head.classification_head.num_anchors, num_classes + 1, norm_layer=partial(nn.GroupNorm, 32))
    elif model == "fasterrcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        net = det.fasterrcnn_resnet50_fpn_v2(weights=det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                             min_size=min_size, max_size=max_size)
        net.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes + 1)
    else:
        raise ValueError(model)
    return net


@torch.inference_mode()
def predict(net, loader, device, score_threshold=0.0):
    net.eval()
    preds, targets = [], []
    for images, tgts in loader:
        out = net([im.to(device) for im in images])
        for o, t in zip(out, tgts):
            keep = o["scores"] >= score_threshold
            preds.append({k: v[keep].cpu() for k, v in o.items()})
            targets.append(t)
    return preds, targets


def coco_map(preds, targets, names, pest_ids=None):
    """COCO mAP@[.5:.95], mAP@.5, mAP@.75 and per-class AP (labels are 1-based in preds/targets)."""
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(preds, targets)
    r = metric.compute()
    per_class = {}
    classes = r["classes"].tolist() if r["classes"].ndim else [int(r["classes"])]
    for c, ap in zip(classes, r["map_per_class"].tolist() if r["map_per_class"].ndim else [float(r["map_per_class"])]):
        per_class[names[c - 1]] = round(ap, 4) if ap >= 0 else None
    out = {"mAP": round(float(r["map"]), 4), "mAP50": round(float(r["map_50"]), 4), "mAP75": round(float(r["map_75"]), 4),
           "mAR100": round(float(r["mar_100"]), 4), "AP_per_class": per_class}
    if pest_ids:
        pest = [per_class[names[i]] for i in pest_ids if per_class.get(names[i]) is not None]
        out["pest_mAP"] = round(float(np.mean(pest)), 4) if pest else None
        m50 = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=[0.5])
        m50.update(preds, targets)
        r50 = m50.compute()
        cls50 = dict(zip(r50["classes"].tolist(), r50["map_per_class"].tolist()))
        out["pest_AP50"] = {names[i]: round(cls50.get(i + 1, -1), 4) for i in pest_ids}
    return out


def train(args):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = class_names(args.data)
    net = build(args.model, len(names)).to(device)
    train_ds, val_ds = YoloDetection(args.data, "train", train=True), YoloDetection(args.data, "val")
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                        persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=args.workers, collate_fn=collate)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    warm = min(500, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out = RUNS / f"{args.data}_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    best, history = -1.0, []
    for epoch in range(args.epochs):
        net.train()
        start, losses = time.time(), []
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = sum(net(images, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
        preds, targets = predict(net, val_loader, device)
        m = coco_map(preds, targets, names)
        row = {"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4), "val_mAP": m["mAP"], "val_mAP50": m["mAP50"],
               "minutes": round((time.time() - start) / 60, 2)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if m["mAP"] > best:
            best = m["mAP"]
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val mAP {best:.4f}")


def load(data, model, device="cpu"):
    names = class_names(data)
    net = build(model, len(names), pretrained=False)
    state = torch.load(RUNS / f"{data}_{model}" / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return net.to(device).eval(), names


def latency(net, data, n=10):
    """Median CPU time per photo (the deployment target is a CPU server)."""
    torch.set_num_threads(4)
    net = net.to("cpu").eval()
    ds = YoloDetection(data, "test")
    times = []
    with torch.inference_mode():
        for i in range(n + 2):
            img = ds[i][0]
            t = time.perf_counter()
            net([img])
            times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times[2:])), 1)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, names = load(args.data, args.model, device)
    loader = DataLoader(YoloDetection(args.data, "test"), batch_size=4, num_workers=args.workers, collate_fn=collate)
    preds, targets = predict(net, loader, device)
    m = coco_map(preds, targets, names, PEST_CLASSES.get(args.data))
    m["cpu_latency_ms"] = latency(net, args.data)
    m["parameters_M"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 1)
    m["test_images"] = len(targets)
    (RUNS / f"{args.data}_{args.model}" / "test_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k != "AP_per_class"}, indent=1))


def yolo(args):
    """YOLO benchmark (ultralytics, AGPL-3.0: used for benchmarking only, not shipped in the app).
    Its test predictions are scored with the same torchmetrics COCO mAP as the torchvision models."""
    from ultralytics import YOLO
    names = class_names(args.data)
    base = DATASETS[args.data]
    yaml_path = RUNS / f"{args.data}_yolo" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f"path: {base.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                         f"names: {json.dumps({i: n for i, n in enumerate(names)})}\n")
    model = YOLO(args.weights)
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=640, batch=16, seed=0, project=str(RUNS),
                name=f"{args.data}_yolo_train", exist_ok=True, workers=args.workers, verbose=False, plots=False)
    best = YOLO(str(RUNS / f"{args.data}_yolo_train" / "weights" / "best.pt"))
    ds = YoloDetection(args.data, "test")
    preds, targets = [], []
    for i in range(len(ds)):
        r = best.predict(str(ds.images[i]), conf=0.001, iou=0.6, max_det=100, verbose=False, imgsz=640)[0]
        b = r.boxes
        preds.append({"boxes": b.xyxy.cpu(), "scores": b.conf.cpu(), "labels": b.cls.cpu().long() + 1})
        targets.append(ds[i][1])
    m = coco_map(preds, targets, names, PEST_CLASSES.get(args.data))
    t = []
    for i in range(12):
        s = time.perf_counter()
        best.predict(str(ds.images[i]), device="cpu", verbose=False, imgsz=640)
        t.append((time.perf_counter() - s) * 1000)
    m["cpu_latency_ms"] = round(float(np.median(t[2:])), 1)
    m["parameters_M"] = round(sum(p.numel() for p in best.model.parameters()) / 1e6, 1)
    m["test_images"] = len(targets)
    m["weights"] = args.weights
    (RUNS / f"{args.data}_yolo" / "test_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k != "AP_per_class"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "eval", "yolo"])
    ap.add_argument("--data", choices=list(DATASETS), required=True)
    ap.add_argument("--model", choices=["retinanet", "fasterrcnn"], default="retinanet")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weights", default="yolov8s.pt", help="YOLO starting weights (COCO)")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "yolo": yolo}[a.action](a)

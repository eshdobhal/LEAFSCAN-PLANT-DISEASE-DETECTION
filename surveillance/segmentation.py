"""Damaged-tissue segmentation, instance-level lesion analysis and severity (brief items 7 and 8).

    python -m surveillance.segmentation train --model deeplab --epochs 40
    python -m surveillance.segmentation train --model fcn --epochs 40
    python -m surveillance.segmentation train --model maskrcnn --epochs 30
    python -m surveillance.segmentation eval --model deeplab

Data: LeafNet75 in-the-wild leaf photos with binary lesion masks (data/leafseg, split by prepare_data.py).
- Semantic segmentation: DeepLabV3-ResNet50 and FCN-ResNet50 -> IoU, Dice, pixel accuracy.
- Instance-level: each connected lesion is an instance. Mask R-CNN-ResNet50 is trained on those
  instances (mask mAP); the semantic models are also scored by lesion counting and lesion-level F1.
- Severity: share of the leaf that is damaged, graded 0-5 (see SEVERITY_BINS), compared with the truth.
"""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "leafseg"
RUNS = ROOT / "surveillance" / "runs"
SIZE = 384
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MIN_LESION_PX = 20  # connected regions smaller than this are noise, not lesions
SEVERITY_BINS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.01]  # grade 0: <1% of leaf damaged ... grade 5: >=50%


def splits():
    return json.loads((DATA / "splits.json").read_text())


class LesionDataset(Dataset):
    def __init__(self, split, train=False, size=SIZE):
        self.names = splits()[split]
        self.train, self.size = train, size

    def __len__(self):
        return len(self.names)

    def load(self, i):
        img = Image.open(DATA / "images" / f"{self.names[i]}.jpg").convert("RGB")
        mask = Image.open(DATA / "masks" / f"{self.names[i]}.png").convert("L")
        return img, mask

    def __getitem__(self, i):
        img, mask = self.load(i)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        x = F.to_tensor(img)
        m = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        if self.train:
            if random.random() < 0.5:
                x, m = x.flip(-1), m.flip(-1)
            if random.random() < 0.5:
                x, m = x.flip(-2), m.flip(-2)
            x = F.adjust_brightness(x, random.uniform(0.8, 1.2))
            x = F.adjust_saturation(x, random.uniform(0.8, 1.2))
        return F.normalize(x, MEAN, STD), m


def build_semantic(model, pretrained=True):
    import torchvision.models.segmentation as seg
    # aux_loss=True always: training uses the auxiliary head, so the saved weights include it and a model
    # rebuilt without pretrained weights must have it too (torchvision leaves it out by default then).
    if model == "deeplab":
        net = seg.deeplabv3_resnet50(weights=seg.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                     aux_loss=True)
    elif model == "fcn":
        net = seg.fcn_resnet50(weights=seg.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                               aux_loss=True)
    else:
        raise ValueError(model)
    net.classifier[-1] = nn.Conv2d(net.classifier[-1].in_channels, 2, 1)  # background, lesion
    if net.aux_classifier is not None:
        net.aux_classifier[-1] = nn.Conv2d(net.aux_classifier[-1].in_channels, 2, 1)
    return net


# ------------------------------------------------------------------ instance-level helpers
def lesion_instances(mask: np.ndarray):
    """Connected lesion regions (8-connectivity) larger than MIN_LESION_PX -> list of boolean masks."""
    labels, n = ndimage.label(mask > 0, structure=np.ones((3, 3)))
    return [labels == k for k in range(1, n + 1) if (labels == k).sum() >= MIN_LESION_PX]


def lesion_f1(pred_mask, true_mask, iou=0.5):
    """Instance-level: a predicted lesion is a hit if it overlaps an unmatched true lesion with IoU >= 0.5."""
    p, t = lesion_instances(pred_mask), lesion_instances(true_mask)
    used, hits = set(), 0
    for pm in p:
        for j, tm in enumerate(t):
            if j in used:
                continue
            inter, union = (pm & tm).sum(), (pm | tm).sum()
            if union and inter / union >= iou:
                used.add(j)
                hits += 1
                break
    return hits, len(p), len(t)


def leaf_area(rgb: np.ndarray, lesion: np.ndarray) -> np.ndarray:
    """Approximate leaf pixels: green vegetation (excess-green index) plus the lesions themselves.
    The dataset has no leaf masks, so severity uses this estimate for both truth and prediction."""
    r, g, b = [rgb[..., i].astype(np.float32) / 255 for i in range(3)]
    exg = 2 * g - r - b
    return ((exg > 0.05) | (lesion > 0))


def severity(rgb, lesion):
    leaf = leaf_area(rgb, lesion)
    frac = float((lesion > 0).sum() / max(1, leaf.sum()))
    grade = int(np.digitize(frac, SEVERITY_BINS) - 1)
    return frac, grade


# ------------------------------------------------------------------ training (semantic)
def run_config(model, epochs):
    """Every setting of a segmentation run (the hyperparameters live in the code, so they are written out here)."""
    common = {"model": model, "epochs": epochs, "seed": 0, "image_size": SIZE, "normalise_mean": MEAN, "normalise_std": STD,
              "augmentation": "horizontal + vertical flips, brightness and saturation x0.8-1.2", "scheduler": "cosine annealing",
              "train_val_test": {k: len(v) for k, v in splits().items()}, "min_lesion_px": MIN_LESION_PX,
              "severity_bins": SEVERITY_BINS}
    if model == "maskrcnn":
        return {**common, "weights": "COCO (torchvision MaskRCNN_ResNet50_FPN_V2)", "batch": 4,
                "optimizer": "SGD lr 0.01 momentum 0.9 weight decay 1e-4", "instances": "connected lesion regions"}
    return {**common, "weights": "COCO with VOC labels (torchvision)", "batch": 8, "aux_loss": True,
            "optimizer": "AdamW lr 1e-4 weight decay 1e-4", "loss": "cross-entropy, class weights [1, 3]"}


def train_semantic(args):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda")
    net = build_semantic(args.model).to(device)
    loader = DataLoader(LesionDataset("train", True), batch_size=8, shuffle=True, num_workers=4, persistent_workers=True)
    val = DataLoader(LesionDataset("val"), batch_size=8, num_workers=4)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))  # lesions are ~12% of pixels
    out = RUNS / f"leafseg_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(run_config(args.model, args.epochs), indent=1))
    best, history = -1, []
    for epoch in range(args.epochs):
        net.train()
        t0, losses = time.time(), []
        for x, m in loader:
            x, m = x.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x)
                loss = loss_fn(o["out"].float(), m) + (0.4 * loss_fn(o["aux"].float(), m) if "aux" in o else 0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        iou = semantic_scores(net, val, device)["IoU_lesion"]
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4), "val_IoU": iou,
                        "minutes": round((time.time() - t0) / 60, 2)})
        print(json.dumps(history[-1]), flush=True)
        if iou > best:
            best = iou
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))


@torch.inference_mode()
def predict_masks(net, loader, device):
    net.eval()
    # Mixed precision only on the GPU (on a CPU-only run, even a disabled bf16 autocast probes CUDA and fails).
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    for x, m in loader:
        with amp:
            p = net(x.to(device))["out"].float().argmax(1).cpu()
        yield p.numpy(), m.numpy()


def semantic_scores(net, loader, device):
    inter = union = tp_fp = tp_fn = correct = total = 0
    for p, m in predict_masks(net, loader, device):
        inter += ((p == 1) & (m == 1)).sum()
        union += ((p == 1) | (m == 1)).sum()
        tp_fp += (p == 1).sum()
        tp_fn += (m == 1).sum()
        correct += (p == m).sum()
        total += m.size
    return {"IoU_lesion": round(float(inter / max(1, union)), 4), "Dice_lesion": round(float(2 * inter / max(1, tp_fp + tp_fn)), 4),
            "pixel_accuracy": round(float(correct / total), 4)}


def eval_semantic(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_semantic(args.model, pretrained=False)
    state = torch.load(RUNS / f"leafseg_{args.model}" / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    net.to(device)
    ds = LesionDataset("test")
    loader = DataLoader(ds, batch_size=8, num_workers=4)
    scores = semantic_scores(net, loader, device)
    # Instance-level and severity, image by image at the evaluation size.
    hits = n_pred = n_true = 0
    count_err, sev_true, sev_pred, frac_err, per_image_iou = [], [], [], [], []
    idx = 0
    for p, m in predict_masks(net, loader, device):
        for pi, mi in zip(p, m):
            h, npd, nt = lesion_f1(pi, mi)
            hits, n_pred, n_true = hits + h, n_pred + npd, n_true + nt
            count_err.append(abs(npd - nt))
            rgb = np.array(ds.load(idx)[0].resize((SIZE, SIZE)))
            ft, gt = severity(rgb, mi)
            fp, gp = severity(rgb, pi)
            sev_true.append(gt)
            sev_pred.append(gp)
            frac_err.append(abs(fp - ft))
            u = ((pi == 1) | (mi == 1)).sum()
            per_image_iou.append(float(((pi == 1) & (mi == 1)).sum() / u) if u else 1.0)
            idx += 1
    prec, rec = hits / max(1, n_pred), hits / max(1, n_true)
    sev_true, sev_pred = np.array(sev_true), np.array(sev_pred)
    scores.update({
        "per_image_IoU_median": round(float(np.median(per_image_iou)), 4),
        "lesion_precision": round(prec, 4), "lesion_recall": round(rec, 4),
        "lesion_F1_at_IoU0.5": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "lesion_count_MAE": round(float(np.mean(count_err)), 2),
        "severity_fraction_MAE": round(float(np.mean(frac_err)), 4),
        "severity_grade_exact": round(float((sev_true == sev_pred).mean()), 4),
        "severity_grade_within_1": round(float((np.abs(sev_true - sev_pred) <= 1).mean()), 4),
        "cpu_latency_ms": semantic_latency(net),
        "parameters_M": round(sum(q.numel() for q in net.parameters()) / 1e6, 1), "test_images": len(ds)})
    (RUNS / f"leafseg_{args.model}" / "test_metrics.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores, indent=1))


def semantic_latency(net, n=10):
    torch.set_num_threads(4)
    net = net.to("cpu").eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    times = []
    with torch.inference_mode():
        for _ in range(n + 2):
            t = time.perf_counter()
            net(x)
            times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times[2:])), 1)


# ------------------------------------------------------------------ Mask R-CNN (instance segmentation)
class LesionInstances(LesionDataset):
    """Same photos; each connected lesion becomes one instance (box + mask) for Mask R-CNN."""

    def __getitem__(self, i):
        img, mask = self.load(i)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        m = np.array(mask.resize((self.size, self.size), Image.NEAREST)) > 127
        x = F.to_tensor(img)
        if self.train and random.random() < 0.5:
            x, m = x.flip(-1), m[:, ::-1].copy()
        inst = lesion_instances(m)
        boxes = []
        for im in inst:
            ys, xs = np.where(im)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(inst), dtype=torch.int64),
                  "masks": torch.from_numpy(np.stack(inst).astype(np.uint8)) if inst else torch.zeros((0, self.size, self.size), dtype=torch.uint8)}
        return x, target


def build_maskrcnn(pretrained=True):
    import torchvision.models.detection as det
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    net = det.maskrcnn_resnet50_fpn_v2(weights=det.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                       min_size=SIZE, max_size=SIZE, box_detections_per_img=200)
    net.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    net.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return net


def collate(batch):
    return tuple(zip(*batch))


def train_maskrcnn(args):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda")
    net = build_maskrcnn().to(device)
    loader = DataLoader(LesionInstances("train", True), batch_size=4, shuffle=True, num_workers=4, collate_fn=collate,
                        persistent_workers=True)
    val = DataLoader(LesionInstances("val"), batch_size=4, num_workers=4, collate_fn=collate)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    out = RUNS / "leafseg_maskrcnn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(run_config("maskrcnn", args.epochs), indent=1))
    best, history = -1, []
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for x, t in loader:
            x = [i.to(device) for i in x]
            t = [{k: v.to(device) for k, v in d.items()} for d in t]
            loss = sum(net(x, t).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        m = maskrcnn_map(net, val, device)
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4), "val_mask_mAP": m["mask_mAP"]})
        print(json.dumps(history[-1]), flush=True)
        if m["mask_mAP"] > best:
            best = m["mask_mAP"]
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))


@torch.inference_mode()
def maskrcnn_map(net, loader, device):
    from torchmetrics.detection import MeanAveragePrecision
    net.eval()
    seg_metric = MeanAveragePrecision(iou_type="segm")
    box_metric = MeanAveragePrecision(iou_type="bbox")
    for x, t in loader:
        out = net([i.to(device) for i in x])
        preds = [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu(),
                  "masks": (o["masks"][:, 0] > 0.5).cpu()} for o in out]
        targets = [{"boxes": d["boxes"], "labels": d["labels"], "masks": d["masks"].bool()} for d in t]
        seg_metric.update(preds, targets)
        box_metric.update([{k: v for k, v in p.items() if k != "masks"} for p in preds],
                          [{k: v for k, v in d.items() if k != "masks"} for d in targets])
    s, b = seg_metric.compute(), box_metric.compute()
    return {"mask_mAP": round(float(s["map"]), 4), "mask_mAP50": round(float(s["map_50"]), 4),
            "box_mAP": round(float(b["map"]), 4), "box_mAP50": round(float(b["map_50"]), 4)}


def eval_maskrcnn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_maskrcnn(pretrained=False)
    state = torch.load(RUNS / "leafseg_maskrcnn" / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    net.to(device)
    m = maskrcnn_map(net, DataLoader(LesionInstances("test"), batch_size=4, num_workers=4, collate_fn=collate), device)
    torch.set_num_threads(4)
    net = net.cpu().eval()
    x = [torch.rand(3, SIZE, SIZE)]
    times = []
    with torch.inference_mode():
        for _ in range(12):
            t0 = time.perf_counter()
            net(x)
            times.append((time.perf_counter() - t0) * 1000)
    m["cpu_latency_ms"] = round(float(np.median(times[2:])), 1)
    m["parameters_M"] = round(sum(q.numel() for q in net.parameters()) / 1e6, 1)
    (RUNS / "leafseg_maskrcnn" / "test_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "eval"])
    ap.add_argument("--model", choices=["deeplab", "fcn", "maskrcnn"], default="deeplab")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    if a.model == "maskrcnn":
        (train_maskrcnn if a.action == "train" else eval_maskrcnn)(a)
    else:
        (train_semantic if a.action == "train" else eval_semantic)(a)

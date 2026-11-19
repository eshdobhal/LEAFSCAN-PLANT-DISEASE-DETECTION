"""Shared pieces for training and evaluating the LeafScan hybrid model."""
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits.json"
RUNS = ROOT / "training" / "runs"
sys.path.insert(0, str(ROOT / "Flask Deployed App"))
from CNN import idx_to_classes  # noqa: E402  (the app's 39 classes, in the app's order)

CLASSES = [idx_to_classes[i] for i in range(len(idx_to_classes))]
ARCH = "caformer_s18"                              # hybrid: conv stages + self-attention stages
PRETRAINED = "caformer_s18.sail_in22k_ft_in1k"     # ImageNet-22k pre-training, fine-tuned on 1k
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = int(os.environ.get("TRAIN_SIZE", "224"))  # input side in pixels (224 or 288)


def eval_transform(size: int):
    """Inference transform: identical in evaluate.py and in the Flask app (model.py).
    Short side -> size * 256/224 (256 for 224, 329 for 288), centre crop size x size."""
    return transforms.Compose([
        transforms.Resize(round(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


EVAL_TF = eval_transform(SIZE)

# Training transform: heavy augmentation so lab photos look more like field photos
# (different framing, lighting, blur, orientation, partial occlusion).
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])


class LeafDataset(Dataset):
    """items: list of (image path relative to data/, class index, domain) with domain "lab" or "field"."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        with Image.open(DATA / path) as img:
            return self.transform(img.convert("RGB")), label


def load_splits():
    return json.loads(SPLITS.read_text(encoding="utf-8"))


def build_model(pretrained: bool):
    import timm
    return timm.create_model(PRETRAINED if pretrained else ARCH, pretrained=pretrained, num_classes=len(CLASSES))


def metrics(y_true, y_pred, labels=None):
    """Accuracy, macro precision/recall/F1 and per-class recall, over the classes present in y_true."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = sorted(set(y_true)) if labels is None else labels
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "per_class_recall": {CLASSES[c]: float(rc) for c, rc in zip(labels, r)},
        "per_class_support": {CLASSES[c]: int(s) for c, s in zip(labels, support)},
    }


@torch.inference_mode()
def predict_probs(model, loader, device, tta=True):
    """Softmax probabilities; with tta, the average of the image and its mirror image."""
    model.eval()
    out, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).float()
            probs = logits.softmax(1)
            if tta:
                probs = (probs + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        out.append(probs.cpu())
        labels.append(y)
    return torch.cat(out), torch.cat(labels)

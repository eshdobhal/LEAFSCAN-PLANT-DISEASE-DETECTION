"""A small, leakage-safe leaf dataset for the fundamentals experiments.

Uses the same leaf-grouped PlantVillage splits as the main model (data/splits.json), so no physical
leaf appears in both training and test. Photos are shrunk to 24 x 24 RGB (1,728 numbers each): small
enough for a NumPy MLP on a CPU, still enough to learn something about colour and texture.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "fundamentals_leaves.npz"
CLASSES = ["Apple___Apple_scab", "Apple___healthy", "Corn___Common_rust", "Grape___Black_rot",
           "Potato___Early_blight", "Potato___Late_blight", "Tomato___Bacterial_spot", "Tomato___healthy"]
PER_CLASS = {"train": 250, "val_lab": 60, "test_lab": 100}
SIZE = 24


def _load_split(items, name_of, split, rng):
    xs, ys = [], []
    for c, cls in enumerate(CLASSES):
        paths = [p for p, label, dom in items if dom == "lab" and name_of[label] == cls]
        for p in rng.choice(paths, min(PER_CLASS[split], len(paths)), replace=False):
            with Image.open(DATA / p) as im:
                xs.append(np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8))
            ys.append(c)
    return np.stack(xs), np.array(ys)


def load(seed=0):
    """Returns dict of (images uint8 N x 24 x 24 x 3, labels) per split; cached after the first call."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {s: (z[f"{s}_x"], z[f"{s}_y"]) for s in PER_CLASS}
    import sys
    sys.path.insert(0, str(ROOT / "Flask Deployed App"))
    from CNN import idx_to_classes
    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    out = {s: _load_split(splits[s], idx_to_classes, s, rng) for s in PER_CLASS}
    np.savez_compressed(CACHE, **{f"{s}_{k}": v for s, (x, y) in out.items() for k, v in (("x", x), ("y", y))})
    return out


def features(images, mean=None, std=None):
    """Flatten to vectors and standardise with the *training* mean/std (never the test set's)."""
    x = images.reshape(len(images), -1).astype(np.float64) / 255.0
    if mean is None:
        mean, std = x.mean(0), x.std(0) + 1e-6
    return (x - mean) / std, mean, std

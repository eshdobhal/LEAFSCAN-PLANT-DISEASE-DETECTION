"""Builds data/splits.json: train / validation / test lists for lab and field images.

    .venv-train/Scripts/python training/prepare_data.py

Sources (all under data/):
- hf_pv/       PlantVillage colour images + the official leaf-grouped 80/20 split lists
                (photos of the same physical leaf are never split between train and test).
- plantdoc/    PlantDoc field photos with its own train/ and test/ folders.
- mendeley/    the "Background_without_leaves" class (no leaf in the photo).

Every item is (path relative to data/, class index in the app's order, "lab" | "field").
"""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from common import CLASSES, DATA, SPLITS

INDEX = {name: i for i, name in enumerate(CLASSES)}
EXTS = {".jpg", ".jpeg", ".png"}

# PlantDoc folder -> app class. Keys are lower-cased with "_" turned into spaces.
PLANTDOC = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple leaf": "Apple___healthy",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry___healthy",
    "corn gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn___Northern_Leaf_Blight",
    "corn rust leaf": "Corn___Common_rust",
    "peach leaf": "Peach___healthy",
    "potato leaf": "Potato___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def pv_class(folder):
    """PlantVillage folder -> app class name ('Corn_(maize)___X' -> 'Corn___X')."""
    return re.sub(r"_\((maize|including_sour)\)", "", folder).rstrip("_ ")  # also 'Common_rust_' -> 'Common_rust'


def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def rel(path):
    return Path(path).relative_to(DATA).as_posix()


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def lab_items():
    files = {}  # "Class/filename" -> path; the zip may nest raw/color under a prefix
    for p in (DATA / "hf_pv").rglob("*"):
        if p.suffix.lower() in EXTS and "color" in p.parts:
            files[f"{p.parent.name}/{p.name}"] = p
    out = {}
    for split in ("train", "test"):
        items = []
        for line in (DATA / "hf_pv" / f"color_{split}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("/")
            if len(parts) < 2:
                continue
            path = files.get(f"{parts[-2]}/{parts[-1]}")
            if path is None:
                continue
            items.append((rel(path), INDEX[pv_class(parts[-2])], "lab"))
        out[split] = items
    missing = sum(1 for _ in (DATA / "hf_pv" / "color_train.txt").open()) - len(out["train"])
    print(f"PlantVillage: {len(out['train'])} train, {len(out['test'])} test ({missing} listed but missing)")
    return out["train"], out["test"]


def background_items(rng):
    paths = sorted(p for p in (DATA / "mendeley").rglob("*") if p.suffix.lower() in EXTS and readable(p))
    rng.shuffle(paths)
    items = [(rel(p), INDEX["Background_without_leaves"], "lab") for p in paths]
    n_test, n_val = len(items) // 5, len(items) // 10
    print(f"Background: {len(items)} images")
    return items[n_test + n_val:], items[n_test:n_test + n_val], items[:n_test]


def field_items(folder):
    items, unknown = [], Counter()
    for class_dir in sorted((DATA / "plantdoc" / folder).iterdir()):
        key = class_dir.name.lower().replace("_", " ").strip()
        if key not in PLANTDOC:
            unknown[class_dir.name] += 1
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in EXTS and readable(p):
                items.append((rel(p), INDEX[PLANTDOC[key]], "field"))
    if unknown:
        print("PlantDoc folders without a matching class (skipped):", list(unknown))
    return items


def main():
    rng = random.Random(0)
    lab_train, test_lab = lab_items()
    # Validation = 10% of the official training list (used only to pick the best epoch).
    rng.shuffle(lab_train)
    n_val = len(lab_train) // 10
    val_lab, lab_train = lab_train[:n_val], lab_train[n_val:]

    bg_train, bg_val, bg_test = background_items(rng)

    field_train = field_items("train")
    test_field = field_items("test")
    # PlantDoc is known to contain a few identical files in train and test: drop them from test.
    train_hashes = {md5(DATA / p) for p, _, _ in field_train}
    before = len(test_field)
    test_field = [it for it in test_field if md5(DATA / it[0]) not in train_hashes]
    print(f"PlantDoc: {len(field_train)} train, {len(test_field)} test ({before - len(test_field)} duplicates removed)")
    rng.shuffle(field_train)
    n_fval = len(field_train) * 15 // 100
    val_field, field_train = field_train[:n_fval], field_train[n_fval:]

    splits = {
        "train": lab_train + bg_train + field_train,
        "val_lab": val_lab + bg_val,
        "val_field": val_field,
        "test_lab": test_lab + bg_test,
        "test_field": test_field,
    }
    SPLITS.write_text(json.dumps(splits), encoding="utf-8")
    for name, items in splits.items():
        print(f"{name:10} {len(items):6d} images, {len({c for _, c, _ in items})} classes")


if __name__ == "__main__":
    main()

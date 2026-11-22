"""Prepare the detection and segmentation datasets (brief item 2: curation, de-duplication, leakage-safe splits).

    python -m surveillance.prepare_data

Inputs (downloaded into data/, git-ignored; provenance and licences in docs/DATASET_CARD.md):
- data/plantdoc_det: KeenForgeAI/PlantDoc-corrected (CC BY 4.0), YOLO format, official test split kept.
- data/tea_pest:     Project-AgML/tea_leaf_disease_pest_detection (CC BY 4.0), one parquet file.
- data/leafseg:      LeafNet75/Leaf-Disease-Segmentation-masks (Apache-2.0), one parquet file.

Outputs: YOLO-format folders for tea (data/tea_pest/yolo) and image/mask folders + a split file for the
lesion masks (data/leafseg/{images,masks}, data/leafseg/splits.json), plus a de-duplication report.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "surveillance" / "results" / "data_report.json"

# Tea: original ids -> kept classes. "Brown blight" (8 boxes) and "disease" (4 boxes) are too rare to
# learn or to measure (a test set would hold 0-2 of them), so those boxes are dropped and reported.
TEA_ORIGINAL = ["Black rot of tea", "Brown blight of tea", "Leaf rust of tea", "Red Spider infested tea leaf",
                "Tea Mosquito bug infested leaf", "Tea leaf", "White spot of tea", "disease"]
TEA_CLASSES = ["black rot", "leaf rust", "red spider mite (pest)", "tea mosquito bug (pest)", "healthy tea leaf",
               "white spot"]
TEA_MAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
TEA_PESTS = [2, 3]
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def prepare_tea(report):
    table = pq.read_table(DATA / "tea_pest" / "data" / "train-00000-of-00001.parquet").to_pylist()
    out = DATA / "tea_pest" / "yolo"
    seen, dup, dropped, kept = {}, 0, Counter(), Counter()
    split_of_hash = {}
    for i, row in enumerate(table):
        data = row["image"]["bytes"]
        h = md5(data)
        split = SPLITS[row["split"]]
        if h in split_of_hash:          # exact duplicate photo: keep only the first copy
            dup += 1
            if split_of_hash[h] != split:
                report.setdefault("tea_cross_split_duplicates", 0)
                report["tea_cross_split_duplicates"] += 1
            continue
        split_of_hash[h] = split
        im = Image.open(io.BytesIO(data))
        w, hgt = im.size
        lines = []
        for (x, y, bw, bh), c in zip(row["objects"]["bbox"], row["objects"]["categories"]):
            if c not in TEA_MAP:
                dropped[TEA_ORIGINAL[c]] += 1
                continue
            kept[TEA_CLASSES[TEA_MAP[c]]] += 1
            lines.append(f"{TEA_MAP[c]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / hgt:.6f} {bw / w:.6f} {bh / hgt:.6f}")
        stem = f"tea_{i:05d}"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "images" / split / f"{stem}.jpg").write_bytes(data)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
    (out / "classes.txt").write_text("\n".join(TEA_CLASSES))
    report["tea"] = {"images": len(table) - dup, "exact_duplicates_removed": dup,
                     "boxes_kept": dict(kept), "boxes_dropped_too_rare": dict(dropped),
                     "images_per_split": {s: len(list((out / "images" / s).glob("*.jpg"))) for s in SPLITS.values()}}


def prepare_leafseg(report, seed=0):
    table = pq.read_table(DATA / "leafseg" / "data" / "train-00000-of-00001.parquet").to_pylist()
    img_dir, mask_dir = DATA / "leafseg" / "images", DATA / "leafseg" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    seen, names, dup, fractions = set(), [], 0, []
    for row in table:
        data = row["image"]["bytes"]
        h = md5(data)
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        name = Path(row["image"]["path"]).stem
        Image.open(io.BytesIO(data)).convert("RGB").save(img_dir / f"{name}.jpg", quality=95)
        mask = (np.array(Image.open(io.BytesIO(row["mask"]["bytes"]))) > 0).astype(np.uint8) * 255
        Image.fromarray(mask).save(mask_dir / f"{name}.png")
        fractions.append(float((mask > 0).mean()))
        names.append(name)
    rng = np.random.default_rng(seed)
    order = rng.permutation(names).tolist()
    n_val, n_test = round(0.15 * len(order)), round(0.15 * len(order))
    splits = {"test": sorted(order[:n_test]), "val": sorted(order[n_test:n_test + n_val]),
              "train": sorted(order[n_test + n_val:])}
    (DATA / "leafseg" / "splits.json").write_text(json.dumps(splits, indent=1))
    report["leafseg"] = {"pairs": len(names), "exact_duplicates_removed": dup,
                         "split_sizes": {k: len(v) for k, v in splits.items()},
                         "lesion_fraction_mean": round(float(np.mean(fractions)), 4),
                         "lesion_fraction_median": round(float(np.median(fractions)), 4)}


def check_plantdoc(report):
    base = DATA / "plantdoc_det"
    classes = (base / "classes.txt").read_text().split("\n") if (base / "classes.txt").exists() else []
    hashes = {}
    for split in ("train", "val", "test"):
        for f in (base / "images" / split).glob("*"):
            hashes.setdefault(md5(f.read_bytes()), set()).add(split)
    cross = sum(1 for s in hashes.values() if len(s) > 1)
    report["plantdoc_detection"] = {"classes": len([c for c in classes if c.strip()]),
                                    "images_per_split": {s: len(list((base / "images" / s).glob("*")))
                                                         for s in ("train", "val", "test")},
                                    "photos_in_more_than_one_split": cross}


def main():
    report = {}
    prepare_tea(report)
    prepare_leafseg(report)
    check_plantdoc(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

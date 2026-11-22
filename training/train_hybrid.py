"""Fine-tune the CAFormer-S18 hybrid (CNN + Transformer) on PlantVillage + PlantDoc + background.

    .venv-train/Scripts/python training/train_hybrid.py --epochs 8

Picks the epoch with the best mean of lab and field macro-recall on the validation sets,
and saves it (fp16) to "Flask Deployed App/leafscan_hybrid.pt".
"""
import argparse
import json
import math
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import (CLASSES, EVAL_TF, ROOT, RUNS, SIZE, TRAIN_TF, LeafDataset, build_model, load_splits, metrics,
                    predict_probs)


def balanced_sampler(items, field_share):
    """Decides how often each image is drawn.

    - `field_share` of all draws are field (PlantDoc) photos, the rest lab photos (+ background),
      so the ~2,000 field photos are seen often without crowding out the 40,000 lab photos.
    - Within a domain, each image's weight is 1/sqrt(class size): rare classes are boosted
      (better recall) but big classes still contribute variety.
    """
    counts = Counter((label, domain) for _, label, domain in items)
    totals = Counter()
    for (label, domain), n in counts.items():
        totals[domain] += n ** 0.5  # sum over classes of count / sqrt(count)
    share = {"field": field_share, "lab": 1.0 - field_share}
    weights = [share[domain] / totals[domain] / counts[(label, domain)] ** 0.5 for _, label, domain in items]
    return WeightedRandomSampler(weights, num_samples=len(items), replacement=True)


def evaluate(model, items, device, batch, workers):
    loader = DataLoader(LeafDataset(items, EVAL_TF), batch_size=batch, num_workers=workers)
    probs, y = predict_probs(model, loader, device, tta=False)
    return metrics(y.tolist(), probs.argmax(1).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)  # fits a 6 GB GPU
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--field-share", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="caformer_s18")
    ap.add_argument("--arch", default="caformer_s18", help="caformer_s18 (shipped), resnet50 or googlenet")
    ap.add_argument("--no-copy", action="store_true", help="don't copy the best weights into the app")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    splits = load_splits()
    train_items = splits["train"]
    print(f"train {len(train_items)} | lab val {len(splits['val_lab'])} | field val {len(splits['val_field'])}")

    loader = DataLoader(LeafDataset(train_items, TRAIN_TF), batch_size=args.batch,
                        sampler=balanced_sampler(train_items, args.field_share),
                        num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)

    model = build_model(pretrained=True, arch=args.arch).to(device, memory_format=torch.channels_last)  # faster convs on GPU
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(loader)
    warmup = len(loader)  # one epoch of linear warm-up, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # softer targets -> better-calibrated confidence

    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "size": SIZE}, indent=1))
    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        start, total, seen = time.time(), 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)

        lab = evaluate(model, splits["val_lab"], device, args.batch * 2, args.workers)
        field = evaluate(model, splits["val_field"], device, args.batch * 2, args.workers)
        score = (lab["macro_recall"] + field["macro_recall"]) / 2
        row = {"epoch": epoch + 1, "loss": total / seen, "lab_acc": lab["accuracy"], "lab_recall": lab["macro_recall"],
               "field_acc": field["accuracy"], "field_recall": field["macro_recall"], "score": score,
               "minutes": (time.time() - start) / 60}
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if score > best:
            best = score
            # fp16 halves the file size (~52 MB); the app converts back to fp32 on load.
            state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save(state, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    print(f"best validation score {best:.4f}")
    if not args.no_copy:
        target = ROOT / "Flask Deployed App" / "leafscan_hybrid.pt"
        target.write_bytes((out_dir / "best.pt").read_bytes())
        print(f"weights copied to {target}")


if __name__ == "__main__":
    main()

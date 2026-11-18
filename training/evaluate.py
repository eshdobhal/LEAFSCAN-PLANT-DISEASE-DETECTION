"""Evaluate the trained hybrid on the held-out lab and field test sets (never seen in training).

    .venv-train/Scripts/python training/evaluate.py

Writes training/runs/<name>/test_metrics.json and prints a Markdown summary.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from common import CLASSES, EVAL_TF, ROOT, RUNS, LeafDataset, build_model, load_splits, metrics, predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="caformer_s18")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    state = torch.load(RUNS / args.name / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.to(device)

    splits = load_splits()
    results = {}
    for split in ("test_lab", "test_field"):
        loader = DataLoader(LeafDataset(splits[split], EVAL_TF), batch_size=96, num_workers=6)
        for tta in (False, True):
            probs, y = predict_probs(model, loader, device, tta=tta)
            conf, pred = probs.max(1)
            m = metrics(y.tolist(), pred.tolist())
            confident = conf >= args.threshold
            m["share_confident"] = float(confident.float().mean())
            m["accuracy_when_confident"] = float((pred[confident] == y[confident]).float().mean()) if confident.any() else None
            results[f"{split}{'_tta' if tta else ''}"] = m

    (RUNS / args.name / "test_metrics.json").write_text(json.dumps(results, indent=1))
    print("| Test set | Images | Accuracy | Macro recall | Macro precision | Macro F1 | >=80% confident | Accuracy when confident |")
    print("|---|---|---|---|---|---|---|---|")
    for key, m in results.items():
        print(f"| {key} | {m['n']} | {m['accuracy']:.2%} | {m['macro_recall']:.2%} | {m['macro_precision']:.2%} | "
              f"{m['macro_f1']:.2%} | {m['share_confident']:.0%} | {m['accuracy_when_confident']:.2%} |")
    field = results["test_field_tta"]["per_class_recall"]
    support = results["test_field_tta"]["per_class_support"]
    print("\nField recall per class (PlantDoc test, TTA):\n\n| Class | Images | Recall |\n|---|---|---|")
    for name in sorted(field, key=field.get):
        print(f"| {name} | {support[name]} | {field[name]:.0%} |")
    lab = results["test_lab_tta"]["per_class_recall"]
    print("\nLowest lab recall:", ", ".join(f"{n} {lab[n]:.1%}" for n in sorted(lab, key=lab.get)[:5]))


if __name__ == "__main__":
    main()

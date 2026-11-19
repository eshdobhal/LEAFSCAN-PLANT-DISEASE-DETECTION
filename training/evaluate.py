"""Evaluate trained hybrid(s) on the held-out lab and field test sets (never seen in training).

    .venv-train/Scripts/python training/evaluate.py                                   # 224 px model
    .venv-train/Scripts/python training/evaluate.py --runs caformer_s18_288:288 --name caformer_s18_288
    .venv-train/Scripts/python training/evaluate.py --runs caformer_s18:224 caformer_s18_288:288 --name ensemble

With several --runs, the models' probabilities are averaged (an ensemble), exactly as the app does.
Writes training/runs/<name>/test_metrics.json and prints a Markdown summary.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from common import RUNS, LeafDataset, build_model, eval_transform, load_splits, metrics, predict_probs


def load(run_name, device):
    model = build_model(pretrained=False)
    state = torch.load(RUNS / run_name / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["caformer_s18:224"], help="run_name:input_size, one or more")
    ap.add_argument("--name", default="caformer_s18", help="where to write test_metrics.json")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [(run.split(":")[0], int(run.split(":")[1])) for run in args.runs]
    splits = load_splits()
    results = {}
    for split in ("test_lab", "test_field"):
        for tta in (False, True):
            member_probs = []
            for run_name, size in members:
                model = load(run_name, device)
                loader = DataLoader(LeafDataset(splits[split], eval_transform(size)), batch_size=64, num_workers=4)
                probs, y = predict_probs(model, loader, device, tta=tta)
                member_probs.append(probs)
                del model
                torch.cuda.empty_cache()
            probs = torch.stack(member_probs).mean(0)  # ensemble = average of the members' probabilities
            conf, pred = probs.max(1)
            m = metrics(y.tolist(), pred.tolist())
            confident = conf >= args.threshold
            m["share_confident"] = float(confident.float().mean())
            m["accuracy_when_confident"] = float((pred[confident] == y[confident]).float().mean()) if confident.any() else None
            results[f"{split}{'_tta' if tta else ''}"] = m

    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps({"runs": args.runs, **results}, indent=1))
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


# The guard is required on Windows: DataLoader workers start by re-importing this file.
if __name__ == "__main__":
    main()

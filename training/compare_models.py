"""Compare the 224 px model, the 288 px model and their average (ensemble) on the VALIDATION sets.

    .venv-train/Scripts/python training/compare_models.py

The validation sets are used so that choosing between the three does not peek at the test sets.
Takes a few minutes on the GPU (each model scores ~4,800 images twice, original + mirror).
"""
import torch
from torch.utils.data import DataLoader

from common import RUNS, LeafDataset, build_model, eval_transform, load_splits, metrics, predict_probs

MODELS = [("224", "caformer_s18", 224), ("288", "caformer_s18_288", 288)]
SPLITS = ("val_field", "val_lab")
THRESHOLD = 0.80


def load(run_name, device):
    model = build_model(pretrained=False)
    state = torch.load(RUNS / run_name / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device)


def summary(probs, y):
    conf, pred = probs.max(1)
    m = metrics(y.tolist(), pred.tolist())
    sure = conf >= THRESHOLD
    acc_sure = (pred[sure] == y[sure]).float().mean().item() if sure.any() else float("nan")
    return (f"| {m['accuracy']:.2%} | {m['macro_recall']:.2%} | {m['macro_precision']:.2%} "
            f"| {sure.float().mean().item():.0%} | {acc_sure:.2%} |")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    print("| Split | Model | Accuracy | Macro recall | Macro precision | >=80% confident | Accuracy when confident |")
    print("|---|---|---|---|---|---|---|")
    for split in SPLITS:
        probs = {}
        for label, run_name, size in MODELS:
            model = load(run_name, device)
            loader = DataLoader(LeafDataset(splits[split], eval_transform(size)), batch_size=64, num_workers=4)
            probs[label], y = predict_probs(model, loader, device, tta=True)
            del model
            torch.cuda.empty_cache()
        probs["ensemble"] = (probs["224"] + probs["288"]) / 2
        for label, p in probs.items():
            print(f"| {split} | {label} {summary(p, y)}", flush=True)


# The guard is required on Windows: DataLoader workers start by re-importing this file.
if __name__ == "__main__":
    main()

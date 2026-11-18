"""Accuracy smoke test + CPU latency benchmark on test_images/.

    .venv/Scripts/python scripts/benchmark.py

Prints a Markdown table (filename -> predicted class -> confidence -> match) and the
median latency of 20 predictions. A "match" means the filename's words appear in the
predicted class name (filenames in test_images/ are named after their disease).
"""
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Flask Deployed App"))

from PIL import Image  # noqa: E402

from config import Config  # noqa: E402
from model import Predictor  # noqa: E402
from CNN import idx_to_classes  # noqa: E402

# Spelling mistakes in the test image filenames -> words used in the class names.
FIXES = {"ceder": "cedar", "northen": "northern", "starwberry": "strawberry", "soyaben": "soybean",
         "haunglongbing": "haunglongbing", "background": "background", "mold": "mold"}
IGNORE = {"jpg", "jpeg", "png", "2", "3", "leaf", "spotted", "two", "mites", "spot", "bell", "virus"}


def words(text):
    return {FIXES.get(w, w) for w in re.split(r"[^a-z]+", text.lower()) if w} - IGNORE


def main():
    predictor = Predictor(Config.MODEL_PATH)
    images = sorted(p for p in (ROOT / "test_images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Model loaded: {predictor.model_loaded}\n")
    print("| File | Predicted class | Confidence | Match |\n|---|---|---|---|")
    matches = 0
    for path in images:
        result = predictor.predict(Image.open(path))
        label = idx_to_classes[result["index"]]
        ok = words(path.stem) <= words(label)
        matches += ok
        print(f"| {path.name} | {label} | {result['confidence']:.1%} | {'yes' if ok else 'no'} |")
    print(f"\n{matches}/{len(images)} predictions match the filename.")

    image = Image.open(images[0])
    latencies = [predictor.predict(image)["latency_ms"] for _ in range(20)]
    print(f"Latency over 20 runs (preprocess + forward + softmax): median {statistics.median(latencies):.1f} ms, "
          f"min {min(latencies):.1f} ms, max {max(latencies):.1f} ms")


if __name__ == "__main__":
    main()

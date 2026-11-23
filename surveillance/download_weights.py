"""Download the trained surveillance weights from the GitHub release into the folders the code expects.

    python -m surveillance.download_weights            # the 5 models the CLI uses (~320 MB)
    python -m surveillance.download_weights --all      # also the benchmark models (~610 MB in total)

Every file is checked against its SHA-256 before it is moved into place; files already present and correct
are skipped. The YOLO benchmark weights are not distributed (AGPL-3.0; see THIRD_PARTY_NOTICES.md).
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/eshdobhal/LEAFSCAN-PLANT-DISEASE-DETECTION/releases/download/v1.1.0/"

# asset name -> (destination, sha256, used by the CLI)
WEIGHTS = {
    "plantdoc_retinanet.pt": ("surveillance/runs/plantdoc_retinanet/best.pt", "70820b59335190d6c3e3d2db6243d45fccec88722ef56a10531489a639cd9450", True),
    "plantdoc_fasterrcnn.pt": ("surveillance/runs/plantdoc_fasterrcnn/best.pt", "37110f35ae09aad31208aebd34f0b27e4c0bbda30c4f85f1c33def7f7737e2c1", True),
    "tea_retinanet.pt": ("surveillance/runs/tea_retinanet/best.pt", "a6687e45f105d80f7a7f0f82e747bb1d1972ed6fae63346416161625d1aac738", True),
    "leafseg_deeplab.pt": ("surveillance/runs/leafseg_deeplab/best.pt", "88867954a3e7f9abba60bdb448a79c83c77ccb9e17cedc8d625fd0db045f8db6", True),
    "progression.pt": ("surveillance/runs/progression.pt", "e770bafc4ed04bee05ac7077e4347d06f7c324477fdf5c0fc42b2c00cd0927ab", True),
    "tea_fasterrcnn.pt": ("surveillance/runs/tea_fasterrcnn/best.pt", "6f35e29e210b95bd4a0a608f91f7e8c4a5132958563a4f8bcb784561986ecf14", False),
    "leafseg_fcn.pt": ("surveillance/runs/leafseg_fcn/best.pt", "5132e5370c25c4d914a4c559553ad8547018ca2f80934cedec9ef6eca4e27e66", False),
    "leafseg_maskrcnn.pt": ("surveillance/runs/leafseg_maskrcnn/best.pt", "74b448e14d63f9aac8d1a066f658d34404af4408508cd3160dea62ed086e4fce", False),
    "generative.pt": ("surveillance/runs/generative.pt", "74d01fefc74dac1512ccbcc2560bdd4a0779d1a272ae2865732477d46adc27ed", False),
    "resnet50.pt": ("training/runs/resnet50/best.pt", "c1028f524e30f1ecfc0c5e5bba2487fe71c5ef088a89f61872c0750d06c62ad6", False),
    "googlenet.pt": ("training/runs/googlenet/best.pt", "36b8128b204afdb722392f5040e41cd783500e0dfe045060139d4e64f236a01f", False),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="also download the benchmark-only models")
    args = ap.parse_args(argv)
    failed = 0
    for asset, (dest, digest, for_cli) in WEIGHTS.items():
        if not (for_cli or args.all):
            continue
        target = ROOT / dest
        if target.exists() and sha256(target) == digest:
            print(f"ok       {dest}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        print(f"download {asset} -> {dest}", flush=True)
        try:
            urllib.request.urlretrieve(RELEASE + asset, part)
        except OSError as exc:
            print(f"  failed: {exc}")
            failed += 1
            continue
        if sha256(part) != digest:
            part.unlink()
            print("  failed: checksum mismatch, file discarded")
            failed += 1
            continue
        part.replace(target)
    print("done" if not failed else f"{failed} file(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

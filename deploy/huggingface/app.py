"""LeafScan on a free Hugging Face Space (Gradio SDK option; no Docker needed).

Hugging Face runs this file and publishes whatever listens on port 7860. It downloads a tagged release
of the GitHub repository (trained weights included) once, then serves the Flask app with waitress.
"""
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

REF = "v1.0.1"  # the GitHub release to run; change it to deploy a newer tag
URL = f"https://github.com/eshdobhal/LEAFSCAN-PLANT-DISEASE-DETECTION/archive/refs/tags/{REF}.zip"
SRC = Path(__file__).resolve().parent / f"leafscan-{REF}"

if not SRC.exists():
    print(f"Downloading LeafScan {REF} from GitHub...", flush=True)
    with urllib.request.urlopen(URL, timeout=600) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    members = [m for m in archive.namelist() if "/Flask Deployed App/" in m]  # only the web app
    archive.extractall(SRC, members)

app_dir = next(SRC.glob("*/Flask Deployed App"))

# Production settings for the Space (see the README's Configuration table).
os.environ.setdefault("REQUIRE_MODEL", "1")       # never start in demo mode
os.environ.setdefault("TRUST_PROXY", "1")         # Spaces sit behind an HTTPS proxy
os.environ.setdefault("FRAME_ANCESTORS", "https://huggingface.co")  # the Space page frames the app
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "60")

os.chdir(app_dir)
sys.path.insert(0, str(app_dir))

from waitress import serve  # noqa: E402

from app import create_app  # noqa: E402

serve(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), threads=4)

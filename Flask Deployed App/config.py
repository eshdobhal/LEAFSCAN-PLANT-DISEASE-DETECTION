"""All settings in one place. Values can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an env var like FLASK_DEBUG=1 / true / yes as a boolean."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_model_path() -> str:
    """The original repo used two different names for the same weights file.
    Use whichever one exists; fall back to the name the original app.py loaded."""
    for name in ("plant_disease_model_1_latest.pt", "plant_disease_model_1.pt"):
        if (BASE_DIR / name).is_file():
            return str(BASE_DIR / name)
    return str(BASE_DIR / "plant_disease_model_1_latest.pt")


class Config:
    # --- Branding (rename the app here) ---
    APP_NAME = "LeafScan"
    APP_TAGLINE = "Leaf disease diagnosis in seconds"
    REPO_URL = os.environ.get("REPO_URL", "https://github.com/manthan89-py/Plant-Disease-Detection")
    ORIGINAL_PROJECT_URL = "https://github.com/manthan89-py/Plant-Disease-Detection"

    # --- Model ---
    # "hybrid" = CAFormer-S18 (CNN + Transformer, trained by training/train_hybrid.py),
    # "cnn" = the original 39-class CNN, "auto" = hybrid if its weights exist, else cnn.
    MODEL_KIND = os.environ.get("MODEL_KIND", "auto").lower()
    HYBRID_MODEL_PATH = os.environ.get("HYBRID_MODEL_PATH", str(BASE_DIR / "leafscan_hybrid.pt"))  # 224 px
    # Second hybrid trained at 288 px; if its file exists and ENSEMBLE is on, both are averaged.
    HYBRID_288_MODEL_PATH = os.environ.get("HYBRID_288_MODEL_PATH", str(BASE_DIR / "leafscan_hybrid_288.pt"))
    ENSEMBLE = _env_bool("ENSEMBLE", default=True)
    # Test-time augmentation: also look at the mirror image and average (hybrid only; ~2x CPU time).
    TTA = _env_bool("TTA", default=True)
    MODEL_PATH = os.environ.get("MODEL_PATH", _default_model_path())
    MODEL_DOWNLOAD_URL = "https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link"
    # If true, refuse to start without real weights (use in production).
    REQUIRE_MODEL = _env_bool("REQUIRE_MODEL")
    # Below this top-1 probability the result page shows a "not sure" warning.
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.80"))

    # --- Data files ---
    DISEASE_CSV = BASE_DIR / "disease_info.csv"
    SUPPLEMENT_CSV = BASE_DIR / "supplement_info.csv"

    # --- Uploads ---
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # Flask rejects bigger requests with 413
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}  # what Pillow must detect inside the file
    PREVIEW_MAX_SIDE = 512  # px, size of the photo echoed back on the result page

    # --- Web ---
    DEBUG = _env_bool("FLASK_DEBUG")
    STATIC_VERSION = "1"  # bump to bust browser caches after changing CSS/JS
    SEND_FILE_MAX_AGE_DEFAULT = 60 * 60 * 24 * 365  # static files cached for 1 year

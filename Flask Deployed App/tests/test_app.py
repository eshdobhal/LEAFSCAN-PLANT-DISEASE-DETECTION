"""End-to-end tests for the LeafScan Flask app. Run: python -m pytest"""
import html
import io

import pytest
from PIL import Image

from catalog import load_catalog
from config import Config
from conftest import TEST_IMAGES

CATALOG = load_catalog(Config.DISEASE_CSV, Config.SUPPLEMENT_CSV)
BY_KEY = {c.key: c for c in CATALOG}

# (test image, expected model class)
SAMPLES = [
    ("Apple_scab.JPG", "Apple___Apple_scab"),
    ("corn_common_rust.JPG", "Corn___Common_rust"),
    ("grape_healthy.JPG", "Grape___healthy"),
    ("potato_late_blight.JPG", "Potato___Late_blight"),
    ("tomato_mosaic_virus.JPG", "Tomato___Tomato_mosaic_virus"),
]


def upload(path_or_bytes, filename="leaf.jpg"):
    data = path_or_bytes.read_bytes() if hasattr(path_or_bytes, "read_bytes") else path_or_bytes
    return {"image": (io.BytesIO(data), filename)}


def png_bytes(mode="RGBA", size=(300, 200)):
    buffer = io.BytesIO()
    Image.new(mode, size, "green").save(buffer, "PNG")
    return buffer.getvalue()


# --- Pages ---------------------------------------------------------------

@pytest.mark.parametrize("url", ["/", "/diagnose", "/treatments", "/about", "/healthz",
                                 "/robots.txt", "/favicon.ico", "/static/site.webmanifest"])
def test_pages_return_200(client, url):
    assert client.get(url).status_code == 200


@pytest.mark.parametrize("old, new", [("/index", "/diagnose"), ("/market", "/treatments"),
                                      ("/contact", "/about"), ("/submit", "/diagnose")])
def test_old_urls_redirect_permanently(client, old, new):
    response = client.get(old)
    assert response.status_code == 301
    assert response.headers["Location"].endswith(new)


def test_unknown_page_is_custom_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert b"Page not found" in response.data


def test_home_numbers_come_from_python(client):
    body = client.get("/").get_data(as_text=True)
    assert "14" in body and "Tomato" in body and "Blueberry" in body


def test_security_headers(client):
    headers = client.get("/").headers
    assert "script-src 'self' 'nonce-" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=(self)" in headers["Permissions-Policy"]
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]  # default: no framing


def test_frame_ancestors_is_configurable():
    """Hugging Face Spaces shows the app in a frame on huggingface.co, so that one site may be allowed."""
    from app import create_app

    class SpacesConfig(Config):
        FRAME_ANCESTORS = "https://huggingface.co"
        ENSEMBLE = False  # one model is enough here and loads faster

    csp = create_app(SpacesConfig).test_client().get("/").headers["Content-Security-Policy"]
    assert "frame-ancestors https://huggingface.co" in csp


def test_no_third_party_assets(client):
    """Every script, stylesheet and font must be served by us (no CDNs)."""
    for url in ["/", "/diagnose", "/treatments", "/about"]:
        body = client.get(url).get_data(as_text=True)
        assert "cdn" not in body.lower()
        assert "<script src=\"http" not in body


# --- Diagnose (HTML form) -----------------------------------------------

@pytest.mark.parametrize("filename, expected", SAMPLES)
def test_diagnose_post_returns_result(client, model_loaded, filename, expected):
    response = client.post("/diagnose", data=upload(TEST_IMAGES / filename, filename),
                           content_type="multipart/form-data")
    assert response.status_code == 200
    body = html.unescape(response.get_data(as_text=True))
    assert "Top 3 possibilities" in body
    assert "data:image/jpeg;base64," in body  # photo echoed back, not saved to disk
    if not model_loaded:
        pytest.skip("demo mode: model weights missing, so the predicted class is random")
    assert BY_KEY[expected].label in body


def test_old_submit_post_keeps_the_upload(client):
    response = client.post("/submit", data=upload(TEST_IMAGES / "Apple_scab.JPG"),
                           content_type="multipart/form-data")
    assert response.status_code == 307  # browser re-POSTs the same file to /diagnose
    assert response.headers["Location"].endswith("/diagnose")


def test_png_with_alpha_and_greyscale_are_accepted(client):
    for mode in ("RGBA", "L"):
        response = client.post("/diagnose", data=upload(png_bytes(mode), "leaf.png"),
                               content_type="multipart/form-data")
        assert response.status_code == 200, mode


def test_missing_file_is_handled(client):
    response = client.post("/diagnose", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Please choose a photo" in response.data


def test_empty_file_is_handled(client):
    response = client.post("/diagnose", data=upload(b"", "empty.jpg"), content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"empty" in response.data


def test_text_file_renamed_to_jpg_is_rejected(client):
    response = client.post("/diagnose", data=upload(b"just some text, not an image", "notes.jpg"),
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"couldn" in response.data


def test_wrong_extension_is_rejected(client):
    response = client.post("/diagnose", data=upload(png_bytes(), "leaf.gif"), content_type="multipart/form-data")
    assert response.status_code == 400


def test_path_traversal_filename_is_harmless(client):
    response = client.post("/diagnose", data=upload(png_bytes(), "../../app.py.png"),
                           content_type="multipart/form-data")
    assert response.status_code == 200


def test_file_over_8mb_is_413(client):
    response = client.post("/diagnose", data=upload(b"0" * (9 * 1024 * 1024), "big.jpg"),
                           content_type="multipart/form-data")
    assert response.status_code == 413
    assert b"Photo too large" in response.data


# --- JSON API ------------------------------------------------------------

def test_api_predict_schema(client, model_loaded):
    response = client.post("/api/predict", data=upload(TEST_IMAGES / "Apple_scab.JPG"),
                           content_type="multipart/form-data")
    assert response.status_code == 200
    data = response.get_json()
    assert set(data) >= {"label", "crop", "condition", "is_healthy", "is_background",
                         "confidence", "top3", "low_confidence"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top3"]) == 3
    assert data["top3"][0]["label"] == data["label"]
    assert data["top3"][0]["confidence"] >= data["top3"][1]["confidence"] >= data["top3"][2]["confidence"]
    assert isinstance(data["low_confidence"], bool)
    if model_loaded:
        assert data["label"] == "Apple___Apple_scab"


def test_api_errors_are_json(client):
    bad = client.post("/api/predict", data=upload(b"nope", "x.jpg"), content_type="multipart/form-data")
    assert bad.status_code == 400 and bad.get_json()["error"] == "invalid_image"
    too_big = client.post("/api/predict", data=upload(b"0" * (9 * 1024 * 1024)), content_type="multipart/form-data")
    assert too_big.status_code == 413 and too_big.is_json
    wrong_method = client.get("/api/predict")
    assert wrong_method.status_code == 405 and wrong_method.is_json


def test_healthz(client):
    data = client.get("/healthz").get_json()
    assert data["status"] == "ok" and isinstance(data["model_loaded"], bool)


# --- Catalog -------------------------------------------------------------

def test_catalog_flags_match_class_names():
    healthy = [c.index for c in CATALOG if c.is_healthy]
    assert healthy == [3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38]  # the indices the old templates hard-coded
    assert [c.index for c in CATALOG if c.is_background] == [4]
    assert all(isinstance(c.description, list) and c.description for c in CATALOG)
    assert "Huanglongbing (citrus greening)" == BY_KEY["Orange___Haunglongbing_(Citrus_greening)"].condition


# --- Sample photos, author credit and rate limiting ---

SAMPLE_EXPECTED = {
    "potato-early-blight": "Potato___Early_blight",
    "peach-bacterial-spot": "Peach___Bacterial_spot",
    "tomato-healthy": "Tomato___healthy",
    "tomato-leaf-curl": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def test_diagnose_page_offers_every_sample(client):
    body = client.get("/diagnose").get_data(as_text=True)
    for name in Config.SAMPLES:
        assert f'value="{name}"' in body
        assert client.get(f"/static/img/samples/{name}.jpg").status_code == 200


@pytest.mark.parametrize("name", list(SAMPLE_EXPECTED))
def test_sample_diagnosis(client, model_loaded, name, app):
    response = client.post("/diagnose/sample", data={"sample": name})
    assert response.status_code == 200
    body = html.unescape(response.get_data(as_text=True))
    assert "sample photo" in body
    if model_loaded:
        expected = next(c for c in load_catalog(Config.DISEASE_CSV, Config.SUPPLEMENT_CSV) if c.key == SAMPLE_EXPECTED[name])
        assert expected.condition in body


@pytest.mark.parametrize("name", ["", "unknown", "../../app", "../config"])
def test_unknown_sample_is_404(client, name):
    assert client.post("/diagnose/sample", data={"sample": name}).status_code == 404


def test_author_credit_in_footer(client):
    body = client.get("/").get_data(as_text=True)
    assert "Esh Dobhal" in body and 'name="author" content="Esh Dobhal"' in body


def test_rate_limit(app, client):
    app.config["RATE_LIMIT_PER_MINUTE"] = 2
    try:
        assert client.post("/api/predict").status_code == 400  # counted, even though it has no file
        assert client.post("/api/predict").status_code == 400
        limited = client.post("/api/predict")
        assert limited.status_code == 429 and limited.get_json()["error"] == "Too many requests"
        page = client.post("/diagnose")
        assert page.status_code == 429 and "Please wait" in page.get_data(as_text=True)
        assert client.get("/diagnose").status_code == 200  # pages are never limited
    finally:
        app.config["RATE_LIMIT_PER_MINUTE"] = 0


def test_rate_limiter_window_expires():
    import time

    from app import RateLimiter
    limiter = RateLimiter(window_seconds=0.05)
    assert limiter.allow("a", 1) and not limiter.allow("a", 1)
    assert limiter.allow("b", 1)  # other clients are counted separately
    time.sleep(0.06)
    assert limiter.allow("a", 1)


# --- Responsible use (requirement R13) -------------------------------------

def test_every_result_carries_the_disclaimer(client):
    from app import DISCLAIMER
    body = html.unescape(client.post("/diagnose", data=upload(TEST_IMAGES / "Apple_scab.JPG"),
                                     content_type="multipart/form-data").get_data(as_text=True))
    assert DISCLAIMER in body
    api = client.post("/api/predict", data=upload(TEST_IMAGES / "Apple_scab.JPG"), content_type="multipart/form-data")
    assert api.get_json()["notice"] == DISCLAIMER


def test_low_confidence_shows_not_sure_warning():
    from app import create_app

    class Strict(Config):
        CONFIDENCE_THRESHOLD = 1.01  # no prediction can reach it, so every result is "not sure"
        RATE_LIMIT_PER_MINUTE = 0
    client = create_app(Strict).test_client()
    body = html.unescape(client.post("/diagnose", data=upload(TEST_IMAGES / "Apple_scab.JPG"),
                                     content_type="multipart/form-data").get_data(as_text=True))
    assert "The model isn't sure." in body
    api = client.post("/api/predict", data=upload(TEST_IMAGES / "Apple_scab.JPG"), content_type="multipart/form-data")
    assert api.get_json()["low_confidence"] is True


def test_uploaded_photos_are_never_written_to_disk(client, tmp_path):
    import tempfile
    from pathlib import Path

    from conftest import APP_DIR

    def snapshot():
        files = {p for p in APP_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
        temp = {p for p in Path(tempfile.gettempdir()).iterdir() if p.is_file()}
        return files, temp
    before_app, before_tmp = snapshot()
    marker = (TEST_IMAGES / "Apple_scab.JPG").read_bytes()
    for _ in range(3):
        client.post("/diagnose", data=upload(marker), content_type="multipart/form-data")
        client.post("/api/predict", data=upload(marker), content_type="multipart/form-data")
    after_app, after_tmp = snapshot()
    assert after_app == before_app
    new_temp = [p for p in after_tmp - before_tmp if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", "")
                and p.stat().st_size >= len(marker) // 2]
    assert not new_temp, new_temp

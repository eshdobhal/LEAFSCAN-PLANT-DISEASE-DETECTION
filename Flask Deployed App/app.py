"""LeafScan web app: application factory and routes.

Run locally:   flask --app app run
Production:    gunicorn "app:create_app()"  (Linux)   |   waitress-serve --call app:create_app  (Windows)
"""
import base64
import io
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from catalog import crop_summary, image_hosts, load_catalog
from config import Config
from model import Predictor

log = logging.getLogger("leafscan")

# Refuse absurd pixel counts (a tiny file can still decode to gigabytes: a "decompression bomb").
Image.MAX_IMAGE_PIXELS = 40_000_000


class UploadError(ValueError):
    """A problem with the uploaded file that the user can fix. The message is shown to them."""


def read_upload(file, config) -> Image.Image:
    """Validate an uploaded file entirely in memory and return it as a PIL image.

    Nothing is written to disk and the user's filename is never used as a path;
    we only look at its extension.
    """
    if file is None or not file.filename:
        raise UploadError("Please choose a photo of a leaf first.")
    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in config["ALLOWED_EXTENSIONS"]:
        raise UploadError("That file type isn't supported. Please use a JPG, PNG or WebP photo.")
    data = file.read()
    if not data:
        raise UploadError("That file is empty. Please choose another photo.")
    try:
        Image.open(io.BytesIO(data)).verify()  # is it really an uncorrupted image?
        image = Image.open(io.BytesIO(data))   # verify() makes the object unusable, so open again
        if image.format not in config["ALLOWED_FORMATS"]:
            raise UploadError("That file type isn't supported. Please use a JPG, PNG or WebP photo.")
        image = ImageOps.exif_transpose(image)  # phone photos: apply the rotation stored in EXIF
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UploadError("We couldn't read that file as an image. Please try another photo.") from exc
    return image


def preview_data_url(image: Image.Image, max_side: int) -> str:
    """Small JPEG copy of the user's photo, embedded in the page instead of saved to disk."""
    thumb = image.convert("RGB")
    thumb.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    thumb.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def load_predictor(config) -> Predictor:
    """MODEL_KIND "auto" uses the hybrid when its weights file exists, otherwise the original CNN.
    The hybrid becomes a 224 px + 288 px ensemble when the 288 px weights exist and ENSEMBLE is on."""
    kind = config["MODEL_KIND"]
    if kind == "auto":
        kind = "hybrid" if Path(config["HYBRID_MODEL_PATH"]).is_file() else "cnn"
    if kind == "hybrid":
        members = [(config["HYBRID_MODEL_PATH"], 224)]
        if config["ENSEMBLE"] and Path(config["HYBRID_288_MODEL_PATH"]).is_file():
            members.append((config["HYBRID_288_MODEL_PATH"], 288))
    else:
        members = [(config["MODEL_PATH"], 224)]
    return Predictor(members, kind=kind, require_weights=config["REQUIRE_MODEL"], tta=config["TTA"])


class RateLimiter:
    """Sliding window: each client may make `limit` requests in any 60 seconds.

    Kept in memory, which is correct for this app because it runs as one process (see the Procfile).
    """

    def __init__(self, window_seconds: float = 60):
        self.window = window_seconds
        self.hits = defaultdict(deque)  # client IP -> times of its recent requests
        self.lock = threading.Lock()    # requests run on several threads at once

    def allow(self, client: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            recent = self.hits[client]
            while recent and now - recent[0] > self.window:  # forget requests older than the window
                recent.popleft()
            if len(recent) >= limit:
                return False
            recent.append(now)
            if len(self.hits) > 10_000:  # don't grow forever: drop clients with nothing recent
                for key in [k for k, q in self.hits.items() if not q or now - q[-1] > self.window]:
                    del self.hits[key]
            return True


ERROR_TEXT = {
    404: ("Page not found", "We couldn't find that page. It may have moved."),
    405: ("Not allowed", "That address doesn't accept this kind of request."),
    413: ("Photo too large", "Photos must be 8 MB or smaller. Try a smaller photo or a screenshot of it."),
    429: ("Too many requests", "You've diagnosed a lot of photos in the last minute. Please wait a moment and try again."),
    500: ("Something went wrong", "An unexpected error happened on our side. Please try again."),
}


DISCLAIMER = "An automated first opinion, not professional advice. Confirm with a local extension officer before treating."


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.jinja_env.trim_blocks = app.jinja_env.lstrip_blocks = True  # tidier HTML output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if app.config["TRUST_PROXY"]:
        # Use the X-Forwarded-For / -Proto headers set by one trusted proxy (real client IP, https URLs).
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    # --- Loaded once, shared by every request ---
    catalog = load_catalog(app.config["DISEASE_CSV"], app.config["SUPPLEMENT_CSV"])
    crops = crop_summary(catalog)
    predictor = load_predictor(app.config)
    stats = {
        "classes": len(catalog),
        "conditions": sum(not c.is_background for c in catalog),
        "diseases": sum(not (c.is_healthy or c.is_background) for c in catalog),
        "crops": len(crops),
    }
    threshold = app.config["CONFIDENCE_THRESHOLD"]
    samples = app.config["SAMPLES"]
    ref_hosts = " ".join(f"https://{host}" for host in sorted(image_hosts(catalog)))
    csp_template = (
        "default-src 'self'; script-src 'self' 'nonce-{nonce}'; style-src 'self'; "
        f"img-src 'self' data: blob: {ref_hosts}; font-src 'self'; connect-src 'self'; "
        "media-src 'self' blob:; manifest-src 'self'; object-src 'none'; base-uri 'self'; "
        f"form-action 'self'; frame-ancestors {app.config['FRAME_ANCESTORS']}"
    )

    def asset(path: str) -> str:
        """URL of a static file with ?v= so a new release busts the browser cache."""
        return url_for("static", filename=path, v=app.config["STATIC_VERSION"])

    def run_prediction(image):
        result = predictor.predict(image)
        best = catalog[result["index"]]
        log.info("prediction label=%s confidence=%.3f latency_ms=%.0f",
                 best.key, result["confidence"], result["latency_ms"])
        return best, result

    def render_result(image, sample=None):
        best, result = run_prediction(image)
        return render_template(
            "result.html",
            c=best,
            confidence=result["confidence"],
            top3=[(catalog[i], p) for i, p in result["top3"]],
            low_confidence=result["confidence"] < threshold,
            photo=preview_data_url(image, app.config["PREVIEW_MAX_SIDE"]),
            sample=sample,
        )

    # --- Request hooks ---
    limiter = RateLimiter()

    @app.before_request
    def rate_limit():
        """Only the endpoints that run the model are limited; pages and static files are not."""
        if request.method == "POST" and request.endpoint in {"diagnose", "diagnose_sample", "api_predict"}:
            if not limiter.allow(request.remote_addr or "unknown", app.config["RATE_LIMIT_PER_MINUTE"]):
                log.warning("rate limit hit for %s", request.remote_addr)
                abort(429)

    @app.before_request
    def make_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)  # lets our one inline <script> run under the CSP

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = csp_template.format(nonce=g.get("csp_nonce", ""))
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        return response

    @app.context_processor
    def template_globals():
        return {
            "app_name": app.config["APP_NAME"],
            "tagline": app.config["APP_TAGLINE"],
            "author": app.config["AUTHOR"],
            "copyright_year": app.config["COPYRIGHT_YEAR"],
            "repo_url": app.config["REPO_URL"],
            "asset": asset,
            "csp_nonce": g.get("csp_nonce", ""),
            "demo_mode": not predictor.model_loaded,
            "model_name": predictor.name,
            "model_kind": predictor.kind,
            "model_sizes": predictor.sizes,
            "stats": stats,
        }

    # --- Pages ---
    @app.get("/")
    def home():
        return render_template("home.html", crops=crops)

    @app.route("/diagnose", methods=["GET", "POST"])
    def diagnose():
        if request.method == "GET":
            return render_template("diagnose.html", samples=samples)
        try:
            image = read_upload(request.files.get("image"), app.config)
        except UploadError as exc:
            return render_template("diagnose.html", samples=samples, error=str(exc)), 400
        return render_result(image)

    @app.post("/diagnose/sample")
    def diagnose_sample():
        """Diagnose one of the bundled sample photos (for trying the app without a leaf at hand)."""
        name = request.form.get("sample", "")
        if name not in samples:  # only our own files: the name never becomes an arbitrary path
            abort(404)
        with Image.open(Path(app.static_folder) / "img" / "samples" / f"{name}.jpg") as image:
            return render_result(image.convert("RGB"), sample=samples[name])

    @app.get("/treatments")
    def treatments():
        products = [c for c in catalog if not c.is_background and c.product_link]
        return render_template("treatments.html", products=products, crops=crops)

    @app.get("/about")
    def about():
        return render_template("about.html", crops=crops, threshold=threshold)

    # --- Old URLs from the original app ---
    @app.get("/index")
    def old_index():
        return redirect(url_for("diagnose"), code=301)

    @app.get("/market")
    def old_market():
        return redirect(url_for("treatments"), code=301)

    @app.get("/contact")
    def old_contact():
        return redirect(url_for("about"), code=301)

    @app.route("/submit", methods=["GET", "POST"])
    def old_submit():
        # 307 keeps the POST method and the uploaded file; a GET just goes to the form.
        return redirect(url_for("diagnose"), code=307 if request.method == "POST" else 301)

    # --- JSON API ---
    @app.post("/api/predict")
    def api_predict():
        try:
            image = read_upload(request.files.get("image"), app.config)
        except UploadError as exc:
            return jsonify(error="invalid_image", message=str(exc)), 400
        best, result = run_prediction(image)
        return jsonify(
            label=best.key,
            crop=best.crop,
            condition=best.condition,
            is_healthy=best.is_healthy,
            is_background=best.is_background,
            confidence=round(result["confidence"], 4),
            top3=[{"label": catalog[i].key, "confidence": round(p, 4)} for i, p in result["top3"]],
            low_confidence=result["confidence"] < threshold,
            demo_mode=not predictor.model_loaded,
            notice=DISCLAIMER,
        )

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", model_loaded=predictor.model_loaded, model=predictor.kind)

    # --- Files browsers ask for at the site root ---
    @app.get("/favicon.ico")
    def favicon():
        return send_from_directory(app.static_folder, "img/favicon.ico")

    @app.get("/robots.txt")
    def robots():
        return send_from_directory(app.static_folder, "robots.txt")

    # --- Errors ---
    @app.errorhandler(HTTPException)
    def http_error(exc):
        title, message = ERROR_TEXT.get(exc.code, (exc.name, exc.description))
        if request.path.startswith("/api/"):
            return jsonify(error=title, message=message), exc.code
        return render_template("error.html", code=exc.code, title=title, message=message), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc):
        log.exception("Unhandled error on %s", request.path)
        title, message = ERROR_TEXT[500]
        if request.path.startswith("/api/"):
            return jsonify(error=title, message=message), 500
        return render_template("error.html", code=500, title=title, message=message), 500

    return app


if __name__ == "__main__":
    # `python app.py` still works; debug only if FLASK_DEBUG=1.
    application = create_app()
    application.run(debug=application.config["DEBUG"])

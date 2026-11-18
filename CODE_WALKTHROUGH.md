# LeafScan — code walkthrough

A plain-English tour of every file, in the order a request flows through them. Line numbers refer to
the files as committed on the `redesign` branch. All app files live in `Flask Deployed App/`.

## Quick answers for the examiner

| If the examiner asks… | The answer is in |
|---|---|
| **How is the image preprocessed? Does it match training?** | `model.py` lines 17–30. `Resize(255)` (short side → 255 px, aspect kept) → `CenterCrop(224)` → `ToTensor()` (pixels → floats 0–1, shape 3×224×224), then `unsqueeze(0)` adds the batch dimension. Same as the notebook (`Model/Plant Disease Detection Code.md` line 35). `.convert("RGB")` on line 29 handles PNG alpha / greyscale. EXIF rotation: `app.py` line 48. |
| **How is the confidence computed?** | `model.py` lines 68–82. The network outputs 39 raw scores (logits). `torch.softmax` (line 74) turns them into probabilities that are positive and sum to 1. `torch.topk(probs, 3)` (line 75) picks the 3 largest. The top one's probability is the "confidence". The 80% warning is `low_confidence=result["confidence"] < threshold` in `app.py` line 152 (threshold from `config.py` line 38). |
| **Why only one gunicorn worker?** | `Flask Deployed App/Procfile` line 1, explained in `DECISIONS.md` ("Production server"). Each worker is a separate process that would load its own ~205 MB copy of the model. One worker loads it once at startup (`app.py` line 81); `--threads 4` lets that one process serve 4 requests at once, sharing the model. PyTorch releases Python's GIL during the forward pass, so the threads really do overlap. |
| **Why are uploads processed in memory?** | `app.py` lines 29–51 (`read_upload`). The original saved files to `static/uploads/` using the user's filename: a path-traversal hole (`../../app.py`), unlimited disk growth, and a privacy problem (anyone could fetch others' photos). Now `file.read()` gives bytes, `io.BytesIO` wraps them so Pillow can open them like a file, and nothing touches the disk. The photo shown on the result page is a small base64 data URL (`preview_data_url`, lines 54–60). |
| **How does a camera capture reach the form?** | `static/js/diagnose.js`. `getUserMedia` (line 194) streams the back camera into a `<video>`. Capture draws the current frame onto a `<canvas>`. "Use photo" (line 230) calls `canvas.toBlob` → wraps it in a `File` → `handleFile()` (line 78), the same function used for picked/dropped/pasted files. That calls `setInputFile()` (line 28), which uses a `DataTransfer` object to put the file into the real `<input type="file" name="image">`. The normal form submit then uploads it as multipart data, exactly like a chosen file. `stopCamera()` (line 177) turns the camera off when the dialog closes. |
| What stops a fake image (a `.txt` renamed `.jpg`)? | `app.py` lines 37–50: extension check, then Pillow `verify()` fails on non-images → `UploadError` → 400 with a friendly message. Tested in `tests/test_app.py`. |
| What happens with a 9 MB file? | `config.py` line 45 `MAX_CONTENT_LENGTH`. Flask refuses to read the body and raises 413; `app.py` lines 217–222 render the custom "Photo too large" page (or JSON for `/api/`). |
| How do templates know if a class is healthy? | `catalog.py` lines 100–101: `is_healthy = key.endswith("healthy")`, `is_background = key == "Background_without_leaves"`. Templates read `c.is_healthy`; no hard-coded index lists. |
| What does `model.eval()` / `inference_mode()` do? | `model.py` lines 58 and 72. `eval()` switches off Dropout and makes BatchNorm use its stored averages. `inference_mode()` stops PyTorch recording operations for gradients, so inference is faster and uses less memory. |
| What is `weights_only=True`? | `model.py` line 48. `.pt` files are Python pickles, which can run code when loaded. `weights_only` only allows tensors and plain containers. `map_location="cpu"` lets weights saved on a GPU load on a CPU server. |
| What is the CSP nonce? | `app.py` lines 108–118. A random value per request; only `<script>` tags carrying it (our one inline theme script in `base.html`) may run inline. Injected scripts can't guess it. |

---

## `config.py` — settings

- `_env_bool` (line 8): reads environment variables like `FLASK_DEBUG=1` as true/false.
- `_default_model_path` (line 16): the original repo used two different weight filenames; use whichever exists.
- `class Config`: plain class attributes. Flask copies them into `app.config` (`from_object`).
  - `APP_NAME` / `APP_TAGLINE` (line 27): rename the app here only.
  - `MODEL_PATH`, `REQUIRE_MODEL`, `CONFIDENCE_THRESHOLD` (lines 33–38): overridable by env vars.
  - `MAX_CONTENT_LENGTH` (line 45): Flask's built-in upload size limit, 8 MB.
  - `ALLOWED_EXTENSIONS` / `ALLOWED_FORMATS`: what the filename may end with / what Pillow must detect inside.
  - `DEBUG` (line 51): only on with `FLASK_DEBUG=1`; never in production.
  - `STATIC_VERSION`, `SEND_FILE_MAX_AGE_DEFAULT` (lines 52–53): cache static files for a year; bump the version to force refresh.

## `CNN.py` — the network (unchanged architecture)

- 4 convolution blocks: each is Conv → ReLU → BatchNorm → Conv → ReLU → BatchNorm → MaxPool(2).
  Channels grow 3 → 32 → 64 → 128 → 256 while the image shrinks 224 → 112 → 56 → 28 → 14.
- `out.view(-1, 50176)`: flattens 256 × 14 × 14 = 50,176 numbers.
- Dense layers: Dropout(0.4) → Linear(50176 → 1024) → ReLU → Dropout(0.4) → Linear(1024 → 39).
- `idx_to_classes`: class index → class name. The order matches both CSV files row by row.
- Only change: removed an unused `import pandas`.

## `catalog.py` — the knowledge base

- `CROP_NAMES` / `CONDITION_NAMES` (lines 14–19): display-only spelling fixes (e.g. Huanglongbing).
- `Condition` dataclass (line 26): one object per class with everything a page needs.
  `label` gives "Tomato · Late blight"; `product_kind` is "fertiliser" for healthy, else "treatment".
- `_split_lines` (line 53): a CSV cell → list of sentences (split on newlines; also adds a missing space after full stops).
- `_display_names` (line 60): "Grape : Esca | Black Measles" → ("Grape", "Esca (black measles)").
- `_read_csv` (line 73): opens with `cp1252` (the files came from Excel on Windows); Python strings are Unicode after that.
- `load_catalog` (line 79): reads both CSVs, checks they line up with the model's classes, builds the 39 `Condition`s.
- `crop_summary` (line 112): one entry per crop with its disease count, for the home-page grid.
- `image_hosts` (line 124): hostnames of reference photos, used to build the CSP.

## `model.py` — inference

- `PREPROCESS` (line 20) and `preprocess()` (line 27): see the quick answers above.
- `Predictor.__init__` (line 34): limits PyTorch threads, loads weights if present (otherwise logs an
  error and stays in demo mode with seeded random weights), calls `eval()`, warms up.
- `_warm_up` (line 61): one dummy forward pass with zeros so the first real user isn't slow.
- `predict` (line 68): preprocess → forward pass → softmax → top-3 → dict with index, confidence, top3, latency in ms.

## `app.py` — the web app

- `MAX_IMAGE_PIXELS` (line 22): refuse images over 40 megapixels (protection against "decompression bombs").
- `UploadError` (line 25): a user-fixable problem; its message is shown on the page.
- `read_upload` (line 29): validation pipeline, all in memory (see quick answers).
- `preview_data_url` (line 54): shrinks the photo to ≤512 px, JPEG-encodes it, base64-encodes it into a `data:` URL.
- `ERROR_TEXT` (line 63): friendly titles/messages for 404, 405, 413, 500.
- `create_app` (line 71):
  - loads config, the catalog, crop summary and the model **once**; computes the home-page numbers;
    builds the CSP string.
  - `asset()` (line 96): `url_for('static', …, v=…)` for cache busting.
  - `run_prediction()` (line 100): calls the model, looks up the `Condition`, logs label/confidence/latency.
  - `make_nonce` / `security_headers` (lines 108–118): per-request nonce; CSP and other headers on every response.
  - `template_globals` (line 121): variables every template can use (app name, `asset`, nonce, demo mode, stats).
  - Routes: `/` home; `/diagnose` GET shows the form, POST validates → predicts → renders `result.html`
    (or re-renders the form with a 400 error); `/treatments`; `/about`; old-URL redirects; `/api/predict`
    (same validation, JSON out); `/healthz`; `/favicon.ico`; `/robots.txt`.
  - Error handlers (lines 217–231): JSON for `/api/…`, the styled `error.html` page otherwise. Unexpected
    exceptions are logged with a traceback and shown as a 500 page.

## `wsgi.py` and `Procfile`

- `wsgi.py` builds the app once for servers that want a module-level `app` object.
- `Procfile`: `gunicorn "app:create_app()" --workers 1 --threads 4 --timeout 60` (see quick answers).

## Templates (`templates/`)

- `base.html`: the page shell. The inline head script (with the nonce) adds the `js` class and applies a
  saved theme before the page paints. Preloads both fonts; links CSS, favicons, manifest, OG tags.
  Each page sets `{% set title = "…" %}` at its top; Jinja makes that visible to the base layout.
- `partials/header.html`: logo, Menu button (JS only), nav with `aria-current="page"`, theme button.
- `partials/footer.html`: disclaimer and the credit line.
- `partials/macros.html`: `icon()` (SVG sprite `<use>`), `status_chip()`, `product_card()`.
- `home.html`: hero, numbers strip, three steps, crop grid (all numbers from Python).
- `diagnose.html`: the form. Works without JS: a visible file input and a submit button.
  `diagnose.js` hides the raw input, shows "Choose photo"/"Use camera", preview and errors.
  The camera is a native `<dialog>`.
- `result.html`: photo + verdict card (status chip, crop, condition, confidence, low-confidence banner),
  top-3 `<progress>` bars, "About this condition"/"What to do" lists (or growing tips/benefits for healthy),
  reference photo, product card with third-party note. Background class hides the advice sections.
- `treatments.html`: filters (JS only) and every product card with `data-crop`, `data-kind`, `data-search`.
- `about.html`, `error.html`: static text; error page used for every HTTP error.

## Static files (`static/`)

- `css/app.css`: tokens (colours, type scale, spacing) at the top, then base, layout, components, pages,
  utilities, motion (only inside `prefers-reduced-motion: no-preference`) and print styles.
  Dark mode = the same token names with different values.
- `js/theme.js`: theme button cycles system/light/dark (saved in `localStorage` inside try/catch);
  phone Menu button toggles `aria-expanded`, which CSS uses to show the nav; Escape closes it.
- `js/diagnose.js`: `downscale()` (line 54) resizes to 1024 px JPEG in the browser; `handleFile()`
  (line 78) checks type → resizes → checks 8 MB → puts the file into the input → shows the preview.
  Drag-and-drop (line 112), paste (line 122), submit guard and loading state (line 129), camera (line 157 on).
- `js/result.js`: print button, hides the reference photo if it fails to load, animates the bars once.
- `js/treatments.js`: keeps `{text, crop, kind}` filter state; `update()` hides cards that don't match
  and writes the count into an `aria-live` region so screen readers hear it.
- `icons.svg`: one `<symbol>` per icon. `img/`: crop and product illustrations, favicons, OG image
  (all generated by `scripts/make_assets.py`). `fonts/`: Inter and Fraunces. `site.webmanifest`: lets
  phones add LeafScan to the home screen.

## Tests and scripts

- `tests/test_app.py`: pages, redirects, uploads (real test images, PNG modes, empty, fake, too big,
  path traversal), API schema and errors, headers, catalog flags. Accuracy assertions skip in demo mode.
- `scripts/benchmark.py`: accuracy table on `test_images/` + median latency of 20 runs.
- `scripts/check_html.py`: tag balance, duplicate ids, missing alt, inline styles.
- `scripts/screenshots.py`: Playwright screenshots + horizontal-overflow check at 320/375/1440 px.
- `scripts/make_assets.py`: draws the SVG art and renders PNG icons and the OG image.

# LeafScan — code walkthrough

A plain-English tour of every file, in the order a request flows through them. Line numbers refer to
the files as committed on the `redesign` branch. All app files live in `Flask Deployed App/`.

## Quick answers for the examiner

| If the examiner asks… | The answer is in |
|---|---|
| **How is the image preprocessed? Does it match training?** | `model.py` lines 25–48. Each model gets the transform it was trained with. Hybrid (default): `Resize(256)` (short side → 256 px, aspect kept) → `CenterCrop(224)` → `ToTensor()` (pixels → floats 0–1) → `Normalize` with ImageNet mean/std, identical to `EVAL_TF` in `training/common.py`. Original CNN: `Resize(255)` → `CenterCrop(224)` → `ToTensor()`, as in its notebook (`Model/Plant Disease Detection Code.md` line 35). `.convert("RGB")` on line 47 handles PNG alpha / greyscale; `unsqueeze(0)` adds the batch dimension. EXIF rotation: `app.py` line 49. |
| **How is the confidence computed?** | `model.py` lines 97–112. The network outputs 39 raw scores (logits) per view. With TTA (line 102) the photo and its mirror image are a batch of 2. `torch.softmax` (line 105) turns each row into probabilities that sum to 1, and `.mean(dim=0)` averages the two views. `torch.topk(probs, 3)` (line 106) picks the 3 largest; the top one is the "confidence". The 80% warning is `low_confidence=result["confidence"] < threshold` in `app.py` line 164 (threshold from `config.py` line 44). Label smoothing during training (`training/train_hybrid.py`) keeps these probabilities from being over-confident. |
| **What is the hybrid model and why that one?** | `model.py` line 51 (`build_network`) creates `caformer_s18` from timm. Stages 1–2 are convolutions (good at local texture: spots, rings, mould), stages 3–4 are self-attention (each patch looks at every other patch: whole-leaf patterns). It was pre-trained on ImageNet-22k, then fine-tuned on PlantVillage + PlantDoc field photos. Why: lab accuracy is saturated in the literature; field photos are the hard part, and pre-trained hybrids + field data + augmentation help most there. Results and sources: `DECISIONS.md` "Hybrid model". |
| **How do you know the test results aren't inflated?** | `training/prepare_data.py`. PlantVillage has several photos of each leaf, so it uses the dataset's official **leaf-grouped** split: no leaf appears in both train and test. PlantDoc test files identical to training files are removed (line with `md5`). Validation sets (for picking the epoch) are separate from the test sets, and the test sets are used once, by `training/evaluate.py`. |
| **Why accuracy *and* recall?** | Recall per class = of all photos that really have disease X, how many the model calls X. Macro recall averages that over classes, so rare diseases count as much as common ones; a missed disease is the costly mistake for a farmer. `training/common.py` `metrics()`; the training loop keeps the epoch with the best mean of lab and field macro recall. |
| **Why only one gunicorn worker?** | `Flask Deployed App/Procfile` line 1, explained in `DECISIONS.md` ("Production server"). Each worker is a separate process that would load its own copy of the model. One worker loads it once at startup (`app.py` line 90); `--threads 4` lets that one process serve 4 requests at once, sharing the model. PyTorch releases Python's GIL during the forward pass, so the threads really do overlap. |
| **Why are uploads processed in memory?** | `app.py` lines 30–52 (`read_upload`). The original saved files to `static/uploads/` using the user's filename: a path-traversal hole (`../../app.py`), unlimited disk growth, and a privacy problem (anyone could fetch others' photos). Now `file.read()` gives bytes, `io.BytesIO` wraps them so Pillow can open them like a file, and nothing touches the disk. The photo shown on the result page is a small base64 data URL (`preview_data_url`, lines 55–61). |
| **How does a camera capture reach the form?** | `static/js/diagnose.js`. `getUserMedia` (line 194) streams the back camera into a `<video>`. Capture draws the current frame onto a `<canvas>`. "Use photo" (line 230) calls `canvas.toBlob` → wraps it in a `File` → `handleFile()` (line 78), the same function used for picked/dropped/pasted files. That calls `setInputFile()` (line 28), which uses a `DataTransfer` object to put the file into the real `<input type="file" name="image">`. The normal form submit then uploads it as multipart data, exactly like a chosen file. `stopCamera()` (line 177) turns the camera off when the dialog closes. |
| What stops a fake image (a `.txt` renamed `.jpg`)? | `app.py` lines 38–51: extension check, then Pillow `verify()` (line 45) fails on non-images → `UploadError` → 400 with a friendly message. Tested in `tests/test_app.py`. |
| What happens with a 9 MB file? | `config.py` line 51 `MAX_CONTENT_LENGTH`. Flask refuses to read the body and raises 413; `app.py` lines 229–234 render the custom "Photo too large" page (or JSON for `/api/`). |
| How do templates know if a class is healthy? | `catalog.py` lines 100–101: `is_healthy = key.endswith("healthy")`, `is_background = key == "Background_without_leaves"`. Templates read `c.is_healthy`; no hard-coded index lists. |
| What does `model.eval()` / `inference_mode()` do? | `model.py` lines 87 and 103. `eval()` switches off Dropout and makes normalisation layers use their stored statistics. `inference_mode()` stops PyTorch recording operations for gradients, so inference is faster and uses less memory. |
| What is `weights_only=True`? | `model.py` line 76. `.pt` files are Python pickles, which can run code when loaded. `weights_only` only allows tensors and plain containers. `map_location="cpu"` lets weights saved on a GPU load on a CPU server. Line 78 converts the hybrid's half-precision (fp16) weights back to fp32. |
| What is the CSP nonce? | `app.py` lines 118–129. A random value per request; only `<script>` tags carrying it (our one inline theme script in `base.html`) may run inline. Injected scripts can't guess it. |

---

## `config.py` — settings

- `_env_bool` (line 8): reads environment variables like `FLASK_DEBUG=1` as true/false.
- `_default_model_path` (line 16): the original repo used two different weight filenames; use whichever exists.
- `class Config`: plain class attributes. Flask copies them into `app.config` (`from_object`).
  - `APP_NAME` / `APP_TAGLINE` (line 27): rename the app here only.
  - `MODEL_KIND`, `HYBRID_MODEL_PATH`, `TTA` (lines 35–38): which model to run, where its weights are,
    and whether to also score the mirror image.
  - `MODEL_PATH`, `REQUIRE_MODEL`, `CONFIDENCE_THRESHOLD` (lines 39–44): overridable by env vars.
  - `MAX_CONTENT_LENGTH` (line 51): Flask's built-in upload size limit, 8 MB.
  - `ALLOWED_EXTENSIONS` / `ALLOWED_FORMATS` (line 52): what the filename may end with / what Pillow must detect inside.
  - `DEBUG` (line 57): only on with `FLASK_DEBUG=1`; never in production.
  - `STATIC_VERSION`, `SEND_FILE_MAX_AGE_DEFAULT` (lines 58–59): cache static files for a year; bump the version to force refresh.

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

- `PREPROCESS` (line 26) and `preprocess()` (line 45): one transform per model kind; see the quick answers.
- `build_network` (line 51): `timm.create_model("caformer_s18", pretrained=False, num_classes=39)` for the
  hybrid (no download: the trained weights come from our own file), or the original `CNN.CNN(39)`.
- `Predictor.__init__` (line 59): limits PyTorch threads, loads weights if present (fp16 → fp32), otherwise
  logs an error and stays in demo mode with seeded random weights; calls `eval()`, warms up.
- `_warm_up` (line 90): one dummy forward pass so the first real user isn't slow.
- `predict` (line 97): preprocess → (mirror copy if TTA) → forward pass → softmax → average → top-3 →
  dict with index, confidence, top3, latency in ms.

## `app.py` — the web app

- `MAX_IMAGE_PIXELS` (line 23): refuse images over 40 megapixels (protection against "decompression bombs").
- `UploadError` (line 26): a user-fixable problem; its message is shown on the page.
- `read_upload` (line 30): validation pipeline, all in memory (see quick answers).
- `preview_data_url` (line 55): shrinks the photo to ≤512 px, JPEG-encodes it, base64-encodes it into a `data:` URL.
- `load_predictor` (line 64): resolves `MODEL_KIND=auto` (hybrid if `leafscan_hybrid.pt` exists, else CNN)
  and builds the `Predictor`.
- `ERROR_TEXT` (line 73): friendly titles/messages for 404, 405, 413, 500.
- `create_app` (line 81):
  - loads config, the catalog, crop summary and the model **once** (line 90); computes the home-page
    numbers; builds the CSP string.
  - `asset()` (line 106): `url_for('static', …, v=…)` for cache busting.
  - `run_prediction()` (line 110): calls the model, looks up the `Condition`, logs label/confidence/latency.
  - `make_nonce` / `security_headers` (lines 118–129): per-request nonce; CSP and other headers on every response.
  - `template_globals` (line 131): variables every template can use (app name, `asset`, nonce, demo mode,
    model name/kind, stats).
  - Routes: `/` home; `/diagnose` GET shows the form, POST validates → predicts → renders `result.html`
    (or re-renders the form with a 400 error); `/treatments`; `/about`; old-URL redirects; `/api/predict`
    (same validation, JSON out); `/healthz`; `/favicon.ico`; `/robots.txt`.
  - Error handlers (lines 229–243): JSON for `/api/…`, the styled `error.html` page otherwise. Unexpected
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

## Training (`training/`, run with `.venv-train`, needs a GPU)

- `common.py`: the 39 classes in the app's order, the architecture name, `TRAIN_TF` (heavy augmentation:
  random crops, flips, TrivialAugment colour/geometry changes, blur, random erasing) and `EVAL_TF` (the
  same transform the app uses), `LeafDataset`, `metrics()` (accuracy, macro precision/recall/F1,
  per-class recall) and `predict_probs()` (with optional mirror TTA).
- `fetch_zip_subset.py`: downloads only the needed files out of a huge remote zip using HTTP Range
  requests, in 8 MB resumable pieces (the connection here kept dropping on 1–2 GB downloads).
- `prepare_data.py`: maps PlantVillage and PlantDoc folder names to the 39 classes and writes
  `data/splits.json` (train / val_lab / val_field / test_lab / test_field). Leaf-grouped PlantVillage
  split; PlantDoc duplicates removed from test.
- `train_hybrid.py`: `balanced_sampler()` (30% field draws, 1/sqrt(class size) inside each domain),
  AdamW + warm-up + cosine schedule, bf16 mixed precision, label smoothing; validates every epoch and keeps
  the best epoch (fp16, copied to `Flask Deployed App/leafscan_hybrid.pt`).
- `evaluate.py`: runs the kept model once on the two test sets, with and without TTA, and prints the
  tables used in `DECISIONS.md`.

## Tests and scripts

- `tests/test_model.py`: `MODEL_KIND=auto` fallback, hybrid top-3 output, hybrid normalisation.
- `tests/test_app.py`: pages, redirects, uploads (real test images, PNG modes, empty, fake, too big,
  path traversal), API schema and errors, headers, catalog flags. Accuracy assertions skip in demo mode.
- `scripts/benchmark.py`: accuracy table on `test_images/` + median latency of 20 runs.
- `scripts/check_html.py`: tag balance, duplicate ids, missing alt, inline styles.
- `scripts/screenshots.py`: Playwright screenshots + horizontal-overflow check at 320/375/1440 px.
- `scripts/make_assets.py`: draws the SVG art and renders PNG icons and the OG image.

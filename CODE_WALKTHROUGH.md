# LeafScan — code walkthrough

A plain-English tour of every file, in the order a request flows through them. Line numbers refer to
the files as committed on the `redesign` branch. Author: Esh Dobhal. All app files live in `Flask Deployed App/`.

## Quick answers for the examiner

| If the examiner asks… | The answer is in |
|---|---|
| **How is the image preprocessed? Does it match training?** | `model.py` `make_transform` (line 29). Each model gets the transform it was trained with. Hybrid: `Resize(size × 256/224)` (short side → 256 px for the 224 px model, 329 px for the 288 px model; aspect kept) → `CenterCrop(size)` → `ToTensor()` (pixels → floats 0–1) → `Normalize` with ImageNet mean/std, identical to `eval_transform()` in `training/common.py`. Original CNN: `Resize(255)` → `CenterCrop(224)` → `ToTensor()`, as in its notebook (`Model/Plant Disease Detection Code.md` line 35). `.convert("RGB")` (line 111) handles PNG alpha / greyscale; `unsqueeze(0)` adds the batch dimension. EXIF rotation: `app.py` line 53. |
| **How is the confidence computed?** | `model.py` `predict` (lines 108–128). For each of the two networks: the photo (and, with TTA, its mirror image, line 116) goes through the network, which outputs 39 raw scores (logits) per view; `torch.softmax` (line 119) turns each row into probabilities that sum to 1. Line 120 stacks all 4 rows (2 networks × 2 views) and averages them. `torch.topk(probs, 3)` (line 121) picks the 3 largest; the top one is the "confidence". The 80% warning is `low_confidence=result["confidence"] < threshold` in `app.py` line 167 (threshold from `config.py` line 49). Label smoothing during training (`training/train_hybrid.py`) keeps these probabilities from being over-confident. |
| **What is the hybrid model and why that one?** | `model.py` line 51 (`build_network`) creates `caformer_s18` from timm. Stages 1–2 are convolutions (good at local texture: spots, rings, mould), stages 3–4 are self-attention (each patch looks at every other patch: whole-leaf patterns). It was pre-trained on ImageNet-22k, then fine-tuned on PlantVillage + PlantDoc field photos. Why: lab accuracy is saturated in the literature; field photos are the hard part, and pre-trained hybrids + field data + augmentation help most there. Results and sources: `DECISIONS.md` "Hybrid model". |
| **Why two models (an ensemble)?** | The same architecture was trained twice, at 224 px and at 288 px (`training/common.py` `TRAIN_SIZE`). They make different mistakes, so averaging their probabilities corrects some of each other's errors. Chosen on the validation sets (`training/compare_models.py`), then confirmed once on the test sets: field accuracy 68.4% → 71.1%, field macro recall 67.8% → 70.4%. Cost: ~190 ms instead of ~70 ms per photo on the CPU. `app.py` `load_predictor` (line 68) adds the 288 px model only if its file exists and `ENSEMBLE` is on (`config.py` lines 38–41). |
| **How do you know the test results aren't inflated?** | `training/prepare_data.py`. PlantVillage has several photos of each leaf, so it uses the dataset's official **leaf-grouped** split: no leaf appears in both train and test. PlantDoc test files identical to training files are removed (line with `md5`). Validation sets (for picking the epoch) are separate from the test sets, and the test sets are used once, by `training/evaluate.py`. |
| **Why accuracy *and* recall?** | Recall per class = of all photos that really have disease X, how many the model calls X. Macro recall averages that over classes, so rare diseases count as much as common ones; a missed disease is the costly mistake for a farmer. `training/common.py` `metrics()`; the training loop keeps the epoch with the best mean of lab and field macro recall. |
| **Why only one gunicorn worker?** | `Flask Deployed App/Procfile` line 1, explained in `DECISIONS.md` ("Production server"). Each worker is a separate process that would load its own copy of the model. One worker loads it once at startup (`app.py` line 132); `--threads 4` lets that one process serve 4 requests at once, sharing the model. PyTorch releases Python's GIL during the forward pass, so the threads really do overlap. |
| **Why are uploads processed in memory?** | `app.py` lines 34–56 (`read_upload`). The original saved files to `static/uploads/` using the user's filename: a path-traversal hole (`../../app.py`), unlimited disk growth, and a privacy problem (anyone could fetch others' photos). Now `file.read()` gives bytes, `io.BytesIO` wraps them so Pillow can open them like a file, and nothing touches the disk. The photo shown on the result page is a small base64 data URL (`preview_data_url`, lines 59–65). |
| **How does a camera capture reach the form?** | `static/js/diagnose.js`. `getUserMedia` (line 194) streams the back camera into a `<video>`. Capture draws the current frame onto a `<canvas>`. "Use photo" (line 230) calls `canvas.toBlob` → wraps it in a `File` → `handleFile()` (line 78), the same function used for picked/dropped/pasted files. That calls `setInputFile()` (line 28), which uses a `DataTransfer` object to put the file into the real `<input type="file" name="image">`. The normal form submit then uploads it as multipart data, exactly like a chosen file. `stopCamera()` (line 177) turns the camera off when the dialog closes. |
| What stops a fake image (a `.txt` renamed `.jpg`)? | `app.py` lines 42–55: extension check, then Pillow `verify()` (line 49) fails on non-images → `UploadError` → 400 with a friendly message. Tested in `tests/test_app.py`. |
| What happens with a 9 MB file? | `config.py` line 56 `MAX_CONTENT_LENGTH`. Flask refuses to read the body and raises 413; `app.py` lines 297–302 render the custom "Photo too large" page (or JSON for `/api/`). |
| How do templates know if a class is healthy? | `catalog.py` lines 100–101: `is_healthy = key.endswith("healthy")`, `is_background = key == "Background_without_leaves"`. Templates read `c.is_healthy`; no hard-coded index lists. |
| What does `model.eval()` / `inference_mode()` do? | `model.py` lines 97 and 117. `eval()` switches off Dropout and makes normalisation layers use their stored statistics. `inference_mode()` stops PyTorch recording operations for gradients, so inference is faster and uses less memory. |
| What is `weights_only=True`? | `model.py` line 92. `.pt` files are Python pickles, which can run code when loaded. `weights_only` only allows tensors and plain containers. `map_location="cpu"` lets weights saved on a GPU load on a CPU server. Line 94 converts the hybrid's half-precision (fp16) weights back to fp32. |
| What is the CSP nonce? | `app.py` lines 183–193. A random value per request; only `<script>` tags carrying it (our one inline theme script in `base.html`) may run inline. Injected scripts can't guess it. |
| How is the server protected from being flooded? | `app.py` `RateLimiter` (line 83) and the `rate_limit` hook (line 176). Only the three endpoints that run the model are limited: each client IP may make `RATE_LIMIT_PER_MINUTE` requests (30 by default, `config.py` line 71) in any 60 seconds; more get a 429 "Too many requests" page or JSON. It keeps a queue of recent request times per IP and drops times older than 60 s ("sliding window"). A `Lock` is needed because four threads serve requests at once. In memory is correct because there is one process. Behind a hosting proxy every request seems to come from the proxy, so `TRUST_PROXY=1` (line 127) makes Flask use the real client IP from `X-Forwarded-For`. |
| How do the sample photos work, and are they safe? | `config.py` `SAMPLES` (line 62) names four photos in `static/img/samples/`, none used in training. The Diagnose page posts the chosen name to `/diagnose/sample` (`app.py` line 228). The name is only accepted if it is one of those keys (line 231), so it can never be turned into a path like `../../app.py`; anything else is a 404. |

---

## `config.py` — settings

- `_env_bool` (line 8): reads environment variables like `FLASK_DEBUG=1` as true/false.
- `_default_model_path` (line 16): the original repo used two different weight filenames; use whichever exists.
- `class Config`: plain class attributes. Flask copies them into `app.config` (`from_object`).
  - `APP_NAME` / `APP_TAGLINE` (line 27), `AUTHOR` / `COPYRIGHT_YEAR` (lines 29–30): shown in the footer and
    page metadata. `REPO_URL` (line 32): footer source link, hidden when unset.
  - `MODEL_KIND`, `HYBRID_MODEL_PATH`, `HYBRID_288_MODEL_PATH`, `ENSEMBLE`, `TTA` (lines 37–43): which model
    to run, where the two hybrids' weights are, whether to average both, and whether to also score the mirror image.
  - `MODEL_PATH`, `REQUIRE_MODEL`, `CONFIDENCE_THRESHOLD` (lines 44–49): overridable by env vars.
  - `MAX_CONTENT_LENGTH` (line 56): Flask's built-in upload size limit, 8 MB.
  - `ALLOWED_EXTENSIONS` / `ALLOWED_FORMATS` (line 57): what the filename may end with / what Pillow must detect inside.
  - `SAMPLES` (line 62): the "Try a sample" photos and their button labels.
  - `RATE_LIMIT_PER_MINUTE`, `TRUST_PROXY` (lines 71–73): flood protection; use the real client IP behind a proxy.
  - `FRAME_ANCESTORS` (line 76): which sites may frame the app (CSP); `'none'` except on Hugging Face Spaces.
  - `DEBUG` (line 79): only on with `FLASK_DEBUG=1`; never in production.
  - `STATIC_VERSION`, `SEND_FILE_MAX_AGE_DEFAULT` (lines 80–81): cache static files for a year; bump the version to force refresh.

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

- `make_transform` (line 29) and `preprocess()` (line 45): the transform for a model kind and input size;
  see the quick answers.
- `build_network` (line 51): `timm.create_model("caformer_s18", pretrained=False, num_classes=39)` for the
  hybrid (no download: the trained weights come from our own files), or the original `CNN.CNN(39)`.
- `Predictor.__init__` (line 61): takes `members`, a list of (weights file, input size): one for the CNN, two
  for the hybrid ensemble. Limits PyTorch threads, builds `self.name` (e.g. "… ensemble (224 px + 288 px)"),
  logs an error and stays in demo mode (seeded random weights) if any file is missing, then loads each member.
- `_load` (line 87): builds one network, loads its weights if present (fp16 → fp32), calls `eval()`.
- `_warm_up` (line 100): one dummy forward pass per network so the first real user isn't slow.
- `predict` (line 108): for each network: its own preprocessing → (mirror copy if TTA) → forward pass →
  softmax; then average all rows → top-3 → dict with index, confidence, top3, latency in ms.

## `app.py` — the web app

- `MAX_IMAGE_PIXELS` (line 27): refuse images over 40 megapixels (protection against "decompression bombs").
- `UploadError` (line 30): a user-fixable problem; its message is shown on the page.
- `read_upload` (line 34): validation pipeline, all in memory (see quick answers).
- `preview_data_url` (line 59): shrinks the photo to ≤512 px, JPEG-encodes it, base64-encodes it into a `data:` URL.
- `load_predictor` (line 68): resolves `MODEL_KIND=auto` (hybrid if `leafscan_hybrid.pt` exists, else CNN),
  adds the 288 px member when its file exists and `ENSEMBLE` is on, and builds the `Predictor`.
- `RateLimiter` (line 83): per-client sliding-window counter (see quick answers).
- `ERROR_TEXT` (line 111): friendly titles/messages for 404, 405, 413, 429, 500.
- `create_app` (line 120):
  - wraps the app in `ProxyFix` when `TRUST_PROXY` is on (line 127); loads config, the catalog, crop summary
    and the model **once** (line 132); computes the home-page numbers; builds the CSP string.
  - `asset()` (line 149): `url_for('static', …, v=…)` for cache busting.
  - `run_prediction()` (line 153): calls the model, looks up the `Condition`, logs label/confidence/latency.
  - `render_result()` (line 160): runs the prediction and renders `result.html`; shared by uploads and samples.
  - `rate_limit` (line 176): before each request, refuses model-running POSTs over the limit with 429.
  - `make_nonce` / `security_headers` (lines 183–193): per-request nonce; CSP and other headers on every response.
  - `template_globals` (line 196): variables every template can use (app name, author, `asset`, nonce,
    demo mode, model name/kind/input sizes, stats).
  - Routes: `/` home; `/diagnose` GET shows the form, POST validates → predicts → renders `result.html`
    (or re-renders the form with a 400 error); `/diagnose/sample` (line 228) diagnoses a bundled sample;
    `/treatments`; `/about`; old-URL redirects; `/api/predict` (same validation, JSON out); `/healthz`;
    `/favicon.ico`; `/robots.txt`.
  - Error handlers (lines 297–310): JSON for `/api/…`, the styled `error.html` page otherwise. Unexpected
    exceptions are logged with a traceback and shown as a 500 page.

## `wsgi.py` and `Procfile`

- `wsgi.py` builds the app once for servers that want a module-level `app` object.
- `Procfile`: `gunicorn "app:create_app()" --workers 1 --threads 4 --timeout 60` (see quick answers).

## Templates (`templates/`)

- `base.html`: the page shell. The inline head script (with the nonce) adds the `js` class and applies a
  saved theme before the page paints. Preloads both fonts; links CSS, favicons, manifest, OG tags.
  Each page sets `{% set title = "…" %}` at its top; Jinja makes that visible to the base layout.
- `partials/header.html`: logo, Menu button (JS only), nav with `aria-current="page"`, theme button.
- `partials/footer.html`: disclaimer, © year and author (from `config.py`), and a source link if `REPO_URL` is set.
- `partials/macros.html`: `icon()` (SVG sprite `<use>`), `status_chip()`, `product_card()`.
- `home.html`: hero, numbers strip, three steps, crop grid (all numbers from Python).
- `diagnose.html`: the form. Works without JS: a visible file input and a submit button.
  `diagnose.js` hides the raw input, shows "Choose photo"/"Use camera", preview and errors.
  The camera is a native `<dialog>`. Below the form, "Try a sample" is a second plain form whose four
  image buttons each submit `sample=<name>`.
- `result.html`: photo + verdict card (status chip, crop, condition, confidence, low-confidence banner),
  top-3 `<progress>` bars, "About this condition"/"What to do" lists (or growing tips/benefits for healthy),
  reference photo, product card with third-party note. Background class hides the advice sections.
- `treatments.html`: filters (JS only) and every product card with `data-crop`, `data-kind`, `data-search`.
- `about.html`: what the app does, how the model works, measured accuracy, limitations, author.
- `error.html`: used for every HTTP error (404, 405, 413, 429, 500).

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

- `common.py`: the 39 classes in the app's order, the architecture name, the input size (`TRAIN_SIZE`
  environment variable, 224 or 288), `TRAIN_TF` (heavy augmentation: random crops, flips, TrivialAugment
  colour/geometry changes, blur, random erasing) and `eval_transform(size)` (the same transform the app
  uses), `LeafDataset`, `metrics()` (accuracy, macro precision/recall/F1, per-class recall) and `predict_probs()` (with optional mirror TTA).
- `fetch_zip_subset.py`: downloads only the needed files out of a huge remote zip using HTTP Range
  requests, in 8 MB resumable pieces (the connection here kept dropping on 1–2 GB downloads).
- `prepare_data.py`: maps PlantVillage and PlantDoc folder names to the 39 classes and writes
  `data/splits.json` (train / val_lab / val_field / test_lab / test_field). Leaf-grouped PlantVillage
  split; PlantDoc duplicates removed from test.
- `train_hybrid.py`: `balanced_sampler()` (30% field draws, 1/sqrt(class size) inside each domain),
  AdamW + warm-up + cosine schedule, bf16 mixed precision, label smoothing; validates every epoch and keeps
  the best epoch (fp16, copied to `Flask Deployed App/leafscan_hybrid.pt` unless `--no-copy`).
- `compare_models.py`: scores the 224 px model, the 288 px model and their average on the **validation**
  sets, so the ensemble decision didn't use the test sets.
- `evaluate.py`: runs the kept model (or, with several `--runs`, the averaged ensemble) once on the two
  test sets, with and without TTA, and prints the tables used in `DECISIONS.md`. Both scripts end with
  `if __name__ == "__main__":`, which Windows needs because DataLoader workers re-import the file.
- `report_figures.py`: runs the shipped ensemble once on both test sets (cached in
  `runs/ensemble/test_predictions.npz`) and draws the README charts in light and dark versions into
  `docs/images/`: training curves, model comparison, per-class field recall and the confusion matrix.

## Browser version (`web/`)

- `export_onnx.py`: rebuilds each CAFormer-S18 from its `.pt` file with timm's fused attention switched
  off (plain matmul + softmax, supported by every ONNX Runtime Web backend), exports to ONNX with a dynamic
  batch size, then `weights_to_fp16()` stores the weights as float16 with a `Cast` back to float32 in
  front of each (half the size, same float32 maths). `--check N` compares with PyTorch on N photos.
- `templates/`: Jinja templates rendered once at build time (autoescaped). `base.html` is the shared
  head, header (nav with `aria-current`) and footer; `index.html` is the diagnose page; `treatments.html`
  lists every product with the same markup and `data-` hooks as the Flask page, so the app's own
  `treatments.js` filters it unchanged.
- `build_site.py`: `render_pages()` renders the templates with the catalog and `crop_summary()`; copies
  `js/app.js`, `js/early.js`, `css/web.css` from `web/` and the stylesheet, icons, fonts, images,
  `diagnose.js`, `theme.js` and `treatments.js` from the Flask app (one design, two versions); writes
  `catalog.json` from `catalog.py`; and exports the models into `models/`.
- `js/app.js`: `fetchModel()` downloads a model with a progress bar and keeps it in Cache Storage;
  `createSession()` tries WebGPU, then WebAssembly; `toTensor()` does the server's preprocessing on a
  canvas (resize so the short side is size × 256/224, centre crop, ImageNet normalisation, plus the mirror
  image); `diagnoseBitmap()` averages 2 models × 2 views like `Predictor.predict`; `renderResult()` builds
  the result with `textContent` only (no HTML from data). The form's submit is handled in the page, so
  nothing is uploaded.
- `js/early.js`: the inline theme script of `base.html`, as a file (a static page has no CSP nonce).
- `test_web.py`: serves the built site under the GitHub Pages sub-path and checks it in Chromium (models
  load, sample, upload, Treatments reached from the nav with 38 products and working filters, no
  console/CSP errors, no overflow at 320/375/1280 px); `--parity` runs all 225 field test photos
  through the page and compares with PyTorch.

## Tests and scripts

- `tests/test_model.py`: `MODEL_KIND=auto` fallback, ensemble only when the 288 px file exists and
  `ENSEMBLE` is on, ensemble top-3 output, hybrid normalisation at 224 and 288 px.
- `tests/test_app.py`: pages, redirects, uploads (real test images, PNG modes, empty, fake, too big,
  path traversal), API schema and errors, headers, catalog flags, samples (each one diagnosed correctly,
  unknown names → 404), author credit, rate limiting. Accuracy assertions skip in demo mode.
- `tests/test_e2e.py` (run with `E2E=1`): a real Chromium browser clicks through the app: upload → result,
  sample button, 4000 px PNG resized to 1024 px in the browser, non-image error, fake-camera capture,
  treatment filters, theme persistence, phone menu, no-JavaScript upload, no sideways scrolling at
  320/375/1440 px, skip link, no console/CSP errors, and the API.
- `scripts/benchmark.py`: accuracy table on `test_images/` + median latency of 20 runs.
- `scripts/check_html.py`: tag balance, duplicate ids, missing alt, inline styles.
- `scripts/screenshots.py`: Playwright screenshots + horizontal-overflow check at 320/375/1440 px.
- `scripts/make_assets.py`: draws the SVG art and renders PNG icons and the OG image.
- `.github/workflows/tests.yml`: GitHub Actions installs the app on Linux (CPU PyTorch) and runs the
  50 app/model tests with the real weights and `REQUIRE_MODEL=1` on every push.

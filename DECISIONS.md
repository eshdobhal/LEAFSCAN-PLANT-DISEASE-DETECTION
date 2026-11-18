# LeafScan — design & engineering decisions

Every non-obvious choice made while rebuilding the original Plant-Disease-Detection Flask app as
**LeafScan**, with the measurements behind them.

## Environment notes

- **Python 3.13 instead of 3.11.** This machine only has Python 3.10 and 3.13 (`py -0`). The code uses
  nothing newer than 3.11, and `scripts/make_assets.py` deliberately avoids 3.12-only f-string syntax.
- **Model weights were not available** (not in the repo, not on disk). The app therefore ran in
  **demo mode** for all testing: the real CNN architecture with seeded random weights. Everything
  except prediction *accuracy* is exercised for real (preprocessing, softmax, top-3, templates, API).
- **Node.js is not installed**, so Lighthouse (`npx lighthouse`) could not be run. See "Quality checks".

## Backend

| Decision | Why |
|---|---|
| Split into `config.py`, `catalog.py`, `model.py`, `app.py` | Each file has one job and fits on a screen; easy to explain in a viva. |
| `create_app()` factory | Tests, gunicorn, waitress and `flask run` all build the app the same way; nothing runs at import time. |
| `MODEL_PATH` defaults to whichever of `plant_disease_model_1_latest.pt` / `plant_disease_model_1.pt` exists | The original README and code disagreed on the name. |
| Missing weights → loud `ERROR` log + demo mode + banner; `REQUIRE_MODEL=1` makes it fatal | Lets the UI be developed and tested without the 200 MB file, while production can refuse to start. |
| `torch.load(map_location="cpu", weights_only=True)` | Works for weights saved on a GPU; `weights_only` refuses to unpickle arbitrary code. |
| Preprocessing `Resize(255) → CenterCrop(224) → ToTensor()` | Identical to training (`Model/Plant Disease Detection Code.md`, line 35). The old `resize((224,224))` squashed images and changed what the network saw. No normalisation, because training used none. |
| `.convert("RGB")` and `ImageOps.exif_transpose` | RGBA/greyscale PNGs used to crash the 3-channel model; phone photos are often stored sideways with an EXIF rotation flag. |
| `torch.inference_mode()`, `model.eval()`, warm-up pass, `set_num_threads(min(4, cpus))` | Faster and lighter inference; the slow first pass happens at startup, not for a user. |
| Softmax + `topk(3)` | Gives a confidence and alternatives instead of a bare argmax. |
| Uploads read with `file.read()` into `BytesIO`, never saved | Removes the path-traversal bug (raw filename used as a path), stops `static/uploads/` growing forever, and protects users' privacy. Only the filename's *extension* is inspected. |
| Validation: extension allow-list → Pillow `verify()` → format allow-list → 8 MB `MAX_CONTENT_LENGTH` → `MAX_IMAGE_PIXELS = 40 MP` | A `.txt` renamed `.jpg` fails at Pillow; huge files get 413; "decompression bombs" are refused. |
| Result photo echoed as a ≤512 px JPEG data URL | The page can show the photo without storing it. ~30–60 KB per result page. |
| Validation errors re-render `/diagnose` with HTTP 400 and the message | No session/secret key needed (so no flash messages); the error is announced by a `role="alert"` region. |
| `/index`, `/market`, `/contact`, `GET /submit` → 301; `POST /submit` → 307 to `/diagnose` | Old bookmarks and any old forms keep working; 307 preserves the POST body. |
| Healthy/background flags derived from class names (`endswith("healthy")`, `== "Background_without_leaves"`) | Replaces the index list `[3,5,7,…,38]` copy-pasted into three templates. A test asserts the derived list equals the old one. |
| CSVs read once with Python's `csv` module (cp1252 → `str`), split on newlines into lists | Proper `<ul>` lists in templates. `pandas` and `numpy` were dropped: they were only used for this and for `argmax`. |
| `catalog.py` checks row count and that `supplement_info.csv` row N is class N | A reordered CSV would silently show the wrong advice; now it fails at startup. |
| Display-name fixes in `catalog.py` only (`Haunglongbing` → `Huanglongbing`, `Pepper bell` → `Bell pepper`, `Cedar rust` → `Cedar apple rust`, `A \| B` → `A (b)`, sentence case) | CSV keys and model class names stay untouched. |
| `http://` reference images upgraded to `https://` | Otherwise blocked as mixed content on an HTTPS deployment. |
| Unused `import pandas as pd` removed from `CNN.py` | It forced a pandas install; the architecture and class list are unchanged. |
| `Flask Deployed App/Readme.md` and `static/uploads/` deleted | Outdated instructions (merged into the root README) and the no-longer-used upload folder. |
| Logging via `logging`: label, confidence, latency in ms | Never logs image bytes or filenames. |

## Security headers

- `Content-Security-Policy`: `default-src 'self'`; scripts only from our origin plus one per-request
  **nonce** for the 3-line theme script in `<head>`; `style-src 'self'` (so **no inline `style=`
  attributes anywhere**; `scripts/check_html.py` enforces it); `img-src 'self' data: blob:` plus the
  30 hosts of the CSV reference photos (computed from the CSV at startup, not hard-coded);
  `frame-ancestors 'none'`, `object-src 'none'`, `form-action 'self'`.
- Because inline `onerror=""` is blocked by that CSP, the broken-reference-image fallback lives in
  `result.js`.
- `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`,
  `Permissions-Policy: camera=(self), microphone=(), geolocation=()`.
- Reference photos also use `referrerpolicy="no-referrer"`; product links use `rel="noopener noreferrer"`.
- Static files: `Cache-Control: max-age=31536000` (Flask `SEND_FILE_MAX_AGE_DEFAULT`) and `?v=1`
  appended by the `asset()` helper; bump `STATIC_VERSION` to bust caches.

## Production server

- **One gunicorn worker, four threads.** Each worker is a separate process with its own copy of the
  ~205 MB model (51 M parameters × 4 bytes, mostly the 50176→1024 dense layer). One worker loads it
  once; threads share it. PyTorch releases the GIL during the forward pass, so threads still overlap.
- `waitress` documented for Windows, where gunicorn does not run.
- `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu` rather than
  `--index-url`: with `--index-url` pip could no longer find Flask etc. on PyPI. On Linux the
  PyTorch CPU index is consulted and provides the CPU wheel; Windows wheels on PyPI are CPU-only anyway.

## Frontend

| Decision | Why |
|---|---|
| No framework, one CSS file (22 KB), JS per page ≤ 9.4 KB, zero CDNs | Brief's constraints; fast on slow networks; every line explainable. |
| Fonts: Inter + Fraunces variable woff2 (latin subset, 48 KB + 37 KB) from Fontsource, preloaded, `font-display: swap` | Self-hosted = no third-party request. OFL licence. |
| Warm-paper / deep-leaf palette, Fraunces headings | "Field notebook meets clinical lab". |
| `--warn` text colour is `#8A5A12`; the specified `#B7791F` is kept as `--warn-line` (borders, icons) | `#B7791F` on the warning tint is only 3.22:1, below AA for text. |
| `--ok-tint` lightened to `#EEF7F1` | `#2E7D4F` on the first tint (`#E3F1E8`) was 4.33:1; now 4.61:1. |
| Confidence bars are native `<progress>` elements | Accessible, work without JS, no inline `style` (CSP). `result.js` animates `value` once (skipped for reduced motion). |
| Phone menu: nav is always visible without JS; with JS (`html.js`) a Menu button with `aria-expanded` toggles it; Escape closes it | Progressive enhancement without a duplicate `<details>` nav. |
| Theme toggle cycles system → light → dark; stored in `localStorage` (try/catch); applied by an inline `<head>` script before paint | No flash of the wrong theme; still works if storage is blocked. |
| Client-side resize to 1024 px JPEG 0.9 *before* the 8 MB size check | A 12 MB phone photo becomes ~200–400 KB and is accepted instead of rejected. The server still enforces 8 MB. Small images (<1024 px and <1 MB) are sent untouched. |
| Camera captures go through the same `handleFile()` path and land in the real `<input type=file>` via `DataTransfer` | One upload path; the normal form submit sends it. The old code never attached the capture. |
| Submit disabled until a photo is chosen (JS only), `aria-busy` + spinner + "Analysing leaf…", double-submit guard, reset on `pageshow` | No double POSTs; the Back button doesn't leave a stuck spinner. |
| Trust strip shows **38 leaf conditions** (26 diseases + 12 healthy), not 39 | The 39th class is "no leaf in the photo", which isn't a condition. All numbers come from `catalog.py`. |
| Local SVG art for crops/products, generated by `scripts/make_assets.py` | The hot-linked images (healthline, webmd, gstatic shopping thumbnails, dead flaticon) break; no copyrighted images. |
| Reference photo on the result page still hot-linked (from the CSV), lazy, hidden on error | It's the only condition-specific photo available. During testing some hosts block hot-linking (`ERR_BLOCKED_BY_RESPONSE`), and the figure correctly disappears. |
| Footer "Source on GitHub" points to the original repository via `REPO_URL` | Your fork's URL isn't known; set `REPO_URL` (or edit `config.py`). |
| Removed: Bootstrap 5 beta + Bootstrap 4 JS + jQuery + ionicons + bttn.css, `/mobile-device`, "open on laptop only" notice, contact page with personal profiles, KrsikX links, emoji headings | See the audit in the brief; all replaced by the above. |

## Contrast ratios (WCAG AA needs 4.5:1 for body text)

Computed from the tokens in `app.css` with the WCAG relative-luminance formula.

| Pair | Use | Light | Dark |
|---|---|---|---|
| `--ink` on `--bg` | Body text | 14.68:1 | 15.32:1 |
| `--ink` on `--surface` | Text on cards | 16.01:1 | 14.04:1 |
| `--ink-muted` on `--bg` | Muted text | 5.54:1 | 8.12:1 |
| `--ink-muted` on `--surface` | Muted text on cards | 6.04:1 | 7.44:1 |
| `--ink-muted` on `--surface-2` | Footer / alt-section muted text | 5.15:1 | 6.70:1 |
| `--brand` on `--bg` | Links, trust numbers | 8.42:1 | 8.79:1 |
| `--brand` on `--surface` | Links on cards | 9.18:1 | 8.06:1 |
| `--brand-ink` on `--brand` | Primary button text | 9.18:1 | 8.65:1 |
| `--brand` on `--brand-tint` | Current nav item, crop tags | 7.73:1 | 6.94:1 |
| `--ok` on `--ok-tint` | "Healthy" chip | 4.61:1 | 7.36:1 |
| `--bad` on `--bad-tint` | "Disease detected" chip | 4.60:1 | 6.79:1 |
| `--bad` on `--surface` | Form error text | 5.52:1 | 7.33:1 |
| `--warn` on `--warn-tint` | Warning icon | 5.24:1 | 8.06:1 |
| `--ink` on `--warn-tint` | Warning / demo banner text | 14.18:1 | 12.73:1 |
| `--ink` on `--brand-tint` | Info banner text | 13.47:1 | 12.10:1 |

`--accent` (`#7FB24A`, 2.3:1 on paper) is only used for decoration (list markers, illustrations), never text.

## Latency (CPU)

`scripts/benchmark.py`, 20 predictions of one test image, this laptop (16 logical CPUs, torch 2.9.1 CPU,
4 threads): **median 23.1 ms** (min 22.2, max 34.9) for preprocess + forward pass + softmax.
Measured in demo mode, but random weights don't change the amount of computation, so this is
representative. Model warm-up at startup: 28 ms. "< 1 s" on the home page is conservative.

## Accuracy smoke test

`scripts/benchmark.py` runs all 43 images in `test_images/` and checks the predicted class against the
filename (spelling mistakes in filenames like `starwberry`, `soyaben`, `ceder`, `northen` are mapped;
each filename was verified to match exactly one class).

**Not yet measured with the real model**: the weights file was unavailable, so the run here used random
weights (1/43 "matches", i.e. chance). After downloading the weights, run:

```powershell
.\.venv\Scripts\python scripts\benchmark.py
```

and paste its Markdown table here. Template:

| File | Predicted class | Confidence | Match |
|---|---|---|---|
| Apple_scab.JPG | *(run benchmark.py)* | | |

## Quality checks performed

- **pytest**: 28 passed, 5 skipped (the 5 "expected class in the HTML" assertions skip themselves in demo
  mode; they run automatically once weights are present). Covers every page, 301/307 redirects, 5 real
  uploads, RGBA/greyscale PNGs, missing/empty files, `.txt` renamed `.jpg` (400), wrong extension,
  path-traversal filename, 9 MB upload (413), API schema and JSON errors (400/413/405), security headers,
  no third-party assets, catalog flags.
- **HTML structure** (`scripts/check_html.py`): every page incl. result and 404: no unclosed or mismatched
  tags, no duplicate ids, every `<img>` has `alt`, no inline styles.
- **Playwright** (`scripts/screenshots.py`): 30 screenshots (375 px and 1440 px, light and dark, every page,
  diagnose with a photo selected, result, open phone menu). No horizontal overflow at 320, 375 or 1440 px.
  Only console errors: the intentional 404 page, and a reference photo whose host blocks hot-linking (hidden).
- **Interaction tests** (Chromium with a fake camera): camera opens, captures, "Use photo" closes the dialog,
  stops the stream, puts `camera-photo.jpg` in the file input and focuses Submit; a 4000×3000 PNG is resized
  to 1024×768 JPEG in the browser; a `.txt` shows the inline error; treatments filters combine correctly
  (Fertilisers + Tomato = 1, "blight" = 2); with JavaScript disabled the plain form uploads and shows a
  result and the nav is visible; keyboard Tab order: skip link → logo → nav → theme → file input → camera → footer.
- **Sizes**: `app.css` 22.3 KB; `diagnose.js` 9.4 KB, `treatments.js` 1.7 KB, `result.js` 1.5 KB,
  `theme.js` 1.7 KB (all uncompressed).
- **Lighthouse: not run** (no Node.js). To run it yourself: install Node LTS, start the app, then
  `npx lighthouse http://127.0.0.1:5000/ --form-factor=mobile --view`.

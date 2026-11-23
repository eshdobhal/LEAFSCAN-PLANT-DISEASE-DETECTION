# LeafScan — design & engineering decisions

Every non-obvious choice made while rebuilding the original Plant-Disease-Detection Flask app as
**LeafScan**, with the measurements behind them.

## Environment notes

- **Python 3.13 instead of 3.11.** This machine only has Python 3.10 and 3.13 (`py -0`). The code uses
  nothing newer than 3.11, and `scripts/make_assets.py` deliberately avoids 3.12-only f-string syntax.
- **The original CNN's weights were never available** (not in the repo, not on disk), so it is only
  exercised in demo mode (seeded random weights). The app's default model is now the **hybrid** trained
  here (see "Hybrid model"), whose weights ship in the repo as `Flask Deployed App/leafscan_hybrid.pt` (224 px) and
  `leafscan_hybrid_288.pt` (288 px); the app averages the two (an ensemble).
- **Training hardware:** NVIDIA RTX 3050 Laptop GPU (6 GB), PyTorch 2.9.1 + CUDA 12.6, in a separate
  `.venv-train` so the app's CPU-only environment stays small.
- **Node.js is not installed**, so Lighthouse (`npx lighthouse`) could not be run. See "Quality checks".

## Backend

| Decision | Why |
|---|---|
| Split into `config.py`, `catalog.py`, `model.py`, `app.py` | Each file has one job and fits on a screen; easy to explain in a viva. |
| `create_app()` factory | Tests, gunicorn, waitress and `flask run` all build the app the same way; nothing runs at import time. |
| `MODEL_PATH` defaults to whichever of `plant_disease_model_1_latest.pt` / `plant_disease_model_1.pt` exists | The original README and code disagreed on the name. |
| Missing weights → loud `ERROR` log + demo mode + banner; `REQUIRE_MODEL=1` makes it fatal | Lets the UI be developed and tested without the weight files, while production can refuse to start. |
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
| Footer shows "Designed and built by Esh Dobhal · © 2026" and a "Source code" link only when `REPO_URL` is set | Author and year live in `config.py` (`AUTHOR`, `COPYRIGHT_YEAR`). The repository URL isn't known yet, so the link is hidden rather than pointing somewhere wrong. |
| "Try a sample" on the Diagnose page: four bundled photos, posted by name to `/diagnose/sample` | Lets anyone (an examiner, a visitor without a sick plant) see a real diagnosis in one click. Three are from the PlantVillage held-out test set and one is a web photo outside all datasets, so none were seen in training; the lab ones are copied byte-for-byte so the results match the benchmark. Names are checked against `SAMPLES`, never used as paths. |
| Rate limit: 30 model-running requests per client IP per minute, in memory, sliding window | Each diagnosis costs ~0.2 s of CPU; without a limit one script could keep the single worker busy. In-memory is correct with one process (a shared store like Redis would be needed with several). Pages and static files are never limited. `TRUST_PROXY=1` is required behind a hosting proxy, otherwise every user shares the proxy's IP and one limit. |
| About page states measured accuracy (99.6% lab, 71% field, 84% when sure) | Users and examiners should see the honest field number, and why the "not sure" banner matters. Shown only when the ensemble is loaded, since the numbers belong to it. |
| Live hosting: browser version on GitHub Pages | Checked in November 2026: Hugging Face now requires PRO for Gradio and Docker Spaces (only static Spaces are free); 512 MB free servers are too small (measured ~600 MB even with one model); other hosts need a card. So the models are exported to ONNX and run in the visitor's browser with ONNX Runtime Web: free, no sleep, unlimited visitors, photos stay on the device. Weights are stored as fp16 in the ONNX file and cast to fp32, so results match PyTorch (225/225 top-1, max difference 6e-6). End to end the page agrees with the server on 219/225 field photos (the browser's canvas resizing differs slightly from Pillow's); accuracy 72.0% vs 71.1%. Costs: ~93 MB first download (then cached), no JSON API. |
| Hosting first planned: Hugging Face Spaces (Docker, free CPU tier) — superseded | Both models need ~700 MB RAM; Render's free tier has 512 MB, and Railway/Fly/Cloud Run need a card. Spaces' free tier has 2 vCPU and 16 GB, HTTPS, and suits ML demos. The Space's Dockerfile clones a tagged GitHub release instead of holding a copy, so the Space runs exactly a version that passed CI and the 49 MB weights never go through Hugging Face's large-file storage. Trade-offs: the Space sleeps after 48 h idle, and 2 vCPUs are slower than the laptop. |
| `FRAME_ANCESTORS` setting, default `'none'` | The Space page shows the app in an iframe on huggingface.co, which `frame-ancestors 'none'` blocks. Only the Space deployment allows that one origin; everywhere else framing stays forbidden. |
| Browser tests (`tests/test_e2e.py`) are opt-in (`E2E=1`) | They need Chromium and take ~15 s; the default `pytest` stays fast and dependency-light. |
| Visible credits name only the author; third-party attributions live in `THIRD_PARTY_NOTICES.md` | Requested by the author. Datasets (PlantDoc is CC BY 4.0), icons (ISC), fonts (OFL), timm (Apache-2.0) and the origin of `CNN.py`, the CSVs and `Model/` still need attribution under their licences, so they are listed there instead of in the footer. |
| Removed: Bootstrap 5 beta + Bootstrap 4 JS + jQuery + ionicons + bttn.css, `/mobile-device`, "open on laptop only" notice, contact page with personal profiles, KrsikX links, emoji headings; later also the unused old templates (`index.html`, `market.html`, `submit.html`, `contact-us.html`) and the old UI screenshots in `demo_images/` | See the audit in the brief; all replaced by the above. |

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

## Hybrid model (CAFormer-S18)

### Why this model

Research summary (sources in the README): on PlantVillage, many CNNs and hybrids already exceed
98-99%, so that number no longer separates models. The real gap is **field photos**: models trained only
on PlantVillage drop to roughly 11-41% on PlantDoc field images, and the best published cross-domain
result found was about 68%. What helps: (1) pre-training on a large general dataset, (2) training on
some field images, (3) heavy augmentation, (4) architectures that combine local texture (convolution)
with global context (attention).

**CAFormer-S18** (MetaFormer family, Yu et al., 2022; timm `caformer_s18.sail_in22k_ft_in1k`) is a true
hybrid: two convolutional stages (local lesion texture) followed by two self-attention stages (relations
across the whole leaf). 24.4 M parameters, ImageNet-22k pre-training, fits a 6 GB GPU at batch 32 with
bf16 autocast, and runs on CPU in about 35 ms per view. Larger hybrids (MaxViT, CoAtNet) were ruled out by
the 6 GB GPU and CPU latency on a free server.

### Data (built by `training/prepare_data.py`)

| Split | Images | Source |
|---|---|---|
| train | 42,029 | PlantVillage official leaf-grouped train list (90%), background (70%), PlantDoc train (85%) |
| val_lab | 4,473 | 10% of the PlantVillage train list + 10% of background (only for picking the epoch) |
| val_field | 351 | 15% of PlantDoc train (only for picking the epoch) |
| **test_lab** | **10,937** | PlantVillage official **leaf-grouped** test list + 20% of background |
| **test_field** | **225** | PlantDoc test, minus 11 files byte-identical to PlantDoc training images |

- **No leakage between photos of the same leaf**: PlantVillage contains several photos of each physical
  leaf. Its maintainers publish a train/test split grouped by leaf (`splits/color_*.txt` on Hugging Face),
  used as-is. A random split (as in the original project) inflates test accuracy.
- **PlantDoc classes mapped** to the app's 39 classes (27 appear in PlantDoc). PlantDoc has no Orange,
  Grape esca/leaf blight, Strawberry scorch, Peach/Cherry disease or Target spot photos, so those are
  learned from lab photos only.
- **Background class** ("no leaf"): 1,143 images from the Mendeley PlantVillage release. Only those images
  were fetched, with HTTP range requests (`training/fetch_zip_subset.py`), because the connection here
  (about 0.5 MB/s with frequent drops) could not finish 1-2 GB downloads in one go.

### Training (`training/train_hybrid.py`)

- AdamW, lr 2e-4, weight decay 0.05, 1 epoch linear warm-up then cosine decay, 8 epochs, batch 32,
  bf16 autocast, gradient clipping 1.0, label smoothing 0.1 (better-calibrated confidence).
- Augmentation: RandomResizedCrop(0.35-1.0), horizontal + vertical flips, TrivialAugmentWide, Gaussian
  blur, RandomErasing: aimed at making lab photos look more like messy field photos.
- **Sampling:** 30% of each epoch's draws are field photos, 70% lab; within a domain each class is
  weighted by 1/sqrt(size), lifting rare classes (recall) without discarding the variety of big ones.
- **First attempt discarded:** the first run multiplied field weights per class, so about 80% of draws
  came from the ~2,000 field photos and lab validation accuracy stalled at 95% (log kept in
  `training/runs/train_v1_unbalanced.log`). Fixed and retrained from scratch.
- The epoch with the best mean of lab and field validation macro-recall is kept (epoch 7), saved in fp16
  (49 MB). Training time: 72 minutes on the RTX 3050.

| Epoch | Train loss | Lab val acc | Lab val recall | Field val acc | Field val recall |
|---|---|---|---|---|---|
| 1 | 1.325 | 94.52% | 93.67% | 62.39% | 60.19% |
| 2 | 1.070 | 97.32% | 96.83% | 67.52% | 66.55% |
| 3 | 0.963 | 97.47% | 97.23% | 69.23% | 67.85% |
| 4 | 0.878 | 98.75% | 98.77% | 67.52% | 63.48% |
| 5 | 0.802 | 99.02% | 98.99% | 71.23% | 67.43% |
| 6 | 0.749 | 99.37% | 99.19% | 75.21% | 72.18% |
| 7 | 0.714 | 99.64% | 99.56% | 76.07% | 73.02% |
| 8 | 0.704 | 99.73% | 99.65% | 76.07% | 72.94% |

### Results of the first (224 px) model on the held-out test sets (`training/evaluate.py`)

These are the single 224 px model's numbers; the shipped ensemble is in the next section.

| Test set | Images | Accuracy | Macro recall | Macro precision | Macro F1 | >=80% confident | Accuracy when confident |
|---|---|---|---|---|---|---|---|
| Lab (PlantVillage, leaf-grouped) | 10,937 | 99.40% | 99.27% | 99.32% | 99.29% | 99% | 99.71% |
| Lab + mirror TTA | 10,937 | **99.45%** | **99.34%** | 99.31% | 99.32% | 99% | 99.72% |
| Field (PlantDoc test) | 225 | 68.00% | 67.30% | 70.42% | 66.22% | 74% | 78.44% |
| Field + mirror TTA | 225 | **68.44%** | **67.76%** | 70.89% | 66.62% | 70% | 80.89% |

How to read this:
- **Lab**: 99.3% macro-recall means that, averaged over conditions, the model finds 99.3% of the photos
  of each one. Lowest: Corn Cercospora/gray leaf spot 94.4%, Peach healthy 97.2%, Corn northern leaf
  blight 97.3%.
- **Field** is the honest number for a phone photo in a garden. 68% is in line with the best published
  PlantDoc classification results found (around 68-70%) and far above PlantVillage-only training
  (11-41%). The author's earlier ResNet-18 (PBL project) scored 0.544 on field photos (different 21-class set, so only roughly
  comparable). The field test has only 225 images (3-12 per class), so the 95% confidence interval is
  about +/-6 percentage points.
- **The 80% threshold works as intended in the field**: when the model is at least 80% sure (70% of
  field photos) it is right 81% of the time; the other 30% get the "not sure, retake" banner.
- Weakest field classes: Tomato mosaic virus 2/10, Corn gray leaf spot 1/3, Tomato healthy 3/8, Cherry
  healthy 4/10. Full per-class numbers: `training/runs/caformer_s18/test_metrics.json`.

**"Best in the market" is not a claim this project can honestly make**: there is no shared field benchmark
across commercial apps, and PlantDoc's test set is small. What can be claimed: on a leak-free split the
hybrid matches the top of published lab results and sits in the top range of published field results, and
the evaluation code is included so anyone can re-run it. (The shipped ensemble reaches 71% on the
same field test, see below.)

### Second model at 288 px, and the shipped ensemble

**Why:** small lesions (bacterial spot, early mosaic mottling) cover few pixels at 224 px; a larger input
was the cheapest next step on the "what would improve" list.

**Training:** same script and recipe, `TRAIN_SIZE=288` (resize 329, crop 288), batch 20 (batch 24 peaked at
5.2 of 6 GB), learning rate 1.5e-4 (scaled down with the batch), 8 epochs, 117 minutes. Best epoch 8.
Run with `--no-copy` so it could not replace the app's model before it was compared.

| Epoch | Train loss | Lab val acc | Lab val recall | Field val acc | Field val recall |
|---|---|---|---|---|---|
| 1 | 1.286 | 95.80% | 94.55% | 68.09% | 65.68% |
| 2 | 1.052 | 95.75% | 95.61% | 65.24% | 63.58% |
| 3 | 0.950 | 98.46% | 98.06% | 72.36% | 69.24% |
| 4 | 0.867 | 98.84% | 98.81% | 70.09% | 65.97% |
| 5 | 0.792 | 99.08% | 99.14% | 74.93% | 71.20% |
| 6 | 0.743 | 99.55% | 99.44% | 76.07% | 72.30% |
| 7 | 0.711 | 99.75% | 99.53% | 78.06% | 74.59% |
| 8 | 0.702 | 99.73% | 99.52% | 77.78% | 74.72% |

**On its own it was not better on the field test** (`training/runs/caformer_s18_288/test_metrics.json`,
with TTA): accuracy 68.44% (same as 224 px), macro recall 67.82% (vs 67.76%), but lower precision (67.3% vs
70.9%) and lower accuracy when confident (77.0% vs 80.9%). Only lab recall improved (99.56% vs 99.34%).
It also costs about 1.65x the CPU time.

**But the two models make different mistakes**, so averaging their probabilities (an ensemble) was tried.
To keep the test sets honest, the choice between "224", "288" and "ensemble" was made on the
**validation** sets (`training/compare_models.py`), and the test sets were then run once:

| Validation (with TTA) | Field acc | Field recall | Field acc when >=80% sure | Lab acc | Lab recall |
|---|---|---|---|---|---|
| 224 px | 76.92% | 73.21% | 84.42% | 99.66% | 99.60% |
| 288 px | 77.78% | 74.39% | 83.92% | 99.84% | 99.78% |
| **Ensemble** | **78.63%** | **74.79%** | **87.50%** | 99.82% | 99.74% |

The ensemble won on field photos, so it is the shipped model. Its **test** results
(`training/evaluate.py --runs caformer_s18:224 caformer_s18_288:288 --name ensemble`, output in
`training/runs/ensemble/test_summary.md`):

| Test set | Images | Accuracy | Macro recall | Macro precision | Macro F1 | >=80% confident | Accuracy when confident |
|---|---|---|---|---|---|---|---|
| Lab (PlantVillage, leaf-grouped) + TTA | 10,937 | **99.63%** | **99.53%** | 99.45% | 99.49% | 99% | 99.85% |
| Field (PlantDoc test) + TTA | 225 | **71.11%** | **70.36%** | 72.86% | 69.27% | 63% | 83.69% |

Compared with the single 224 px model: **+2.7 points field accuracy, +2.6 points field macro recall,
+2.0 points field precision**, +0.2 points lab accuracy. With only 225 field test images the 95% interval
is still about +/-6 points, so the gain is suggestive rather than proven; but validation and test agree
in direction, which is why it was kept.

- **The 80% threshold is now more conservative**: 63% of field photos pass it, and those are right 84% of
  the time; the other 37% get the "not sure, retake" banner.
- Weakest field classes are unchanged in kind: Tomato mosaic virus 2/10, Corn gray leaf spot 1/3, Tomato
  bacterial spot 3/8, Potato early blight 3/7.
- **Cost:** two networks instead of one: about 190 ms per photo on the CPU instead of 70 ms, and 98 MB of
  weights instead of 49 MB. `ENSEMBLE=0` (or deleting `leafscan_hybrid_288.pt`) goes back to the single
  224 px model.

### What would improve field accuracy further

More field photos (the biggest factor by far: PlantDoc has only ~2,000), a third ensemble member of a
different architecture (e.g. a ConvNeXt or EfficientNet, so the mistakes are less correlated), or
distilling the ensemble into one network to get its accuracy at single-model speed.

## Latency (CPU)

`scripts/benchmark.py`, 20 predictions of one image, this laptop (16 logical CPUs, torch 2.9.1 CPU,
4 threads):

| Model | Median | Notes |
|---|---|---|
| Hybrid ensemble 224 + 288 px + mirror TTA (default) | **190 ms** (min 184, max 210) | 2 networks x 2 views |
| Single 224 px hybrid + mirror TTA (`ENSEMBLE=0`) | 70 ms (min 66, max 83) | 2 views per photo |
| Original CNN (random weights, same compute) | 23.1 ms | demo mode only |

Re-measured later on the same laptop and code: ensemble 359 ms, single 224 px model 139–147 ms, single
288 px model 223 ms. The difference is the laptop's power profile (about 2× CPU speed), not the code, so
the README quotes both ends of the range. Set `TTA=0` to roughly halve either. "< 1 s" on the home page remains true with margin. (A first
ensemble measurement of 416 ms was taken while a GPU job's data loaders were still busy; the repeated
runs above are the reliable figures.)

## `test_images/` sanity check

`scripts/benchmark.py` with the shipped ensemble: **41/43** filenames matched (same as the single model).
**This is not a fair accuracy measure**: file hashes show 24 of the 43 images are in our training set, 6 in
validation, 9 in the lab test set, and only 4 are not from PlantVillage. Both misses are web images outside
PlantVillage: `tomato-mold.jpg` is predicted Tomato late blight at 36.9% (the single model said 53.5%; either
way the "not sure" banner shows), and `tomato_yellow_leaf_curl_virus2.jpg` is predicted Soybean healthy at
92.3%, a confident mistake.

| File | Predicted class | Confidence | Match |
|---|---|---|---|
| apple_black_rot.JPG | Apple___Black_rot | 90.0% | yes |
| Apple_ceder_apple_rust.JPG | Apple___Cedar_apple_rust | 90.2% | yes |
| apple_healthy.JPG | Apple___healthy | 89.6% | yes |
| Apple_scab.JPG | Apple___Apple_scab | 90.2% | yes |
| background_without_leaves.jpg | Background_without_leaves | 89.9% | yes |
| blueberry_healthy.JPG | Blueberry___healthy | 90.5% | yes |
| cherry_healthy.JPG | Cherry___healthy | 90.3% | yes |
| cherry_powdery_mildew.JPG | Cherry___Powdery_mildew | 90.2% | yes |
| corn_cercospora_leaf.JPG | Corn___Cercospora_leaf_spot Gray_leaf_spot | 89.2% | yes |
| corn_common_rust.JPG | Corn___Common_rust | 90.1% | yes |
| corn_healthy.jpg | Corn___healthy | 90.2% | yes |
| corn_northen_leaf_blight.JPG | Corn___Northern_Leaf_Blight | 89.9% | yes |
| grape_black_rot.JPG | Grape___Black_rot | 90.2% | yes |
| Grape_esca.JPG | Grape___Esca_(Black_Measles) | 90.3% | yes |
| grape_healthy.JPG | Grape___healthy | 90.7% | yes |
| grape_leaf_blight.JPG | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 90.2% | yes |
| orange_haunglongbing.JPG | Orange___Haunglongbing_(Citrus_greening) | 90.3% | yes |
| peach_bacterial_spot.JPG | Peach___Bacterial_spot | 90.3% | yes |
| peach_healthy.JPG | Peach___healthy | 90.1% | yes |
| pepper_bacterial_spot.JPG | Pepper,_bell___Bacterial_spot | 90.1% | yes |
| pepper_bell_healthy.JPG | Pepper,_bell___healthy | 90.3% | yes |
| potato_early_blight.JPG | Potato___Early_blight | 90.5% | yes |
| potato_healthy.JPG | Potato___healthy | 89.6% | yes |
| potato_late_blight.JPG | Potato___Late_blight | 90.3% | yes |
| raspberry_healthy.JPG | Raspberry___healthy | 90.3% | yes |
| soyaben healthy.JPG | Soybean___healthy | 90.3% | yes |
| squash_powdery_mildew.JPG | Squash___Powdery_mildew | 90.3% | yes |
| starwberry_healthy.JPG | Strawberry___healthy | 90.4% | yes |
| starwberry_leaf_scorch.JPG | Strawberry___Leaf_scorch | 90.1% | yes |
| tomato-bacterial-spot2.jpg | Tomato___Bacterial_spot | 89.3% | yes |
| tomato-leaf-curl-virus3.jpg | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 90.1% | yes |
| tomato-mold.jpg | Tomato___Late_blight | 36.9% | no |
| tomato_bacterial_spot.JPG | Tomato___Bacterial_spot | 90.4% | yes |
| tomato_early_blight.JPG | Tomato___Early_blight | 90.2% | yes |
| tomato_healthy.JPG | Tomato___healthy | 90.3% | yes |
| tomato_late_blight.JPG | Tomato___Late_blight | 90.2% | yes |
| tomato_leaf_mold.JPG | Tomato___Leaf_Mold | 90.4% | yes |
| tomato_mosaic_virus.JPG | Tomato___Tomato_mosaic_virus | 90.3% | yes |
| tomato_septoria_leaf_spot.JPG | Tomato___Septoria_leaf_spot | 89.9% | yes |
| tomato_spider_mites_two_spotted_spider_mites.JPG | Tomato___Spider_mites Two-spotted_spider_mite | 90.5% | yes |
| tomato_target_spot.JPG | Tomato___Target_Spot | 89.4% | yes |
| tomato_yellow_leaf_curl_virus.JPG | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 90.3% | yes |
| tomato_yellow_leaf_curl_virus2.jpg | Soybean___healthy | 92.3% | no |

## Quality checks performed

- **pytest**: 65 passed with `E2E=1` (50 app/model tests + 15 real-browser tests; without `E2E` the
  browser tests are skipped). Earlier: 37 passed with the ensemble loaded (in demo mode the 5 "expected class in the HTML"
  assertions skip themselves). Also covers model selection (`MODEL_KIND=auto` falls back to the CNN; the
  ensemble is used only when the 288 px file exists and `ENSEMBLE` is on), the ensemble's top-3 output,
  and the hybrid's ImageNet normalisation at 224 and 288 px. Covers every page, 301/307 redirects, 5 real
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
  (Fertilisers + Tomato = 1; "blight" = 2 with Tomato, 6 across all crops); with JavaScript disabled the plain form uploads and shows a
  result and the nav is visible; keyboard Tab order: skip link → logo → nav → theme → file input → camera → footer.
- **Sizes**: `app.css` 22.3 KB; `diagnose.js` 9.4 KB, `treatments.js` 1.7 KB, `result.js` 1.5 KB,
  `theme.js` 1.7 KB (all uncompressed).
- **Lighthouse: not run** (no Node.js). To run it yourself: install Node LTS, start the app, then
  `npx lighthouse http://127.0.0.1:5000/ --form-factor=mobile --view`.

## Surveillance system (Project 05)

The brief asked for classification, damaged-tissue segmentation, pest detection and progression estimation,
benchmarked across model families, with a full evaluation package. Decisions, each with its reason:

| Decision | Reason |
|---|---|
| Acceptance tests (R1–R14) written before training (`docs/REQUIREMENTS.md`) | So the go/no-go cannot be tuned to the results; two tests (R6, R7) are reported as failed |
| PlantDoc-corrected for disease detection, not the original PlantDoc boxes | The corrected release removes 12 train/test duplicates and fixes class names and EXIF orientation |
| Tea dataset for pests, de-duplicated first | The only licensed, box-annotated field pest set found; 72% of its files were byte-identical copies, which would have inflated every metric |
| A rejected candidate: a synthetic pest set | It contained kitchen pests, not crop pests |
| RetinaNet-ResNet50-FPN is the default detector; Faster R-CNN behind `--detector fasterrcnn` | Brief names "YOLO or RetinaNet with ResNet"; RetinaNet passes R4, is best on tea at strict IoU and 2.3× faster than Faster R-CNN, but Faster R-CNN is clearly more accurate on PlantDoc (mAP@0.5 0.607 vs 0.489), so it is one flag away |
| YOLOv8s only as a benchmark | AGPL-3.0 conflicts with the redistributable-licence constraint, although it is 8× faster on a CPU |
| Faster R-CNN stands for the R-CNN family | It is R-CNN and Fast R-CNN with the slow external proposals replaced by a learned network; training R-CNN itself (selective search, one CNN pass per region) would teach nothing new at 100× the cost |
| DeepLabV3 for damaged tissue, Mask R-CNN as the instance benchmark | DeepLab beat FCN on every metric; Mask R-CNN gives true instance masks for the lesion-count question |
| Severity from an excess-green leaf estimate | No leaf masks exist; the same estimate is applied to the truth and to predictions, so the comparison is fair |
| Progression on simulated epidemics, with noise equal to the measured segmentation error | No public dataset photographs the same plants over a season; simulation shows the method, and the caveat travels with every forecast |
| Forecaster predicts the change from today, not the level | Persistence is a strong baseline; predicting the change starts every model from it, and both RNN and LSTM end more than 50% below its error |
| Pest detection only when the crop is declared as tea | Measured: 89% of photos of 13 other crops got a false pest alarm; the model learned "spots", not "pests" |
| Classifier skipped for tea | It was never trained on tea and answered "Cherry, healthy 35%" on the demo photo |
| Temperature scaling fitted on validation, checked on synthetic mis-calibration first | ECE must be fixed without touching the test set; the check proved the fitting code recovers T = 2.98 / 1.00 / 0.50 |
| Demo video from openly licensed photos only | PlantDoc images are scraped from the web, some watermarked; the demo uses PlantVillage (CC0), LeafNet75 (Apache-2.0) and tea (CC BY 4.0) |
| Synthetic scouting clip labelled SYNTHETIC in its file name, the video and every document | No openly licensed annotated field video exists; the label prevents it being mistaken for field evidence |
| Disclaimer on every result (web page, browser version, API `notice`, CLI notes) | Requirement R13; previously only the About page carried it |
| Temperature fitted as exp(log T) | A raw T overshot to a negative value when nearly every prediction is right (lab photos); found by a test written after the first evaluation run crashed |
| Generative models not used in the product | Measured: VAE and GAN leaves were no better than repeating the 25 real photos; the GAN mode-collapsed |

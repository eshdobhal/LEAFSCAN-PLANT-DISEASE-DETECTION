# LeafScan

**Leaf disease diagnosis in seconds.** Photograph a single leaf; LeafScan names the most likely of
26 diseases across 14 crops, says how confident it is, and gives practical next steps.

<img src="screenshots/home-desktop-light.png" alt="LeafScan home page" width="720">

<p>
  <img src="screenshots/diagnose-selected-mobile-light.png" alt="Diagnose page on a phone" width="220">
  <img src="screenshots/result-mobile-light.png" alt="Result page on a phone" width="220">
  <img src="screenshots/treatments-mobile-dark.png" alt="Treatments page in dark mode" width="220">
</p>

## Features

- **Mobile-first diagnosis**: choose a photo, drag and drop, paste from the clipboard, or use the
  live camera (back camera on phones). Photos are resized in the browser to 1024 px before upload.
- **Honest results**: top-1 condition with its confidence, a top-3 bar chart, and a warning when the
  model is less than 80% sure.
- **Care steps** for each condition (or growing tips for healthy plants) and one suggested product.
- **Treatments page** with instant search, crop filter and treatment/fertiliser toggle.
- **Works without JavaScript** (plain form), keyboard-accessible, light/dark theme, print stylesheet.
- **JSON API** at `/api/predict` and a health check at `/healthz`.
- **Private by design**: uploads are processed in memory and never written to disk.
- No CDNs, no frameworks: Flask + Jinja, one CSS file, small vanilla JS files.

## Quick start (Windows PowerShell)

```powershell
cd Plant-Disease-Detection-main
py -3.13 -m venv .venv            # Python 3.11+ works; use whichever you have
.\.venv\Scripts\Activate.ps1
pip install -r "Flask Deployed App\requirements.txt"
```

**Download the model weights** (≈200 MB, not stored in git): get `plant_disease_model_1_latest.pt`
from the original project's
[Google Drive folder](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link)
and put it in `Flask Deployed App\`. (A file named `plant_disease_model_1.pt` is also picked up.)
Without it the app still runs in **demo mode**, with a banner and random predictions.

```powershell
cd "Flask Deployed App"
flask --app app run               # http://127.0.0.1:5000
```

Debug mode with auto-reload: `$env:FLASK_DEBUG = "1"; flask --app app run`.

## Project structure

```
Flask Deployed App/
  app.py            create_app(): routes, upload validation, security headers, error pages
  config.py         every setting (env-overridable): model path, threshold, upload limit, app name
  model.py          loads the CNN once; preprocess() and Predictor.predict()
  catalog.py        reads both CSVs once into Condition objects (names, flags, text, product)
  CNN.py            the original network architecture and class list (unchanged)
  wsgi.py, Procfile production entry points
  disease_info.csv, supplement_info.csv
  templates/        base.html, partials/, home, diagnose, result, treatments, about, error
  static/           css/app.css, js/*.js, icons.svg, img/, fonts/, site.webmanifest
  tests/            pytest suite
scripts/            make_assets.py, screenshots.py, check_html.py, benchmark.py
screenshots/        375 px and 1440 px, light and dark, every page
DECISIONS.md        why things are the way they are
CODE_WALKTHROUGH.md file-by-file explanation
```

## How a prediction works

1. The photo is validated in memory (extension, Pillow decode, 8 MB limit) and rotated per its EXIF data.
2. It is converted to RGB, resized so the short side is 255 px, centre-cropped to 224 × 224 and turned
   into a tensor: exactly the training transform.
3. The CNN (4 convolution blocks + 2 dense layers) outputs 39 scores; softmax turns them into probabilities.
4. The top 3 classes and their probabilities are looked up in the catalog built from the CSVs.
5. If the best probability is below `CONFIDENCE_THRESHOLD`, the page warns that the model isn't sure.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `Flask Deployed App/plant_disease_model_1_latest.pt` | Weights file |
| `REQUIRE_MODEL` | off | `1` = refuse to start without weights (use in production) |
| `CONFIDENCE_THRESHOLD` | `0.80` | Below this, show the "not sure" warning |
| `FLASK_DEBUG` | off | `1` = debugger and auto-reload (never in production) |
| `REPO_URL` | original project | GitHub link in the footer |

The app name, tagline, upload limit (8 MB) and static-file version live in `config.py`.

## Testing

```powershell
pip install -r "Flask Deployed App\requirements-dev.txt"
python -m playwright install chromium        # only for screenshots
cd "Flask Deployed App"; python -m pytest; cd ..
python scripts\check_html.py                 # tag balance, duplicate ids, alt text, inline styles
python scripts\benchmark.py                  # accuracy table + median latency on test_images/
python scripts\screenshots.py                # writes screenshots/ and reports horizontal overflow
```

## Deployment

**Render / Railway (Linux):** set the root directory to `Flask Deployed App`, build command
`pip install -r requirements.txt`, start command from the `Procfile`:

```
gunicorn "app:create_app()" --workers 1 --threads 4 --timeout 60
```

One worker means the model is loaded once (~200 MB RAM); four threads serve requests concurrently.
Put the weights on the server (persistent disk or download at build time) and set `REQUIRE_MODEL=1`.

**Windows server:** gunicorn doesn't run on Windows; use waitress (already in requirements):

```powershell
cd "Flask Deployed App"
waitress-serve --listen=0.0.0.0:8000 --threads=4 --call app:create_app
```

The live camera needs HTTPS (or `localhost`); put the app behind an HTTPS proxy in production.

## Limitations

- Trained on PlantVillage lab photos (single leaf, plain background); field photos are harder.
- Only the 38 listed leaf conditions are known; anything else is mapped to the closest one.
- Product suggestions and reference photos come from the original project's CSVs and are not verified;
  reference photos are hot-linked and are hidden automatically if their host blocks them.
- An automated first opinion, not professional advice.

## Credit and licence

Model and dataset pipeline based on the open-source
[Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection) project by
Manthan Bhikadiya & Krishna Baldaniya. The original training notebook is in `Model/`.
Fonts: Fraunces and Inter (SIL Open Font License 1.1). Icons adapted from Lucide (ISC).
Crop and product illustrations were drawn for this project (`scripts/make_assets.py`).
Check the original repository's licence terms before redistributing the model weights.

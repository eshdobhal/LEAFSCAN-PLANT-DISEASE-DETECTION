# LeafScan user guide

*Author: Esh Dobhal.* LeafScan comes in three forms. Pick the one that fits:

| You want to… | Use | Needs |
|---|---|---|
| Check one leaf on your phone, nothing to install | **Live browser version**: <https://eshdobhal.github.io/LEAFSCAN-PLANT-DISEASE-DETECTION/> | A modern browser; ~93 MB download on first visit |
| Run LeafScan for others on a server, or use the JSON API | **Web app** (`Flask Deployed App/`) | Python 3.11+, ~1 GB RAM |
| Scout many photos or a video: boxes, damaged tissue, severity, forecasts | **Surveillance CLI** (`surveillance/`) | Python 3.11+, the research environment, ~2.5 GB RAM |

---

## 1. Taking a good photo

- One leaf, filling most of the frame; daylight but not harsh sun; in focus.
- A plain background (your palm, a sheet of paper) helps: the classifier partly learned backgrounds.
- If LeafScan says **"The model isn't sure"** (confidence below 80%), retake the photo before acting.

## 2. Live browser version

1. Open the link above. The first visit downloads the AI model (progress bar); later visits start at once.
2. **Choose photo**, drag a photo in, paste one, **Use camera**, or try a **sample**.
3. The result shows the most likely condition, confidence, the top 3 possibilities, what to do, and a
   suggested product. **Treatments** lists every product with search and filters.

The photo is analysed on your device and never uploaded.

## 3. Web app (server)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "Flask Deployed App\requirements.txt"
cd "Flask Deployed App"
flask --app app run                     # http://127.0.0.1:5000
```

JSON API for other programs:

```powershell
curl.exe -F "image=@leaf.jpg" http://127.0.0.1:5000/api/predict
```

Settings (environment variables) are listed in the README, section 12.

## 4. Surveillance CLI

### Install

```powershell
py -3.13 -m venv .venv-train
.\.venv-train\Scripts\python -m pip install -r requirements-research.txt
```

Download the trained detection, pest, segmentation and forecast models (about 320 MB, each checked by
SHA-256) from the v1.1.0 release, or train them yourself (section 6). `status` shows what is available:

```powershell
.\.venv-train\Scripts\python -m surveillance.download_weights        # add --all for the benchmark models
.\.venv-train\Scripts\python -m surveillance.cli status
```

### Analyse photos

```powershell
.\.venv-train\Scripts\python -m surveillance.cli analyze field\leaf1.jpg field\leaf2.jpg --out reports
```

For each photo it writes `reports\<name>.json` (the full report) and `reports\<name>_annotated.png`:
red overlay = damaged tissue, blue boxes = diseased leaves, orange boxes = pest damage, and a summary panel.

Options:
- `--only classify,segment` runs only some components (`classify`, `detect`, `pests`, `segment`, `forecast`).
- `--device cpu` forces the CPU.
- `--detector fasterrcnn` uses Faster R-CNN for the diseased-leaf boxes: more accurate on field photos
  (mAP@0.5 0.61 vs 0.49) but about twice as slow. The default is RetinaNet.
- `--history field_history.csv` adds the 7-day severity forecast (below).
- `--crop tea` turns on **pest detection** (and turns off the disease classifier, which does not know tea).
  The pest model was trained on tea only; on other crops it mistook disease spots for pests in 89% of
  test photos, so without `--crop tea` it is skipped and the report says why.

**Exit codes** (for scripts and schedulers): `0` no alerts, `2` at least one alert, `1` error.

### Forecast from a field history

A CSV with one row per day, at least 14 rows, newest last:

```csv
day,severity,temperature_c,humidity_pct,rain_mm,spray
1,0.2,19.5,82,0,0
2,0.3,21.0,88,4.2,0
...
```

`severity` may be a fraction (0–1) or a percentage. `spray` is 1 on days a fungicide was applied. The
report then contains the predicted severity for each of the next 7 days and whether it will cross the 5%
action threshold. **The forecaster was trained on simulated epidemics**: treat it as a guide, not a fact.

A worked example with real outputs (photos, a tea photo, a forecast and a video) is in `demo/`.

### Scan a video

```powershell
.\.venv-train\Scripts\python -m surveillance.cli video walk.mp4 --every 15 --out reports
```

Every 15th frame is analysed; annotated frames go to `reports\walk_frames\` and a per-frame timeline
(label, confidence, severity, pests, alerts) to `reports\walk_timeline.json`.

## 5. Reading a report

| Field | Meaning | How far to trust it |
|---|---|---|
| `classification.label`, `confidence` | Most likely condition and its probability | ~71% right on field photos; ~84% right when ≥ 80% sure |
| `classification.low_confidence` | True below 80% | Retake the photo |
| `detections` | Boxes around diseased leaves (PlantDoc model, 13 crops) | mAP@0.5 0.49 (RetinaNet) or 0.61 (`--detector fasterrcnn`): a locator; the classifier names the disease |
| `pests` | Boxes around pest damage (only with `--crop tea`) | Trained on **tea** pests only; AP@0.5 0.96–0.98 on tea |
| `segmentation.severity_percent`, `severity_grade_0_to_5` | Share of the leaf that is damaged | Grade within ±1 for 87.5% of test photos; unreliable for tiny rust pustules and plain-backdrop lab photos |
| `forecast` | Severity for the next 7 days | Simulated training data: a guide only |
| `alerts` | Things that need attention | Always confirm with an extension officer before spraying |

## 6. Reproducing the models

```powershell
.\.venv-train\Scripts\python training\prepare_data.py           # classification splits
.\.venv-train\Scripts\python -m surveillance.prepare_data       # detection + lesion data
bash surveillance/run_experiments.sh                            # train and evaluate everything (GPU, hours)
.\.venv-train\Scripts\python -m surveillance.evaluate_all       # the evaluation package
.\.venv-train\Scripts\python -m pytest fundamentals\tests surveillance\tests
```

## 7. Responsible use

LeafScan gives an automated first opinion, not professional advice. It knows 38 leaf conditions and will
map anything else to the closest one. It never decides to spray: before applying any product, confirm the
diagnosis with a local agricultural extension officer and follow the product label.

# Model card: LeafScan Surveillance

*Author: Esh Dobhal. Covers every trained model in LeafScan: what it is for, how it was trained, how well it
works on held-out test data, and where it must not be used. Data details are in
[DATASET_CARD.md](DATASET_CARD.md); methods and the full analysis are in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
All numbers come from result files in the repository (paths given) and were measured once, on test splits
that were not used for training or model selection.*

## 1. Models at a glance

| Component | Model | Shipped in | Trained on | Headline test result |
|---|---|---|---|---|
| Disease classification | **CAFormer-S18 ensemble** (224 px + 288 px, mirror TTA) | web app, browser version, CLI | PlantVillage + PlantDoc (39 classes) | lab accuracy 99.63%; **field accuracy 71.11%, macro F1 0.693** |
| Disease detection | **RetinaNet, ResNet-50-FPN** | CLI | PlantDoc-corrected (29 classes, 13 crops) | **mAP@0.5 0.489**, mAP@[.5:.95] 0.376 |
| Pest detection | **RetinaNet, ResNet-50-FPN** | CLI (`--crop tea` only) | Tea field photos (6 classes, 2 pests) | **pest AP@0.5 0.96 / 0.98**, mAP@[.5:.95] 0.710 |
| Damaged-tissue segmentation | **DeepLabV3, ResNet-50** | CLI | LeafNet75 lesion masks | **IoU 0.581**, Dice 0.735, severity grade within ±1 for 87.5% |
| Lesion instances (benchmark) | Mask R-CNN, ResNet-50-FPN | research only | LeafNet75 (connected lesion regions) | mask mAP@0.5 0.576 |
| Progression forecast | **LSTM** (2 layers, 64 units) | CLI (`--history`) | **simulated** epidemics | 7-day MAE 1.68 points (persistence 4.64); alert F1 0.744 |
| Augmentation study | conditional VAE, conditional DCGAN | research only | 8-class PlantVillage subset, 64 px | ⟪PENDING:generative⟫ |
| Baselines | ResNet-50, GoogLeNet (classification); Faster R-CNN, YOLOv8s (detection); FCN (segmentation); vanilla RNN (forecast) | research only | same data as above | see §4 |

## 2. Intended use

- **Users:** growers, field scouts, agronomists and extension officers.
- **Use:** a fast first opinion on a leaf photo or a scouting video, which a person then checks. That
  means: which disease is most likely; where the diseased leaves and pest damage are; how much tissue is
  damaged; and whether severity is heading past the 5% action threshold in the next week.
- **Decisions it supports:** whether to look closer, which field to scout first, and whether to call an
  extension officer.
- **Decisions it must not make:** whether to spray, and which product or dose. The system never triggers
  an action itself.

## 3. Out-of-scope use (measured, not guessed)

| Misuse | What happens | Evidence |
|---|---|---|
| Pest detection on crops other than tea | **89% of photos of other crops get a false pest alarm** (210 of 236 PlantDoc test photos); the model learned "spots on a leaf" rather than the pest | `surveillance/results/evaluation.json` → `pest_out_of_domain`. The pipeline therefore runs the pest model only when the crop is declared as tea |
| Classifying crops or diseases outside the 39 classes (e.g. tea) | Maps the photo to the closest known class, usually with low confidence (35% on the tea demo photo) | `demo/`; the pipeline skips classification for tea |
| Severity on lab-style photos (plain backdrop, single leaf) | Can miss the lesions completely: a PlantVillage late-blight leaf was graded 0% | `demo/reports/plantvillage_potato_late_blight.json`; LeafNet75 masks are in-the-wild photos |
| Tiny pustules (rusts) | Often below the segmenter's resolution: rust leaves get near-0% severity | error analysis in the technical report |
| Forecasts for real fields | The LSTM has only seen simulated epidemics | `surveillance/progression.py` |
| Photos of fruit, stems, roots, whole plants or canopy | Not trained for any of these | — |

## 4. Performance

### 4.1 Classification (39 classes; `training/runs/*/test_summary.md`, `surveillance/results/evaluation.json`)

| Model | Params | Lab acc. | Field acc. | Field macro F1 | Field ECE raw → scaled | Answers ≥ 80% sure | Acc. when ≥ 80% sure |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CAFormer ensemble (shipped)** | 2 × 24.4 M | **99.63%** | 71.11% | 0.693 | ⟪PENDING:ece_ens⟫ | 63% | 83.69% |
| CAFormer-S18 224 | 24.4 M | 99.45% | 68.44% | — | ⟪PENDING:ece_224⟫ | — | 80.89% |
| ResNet-50 | 23.6 M | 99.42% | 70.67% | 0.698 | ⟪PENDING:ece_r50⟫ | 61% | 87.59% |
| GoogLeNet | 5.6 M | 96.48% | **72.00%** | **0.708** | ⟪PENDING:ece_goog⟫ | 40% | 87.64% |

Test sets: 10,937 lab photos (leaf-grouped PlantVillage split) and 225 real field photos (PlantDoc).
**On field photos the four models are statistically indistinguishable.** The 95% interval on 225 photos is
about ±6 points, and they span 68–72%. The ensemble is shipped because it was chosen on validation
before testing, and it gives the most useful confidence (63% of answers are ≥ 80% sure; GoogLeNet: 40%).
It is not shipped because it is more accurate on field photos.

**Robustness** (shipped ensemble, field photos, 16 corruptions): ⟪PENDING:robustness⟫

### 4.2 Detection (COCO mAP on the untouched test splits; `surveillance/runs/*/test_metrics.json`)

| Dataset | Model | mAP@[.5:.95] | mAP@0.5 | Pest AP@0.5 (mite / bug) | CPU ms | Params |
|---|---|---:|---:|---:|---:|---:|
| PlantDoc (236 photos) | **RetinaNet R50-FPN** | 0.376 | **0.489** | — | 458 | 36.9 M |
| PlantDoc | Faster R-CNN R50-FPN | ⟪PENDING:pd_frcnn⟫ | | — | | |
| PlantDoc | YOLOv8s | ⟪PENDING:pd_yolo⟫ | | — | | |
| Tea (406 photos) | **RetinaNet R50-FPN** | **0.710** | 0.988 | 0.964 / 0.975 | 284 | 36.5 M |
| Tea | Faster R-CNN R50-FPN | 0.694 | **0.989** | 0.972 / 0.977 | 935 | 43.3 M |
| Tea | YOLOv8s | 0.696 | 0.985 | 0.958 / 0.976 | **34** | 11.1 M |

PlantDoc's original paper reported mAP@0.5 0.389 (Faster R-CNN, Singh et al. 2020), but on the uncorrected
labels, so the two numbers are not directly comparable. Weakest PlantDoc classes: tomato mould leaf (AP 0.06),
bell-pepper leaf (0.11), tomato bacterial spot (0.11), potato early blight (0.14). These are small, similar
spots, where boxes are also ambiguous for people.

### 4.3 Segmentation and instances (LeafNet75, 88 test photos; `surveillance/runs/leafseg_*/test_metrics.json`)

| Model | Lesion IoU | Dice | Lesion F1 @ IoU 0.5 | Count MAE | Severity MAE | Grade exact / ±1 | CPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DeepLabV3-R50** | **0.581** | **0.735** | **0.290** | **2.61** | **4.9 pts** | 71.6% / **87.5%** | 526 |
| FCN-R50 | 0.535 | 0.697 | 0.278 | 3.08 | 5.5 pts | 71.6% / 86.4% | 428 |
| Mask R-CNN-R50 (instances) | — | — | mask mAP@0.5 0.576, mask mAP 0.250, box mAP@0.5 0.440 | | | | 778 |

Lesion-level F1 is low because neighbouring lesions merge or split between the ground truth and the
prediction. **Total damaged area** (IoU, severity) is far more reliable than **lesion counts**.

### 4.4 Progression forecasting (simulated; `surveillance/results/progression.json`)

| Model | MAE, all 7 days | MAE day 1 | MAE day 7 | Alert precision / recall / F1 |
|---|---:|---:|---:|---|
| Persistence (tomorrow = today) | 0.0464 | 0.0258 | 0.0705 | 0 / 0 / 0 |
| Logistic-curve extrapolation | 0.0862 | 0.0310 | 0.1475 | 0.33 / 0.41 / 0.36 |
| Vanilla RNN (tanh) | 0.0194 | 0.0125 | 0.0281 | 0.73 / **0.80** / **0.761** |
| **LSTM** | **0.0168** | **0.0108** | **0.0239** | **0.83** / 0.68 / 0.744 |

The LSTM is shipped for its lower error and fewer false alarms; the RNN catches slightly more crossings.

### 4.5 Deployment (CPU, 4 threads; `surveillance/results/evaluation.json`)

⟪PENDING:deployment⟫

## 5. Training details

| Model | Initial weights | Recipe | Hardware / time |
|---|---|---|---|
| CAFormer-S18 (224, 288) | ImageNet-22k → 1k (`timm`) | AdamW 2e-4 / 1.5e-4, cosine, 8 epochs, label smoothing 0.1, heavy augmentation, 30% field draws; selected on validation macro recall | RTX 3050 Laptop 6 GB: 72 + 117 min |
| ResNet-50, GoogLeNet | ImageNet-1k (torchvision) | same recipe as CAFormer 224 (`training/train_hybrid.py --arch`) | 34 min, 21 min |
| RetinaNet / Faster R-CNN | COCO (torchvision v2 weights) | SGD 0.01, momentum 0.9, wd 1e-4, warm-up + cosine, 16 epochs, batch 4, fp16 autocast, horizontal flip; image short side 512 | 36–110 min per run |
| YOLOv8s | COCO (`ultralytics`, AGPL-3.0) | ultralytics defaults, 640 px, 50 epochs, seed 0 | ~40 min |
| DeepLabV3 / FCN | COCO with VOC labels (torchvision) | AdamW 1e-4, cosine, 40 epochs, 384 px, class weights [1, 3], flips | 23 min, 20 min |
| Mask R-CNN (v2) | COCO | SGD 0.01, cosine, 30 epochs, instances = connected lesion regions > 20 px | 42 min |
| RNN / LSTM | random (seed 0) | Adam 2e-3, 40 epochs, 14-day look-back, predicts the change from today | minutes |

Validation splits choose epochs and models; test splits are run once, by the evaluation scripts.

## 6. Ethical considerations and responsible use

- **Wrong advice costs money and can harm crops and people.** Every result says it is a first opinion; low
  confidence (< 80%) triggers a "retake the photo" warning; product suggestions say to follow the label and
  local advice.
- **Shortcut learning:** Grad-CAM shows the classifier partly uses PlantVillage backdrops
  (`surveillance/results/conv_gradcam.png`). This is a major reason lab accuracy (99.6%) does not carry over to
  field photos (71%).
- **Geographic and crop bias:** most data come from North America (PlantVillage), India and China (tea);
  other regions, cultivars and growth stages are under-represented.
- **Privacy:** photos are processed in memory and never stored or logged; the browser version never uploads them.
- **Environmental cost:** all training here took about 12 GPU-hours on a 6 GB laptop GPU.
- **Licences:** shipped weights come only from CC0, CC BY 4.0 and Apache-2.0 data. YOLO (AGPL-3.0) was used
  only as a benchmark and is not part of the shipped application.

## 7. Caveats and recommendations

1. Retrain the pest detector on local pests before using it on any crop other than tea.
2. Validate the forecaster on real season-long severity series before trusting its numbers.
3. Collect more real field photos. That is the biggest lever on field accuracy: PlantDoc has only about 2,500.
4. Re-measure on a local test set before deployment in a new region; small test sets here mean ±6 points.

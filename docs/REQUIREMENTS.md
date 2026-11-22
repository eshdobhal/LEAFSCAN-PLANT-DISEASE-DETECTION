# LeafScan Surveillance: use case, requirements and acceptance tests

*Author: Esh Dobhal · Project 05, Crop Disease and Pest Surveillance System (precision agriculture)*

> **When these targets were set.** The detection, segmentation, progression, latency, memory and robustness
> targets below were written on 22 November 2026, **before** any of those models had been trained or
> measured, so the go/no-go decision in the technical report cannot be tuned to the results. The
> classification targets were set after the classifier existed; they are deliberately the targets a
> stakeholder would ask for, not its measured numbers.

## 1. Industry use case

**Scouting for disease and pests on smallholder and mid-size farms.** Today a grower or field scout walks
the crop, spots damaged leaves, and either guesses the cause or waits days for an extension officer.
Late or wrong diagnosis means sprays at the wrong time, with the wrong product, or not at all.

LeafScan Surveillance turns a phone photo (or a scouting video) into:
1. the most likely disease, with an honest confidence;
2. where on the plant the problem is (boxes around diseased leaves and pest damage);
3. how much tissue is damaged (severity %, grade 0–5);
4. where severity is heading over the next 7 days, and whether it will cross the action threshold.

## 2. Stakeholders and their needs

| Stakeholder | Need | What it means for the system |
|---|---|---|
| Smallholder grower | "What is wrong, and what do I do?" on a cheap phone, poor network | Fast, works offline-ish (browser version), plain language, care steps |
| Field scout / agronomist | Map where problems are and how bad, across many photos | Detection boxes, severity grade, batch CLI, video scanning, JSON reports |
| Extension officer | Trust the tool and know when not to | Calibrated confidence, "not sure" warnings, documented limits, explainability |
| Farm manager | Decide *when* to spray | Severity trend and 7-day forecast with an action-threshold alert |
| Operator / IT | Run it cheaply and safely | CPU-only, bounded memory, reproducible builds, tests, no data retention |
| Regulator / public | No harm from wrong advice | Disclaimers, no autonomous spraying decisions, privacy by design |

## 3. Measurable requirements and acceptance tests

| ID | Requirement (from need) | Acceptance test (pass if) | How it is measured |
|---|---|---|---|
| R1 | Correct disease on lab-quality photos | macro recall ≥ 95% on the leaf-grouped PlantVillage test set | `training/evaluate.py` |
| R2 | Useful on real field photos | accuracy ≥ 65% and macro F1 ≥ 0.60 on PlantDoc field test | `training/evaluate.py`, `surveillance/evaluate_all.py` |
| R3 | Honest confidence | field accuracy ≥ 80% when confidence ≥ 80%; ECE ≤ 0.10 (after temperature scaling) | `surveillance/evaluate_all.py` |
| R4 | Locate diseased leaves | PlantDoc detection mAP@0.5 ≥ 0.40 (the PlantDoc paper's best was 0.389) | `surveillance/detection.py eval` |
| R5 | Detect pest damage | pest AP@0.5 ≥ 0.50 for each pest class on tea field test photos | `surveillance/detection.py eval` |
| R6 | Measure damaged tissue | lesion IoU ≥ 0.60; severity grade within ±1 for ≥ 85% of test photos | `surveillance/segmentation.py eval` |
| R7 | Count lesions (instance level) | lesion-level F1 ≥ 0.50 at IoU 0.5 | `surveillance/segmentation.py eval` |
| R8 | Anticipate progression | LSTM 7-day MAE ≥ 30% lower than persistence; threshold-alert F1 ≥ 0.70 (simulated data) | `surveillance/progression.py` |
| R9 | Robust to phone photos | accuracy drop ≤ 15 points under mild corruptions (noise σ 0.05, blur 1, brightness ×0.5/×1.5, JPEG 20) | `surveillance/evaluate_all.py` |
| R10 | Fast enough in the field | classifier ≤ 1 s and full pipeline ≤ 3 s per photo on a 4-thread CPU | `surveillance/evaluate_all.py` |
| R11 | Cheap to host | web app ≤ 1 GB RAM; full pipeline ≤ 2.5 GB RAM | `surveillance/evaluate_all.py` |
| R12 | Explainable | Grad-CAM evidence maps available for the classifier | `surveillance/convviz.py` |
| R13 | Safe use | "not sure" warning below 80%; disclaimer on every result; no photo stored | `Flask Deployed App/tests/` (68 tests, 3 of them for exactly this) |
| R14 | Reproducible | pinned dependencies, fixed seeds, one-command test suites, documented data splits | CI + `README.md` |

## 4. Deployment constraints

- **Hardware:** CPU-only servers (no GPU at inference); laptops and phones for the browser version.
- **Memory:** a 1 GB instance must run the web classifier; the full pipeline targets ≤ 2.5 GB.
- **Network:** photos up to 8 MB; the browser version must work after a one-time model download.
- **Privacy:** photos processed in memory, never written to disk or logged.
- **Licences:** only datasets and weights whose licences allow redistribution of trained models
  (CC BY 4.0, Apache-2.0, MIT, CC0); AGPL tools (ultralytics YOLO) may be used for benchmarking but not shipped.
- **Human in the loop:** the system advises; it never triggers spraying by itself.

## 5. Traceability: project brief → evidence in this repository

| Brief item | Where it is done | Evidence |
|---|---|---|
| 1. Use case, requirements, acceptance tests, constraints | this document | §1–4 |
| 2. Data: acquisition, provenance, licensing, de-duplication, labels, leakage-safe splits | `training/prepare_data.py`, `surveillance/prepare_data.py`, [DATASET_CARD.md](DATASET_CARD.md) | `surveillance/results/data_report.json` |
| 3. Tensor sandbox: broadcasting, inner products, reshaping, SVD, gradient checking, visuals | `fundamentals/tensors.py`, `fundamentals/experiments/sandbox.py` | `fundamentals/results/sandbox_*` |
| 4. McCulloch-Pitts, sigmoid neurons, MLP with backprop, loss, parameter vs hyperparameter tracking | `fundamentals/neurons.py`, `fundamentals/mlp.py` | `fundamentals/results/mlp_baseline*` |
| 5. Batch GD, SGD, Momentum, Nesterov; initialisation, normalisation, regularisation, augmentation, overfitting | `fundamentals/optim.py`, `fundamentals/experiments/mlp_ablations.py` | `fundamentals/results/mlp_ablation_*` |
| 6. Hebbian, modified Hebbian, competitive, error-correction, Boltzmann | `fundamentals/hebbian.py`, `fundamentals/experiments/learning_rules.py` | `fundamentals/results/rules_*` |
| 7. Main product: classify, segment, detect pests, estimate progression; RetinaNet + ResNet; 2D/3D conv visuals | `surveillance/pipeline.py`, `surveillance/cli.py`, `surveillance/convviz.py` | `surveillance/results/conv_*` |
| 8. Benchmark classification, segmentation, instance analysis, detection (R-CNN family, YOLO, ResNet, GoogLeNet, RetinaNet) | `training/train_hybrid.py --arch`, `surveillance/detection.py`, `surveillance/segmentation.py` | `surveillance/results/evaluation.json` |
| 9. RNN/LSTM progression, GAN/VAE augmentation, latent space, step/sigmoid/tanh | `surveillance/progression.py`, `surveillance/generative.py`, `fundamentals/experiments/sandbox.py` | `surveillance/results/progression_*`, `generative_*` |
| 10. Integrated, validated application: macro F1, mAP, IoU, calibration, latency, robustness, error analysis, limits | `surveillance/evaluate_all.py`, [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | `surveillance/results/EVALUATION_SUMMARY.md` |

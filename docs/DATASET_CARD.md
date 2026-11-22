# Dataset card: LeafScan Surveillance

*Author: Esh Dobhal. Covers every dataset used for training or evaluation, how it was obtained, cleaned and
split, and what it cannot tell us. Image data are not stored in this repository (git-ignored `data/`); the
scripts below re-create every split exactly.*

## Summary

| Dataset | Task in LeafScan | Images used | Licence | Source |
|---|---|---:|---|---|
| PlantVillage (colour) | Classification (39 classes), fundamentals experiments, generative study | 54,303 | CC0 (Hughes & Salathé 2015) | [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage) (official leaf-grouped split lists) |
| PlantVillage background class | "No leaf in the photo" class | 1,143 | as released | [Mendeley Data tywbtsjrjv/1](https://data.mendeley.com/datasets/tywbtsjrjv/1) |
| PlantDoc (classification) | Field photos for classification training and test | 2,322 | CC BY 4.0 (Singh et al. 2020) | [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) |
| PlantDoc-corrected (detection) | Disease detection, 29 classes, 13 crops | 2,565 | CC BY 4.0 | [KeenForgeAI/PlantDoc-corrected](https://huggingface.co/datasets/KeenForgeAI/PlantDoc-corrected) (DOI 10.57967/hf/10678) |
| Tea leaf disease & pest detection | Pest detection (2 pests) + 4 other classes | 2,715 unique (of 9,591 files) | CC BY 4.0 | [Project-AgML/tea_leaf_disease_pest_detection](https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_pest_detection) (Lyu, Yu & Song, PLOS One 2026) |
| LeafNet75 lesion masks | Damaged-tissue segmentation, lesion instances, severity | 588 | Apache-2.0 | [LeafNet75/Leaf-Disease-Segmentation-masks](https://huggingface.co/datasets/LeafNet75/Leaf-Disease-Segmentation-masks) |
| Simulated epidemics | Progression forecasting (RNN/LSTM) | 1,100 fields × 90 days | generated here | `surveillance/progression.py` |

## Acquisition and provenance

- **PlantVillage colour images** were fetched selectively from the Hugging Face archive with HTTP range
  requests (`training/fetch_zip_subset.py`), because the full 1–2 GB downloads kept failing on the
  available connection. The dataset maintainers' **leaf-grouped** split lists were used unchanged.
- **PlantDoc** (classification) was taken from the authors' GitHub repository; filenames containing `?` or
  longer than 260 characters were renamed to `stem[:40]_md5[:8].ext` so they work on Windows.
- **PlantDoc-corrected** is a cleaned re-release of the PlantDoc detection set by KeenForgeAI: it removes
  12 train/test duplicates, normalises the 29 class names, fixes filenames and EXIF orientation, and keeps
  the official test split. Only the images and YOLO labels were downloaded (the VOC copies are redundant).
- **Tea** and **LeafNet75** arrive as Parquet files and are converted by `surveillance/prepare_data.py`.

## Cleaning and de-duplication (`surveillance/results/data_report.json`)

| Check | Finding | Action |
|---|---|---|
| PlantVillage: several photos of the same physical leaf | A random split would put the same leaf in train and test | Official leaf-grouped split used |
| PlantDoc classification: test files identical to training files | 11 byte-identical files | Removed from the test set |
| **Tea: exact duplicate photos** | **6,876 of 9,591 files (72%) are byte-identical copies**; all copies lie within one split, so no cross-split leakage | Only the first copy kept: 2,715 unique photos |
| Tea: classes too rare to learn or measure | "Brown blight" 2 boxes and "disease" 1 box after de-duplication | Those 3 boxes dropped, reported |
| LeafNet75 duplicates | none (MD5) | — |
| PlantDoc-corrected cross-split duplicates | none after the upstream fix (checked again by MD5) | — |
| PlantDoc-corrected label files the Hub cannot serve | 2 (one train, one validation; both stock photos of strawberry leaves) | Those 2 images excluded rather than trained as "no objects" |

## Labels

- **Classification:** 39 classes, in the order of the original project's `CNN.py` (26 diseases, 12 healthy
  states, 1 background). PlantDoc folder names are mapped onto them in `training/prepare_data.py`
  (27 of the 39 classes occur in PlantDoc).
- **Disease detection:** 29 PlantDoc classes (diseased and healthy leaves of 13 crops), boxes as released.
- **Pest detection:** 6 tea classes kept: black rot, leaf rust, **red spider mite (pest)**,
  **tea mosquito bug (pest)**, healthy tea leaf, white spot. Boxes mark the infested leaf or region.
- **Lesions:** binary masks (damaged tissue vs everything else). Lesion *instances* are the connected
  regions of the mask larger than 20 pixels (8-connectivity).
- **Severity:** share of leaf pixels that are lesion, graded 0 (< 1%), 1 (1–5%), 2 (5–10%), 3 (10–25%),
  4 (25–50%), 5 (≥ 50%). There are no leaf masks, so leaf area is estimated with an excess-green index plus
  the lesion itself, the same way for the truth and for predictions.

## Splits (all leakage-safe)

| Dataset | Train | Validation | Test | Rule |
|---|---:|---:|---:|---|
| Classification (lab + field) | 42,029 | 4,473 lab + 351 field | 10,937 lab + 225 field | leaf-grouped PlantVillage; PlantDoc official test; validation carved from train only |
| PlantDoc-corrected detection | 2,096 | 233 | 236 | upstream split (official test kept) |
| Tea detection | 2,040 | 269 | 406 | dataset's own split column, after de-duplication |
| LeafNet75 lesions | 412 | 88 | 88 | random 70/15/15 with seed 0, after de-duplication |
| Simulated epidemics | 800 fields | 100 fields | 200 fields | split by field, never by window |
| Fundamentals subset (8 classes, 24 px) | 2,000 | 480 | 800 | drawn from the leaf-grouped split above |

Validation sets choose epochs and models; **test sets are used once**, by the evaluation scripts.

## Known biases and limitations

1. **PlantVillage backdrop shortcut.** Each PlantVillage class was photographed against similar backdrops.
   Grad-CAM (`surveillance/results/conv_gradcam.png`) shows the classifier partly using the background for
   some classes. This is one reason field accuracy (~71%) is far below lab accuracy (~99.6%).
2. **Pests are demonstrated on tea**, not on the 14 LeafScan crops: no licensed, box-annotated pest dataset
   for those crops was found. The pest detector is evidence that the approach works on real field photos;
   it must be retrained on local pests before use on other crops.
3. **No field video.** No openly licensed, annotated field video of these diseases was found. The video
   mode is tested on a clip assembled from field photos and labelled as such.
4. **Progression data are simulated** from a logistic disease-progress model with weather dependence.
   They show the method works; real season-long photo series are needed before trusting the forecasts.
5. **Small test sets** for field photos (225), detection (236 / 406) and lesions (88): 95% intervals are
   several points wide; numbers should be read as approximate.
6. **Web-scraped images** (PlantDoc) can carry label noise; the corrected release fixes names and
   duplicates but not every box.
7. **Geography and cultivars:** most images come from North America (PlantVillage), India and China (tea);
   other regions, cultivars and growth stages are under-represented.

## How to re-create everything

```powershell
.\.venv-train\Scripts\python training\prepare_data.py        # classification splits -> data/splits.json
.\.venv-train\Scripts\python -m surveillance.prepare_data    # tea + lesion data, de-duplication report
```

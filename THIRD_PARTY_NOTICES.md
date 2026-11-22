# Third-party notices

LeafScan is by Esh Dobhal and is released under the MIT License (see `LICENSE`). That licence covers
Esh Dobhal's own work in this repository. It includes or was built from the third-party material below,
which keeps its own terms and is **not** relicensed under MIT. These attributions are kept because the
licences require them.

## Code and data inherited from an earlier project

- `Flask Deployed App/CNN.py` (the original CNN architecture and the 39-class list),
  `disease_info.csv`, `supplement_info.csv`, the `Model/` notebook and the `test_images/` samples (four of which are copied into
  `Flask Deployed App/static/img/samples/` for the Diagnose page) come from
  [Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection) by Manthan Bhikadiya
  and Krishna Baldaniya. That repository publishes no licence file; check with its authors before
  redistributing those files or the original CNN weights.

## Datasets used to train the hybrid model

- **PlantVillage**: Hughes, D. P. & Salathé, M. (2015), "An open access repository of images on plant
  health to enable the development of mobile disease diagnostics", arXiv:1511.08060.
  Images and leaf-grouped splits from [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage).
- **PlantDoc**: Singh, D. et al. (2020), "PlantDoc: A Dataset for Visual Plant Disease Detection",
  CoDS-COMAD 2020. [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset),
  licensed CC BY 4.0. Images were resized and relabelled to this project's classes.
- **Background class**: the `Background_without_leaves` folder of the
  [PlantVillage release on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1).

## Datasets used by the surveillance system (`surveillance/`)

- **PlantDoc-corrected** (disease detection): KeenForgeAI, cleaned re-release of the PlantDoc detection set,
  [KeenForgeAI/PlantDoc-corrected](https://huggingface.co/datasets/KeenForgeAI/PlantDoc-corrected),
  DOI 10.57967/hf/10678, CC BY 4.0. Original: Singh, D. et al. (2020), PlantDoc.
- **Tea leaf disease and pest detection** (pest detection): Lyu, Yu & Song, PLOS One (2026),
  [Project-AgML/tea_leaf_disease_pest_detection](https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_pest_detection),
  CC BY 4.0. De-duplicated and relabelled to 6 classes by `surveillance/prepare_data.py`.
- **LeafNet75 lesion masks** (segmentation): [LeafNet75/Leaf-Disease-Segmentation-masks](https://huggingface.co/datasets/LeafNet75/Leaf-Disease-Segmentation-masks),
  Apache License 2.0. Six of its test photos are copied into `demo/photos/` and appear in the demo video.
- The demo also uses `test_images/potato_late_blight.JPG` (PlantVillage, see above) and one tea test photo
  (`demo/photos_tea/`, CC BY 4.0, Lyu, Yu & Song).
- Progression data are simulated by `surveillance/progression.py` (no third-party data).

## Model architecture and pre-trained weights

- **CAFormer-S18** and its ImageNet pre-trained weights (`caformer_s18.sail_in22k_ft_in1k`) from
  [timm / pytorch-image-models](https://github.com/huggingface/pytorch-image-models), Apache License 2.0.
  Yu, W. et al. (2022), "MetaFormer Baselines for Vision".
- **ResNet-50, GoogLeNet, RetinaNet-ResNet50-FPN (v2), Faster R-CNN-ResNet50-FPN (v2), Mask R-CNN-ResNet50-FPN
  (v2), DeepLabV3-ResNet50 and FCN-ResNet50** architectures and their ImageNet / COCO pre-trained weights from
  [torchvision](https://github.com/pytorch/vision), BSD-3-Clause. Papers: He et al. 2016 (ResNet), Szegedy et
  al. 2015 (GoogLeNet), Lin et al. 2017 (RetinaNet, focal loss), Ren et al. 2015 (Faster R-CNN), He et al. 2017
  (Mask R-CNN), Chen et al. 2017 (DeepLabV3), Long et al. 2015 (FCN).
- **YOLOv8s** and its COCO weights from [Ultralytics](https://github.com/ultralytics/ultralytics),
  **AGPL-3.0**. Used only to benchmark detection (`surveillance/detection.py yolo`); YOLO weights and code
  are not part of the shipped application, and the trained YOLO weights are not distributed.

## Assets

- **Fonts**: Inter and Fraunces, SIL Open Font License 1.1 (full text in
  `Flask Deployed App/static/fonts/LICENSE.txt`).
- **Icons**: paths in `Flask Deployed App/static/icons.svg` are adapted from [Lucide](https://lucide.dev),
  ISC License: Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather
  (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022. Permission to use, copy,
  modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided
  that the above copyright notice and this permission notice appear in all copies.
- Crop and product illustrations, favicon and social image were drawn for LeafScan
  (`scripts/make_assets.py`).

## Libraries (installed from PyPI, not bundled)

Flask (BSD-3-Clause), PyTorch and torchvision (BSD-3-Clause), Pillow (MIT-CMU), timm (Apache-2.0),
huggingface_hub (Apache-2.0), gunicorn (MIT), waitress (ZPL 2.1), pytest (MIT), Playwright (Apache-2.0).
Research environment (`requirements-research.txt`): NumPy (BSD-3-Clause), SciPy (BSD-3-Clause),
scikit-learn (BSD-3-Clause), matplotlib (PSF-based), torchmetrics (Apache-2.0), pycocotools (BSD-2-Clause),
pyarrow (Apache-2.0), OpenCV (Apache-2.0), PyYAML (MIT), psutil (BSD-3-Clause), ONNX and ONNX Runtime (MIT),
imageio-ffmpeg (BSD-2-Clause; its bundled FFmpeg build is GPL, used only to encode the demo video),
Ultralytics (AGPL-3.0, benchmark only).

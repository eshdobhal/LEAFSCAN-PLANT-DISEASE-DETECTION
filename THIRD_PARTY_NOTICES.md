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

## Model architecture and pre-trained weights

- **CAFormer-S18** and its ImageNet pre-trained weights (`caformer_s18.sail_in22k_ft_in1k`) from
  [timm / pytorch-image-models](https://github.com/huggingface/pytorch-image-models), Apache License 2.0.
  Yu, W. et al. (2022), "MetaFormer Baselines for Vision".

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

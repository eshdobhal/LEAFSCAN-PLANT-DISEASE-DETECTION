"""Loads the chosen model once and turns a PIL image into a prediction dictionary.

Two models share the same 39 classes (same order as CNN.idx_to_classes):
- "hybrid": CAFormer-S18, convolution stages + self-attention stages, fine-tuned by
  training/train_hybrid.py on PlantVillage lab photos + PlantDoc field photos.
- "cnn": the original project's CNN trained on PlantVillage only.
"""
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import CNN

log = logging.getLogger(__name__)

NUM_CLASSES = len(CNN.idx_to_classes)  # 39
HYBRID_ARCH = "caformer_s18"
NAMES = {"hybrid": "CAFormer-S18 hybrid CNN-Transformer", "cnn": "Original CNN"}

# Each model must see images prepared exactly as during its training.
PREPROCESS = {
    # Original CNN (Model/Plant Disease Detection Code.md): short side -> 255 px, centre 224x224,
    # float tensor in [0, 1]. No normalisation was used in that training.
    "cnn": transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    # Hybrid (training/common.py EVAL_TF): short side -> 256 px, centre 224x224, then the
    # ImageNet mean/std normalisation its pre-trained weights expect.
    "hybrid": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
}


def preprocess(image: Image.Image, kind: str = "cnn") -> torch.Tensor:
    """PIL image (any mode) -> tensor of shape [1, 3, 224, 224]."""
    image = image.convert("RGB")  # PNG with alpha or greyscale photos would otherwise break the 3-channel model
    return PREPROCESS[kind](image).unsqueeze(0)  # add the batch dimension


def build_network(kind: str) -> torch.nn.Module:
    if kind == "hybrid":
        import timm  # only needed for the hybrid
        return timm.create_model(HYBRID_ARCH, pretrained=False, num_classes=NUM_CLASSES)
    return CNN.CNN(NUM_CLASSES)


class Predictor:
    def __init__(self, model_path: str, kind: str = "cnn", require_weights: bool = False, tta: bool = False):
        # A few threads are faster on a small server; more just fight each other.
        torch.set_num_threads(min(4, os.cpu_count() or 1))

        self.kind = kind
        self.name = NAMES[kind]
        self.tta = tta and kind == "hybrid"
        self.model_path = Path(model_path)
        self.model_loaded = self.model_path.is_file()
        if not self.model_loaded and require_weights:
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        torch.manual_seed(0)  # only matters in demo mode: random weights stay the same between runs
        self.model = build_network(kind)
        if self.model_loaded:
            # map_location: weights may have been saved on a GPU machine.
            # weights_only: refuse to unpickle arbitrary Python objects (safer).
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            # The hybrid is stored in half precision to halve the file; compute in full precision.
            state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            self.model.load_state_dict(state)
            log.info("Loaded %s weights from %s", self.name, self.model_path)
        else:
            log.error(
                "MODEL WEIGHTS NOT FOUND at %s. Running in DEMO MODE: predictions are random. "
                "See README.md for where to get the weights (or set MODEL_PATH / HYBRID_MODEL_PATH).",
                self.model_path,
            )
        self.model.eval()  # turns off dropout and uses normalisation layers' stored statistics
        self._warm_up()

    def _warm_up(self):
        """The first forward pass is slow (memory allocation); do it at startup, not for a user."""
        start = time.perf_counter()
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, 224, 224))
        log.info("Model warm-up took %.0f ms", (time.perf_counter() - start) * 1000)

    def predict(self, image: Image.Image) -> dict:
        """Returns {"index", "confidence", "top3": [(index, probability), ...], "latency_ms"}."""
        start = time.perf_counter()
        batch = preprocess(image, self.kind)
        if self.tta:  # the photo and its mirror image in one batch of 2
            batch = torch.cat([batch, torch.flip(batch, dims=[3])])
        with torch.inference_mode():  # no gradient bookkeeping -> faster, less memory
            logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).mean(dim=0)  # scores -> probabilities (averaged over TTA views)
        top_probs, top_idx = torch.topk(probs, k=3)
        top3 = [(int(i), float(p)) for i, p in zip(top_idx, top_probs)]
        return {
            "index": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

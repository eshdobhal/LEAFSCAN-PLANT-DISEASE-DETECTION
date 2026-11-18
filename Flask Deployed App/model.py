"""Loads the CNN once and turns a PIL image into a prediction dictionary."""
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

# Exactly the transform used during training (Model/Plant Disease Detection Code.md):
# shorter side -> 255 px (keeps aspect ratio), cut the central 224x224 square,
# then convert to a float tensor in [0, 1]. No normalisation was used in training.
PREPROCESS = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL image (any mode) -> tensor of shape [1, 3, 224, 224]."""
    image = image.convert("RGB")  # PNG with alpha or greyscale photos would otherwise break the 3-channel model
    return PREPROCESS(image).unsqueeze(0)  # add the batch dimension


class Predictor:
    def __init__(self, model_path: str, require_weights: bool = False):
        # A few threads are faster on a small server; more just fight each other.
        torch.set_num_threads(min(4, os.cpu_count() or 1))

        self.model_path = Path(model_path)
        self.model_loaded = self.model_path.is_file()
        if not self.model_loaded and require_weights:
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        torch.manual_seed(0)  # only matters in demo mode: random weights stay the same between runs
        self.model = CNN.CNN(NUM_CLASSES)
        if self.model_loaded:
            # map_location: weights may have been saved on a GPU machine.
            # weights_only: refuse to unpickle arbitrary Python objects (safer).
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            log.info("Loaded model weights from %s", self.model_path)
        else:
            log.error(
                "MODEL WEIGHTS NOT FOUND at %s. Running in DEMO MODE: predictions are random. "
                "Download plant_disease_model_1_latest.pt from the link in README.md and put it in "
                "'Flask Deployed App/' (or set the MODEL_PATH environment variable).",
                self.model_path,
            )
        self.model.eval()  # turns off dropout and uses BatchNorm's stored statistics
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
        batch = preprocess(image)
        with torch.inference_mode():  # no gradient bookkeeping -> faster, less memory
            logits = self.model(batch)[0]
        probs = torch.softmax(logits, dim=0)  # raw scores -> probabilities that sum to 1
        top_probs, top_idx = torch.topk(probs, k=3)
        top3 = [(int(i), float(p)) for i, p in zip(top_idx, top_probs)]
        return {
            "index": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

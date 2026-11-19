"""Loads the chosen model once and turns a PIL image into a prediction dictionary.

Two models share the same 39 classes (same order as CNN.idx_to_classes):
- "hybrid": CAFormer-S18, convolution stages + self-attention stages, fine-tuned by
  training/train_hybrid.py on PlantVillage lab photos + PlantDoc field photos. Two copies were
  trained, one at 224 px and one at 288 px; when both weight files exist their probabilities are
  averaged (an ensemble), which was best on the validation photos.
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
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_transform(kind: str, size: int = 224):
    """Each model must see images prepared exactly as during its training."""
    if kind == "cnn":
        # Original CNN (Model/Plant Disease Detection Code.md): short side -> 255 px, centre 224x224,
        # float tensor in [0, 1]. No normalisation was used in that training.
        return transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    # Hybrid (training/common.py eval_transform): short side -> size * 256/224 (256 for 224 px,
    # 329 for 288 px), centre crop size x size, then the ImageNet mean/std its pre-trained weights expect.
    return transforms.Compose([
        transforms.Resize(round(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess(image: Image.Image, kind: str = "cnn", size: int = 224) -> torch.Tensor:
    """PIL image (any mode) -> tensor of shape [1, 3, size, size]."""
    image = image.convert("RGB")  # PNG with alpha or greyscale photos would otherwise break the 3-channel model
    return make_transform(kind, size)(image).unsqueeze(0)  # add the batch dimension


def build_network(kind: str) -> torch.nn.Module:
    if kind == "hybrid":
        import timm  # only needed for the hybrid
        return timm.create_model(HYBRID_ARCH, pretrained=False, num_classes=NUM_CLASSES)
    return CNN.CNN(NUM_CLASSES)


class Predictor:
    """members: list of (weights path, input size). One member for the CNN; one or two for the hybrid."""

    def __init__(self, members, kind: str = "cnn", require_weights: bool = False, tta: bool = False):
        # A few threads are faster on a small server; more just fight each other.
        torch.set_num_threads(min(4, os.cpu_count() or 1))

        self.kind = kind
        self.sizes = [size for _, size in members]
        self.name = NAMES[kind] + (" ensemble (" + " + ".join(f"{s} px" for s in self.sizes) + ")"
                                   if len(members) > 1 else "")
        self.tta = tta and kind == "hybrid"
        missing = [str(path) for path, _ in members if not Path(path).is_file()]
        self.model_loaded = not missing
        if missing and require_weights:
            raise FileNotFoundError(f"Model weights not found at {', '.join(missing)}")
        if missing:
            log.error(
                "MODEL WEIGHTS NOT FOUND at %s. Running in DEMO MODE: predictions are random. "
                "See README.md for where to get the weights (or set MODEL_PATH / HYBRID_MODEL_PATH).",
                ", ".join(missing),
            )

        torch.manual_seed(0)  # only matters in demo mode: random weights stay the same between runs
        self.members = []  # list of (network, transform)
        for path, size in members:
            self.members.append((self._load(Path(path), kind), make_transform(kind, size)))
        self._warm_up()

    def _load(self, path: Path, kind: str) -> torch.nn.Module:
        network = build_network(kind)
        if path.is_file():
            # map_location: weights may have been saved on a GPU machine.
            # weights_only: refuse to unpickle arbitrary Python objects (safer).
            state = torch.load(path, map_location="cpu", weights_only=True)
            # The hybrid is stored in half precision to halve the file; compute in full precision.
            state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            network.load_state_dict(state)
            log.info("Loaded %s weights from %s", NAMES[kind], path)
        network.eval()  # turns off dropout and uses normalisation layers' stored statistics
        return network

    def _warm_up(self):
        """The first forward pass is slow (memory allocation); do it at startup, not for a user."""
        start = time.perf_counter()
        with torch.inference_mode():
            for network, _ in self.members:
                network(torch.zeros(1, 3, 224, 224))
        log.info("Model warm-up took %.0f ms", (time.perf_counter() - start) * 1000)

    def predict(self, image: Image.Image) -> dict:
        """Returns {"index", "confidence", "top3": [(index, probability), ...], "latency_ms"}."""
        start = time.perf_counter()
        image = image.convert("RGB")  # PNG with alpha or greyscale photos would otherwise break the model
        all_probs = []
        for network, transform in self.members:
            batch = transform(image).unsqueeze(0)  # add the batch dimension
            if self.tta:  # the photo and its mirror image in one batch of 2
                batch = torch.cat([batch, torch.flip(batch, dims=[3])])
            with torch.inference_mode():  # no gradient bookkeeping -> faster, less memory
                logits = network(batch)
            all_probs.append(torch.softmax(logits, dim=1))  # scores -> probabilities, one row per view
        probs = torch.cat(all_probs).mean(dim=0)  # average over views and ensemble members
        top_probs, top_idx = torch.topk(probs, k=3)
        top3 = [(int(i), float(p)) for i, p in zip(top_idx, top_probs)]
        return {
            "index": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

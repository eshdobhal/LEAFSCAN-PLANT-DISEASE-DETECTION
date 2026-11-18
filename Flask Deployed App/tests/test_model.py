"""Tests for model selection and the hybrid predictor (works with or without trained weights)."""
from PIL import Image

from app import load_predictor
from config import Config
from model import Predictor, preprocess


def test_auto_falls_back_to_cnn_without_hybrid_weights(tmp_path):
    config = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    config.update(MODEL_KIND="auto", HYBRID_MODEL_PATH=str(tmp_path / "missing.pt"))
    assert load_predictor(config).kind == "cnn"


def test_hybrid_predictor_returns_top3_probabilities(tmp_path):
    predictor = Predictor(str(tmp_path / "missing.pt"), kind="hybrid", tta=True)  # demo weights
    result = predictor.predict(Image.new("RGB", (640, 480), "green"))
    probs = [p for _, p in result["top3"]]
    assert len(probs) == 3 and probs == sorted(probs, reverse=True)
    assert 0 < sum(probs) <= 1.0001


def test_hybrid_preprocessing_is_normalised():
    x = preprocess(Image.new("RGB", (300, 200), "white"), "hybrid")
    assert tuple(x.shape) == (1, 3, 224, 224)
    assert x.max() > 1.5  # white after ImageNet normalisation is ~2.2-2.6, not 1.0

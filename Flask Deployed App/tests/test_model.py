"""Tests for model selection and the hybrid predictor (works with or without trained weights)."""
from pathlib import Path

import pytest
from PIL import Image

from app import load_predictor
from config import Config
from model import Predictor, preprocess


def test_auto_falls_back_to_cnn_without_hybrid_weights(tmp_path):
    config = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    config.update(MODEL_KIND="auto", HYBRID_MODEL_PATH=str(tmp_path / "missing.pt"))
    assert load_predictor(config).kind == "cnn"


def test_auto_builds_ensemble_only_when_288_weights_exist(tmp_path):
    config = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    if not Path(config["HYBRID_MODEL_PATH"]).is_file():
        pytest.skip("hybrid weights not present")
    config.update(MODEL_KIND="auto", HYBRID_288_MODEL_PATH=str(tmp_path / "missing.pt"))
    assert load_predictor(config).sizes == [224]
    if Path(Config.HYBRID_288_MODEL_PATH).is_file():
        config.update(HYBRID_288_MODEL_PATH=Config.HYBRID_288_MODEL_PATH)
        assert load_predictor(config).sizes == [224, 288]
        config.update(ENSEMBLE=False)
        assert load_predictor(config).sizes == [224]


def test_hybrid_ensemble_returns_top3_probabilities(tmp_path):
    members = [(tmp_path / "missing.pt", 224), (tmp_path / "missing_288.pt", 288)]
    predictor = Predictor(members, kind="hybrid", tta=True)  # demo weights
    assert not predictor.model_loaded and "ensemble" in predictor.name
    result = predictor.predict(Image.new("RGBA", (640, 480), "green"))
    probs = [p for _, p in result["top3"]]
    assert len(probs) == 3 and probs == sorted(probs, reverse=True)
    assert 0 < sum(probs) <= 1.0001


def test_hybrid_preprocessing_is_normalised():
    x = preprocess(Image.new("RGB", (300, 200), "white"), "hybrid")
    assert tuple(x.shape) == (1, 3, 224, 224)
    assert x.max() > 1.5  # white after ImageNet normalisation is ~2.2-2.6, not 1.0
    assert tuple(preprocess(Image.new("RGB", (400, 300)), "hybrid", 288).shape) == (1, 3, 288, 288)

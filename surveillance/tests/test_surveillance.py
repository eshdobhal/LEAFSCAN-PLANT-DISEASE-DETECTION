"""Unit tests for the surveillance package (no trained weights or GPU needed, except where skipped)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


# ------------------------------------------------------------------ detection data
def test_yolo_labels_become_pixel_boxes(tmp_path, monkeypatch):
    from surveillance import detection
    (tmp_path / "images" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(tmp_path / "images" / "test" / "a.jpg")
    (tmp_path / "labels" / "test" / "a.txt").write_text("2 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.0001 0.1\n")
    monkeypatch.setitem(detection.DATASETS, "fake", tmp_path)
    img, target = detection.YoloDetection("fake", "test")[0]
    assert img.shape == (3, 100, 200)
    assert target["boxes"].tolist() == [[80.0, 30.0, 120.0, 70.0]]  # degenerate 0.02-px box dropped
    assert target["labels"].tolist() == [3]  # +1: torchvision reserves 0 for background


def test_flip_augmentation_mirrors_boxes(tmp_path, monkeypatch):
    from surveillance import detection
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.2 0.5 0.2 0.2\n")
    monkeypatch.setitem(detection.DATASETS, "fake", tmp_path)
    ds = detection.YoloDetection("fake", "train", train=True)
    seen = {tuple(round(v) for v in ds[0][1]["boxes"][0].tolist()) for _ in range(30)}
    assert seen == {(10, 40, 30, 60), (70, 40, 90, 60)}


# ------------------------------------------------------------------ segmentation helpers
def test_lesion_instances_and_f1():
    from surveillance.segmentation import lesion_f1, lesion_instances
    true = np.zeros((50, 50), np.uint8)
    true[5:15, 5:15] = 1
    true[30:40, 30:40] = 1
    true[45, 45] = 1  # 1-pixel speck: below MIN_LESION_PX, not a lesion
    pred = np.zeros_like(true)
    pred[6:15, 5:15] = 1  # matches the first lesion (IoU 0.9)
    pred[20:25, 0:10] = 1  # a false positive
    assert len(lesion_instances(true)) == 2
    hits, n_pred, n_true = lesion_f1(pred, true)
    assert (hits, n_pred, n_true) == (1, 2, 2)


@pytest.mark.parametrize("fraction,grade", [(0.0, 0), (0.005, 0), (0.03, 1), (0.07, 2), (0.2, 3), (0.4, 4), (0.8, 5)])
def test_severity_grades(fraction, grade):
    from surveillance.segmentation import severity
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[..., 1] = 200  # all green: every pixel counts as leaf
    lesion = np.zeros((100, 100), np.uint8)
    lesion.flat[: int(fraction * 10000)] = 1
    frac, g = severity(rgb, lesion)
    assert abs(frac - fraction) < 1e-3 and g == grade


# ------------------------------------------------------------------ calibration
def test_temperature_scaling_recovers_known_overconfidence():
    from surveillance.evaluate_all import apply_t, ece, fit_temperature
    torch.manual_seed(0)
    z = torch.randn(4000, 10) * 1.5
    y = torch.multinomial(torch.softmax(z, 1), 1)[:, 0]
    p = torch.softmax(z * 3, 1)  # three times too confident
    t = fit_temperature(p, y)
    assert abs(t - 3) < 0.2
    assert ece(apply_t(p, t), y)[0] < ece(p, y)[0] / 5


def test_ece_is_zero_when_confidence_equals_accuracy():
    from surveillance.evaluate_all import ece
    p = torch.tensor([[0.75, 0.25]] * 4)
    y = torch.tensor([0, 0, 0, 1])  # 75% confident, 75% right
    assert ece(p, y)[0] < 1e-6


# ------------------------------------------------------------------ convolutions by hand
def test_conv2d_and_conv3d_by_hand_match_libraries():
    from scipy import ndimage

    from surveillance.convviz import conv2d_by_hand, conv3d_by_hand
    rng = np.random.default_rng(0)
    img, k = rng.random((12, 10)), rng.random((3, 3))
    assert np.allclose(conv2d_by_hand(img, k), ndimage.correlate(img, k, mode="constant")[1:-1, 1:-1])
    vol, k3 = rng.random((5, 8, 9)), rng.random((2, 3, 3))
    ref = torch.nn.functional.conv3d(torch.from_numpy(vol)[None, None], torch.from_numpy(k3)[None, None])[0, 0].numpy()
    assert np.allclose(conv3d_by_hand(vol, k3), ref)


# ------------------------------------------------------------------ progression
def test_simulated_epidemics_are_valid_and_sprays_slow_disease():
    from surveillance.progression import DAYS, FEATURES, simulate
    seq, truth = simulate(40, seed=0, noise=0.0)
    assert seq.shape == (40, DAYS, len(FEATURES)) and truth.shape == (40, DAYS)
    assert (np.diff(truth, axis=1) >= -1e-9).all()  # disease never heals in the model
    assert truth.min() >= 0 and truth.max() <= 1


def test_forecaster_shapes():
    from surveillance.progression import FEATURES, HORIZON, LOOKBACK, Forecaster
    for cell in ("rnn", "lstm"):
        out = Forecaster(cell)(torch.rand(3, LOOKBACK, len(FEATURES)))
        assert out.shape == (3, HORIZON) and out.min() >= 0 and out.max() <= 1


# ------------------------------------------------------------------ application
@pytest.mark.skipif(not (ROOT / "Flask Deployed App" / "leafscan_hybrid.pt").exists(), reason="needs the classifier weights")
def test_cli_writes_report_and_alert_exit_code(tmp_path):
    from surveillance import cli
    code = cli.main(["analyze", str(ROOT / "test_images" / "tomato-mold.jpg"), "--only", "classify", "--out", str(tmp_path),
                     "--device", "cpu"])
    report = json.loads((tmp_path / "tomato-mold.json").read_text())
    assert report["classification"]["low_confidence"] is True and code == 2  # uncertain -> alert -> exit 2
    assert (tmp_path / "tomato-mold_annotated.png").exists()

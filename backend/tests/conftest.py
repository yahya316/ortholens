"""Shared fixtures: synthetic images and mocked models (no weights needed)."""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.model import ComboConfig, Detection, Models


class FakeDetector:
    """Stands in for the YOLO detector; returns preset detections for every call."""

    loaded = True
    device = "cpu"
    version = "fake@test"

    def __init__(self, detections: list[Detection] | None = None) -> None:
        """Store the detections returned by predict."""
        self.detections = detections if detections is not None else []
        self.calls = 0
        self.last_kwargs: dict = {}
        self.last_shape: tuple | None = None

    def predict(self, image: np.ndarray, **kwargs) -> list[Detection]:
        """Return preset detections and record the call."""
        self.calls += 1
        self.last_kwargs = kwargs
        self.last_shape = image.shape
        return list(self.detections)


class FakeResNet:
    """Stands in for the ResNet50 classifier; returns a preset probability."""

    loaded = True
    device = "cpu"

    def __init__(self, prob: float = 0.05) -> None:
        """Store the probability returned for every image."""
        self.prob = prob
        self.calls = 0

    def predict_proba(self, cleaned_bgr: np.ndarray) -> float:
        """Return the preset probability."""
        self.calls += 1
        return self.prob


def make_config() -> ComboConfig:
    """Small fake combo config: refs 0.1..1.0, thresholds 0.4 (screening) < 0.6 (balanced)."""
    ref = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return ComboConfig.from_dict({
        "w_yolo": 0.6, "w_resnet": 0.4, "ref_yolo": ref, "ref_resnet": ref,
        "thr_balanced": 0.6, "thr_screening": 0.4, "yolo_score_conf": 0.01, "yolo_box_conf": 0.33,
        "imgsz": 1024, "resnet_size": 384,
    })


def _encode(arr_bgr: np.ndarray, fmt: str = "JPEG") -> bytes:
    """Encode a BGR array to bytes with Pillow."""
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def sharp_jpeg() -> bytes:
    """A bright, high-detail image that passes the quality check."""
    rng = np.random.default_rng(0)
    img = rng.integers(90, 230, size=(480, 640, 3), dtype=np.uint8)
    return _encode(img)


@pytest.fixture
def blurry_jpeg() -> bytes:
    """A bright but featureless (blurred) image that fails the sharpness check."""
    rng = np.random.default_rng(1)
    img = rng.integers(90, 230, size=(480, 640, 3), dtype=np.uint8)
    return _encode(cv2.GaussianBlur(img, (0, 0), 40))


@pytest.fixture
def lowcontrast_jpeg() -> bytes:
    """A sharp-edged but dark, low-contrast image (grey values 20-40)."""
    rng = np.random.default_rng(2)
    img = rng.integers(20, 40, size=(480, 640, 3), dtype=np.uint8)
    return _encode(img)


@pytest.fixture
def xray_like_jpeg() -> bytes:
    """Mostly dark background with a bright textured 'bone' region, like an X-ray photo."""
    rng = np.random.default_rng(3)
    img = rng.integers(5, 35, size=(480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    cv2.ellipse(mask, (320, 240), (200, 80), 20, 0, 360, 255, -1)
    bone = rng.integers(150, 230, size=(480, 640, 3), dtype=np.uint8)
    img[mask > 0] = bone[mask > 0]
    return _encode(img)


@pytest.fixture
def fake_detector() -> FakeDetector:
    """A FakeDetector with no detections by default."""
    return FakeDetector()


@pytest.fixture
def fake_resnet() -> FakeResNet:
    """A FakeResNet with a low probability by default (so decisions follow YOLO)."""
    return FakeResNet()


@pytest.fixture
def client(fake_detector: FakeDetector, fake_resnet: FakeResNet):
    """TestClient with fake models installed (lifespan not run)."""
    app.state.models = Models(fake_detector, fake_resnet, make_config())
    yield TestClient(app)
    app.state.models = None


@pytest.fixture(autouse=True)
def _no_disk_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SAVE_UPLOADS is off and the default mode is screening during tests."""
    monkeypatch.setattr(settings, "SAVE_UPLOADS", False)
    monkeypatch.setattr(settings, "DECISION_MODE", "screening")
    monkeypatch.setattr(settings, "SPLIT_VIEWS", False)

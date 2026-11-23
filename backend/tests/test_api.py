"""API and pipeline tests."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import NOTICE, settings
from app.main import app
from app.model import Detection
from app.pipeline import build_findings
from app import restore

from .conftest import FakeDetector


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"models_loaded": {"yolo": True, "resnet": True, "config": True}, "device": "cpu"}


def test_health_without_weights(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "missing.pt")
    with TestClient(app) as c:  # runs lifespan: must start despite missing weights
        r = c.get("/api/health")
        assert r.status_code == 200 and r.json()["models_loaded"]["yolo"] is False
        r2 = c.post("/api/analyses", files={"file": ("a.png", _png(), "image/png")})
        assert r2.status_code == 503
        assert "missing.pt" in r2.json()["message"]
        assert r2.json()["notice"] == NOTICE


def _png() -> bytes:
    from .conftest import _encode
    return _encode(np.full((64, 64, 3), 128, np.uint8), "PNG")


def test_bad_file_type(client: TestClient) -> None:
    r = client.post("/api/analyses", files={"file": ("x.txt", b"hello world", "text/plain")})
    assert r.status_code == 415
    body = r.json()
    assert body["status"] == "error" and body["notice"] == NOTICE


def test_non_jpeg_png_image_rejected(client: TestClient) -> None:
    from .conftest import _encode
    gif = _encode(np.full((64, 64, 3), 128, np.uint8), "GIF")
    r = client.post("/api/analyses", files={"file": ("x.jpg", gif, "image/jpeg")})
    assert r.status_code == 415


def test_corrupt_image(client: TestClient, sharp_jpeg: bytes) -> None:
    r = client.post("/api/analyses", files={"file": ("x.jpg", sharp_jpeg[:300], "image/jpeg")})
    assert r.status_code in (400, 415)
    assert r.json()["status"] == "error"


def test_missing_field(client: TestClient) -> None:
    r = client.post("/api/analyses")
    assert r.status_code == 400 and r.json()["notice"] == NOTICE


def test_oversized_file(client: TestClient, monkeypatch: pytest.MonkeyPatch, sharp_jpeg: bytes) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/analyses", files={"file": ("big.jpg", sharp_jpeg, "image/jpeg")})
    assert r.status_code == 413
    assert "too large" in r.json()["message"].lower()


def test_blurry_image_retake(client: TestClient, blurry_jpeg: bytes, fake_detector: FakeDetector, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "QUALITY_MODE", "block")
    r = client.post("/api/analyses", files={"file": ("b.jpg", blurry_jpeg, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "retake" and "blurry" in body["message"]
    assert body["notice"] == NOTICE and isinstance(body["warnings"], list)
    assert fake_detector.calls == 0


def test_happy_path_with_mock_model(client: TestClient, fake_detector: FakeDetector, sharp_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((100, 120, 200, 220), 0.91)]
    r = client.post("/api/analyses", files={"file": ("secret_name.jpg", sharp_jpeg, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["findings"]) >= 1
    f = body["findings"][0]
    assert f["low_confidence"] is False and len(f["box"]) == 4
    assert body["original_image"].startswith("data:image/jpeg;base64,")
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert body["model_version"] == "fake@test"
    assert body["thresholds"] == {"conf": 0.33, "low_conf": settings.LOW_CONF, "imgsz": 1024,
                                  "quality_mode": settings.QUALITY_MODE}
    assert body["decision"] == "fracture" and body["scores"]["yolo_top"] == 0.91
    assert "secret_name" not in r.text
    assert body["notice"] == NOTICE


def test_no_fracture_detected(client: TestClient, sharp_jpeg: bytes) -> None:
    r = client.post("/api/analyses", files={"file": ("a.jpg", sharp_jpeg, "image/jpeg")})
    assert r.json()["status"] == "no_fracture_detected"


def test_low_confidence_flag() -> None:
    dets = [Detection((0, 0, 10, 10), 0.90), Detection((0, 0, 10, 10), 0.40), Detection((0, 0, 10, 10), 0.10)]
    out = build_findings([(0, 0, dets)], (100, 100, 3), settings)
    assert [f["low_confidence"] for f in out] == [False, True]  # 0.10 < CONF is dropped
    assert all(f["confidence"] >= settings.CONF for f in out)


def test_boxes_mapped_to_full_image_for_second_view() -> None:
    out = build_findings([(1, 300, [Detection((10, 20, 50, 60), 0.8)])], (200, 600, 3), settings)
    assert out[0]["box"] == [310.0, 20.0, 350.0, 60.0] and out[0]["view"] == 1


def test_out_of_scope(client: TestClient, sharp_jpeg: bytes, fake_detector: FakeDetector) -> None:
    from app import scope
    scope.set_classifier(lambda img: (False, "nope"))
    try:
        r = client.post("/api/analyses", files={"file": ("a.jpg", sharp_jpeg, "image/jpeg")})
    finally:
        scope.set_classifier(None)
    assert r.json()["status"] == "out_of_scope"
    assert r.json()["message"].startswith("This looks like a body region OrthoLens does not support")
    assert fake_detector.calls == 0


def test_split_views_detects_divider(monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.full((200, 400, 3), 200, np.uint8)
    img[:, 195:205] = 10
    monkeypatch.setattr(settings, "SPLIT_VIEWS", True)
    views = restore.split_views(img, settings)
    assert len(views) == 2 and 150 < views[1][1] < 250
    assert len(restore.split_views(np.full((200, 400, 3), 200, np.uint8), settings)) == 1


def test_no_files_saved_by_default(client: TestClient, sharp_jpeg: bytes, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "up")
    client.post("/api/analyses", files={"file": ("a.jpg", sharp_jpeg, "image/jpeg")})
    assert not (tmp_path / "up").exists()


def test_save_uploads_opt_in(client: TestClient, sharp_jpeg: bytes, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(settings, "SAVE_UPLOADS", True)
    client.post("/api/analyses", files={"file": ("patient.jpg", sharp_jpeg, "image/jpeg")})
    files = list((tmp_path / "up").iterdir())
    assert len(files) == 1 and files[0].suffix == ".jpg" and "patient" not in files[0].name

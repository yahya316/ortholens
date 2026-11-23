"""Tests for the QUALITY_MODE logic and response contract."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import NOTICE, settings
from app.model import Detection

from .conftest import FakeDetector


def _post(client: TestClient, data: bytes):
    return client.post("/api/analyses", files={"file": ("p.jpg", data, "image/jpeg")})


def test_warn_mode_low_contrast_still_runs_model(client: TestClient, fake_detector: FakeDetector, lowcontrast_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((10, 10, 60, 60), 0.8)]
    body = _post(client, lowcontrast_jpeg).json()
    assert settings.QUALITY_MODE == "warn"
    assert body["status"] == "ok" and fake_detector.calls >= 1
    assert body["findings"][0]["confidence"] == 0.8
    assert any("contrast" in w.lower() for w in body["warnings"])


def test_warn_mode_blurry_still_runs_model(client: TestClient, fake_detector: FakeDetector, blurry_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((10, 10, 60, 60), 0.8)]
    body = _post(client, blurry_jpeg).json()
    assert body["status"] == "ok" and fake_detector.calls >= 1
    assert any("blurry" in w.lower() for w in body["warnings"])


def test_block_mode_rejects_only_unusable(client: TestClient, monkeypatch: pytest.MonkeyPatch, blurry_jpeg: bytes,
                                          xray_like_jpeg: bytes, fake_detector: FakeDetector) -> None:
    monkeypatch.setattr(settings, "QUALITY_MODE", "block")
    assert _post(client, blurry_jpeg).json()["status"] == "retake"
    assert fake_detector.calls == 0
    body = _post(client, xray_like_jpeg).json()
    assert body["status"] != "retake" and fake_detector.calls >= 1


def test_block_mode_does_not_reject_merely_poor(client: TestClient, monkeypatch: pytest.MonkeyPatch, lowcontrast_jpeg: bytes) -> None:
    """Contrast between MIN/2 and MIN is a warning, not a retake."""
    monkeypatch.setattr(settings, "QUALITY_MODE", "block")
    monkeypatch.setattr(settings, "MIN_CONTRAST", 12.0)  # image contrast is ~7: below 12 but above 6
    body = _post(client, lowcontrast_jpeg).json()
    assert body["status"] != "retake"
    assert any("contrast" in w.lower() for w in body["warnings"])


def test_off_mode_skips_check(client: TestClient, monkeypatch: pytest.MonkeyPatch, blurry_jpeg: bytes) -> None:
    monkeypatch.setattr(settings, "QUALITY_MODE", "off")
    body = _post(client, blurry_jpeg).json()
    assert body["status"] != "retake"
    assert not any(w for w in body["warnings"] if "blurry" in w.lower() or "contrast" in w.lower())
    assert body["quality"]["sharpness"] is None


def test_every_response_has_notice_and_warnings(client: TestClient, sharp_jpeg: bytes, blurry_jpeg: bytes,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        client.post("/api/analyses", files={"file": ("x.txt", b"nope", "text/plain")}),
        _post(client, sharp_jpeg),
        _post(client, blurry_jpeg),
    ]
    monkeypatch.setattr(settings, "QUALITY_MODE", "block")
    responses.append(_post(client, blurry_jpeg))
    for r in responses:
        body = r.json()
        assert body["notice"] == NOTICE
        assert isinstance(body["warnings"], list)
        assert set(body["quality"]) == {"sharpness", "contrast", "film_found", "views_found"}
        assert "quality_mode" in body["thresholds"]


def test_low_confidence_finding_in_response(client: TestClient, fake_detector: FakeDetector, sharp_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((10, 10, 60, 60), 0.40), Detection((100, 100, 160, 160), 0.9)]
    findings = _post(client, sharp_jpeg).json()["findings"]
    by_conf = {f["confidence"]: f["low_confidence"] for f in findings}
    assert by_conf == {0.4: True, 0.9: False}


def test_mean_brightness_not_used() -> None:
    app_dir = Path(__file__).resolve().parent.parent / "app"
    for py in app_dir.glob("*.py"):
        assert "brightness" not in py.read_text(encoding="utf-8").lower(), py.name

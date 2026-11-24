"""Tests for the combined score, decision modes, boxes, 503 and the response contract."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import BODY_PART_WARNING, NOTICE, settings
from app.main import app
from app.model import Detection, cdf, combined_score, is_fracture

from .conftest import FakeDetector, FakeResNet, make_config


def _post(client: TestClient, data: bytes, **form):
    return client.post("/api/analyses", files={"file": ("p.jpg", data, "image/jpeg")}, data=form)


def test_cdf_matches_reference_formula() -> None:
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    assert cdf(ref, 0.0) == 0.0
    assert cdf(ref, 0.2) == 0.5  # side="right": ties count
    assert cdf(ref, 5.0) == 1.0


def test_combined_score_monotonic_and_bounded() -> None:
    cfg = make_config()
    xs = np.linspace(0, 1, 41)
    by_yolo = [combined_score(cfg, x, 0.5) for x in xs]
    by_resnet = [combined_score(cfg, 0.5, x) for x in xs]
    assert by_yolo == sorted(by_yolo) and by_resnet == sorted(by_resnet)
    for a in xs:
        for b in xs:
            assert 0.0 <= combined_score(cfg, a, b) <= 1.0
    assert combined_score(cfg, 0.0, 0.0) == 0.0
    assert combined_score(cfg, 1.0, 1.0) == pytest.approx(1.0)


def test_screening_threshold_lower_than_balanced_in_real_config() -> None:
    from app.model import load_config
    real = load_config(settings)
    if real is None:
        pytest.skip("combo_config.json not present")
    assert real.thr_screening < real.thr_balanced


def test_between_thresholds_screening_fracture_balanced_not() -> None:
    cfg = make_config()
    assert cfg.thr_screening < cfg.thr_balanced
    score = combined_score(cfg, 0.55, 0.35)  # 0.6*0.5 + 0.4*0.3 = 0.42
    assert cfg.thr_screening <= score < cfg.thr_balanced
    assert is_fracture(cfg, score, "screening") and not is_fracture(cfg, score, "balanced")


def test_api_modes_between_thresholds(client: TestClient, fake_detector: FakeDetector, fake_resnet: FakeResNet,
                                      sharp_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((100, 100, 200, 200), 0.55)]
    fake_resnet.prob = 0.35
    scr = _post(client, sharp_jpeg, mode="screening").json()
    bal = _post(client, sharp_jpeg, mode="balanced").json()
    default = _post(client, sharp_jpeg).json()  # DECISION_MODE=screening
    assert scr["decision"] == "fracture" and scr["decision_mode"] == "screening"
    assert bal["decision"] == "no_fracture" and bal["decision_mode"] == "balanced"
    assert default["decision"] == "fracture"
    assert scr["scores"]["threshold"] < bal["scores"]["threshold"]
    assert scr["scores"]["combined"] == pytest.approx(0.42, abs=1e-3)


def test_invalid_mode_rejected(client: TestClient, sharp_jpeg: bytes) -> None:
    r = _post(client, sharp_jpeg, mode="aggressive")
    assert r.status_code == 400 and r.json()["notice"] == NOTICE


def test_no_fracture_has_no_boxes(client: TestClient, fake_detector: FakeDetector, fake_resnet: FakeResNet,
                                  sharp_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((10, 10, 60, 60), 0.5)]  # box exists, score still below balanced
    fake_resnet.prob = 0.05
    body = _post(client, sharp_jpeg, mode="balanced").json()
    assert body["decision"] == "no_fracture" and body["status"] == "no_fracture_detected"
    assert body["findings"] == []


def test_fracture_without_confident_box_is_location_uncertain(client: TestClient, fake_detector: FakeDetector,
                                                             fake_resnet: FakeResNet, sharp_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((10, 10, 60, 60), 0.25)]  # < 0.33
    fake_resnet.prob = 0.95
    body = _post(client, sharp_jpeg).json()
    assert body["decision"] == "fracture" and body["status"] == "ok"
    assert body["message"] == "Fracture suspected, location uncertain"
    assert body["findings"] == []


def test_recipe_inputs_and_outputs(client: TestClient, fake_detector: FakeDetector, fake_resnet: FakeResNet,
                                   sharp_jpeg: bytes) -> None:
    fake_detector.detections = [Detection((10, 10, 60, 60), 0.9), Detection((5, 5, 20, 20), 0.2)]
    fake_resnet.prob = 0.7
    body = _post(client, sharp_jpeg).json()
    assert fake_detector.calls == 1  # no split by default
    assert fake_detector.last_shape == (480, 640, 3)  # YOLO saw the RAW photo, not the cleaned one
    assert fake_detector.last_kwargs == {"imgsz": 1024, "conf": 0.01}
    assert body["scores"]["yolo_top"] == 0.9 and body["scores"]["resnet_prob"] == 0.7
    assert [f["confidence"] for f in body["findings"]] == [0.9]  # 0.2 < 0.33 not displayed
    assert body["cleaned_image"].startswith("data:image/jpeg;base64,")
    t = body["timings_ms"]
    assert set(t) == {"clean", "yolo", "resnet", "total"} and t["total"] >= 0


def test_show_cleaned_off(client: TestClient, monkeypatch: pytest.MonkeyPatch, sharp_jpeg: bytes) -> None:
    monkeypatch.setattr(settings, "SHOW_CLEANED", False)
    assert not _post(client, sharp_jpeg).json().get("cleaned_image")


def test_split_views_never_run_by_default(client: TestClient, fake_detector: FakeDetector) -> None:
    img = np.full((200, 400, 3), 200, np.uint8)
    img[:, 195:205] = 10
    from app import restore
    assert len(restore.split_views(img, settings)) == 1


def test_missing_weights_give_503_naming_files(monkeypatch: pytest.MonkeyPatch, tmp_path, sharp_jpeg: bytes) -> None:
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "no_yolo.pt")
    monkeypatch.setattr(settings, "RESNET_PATH", tmp_path / "no_resnet.pt")
    monkeypatch.setattr(settings, "COMBO_CONFIG_PATH", tmp_path / "no_config.json")
    with TestClient(app) as c:  # runs lifespan: must start anyway
        h = c.get("/api/health").json()
        assert h["models_loaded"] == {"yolo": False, "resnet": False, "config": False}
        assert isinstance(h["device"], str)
        r = _post(c, sharp_jpeg)
        assert r.status_code == 503
        msg = r.json()["message"]
        assert "no_yolo.pt" in msg and "no_resnet.pt" in msg and "no_config.json" in msg
        body = r.json()
        assert body["notice"] == NOTICE and BODY_PART_WARNING in body["warnings"]


def test_one_missing_file_named(monkeypatch: pytest.MonkeyPatch, tmp_path, sharp_jpeg: bytes) -> None:
    import shutil
    from app.config import BASE_DIR
    cfg_src = BASE_DIR / "models" / "combo_config.json"
    if not cfg_src.exists():
        pytest.skip("combo_config.json not present")
    shutil.copy(cfg_src, tmp_path / "combo_config.json")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "best.pt")
    monkeypatch.setattr(settings, "RESNET_PATH", tmp_path / "resnet50.pt")
    monkeypatch.setattr(settings, "COMBO_CONFIG_PATH", tmp_path / "combo_config.json")
    with TestClient(app) as c:
        assert c.get("/api/health").json()["models_loaded"] == {"yolo": False, "resnet": False, "config": True}


def test_health_reports_loaded(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"models_loaded": {"yolo": True, "resnet": True, "config": True}, "device": "cpu"}


def test_every_response_has_notice_warnings_and_body_part_warning(client: TestClient, sharp_jpeg: bytes,
                                                                  blurry_jpeg: bytes,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        client.post("/api/analyses", files={"file": ("x.txt", b"nope", "text/plain")}),
        client.post("/api/analyses"),
        _post(client, sharp_jpeg),
        _post(client, sharp_jpeg, mode="balanced"),
        _post(client, sharp_jpeg, mode="bogus"),
    ]
    monkeypatch.setattr(settings, "QUALITY_MODE", "block")
    responses.append(_post(client, blurry_jpeg))
    for r in responses:
        body = r.json()
        assert body["notice"] == NOTICE
        assert BODY_PART_WARNING in body["warnings"]


def test_score_never_called_probability_or_percentage() -> None:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    text = (root / "app" / "pipeline.py").read_text(encoding="utf-8") + (root / "app" / "schemas.py").read_text(encoding="utf-8")
    assert "combined probability" not in text.lower()

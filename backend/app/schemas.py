"""Pydantic response schemas (used for the OpenAPI docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Finding(BaseModel):
    """One suspected fracture region in raw-photo coordinates."""

    box: list[float]
    confidence: float
    view: int
    low_confidence: bool


class Quality(BaseModel):
    """Photo quality (measured on the film region) and preprocessing information.

    sharpness and contrast are null when QUALITY_MODE is "off".
    """

    sharpness: float | None
    contrast: float | None
    film_found: bool
    views_found: int


class Thresholds(BaseModel):
    """Settings in effect for this analysis. ``conf`` is the box display threshold."""

    conf: float
    low_conf: float
    imgsz: int
    quality_mode: str


class Scores(BaseModel):
    """Recipe scores. ``combined`` is a rank-based 0-1 score, NOT a probability."""

    yolo_top: float
    resnet_prob: float
    combined: float
    threshold: float


class Timings(BaseModel):
    """Milliseconds per step."""

    clean: int
    yolo: int
    resnet: int
    total: int


class AnalysisResponse(BaseModel):
    """Result of POST /api/analyses."""

    status: Literal["ok", "no_fracture_detected", "retake", "out_of_scope", "error"]
    message: str
    warnings: list[str]
    findings: list[Finding]
    original_image: str
    annotated_image: str
    quality: Quality
    model_version: str
    thresholds: Thresholds
    elapsed_ms: int
    notice: str
    decision: Literal["fracture", "no_fracture"] | None = None
    decision_mode: str | None = None
    scores: Scores | None = None
    cleaned_image: str | None = None
    timings_ms: Timings | None = None


class ModelsLoaded(BaseModel):
    """Which model files are loaded."""

    yolo: bool
    resnet: bool
    config: bool


class HealthResponse(BaseModel):
    """Result of GET /api/health."""

    models_loaded: ModelsLoaded
    device: str

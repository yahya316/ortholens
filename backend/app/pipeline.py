"""Orchestrates the analysis: validate, quality, CLEAN, YOLO (raw) + ResNet50 (cleaned), combine, decide, annotate."""
from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from . import restore as restore_mod
from . import scope
from .config import BODY_PART_WARNING, MODES, NOTICE, OUT_OF_SCOPE_MESSAGE, Settings, settings as default_settings
from .model import Models, combined_score, is_fracture

logger = logging.getLogger("ortholens.pipeline")

ALLOWED_FORMATS = {"JPEG", "PNG"}
WARN_BLURRY = "Image looks blurry; result may be less reliable"
WARN_LOW_CONTRAST = "Low contrast; result may be less reliable"
WARN_NO_FILM = "Film region not found; full photo used"
WARN_TWO_VIEWS = "Two views detected and analysed separately"
MSG_UNCERTAIN = "Fracture suspected, location uncertain"
BLUE_BGR = (200, 110, 20)
ORANGE_BGR = (0, 140, 255)


class PipelineError(Exception):
    """A user-facing failure with an HTTP status code."""

    def __init__(self, message: str, http_status: int = 400) -> None:
        """Store the message and HTTP status."""
        super().__init__(message)
        self.message = message
        self.http_status = http_status


def resolve_mode(mode: str | None, cfg: Settings) -> str:
    """Pick the decision mode: the request's, else DECISION_MODE (invalid env falls back to screening).

    Raises:
        PipelineError: 400 if an explicit mode is not 'screening' or 'balanced'.
    """
    if mode is not None and mode.strip() != "":
        m = mode.strip().lower()
        if m not in MODES:
            raise PipelineError("Invalid mode. Use 'screening' or 'balanced'.", 400)
        return m
    env = cfg.DECISION_MODE
    return env if env in MODES else "screening"


def validate_and_decode(data: bytes, cfg: Settings) -> np.ndarray:
    """Validate an upload and return the raw photo as a BGR array (in memory only).

    Checks size, real format (JPEG/PNG by content, not file name) and readability, then
    decodes with ``cv2.imdecode`` (EXIF orientation applied). Only pixels are kept, so
    all metadata is dropped. Photos are not downscaled unless MAX_SIDE > 0.

    Raises:
        PipelineError: with 400/413/415 for invalid input.
    """
    if not data:
        raise PipelineError("The uploaded file is empty.", 400)
    if len(data) > cfg.MAX_UPLOAD_BYTES:
        raise PipelineError(f"File is too large. Maximum size is {cfg.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
    Image.MAX_IMAGE_PIXELS = cfg.MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise PipelineError("Unsupported file type. Please upload a JPEG or PNG image.", 415)
            probe.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()  # full decode: catches truncated files
    except PipelineError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise PipelineError("The file could not be read as a valid JPEG or PNG image.", 415)
    except Exception:
        raise PipelineError("The image appears to be corrupt and could not be read.", 400)
    bgr = decode_bgr(data)
    if bgr is None:
        raise PipelineError("The image appears to be corrupt and could not be read.", 400)
    h, w = bgr.shape[:2]
    if min(h, w) < 32:
        raise PipelineError("The image is too small to analyse.", 400)
    longest = max(h, w)
    if cfg.MAX_SIDE > 0 and longest > cfg.MAX_SIDE:
        scale = cfg.MAX_SIDE / longest
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return bgr


def decode_bgr(data: bytes) -> np.ndarray | None:
    """Decode image bytes to BGR with cv2.imdecode (None if undecodable)."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _display(image: np.ndarray, cfg: Settings) -> np.ndarray:
    """Downscale for transport only (DISPLAY_MAX_SIDE); analysis always uses the full photo."""
    longest = max(image.shape[:2])
    if cfg.DISPLAY_MAX_SIDE > 0 and longest > cfg.DISPLAY_MAX_SIDE:
        s = cfg.DISPLAY_MAX_SIDE / longest
        return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return image


def encode_data_uri(image: np.ndarray, cfg: Settings) -> str:
    """Encode a BGR image as a metadata-free JPEG data URI."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
    if not ok:
        raise PipelineError("Could not encode the result image.", 500)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], t: int, dash: int = 14) -> None:
    """Draw a dashed rectangle on ``img`` in place."""
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, t)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, t)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, t)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, t)


def annotate(image: np.ndarray, findings: list[dict[str, Any]]) -> np.ndarray:
    """Draw findings: solid blue boxes, dashed orange boxes for low confidence.

    Every box carries a text label with the confidence; low-confidence ones read
    "low confidence".
    """
    out = image.copy()
    scale = max(out.shape[:2]) / 1000.0
    thick = max(2, int(round(3 * scale)))
    font_scale = max(0.5, 0.6 * scale)
    for f in findings:
        x1, y1, x2, y2 = (int(round(v)) for v in f["box"])
        low = f["low_confidence"]
        color = ORANGE_BGR if low else BLUE_BGR
        if low:
            _dashed_rect(out, (x1, y1), (x2, y2), color, thick)
            label = f"low confidence {f['confidence']:.2f}"
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            label = f"fracture {f['confidence']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thick - 1))
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y1 + th + base + 4
        tx = min(max(0, x1), max(0, out.shape[1] - tw - 6))
        cv2.rectangle(out, (tx, ty - th - base), (tx + tw + 6, ty + base // 2), color, -1)
        cv2.putText(out, label, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return out


def _save_deidentified(image: np.ndarray, cfg: Settings) -> None:
    """Optionally persist the de-identified image (SAVE_UPLOADS). Atomic; no partial files."""
    if not cfg.SAVE_UPLOADS:
        return
    cfg.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    final = Path(cfg.UPLOAD_DIR) / f"{uuid.uuid4().hex}.jpg"
    tmp = final.with_suffix(".tmp")
    try:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
        if not ok:
            return
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _response(
    cfg: Settings,
    models: Models | None,
    t0: float,
    *,
    status: str,
    message: str,
    original: str,
    annotated: str,
    findings: list[dict[str, Any]] | None = None,
    sharpness: float | None = None,
    contrast: float | None = None,
    film_found: bool = False,
    views_found: int = 0,
    warnings: list[str] | None = None,
    decision: str | None = None,
    decision_mode: str | None = None,
    scores: dict[str, float] | None = None,
    cleaned_image: str | None = None,
    timings: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compose the standard response envelope (always includes the notice and body-part warning)."""
    combo = models.config if models is not None else None
    warn = list(warnings or [])
    if BODY_PART_WARNING not in warn:
        warn.append(BODY_PART_WARNING)
    elapsed = int((time.perf_counter() - t0) * 1000)
    out: dict[str, Any] = {
        "status": status,
        "message": message,
        "findings": findings or [],
        "original_image": original,
        "annotated_image": annotated,
        "quality": {
            "sharpness": None if sharpness is None else round(sharpness, 2),
            "contrast": None if contrast is None else round(contrast, 2),
            "film_found": film_found,
            "views_found": views_found,
        },
        "model_version": models.version if models is not None else "none",
        "thresholds": {
            "conf": combo.yolo_box_conf if combo else cfg.CONF,
            "low_conf": cfg.LOW_CONF,
            "imgsz": combo.imgsz if combo else cfg.IMGSZ,
            "quality_mode": cfg.QUALITY_MODE,
        },
        "warnings": warn,
        "elapsed_ms": elapsed,
        "notice": NOTICE,
        "decision": decision,
        "decision_mode": decision_mode,
        "scores": scores,
        "timings_ms": timings,
    }
    if cleaned_image is not None:
        out["cleaned_image"] = cleaned_image
    return out


def error_envelope(message: str, cfg: Settings | None = None, models: Models | None = None) -> dict[str, Any]:
    """Build an ``error`` response with empty images (used for HTTP error replies)."""
    return _response(cfg or default_settings, models, time.perf_counter(), status="error", message=message, original="", annotated="")


def build_findings(
    per_view: list[tuple[int, int, list[Any]]], image_shape: tuple[int, ...], cfg: Settings, min_conf: float | None = None
) -> list[dict[str, Any]]:
    """Map per-view detections to full-image findings.

    Args:
        per_view: ``(view_index, x_offset, detections)`` triples.
        image_shape: shape of the image the boxes refer to (the raw photo), used to clamp boxes.
        cfg: settings supplying LOW_CONF.
        min_conf: detections below this are dropped (the combo config's ``yolo_box_conf``).

    Returns:
        Findings, best first, with ``low_confidence`` set when confidence < LOW_CONF.
    """
    min_conf = cfg.CONF if min_conf is None else min_conf
    h, w = image_shape[:2]
    findings: list[dict[str, Any]] = []
    for view, x_off, dets in per_view:
        for d in dets:
            if d.confidence < min_conf:
                continue
            x1, y1, x2, y2 = d.box
            box = [
                round(min(max(x1 + x_off, 0), w), 1),
                round(min(max(y1, 0), h), 1),
                round(min(max(x2 + x_off, 0), w), 1),
                round(min(max(y2, 0), h), 1),
            ]
            findings.append({"box": box, "confidence": round(float(d.confidence), 4), "view": view,
                             "low_confidence": bool(d.confidence < cfg.LOW_CONF)})
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings


@dataclass
class RecipeResult:
    """Raw outputs of the detection recipe for one photo."""

    cleaned: np.ndarray
    film_found: bool
    per_view: list[tuple[int, int, list[Any]]]
    views_found: int
    yolo_top: float
    resnet_prob: float
    combined: float
    ms: dict[str, int] = field(default_factory=dict)


def score_photo(raw: np.ndarray, cleaned: np.ndarray, film_found: bool, models: Models, cfg: Settings,
                clean_ms: int = 0) -> RecipeResult:
    """Steps 2-4 of the recipe: YOLO on the RAW photo, ResNet50 on the CLEANED one, combined score."""
    combo = models.config
    assert combo is not None
    t = time.perf_counter()
    views = restore_mod.split_views(raw, cfg)  # one view unless SPLIT_VIEWS is true
    per_view = [(i, off, models.detector.predict(crop, imgsz=combo.imgsz, conf=combo.yolo_score_conf))
                for i, (crop, off) in enumerate(views)]
    confs = [d.confidence for _, _, dets in per_view for d in dets]
    yolo_top = float(max(confs)) if confs else 0.0
    yolo_ms = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    resnet_prob = float(models.resnet.predict_proba(cleaned))
    resnet_ms = int((time.perf_counter() - t) * 1000)

    combined = float(combined_score(combo, yolo_top, resnet_prob))
    return RecipeResult(cleaned, film_found, per_view, len(views), yolo_top, resnet_prob, combined,
                        {"clean": clean_ms, "yolo": yolo_ms, "resnet": resnet_ms})


def run_recipe(raw: np.ndarray, models: Models, cfg: Settings | None = None) -> RecipeResult:
    """Full recipe on a decoded BGR photo: clean, YOLO, ResNet50, combine (no quality check, no drawing)."""
    cfg = cfg or default_settings
    t = time.perf_counter()
    cleaned, film_found = restore_mod.restore(raw)
    return score_photo(raw, cleaned, film_found, models, cfg, int((time.perf_counter() - t) * 1000))


def analyze(data: bytes, models: Models, cfg: Settings | None = None, mode: str | None = None) -> dict[str, Any]:
    """Run the full pipeline on raw upload bytes. Nothing is written to disk unless SAVE_UPLOADS.

    In QUALITY_MODE "warn" the models always run and poor quality only adds warnings;
    in "block" only truly unusable images return ``retake``; "off" skips the check.

    Raises:
        PipelineError: for invalid input (400/413/415).
    """
    cfg = cfg or default_settings
    t0 = time.perf_counter()
    mode = resolve_mode(mode, cfg)
    combo = models.config
    assert combo is not None
    raw = validate_and_decode(data, cfg)
    original_uri = encode_data_uri(_display(raw, cfg), cfg)
    warnings: list[str] = []

    t = time.perf_counter()
    cleaned, film_found, film_crop = restore_mod.restore_with_film(raw)  # step 1
    clean_ms = int((time.perf_counter() - t) * 1000)
    if not film_found:
        warnings.append(WARN_NO_FILM)

    sharpness: float | None = None
    contrast: float | None = None
    if cfg.QUALITY_MODE != "off":
        sharpness, contrast = restore_mod.measure_quality(film_crop, cfg)
        if cfg.DEBUG_QUALITY:
            logger.info("quality sharpness=%.2f contrast=%.2f film_found=%s", sharpness, contrast, film_found)
        poor_s, poor_c, bad_s, bad_c = restore_mod.assess_quality(sharpness, contrast, cfg)
        if poor_s:
            warnings.append(WARN_BLURRY)
        if poor_c:
            warnings.append(WARN_LOW_CONTRAST)
        if cfg.QUALITY_MODE == "block" and (bad_s or bad_c):
            reasons = " and ".join(r for r, bad in (("blurry", bad_s), ("low contrast", bad_c)) if bad)
            return _response(
                cfg, models, t0, status="retake",
                message=f"The photo is too {reasons} to analyse. Please retake it with the film in focus and well lit.",
                original=original_uri, annotated=original_uri, sharpness=sharpness, contrast=contrast,
                film_found=film_found, warnings=warnings, decision_mode=mode,
            )

    in_scope, _reason = scope.is_in_scope(cleaned)  # body-part check (stub)
    if not in_scope:
        return _response(cfg, models, t0, status="out_of_scope", message=OUT_OF_SCOPE_MESSAGE,
                         original=original_uri, annotated=original_uri, sharpness=sharpness, contrast=contrast,
                         film_found=film_found, warnings=warnings, decision_mode=mode)

    res = score_photo(raw, cleaned, film_found, models, cfg, clean_ms)  # steps 2-4
    if res.views_found == 2:
        warnings.append(WARN_TWO_VIEWS)
    threshold = combo.threshold(mode)
    fracture = is_fracture(combo, res.combined, mode)  # step 5

    findings: list[dict[str, Any]] = []
    if fracture:  # step 6
        findings = build_findings(res.per_view, raw.shape, cfg, combo.yolo_box_conf)
    if fracture and findings:
        n = len(findings)
        low = sum(f["low_confidence"] for f in findings)
        message = f"{n} suspected fracture region{'s' if n != 1 else ''} found" + (f" ({low} low confidence)." if low else ".")
    elif fracture:
        message = MSG_UNCERTAIN
    else:
        message = "No fracture detected. This does not rule out a fracture."

    annotated_uri = encode_data_uri(_display(annotate(raw, findings), cfg), cfg)
    cleaned_uri = encode_data_uri(cleaned, cfg) if cfg.SHOW_CLEANED else None
    _save_deidentified(raw, cfg)
    timings = {**res.ms, "total": int((time.perf_counter() - t0) * 1000)}
    return _response(
        cfg, models, t0, status="ok" if fracture else "no_fracture_detected", message=message,
        original=original_uri, annotated=annotated_uri, findings=findings,
        sharpness=sharpness, contrast=contrast, film_found=film_found, views_found=res.views_found,
        warnings=warnings, decision="fracture" if fracture else "no_fracture", decision_mode=mode,
        scores={"yolo_top": round(res.yolo_top, 4), "resnet_prob": round(res.resnet_prob, 4),
                "combined": round(res.combined, 4), "threshold": round(threshold, 4)},
        cleaned_image=cleaned_uri, timings=timings,
    )

"""Central settings for OrthoLens.

Every value can be overridden by an environment variable of the same name.
Other modules must read values from the shared ``settings`` object and never
hardcode them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable."""
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable (1/true/yes/on are truthy)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    """Read a comma-separated list environment variable."""
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]


def _models_dir() -> Path:
    """Resolve MODELS_DIR from the environment."""
    return Path(_env_str("MODELS_DIR", str(BASE_DIR / "models")))


@dataclass
class Settings:
    """All tunable settings, read from the environment at construction."""

    # --- Model / inference ---
    MODELS_DIR: Path = field(default_factory=lambda: Path(_env_str("MODELS_DIR", str(BASE_DIR / "models"))))
    MODEL_PATH: Path = field(default_factory=lambda: Path(_env_str("MODEL_PATH", str(_models_dir() / "best.pt"))))  # YOLO weights
    RESNET_PATH: Path = field(default_factory=lambda: Path(_env_str("RESNET_PATH", str(_models_dir() / "resnet50.pt"))))
    COMBO_CONFIG_PATH: Path = field(default_factory=lambda: Path(_env_str("COMBO_CONFIG_PATH", str(_models_dir() / "combo_config.json"))))
    DECISION_MODE: str = field(default_factory=lambda: _env_str("DECISION_MODE", "screening").strip().lower())  # screening | balanced
    SHOW_CLEANED: bool = field(default_factory=lambda: _env_bool("SHOW_CLEANED", True))
    DISPLAY_MAX_SIDE: int = field(default_factory=lambda: _env_int("DISPLAY_MAX_SIDE", 1600))  # size of returned display images only
    IMGSZ: int = field(default_factory=lambda: _env_int("IMGSZ", 1024))
    CONF: float = field(default_factory=lambda: _env_float("CONF", 0.33))  # DEPRECATED: not used for the decision (see combo_config.json)
    LOW_CONF: float = field(default_factory=lambda: _env_float("LOW_CONF", 0.45))
    IOU: float = field(default_factory=lambda: _env_float("IOU", 0.7))
    DEVICE: str = field(default_factory=lambda: _env_str("DEVICE", "auto"))  # auto | cpu | 0 | cuda:0

    # --- Upload validation / privacy ---
    MAX_UPLOAD_BYTES: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    MAX_PIXELS: int = field(default_factory=lambda: _env_int("MAX_PIXELS", 60_000_000))
    MAX_SIDE: int = field(default_factory=lambda: _env_int("MAX_SIDE", 0))  # 0 = never downscale (needed for parity with the offline recipe)
    JPEG_QUALITY: int = field(default_factory=lambda: _env_int("JPEG_QUALITY", 90))
    SAVE_UPLOADS: bool = field(default_factory=lambda: _env_bool("SAVE_UPLOADS", False))
    UPLOAD_DIR: Path = field(default_factory=lambda: Path(_env_str("UPLOAD_DIR", str(BASE_DIR / "uploads"))))

    # --- Quality check ---
    QUALITY_MODE: str = field(default_factory=lambda: _env_str("QUALITY_MODE", "warn").strip().lower())  # warn | block | off
    MIN_SHARPNESS: float = field(default_factory=lambda: _env_float("MIN_SHARPNESS", 15.0))  # variance of Laplacian (placeholder)
    MIN_CONTRAST: float = field(default_factory=lambda: _env_float("MIN_CONTRAST", 40.0))  # p95 - p5 of grey (placeholder)
    QUALITY_WIDTH: int = field(default_factory=lambda: _env_int("QUALITY_WIDTH", 1024))  # resize width for measuring
    DEBUG_QUALITY: bool = field(default_factory=lambda: _env_bool("DEBUG_QUALITY", False))

    # --- Restore steps ---
    ENABLE_FILM_CROP: bool = field(default_factory=lambda: _env_bool("ENABLE_FILM_CROP", True))
    ENABLE_GLARE_REDUCTION: bool = field(default_factory=lambda: _env_bool("ENABLE_GLARE_REDUCTION", True))
    ENABLE_CLAHE: bool = field(default_factory=lambda: _env_bool("ENABLE_CLAHE", True))
    FILM_MIN_AREA_FRAC: float = field(default_factory=lambda: _env_float("FILM_MIN_AREA_FRAC", 0.20))
    FILM_MAX_AREA_FRAC: float = field(default_factory=lambda: _env_float("FILM_MAX_AREA_FRAC", 0.98))
    GLARE_THRESHOLD: int = field(default_factory=lambda: _env_int("GLARE_THRESHOLD", 250))
    GLARE_MAX_AREA_FRAC: float = field(default_factory=lambda: _env_float("GLARE_MAX_AREA_FRAC", 0.05))
    CLAHE_CLIP: float = field(default_factory=lambda: _env_float("CLAHE_CLIP", 2.0))
    CLAHE_TILE: int = field(default_factory=lambda: _env_int("CLAHE_TILE", 8))

    # --- Double-view split ---
    SPLIT_VIEWS: bool = field(default_factory=lambda: _env_bool("SPLIT_VIEWS", False))  # not validated: off by default
    ENABLE_SPLIT: bool = field(default_factory=lambda: _env_bool("ENABLE_SPLIT", True))  # only used when SPLIT_VIEWS is true
    SPLIT_MIN_DEPTH: float = field(default_factory=lambda: _env_float("SPLIT_MIN_DEPTH", 0.25))

    # --- Web ---
    CORS_ORIGINS: list[str] = field(
        default_factory=lambda: _env_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    )


settings = Settings()

NOTICE: str = (
    "OrthoLens is a screening and teaching aid, not a diagnosis. "
    "A qualified doctor must confirm every finding."
)
OUT_OF_SCOPE_MESSAGE: str = (
    "This looks like a body region OrthoLens does not support. Supported: wrist and forearm."
)

BODY_PART_WARNING: str = (
    "Body-part check not installed: OrthoLens supports wrist and forearm photos only. "
    "Results on other body regions are not valid."
)
MODES: tuple[str, ...] = ("screening", "balanced")

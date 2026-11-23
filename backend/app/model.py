"""Models: YOLOv8 detector, ResNet50 classifier and the combo config. Loaded once at startup."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import MODES, Settings, settings as default_settings

logger = logging.getLogger("ortholens.model")

CONFIG_KEYS = ("w_yolo", "w_resnet", "ref_yolo", "ref_resnet", "thr_balanced", "thr_screening",
               "yolo_score_conf", "yolo_box_conf", "imgsz", "resnet_size")


@dataclass
class Detection:
    """One raw detection in the coordinates of the image passed to predict."""

    box: tuple[float, float, float, float]
    confidence: float


# ------------------------------------------------------------------ combo config
@dataclass
class ComboConfig:
    """Contents of combo_config.json (reference distributions are kept as sorted arrays)."""

    w_yolo: float
    w_resnet: float
    ref_yolo: np.ndarray
    ref_resnet: np.ndarray
    thr_balanced: float
    thr_screening: float
    yolo_score_conf: float
    yolo_box_conf: float
    imgsz: int
    resnet_size: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ComboConfig":
        """Build from a parsed JSON dict; raises KeyError/ValueError if invalid."""
        missing = [k for k in CONFIG_KEYS if k not in d]
        if missing:
            raise KeyError(f"combo_config.json is missing keys: {', '.join(missing)}")
        return cls(
            w_yolo=float(d["w_yolo"]), w_resnet=float(d["w_resnet"]),
            ref_yolo=np.asarray(d["ref_yolo"], dtype=np.float64), ref_resnet=np.asarray(d["ref_resnet"], dtype=np.float64),
            thr_balanced=float(d["thr_balanced"]), thr_screening=float(d["thr_screening"]),
            yolo_score_conf=float(d["yolo_score_conf"]), yolo_box_conf=float(d["yolo_box_conf"]),
            imgsz=int(d["imgsz"]), resnet_size=int(d["resnet_size"]),
        )

    def threshold(self, mode: str) -> float:
        """Decision threshold for a mode ('screening' or 'balanced')."""
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        return self.thr_screening if mode == "screening" else self.thr_balanced


def cdf(ref: np.ndarray, x: float) -> float:
    """Empirical CDF of ``x`` against a sorted reference array (rank on a 0-1 scale)."""
    return float(np.searchsorted(ref, x, side="right") / len(ref))


def combined_score(cfg: ComboConfig, yolo_top: float, resnet_prob: float) -> float:
    """Rank-based combined score in [0, 1]. NOT a probability."""
    return cfg.w_yolo * cdf(cfg.ref_yolo, yolo_top) + cfg.w_resnet * cdf(cfg.ref_resnet, resnet_prob)


def is_fracture(cfg: ComboConfig, combined: float, mode: str) -> bool:
    """Decision: combined score >= the mode's threshold."""
    return combined >= cfg.threshold(mode)


def resolve_device(cfg: Settings) -> str:
    """Resolve the DEVICE setting; 'auto' picks the GPU if available, else CPU."""
    wanted = cfg.DEVICE.strip().lower()
    if wanted != "auto":
        return cfg.DEVICE
    try:
        import torch  # imported lazily: heavy

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken
        return "cpu"


# --------------------------------------------------------------------- YOLO
class Detector:
    """Loads YOLO weights once and runs inference. ``loaded`` stays False if the file is missing."""

    def __init__(self, cfg: Settings | None = None) -> None:
        """Create an unloaded detector bound to a settings object."""
        self.cfg = cfg or default_settings
        self.model = None
        self.device: str = "cpu"
        self.version: str = "none"
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """True when weights are loaded and inference is possible."""
        return self.model is not None

    def load(self) -> bool:
        """Load the weights from MODEL_PATH. Returns True on success, never raises."""
        self.device = resolve_device(self.cfg)
        path = self.cfg.MODEL_PATH
        if not path.is_file():
            logger.warning("YOLO weights not found at %s", path)
            return False
        try:
            from ultralytics import YOLO  # imported lazily: heavy

            self.model = YOLO(str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
            self.version = f"{path.name}@{digest}"
            logger.info("Loaded %s on %s", self.version, self.device)
            return True
        except Exception:
            logger.exception("Failed to load YOLO weights from %s", path)
            self.model = None
            return False

    def predict(self, image: np.ndarray, imgsz: int | None = None, conf: float = 0.01) -> list[Detection]:
        """Run inference on a BGR image; returns every detection with confidence >= ``conf``.

        Raises:
            RuntimeError: if the model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded")
        with self._lock:
            results = self.model.predict(image, imgsz=imgsz or self.cfg.IMGSZ, conf=conf,
                                         device=self.device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                detections.append(Detection((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c)))
        return detections


# ------------------------------------------------------------------- ResNet50
class ResNetClassifier:
    """torchvision resnet50 with a 2-class head; class index 1 = fracture."""

    def __init__(self, cfg: Settings | None = None) -> None:
        """Create an unloaded classifier."""
        self.cfg = cfg or default_settings
        self.net = None
        self.device: str = "cpu"
        self.size: int = 384
        self._tf = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """True when weights are loaded."""
        return self.net is not None

    def load(self, size: int = 384) -> bool:
        """Load the state_dict from RESNET_PATH. Returns True on success, never raises."""
        self.device = resolve_device(self.cfg)
        path = self.cfg.RESNET_PATH
        if not path.is_file():
            logger.warning("ResNet50 weights not found at %s", path)
            return False
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms as T

            net = models.resnet50()
            net.fc = nn.Linear(net.fc.in_features, 2)
            net.load_state_dict(torch.load(str(path), map_location=self.device, weights_only=True))
            self.net = net.to(self.device).eval()
            self.size = size
            self._tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            logger.info("Loaded %s on %s", path.name, self.device)
            return True
        except Exception:
            logger.exception("Failed to load ResNet50 weights from %s", path)
            self.net = None
            return False

    def predict_proba(self, cleaned_bgr: np.ndarray) -> float:
        """Probability of class 1 (fracture) for a cleaned BGR image.

        The image is JPEG-encoded at quality 95 and decoded again first, to match
        the offline evaluation (which read cleaned JPEG files).
        """
        if self.net is None or self._tf is None:
            raise RuntimeError("ResNet50 model is not loaded")
        import torch
        from PIL import Image

        ok, buf = cv2.imencode(".jpg", cleaned_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not JPEG-encode the cleaned image")
        rt = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        pil = Image.fromarray(cv2.cvtColor(rt, cv2.COLOR_BGR2RGB))
        with self._lock, torch.inference_mode():
            x = self._tf(pil).unsqueeze(0).to(self.device)
            return float(torch.softmax(self.net(x), 1)[0, 1])


# ------------------------------------------------------------------ container
@dataclass
class Models:
    """Everything needed to run the recipe. ``missing`` lists absent/unloadable files by name."""

    detector: Any
    resnet: Any
    config: ComboConfig | None
    config_path_name: str = "combo_config.json"
    yolo_name: str = "best.pt"
    resnet_name: str = "resnet50.pt"

    @property
    def status(self) -> dict[str, bool]:
        """What is loaded."""
        return {"yolo": bool(self.detector and self.detector.loaded),
                "resnet": bool(self.resnet and self.resnet.loaded),
                "config": self.config is not None}

    @property
    def missing(self) -> list[str]:
        """File names (as configured) that are not loaded."""
        s = self.status
        names = {"yolo": self.yolo_name, "resnet": self.resnet_name, "config": self.config_path_name}
        return [names[k] for k in ("yolo", "resnet", "config") if not s[k]]

    @property
    def device(self) -> str:
        """Device in use (YOLO's, else ResNet's, else cpu)."""
        return str(getattr(self.detector, "device", None) or getattr(self.resnet, "device", None) or "cpu")

    @property
    def version(self) -> str:
        """Model version string for responses."""
        return str(getattr(self.detector, "version", "none"))


def load_config(cfg: Settings) -> ComboConfig | None:
    """Load combo_config.json; None (with a log) if missing or invalid."""
    path = cfg.COMBO_CONFIG_PATH
    if not path.is_file():
        logger.warning("combo config not found at %s", path)
        return None
    try:
        return ComboConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.exception("Invalid combo config %s", path)
        return None


def load_models(cfg: Settings | None = None) -> Models:
    """Load YOLO, ResNet50 and the combo config once. Never raises; missing parts stay unloaded."""
    cfg = cfg or default_settings
    combo = load_config(cfg)
    det = Detector(cfg)
    det.load()
    res = ResNetClassifier(cfg)
    res.load(combo.resnet_size if combo else 384)
    return Models(det, res, combo, cfg.COMBO_CONFIG_PATH.name, cfg.MODEL_PATH.name, cfg.RESNET_PATH.name)

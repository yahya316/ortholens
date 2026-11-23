"""OpenCV photo cleanup (ported from reference/restore.py), quality check and optional view split.

``find_film``, ``straighten``, ``fix_light`` and ``restore`` are the offline cleanup
algorithm, unchanged. The cleaned image feeds ResNet50 only; YOLO sees the raw photo.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Settings


# ---------------------------------------------------------------- quality check
def measure_quality(image: np.ndarray, cfg: Settings) -> tuple[float, float]:
    """Return (sharpness, contrast) for a BGR image (the film region).

    The image is first resized to ``QUALITY_WIDTH`` px wide so values are comparable
    between photos. Sharpness is the variance of the Laplacian of the grey image;
    contrast is the 95th minus the 5th percentile of grey values.
    """
    h, w = image.shape[:2]
    if w != cfg.QUALITY_WIDTH:
        new_h = max(1, int(round(h * cfg.QUALITY_WIDTH / w)))
        image = cv2.resize(image, (cfg.QUALITY_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    p5, p95 = np.percentile(gray, [5, 95])
    return sharpness, float(p95 - p5)


def assess_quality(sharpness: float, contrast: float, cfg: Settings) -> tuple[bool, bool, bool, bool]:
    """Classify quality values against the thresholds.

    Returns:
        ``(poor_sharpness, poor_contrast, unusable_sharpness, unusable_contrast)``;
        "unusable" means below half the threshold (used for hard rejects in block mode).
    """
    return (
        sharpness < cfg.MIN_SHARPNESS,
        contrast < cfg.MIN_CONTRAST,
        sharpness < cfg.MIN_SHARPNESS / 2,
        contrast < cfg.MIN_CONTRAST / 2,
    )


# ------------------------------------------------- cleanup (port of reference/restore.py)
def order_pts(p: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    p = p.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def find_film(img: np.ndarray) -> np.ndarray | None:
    """Return 4 corner points of the film, or None."""
    H, W = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # background level = median of the photo border
    border = np.concatenate([g[:15].ravel(), g[-15:].ravel(), g[:, :15].ravel(), g[:, -15:].ravel()])
    bgv = np.median(border)

    diff = cv2.absdiff(g, np.full_like(g, int(bgv)))
    edges = cv2.Canny(g, 20, 60)
    mask = ((diff > 12).astype(np.uint8) * 255) | edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * H * W:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_pts(ap)
    return order_pts(cv2.boxPoints(cv2.minAreaRect(hull)))


def straighten(img: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """Perspective-correct the film quad; None if it is too small."""
    tl, tr, br, bl = pts
    w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    if w < 100 or h < 100:
        return None
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h))


def fix_light(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven light / soft glare, then boost contrast."""
    g = gray.astype(np.float32)
    sigma = max(gray.shape) / 6
    illum = cv2.GaussianBlur(g, (0, 0), sigma)
    flat = g / (illum + 10) * float(illum.mean())
    flat = 0.7 * flat + 0.3 * g  # keep some original look
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    # inpaint any still-saturated pixels
    sat = (flat > 245).astype(np.uint8) * 255
    sat = cv2.dilate(sat, np.ones((5, 5), np.uint8))
    if sat.any():
        flat = cv2.inpaint(flat, sat, 5, cv2.INPAINT_TELEA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(flat)


def restore_with_film(img: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    """Run the cleanup and also return the film crop (before the margin trim).

    Returns:
        ``(cleaned_bgr, film_found, film_crop_bgr)``.
    """
    pts = find_film(img)
    ok = pts is not None
    film = straighten(img, pts) if ok else None
    if film is None:
        film, ok = img, False
    crop = film
    h, w = film.shape[:2]
    m = int(0.03 * min(h, w))  # cut the film edge / background
    film = film[m:h - m, m:w - m]
    gray = cv2.cvtColor(film, cv2.COLOR_BGR2GRAY)
    out = fix_light(gray)
    s = 1024 / max(out.shape)
    if s < 1:
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), ok, crop


def restore(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clean a raw photo. Returns ``(cleaned_bgr, film_found)``."""
    out, ok, _ = restore_with_film(img)
    return out, ok


def locate_film(image: np.ndarray, cfg: Settings | None = None) -> tuple[np.ndarray, bool]:
    """Return ``(film_crop, film_found)``; the full frame when no film is found (for quality stats)."""
    _, ok, crop = restore_with_film(image)
    return crop, ok


# ------------------------------------------------ double-view split (disabled by default)
def find_divider(image: np.ndarray, cfg: Settings) -> int | None:
    """Locate a vertical divider (dark valley) in the middle third, or None.

    The column-intensity profile is smoothed; the darkest column in the middle
    third is a divider only if it is at least ``SPLIT_MIN_DEPTH`` (relative)
    darker than the neighbouring regions on both sides.
    """
    h, w = image.shape[:2]
    if w < 60:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray[int(h * 0.1): max(int(h * 0.9), int(h * 0.1) + 1)].mean(axis=0)
    k = max(3, w // 100)
    profile = np.convolve(profile, np.ones(k, dtype=np.float32) / k, mode="same")
    lo, hi = w // 3, (2 * w) // 3
    idx = lo + int(np.argmin(profile[lo:hi]))
    near, far = max(2, int(w * 0.03)), max(4, int(w * 0.15))
    left = profile[max(0, idx - far): max(1, idx - near)]
    right = profile[min(w - 1, idx + near): min(w, idx + far)]
    if left.size == 0 or right.size == 0:
        return None
    ref = float(min(left.mean(), right.mean()))
    depth = (ref - float(profile[idx])) / max(ref, 1.0)
    return idx if depth >= cfg.SPLIT_MIN_DEPTH else None


def split_views(image: np.ndarray, cfg: Settings) -> list[tuple[np.ndarray, int]]:
    """Split a side-by-side double view into crops (only when SPLIT_VIEWS and ENABLE_SPLIT are true).

    Returns:
        List of ``(crop, x_offset)``; one entry when disabled or no clear divider exists.
    """
    if not (cfg.SPLIT_VIEWS and cfg.ENABLE_SPLIT):
        return [(image, 0)]
    idx = find_divider(image, cfg)
    if idx is None:
        return [(image, 0)]
    return [(image[:, :idx], 0), (image[:, idx:], idx)]

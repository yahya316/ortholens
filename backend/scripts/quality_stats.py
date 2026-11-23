"""Print quality measures for a folder of photos, to help choose MIN_SHARPNESS / MIN_CONTRAST.

Usage (from the backend folder):
    python scripts/quality_stats.py path/to/photos

Uses the same functions as the pipeline (decode, film crop, measure_quality).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import restore  # noqa: E402
from app.config import settings  # noqa: E402
from app.pipeline import PipelineError, validate_and_decode  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main(argv: list[str]) -> int:
    """Run the report for the folder given in argv[1]. Returns a process exit code."""
    if len(argv) != 2:
        print("Usage: python scripts/quality_stats.py path/to/photos")
        return 2
    folder = Path(argv[1])
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        print("No JPEG/PNG images found.")
        return 1

    print(f"{'filename':40} {'sharpness':>10} {'contrast':>10} {'film_found':>11}")
    sharp_vals: list[float] = []
    contrast_vals: list[float] = []
    for path in files:
        try:
            image = validate_and_decode(path.read_bytes(), settings)
        except PipelineError as exc:
            print(f"{path.name[:40]:40} skipped: {exc.message}")
            continue
        film_img, found = restore.locate_film(image, settings)
        sharpness, contrast = restore.measure_quality(film_img, settings)
        sharp_vals.append(sharpness)
        contrast_vals.append(contrast)
        print(f"{path.name[:40]:40} {sharpness:10.2f} {contrast:10.2f} {str(found):>11}")

    if sharp_vals:
        print()
        print(f"{'measure':12} {'min':>10} {'median':>10} {'max':>10}")
        for name, vals in (("sharpness", sharp_vals), ("contrast", contrast_vals)):
            print(f"{name:12} {min(vals):10.2f} {statistics.median(vals):10.2f} {max(vals):10.2f}")
        print(f"\nCurrent thresholds: MIN_SHARPNESS={settings.MIN_SHARPNESS}, MIN_CONTRAST={settings.MIN_CONTRAST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

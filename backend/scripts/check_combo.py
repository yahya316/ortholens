"""Run the full OrthoLens pipeline on every image in a folder, to compare with reference/predict_combo.py.

Usage (from the backend folder):
    python scripts/check_combo.py path/to/photos

Columns match predict_combo.py: file, yolo_top, resnet_prob, combined, balanced, screening, boxes_shown.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.model import is_fracture, load_models  # noqa: E402
from app.pipeline import build_findings, decode_bgr, run_recipe  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main(argv: list[str]) -> int:
    """Process the folder in argv[1]. Returns a process exit code."""
    if len(argv) != 2:
        print("Usage: python scripts/check_combo.py path/to/photos")
        return 2
    folder = Path(argv[1])
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2
    models = load_models(settings)
    if models.missing:
        print("Missing model file(s): " + ", ".join(models.missing))
        return 1
    combo = models.config
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        print("No JPEG/PNG images found.")
        return 1

    print(f"{'file':52s} yolo  resnet  comb  bal scr boxes")
    totals: dict[str, list[int]] = {"clean": [], "yolo": [], "resnet": []}
    for f in files:
        raw = decode_bgr(np.fromfile(str(f), np.uint8).tobytes())  # safe for odd file names
        if raw is None:
            print(f"{f.name[:50]:52s} unreadable")
            continue
        r = run_recipe(raw, models, settings)
        bal = is_fracture(combo, r.combined, "balanced")
        scr = is_fracture(combo, r.combined, "screening")
        shown = len(build_findings(r.per_view, raw.shape, settings, combo.yolo_box_conf))
        for k in totals:
            totals[k].append(r.ms[k])
        print(f"{f.name[:50]:52s} {r.yolo_top:.2f}  {r.resnet_prob:.2f}    {r.combined:.2f}   {int(bal)}   {int(scr)}   {shown}")
    if totals["clean"]:
        med = {k: int(np.median(v)) for k, v in totals.items()}
        print(f"\nmedian ms per step: clean {med['clean']}, yolo {med['yolo']}, resnet {med['resnet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

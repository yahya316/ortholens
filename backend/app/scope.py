"""Body-part scope check (stub).

Replace the body of :func:`is_in_scope` (or assign a classifier to
``set_classifier``) to plug in a small wrist/forearm-vs-other model later.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# A classifier takes a BGR image and returns (in_scope, message).
ScopeClassifier = Callable[[np.ndarray], tuple[bool, str]]

_classifier: ScopeClassifier | None = None


def set_classifier(classifier: ScopeClassifier | None) -> None:
    """Install (or clear, with None) a body-part classifier used by is_in_scope."""
    global _classifier
    _classifier = classifier


def is_in_scope(image: np.ndarray) -> tuple[bool, str]:
    """Return whether the image shows a supported body region (wrist/forearm).

    Args:
        image: BGR uint8 image.

    Returns:
        ``(in_scope, reason)``. Currently always ``(True, "check not installed")``
        unless a classifier was installed with ``set_classifier``.
    """
    if _classifier is not None:
        return _classifier(image)
    return True, "check not installed"

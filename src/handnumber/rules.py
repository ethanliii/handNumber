"""A hand-written geometric finger counter, kept as a baseline for the learned model.

This is the classic approach you find in most webcam finger-counting tutorials: a
finger is "up" if it's roughly straight and its tip is farther from the wrist than
its middle joint. It's here so train.py can show what the ML model actually buys us.
"""

from __future__ import annotations

import numpy as np

from .features import canonicalize
from .landmarks import FINGERS


def rule_based_fingers(points: np.ndarray) -> np.ndarray:
    """(N, 21, 2) square-pixel landmarks -> (N, 5) 0/1 finger states."""
    p = canonicalize(np.asarray(points, dtype=np.float64))
    out = []
    for name, (base, mid, _, tip) in FINGERS.items():
        if name == "thumb":
            # thumb is out if its tip is well away from the index knuckle
            out.append(np.linalg.norm(p[:, tip] - p[:, 5], axis=-1) > 0.6)
            continue
        v1, v2 = p[:, mid] - p[:, base], p[:, tip] - p[:, mid]
        cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
        tip_far = np.linalg.norm(p[:, tip], axis=-1) > np.linalg.norm(p[:, mid], axis=-1)
        out.append((cos > 0.5) & tip_far)
    return np.stack(out, axis=1).astype(np.int8)

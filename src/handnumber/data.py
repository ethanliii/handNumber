"""Turn HaGRID v2 gesture annotations into a per-finger training set.

HaGRID labels gestures ("peace", "four", "call", ...), not finger counts, but each
gesture fixes which fingers are up. We translate every gesture into a 5-vector
(thumb, index, middle, ring, pinky) with 1 = extended, 0 = folded, and -1 = "this
gesture doesn't pin the finger down" (masked out of the loss).

The patterns below were checked against the data itself: a geometric heuristic run
over the HaGRID val split agrees with them on 88-100% of hands for every
unmasked finger. Gestures whose fingers are ambiguous in a 2D photo (e.g. "point",
where the index aims at the camera) are left out entirely.
"""

from __future__ import annotations

import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .landmarks import ROOT

DATA_DIR = ROOT / "data"
ANNOTATIONS_ZIP = DATA_DIR / "hagrid" / "annotations.zip"
ANNOTATIONS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
    "annotations_with_landmarks/annotations.zip"
)
CACHE = DATA_DIR / "hagrid_fingers.npz"

U = -1  # unknown / masked
#                      thumb index middle ring pinky
GESTURE_FINGERS: dict[str, tuple[int, int, int, int, int]] = {
    "fist":            (0, 0, 0, 0, 0),
    "one":             (0, 1, 0, 0, 0),
    "mute":            (U, 1, 0, 0, 0),
    "middle_finger":   (U, 0, 1, 0, 0),
    "little_finger":   (0, 0, 0, 0, 1),
    "like":            (1, 0, 0, 0, 0),
    "dislike":         (1, 0, 0, 0, 0),
    "peace":           (0, 1, 1, 0, 0),
    "peace_inverted":  (0, 1, 1, 0, 0),
    "two_up":          (0, 1, 1, 0, 0),
    "two_up_inverted": (0, 1, 1, 0, 0),
    "call":            (1, 0, 0, 0, 1),
    "thumb_index":     (1, 1, 0, 0, 0),
    "thumb_index2":    (1, 1, 0, 0, 0),
    "rock":            (0, 1, 0, 0, 1),
    "three":           (0, 1, 1, 1, 0),
    "three2":          (1, 1, 1, 0, 0),
    "three3":          (0, 1, 1, 0, 1),
    "three_gun":       (1, 1, 1, 0, 0),
    "ok":              (0, 0, 1, 1, 1),
    "four":            (0, 1, 1, 1, 1),
    "palm":            (1, 1, 1, 1, 1),
    "stop":            (U, 1, 1, 1, 1),
    "stop_inverted":   (U, 1, 1, 1, 1),
}  # fmt: skip
GESTURES = tuple(GESTURE_FINGERS)
SPLITS = ("train", "val", "test")

# HaGRID stores landmarks normalised separately by image width and height. Most of its
# photos are 1440x1920 portrait, so x is rescaled by 3/4 to get square-pixel
# coordinates; training adds aspect jitter on top to absorb the images that aren't.
HAGRID_ASPECT = 1440 / 1920


def _load_one(args: tuple[str, str]) -> tuple[str, str, np.ndarray]:
    split, gesture = args
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as z:
        records = json.loads(z.read(f"annotations/{split}/{gesture}.json"))
    pts = [
        lm
        for rec in records.values()
        for label, lm in zip(rec["labels"], rec["hand_landmarks"])
        # Images often contain a second, relaxed hand labelled "no_gesture"; skip it.
        if label == gesture and len(lm) == 21
    ]
    return split, gesture, np.asarray(pts, dtype=np.float32).reshape(-1, 21, 2)


def build_dataset(force: bool = False) -> dict[str, np.ndarray]:
    """Return {points, fingers, gesture, split} arrays, building the cache if needed."""
    if CACHE.exists() and not force:
        return dict(np.load(CACHE))
    if not ANNOTATIONS_ZIP.exists():
        raise SystemExit(
            f"Missing {ANNOTATIONS_ZIP}.\nDownload it (719 MB) with:\n"
            f"  mkdir -p {ANNOTATIONS_ZIP.parent} && curl -L -o {ANNOTATIONS_ZIP} {ANNOTATIONS_URL}"
        )

    jobs = [(s, g) for s in SPLITS for g in GESTURES]
    points, fingers, gesture_ids, split_ids = [], [], [], []
    with ProcessPoolExecutor() as pool:
        for split, gesture, pts in pool.map(_load_one, jobs):
            n = len(pts)
            points.append(pts)
            fingers.append(np.tile(np.array(GESTURE_FINGERS[gesture], np.int8), (n, 1)))
            gesture_ids.append(np.full(n, GESTURES.index(gesture), np.int16))
            split_ids.append(np.full(n, SPLITS.index(split), np.int8))
            print(f"  {split:5s} {gesture:16s} {n:6d} hands")

    data = {
        "points": np.concatenate(points),
        "fingers": np.concatenate(fingers),
        "gesture": np.concatenate(gesture_ids),
        "split": np.concatenate(split_ids),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **data)
    return data


def hagrid_to_square(points: np.ndarray, aspect: float = HAGRID_ASPECT) -> np.ndarray:
    """Normalised HaGRID landmarks -> coordinates with equal x and y units."""
    out = points.copy()
    out[..., 0] *= aspect
    return out


if __name__ == "__main__":
    d = build_dataset(force=True)
    for i, s in enumerate(SPLITS):
        print(s, int((d["split"] == i).sum()), "hands")

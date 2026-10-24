"""Landmarks -> feature vector.

The same function runs on HaGRID photos during training and on webcam frames at
runtime, so it only uses what both sources share: 2D keypoints in square-pixel units.

Every hand is first put in a canonical pose, which makes the features invariant
to where the hand is, how big it is, how it is rotated, and whether it's a left or
right hand / palm or back:
  1. wrist moved to the origin
  2. scaled by palm size
  3. rotated so wrist -> middle knuckle points straight up
  4. mirrored so the index knuckle is always on the right of the pinky knuckle

The vector is three blocks, which the model reads separately (see model.py):
  fingers  FINGER_DIM numbers for each of index, middle, ring, pinky, describing
           only that finger's own joints
  thumb    THUMB_DIM numbers about the thumb, relative to the palm. It deliberately
           can't see the index finger's joints: with them, the model learned to
           guess the thumb from the index finger's state (a HaGRID shortcut), which
           failed on photos from other sources.
  palm     PALM_DIM numbers on the palm's shape, which reveals the viewing angle
"""

from __future__ import annotations

import numpy as np

from .landmarks import FINGERS

PALM_POINTS = (5, 9, 13, 17)  # knuckles (MCP joints)
LONG_FINGERS = [FINGERS[name] for name in ("index", "middle", "ring", "pinky")]
PALM_PAIRS = ((5, 17), (0, 5), (0, 17), (0, 9), (5, 9), (13, 17))


def _upright(points: np.ndarray) -> np.ndarray:
    """Steps 1-3: centred, scaled, rotated, but not yet mirrored."""
    p = points - points[:, :1]
    span = np.linalg.norm(p[:, 5] - p[:, 17], axis=-1)
    reach = np.linalg.norm(p[:, PALM_POINTS], axis=-1).mean(-1)
    scale = np.maximum((reach * 4 + span) / 5, 1e-6)
    p = p / scale[:, None, None]
    return np.einsum("nij,nkj->nki", _rotation_to_up(p[:, 9]), p)


def _rotation_to_up(v: np.ndarray) -> np.ndarray:
    """(N, 2) directions -> (N, 2, 2) rotations taking each one to straight up (0, -1)."""
    theta = np.arctan2(v[:, 0], -v[:, 1])
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def index_left_of_pinky(points: np.ndarray) -> np.ndarray:
    """(N,) bool: with the hand held upright, is the index on the left? (for drawing)"""
    p = _upright(np.asarray(points, dtype=np.float64))
    return p[:, 5, 0] < p[:, 17, 0]


def canonicalize(points: np.ndarray) -> np.ndarray:
    """(N, 21, 2) points -> (N, 21, 2) in the canonical hand frame."""
    p = _upright(points)
    flip = p[:, 5, 0] < p[:, 17, 0]
    p[flip, :, 0] *= -1
    return p


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def _chain_shape(joints: np.ndarray) -> list[np.ndarray]:
    """Bend and straightness of a chain of 5 points (root + 4 joints)."""
    bones = joints[:, 1:] - joints[:, :-1]
    dirs = _unit(bones)
    bend = (dirs[:, :-1] * dirs[:, 1:]).sum(-1)  # cos of each joint angle, 1 = straight
    length = np.linalg.norm(bones[:, 1:], axis=-1).sum(-1)
    straightness = np.linalg.norm(joints[:, 4] - joints[:, 1], axis=-1) / np.maximum(length, 1e-6)
    return [bend, straightness[:, None], length[:, None]]


def _finger_block(p: np.ndarray, palm_centre: np.ndarray, chain: tuple[int, ...]) -> np.ndarray:
    n = len(p)
    joints = p[:, (0, *chain)]  # wrist, knuckle, two middle joints, tip
    # The finger's joints seen from its own knuckle, with its palm bone pointing up.
    local = np.einsum("nij,nkj->nki", _rotation_to_up(joints[:, 1] - joints[:, 0]), joints[:, 2:] - joints[:, 1:2])
    tip_wrist = np.linalg.norm(joints[:, 4], axis=-1)
    mid_wrist = np.linalg.norm(joints[:, 2], axis=-1)
    return np.concatenate(
        [
            *_chain_shape(joints),
            local.reshape(n, -1),
            np.stack(
                [
                    tip_wrist,
                    mid_wrist,
                    tip_wrist / np.maximum(mid_wrist, 1e-6),
                    np.linalg.norm(joints[:, 4] - palm_centre, axis=-1),
                ],
                1,
            ),
        ],
        1,
    )


def _thumb_block(p: np.ndarray, palm_centre: np.ndarray) -> np.ndarray:
    n = len(p)
    tip = p[:, 4]
    thumb_dir = _unit(tip - p[:, 2])
    index_bone = _unit(p[:, 5] - p[:, 0])
    return np.concatenate(
        [
            p[:, 1:5].reshape(n, -1),
            *_chain_shape(p[:, :5]),
            np.stack([np.linalg.norm(tip - p[:, k], axis=-1) for k in PALM_POINTS], 1),
            np.linalg.norm(tip - palm_centre, axis=-1)[:, None],
            (thumb_dir * index_bone).sum(-1, keepdims=True),
        ],
        1,
    )


def _palm_block(p: np.ndarray) -> np.ndarray:
    n = len(p)
    return np.concatenate(
        [
            p[:, (1, *PALM_POINTS)].reshape(n, -1),
            np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=-1) for a, b in PALM_PAIRS], 1),
        ],
        1,
    )


def featurize(points: np.ndarray) -> np.ndarray:
    """(N, 21, 2) square-pixel landmarks -> (N, NUM_FEATURES) float32.

    Layout: [index | middle | ring | pinky] (FINGER_DIM each), thumb, palm.
    """
    p = canonicalize(np.asarray(points, dtype=np.float64))
    palm_centre = p[:, (0, *PALM_POINTS)].mean(1)
    blocks = [_finger_block(p, palm_centre, chain) for chain in LONG_FINGERS]
    blocks += [_thumb_block(p, palm_centre), _palm_block(p)]
    return np.concatenate(blocks, axis=1).astype(np.float32)


_probe = np.random.default_rng(0).normal(size=(1, 21, 2))
FINGER_DIM = _finger_block(canonicalize(_probe), np.zeros((1, 2)), LONG_FINGERS[0]).shape[1]
THUMB_DIM = _thumb_block(canonicalize(_probe), np.zeros((1, 2))).shape[1]
PALM_DIM = _palm_block(canonicalize(_probe)).shape[1]
NUM_FEATURES = 4 * FINGER_DIM + THUMB_DIM + PALM_DIM

"""Hand landmark detection with MediaPipe's HandLandmarker.

MediaPipe finds up to N hands per frame and returns 21 keypoints per hand.
Our own model (see model.py) only ever sees those keypoints, never pixels.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices (MediaPipe hand topology).
WRIST = 0
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_NAMES = tuple(FINGERS)


@dataclass
class Hand:
    points: np.ndarray  # (21, 2) pixel coordinates in the frame
    handedness: str  # "Left" / "Right" as MediaPipe reports it
    score: float


def ensure_landmarker_model() -> Path:
    if not LANDMARKER_PATH.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand landmarker to {LANDMARKER_PATH} ...")
        tmp = LANDMARKER_PATH.with_suffix(".part")
        urllib.request.urlretrieve(LANDMARKER_URL, tmp)
        tmp.rename(LANDMARKER_PATH)
    return LANDMARKER_PATH


class HandDetector:
    """Thin wrapper that turns RGB frames into a list of `Hand`s."""

    def __init__(self, num_hands: int = 2, video: bool = True, min_confidence: float = 0.5):
        self.video = video
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_landmarker_model())),
            running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> list[Hand]:
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            # VIDEO mode needs strictly increasing timestamps.
            ts = max(int(timestamp_ms or 0), self._last_ts + 1)
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
            hands.append(Hand(pts, handed[0].category_name, float(handed[0].score)))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

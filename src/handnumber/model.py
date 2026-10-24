"""FingerNet: decides, finger by finger, whether each one is up.

The number shown is just how many fingers are up, so the model never needs to have
seen a particular combination (say, thumb + ring + pinky) to count it correctly.

It is two small MLPs rather than one big one:
  - a finger net, shared by index/middle/ring/pinky, that sees one finger's own
    geometry plus the palm's shape (and which finger it is)
  - a thumb net that sees the thumb plus the palm's shape

A single MLP over the whole hand scored higher on HaGRID's own test set (99.3% vs
98.4%), but it learned which fingers tend to be up *together* in HaGRID's gestures
rather than looking at each finger. On photos from another source (Google's
rock-paper-scissors set) it fell to 81%, where this design gets 93%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FINGER_DIM, NUM_FEATURES, PALM_DIM, THUMB_DIM, featurize
from .landmarks import MODELS_DIR

CHECKPOINT = MODELS_DIR / "fingernet.pt"


def _mlp(n_in: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, 1),
    )  # fmt: skip


class FingerNet(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        # Feature standardisation lives in the model so the checkpoint is self-contained.
        self.register_buffer("mean", torch.zeros(NUM_FEATURES))
        self.register_buffer("std", torch.ones(NUM_FEATURES))
        self.register_buffer("finger_id", torch.eye(4))
        self.finger = _mlp(FINGER_DIM + PALM_DIM + 4, hidden, dropout)
        self.thumb = _mlp(THUMB_DIM + PALM_DIM, hidden, dropout)
        self.config = {"hidden": hidden, "dropout": dropout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, NUM_FEATURES) -> (N, 5) logits: thumb, index, middle, ring, pinky."""
        x = (x - self.mean) / self.std
        n = len(x)
        fingers = x[:, : 4 * FINGER_DIM].reshape(n, 4, FINGER_DIM)
        thumb = x[:, 4 * FINGER_DIM : 4 * FINGER_DIM + THUMB_DIM]
        palm = x[:, 4 * FINGER_DIM + THUMB_DIM :]
        per_finger = torch.cat(
            [fingers, palm[:, None].expand(n, 4, PALM_DIM), self.finger_id.expand(n, 4, 4)], dim=-1
        )
        return torch.cat([self.thumb(torch.cat([thumb, palm], 1)), self.finger(per_finger).squeeze(-1)], 1)

    def save(self, path: Path = CHECKPOINT, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: Path = CHECKPOINT) -> "FingerNet":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()


class FingerCounter:
    """Runtime wrapper: square-pixel landmarks in, per-finger probabilities out."""

    def __init__(self, path: Path = CHECKPOINT):
        if not path.exists():
            raise SystemExit(f"No trained model at {path}. Train one with: uv run handnumber-train")
        self.model = FingerNet.load(path)
        torch.set_num_threads(1)  # the model is tiny; threads only add overhead

    @torch.no_grad()
    def finger_probs(self, points: np.ndarray) -> np.ndarray:
        """(N, 21, 2) -> (N, 5) probability that each finger is extended."""
        if len(points) == 0:
            return np.zeros((0, 5), np.float32)
        x = torch.from_numpy(featurize(points))
        return torch.sigmoid(self.model(x)).numpy()

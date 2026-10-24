import numpy as np
import pytest

from handnumber.app import FingerTracker
from handnumber.evaluate import EVAL_DIR, detect_set
from handnumber.landmarks import Hand
from handnumber.model import CHECKPOINT, FingerCounter, FingerNet


def hand_at(x: float) -> Hand:
    pts = np.random.default_rng(0).normal(size=(21, 2)).astype(np.float32) * 10
    pts[9] = (0, -60)  # give it a palm size
    return Hand(pts + (x, 300), "Right", 0.99)


def test_tracker_smooths_out_a_one_frame_blip():
    tracker = FingerTracker()
    up = np.array([[0, 1, 1, 0, 0]], np.float32)
    for t in range(5):
        tracker.update([hand_at(100)], up, t / 30)
    blip = np.array([[0, 1, 1, 1, 0]], np.float32)  # ring finger flickers up for one frame
    (track,) = tracker.update([hand_at(102)], blip, 5 / 30)
    assert track.up.sum() == 2
    for t in range(6, 9):  # ...but if it stays up, it's counted within a few frames
        (track,) = tracker.update([hand_at(102)], blip, t / 30)
    assert track.up.sum() == 3


def test_tracker_follows_two_hands_and_forgets_lost_ones():
    tracker = FingerTracker(ttl=0.3)
    probs = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 0, 0]], np.float32)
    tracks = tracker.update([hand_at(900), hand_at(100)], probs, 0.0)
    assert [int(t.up.sum()) for t in tracks] == [1, 5]  # sorted left to right on screen
    tracker.update([hand_at(100)], probs[1:], 0.1)
    assert len(tracker.update([hand_at(100)], probs[1:], 0.5)) == 1


def test_model_shapes():
    out = FingerNet()(__import__("torch").zeros(3, FingerNet().mean.shape[0]))
    assert out.shape == (3, 5)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no trained model")
@pytest.mark.skipif(not (EVAL_DIR / "hagrid").exists(), reason="eval photos not downloaded")
def test_end_to_end_on_held_out_photos():
    d = detect_set(EVAL_DIR / "hagrid")
    probs = FingerCounter().finger_probs(d["points"][d["found"]])
    accuracy = np.mean((probs > 0.5).sum(1) == d["true"][d["found"]])
    assert d["found"].mean() > 0.9
    assert accuracy > 0.95

import numpy as np
import pytest

from handnumber.data import build_dataset, hagrid_to_square
from handnumber.features import NUM_FEATURES, canonicalize, featurize


@pytest.fixture(scope="module")
def hands():
    d = build_dataset()
    idx = np.random.default_rng(0).choice(len(d["points"]), 256, replace=False)
    return hagrid_to_square(d["points"][idx]).astype(np.float64)


def rotate(p, angle):
    c, s = np.cos(angle), np.sin(angle)
    return p @ np.array([[c, -s], [s, c]]).T


def test_shape(hands):
    f = featurize(hands)
    assert f.shape == (len(hands), NUM_FEATURES)
    assert np.isfinite(f).all()


def test_canonical_pose(hands):
    p = canonicalize(hands)
    assert np.allclose(p[:, 0], 0)  # wrist at origin
    assert np.allclose(p[:, 9, 0], 0, atol=1e-6)  # middle knuckle straight above it...
    assert (p[:, 9, 1] < 0).all()  # ...(image y grows downwards)
    assert (p[:, 5, 0] >= p[:, 17, 0]).all()  # index knuckle right of pinky knuckle


@pytest.mark.parametrize(
    "transform",
    [
        lambda p: p + np.array([123.0, -45.0]),
        lambda p: p * 3.7,
        lambda p: rotate(p, 0.9),
        lambda p: rotate(p, np.pi),
        lambda p: p * np.array([-1.0, 1.0]),  # mirror: left hand <-> right hand
    ],
    ids=["translate", "scale", "rotate", "upside-down", "mirror"],
)
def test_invariance(hands, transform):
    assert np.allclose(featurize(hands), featurize(transform(hands)), atol=1e-4)

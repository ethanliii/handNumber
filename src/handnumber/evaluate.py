"""End-to-end test on real photos: pixels -> MediaPipe -> FingerNet -> count.

Expects data/eval/<set>/<true count>/*.jpg (scripts/fetch_eval_images.py makes these).

    uv run handnumber-eval                 # every set under data/eval
    uv run handnumber-eval --errors out/   # also save annotated images of mistakes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .data import DATA_DIR
from .landmarks import FINGER_NAMES, HandDetector
from .model import FingerCounter
from .render import draw_skeleton
from .rules import rule_based_fingers

EVAL_DIR = DATA_DIR / "eval"


def central_hand(hands, shape):
    """The photo's subject: the detected hand nearest the image centre."""
    h, w = shape[:2]
    return min(hands, key=lambda hd: np.linalg.norm(hd.points.mean(0) - (w / 2, h / 2)))


def detect_set(root: Path, detector: HandDetector | None = None) -> dict[str, np.ndarray]:
    """MediaPipe landmarks for every photo in an eval set, cached next to the photos.

    Returns {files, true, found, points}; `points` is NaN where no hand was found.
    """
    files = sorted(str(p.relative_to(root)) for p in root.glob("*/*.jpg") if p.parent.name.isdigit())
    cache = root / "landmarks.npz"
    if cache.exists():
        cached = dict(np.load(cache))
        if cached["files"].tolist() == files:
            return cached
    own = detector is None
    detector = detector or HandDetector(num_hands=2, video=False)
    points = np.full((len(files), 21, 2), np.nan, np.float32)
    for i, f in enumerate(files):
        bgr = cv2.imread(str(root / f))
        hands = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if hands:
            points[i] = central_hand(hands, bgr.shape).points
    if own:
        detector.close()
    out = {
        "files": np.array(files),
        "true": np.array([int(Path(f).parent.name) for f in files]),
        "found": ~np.isnan(points[:, 0, 0]),
        "points": points,
    }
    np.savez(cache, **out)
    return out


def evaluate_set(root: Path, counter: FingerCounter, errors_dir: Path | None) -> dict:
    d = detect_set(root)
    pts, true = d["points"][d["found"]], d["true"][d["found"]]
    files = d["files"][d["found"]]
    probs = counter.finger_probs(pts)
    ml = (probs > 0.5).sum(1)
    rules = rule_based_fingers(pts).sum(1)

    n = len(d["files"])
    print(f"\n[{root.name}] {n} photos, hand found in {len(pts)} ({len(pts) / max(n, 1):.1%})")
    print(f"  count accuracy   FingerNet {np.mean(ml == true):.1%}   rules {np.mean(rules == true):.1%}")
    for c in np.unique(true):
        sel = true == c
        print(
            f"  true {c}: n={sel.sum():3d}  FingerNet {np.mean(ml[sel] == c):6.1%}  "
            f"rules {np.mean(rules[sel] == c):6.1%}   predicted counts {np.bincount(ml[sel], minlength=6).tolist()}"
        )

    if errors_dir:
        for f, p, hand_probs, t, m in zip(files, pts, probs, true, ml):
            if m == t:
                continue
            bgr = cv2.imread(str(root / f))
            draw_skeleton(bgr, p, hand_probs)
            label = " ".join(f"{n[0].upper()}{v:.2f}" for n, v in zip(FINGER_NAMES, hand_probs))
            cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            out = errors_dir / root.name / f"true{t}_pred{m}_{Path(f).name}"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), bgr)
    return {"photos": n, "found": len(pts), "accuracy": float(np.mean(ml == true)), "rules": float(np.mean(rules == true))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="*", help="subfolders of data/eval (default: all)")
    ap.add_argument("--errors", type=Path, help="write annotated images of mistakes here")
    args = ap.parse_args()

    roots = [EVAL_DIR / s for s in args.sets] if args.sets else sorted(p for p in EVAL_DIR.iterdir() if p.is_dir())
    if not roots:
        raise SystemExit(f"No eval sets in {EVAL_DIR}. Run: uv run python scripts/fetch_eval_images.py")
    counter = FingerCounter()
    for root in roots:
        evaluate_set(root, counter, args.errors)


if __name__ == "__main__":
    main()

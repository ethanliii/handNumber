"""Live webcam finger counter.

    uv run handnumber                    # default camera
    uv run handnumber --camera 1         # another camera (e.g. iPhone Continuity Camera)
    uv run handnumber --image hand.jpg --out counted.png   # run on a photo instead
    uv run handnumber --video clip.mp4 --out counted.mp4   # ...or a video

Keys:  Q / Esc quit   C next camera   M mirror on/off   D details (probabilities, fps)   F fullscreen
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import index_left_of_pinky
from .landmarks import Hand, HandDetector
from .model import FingerCounter
from .render import Hud, draw_skeleton, hand_pill

WINDOW = "handNumber"


@dataclass(eq=False)  # compare tracks by identity, not by their numpy fields
class Track:
    """One hand followed across frames, with smoothed per-finger probabilities."""

    points: np.ndarray
    probs: np.ndarray
    up: np.ndarray
    mirror: bool
    seen: float
    fresh: bool = True
    born: float = field(default_factory=time.monotonic)


class FingerTracker:
    """Matches hands frame to frame and stabilises what each finger is doing.

    Raw per-frame predictions flicker when a finger is half-bent, so each hand keeps
    an exponential moving average of its probabilities, and a finger only flips
    state once the average clearly crosses the line (hysteresis).
    """

    def __init__(self, smoothing: float = 0.5, on: float = 0.6, off: float = 0.4, ttl: float = 0.3):
        self.smoothing, self.on, self.off, self.ttl = smoothing, on, off, ttl
        self.tracks: list[Track] = []

    def update(self, hands: list[Hand], probs: np.ndarray, now: float) -> list[Track]:
        for t in self.tracks:
            t.fresh = False
        free = list(self.tracks)
        mirrors = index_left_of_pinky(np.stack([h.points for h in hands])) if hands else []
        for hand, p, mirror in zip(hands, probs, mirrors):
            centre = hand.points.mean(0)
            size = np.linalg.norm(hand.points[9] - hand.points[0])
            match = min(free, key=lambda t: np.linalg.norm(t.points.mean(0) - centre), default=None)
            if match is not None and np.linalg.norm(match.points.mean(0) - centre) < 3 * size:
                free.remove(match)
                match.probs = self.smoothing * p + (1 - self.smoothing) * match.probs
                match.up = np.where(match.up, match.probs > self.off, match.probs > self.on)
                match.points, match.mirror, match.seen, match.fresh = hand.points, bool(not mirror), now, True
            else:
                self.tracks.append(Track(hand.points, p.copy(), p > 0.5, bool(not mirror), now))
        self.tracks = [t for t in self.tracks if now - t.seen < self.ttl]
        return sorted(self.tracks, key=lambda t: t.points[:, 0].mean())


def compose(frame: np.ndarray, tracks: list[Track], hud: Hud, now: float, details: bool, stats: str) -> None:
    for t in tracks:
        if t.fresh:
            draw_skeleton(frame, t.points, t.probs)
            hand_pill(frame, t.points, t.up)
    hud.draw(frame, [{"probs": t.probs, "up": t.up, "mirror": t.mirror} for t in tracks], now, details, stats)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # The first frames can take a moment (or wait on the macOS permission prompt).
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


CAMERA_HELP = """Couldn't read from camera {index}.
  - macOS: System Settings > Privacy & Security > Camera, and allow the app you ran this
    from (Terminal, iTerm, Ghostty, VS Code, ...). Then quit and reopen that app.
  - Another camera? Try:  uv run handnumber --camera 1"""


def run_camera(args) -> None:
    counter = FingerCounter()
    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        raise SystemExit(CAMERA_HELP.format(index=args.camera))
    camera, mirror, details, fullscreen = args.camera, not args.no_mirror, False, False
    tracker, hud = FingerTracker(), Hud()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fps, last = 0.0, time.monotonic()

    with HandDetector(num_hands=args.hands, video=True) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = open_camera(camera, args.width, args.height)
                if cap is None:
                    raise SystemExit(CAMERA_HELP.format(index=camera))
                continue
            if mirror:
                frame = cv2.flip(frame, 1)  # selfie view: your right hand on the right

            now = time.monotonic()
            t0 = time.perf_counter()
            hands = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(now * 1000))
            probs = counter.finger_probs(np.stack([h.points for h in hands])) if hands else np.zeros((0, 5))
            infer_ms = (time.perf_counter() - t0) * 1000
            tracks = tracker.update(hands, probs, now)

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            compose(frame, tracks, hud, now, details, f"{fps:.0f} FPS  ·  {infer_ms:.0f} MS")
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                mirror = not mirror
            elif key == ord("d"):
                details = not details
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                )
            elif key == ord("c"):
                for step in range(1, 5):
                    nxt = (camera + step) % 5
                    new = open_camera(nxt, args.width, args.height)
                    if new is not None:
                        cap.release()
                        cap, camera = new, nxt
                        print(f"camera {camera}")
                        break

    cap.release()
    cv2.destroyAllWindows()


def run_video(args) -> None:
    """Same pipeline as the webcam, frame by frame over a video file."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Can't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    counter, tracker, hud = FingerCounter(), FingerTracker(), Hud()
    writer = None
    counts = []
    with HandDetector(num_hands=args.hands, video=True) as detector:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = i / fps  # media time, so smoothing and animations match the clip
            hands = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(now * 1000))
            probs = counter.finger_probs(np.stack([h.points for h in hands])) if hands else np.zeros((0, 5))
            tracks = tracker.update(hands, probs, now)
            counts.append(sum(int(t.up.sum()) for t in tracks) if tracks else None)
            compose(frame, tracks, hud, now, args.details, "")
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            else:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF in (ord("q"), 27):
                    break
            i += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"{i} frames; count per frame: {counts}")


def run_image(args) -> None:
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Can't read {args.image}")
    if min(frame.shape[:2]) < 720:  # the HUD is sized for a 720p webcam frame
        scale = 720 / min(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    counter = FingerCounter()
    with HandDetector(num_hands=args.hands, video=False) as detector:
        hands = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    probs = counter.finger_probs(np.stack([h.points for h in hands])) if hands else np.zeros((0, 5))
    tracks = FingerTracker().update(hands, probs, 0.0)
    total = sum(int(t.up.sum()) for t in tracks)
    print(f"{len(tracks)} hand(s), {total} finger(s): " + ", ".join(str(int(t.up.sum())) for t in tracks))
    hud = Hud()
    hud.shown, hud.changed_at = total, -10.0  # no pop animation in a still
    compose(frame, tracks, hud, 0.0, args.details, "")
    if args.out:
        cv2.imwrite(args.out, frame)
    else:
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hands", type=int, default=2, help="max hands to count (default 2)")
    ap.add_argument("--no-mirror", action="store_true", help="show the raw camera image instead of a selfie view")
    ap.add_argument("--image", help="count fingers in a photo instead of the webcam")
    ap.add_argument("--video", help="count fingers in a video file instead of the webcam")
    ap.add_argument("--out", help="with --image/--video: save the annotated result here instead of showing it")
    ap.add_argument("--details", action="store_true", help="with --image/--video: show per-finger probabilities")
    args = ap.parse_args()
    if args.image:
        run_image(args)
    elif args.video:
        run_video(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()

"""Download real photos for an end-to-end test (pixels -> MediaPipe -> FingerNet).

Writes data/eval/<set>/<finger count>/<name>.jpg for two sets:
  rps     Google's rock/paper/scissors sample photos (rock = 0, scissors = 2, paper = 5)
  hagrid  photos of HaGRID *test-split* people, cropped around the labelled hand.
          Each gesture class lives in its own ~40 GB zip; we read only the bytes we need.

    uv run python scripts/fetch_eval_images.py --per-gesture 20   # safe to re-run; resumes
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import RangeFile  # noqa: E402

from handnumber.data import ANNOTATIONS_ZIP, DATA_DIR, GESTURE_FINGERS  # noqa: E402

EVAL_DIR = DATA_DIR / "eval"
RPS_URL = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip"
V1 = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/"
V2 = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip/"
V1_GESTURES = {
    "call", "dislike", "fist", "four", "like", "mute", "ok", "palm", "peace",
    "peace_inverted", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
}  # fmt: skip


def fetch_rps() -> None:
    out = EVAL_DIR / "rps"
    with urllib.request.urlopen(RPS_URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in z.namelist():
        parts = Path(name).parts
        if len(parts) == 3 and parts[1] in ("rock", "paper", "scissors") and name.lower().endswith((".jpg", ".png")):
            count = {"rock": 0, "scissors": 2, "paper": 5}[parts[1]]
            dest = out / str(count) / f"{parts[1]}_{Path(name).stem}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(z.read(name))).convert("RGB").save(dest, quality=92)
    print("rps:", sum(1 for _ in out.rglob("*.jpg")), "images")


def fetch_hagrid_gesture(gesture: str, per_gesture: int, seed: int) -> int:
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as z:
        records = json.loads(z.read(f"annotations/test/{gesture}.json"))
    remote = zipfile.ZipFile(RangeFile((V1 if gesture in V1_GESTURES else V2) + f"{gesture}.zip"))
    members = {Path(n).stem: n for n in remote.namelist() if n.endswith(".jpg")}
    keys = [k for k in records if k in members and gesture in records[k]["labels"]]
    random.Random(seed).shuffle(keys)

    count = sum(GESTURE_FINGERS[gesture])
    saved = 0
    for key in keys[:per_gesture]:
        dest = EVAL_DIR / "hagrid" / str(count) / f"{gesture}_{key[:8]}.jpg"
        if dest.exists():
            saved += 1
            continue
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(remote.read(members[key])))).convert("RGB")
        W, H = img.size
        x, y, w, h = records[key]["bboxes"][records[key]["labels"].index(gesture)]
        cx, cy = (x + w / 2) * W, (y + h / 2) * H
        side = max(w * W, h * H) * 2.4
        box = (max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2))
        crop = img.crop(tuple(int(v) for v in box))
        crop.thumbnail((720, 720))
        dest.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dest, quality=92)
        saved += 1
    print(f"hagrid {gesture:16s} {saved} images")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gesture", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (EVAL_DIR / "rps").exists():
        fetch_rps()
    # Only gestures where every finger's state is known, so the true count is too.
    gestures = [g for g, f in GESTURE_FINGERS.items() if min(f) >= 0 and g != "thumb_index2"]
    with ThreadPoolExecutor(4) as pool:
        total = sum(pool.map(lambda g: fetch_hagrid_gesture(g, args.per_gesture, args.seed), gestures))
    print("hagrid:", total, "images")


if __name__ == "__main__":
    main()

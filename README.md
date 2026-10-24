# handNumber

Hold fingers up to your webcam and it counts them, with up to two hands (0–10).

```bash
uv run handnumber
```

The first run downloads MediaPipe's hand landmark model (8 MB). The trained finger
model ships in `models/fingernet.pt`, so there's no training step.

On macOS the first run triggers a camera permission prompt for your terminal app. If
you get nothing, go to System Settings → Privacy & Security → Camera and allow
Terminal / iTerm / Ghostty / VS Code, whichever you ran it from, then restart that app.

| Key | |
|---|---|
| `Q` / `Esc` | quit |
| `C` | next camera (e.g. if your iPhone's Continuity Camera grabbed camera 0) |
| `M` | mirror on/off |
| `D` | details: per-finger probability bars, FPS, latency |
| `F` | fullscreen |

It also runs on files: `uv run handnumber --image hand.jpg --out counted.png` or
`--video clip.mp4 --out counted.mp4`.

## How it works

```
webcam frame
  → MediaPipe HandLandmarker        21 keypoints per hand (pretrained, off the shelf)
  → canonical pose                  wrist at origin, palm-sized, rotated upright, mirrored to one chirality
  → features                        per-finger joint angles & distances, thumb geometry, palm shape
  → FingerNet (ours, PyTorch)       P(finger is up) for thumb, index, middle, ring, pinky
  → smoothing + hysteresis          per tracked hand, so a half-bent finger doesn't flicker
  → count                           fingers up, summed over hands
```

**The model predicts each finger's state, not "the number".** The count is the sum. A
model trained to output the number directly can only count finger combinations it
has seen. This one can count thumb + ring + pinky even though nobody in the training
data holds that up.

**Training data: [HaGRID v2](https://github.com/hukenovs/hagrid).** It has 1.3M hands
from many thousands of people, already annotated with MediaPipe landmarks. HaGRID labels
gestures ("peace", "call", "four"…), not finger counts, so `data.py` maps each
gesture to the five finger states it implies:
- I checked each mapping against the data with a geometric heuristic.
- Where a gesture doesn't settle a finger (the thumb in "stop", "mute"), that finger
  is masked out of the loss rather than guessed.
- Gestures that are unreadable in 2D ("point", which aims at the camera) are dropped.

That leaves 586k training hands. Train, val and test are different people.

**Architecture: two small MLPs instead of one.** A finger net, shared by
index/middle/ring/pinky, sees only that finger's geometry plus the palm's shape. A
thumb net sees the thumb plus the palm. My first version was a single MLP over the
whole hand. It scored 99.3% on HaGRID's test set but only 81% on photos from another
source. It had learned which fingers tend to be up *together* in HaGRID's gestures
instead of looking at each finger. Splitting the network fixed most of that. The
rest came from hiding the index finger's joints from the thumb net, which had been
inferring the thumb from the index finger.

## Results

Count accuracy (the number shown is exactly right):

| Test set | FingerNet | hand-written rules¹ | single MLP (v1) |
|---|---|---|---|
| HaGRID test split: 123k hands from unseen people (landmarks) | **98.4%** | 83.0% | 99.3% |
| HaGRID test photos, end to end: 370 photos of counts 0–5 | **97.0%** | 80.0% | – |
| Google rock-paper-scissors photos, end to end: 363 photos² | **93.4%** | 57.9% | 81.3% |

¹ The classic tutorial approach: a finger is up if it's straight and its tip is
farther from the wrist than its middle joint (`rules.py`).
² From a different source, with lots of sideways and top-down hands. I used this set
to choose the architecture, so treat it as a dev set. The HaGRID photos were held out
until the end.

What still goes wrong:
- **The thumb.** It's the only finger that bends sideways, so in a 2D photo a thumb
  held flat against the hand is often genuinely ambiguous. Almost all remaining
  errors on open hands are thumbs.
- **MediaPipe misplacing landmarks.** Sometimes it lays the skeleton over the curled
  fingers and misses a raised one entirely. FingerNet only sees landmarks, so it
  can't recover. Live video helps, because MediaPipe tracks between frames.
- **2D only.** HaGRID's landmarks have no depth, so the model never sees it. Hands
  held edge-on or pointing at the camera are the hardest cases.

## Retrain / re-evaluate

```bash
mkdir -p data/hagrid && curl -L -o data/hagrid/annotations.zip \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip  # 719 MB
uv run handnumber-train                           # ~1 min on an M-series Mac; writes models/fingernet.pt
uv run python scripts/fetch_eval_images.py        # test photos (reads single files out of HaGRID's 40 GB zips)
uv run handnumber-eval --errors errors/           # end-to-end accuracy, plus images of every mistake
uv run pytest
```

## Layout

```
src/handnumber/
  landmarks.py   MediaPipe wrapper (+ model download)
  data.py        HaGRID gestures -> per-finger labels
  features.py    canonical pose + feature vector (shared by training and the app)
  model.py       FingerNet
  train.py       training + HaGRID test report
  rules.py       hand-written baseline
  evaluate.py    end-to-end photo evaluation
  render.py      everything drawn on screen
  app.py         webcam / image / video front end, hand tracking & smoothing
scripts/         downloading evaluation photos
tests/           feature invariances, tracker, end-to-end accuracy
```

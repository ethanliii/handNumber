"""Train FingerNet on HaGRID and report how well it counts on held-out people.

    uv run handnumber-train            # ~1-2 min on an M-series Mac
    uv run handnumber-train --epochs 5 # quicker

HaGRID's train / val / test splits contain different people, so the test numbers
are for hands the model has never seen.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from .data import GESTURES, HAGRID_ASPECT, SPLITS, build_dataset, hagrid_to_square
from .features import featurize
from .landmarks import FINGER_NAMES, MODELS_DIR
from .model import CHECKPOINT, FingerNet
from .rules import rule_based_fingers

METRICS_PATH = MODELS_DIR / "fingernet_metrics.json"


def augment(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random distortions of normalised HaGRID landmarks, returned in square units.

    - aspect jitter: we only know most photos are 3:4, not which ones aren't
    - a random stretch along a random axis, a cheap stand-in for viewing angle
    - per-keypoint noise, since webcam landmarks wobble frame to frame
    """
    n = len(points)
    p = points.astype(np.float64)
    p[..., 0] *= HAGRID_ASPECT * np.exp(rng.normal(0, 0.15, n))[:, None]

    phi = rng.uniform(0, np.pi, n)
    c, s = np.cos(phi), np.sin(phi)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    stretch = np.zeros((n, 2, 2))
    stretch[:, 0, 0] = np.exp(rng.normal(0, 0.12, n))
    stretch[:, 1, 1] = 1.0
    affine = rot @ stretch @ rot.transpose(0, 2, 1)
    p = np.einsum("nij,nkj->nki", affine, p - p[:, :1])

    palm = np.linalg.norm(p[:, 9] - p[:, 0], axis=-1)
    p += rng.normal(0, 1, p.shape) * (0.02 * palm)[:, None, None]
    return p


def finger_metrics(pred: np.ndarray, true: np.ndarray, gesture: np.ndarray) -> dict:
    known = true >= 0
    complete = known.all(1)
    pred_count, true_count = pred.sum(1), true.sum(1)
    correct = (pred_count == true_count) & complete

    confusion = np.zeros((6, 6), int)
    np.add.at(confusion, (true_count[complete], pred_count[complete]), 1)
    per_gesture = {
        GESTURES[g]: round(float(correct[(gesture == g) & complete].mean()), 4)
        for g in np.unique(gesture)
        if ((gesture == g) & complete).any()
    }
    return {
        "count_accuracy": round(float(correct.sum() / complete.sum()), 4),
        "finger_accuracy": {
            name: round(float((pred[:, i] == true[:, i])[known[:, i]].mean()), 4)
            for i, name in enumerate(FINGER_NAMES)
        },
        "count_accuracy_by_gesture": per_gesture,
        "count_confusion": confusion.tolist(),  # rows = true count, cols = predicted
    }


@torch.no_grad()
def predict(model: FingerNet, x: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    logits = torch.cat([model(b.to(device)).cpu() for b in x.split(65536)])
    return (logits > 0).numpy().astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = build_dataset()
    sets = {name: {k: v[data["split"] == i] for k, v in data.items()} for i, name in enumerate(SPLITS)}
    train = sets["train"]
    for name, s in sets.items():
        print(f"{name:5s} {len(s['points']):7d} hands")

    # Each gesture gets the same total weight, so "four" (one gesture) counts as much
    # as the six different gestures that all show a single finger.
    counts = np.bincount(train["gesture"], minlength=len(GESTURES))
    sample_w = torch.from_numpy((len(train["gesture"]) / (len(GESTURES) * counts))[train["gesture"]]).float()
    y = torch.from_numpy(train["fingers"].astype(np.float32))
    mask = (y >= 0).float()
    y = y.clamp(min=0)

    eval_x = {k: torch.from_numpy(featurize(hagrid_to_square(sets[k]["points"]))) for k in ("val", "test")}

    model = FingerNet()
    x0 = torch.from_numpy(featurize(augment(train["points"], rng)))
    model.mean.copy_(x0.mean(0))
    model.std.copy_(x0.std(0).clamp(min=1e-4))
    model.to(device)

    steps_per_epoch = (len(y) + args.batch_size - 1) // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        t0 = time.time()
        x = x0 if epoch == 0 else torch.from_numpy(featurize(augment(train["points"], rng)))
        perm = torch.randperm(len(y))
        model.train()
        total = 0.0
        for idx in perm.split(args.batch_size):
            xb, yb, mb, wb = (t[idx].to(device) for t in (x, y, mask, sample_w))
            # Light label smoothing: HaGRID labels are per gesture, and people don't
            # always hold a gesture exactly as defined.
            loss = (bce(model(xb), yb * 0.96 + 0.02) * mb).sum(1) * wb
            loss = loss.sum() / mb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)

        val = finger_metrics(predict(model, eval_x["val"], device), sets["val"]["fingers"], sets["val"]["gesture"])
        acc = val["count_accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:2d}  loss {total / len(y):.4f}  val count acc {acc:.4f}  ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_state)
    model.to("cpu")
    test = sets["test"]
    ml = finger_metrics(predict(model, eval_x["test"], "cpu"), test["fingers"], test["gesture"])
    rules = finger_metrics(rule_based_fingers(hagrid_to_square(test["points"])), test["fingers"], test["gesture"])

    print("\n=== HaGRID test set (people never seen in training) ===")
    print(f"{'':22s}{'FingerNet':>10s}{'rules':>10s}")
    print(f"{'count accuracy':22s}{ml['count_accuracy']:10.2%}{rules['count_accuracy']:10.2%}")
    for name in FINGER_NAMES:
        print(f"{name + ' accuracy':22s}{ml['finger_accuracy'][name]:10.2%}{rules['finger_accuracy'][name]:10.2%}")
    print("\ncount accuracy by gesture:")
    for g, a in ml["count_accuracy_by_gesture"].items():
        print(f"  {g:16s} {a:7.2%}   (rules {rules['count_accuracy_by_gesture'][g]:.2%})")
    print("\ncount confusion (rows = true, cols = predicted):")
    for i, row in enumerate(ml["count_confusion"]):
        print(f"  {i}: " + " ".join(f"{v:6d}" for v in row))

    model.save(CHECKPOINT, metrics={"test": ml, "rules_test": rules})
    METRICS_PATH.write_text(json.dumps({"fingernet": ml, "rules": rules}, indent=2))
    print(f"\nsaved {CHECKPOINT} and {METRICS_PATH}")


if __name__ == "__main__":
    main()

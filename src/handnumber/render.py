"""Everything drawn on screen: a glowing per-finger skeleton, a frosted-glass HUD
with a gradient numeral, and a hand glyph whose fingers grow as the model grows
more confident. All sprites are premultiplied BGRA so anti-aliased edges composite
cleanly onto the video.
"""

from __future__ import annotations

import functools
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .landmarks import FINGERS

# One colour per finger, thumb -> pinky: a warm-to-cool sweep (RGB).
FINGER_RGB = ((255, 94, 120), (255, 166, 77), (214, 240, 92), (70, 222, 205), (142, 128, 255))
FINGER_BGR = tuple(c[::-1] for c in FINGER_RGB)
GLASS_TINT = (34, 26, 22)  # BGR, a cool near-black
STUB_BGR = (88, 80, 76)
PALM_BGR = (52, 44, 40)
WHITE = (255, 255, 255)
PALM_EDGES = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
FONT_PATH = "/System/Library/Fonts/SFNSRounded.ttf"


# ---------------------------------------------------------------- compositing


def blend(dst: np.ndarray, sprite: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Composite a premultiplied BGRA sprite onto a BGR frame, top-left at (x, y)."""
    h, w = sprite.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    s = sprite[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32) * opacity
    roi = dst[y0:y1, x0:x1]
    roi[:] = np.clip(s[..., :3] + roi * (1 - s[..., 3:] / 255), 0, 255).astype(np.uint8)


def blend_center(dst, sprite, cx, cy, opacity=1.0, scale=1.0) -> None:
    if scale != 1.0:
        h, w = sprite.shape[:2]
        sprite = cv2.resize(sprite, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_LINEAR)
    h, w = sprite.shape[:2]
    blend(dst, sprite, round(cx - w / 2), round(cy - h / 2), opacity)


def _over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Premultiplied 'top over bottom' for float arrays in 0..255."""
    return top + bottom * (1 - top[..., 3:] / 255)


# ---------------------------------------------------------------- text


@functools.lru_cache(maxsize=None)
def _font(size: int, weight: str) -> ImageFont.FreeTypeFont:
    try:
        f = ImageFont.truetype(FONT_PATH, size)
        f.set_variation_by_name(weight)
        return f
    except (OSError, ValueError):  # not on macOS: fall back to Pillow's bundled font
        return ImageFont.load_default(size)


def _gradient(w: int, h: int, stops=FINGER_RGB, tilt: float = 0.45) -> np.ndarray:
    """(h, w, 3) BGR diagonal gradient through `stops`."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = xx / max(w - 1, 1) + tilt * yy / max(h - 1, 1)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    stops_arr = np.array([s[::-1] for s in stops], np.float32)
    pos = t * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = (pos - i)[..., None]
    return stops_arr[i] * (1 - f) + stops_arr[i + 1] * f


def _with_shadow(sprite: np.ndarray, blur: float, dy: float, opacity: float) -> np.ndarray:
    pad = int(blur * 3 + abs(dy)) + 2
    h, w = sprite.shape[:2]
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    out[pad : pad + h, pad : pad + w] = sprite
    shadow = np.zeros_like(out)
    shadow[pad + int(dy) : pad + int(dy) + h, pad : pad + w, 3] = sprite[..., 3] * opacity
    shadow[..., 3] = cv2.GaussianBlur(shadow[..., 3], (0, 0), blur)
    return np.clip(_over(out, shadow), 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=512)
def text_sprite(
    text: str,
    size: int,
    weight: str = "Bold",
    color: tuple[int, int, int] = WHITE,
    alpha: float = 1.0,
    tracking: float = 0.0,
    gradient: bool = False,
    shadow: bool = False,
) -> np.ndarray:
    """Tightly cropped, premultiplied BGRA rendering of `text` (BGR `color`)."""
    f = _font(size, weight)
    pad = size // 2
    advances = [f.getlength(ch) for ch in text]
    width = int(sum(advances) + tracking * size * max(len(text) - 1, 0)) + 2 * pad
    mask = Image.new("L", (max(width, 1), int(size * 1.6) + 2 * pad), 0)
    draw = ImageDraw.Draw(mask)
    x = float(pad)
    for ch, adv in zip(text, advances):
        draw.text((x, pad), ch, font=f, fill=255)
        x += adv + tracking * size
    box = mask.getbbox()
    a = np.asarray(mask.crop(box) if box else mask.crop((0, 0, 1, 1)), np.float32) * alpha
    h, w = a.shape
    rgb = _gradient(w, h) if gradient else np.broadcast_to(np.array(color, np.float32), (h, w, 3))
    sprite = np.dstack([rgb * a[..., None] / 255, a])
    if shadow:
        return _with_shadow(sprite, blur=size * 0.07, dy=size * 0.05, opacity=0.55)
    return np.clip(sprite, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- glass panels


@functools.lru_cache(maxsize=64)
def _rounded_mask(w: int, h: int, r: int, ss: int = 4) -> np.ndarray:
    big = np.zeros((h * ss, w * ss), np.uint8)
    R = max(1, r * ss)
    cv2.rectangle(big, (R, 0), (w * ss - R - 1, h * ss - 1), 255, -1)
    cv2.rectangle(big, (0, R), (w * ss - 1, h * ss - R - 1), 255, -1)
    for cx, cy in ((R, R), (w * ss - R - 1, R), (R, h * ss - R - 1), (w * ss - R - 1, h * ss - R - 1)):
        cv2.circle(big, (cx, cy), R, 255, -1, cv2.LINE_AA)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


@functools.lru_cache(maxsize=64)
def _glass_layers(w: int, h: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    fill = _rounded_mask(w, h, r)
    inner = np.zeros_like(fill)
    inner[1:-1, 1:-1] = _rounded_mask(w - 2, h - 2, max(r - 1, 1))
    rim = np.clip(fill - inner, 0, 1) * 0.22
    sheen = inner * np.clip(1 - np.linspace(0, 1, h) / 0.5, 0, 1)[:, None] * 0.07  # light from above
    return fill[..., None], ((rim + sheen) * 255)[..., None]


def glass(dst: np.ndarray, x: int, y: int, w: int, h: int, r: int = 24, tint: float = 0.6) -> None:
    """Frosted-glass rounded rectangle: blurred, darkened video with a lit rim."""
    H, W = dst.shape[:2]
    x, y = int(np.clip(x, 0, max(W - w, 0))), int(np.clip(y, 0, max(H - h, 0)))
    w, h = min(w, W - x), min(h, H - y)
    if w < 4 or h < 4:
        return
    roi = dst[y : y + h, x : x + w]
    small = cv2.resize(roi, (max(w // 12, 1), max(h // 12, 1)), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
    frosted = blurred.astype(np.float32) * (1 - tint) + np.array(GLASS_TINT, np.float32) * tint
    fill, light = _glass_layers(w, h, r)
    roi[:] = np.clip(roi * (1 - fill) + frosted * fill + light, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- hand skeleton


def draw_skeleton(frame: np.ndarray, points: np.ndarray, probs: np.ndarray, glow: bool = True) -> None:
    """Draw one hand: raised fingers in their own colour with a glow, the rest faint."""
    pts = np.asarray(points, np.float32)
    palm = float(np.linalg.norm(pts[9] - pts[0]))
    th = int(np.clip(palm * 0.07, 2, 12))
    margin = th * 6
    H, W = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - margin, 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + margin, (W, H))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    fixed = np.round((pts - (x0, y0)) * 16).astype(int)  # 4 bits of sub-pixel precision

    def P(i):
        return tuple(int(v) for v in fixed[i])

    def bones(chain):
        return zip(chain[:-1], chain[1:])

    up = probs >= 0.5
    chains = list(FINGERS.values())

    ghost = roi.copy()
    thin = max(th // 2, 1)
    for a, b in PALM_EDGES:
        cv2.line(ghost, P(a), P(b), WHITE, thin, cv2.LINE_AA, 4)
    for i, chain in enumerate(chains):
        if not up[i]:
            for a, b in bones(chain):
                cv2.line(ghost, P(a), P(b), WHITE, thin, cv2.LINE_AA, 4)
    roi[:] = cv2.addWeighted(ghost, 0.6, roi, 0.4, 0)

    if glow and up.any():
        layer = np.zeros_like(roi)
        for i, chain in enumerate(chains):
            if up[i]:
                for a, b in bones(chain):
                    cv2.line(layer, P(a), P(b), FINGER_BGR[i], th * 3, cv2.LINE_AA, 4)
        roi[:] = cv2.add(roi, cv2.GaussianBlur(layer, (0, 0), th * 1.8))

    for i, chain in enumerate(chains):
        if up[i]:
            for a, b in bones(chain):
                cv2.line(roi, P(a), P(b), FINGER_BGR[i], th, cv2.LINE_AA, 4)
    for j in range(21):
        cv2.circle(roi, P(j), max(th // 2, 2) * 16, WHITE, -1, cv2.LINE_AA, 4)
    for i, chain in enumerate(chains):
        if up[i]:
            cv2.circle(roi, P(chain[-1]), int(th * 1.4 * 16), FINGER_BGR[i], -1, cv2.LINE_AA, 4)
            cv2.circle(roi, P(chain[-1]), int(th * 0.6 * 16), WHITE, -1, cv2.LINE_AA, 4)


# ---------------------------------------------------------------- hand glyph

# A stylised mitten. Units: hand width = 1. For each finger:
# (base x, base y, lean from vertical in degrees, full length, thickness)
_GLYPH_W, _GLYPH_H = 1.0, 1.18
_GLYPH_PAD = 0.22  # transparent margin so the glow isn't clipped
_GLYPH = (
    (0.30, 0.86, -52, 0.36, 0.125),  # thumb
    (0.37, 0.62, -9, 0.43, 0.112),  # index
    (0.50, 0.60, -1, 0.48, 0.112),  # middle
    (0.63, 0.62, 7, 0.43, 0.112),  # ring
    (0.75, 0.67, 15, 0.34, 0.100),  # pinky
)
_PALM = (0.27, 0.57, 0.85, 1.10, 0.22)  # x0, y0, x1, y1, corner radius
PALM_TOP_BGR, PALM_BOTTOM_BGR = (78, 68, 62), (44, 37, 33)


def _mix(a, b, t):
    return tuple(float(u + (v - u) * t) for u, v in zip(a, b))


def _glyph_frame(height: int, ss: int) -> tuple[float, float, int, int]:
    U = height * ss / _GLYPH_H
    return U, _GLYPH_PAD * U, int((_GLYPH_W + 2 * _GLYPH_PAD) * U), int((_GLYPH_H + 2 * _GLYPH_PAD) * U)


@functools.lru_cache(maxsize=8)
def _glyph_palm(height: int, ss: int) -> tuple[np.ndarray, tuple[float, float]]:
    """The palm never changes, so it's built once: a rounded block lit from above."""
    U, pad, w, h = _glyph_frame(height, ss)
    x0, y0, x1, y1, r = (v * U for v in _PALM)
    mask = np.zeros((h, w), np.float32)
    pm = _rounded_mask(int(x1 - x0), int(y1 - y0), int(r), ss=1)
    mask[int(y0 + pad) : int(y0 + pad) + pm.shape[0], int(x0 + pad) : int(x0 + pad) + pm.shape[1]] = pm
    t = np.clip((np.arange(h, dtype=np.float32) - (y0 + pad)) / (y1 - y0), 0, 1)[:, None, None]
    shade = np.array(PALM_TOP_BGR, np.float32) * (1 - t) + np.array(PALM_BOTTOM_BGR, np.float32) * t
    palm = np.dstack([shade * mask[..., None], mask * 255])
    rim = np.clip(cv2.dilate(mask, np.ones((2 * ss + 1,) * 2, np.uint8)) - mask, 0, 1) * 0.3
    rim_layer = np.dstack([np.full((h, w, 3), 255, np.float32) * rim[..., None], rim * 255])
    layer = cv2.resize(_over(rim_layer, palm), (w // ss, h // ss), interpolation=cv2.INTER_AREA)
    return layer, (((x0 + x1) / 2 + pad) / ss, ((y0 + y1) / 2 + pad) / ss)


def hand_glyph(probs: np.ndarray, height: int = 104, mirror: bool = False):
    """Premultiplied BGRA glyph of a hand whose fingers grow with their probability.

    Returns (sprite, (x, y) of the palm centre inside the sprite). Probabilities are
    rounded to 1/40 so a steady hand reuses a cached glyph.
    """
    steps = tuple(int(round(float(np.clip(p, 0, 1)) * 40)) for p in probs)
    return _hand_glyph(steps, height, mirror)


@functools.lru_cache(maxsize=512)
def _hand_glyph(steps: tuple[int, ...], height: int, mirror: bool, ss: int = 3):
    U, pad, w, h = _glyph_frame(height, ss)
    # Fingers are drawn supersampled on uint8 canvases (OpenCV only anti-aliases 8-bit).
    body = np.zeros((h, w, 4), np.uint8)
    halo = np.zeros((h, w, 4), np.uint8)
    for i, (bx, by, lean, length, thick) in enumerate(_GLYPH):
        p = steps[i] / 40
        ease = p * p * (3 - 2 * p)
        reach = length * (0.28 + 0.72 * ease)
        a = (bx * U + pad, by * U + pad)
        b = (a[0] + math.sin(math.radians(lean)) * reach * U, a[1] - math.cos(math.radians(lean)) * reach * U)
        pa, pb = tuple(int(v * 16) for v in a), tuple(int(v * 16) for v in b)
        color = _mix(STUB_BGR, FINGER_BGR[i], ease)
        cv2.line(body, pa, pb, (*color, 255), int(thick * U), cv2.LINE_AA, 4)
        if p > 0.5:
            glow = (*(c * 0.8 * ease for c in FINGER_BGR[i]), 200 * ease)  # premultiplied
            cv2.line(halo, pa, pb, glow, int(thick * U * 2.0), cv2.LINE_AA, 4)

    size = (w // ss, h // ss)
    body = cv2.resize(body, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    # The glow is soft anyway, so blur it after shrinking: ~50x cheaper.
    halo = cv2.GaussianBlur(cv2.resize(halo, size, interpolation=cv2.INTER_AREA), (0, 0), 0.08 * U / ss)
    palm, (cx, cy) = _glyph_palm(height, ss)
    out = _over(palm, _over(body, halo.astype(np.float32)))
    if mirror:
        out, cx = out[:, ::-1], out.shape[1] - cx
    return np.clip(out, 0, 255).astype(np.uint8), (cx, cy)


# ---------------------------------------------------------------- small pieces


def finger_dots(dst, cx, cy, up, spacing=10, r=3) -> None:
    for i in range(5):
        x = cx + (i - 2) * spacing
        color = FINGER_BGR[i] if up[i] else (96, 90, 86)
        cv2.circle(dst, (int(x * 16), int(cy * 16)), r * 16, color, -1, cv2.LINE_AA, 4)


def hand_pill(frame: np.ndarray, points: np.ndarray, up: np.ndarray) -> None:
    """A little glass tag floating above the hand with its own count."""
    w, h = 62, 62
    top = points[:, 1].min()
    cx = float(points[:, 0].mean())
    y = top - h - 18 if top - h - 18 > 8 else points[:, 1].max() + 18
    x = int(np.clip(cx - w / 2, 8, frame.shape[1] - w - 8))
    y = int(np.clip(y, 8, frame.shape[0] - h - 8))
    glass(frame, x, y, w, h, r=20, tint=0.55)
    blend_center(frame, text_sprite(str(int(up.sum())), 30, "Heavy"), x + w / 2, y + 26)
    finger_dots(frame, x + w / 2, y + 49, up, spacing=9, r=3)


@functools.lru_cache(maxsize=8)
def keycaps(items: tuple[tuple[str, str], ...], size: int = 13) -> np.ndarray:
    """Premultiplied sprite of key hints drawn as little keycaps."""
    f_key, f_label = _font(size, "Bold"), _font(size, "Medium")
    cap = int(size * 1.75)
    gap, inner = int(size * 1.4), int(size * 0.55)
    widths = [cap + inner + int(f_label.getlength(label)) for _, label in items]
    W, H = sum(widths) + gap * (len(items) - 1) + 4, cap + 4
    ss = 3
    mask = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(mask)
    fk, fl = _font(size * ss, "Bold"), _font(size * ss, "Medium")
    x = 2
    for (key, label), width in zip(items, widths):
        d.rounded_rectangle(
            [x * ss, 2 * ss, (x + cap) * ss, (2 + cap) * ss], radius=6 * ss, outline=150, width=int(1.4 * ss)
        )
        d.text(((x + cap / 2) * ss, (2 + cap / 2) * ss), key, font=fk, fill=255, anchor="mm")
        d.text(((x + cap + inner) * ss, (2 + cap / 2) * ss), label, font=fl, fill=185, anchor="lm")
        x += width + gap
    a = np.asarray(mask.resize((W, H), Image.LANCZOS), np.float32)
    return np.dstack([a, a, a, a]).astype(np.uint8)


# ---------------------------------------------------------------- HUD


class Hud:
    """The frosted card in the corner: total count, one glyph per hand, key hints."""

    CARD_W = 304

    def __init__(self):
        self.shown: int | None = None
        self.changed_at = -1e9

    def draw(self, frame: np.ndarray, hands: list[dict], now: float, details: bool = False, stats: str = "") -> None:
        total = sum(int(hd["up"].sum()) for hd in hands) if hands else None
        if total != self.shown:
            self.shown, self.changed_at = total, now
        age = now - self.changed_at

        x, y, w = 24, 24, self.CARD_W
        glyph_h = 104
        gy = y + 186  # top of the hand glyphs
        h = 186 + glyph_h + (34 if not hands else 22) + (44 if details and hands else 0)
        glass(frame, x, y, w, h, r=30, tint=0.62)
        cx = x + w / 2

        # Numeral: a damped spring 'pop' whenever the count changes, plus a fading ring.
        ncy = y + 110
        if total is not None and age < 0.6:
            t = age / 0.6
            card = frame[y : y + h, x : x + w]
            ring = card.copy()
            cv2.circle(ring, (int(cx - x), int(ncy - y)), int(46 + 64 * (1 - (1 - t) ** 3)), WHITE, 2, cv2.LINE_AA)
            card[:] = cv2.addWeighted(ring, 0.45 * (1 - t), card, 1 - 0.45 * (1 - t), 0)

        blend(frame, text_sprite("FINGERS UP", 12, "Bold", alpha=0.55, tracking=0.32), x + 26, y + 24)
        if total is None:
            blend_center(frame, text_sprite("–", 120, "Heavy", alpha=0.35), cx, ncy)
        else:
            pop = 1 + 0.22 * math.exp(-age / 0.11) * math.cos(age * 26)
            blend_center(frame, text_sprite(str(total), 150, "Black", gradient=True, shadow=True), cx, ncy, scale=pop)

        if not hands:
            sprite, _ = hand_glyph(np.zeros(5), glyph_h)
            blend_center(frame, sprite, cx, gy + glyph_h / 2, opacity=0.4)
            blend_center(frame, text_sprite("hold up a hand", 15, "Semibold", alpha=0.6), cx, gy + glyph_h + 18)
        else:
            slots = [cx] if len(hands) == 1 else [x + w * 0.27, x + w * 0.73]
            for slot, hd in zip(slots, hands):
                sprite, (px, py) = hand_glyph(hd["probs"], glyph_h, mirror=hd["mirror"])
                left, top = round(slot - sprite.shape[1] / 2), round(gy + glyph_h / 2 - sprite.shape[0] / 2)
                blend(frame, sprite, left, top)
                blend_center(frame, text_sprite(str(int(hd["up"].sum())), 24, "Heavy", alpha=0.92), left + px, top + py)
                if details:
                    for i, p in enumerate(hd["probs"]):
                        bx, base = int(slot + (i - 2) * 13), gy + glyph_h + 52
                        cv2.rectangle(frame, (bx - 4, base - 28), (bx + 4, base), (70, 64, 60), -1)
                        cv2.rectangle(frame, (bx - 4, base - int(28 * p)), (bx + 4, base), FINGER_BGR[i], -1)
            if len(hands) == 2:
                blend_center(frame, text_sprite("+", 22, "Heavy", alpha=0.45), cx, gy + glyph_h * 0.78)

        self._live_chip(frame, now, stats if details else "")
        H = frame.shape[0]
        hints = keycaps((("Q", "quit"), ("C", "camera"), ("M", "mirror"), ("D", "details")))
        glass(frame, 24, H - 24 - 46, hints.shape[1] + 32, 46, r=23, tint=0.55)
        blend(frame, hints, 24 + 16, H - 24 - 46 + (46 - hints.shape[0]) // 2)

    def _live_chip(self, frame, now, stats: str) -> None:
        label = text_sprite(f"LIVE{'  ·  ' + stats if stats else ''}", 12, "Bold", alpha=0.85, tracking=0.12)
        w, h = label.shape[1] + 52, 36
        x, y = frame.shape[1] - 24 - w, 24
        glass(frame, x, y, w, h, r=18, tint=0.55)
        # A sonar ping around the red dot.
        phase = (now * 1.1) % 1.0
        chip = frame[y : y + h, x : x + w]
        ping = chip.copy()
        cv2.circle(ping, (20 * 16, h * 8), int((5 + 10 * phase) * 16), (96, 86, 255), -1, cv2.LINE_AA, 4)
        chip[:] = cv2.addWeighted(ping, 0.5 * (1 - phase), chip, 1 - 0.5 * (1 - phase), 0)
        cv2.circle(chip, (20 * 16, h * 8), 5 * 16, (96, 86, 255), -1, cv2.LINE_AA, 4)
        blend(frame, label, x + 34, y + (h - label.shape[0]) // 2)

"""
utils.py — Production-grade Visual Processing & Helper Utilities
High-performance frame annotation with semi-transparent overlays,
rolling FPS calculation, zone breach detection, and screenshot saving.
"""

import cv2
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Colour palette: BGR format, one distinct hue per common COCO class
# ---------------------------------------------------------------------------
_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person":       (255, 128,   0),
    "car":          (  0, 165, 255),
    "truck":        (  0, 200, 200),
    "bus":          ( 30, 220, 100),
    "motorcycle":   (200,  80, 255),
    "bicycle":      (255, 200,  50),
    "bottle":       ( 80, 220,  80),
    "cup":          (100, 255, 180),
    "laptop":       (220,  80, 220),
    "cell phone":   (255,  80, 150),
    "chair":        ( 50, 180, 255),
    "dog":          (  0, 120, 255),
    "cat":          (180, 100, 255),
    "backpack":     (255, 160,  60),
    "handbag":      (160, 255, 100),
    "umbrella":     (  0, 220, 255),
}
_DEFAULT_COLOR: Tuple[int, int, int] = (0, 210, 210)

# ---------------------------------------------------------------------------
# FPS — rolling average over last N frames
# ---------------------------------------------------------------------------
_fps_times: deque = deque(maxlen=30)


def calculate_fps() -> int:
    """
    Push current timestamp and return rolling-average FPS over last 30 frames.
    Call once per rendered frame.
    """
    _fps_times.append(time.perf_counter())
    if len(_fps_times) < 2:
        return 0
    elapsed = _fps_times[-1] - _fps_times[0]
    return int((len(_fps_times) - 1) / elapsed) if elapsed > 0 else 0


def reset_fps() -> None:
    """Clear FPS history (call when stream starts)."""
    _fps_times.clear()


# ---------------------------------------------------------------------------
# Zone breach
# ---------------------------------------------------------------------------
def check_zone_breach(
    box: List[int],
    zone: Optional[Dict[str, float]],
    frame_w: int,
    frame_h: int,
) -> bool:
    """
    Return True when the centre of `box` lies inside the normalised `zone`.
    Zone dict keys: xmin, ymin, xmax, ymax — all in [0, 1].
    """
    if not zone:
        return False
    zx0 = int(zone["xmin"] * frame_w)
    zy0 = int(zone["ymin"] * frame_h)
    zx1 = int(zone["xmax"] * frame_w)
    zy1 = int(zone["ymax"] * frame_h)
    xmin, ymin, xmax, ymax = box
    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    return zx0 <= cx <= zx1 and zy0 <= cy <= zy1


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------
def save_screenshot(frame: np.ndarray, screenshots_dir: str = "screenshots") -> str:
    """Save BGR frame as JPEG; return the full file path."""
    os.makedirs(screenshots_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(screenshots_dir, f"detection_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------
def _color_for(cls_name: str) -> Tuple[int, int, int]:
    return _CLASS_COLORS.get(cls_name.lower(), _DEFAULT_COLOR)


def draw_annotations(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    tracker_active: bool,
    active_counts: Dict[str, int],
    zone: Optional[Dict[str, float]],
    zone_breached: bool,
    fps: int = 0,
) -> np.ndarray:
    """
    Composites all HUD layers onto a copy of `frame` and returns it.

    Layers (bottom-up):
      1. Restricted zone translucent fill + border
      2. Bounding boxes + label chips
      3. Active-object count panel (top-left)
      4. FPS badge (top-right)
      5. Alert banner (bottom-centre, only when zone is breached)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # ── 1. Restricted zone ────────────────────────────────────────────
    if zone:
        zx0 = int(zone["xmin"] * w);  zy0 = int(zone["ymin"] * h)
        zx1 = int(zone["xmax"] * w);  zy1 = int(zone["ymax"] * h)
        z_color = (0, 0, 230) if zone_breached else (0, 140, 255)
        z_thick = 3 if zone_breached else 2

        if zone_breached:
            overlay = out.copy()
            cv2.rectangle(overlay, (zx0, zy0), (zx1, zy1), z_color, -1)
            cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

        cv2.rectangle(out, (zx0, zy0), (zx1, zy1), z_color, z_thick)
        z_label = "⚠ ZONE BREACHED!" if zone_breached else "RESTRICTED AREA"
        cv2.putText(out, z_label, (zx0 + 6, zy0 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, z_color, 1, cv2.LINE_AA)

    # ── 2. Bounding boxes & label chips ───────────────────────────────
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        cls_name = det["class_name"]
        conf     = det["confidence"]
        tid      = det.get("track_id")
        color    = _color_for(cls_name)

        # Box
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)

        # Label
        label = (
            f"{cls_name.capitalize()} #{tid} {int(conf*100)}%"
            if (tracker_active and tid is not None)
            else f"{cls_name.capitalize()} {int(conf*100)}%"
        )
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        ty = max(ymin - 5, lh + 5)
        cv2.rectangle(out, (xmin, ty - lh - 4), (xmin + lw + 8, ty + bl), color, -1)
        cv2.putText(out, label, (xmin + 4, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 3. Active-object count panel (top-left) ───────────────────────
    if active_counts:
        n_rows = len(active_counts)
        panel_h = 28 + n_rows * 22
        panel_w = 190
        ov = out.copy()
        cv2.rectangle(ov, (8, 8), (8 + panel_w, 8 + panel_h), (8, 12, 22), -1)
        cv2.addWeighted(ov, 0.70, out, 0.30, 0, out)
        cv2.rectangle(out, (8, 8), (8 + panel_w, 8 + panel_h), (0, 210, 210), 1)

        cv2.putText(out, "LIVE OBJECTS", (18, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 210, 210), 1, cv2.LINE_AA)
        for i, (cls, cnt) in enumerate(active_counts.items()):
            cv2.putText(out, f"  {cls.capitalize()}: {cnt}",
                        (18, 26 + (i + 1) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # ── 4. FPS badge (top-right) ──────────────────────────────────────
    fps_label = f"FPS  {fps}"
    (fw, fh), fbl = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    fx0 = w - fw - 18
    ov = out.copy()
    cv2.rectangle(ov, (fx0 - 4, 8), (w - 8, 8 + fh + 10), (8, 12, 22), -1)
    cv2.addWeighted(ov, 0.70, out, 0.30, 0, out)
    fps_color = (80, 255, 80) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 60, 255)
    cv2.putText(out, fps_label, (fx0, 8 + fh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, fps_color, 1, cv2.LINE_AA)

    # ── 5. Alert banner (bottom-centre) ───────────────────────────────
    if zone_breached:
        banner = "! ZONE BREACH DETECTED !"
        (bw, bh), bbl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        bx = max((w - bw) // 2 - 12, 0)
        by = h - 50
        ov = out.copy()
        cv2.rectangle(ov, (bx, by), (bx + bw + 24, by + bh + 18), (0, 0, 200), -1)
        cv2.addWeighted(ov, 0.80, out, 0.20, 0, out)
        cv2.putText(out, banner, (bx + 12, by + bh + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    return out

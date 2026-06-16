"""
heatmap.py — Production Movement Heatmap Generator
Accumulates Gaussian-blurred coordinate points on a 2D float grid.
Uses a threshold mask so only active zones glow — background stays clean.
Supports configurable temporal decay for fade-out effects.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional


class MovementHeatmap:
    """
    Builds a live traffic heatmap by accumulating object-centre coordinates
    onto a float32 grid and rendering it as a JET-colourmap overlay.

    Colour coding:
        Blue  → low activity
        Green → medium activity
        Red   → high-traffic hotspot
    """

    BLUR_KERNEL = (31, 31)   # Gaussian kernel — larger = smoother glow
    RADIUS      = 20          # Gaussian circle radius per point
    INTENSITY   = 2.0         # Accumulation weight per point

    def __init__(self, decay: float = 1.0) -> None:
        """
        Args:
            decay: Multiplier applied to the grid each frame.
                   1.0  → infinite accumulation (persistent trails)
                   0.97 → gentle fade-out (~30 frame half-life)
        """
        self.decay = decay
        self._grid: Optional[np.ndarray] = None   # float32 h×w
        self._h: int = 0
        self._w: int = 0

    # ------------------------------------------------------------------
    def _ensure_grid(self, h: int, w: int) -> None:
        """Lazy-initialise or reinitialise grid when resolution changes."""
        if self._grid is None or self._h != h or self._w != w:
            self._grid = np.zeros((h, w), dtype=np.float32)
            self._h, self._w = h, w

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Accumulate centre-points from `detections` onto the heat grid.
        Call once per frame after tracker.update().
        """
        self._ensure_grid(frame_h, frame_w)

        # Temporal decay (fade old trails)
        if self.decay < 1.0:
            self._grid *= self.decay

        for det in detections:
            cx, cy = det["center"]
            if not (0 <= cx < frame_w and 0 <= cy < frame_h):
                continue
            # Draw a filled circle on a temporary mask, then Gaussian-blur it
            mask = np.zeros((frame_h, frame_w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), self.RADIUS, self.INTENSITY, -1)
            mask = cv2.GaussianBlur(mask, self.BLUR_KERNEL, 0)
            self._grid = cv2.add(self._grid, mask)

    # ------------------------------------------------------------------
    def get_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the JET colourmap onto `frame` and return the result.
        Only active (above-threshold) regions are coloured;
        the background remains from the original frame.
        """
        if self._grid is None or np.max(self._grid) == 0:
            return frame.copy()

        h, w = frame.shape[:2]
        self._ensure_grid(h, w)

        # Normalise → uint8
        norm = cv2.normalize(self._grid, None, 0, 255,
                             cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Threshold mask: only paint pixels that have meaningful activity
        _, mask = cv2.threshold(norm, 8, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Background: keep original where activity is zero
        background = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Foreground: blend coloured heatmap 55 % / original 45 %
        blended   = cv2.addWeighted(frame, 0.45, coloured, 0.55, 0)
        foreground = cv2.bitwise_and(blended, blended, mask=mask)

        return cv2.add(background, foreground)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Wipe the accumulation grid."""
        self._grid = None

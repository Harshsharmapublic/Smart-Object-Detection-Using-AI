"""
line_counter.py — Production Virtual Line Crossing Counter
Tracks Entry / Exit events for every tracked object using two-frame
vector analysis. Crossing state is persisted per track ID to prevent
double-counting on jitter at the line boundary.
"""

import cv2
import numpy as np
from typing import Any, Dict, List

import logging
logger = logging.getLogger(__name__)


class LineCrossingCounter:
    """
    Detects when tracked objects cross a configurable horizontal line.

      - Moving downward  (prev_cy < line_y  →  cy >= line_y) = ENTRY
      - Moving upward    (prev_cy > line_y  →  cy <= line_y) = EXIT

    Each track_id can only trigger one state change per direction to
    prevent jitter-induced duplicate counts.
    """

    def __init__(self, line_y_pct: float = 0.5) -> None:
        self.line_y_pct = line_y_pct      # 0.0–1.0 fractional frame height
        self.entry_count: int = 0
        self.exit_count: int = 0
        # Maps track_id → last crossing direction ('entry' | 'exit')
        self._last_direction: Dict[int, str] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Dict[str, Any]],
        paths: Dict[int, Any],          # id → deque[(cx, cy)]
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Evaluate every tracked object's last two path points against the line.
        Call once per rendered frame after tracker.update().
        """
        line_y = int(self.line_y_pct * frame_h)

        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue

            path = paths.get(tid)
            if path is None or len(path) < 2:
                continue

            path_list = list(path)
            prev_cy = path_list[-2][1]
            curr_cy = path_list[-1][1]

            if prev_cy < line_y and curr_cy >= line_y:
                # Downward crossing → Entry
                if self._last_direction.get(tid) != "entry":
                    self._last_direction[tid] = "entry"
                    self.entry_count += 1
                    logger.debug(f"ENTRY  track_id={tid}")

            elif prev_cy > line_y and curr_cy <= line_y:
                # Upward crossing → Exit
                if self._last_direction.get(tid) != "exit":
                    self._last_direction[tid] = "exit"
                    self.exit_count += 1
                    logger.debug(f"EXIT   track_id={tid}")

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay the virtual counting line, directional arrows,
        and Entry/Exit counter chip onto a copy of `frame`.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        line_y = int(self.line_y_pct * h)

        # ── Line ──────────────────────────────────────────────────────
        LINE_COLOR = (0, 200, 255)  # Amber
        cv2.line(out, (0, line_y), (w, line_y), LINE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(out, "COUNTING LINE", (12, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, LINE_COLOR, 1, cv2.LINE_AA)

        # ── Directional arrows ────────────────────────────────────────
        mid_x = w // 2
        # Entry (↓) green
        cv2.arrowedLine(out, (mid_x - 60, line_y - 22),
                         (mid_x - 60, line_y + 22),
                         (60, 220, 60), 2, tipLength=0.35)
        cv2.putText(out, "ENTRY", (mid_x - 58, line_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 220, 60), 1, cv2.LINE_AA)
        # Exit (↑) orange
        cv2.arrowedLine(out, (mid_x + 60, line_y + 22),
                         (mid_x + 60, line_y - 22),
                         (0, 165, 255), 2, tipLength=0.35)
        cv2.putText(out, "EXIT", (mid_x + 47, line_y - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)

        # ── Counter chip (top-right) ──────────────────────────────────
        chip_x, chip_y = w - 185, 55
        ov = out.copy()
        cv2.rectangle(ov, (chip_x, chip_y),
                      (chip_x + 175, chip_y + 55), (8, 12, 22), -1)
        cv2.addWeighted(ov, 0.72, out, 0.28, 0, out)
        cv2.rectangle(out, (chip_x, chip_y),
                      (chip_x + 175, chip_y + 55), LINE_COLOR, 1)
        cv2.putText(out, f"Entry : {self.entry_count}",
                    (chip_x + 10, chip_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 220, 60), 1, cv2.LINE_AA)
        cv2.putText(out, f"Exit  : {self.exit_count}",
                    (chip_x + 10, chip_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 165, 255), 1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.entry_count = 0
        self.exit_count = 0
        self._last_direction.clear()

"""
tracker.py — Production-grade Multi-Object Tracking State Manager
Wraps the detector's track() output, maintains per-ID path history
(bounded to MAX_PATH_LEN frames), and prunes stale tracks automatically.
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any

MAX_PATH_LEN = 30       # Max coordinate history per track ID
STALE_AFTER_FRAMES = 60 # Frames of absence before path is purged

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Stateful multi-object tracking coordinator.

    Wraps the YOLODetector and adds:
    - Per-ID coordinate path history (for heatmap & line crossing)
    - Active / historical category counts
    - Automatic stale-path pruning to prevent memory leak
    """

    def __init__(self, detector) -> None:
        self.detector = detector
        self.active_tracks: Dict[int, str] = {}            # id → class_name (current frame)
        self.historical_ids: Dict[str, Set[int]] = defaultdict(set)  # class → set of unique ids
        self.paths: Dict[int, deque] = {}                  # id → deque[(cx, cy)]
        self._inactive_frames: Dict[int, int] = {}         # id → consecutive absent frames

    # ------------------------------------------------------------------
    def update(
        self,
        frame,
        conf_threshold: float = 0.25,
        classes: Optional[List[int]] = None,
        use_tracking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run detection/tracking on a frame and update internal state.

        Returns list of detection dicts (same schema as detector outputs).
        """
        if use_tracking:
            detections = self.detector.track(frame, conf_threshold, classes)
        else:
            detections = self.detector.detect(frame, conf_threshold, classes)
            # Assign synthetic IDs so downstream code remains consistent
            for i, det in enumerate(detections):
                det["track_id"] = i  # ephemeral, resets each frame

        active_ids: List[int] = []
        current_active: Dict[int, str] = {}

        for det in detections:
            tid = det.get("track_id")
            cls_name = det["class_name"]
            cx, cy = det["center"]

            if tid is not None:
                current_active[tid] = cls_name
                active_ids.append(tid)
                self.historical_ids[cls_name].add(tid)

                # Update path history
                if tid not in self.paths:
                    self.paths[tid] = deque(maxlen=MAX_PATH_LEN)
                self.paths[tid].append((cx, cy))
                self._inactive_frames[tid] = 0  # reset absence counter

        self.active_tracks = current_active
        self._prune_stale(set(active_ids))
        return detections

    # ------------------------------------------------------------------
    def _prune_stale(self, active_ids: Set[int]) -> None:
        """Remove paths of IDs not seen for STALE_AFTER_FRAMES consecutive frames."""
        to_delete = []
        for tid in list(self.paths.keys()):
            if tid not in active_ids:
                self._inactive_frames[tid] = self._inactive_frames.get(tid, 0) + 1
                if self._inactive_frames[tid] > STALE_AFTER_FRAMES:
                    to_delete.append(tid)
        for tid in to_delete:
            self.paths.pop(tid, None)
            self._inactive_frames.pop(tid, None)

    # ------------------------------------------------------------------
    def get_active_counts(self) -> Dict[str, int]:
        """Count of each class currently visible in the frame."""
        counts: Dict[str, int] = {}
        for cls_name in self.active_tracks.values():
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts

    def get_total_unique_counts(self) -> Dict[str, int]:
        """Total unique objects (by ID) seen per class since last reset."""
        return {cls: len(ids) for cls, ids in self.historical_ids.items()}

    def get_total_objects_ever(self) -> int:
        """Sum of all unique tracked IDs across all classes."""
        return sum(len(ids) for ids in self.historical_ids.values())

    def get_path(self, track_id: int) -> List[Tuple[int, int]]:
        """Return coordinate history for a given track ID."""
        return list(self.paths.get(track_id, []))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Full state wipe — counters, paths, and track records."""
        self.active_tracks.clear()
        self.historical_ids.clear()
        self.paths.clear()
        self._inactive_frames.clear()
        logger.info("ObjectTracker state reset.")

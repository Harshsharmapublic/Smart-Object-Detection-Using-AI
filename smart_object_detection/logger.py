"""
logger.py — Production-grade Thread-safe Detection & Alert Logger
Writes daily-rolling CSV logs. Uses a seen-ID set to prevent flooding
the CSV with every single frame for the same tracked object.
Alert logging is always written (violations & line crossings).
"""

import csv
import logging
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DETECTION_COLS = ["Timestamp", "Object", "Confidence", "Tracking_ID"]
ALERT_COLS = ["Timestamp", "Event", "Tracking_ID", "Confidence"]


class DetectionLogger:
    """
    Thread-safe, daily-rolling CSV logger.

    Detections are de-duplicated per tracking_id (each new object is
    written once when first seen).  Alerts (zone breach, line crossing)
    are always appended.
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._seen_ids: set = set()  # track_ids already written this session

        date_str = datetime.now().strftime("%Y%m%d")
        self.detection_log_path = os.path.join(logs_dir, f"detections_{date_str}.csv")
        self.alert_log_path = os.path.join(logs_dir, f"alerts_{date_str}.csv")
        self._init_files()

    # ------------------------------------------------------------------
    def _init_files(self) -> None:
        """Create CSV files with headers if they don't already exist."""
        with self._lock:
            if not os.path.exists(self.detection_log_path):
                self._write_header(self.detection_log_path, DETECTION_COLS)
            if not os.path.exists(self.alert_log_path):
                self._write_header(self.alert_log_path, ALERT_COLS)

    def _write_header(self, path: str, cols: List[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

    # ------------------------------------------------------------------
    def log_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Batch-log detections, skipping already-seen tracking IDs."""
        if not detections:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        with self._lock:
            for det in detections:
                tid = det.get("track_id")
                # De-duplicate: skip if this ID was already logged
                if tid is not None and tid in self._seen_ids:
                    continue
                if tid is not None:
                    self._seen_ids.add(tid)
                rows.append([ts, det["class_name"], f"{det['confidence']:.2f}", tid if tid is not None else "N/A"])
            if rows:
                with open(self.detection_log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

    # ------------------------------------------------------------------
    def log_alert(self, event: str, tracking_id: Optional[int], confidence: float) -> None:
        """Always append an alert row (no de-duplication)."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tid_val = tracking_id if tracking_id is not None else "N/A"
        with self._lock:
            with open(self.alert_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ts, event, tid_val, f"{confidence:.2f}"])

    # ------------------------------------------------------------------
    def get_recent_detections(self, limit: int = 200) -> pd.DataFrame:
        """Return last `limit` detection rows, most-recent first."""
        return self._read_tail(self.detection_log_path, DETECTION_COLS, limit)

    def get_recent_alerts(self, limit: int = 100) -> pd.DataFrame:
        """Return last `limit` alert rows, most-recent first."""
        return self._read_tail(self.alert_log_path, ALERT_COLS, limit)

    def _read_tail(self, path: str, cols: List[str], limit: int) -> pd.DataFrame:
        with self._lock:
            try:
                if os.path.exists(path) and os.path.getsize(path) > 10:
                    df = pd.read_csv(path)
                    return df.tail(limit).iloc[::-1].reset_index(drop=True)
            except Exception as exc:
                logger.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------
    def clear_logs(self) -> None:
        """Wipe both CSVs and reset the seen-ID set."""
        with self._lock:
            self._seen_ids.clear()
            self._write_header(self.detection_log_path, DETECTION_COLS)
            self._write_header(self.alert_log_path, ALERT_COLS)
        logger.info("Logs cleared.")

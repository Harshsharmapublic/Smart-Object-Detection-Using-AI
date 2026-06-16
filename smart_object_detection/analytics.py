"""
analytics.py — Production Surveillance Analytics & Report Engine
Aggregates detection logs into statistical summaries and exports
multi-worksheet Excel workbooks using Pandas + openpyxl.
"""

import io
import logging
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class SurveillanceAnalytics:
    """
    Reads rolling CSV logs produced by DetectionLogger and provides:
      - Summary statistics (distribution, most-detected, hourly freq.)
      - Binary Excel workbook for Streamlit download_button
      - Raw CSV string for Streamlit download_button
    """

    def __init__(self, logs_dir: str = "logs", reports_dir: str = "reports") -> None:
        self.logs_dir = logs_dir
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _safe_read(self, path: str) -> pd.DataFrame:
        """Read a CSV safely; return empty DataFrame on any error."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 10:
                return pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    def get_summary_statistics(
        self,
        detection_log_path: str,
        alert_log_path: str,
        entry_count: int = 0,
        exit_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Return a dict of analytical metrics derived from the current log files.
        Safe to call even when logs are empty.
        """
        stats: Dict[str, Any] = {
            "total_detections": 0,
            "most_detected": "—",
            "most_detected_count": 0,
            "total_alerts": 0,
            "entry_count": entry_count,
            "exit_count": exit_count,
            "distribution": {},
            "frequency_by_hour": {},
        }

        df = self._safe_read(detection_log_path)
        if not df.empty and "Object" in df.columns:
            stats["total_detections"] = len(df)
            vc = df["Object"].value_counts()
            stats["distribution"] = vc.to_dict()
            if not vc.empty:
                stats["most_detected"] = str(vc.index[0]).capitalize()
                stats["most_detected_count"] = int(vc.iloc[0])

            # Hourly frequency
            if "Timestamp" in df.columns:
                try:
                    df["_hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
                    stats["frequency_by_hour"] = (
                        df["_hour"].value_counts().sort_index().to_dict()
                    )
                except Exception:
                    pass

        df_alerts = self._safe_read(alert_log_path)
        if not df_alerts.empty:
            stats["total_alerts"] = len(df_alerts)

        return stats

    # ------------------------------------------------------------------
    def generate_excel_report(
        self,
        detection_log_path: str,
        alert_log_path: str,
        stats: Dict[str, Any],
    ) -> bytes:
        """
        Build and return a binary Excel workbook with three worksheets:
          1. Executive Summary — key KPIs
          2. Detection Log     — full raw CSV
          3. Alerts Log        — zone & crossing events
        """
        df_dets   = self._safe_read(detection_log_path)
        df_alerts = self._safe_read(alert_log_path)

        summary_rows = {
            "Metric": [
                "Total Objects Logged",
                "Most Detected Class",
                "Most Detected Count",
                "Zone Violations",
                "Line Entry Count",
                "Line Exit Count",
                "Report Generated",
            ],
            "Value": [
                stats["total_detections"],
                stats["most_detected"],
                stats["most_detected_count"],
                stats["total_alerts"],
                stats["entry_count"],
                stats["exit_count"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ],
        }
        df_summary = pd.DataFrame(summary_rows)

        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_summary.to_excel(writer, sheet_name="Executive Summary", index=False)
                if not df_dets.empty:
                    df_dets.to_excel(writer, sheet_name="Detection Log", index=False)
                if not df_alerts.empty:
                    df_alerts.to_excel(writer, sheet_name="Alerts Log", index=False)
        except Exception as exc:
            logger.error(f"Excel generation failed: {exc}")
        return buf.getvalue()

    # ------------------------------------------------------------------
    def generate_csv_report(self, detection_log_path: str) -> str:
        """Return the raw detection CSV as a UTF-8 string."""
        try:
            if os.path.exists(detection_log_path) and os.path.getsize(detection_log_path) > 10:
                with open(detection_log_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as exc:
            logger.warning(f"CSV read failed: {exc}")
        return "Timestamp,Object,Confidence,Tracking_ID\n"

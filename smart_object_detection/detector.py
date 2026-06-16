"""
detector.py — Production-grade YOLOv8 Inference Engine
Wraps Ultralytics YOLO with robust model management, consistent output schema,
and full error handling. Every detection dict carries: box, confidence,
class_id, class_name, and center (cx, cy).
"""

import os
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


class YOLODetector:
    """
    High-performance YOLOv8 object detector with lazy model loading,
    local weight caching, and graceful fallback on failure.
    """

    def __init__(self, model_name: str = "yolov8n.pt", models_dir: str = "models") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_name = model_name
        self.model = None
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load model from local cache or download from Ultralytics hub."""
        from ultralytics import YOLO

        local_path = os.path.join(self.models_dir, model_name)
        try:
            self.logger.info(f"Loading model → {local_path}")
            self.model = YOLO(local_path)
            self.logger.info("Model loaded successfully.")
        except Exception as primary_err:
            self.logger.warning(f"Local load failed ({primary_err}). Downloading {model_name}…")
            try:
                self.model = YOLO(model_name)
                # Persist weight to models_dir for future runs
                if os.path.exists(model_name) and not os.path.exists(local_path):
                    os.rename(model_name, local_path)
                    self.model = YOLO(local_path)
                self.logger.info(f"Download complete. Cached to {local_path}.")
            except Exception as fallback_err:
                self.logger.critical(f"Cannot load model: {fallback_err}")
                raise RuntimeError(f"Failed to load YOLOv8 model '{model_name}'.") from fallback_err

    # ------------------------------------------------------------------
    def get_class_names(self) -> Dict[int, str]:
        """Return {class_id: class_name} mapping from the loaded model."""
        return dict(self.model.names) if self.model else {}

    # ------------------------------------------------------------------
    def _parse_boxes(self, result, track_ids=None) -> List[Dict[str, Any]]:
        """
        Shared parser for both predict() and track() results.
        Returns a list of dicts with: box, confidence, class_id, class_name, center, track_id.
        """
        detections: List[Dict[str, Any]] = []
        if result is None or result.boxes is None:
            return detections

        boxes = result.boxes
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            # Tracking ID (None when running plain detect)
            if track_ids is not None and box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = None

            detections.append({
                "box": [xmin, ymin, xmax, ymax],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (cx, cy),
                "track_id": track_id,
            })
        return detections

    # ------------------------------------------------------------------
    def detect(
        self,
        frame,
        conf_threshold: float = 0.25,
        classes: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run stateless single-frame inference (no tracking).

        Returns list of detection dicts — each has a `track_id` of None.
        """
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                classes=classes if classes else None,
                verbose=False,
                stream=False,
            )
            return self._parse_boxes(results[0] if results else None)
        except Exception as exc:
            self.logger.error(f"Inference error: {exc}")
            return []

    # ------------------------------------------------------------------
    def track(
        self,
        frame,
        conf_threshold: float = 0.25,
        classes: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run single-frame tracking inference using YOLOv8's built-in ByteTrack.

        Returns list of detection dicts — each has a `track_id` integer.
        Falls back to stateless detect() on tracker failure.
        """
        if self.model is None:
            return []
        try:
            results = self.model.track(
                source=frame,
                conf=conf_threshold,
                classes=classes if classes else None,
                persist=True,
                verbose=False,
                stream=False,
            )
            result = results[0] if results else None
            return self._parse_boxes(result, track_ids=True)
        except Exception as exc:
            self.logger.warning(f"Tracker error ({exc}). Falling back to detect().")
            return self.detect(frame, conf_threshold, classes)

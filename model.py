"""
Label Studio ML backend for **Ultralytics YOLO‑pose** that feeds filtered
key-point predictions back to the UI and includes thorough logging and
task‑data verification.

Constants below are your only settings—no env vars.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MODEL_PATH = "yolo11n-pose.pt"   # YOLO checkpoint
BOX_THR = 0.15                    # min box confidence
KP_THR = 0.05                     # min key-point confidence
IMGSZ = (192, 128)                # (height, width)
DEVICE = "cpu"                    # "cpu", "0", "cuda:1", …
IMAGE_FIELD = "img"               # key in task["data"] for image URL

# Label names as declared in Label Studio config
RECT_LABELS = {0: ("bbox_face", "face"),    # class 0 → (rect tag name, label)
               1: ("bbox_tube", "juice_tube")}  # class 1 → (rect tag name, label)
KPT_LABELS = {0: ("kp_face", ["left_pupil", "right_pupil", "nose_bridge"]),
              1: ("kp_tube", ["spout_top", "spout_bottom"])}
# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewModel(LabelStudioMLBase):
    """Ultralytics YOLO‑pose wrapper with filtering and detailed logging."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info("Initializing YOLO‑pose backend")
        try:
            self.model = YOLO(MODEL_PATH)
            logger.info("Loaded model from %s", MODEL_PATH)
        except Exception as exc:
            logger.exception("Failed to load model from %s: %s", MODEL_PATH, exc)
            raise

        self.box_thr = BOX_THR
        self.kp_thr = KP_THR
        self.imgsz = IMGSZ
        self.device = DEVICE

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------
    def predict(self, tasks: List[dict], **kwargs):  # noqa: D401
        """Return filtered YOLO‑pose predictions for Label Studio tasks."""
        predictions = []

        for task in tasks:
            task_id = task.get("id")
            data = task.get("data", {})
            logger.debug("Task %s data fields: %s", task_id, list(data.keys()))

            # Determine image URL
            img_url = data.get(IMAGE_FIELD)
            if not img_url:
                if data:
                    img_url = next(iter(data.values()))
                    logger.warning("Task %s: '%s' not found, using fallback URL", task_id, IMAGE_FIELD)
                else:
                    logger.error("Task %s has no data fields", task_id)
                    predictions.append({"result": [], "score": 0.0})
                    continue

            logger.debug("Task %s → processing URL: %s", task_id, img_url)
            img_path_str = self.get_local_path(img_url)
            img_path = Path(img_path_str)
            logger.debug("Task %s → local path: %s", task_id, img_path)

            if not img_path.exists():
                logger.error("Task %s: file does not exist: %s", task_id, img_path)
                predictions.append({"result": [], "score": 0.0})
                continue

            try:
                res = self.model.predict(
                    source=str(img_path),
                    imgsz=self.imgsz,
                    device=self.device,
                    conf=self.box_thr,
                    verbose=False,
                )[0]
                logger.debug("Task %s: raw detections = %d", task_id, len(res.boxes))
            except Exception as exc:
                logger.exception("Task %s: YOLO inference error: %s", task_id, exc)
                predictions.append({"result": [], "score": 0.0})
                continue

            h_orig, w_orig = res.orig_shape
            task_results = []

            for det_idx, (box, det_conf, cls) in enumerate(zip(res.boxes.xywh, res.boxes.conf, res.boxes.cls)):
                cls = int(cls)
                logger.debug("Task %s: det %d cls=%d conf=%.3f", task_id, det_idx, cls, det_conf)
                if det_conf < self.box_thr:
                    logger.debug("Task %s: skip box %d below thr", task_id, det_idx)
                    continue

                rect_tag, rect_label = RECT_LABELS.get(cls, (None, None))
                kp_tag, kp_names = KPT_LABELS.get(cls, (None, []))
                if rect_tag is None or kp_tag is None:
                    logger.warning("Task %s: unknown class %d", task_id, cls)
                    continue

                x_c, y_c, bw, bh = box.tolist()
                tlx_pct = (x_c - bw / 2) / w_orig * 100
                tly_pct = (y_c - bh / 2) / h_orig * 100
                w_pct = bw / w_orig * 100
                h_pct = bh / h_orig * 100

                # Rectangle annotation
                task_results.append({
                    "id": str(uuid4()),
                    "from_name": rect_tag,
                    "to_name": IMAGE_FIELD,
                    "type": "rectanglelabels",
                    "value": {
                        "x": tlx_pct,
                        "y": tly_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rectanglelabels": [rect_label]
                    }
                })

                # Key-points for this class
                kp_xy = res.keypoints.xy[det_idx].cpu().numpy()
                kp_conf = res.keypoints.conf[det_idx].cpu().numpy()
                points = []
                for kp_idx, ((x, y), kconf) in enumerate(zip(kp_xy, kp_conf)):
                    label_name = kp_names[kp_idx] if kp_idx < len(kp_names) else None
                    logger.debug("Task %s: KP %d cls=%d conf=%.3f", task_id, kp_idx, cls, kconf)
                    if kconf < self.kp_thr or label_name is None:
                        continue
                    points.append({
                        "x": float(x / w_orig * 100),
                        "y": float(y / h_orig * 100),
                        "label": [label_name],
                        "confidence": float(kconf),
                    })

                if points:
                    task_results.append({
                        "id": str(uuid4()),
                        "from_name": kp_tag,
                        "to_name": IMAGE_FIELD,
                        "type": "keypointlabels",
                        "score": float(det_conf),
                        "value": {
                            "points": points,
                            "x": tlx_pct,
                            "y": tly_pct,
                            "width": w_pct,
                            "height": h_pct
                        },
                    })
                    logger.info("Task %s: kept cls=%d box %d with %d KPs", task_id, cls, det_idx, len(points))

            logger.debug("Task %s: total results = %d", task_id, len(task_results))
            predictions.append({
                "result": task_results,
                "score": float(np.mean([r.get("score", 0.0) for r in task_results]) if task_results else 0.0),
            })

        return predictions

    # ------------------------------------------------------------------
    # Training hook (unused)
    # ------------------------------------------------------------------
    def fit(self, **kwargs):  # noqa: D401
        return {}

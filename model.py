"""
Label Studio ML backend for **Ultralytics YOLO‑pose** that feeds filtered
key‑point predictions back to the UI and includes thorough logging and
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
MODEL_PATH = "yolo11n-pose.pt"  # YOLO checkpoint
BOX_THR = 0.15                  # min box confidence
KP_THR = 0.05                   # min key‑point confidence
IMGSZ = (192, 128)              # (height, width)
DEVICE = "cpu"                  # "cpu", "0", "cuda:1", …
IMAGE_FIELD = "image"           # preferred key in task["data"] for image URL

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

        # Key‑point labels mapping
        self.kpt_labels = kwargs.get("keypoint_labels")
        if self.kpt_labels is None:
            try:
                kpt_count = int(self.model.model.kpt_shape[0])  # e.g. [3,3]
                self.kpt_labels = [f"kp_{i}" for i in range(kpt_count)]
                logger.debug("Auto-generated %d key-point labels", kpt_count)
            except Exception as exc:
                logger.exception("Failed to derive key-point labels: %s", exc)
                self.kpt_labels = []
        logger.debug("Key‑point labels: %s", self.kpt_labels)

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
                    # fallback to first field value
                    img_url = next(iter(data.values()))
                    logger.warning("Task %s: '%s' not found, using fallback URL from data", task_id, IMAGE_FIELD)
                else:
                    logger.error("Task %s has no data fields", task_id)
                    predictions.append({"result": [], "score": 0.0})
                    continue

            logger.debug("Task %s → processing image URL: %s", task_id, img_url)
            img_path: Path = self.get_local_path(img_url)
            logger.debug("Task %s → local image path: %s", task_id, img_path)

            # Check file existence
            if not img_path.exists():
                logger.error("Task %s: image file does not exist at %s", task_id, img_path)
                predictions.append({"result": [], "score": 0.0})
                continue

            # YOLO forward pass
            try:
                res = self.model.predict(
                    source=str(img_path),
                    imgsz=self.imgsz,
                    device=self.device,
                    conf=self.box_thr,
                    verbose=False,
                )[0]
                logger.debug("Task %s: YOLO returned %d raw detections", task_id, len(res.boxes))
            except Exception as exc:
                logger.exception("Task %s: YOLO inference error: %s", task_id, exc)
                predictions.append({"result": [], "score": 0.0})
                continue

            h_orig, w_orig = res.orig_shape
            task_results = []

            # Process each detection
            for det_idx, (box, det_conf) in enumerate(zip(res.boxes.xywh, res.boxes.conf)):
                logger.debug("Task %s: detection %d with conf=%.3f", task_id, det_idx, det_conf)
                if det_conf < self.box_thr:
                    logger.debug("Task %s: skip box %d below threshold", task_id, det_idx)
                    continue

                x_c, y_c, bw, bh = box.tolist()
                tlx_pct = (x_c - bw / 2) / w_orig * 100
                tly_pct = (y_c - bh / 2) / h_orig * 100
                w_pct = bw / w_orig * 100
                h_pct = bh / h_orig * 100

                kp_xy = res.keypoints.xy[det_idx].cpu().numpy()
                kp_conf = res.keypoints.conf[det_idx].cpu().numpy()
                points = []

                for kp_idx, ((x, y), kconf) in enumerate(zip(kp_xy, kp_conf)):
                    logger.debug("Task %s: KP %d conf=%.3f", task_id, kp_idx, kconf)
                    if kconf < self.kp_thr:
                        logger.debug("Task %s: drop KP %d below threshold", task_id, kp_idx)
                        continue
                    points.append({
                        "x": float(x / w_orig * 100),
                        "y": float(y / h_orig * 100),
                        "label": [self.kpt_labels[kp_idx]] if kp_idx < len(self.kpt_labels) else [],
                        "confidence": float(kconf),
                    })

                if not points:
                    logger.debug("Task %s: no KPs above threshold for box %d", task_id, det_idx)
                    continue

                logger.info("Task %s: kept box %d with %d key-points", task_id, det_idx, len(points))
                task_results.append({
                    "id": str(uuid4()),
                    "from_name": "keypoints",  # must match LS labeling config
                    "to_name": "image",
                    "type": "keypointlabels",
                    "score": float(det_conf),
                    "value": {
                        "points": points,
                        "x": tlx_pct,
                        "y": tly_pct,
                        "width": w_pct,
                        "height": h_pct,
                    },
                })

            logger.debug("Task %s: final results count = %d", task_id, len(task_results))
            predictions.append({
                "result": task_results,
                "score": float(np.mean([r["score"] for r in task_results]) if task_results else 0.0),
            })

        return predictions

    # ------------------------------------------------------------------
    # Training hook (unused)
    # ------------------------------------------------------------------
    def fit(self, **kwargs):  # noqa: D401
        return {}

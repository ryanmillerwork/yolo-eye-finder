"""
Label Studio ML backend for **Ultralytics YOLO‑pose** that feeds filtered
key‑point predictions back to the UI and now includes robust logging and
error‑handling.

Constants below are your only settings—no env vars required.
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
IMAGE_FIELD = "image"           # key in task["data"] that holds the image URL

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # force DEBUG even if LS reset root logger


class NewModel(LabelStudioMLBase):
    """Ultralytics YOLO‑pose wrapper with filtering and verbose logging."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info("Loading YOLO model from %s", MODEL_PATH)
        self.model = YOLO(MODEL_PATH)
        self.box_thr = BOX_THR
        self.kp_thr = KP_THR
        self.imgsz = IMGSZ
        self.device = DEVICE

        # Map key‑point indices → labels; fall back to generic names
        self.kpt_labels = kwargs.get("keypoint_labels")
        if self.kpt_labels is None:
            kpt_count = int(self.model.model.kpt_shape[0])  # e.g. 3 in [3,3]
            self.kpt_labels = [f"kp_{i}" for i in range(kpt_count)]
        logger.debug("Key‑point labels: %s", self.kpt_labels)

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------
    def predict(self, tasks: List[dict], **kwargs):  # noqa: D401
        """Return filtered YOLO‑pose predictions for Label Studio *tasks*."""
        predictions = []

        for task in tasks:
            img_url = task["data"].get(IMAGE_FIELD)
            if not img_url:
                logger.error("Task %s missing '%s' field", task.get("id"), IMAGE_FIELD)
                predictions.append({"result": [], "score": 0.0})
                continue

            img_path: Path = self.get_local_path(img_url)
            logger.debug("Task %s → %s", task.get("id"), img_path)

            # ------------------------------ YOLO forward pass
            try:
                res = self.model.predict(
                    source=str(img_path),
                    imgsz=self.imgsz,
                    device=self.device,
                    conf=self.box_thr,
                    verbose=False,
                )[0]
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("YOLO inference failed on %s: %s", img_path, exc)
                predictions.append({"result": [], "score": 0.0})
                continue

            logger.debug("%d detections returned", len(res.boxes))

            h_orig, w_orig = res.orig_shape
            task_results = []

            for det_idx, (box, det_conf) in enumerate(zip(res.boxes.xywh, res.boxes.conf)):
                if det_conf < self.box_thr:
                    logger.debug("Skip box %d: conf=%.3f < %.3f", det_idx, det_conf, self.box_thr)
                    continue

                # Convert box (cx,cy,w,h) → LS % coordinates (top‑left + size)
                x_c, y_c, bw, bh = box.tolist()
                tlx_pct = (x_c - bw / 2) / w_orig * 100
                tly_pct = (y_c - bh / 2) / h_orig * 100
                w_pct = bw / w_orig * 100
                h_pct = bh / h_orig * 100

                # Filter key‑points
                kp_xy = res.keypoints.xy[det_idx].cpu().numpy()
                kp_conf = res.keypoints.conf[det_idx].cpu().numpy()

                points = []
                for kp_idx, ((x, y), kconf) in enumerate(zip(kp_xy, kp_conf)):
                    if kconf < self.kp_thr:
                        logger.debug("  • drop KP %d conf=%.3f", kp_idx, kconf)
                        continue
                    points.append({
                        "x": float(x / w_orig * 100),
                        "y": float(y / h_orig * 100),
                        "label": [self.kpt_labels[kp_idx]],
                        "confidence": float(kconf),
                    })

                if not points:
                    logger.debug("All key‑points dropped for box %d", det_idx)
                    continue

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

            logger.debug("Task %s finished with %d results", task.get("id"), len(task_results))
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

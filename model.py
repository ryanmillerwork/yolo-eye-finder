"""
Label Studio ML backend for **Ultralytics YOLO‑pose** that sends filtered
key‑point predictions back to the UI.

• All tunables are simple constants below.
• Excessive ASCII art banner removed per user request.
• Added DEBUG‑level console logging so you can trace exactly what happens
  for every task the backend receives.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import uuid4

import logging
import numpy as np
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

# ---------------------------------------------------------------------------
# Configuration constants — tweak as needed
# ---------------------------------------------------------------------------
MODEL_PATH = "yolo11n-pose.pt"  # checkpoint to load
BOX_THR    = 0.15               # minimum box confidence
KP_THR     = 0.05               # minimum key‑point confidence
IMGSZ      = (192, 128)         # (height, width) sent to YOLO
DEVICE     = "cpu"              # "cpu", "0", "cuda:1", …

# ---------------------------------------------------------------------------
# Logging setup — everything goes to the console
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):
    """YOLO‑pose wrapper that filters low‑confidence key‑points and logs steps."""

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
    # Public API required by Label Studio ML backend
    # ------------------------------------------------------------------
    def predict(self, tasks: List[dict], **kwargs):  # noqa: D401
        """Generate filtered pre‑annotations for the given Label Studio *tasks*."""
        predictions = []

        for task in tasks:
            img_url = next(iter(task["data"].values()))  # first data field
            img_path = self.get_local_path(img_url)
            logger.debug("Processing task id=%s • image=%s", task.get("id"), img_url)

            # Run the model
            res = self.model.predict(
                source=img_path,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.box_thr,  # global box filter
                verbose=False,
            )[0]

            logger.debug("Model returned %d detections", len(res.boxes))

            h_orig, w_orig = res.orig_shape  # needed for % conversion
            task_results = []

            for det_idx, (box, det_conf) in enumerate(zip(res.boxes.xywh, res.boxes.conf)):
                if det_conf < self.box_thr:
                    logger.debug("Skipping box %d: conf %.3f < %.3f", det_idx, det_conf, self.box_thr)
                    continue  # skip weak detection

                # Convert box from center‑xywh to top‑left‑width‑height in %
                x_c, y_c, bw, bh = box.tolist()
                tlx_pct = (x_c - bw / 2) / w_orig * 100
                tly_pct = (y_c - bh / 2) / h_orig * 100
                w_pct = bw / w_orig * 100
                h_pct = bh / h_orig * 100

                # Collect key‑points above threshold
                kp_xy = res.keypoints.xy[det_idx].cpu().numpy()
                kp_conf = res.keypoints.conf[det_idx].cpu().numpy()

                points = []
                for kp_idx, ((x, y), kconf) in enumerate(zip(kp_xy, kp_conf)):
                    if kconf < self.kp_thr:
                        logger.debug("  • drop KP %d: conf %.3f < %.3f", kp_idx, kconf, self.kp_thr)
                        continue  # drop low‑conf kp
                    points.append({
                        "x": float(x / w_orig * 100),
                        "y": float(y / h_orig * 100),
                        "label": [self.kpt_labels[kp_idx]],
                        "confidence": float(kconf),
                    })

                if not points:
                    logger.debug("Box %d discarded – all KPs below threshold", det_idx)
                    continue

                logger.debug("Box %d kept with %d key‑points (conf %.3f)",
                             det_idx, len(points), det_conf)

                task_results.append({
                    "id": str(uuid4()),
                    "from_name": "keypoints",  # MUST match labeling config
                    "to_name": "image",        # MUST match labeling config
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

            logger.debug("Task finished with %d detections after filtering", len(task_results))

            predictions.append({
                "result": task_results,
                "score": float(np.mean([rr["score"] for rr in task_results]) if task_results else 0.0),
            })

        return predictions

    # ------------------------------------------------------------------
    # Training hook (not used here)
    # ------------------------------------------------------------------
    def fit(self, **kwargs):  # noqa: D401
        """Dummy *fit* so the ML backend can be registered without training."""
        return {}

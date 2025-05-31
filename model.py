"""
Label Studio ML backend for Ultralytics YOLO‑pose that filters out low‑confidence
boxes and key‑points before sending pre‑annotations to the Label Studio UI.

All runtime settings are plain **constants** in this file so you don’t have to
fiddle with environment variables anymore.
--------------------------------------------------------------------------
MODEL_PATH  – path to a *.pt* checkpoint (str)
BOX_THR     – minimum confidence for a detection box         (float)
KP_THR      – minimum confidence for an individual key‑point (float)
IMGSZ       – (height, width) image size fed to the model    (tuple[int,int])
DEVICE      – "cpu", "0", "cuda:1", …                       (str)

The script assumes you created an *Image* tag called "image" and a
*KeyPointLabels* tag called "keypoints" in the Label Studio labeling config.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

MODEL_PATH = "yolo11n-pose.pt"   # checkpoint to load
BOX_THR    = 0.20                # minimum box confidence
KP_THR     = 0.25                # minimum key‑point confidence
IMGSZ      = (192, 128)          # height, width sent to YOLO
DEVICE     = "cpu"               # "cpu", "0", "cuda:1", …


class NewModel(LabelStudioMLBase):
    """Ultralytics YOLO‑pose wrapper with per‑key‑point confidence filtering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------
        # Model & runtime settings (constants defined above)
        # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Public API required by Label Studio ML backend
    # ------------------------------------------------------------------
    def predict(self, tasks: List[dict], **kwargs):  # noqa: D401
        """Generate filtered pre‑annotations for the given Label Studio *tasks*."""
        predictions = []

        for task in tasks:
            img_url = next(iter(task["data"].values()))  # first data field
            img_path = self.get_local_path(img_url)

            # Run the model
            res = self.model.predict(
                source=img_path,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.box_thr,   # global box filter
                verbose=False,
            )[0]

            h_orig, w_orig = res.orig_shape  # needed for % conversion
            task_results = []

            for det_idx, (box, det_conf) in enumerate(zip(res.boxes.xywh, res.boxes.conf)):
                if det_conf < self.box_thr:
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
                        continue  # drop low‑conf kp
                    points.append({
                        "x": float(x / w_orig * 100),
                        "y": float(y / h_orig * 100),
                        "label": [self.kpt_labels[kp_idx]],
                        "confidence": float(kconf),
                    })

                if not points:  # ignore boxes with no good key‑points
                    continue

                task_results.append({
                    "id": str(uuid4()),
                    "from_name": "keypoints",  # MUST match your labeling config
                    "to_name": "image",        # MUST match your labeling config
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

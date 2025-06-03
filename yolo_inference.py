#!/usr/bin/env python3
import time
import json
import numpy as np
import cv2
import onnxruntime as ort

def compute_iou(box, boxes):
    """Compute IoU between one box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    return inter / (area1 + area2 - inter + 1e-6)

# 1) Load ONNX model
sess = ort.InferenceSession("yolo11n-pose.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0].name
_, _, inp_h, inp_w = inp.shape

# 2) Read & letterbox-pad input
orig = cv2.imread("test.png")
h0, w0 = orig.shape[:2]
scale = min(inp_w / w0, inp_h / h0)
nw, nh = int(w0 * scale), int(h0 * scale)
pad_w, pad_h = (inp_w - nw) / 2, (inp_h - nh) / 2

resized = cv2.resize(orig, (nw, nh))
padded = np.full((inp_h, inp_w, 3), 114, dtype=np.uint8)
y1, y2 = int(pad_h), int(pad_h) + nh
x1, x2 = int(pad_w), int(pad_w) + nw
padded[y1:y2, x1:x2] = resized

# 3) Preprocess to (1,3,inp_h,inp_w)
img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = img.transpose(2, 0, 1)[None]

# 4) Warm-up & benchmark
for _ in range(5):
    _ = sess.run([out], {inp.name: tensor})
runs = 50
t0 = time.perf_counter()
for _ in range(runs):
    raw = sess.run([out], {inp.name: tensor})[0]
t1 = time.perf_counter()
print(f"Average inference time: {(t1 - t0)/runs*1000:.2f} ms")

# 5) Decode raw outputs: shape (1, feat_dim, num_preds)
pred = raw[0].T  # → (num_preds, feat_dim)

# 6) Multi-class scores & IDs
#   layout: [xc,yc,w,h, score_face, score_tube, kpx1,kpy1,kpconf1, ...]
class_scores = pred[:, 4:6]                           # (num_preds, 2)
class_ids = np.argmax(class_scores, axis=1)           # 0=face,1=juice_tube
confs = class_scores[np.arange(len(class_scores)), class_ids]

# 7) Confidence filtering
conf_thresh = 0.3
mask = confs > conf_thresh
if not mask.any():
    print("No detections above threshold")
    exit()

pred = pred[mask]
class_ids = class_ids[mask]
confs = confs[mask]

# 8) Extract & convert KPs and boxes
# Keypoints raw
kp_data = pred[:, 6:]
total_kps = kp_data.shape[1] // 3
kps_raw = kp_data.reshape(-1, total_kps, 3)  # (num_dets, k, [x,y,conf])

# Boxes from (xc,yc,w,h)
xc, yc, wb, hb = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
x1 = xc - wb/2; y1 = yc - hb/2
x2 = xc + wb/2; y2 = yc + hb/2
boxes = np.stack([x1, y1, x2, y2], axis=1)

# 9) Un-pad & un-scale boxes
boxes[:, [0,2]] = (boxes[:, [0,2]] - pad_w) / scale
boxes[:, [1,3]] = (boxes[:, [1,3]] - pad_h) / scale
boxes = boxes.astype(int)

# 10) Un-pad & un-scale keypoints
kps_xy = kps_raw[..., :2].copy()
kps_xy[..., 0] = (kps_xy[..., 0] - pad_w) / scale
kps_xy[..., 1] = (kps_xy[..., 1] - pad_h) / scale
kps_xy = kps_xy.astype(int)
kps_conf = kps_raw[..., 2]

# 11) Per-class NMS
keep = []
for cls in np.unique(class_ids):
    idxs = np.where(class_ids == cls)[0]
    b = boxes[idxs]; s = confs[idxs]
    order = s.argsort()[::-1]
    while order.size:
        i = order[0]; keep.append(idxs[i])
        if order.size == 1: break
        ious = compute_iou(b[i], b[order[1:]])
        order = order[1:][ious < 0.5]
keep = sorted(set(keep))

final_boxes   = boxes[keep]
final_classes = class_ids[keep]
final_confs   = confs[keep]
final_kps_xy  = kps_xy[keep]
final_kps_conf= kps_conf[keep]

# 12) Build and print JSON
names = ["face", "juice_tube"]
kp_labels = {
    "face":        ["left_pupil", "right_pupil", "nose_bridge"],
    "juice_tube":  ["spout_top", "spout_bottom"],
}
face_kp_count = len(kp_labels["face"])

results = {}
for box, cls, conf, kp_xy, kp_c in zip(final_boxes, final_classes, final_confs, final_kps_xy, final_kps_conf):
    cname = names[cls]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    results[cname] = {
        "box": [int(x1), int(y1), int(bw), int(bh), float(conf)]
    }
    offset = 0 if cls == 0 else face_kp_count
    for j, label in enumerate(kp_labels[cname]):
        idx = offset + j
        if idx < kp_xy.shape[0]:
            xkp, ykp = int(kp_xy[idx][0]), int(kp_xy[idx][1])
            ckp = float(kp_c[idx])
            results[cname][label] = [xkp, ykp, ckp]
        else:
            results[cname][label] = None

print(json.dumps(results, indent=2))

# 13) (Optional) Draw & save image
vis = orig.copy()
colors = [(0,255,0), (255,0,0)]
for box, cls, kp in zip(final_boxes, final_classes, final_kps_xy):
    x1, y1, x2, y2 = box
    c = colors[cls]
    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
    for (kx, ky) in kp:
        cv2.circle(vis, (kx, ky), 3, c, -1)
cv2.imwrite("output.jpg", vis)

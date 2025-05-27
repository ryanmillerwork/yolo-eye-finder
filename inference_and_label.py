#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import cv2
from ultralytics import YOLO

# ----------------------
# Configuration Constants
# ----------------------
DOT_RADIUS = 2         # Radius of keypoint circles
DOT_THICKNESS = -1     # Filled circle
BOX_THICKNESS = 1      # Thickness for bounding box lines

# Colors in BGR format
FACE_BOX_COLOR = (0, 255, 0)    # Green for face
TUBE_BOX_COLOR = (255, 0, 0)    # Blue for tube
FACE_DOT_COLOR = (0, 128, 0)    # Dark green for face keypoints
TUBE_DOT_COLOR = (128, 0, 0)    # Dark blue for tube keypoints

# Class IDs (as in your training)
FACE_CLASS_ID = 0
TUBE_CLASS_ID = 1
# ----------------------

def draw_detections(img, boxes, keypoints, class_id, box_color, dot_color):
    """
    Draw at most one bounding box and its keypoints for the given class_id.
    Handles keypoints arrays of shape (K,2) or (K,3).
    """
    # Convert to numpy
    cls_array  = boxes.cls.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy()
    xyxy_array = boxes.xyxy.cpu().numpy()
    kp_array   = keypoints.cpu().numpy()  # shape (N, K, 2 or 3)

    # Filter indices of this class
    indices = [i for i, c in enumerate(cls_array) if int(c) == class_id]
    if not indices:
        return

    # Pick highest-confidence detection
    best_i = max(indices, key=lambda i: conf_array[i])

    # Draw the box
    x1, y1, x2, y2 = map(int, xyxy_array[best_i])
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

    h, w = img.shape[:2]
    # Draw keypoints
    for pt in kp_array[best_i]:
        # pt can be [x,y] or [x,y,vis]
        nx, ny = pt[0], pt[1]
        vis = pt[2] if len(pt) == 3 else 1
        if vis > 0:
            px = int(nx * w)
            py = int(ny * h)
            cv2.circle(img, (px, py), DOT_RADIUS, dot_color, DOT_THICKNESS)



def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO-Pose inference on a random subset and save annotated images")
    parser.add_argument("images_dir",  help="Directory containing source images (PNG/JPG)")
    parser.add_argument("model_path",  help="Path to your trained YOLO-Pose best.pt")
    parser.add_argument("num_images", type=int,
                        help="Number of random images to process")
    parser.add_argument("output_dir",   help="Directory to save output images")
    parser.add_argument("--imgsz", type=int, default=192, help="Inference image size")
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample images
    all_imgs = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    random.seed(42)
    selected = random.sample(all_imgs, min(args.num_images, len(all_imgs)))

    # Load the YOLO-Pose model
    model = YOLO(args.model_path)

    # Perform inference (no display, no txt save)
    results = model.predict(source=[str(p) for p in selected],
                            imgsz=args.imgsz,
                            save_txt=False,
                            show=False)

    # Annotate and save each image
    for img_path, res in zip(selected, results):
        img = cv2.imread(str(img_path))
        draw_detections(img, res.boxes, res.keypoints.xyn, FACE_CLASS_ID, FACE_BOX_COLOR, FACE_DOT_COLOR)
        draw_detections(img, res.boxes, res.keypoints.xyn, TUBE_CLASS_ID, TUBE_BOX_COLOR, TUBE_DOT_COLOR)
        cv2.imwrite(str(out_dir / img_path.name), img)

if __name__ == "__main__":
    main()

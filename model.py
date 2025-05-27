from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image

class YoloPoseBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load your YOLOv11-pose-nano weights
        self.model = YOLO("./yolo-pose-backend/best-nano.pt")

    def predict(self, tasks, context=None, **kwargs):
        predictions = []
        for task in tasks:
            # 1) Get the image URI from the task data
            uri = task.get('data', {}).get('img')
            if not uri:
                predictions.append({'result': []})
                continue

            # 2) Resolve Label Studio–managed URLs to a local filesystem path
            try:
                image_path = self.get_local_path(uri, task_id=task['id'])
            except Exception:
                image_path = uri

            # 3) Load original image size for percentage calculations
            with Image.open(image_path) as img:
                orig_w, orig_h = img.size  # (width, height)

            # 4) Run YOLO inference
            results = self.model.predict(
                source=image_path, device='cpu', imgsz=640
            )[0]

            ls_results = []
            class_names = self.model.names  # {0: 'face', 1: 'juice_tube', ...}

            for box, keypoints, cls_id in zip(
                results.boxes.xyxy, results.keypoints, results.boxes.cls
            ):
                # Extract box in pixels
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                w_px, h_px = x2 - x1, y2 - y1

                # Convert box to percentages
                x_pct = (x1 / orig_w) * 100
                y_pct = (y1 / orig_h) * 100
                w_pct = (w_px / orig_w) * 100
                h_pct = (h_px / orig_h) * 100

                # Determine class label
                cls = int(cls_id.cpu().numpy().item())
                cls_name = class_names.get(cls, 'unknown')

                # Map to your Label Studio tag names
                if cls_name == 'face':
                    bbox_from, kp_from = 'bbox_face', 'kp_face'
                    rect_label = ['face']
                    kp_labels = ['left_pupil', 'right_pupil', 'nose_bridge']
                elif cls_name == 'juice_tube':
                    bbox_from, kp_from = 'bbox_tube', 'kp_tube'
                    rect_label = ['juice_tube']
                    kp_labels = ['spout_top', 'spout_bottom']
                else:
                    # skip unknown classes
                    continue

                # RectangleLabels entry (percent coords)
                ls_results.append({
                    'from_name': bbox_from,
                    'to_name': 'img',
                    'type': 'rectanglelabels',
                    'value': {
                        'x':  x_pct,  # % from left
                        'y':  y_pct,  # % from top
                        'width':  w_pct,
                        'height': h_pct,
                        'rotation': 0,
                        'rectanglelabels': rect_label
                    }
                })

                # KeyPointLabels entry (percent coords & correct 'id' field)
                raw_pts = keypoints.data.cpu().numpy().tolist()
                points = []
                for idx, (x_px, y_px, conf) in enumerate(raw_pts):
                    points.append({
                        'x': (x_px / orig_w) * 100,
                        'y': (y_px / orig_h) * 100,
                        'id': idx     # Label Studio expects 'id', not 'key_id'
                    })

                ls_results.append({
                    'from_name': kp_from,
                    'to_name': 'img',
                    'type': 'keypointlabels',
                    'value': {
                        'points': points,
                        'labels': kp_labels
                    }
                })

            predictions.append({'result': ls_results})

        return predictions

# Alias so Label Studio’s _wsgi.py can import it
NewModel = YoloPoseBackend

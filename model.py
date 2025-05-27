from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

class YoloPoseBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load your trained YOLOv11 pose nano weights
        self.model = YOLO("./yolo-pose-backend/best-nano.pt")

    def predict(self, tasks, context=None, **kwargs):
        predictions = []
        for task in tasks:
            # 1) Get the image URI from the task data (your config uses name="img")
            uri = task.get('data', {}).get('img') or task['data'].get('image')
            if not uri:
                predictions.append({'result': []})
                continue

            # 2) Resolve LS-managed URLs to a local filesystem path
            try:
                image_path = self.get_local_path(uri, task_id=task['id'])
            except Exception:
                image_path = uri  # fallback to original URI

            # 3) Run YOLO inference
            results = self.model.predict(source=image_path, device='cpu', imgsz=640)[0]

            ls_results = []
            # 4) Build LS annotations for each detection
            for box, keypoints, cls_id in zip(
                results.boxes.xyxy, results.keypoints, results.boxes.cls
            ):
                # Bounding box
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                w, h = x2 - x1, y2 - y1
                cls = int(cls_id.cpu().numpy().item())

                if cls == 0:
                    bbox_from, kp_from = 'bbox_face', 'kp_face'
                    rect_label = 'face'
                    kp_labels = ['left_pupil', 'right_pupil', 'nose_bridge']
                else:
                    bbox_from, kp_from = 'bbox_tube', 'kp_tube'
                    rect_label = 'juice_tube'
                    kp_labels = ['spout_top', 'spout_bottom']

                # RectangleLabels entry
                ls_results.append({
                    'from_name': bbox_from,
                    'to_name': 'img',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': x1, 'y': y1,
                        'width': w, 'height': h,
                        'rotation': 0,
                        'rectanglelabels': [rect_label]
                    }
                })

                # KeyPointLabels entry (extract raw data tensor → numpy → list)
                raw_pts = keypoints.data.cpu().numpy().tolist()
                formatted = [
                    {'x': x, 'y': y, 'key_id': idx, 'probability': conf}
                    for idx, (x, y, conf) in enumerate(raw_pts)
                ]
                ls_results.append({
                    'from_name': kp_from,
                    'to_name': 'img',
                    'type': 'keypointlabels',
                    'value': {
                        'points': formatted,
                        'labels': kp_labels
                    }
                })

            predictions.append({'result': ls_results})

        return predictions

# Alias so Label Studio’s _wsgi.py can import it
NewModel = YoloPoseBackend

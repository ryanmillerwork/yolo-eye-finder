from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

class YoloPoseBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO("./yolo-pose-backend/best-nano.pt")

    def predict(self, tasks, context=None, **kwargs):
        predictions = []
        for task in tasks:
            uri = task.get('data', {}).get('img') or task['data'].get('image')
            if not uri:
                predictions.append({'result': []})
                continue

            # 1) Try to resolve LS-managed URLs into a local filesystem path
            try:
                # task['id'] gives the unique task identifier
                image_path = self.get_local_path(uri, task_id=task['id'])
            except Exception:
                # fallback to original URI (e.g. external HTTP URL)
                image_path = uri
            :contentReference[oaicite:5]{index=5}

            # 2) Run YOLO inference on the resolved path/URL
            results = self.model.predict(source=image_path, device='cpu', imgsz=640)[0]

            ls_results = []
            for box, keypoints, cls_id in zip(results.boxes.xyxy, results.keypoints, results.boxes.cls):
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                w, h = x2 - x1, y2 - y1
                cls = int(cls_id.cpu().numpy().item())

                if cls == 0:
                    bbox_from, kp_from = 'bbox_face', 'kp_face'
                    rect_label = 'face'
                    kp_labels = ['left_pupil','right_pupil','nose_bridge']
                else:
                    bbox_from, kp_from = 'bbox_tube', 'kp_tube'
                    rect_label = 'juice_tube'
                    kp_labels = ['spout_top','spout_bottom']

                # RectangleLabels
                ls_results.append({
                    'from_name': bbox_from,
                    'to_name': 'img',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': x1, 'y': y1, 'width': w, 'height': h,
                        'rotation': 0,
                        'rectanglelabels': [rect_label]
                    }
                })

                # KeyPointLabels
                pts = keypoints.cpu().numpy().tolist()
                formatted = [
                    {'x': x, 'y': y, 'key_id': i, 'probability': prob}
                    for i,(x,y,prob) in enumerate(pts)
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

# Alias for Label Studio
NewModel = YoloPoseBackend

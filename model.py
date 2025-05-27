from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO


class YoloPoseBackend(LabelStudioMLBase):
    """
    ML backend for YOLOv11 Pose Nano model, matching your Label Studio config.
    It returns both bounding boxes and keypoints for 'face' and 'juice_tube' instances,
    formatted to <RectangleLabels> and <KeyPointLabels> tags.
    """
    def __init__(self, **kwargs):
        super(YoloPoseBackend, self).__init__(**kwargs)
        # Path to your trained YOLOv11 pose nano weights
        weights_path = "./yolo-pose-backend/best-nano.pt"
        self.model = YOLO(weights_path)

    def predict(self, tasks, **kwargs):
        """
        Run inference on a batch of tasks and convert results to LS format.
        """
        predictions = []

        for task in tasks:
            data = task.get('data', {})
            image_url = data.get('img') or data.get('image') or data.get('url')
            if not image_url:
                predictions.append({'result': []})
                continue

            # Run inference (CPU, 640px)
            results = self.model.predict(source=image_url, device='cpu', imgsz=640)[0]

            ls_results = []
            # For each detected instance
            for box, keypoints, cls_id in zip(results.boxes.xyxy, results.keypoints, results.boxes.cls):
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                width = x2 - x1
                height = y2 - y1
                cls = int(cls_id.cpu().numpy().item())

                # Map class ID to your LS tags
                if cls == 0:
                    # Face
                    bbox_from = 'bbox_face'
                    kp_from = 'kp_face'
                    rect_label = 'face'
                    kp_labels = ['left_pupil', 'right_pupil', 'nose_bridge']
                else:
                    # Juice tube
                    bbox_from = 'bbox_tube'
                    kp_from = 'kp_tube'
                    rect_label = 'juice_tube'
                    kp_labels = ['spout_top', 'spout_bottom']

                # Rectangle annotation
                ls_results.append({
                    'from_name': bbox_from,
                    'to_name': 'img',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height,
                        'rotation': 0,
                        'rectanglelabels': [rect_label]
                    }
                })

                # Keypoint annotations
                pts = keypoints.cpu().numpy().tolist()
                formatted_points = []
                for idx, (x, y, prob) in enumerate(pts):
                    formatted_points.append({
                        'x': x,
                        'y': y,
                        'key_id': idx,
                        'probability': prob
                    })

                ls_results.append({
                    'from_name': kp_from,
                    'to_name': 'img',
                    'type': 'keypointlabels',
                    'value': {
                        'points': formatted_points,
                        'labels': kp_labels
                    }
                })

            predictions.append({'result': ls_results})

        return predictions


# Label Studio expects a class named "NewModel" in model.py
NewModel = YoloPoseBackend

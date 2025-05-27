from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from label_studio_tools.core.utils.io import get_local_path


class YoloPoseBackend(LabelStudioMLBase):
    """
    ML backend for YOLOv11 Pose Nano model, matching your Label Studio config.
    Uses get_local_path to resolve uploaded/local image URIs to filesystem paths.
    Returns both bounding boxes and keypoints formatted for Label Studio.
    """
    def __init__(self, **kwargs):
        super(YoloPoseBackend, self).__init__(**kwargs)
        # Load your trained YOLOv11 pose nano weights
        weights_path = "./yolo-pose-backend/best-nano.pt"
        self.model = YOLO(weights_path)

    def predict(self, tasks, context=None, **kwargs):
        """
        Run inference on a batch of tasks and convert results to LS format.
        Resolves image URIs (uploaded/local) to a local filesystem path for inference.
        """
        predictions = []

        for task in tasks:
            data = task.get('data', {})
            # Extract the image reference key matching your LS config
            image_ref = data.get('img') or data.get('image') or data.get('url')
            if not image_ref:
                predictions.append({'result': []})
                continue

            # Resolve to local path (requires LABEL_STUDIO_URL & API_KEY env vars)
            try:
                image_path = get_local_path(image_ref, task_id=task.get('id'))
            except Exception as e:
                # fallback: use original ref if local resolution fails
                image_path = image_ref

            # Perform inference on the local file
            results = self.model.predict(source=image_path, device='cpu', imgsz=640)[0]

            ls_results = []
            # Iterate over detected instances
            for box, keypoints, cls_id in zip(results.boxes.xyxy, results.keypoints, results.boxes.cls):
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                width = x2 - x1
                height = y2 - y1
                cls = int(cls_id.cpu().numpy().item())

                # Map class ID to LS tag names and labels
                if cls == 0:
                    bbox_from = 'bbox_face'
                    kp_from = 'kp_face'
                    rect_label = 'face'
                    kp_labels = ['left_pupil', 'right_pupil', 'nose_bridge']
                else:
                    bbox_from = 'bbox_tube'
                    kp_from = 'kp_tube'
                    rect_label = 'juice_tube'
                    kp_labels = ['spout_top', 'spout_bottom']

                # RectangleLabels annotation
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

                # KeyPointLabels annotation
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


# Alias for Label Studio import
NewModel = YoloPoseBackend

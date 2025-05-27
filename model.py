from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image

class YoloPoseBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO("./yolo-pose-backend/best-nano.pt")

    def predict(self, tasks, context=None, **kwargs):
        predictions = []
        for task in tasks:
            # ▶︎ FIXED: pull either your 'img' alias or the raw 'image' key
            data = task.get('data', {})
            uri  = data.get('img') or data.get('image')
            if not uri:
                predictions.append({'result': []})
                continue

            # resolve the path & load image size
            try:
                image_path = self.get_local_path(uri, task_id=task['id'])
            except Exception:
                image_path = uri
            with Image.open(image_path) as img:
                orig_w, orig_h = img.size

            # run inference
            results = self.model.predict(source=image_path, device='cpu', imgsz=640)[0]
            ls_results = []
            class_names = self.model.names

            for box, keypoints, cls_id in zip(results.boxes.xyxy,
                                               results.keypoints,
                                               results.boxes.cls):
                # box → px → %
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                w_px, h_px = x2 - x1, y2 - y1
                x_pct = x1 / orig_w * 100
                y_pct = y1 / orig_h * 100
                w_pct = w_px / orig_w * 100
                h_pct = h_px / orig_h * 100

                cls      = int(cls_id.cpu().numpy().item())
                cls_name = class_names.get(cls, '')

                if cls_name == 'face':
                    bbox_from, kp_from = 'bbox_face', 'kp_face'
                    rect_label = ['face']
                    kp_labels  = ['left_pupil', 'right_pupil', 'nose_bridge']
                elif cls_name == 'juice_tube':
                    bbox_from, kp_from = 'bbox_tube', 'kp_tube'
                    rect_label = ['juice_tube']
                    kp_labels  = ['spout_top', 'spout_bottom']
                else:
                    continue  # skip anything else

                # rectangle
                ls_results.append({
                    'from_name': bbox_from,
                    'to_name':   'img',
                    'type':      'rectanglelabels',
                    'value': {
                        'x': x_pct, 'y': y_pct,
                        'width':  w_pct, 'height': h_pct,
                        'rotation': 0,
                        'rectanglelabels': rect_label
                    }
                })

                # keypoints
                raw_pts = keypoints.data.cpu().numpy().tolist()
                points  = [{
                    'x': pt[0]/orig_w*100,
                    'y': pt[1]/orig_h*100,
                    'id': idx
                } for idx, pt in enumerate(raw_pts)]

                ls_results.append({
                    'from_name': kp_from,
                    'to_name':   'img',
                    'type':      'keypointlabels',
                    'value': {
                        'points': points,
                        'labels': kp_labels
                    }
                })

            predictions.append({'result': ls_results})

        return predictions

# alias for Label Studio ML entrypoint
NewModel = YoloPoseBackend

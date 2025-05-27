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

                # KEYPOINTS Section
                # keypoints_item is the tensor for the current instance's keypoints from the zip loop.
                # Expected shape (K,D) e.g. (num_keypoints, 3 for x,y,visibility)
                # The TypeError suggests it might sometimes be (1,K,D) leading to an extra list nesting.
                
                kp_tensor_as_list = keypoints.cpu().numpy().tolist() # Use keypoints directly, not keypoints.data
                
                individual_kps_data = []
                if kp_tensor_as_list: # Check if list is not empty
                    if isinstance(kp_tensor_as_list[0], list) and \
                       kp_tensor_as_list[0] and isinstance(kp_tensor_as_list[0][0], list):
                        # Handles cases like [[[x,y,v], ...]] (from a (1,K,D) tensor)
                        individual_kps_data = kp_tensor_as_list[0]
                    elif isinstance(kp_tensor_as_list[0], list):
                        # Handles cases like [[x,y,v], ...] (from a (K,D) tensor)
                        individual_kps_data = kp_tensor_as_list
                    # Else, structure is not recognized or empty, individual_kps_data remains []

                # kp_labels was defined earlier, e.g., ['left_pupil', 'right_pupil', 'nose_bridge']
                
                for kp_idx, single_kp_coords in enumerate(individual_kps_data):
                    if kp_idx >= len(kp_labels):
                        # More keypoints detected than labels defined for this class
                        break 

                    if not (isinstance(single_kp_coords, (list, tuple)) and len(single_kp_coords) >= 2):
                        # Malformed keypoint data for this specific keypoint
                        continue 

                    x_px = float(single_kp_coords[0])
                    y_px = float(single_kp_coords[1])
                    visibility = 1.0 # Default if not present
                    if len(single_kp_coords) > 2:
                        visibility = float(single_kp_coords[2])

                    # Skip keypoints that are often padding/non-existent in YOLO outputs
                    # (i.e., at origin with low/zero visibility)
                    if visibility < 0.1 and abs(x_px) < 1e-3 and abs(y_px) < 1e-3:
                        continue
                    
                    ls_results.append({
                        'from_name': kp_from, 
                        'to_name': 'img', 
                        'type': 'keypointlabels',
                        'original_width': orig_w,
                        'original_height': orig_h,
                        'image_rotation': 0,
                        'value': {
                            'x': (x_px / orig_w) * 100.0 if orig_w > 0 else 0,
                            'y': (y_px / orig_h) * 100.0 if orig_h > 0 else 0,
                            'width': 1.0, # Label Studio UI often expects a small width
                            'keypointlabels': [kp_labels[kp_idx]]
                        }
                    })

            predictions.append({'result': ls_results})

        return predictions

# alias for Label Studio ML entrypoint
NewModel = YoloPoseBackend

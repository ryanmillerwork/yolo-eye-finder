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
            data = task.get('data', {})
            uri  = data.get('img') or data.get('image')
            if not uri:
                predictions.append({'result': []})
                continue

            try:
                image_path = self.get_local_path(uri, task_id=task['id'])
            except Exception:
                image_path = uri
            with Image.open(image_path) as img:
                orig_w, orig_h = img.size

            # run inference
            results = self.model.predict(source=image_path, device='cpu', imgsz=640)[0]
            class_names = self.model.names

            # Store all valid, processed detections before filtering
            # Structure: {'face': [{'confidence': 0.9, 'box_ls_item': {...}, 'kps_ls_items': [...]}, ...], 'juice_tube': []}
            all_processed_detections = {'face': [], 'juice_tube': []}

            num_detections = results.boxes.shape[0]
            if num_detections == 0:
                predictions.append({'result': []})
                continue

            # Get all tensor data once
            boxes_xyxy_all = results.boxes.xyxy.cpu().numpy()
            boxes_cls_all = results.boxes.cls.cpu().numpy()
            boxes_conf_all = results.boxes.conf.cpu().numpy()
            
            keypoints_data_all_detections_np = None
            if results.keypoints is not None and results.keypoints.data is not None:
                keypoints_data_all_detections_np = results.keypoints.data.cpu().numpy()

            for i in range(num_detections):
                box_coords_px = boxes_xyxy_all[i]
                cls_id_val = int(boxes_cls_all[i])
                conf_score_val = float(boxes_conf_all[i])
                
                # Bounding Box Processing
                x1_px, y1_px, x2_px, y2_px = box_coords_px
                box_w_px, box_h_px = x2_px - x1_px, y2_px - y1_px
                
                x_pct = (x1_px / orig_w) * 100.0 if orig_w > 0 else 0
                y_pct = (y1_px / orig_h) * 100.0 if orig_h > 0 else 0
                w_pct = (box_w_px / orig_w) * 100.0 if orig_w > 0 else 0
                h_pct = (box_h_px / orig_h) * 100.0 if orig_h > 0 else 0

                cls_name_str = class_names.get(cls_id_val, '')
                
                current_box_ls_item = None
                kp_from_name = ''
                defined_kp_labels_for_class = []

                if cls_name_str == 'face':
                    bbox_from_name, kp_from_name = 'bbox_face', 'kp_face'
                    rect_label_val = ['face']
                    defined_kp_labels_for_class  = ['left_pupil', 'right_pupil', 'nose_bridge']
                elif cls_name_str == 'juice_tube':
                    bbox_from_name, kp_from_name = 'bbox_tube', 'kp_tube'
                    rect_label_val = ['juice_tube']
                    defined_kp_labels_for_class  = ['spout_top', 'spout_bottom']
                else:
                    continue # Skip other classes

                current_box_ls_item = {
                    'from_name': bbox_from_name,
                    'to_name':   'img',
                    'type':      'rectanglelabels',
                    'original_width': orig_w,
                    'original_height': orig_h,
                    'image_rotation': 0,
                    'value': {
                        'x': x_pct, 'y': y_pct,
                        'width':  w_pct, 'height': h_pct,
                        'rotation': 0,
                        'rectanglelabels': rect_label_val
                    }
                }

                # Keypoint Processing for current detection i
                current_kps_ls_items = []
                # keypoints_for_this_detection_np will be a (K,D) numpy array or None
                keypoints_for_this_detection_np = None
                if keypoints_data_all_detections_np is not None and i < keypoints_data_all_detections_np.shape[0]:
                    keypoints_for_this_detection_np = keypoints_data_all_detections_np[i]

                if keypoints_for_this_detection_np is not None and defined_kp_labels_for_class:
                    # Iterate over rows of the (K,D) numpy array for this instance
                    for kp_idx, single_kp_coords_np_row in enumerate(keypoints_for_this_detection_np):
                        if kp_idx >= len(defined_kp_labels_for_class):
                            break # Processed all defined keypoints for this class

                        if not (isinstance(single_kp_coords_np_row, (list, tuple)) or type(single_kp_coords_np_row).__name__ == 'ndarray') or len(single_kp_coords_np_row) < 2:
                            continue 

                        kp_x_px = float(single_kp_coords_np_row[0])
                        kp_y_px = float(single_kp_coords_np_row[1])
                        visibility = 1.0 
                        if len(single_kp_coords_np_row) > 2:
                            visibility = float(single_kp_coords_np_row[2])

                        if visibility < 0.1 and abs(kp_x_px) < 1e-3 and abs(kp_y_px) < 1e-3:
                            continue
                        
                        current_kps_ls_items.append({
                            'from_name': kp_from_name, 
                            'to_name': 'img', 
                            'type': 'keypointlabels',
                            'original_width': orig_w,
                            'original_height': orig_h,
                            'image_rotation': 0,
                            'value': {
                                'x': (kp_x_px / orig_w) * 100.0 if orig_w > 0 else 0,
                                'y': (kp_y_px / orig_h) * 100.0 if orig_h > 0 else 0,
                                'width': 1.0, 
                                'keypointlabels': [defined_kp_labels_for_class[kp_idx]]
                            }
                        })
                
                # Store this processed detection (box and its keypoints)
                if cls_name_str in all_processed_detections:
                    all_processed_detections[cls_name_str].append({
                        'confidence': conf_score_val,
                        'box_ls_item': current_box_ls_item,
                        'kps_ls_items': current_kps_ls_items
                    })

            # Filter to best detection per class and build final ls_results for this task
            final_ls_results_for_task = []
            for class_key_to_filter in ['face', 'juice_tube']: 
                detections_of_this_class = all_processed_detections[class_key_to_filter]
                if detections_of_this_class:
                    # Sort by confidence, highest first
                    best_detection = sorted(detections_of_this_class, key=lambda x: x['confidence'], reverse=True)[0]
                    if best_detection['box_ls_item']:
                         final_ls_results_for_task.append(best_detection['box_ls_item'])
                    if best_detection['kps_ls_items']:
                         final_ls_results_for_task.extend(best_detection['kps_ls_items'])
            
            predictions.append({'result': final_ls_results_for_task})

        return predictions

# alias for Label Studio ML entrypoint
NewModel = YoloPoseBackend

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image

class YoloPoseBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO("./yolo-pose-backend/best-small.pt")

    def predict(self, tasks, context=None, **kwargs):
        predictions = []
        KEYPOINT_CONFIDENCE_THRESHOLD = 0.7 # Threshold for filtering keypoints

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
            results = self.model.predict(source=image_path, device='cpu', imgsz=(192, 128))[0]
            class_names = self.model.names

            # Store all valid, processed detections before filtering
            # Structure: {'face': [{'confidence': 0.9, 'box_ls_item': {...}, 'kps_ls_items': [...]}, ...], 'juice_tube': []}
            all_processed_detections = {'face': [], 'juice_tube': []}

            num_detections = results.boxes.shape[0]
            if num_detections == 0:
                predictions.append({'result': []})
                continue

            # Get all tensor data once
            boxes_xyxyn_all = results.boxes.xyxyn.cpu().numpy()
            boxes_cls_all = results.boxes.cls.cpu().numpy()
            boxes_conf_all = results.boxes.conf.cpu().numpy()
            
            keypoints_xyn_all_detections_np = None
            keypoints_conf_all_detections_np = None # For keypoint confidences

            if results.keypoints is not None:
                if results.keypoints.xyn is not None:
                    keypoints_xyn_all_detections_np = results.keypoints.xyn.cpu().numpy()
                if hasattr(results.keypoints, 'conf') and results.keypoints.conf is not None: # Check for conf attribute
                    keypoints_conf_all_detections_np = results.keypoints.conf.cpu().numpy()
                else:
                    print("DEBUG YoloPoseBackend: results.keypoints.conf is None or not available.")

            for i in range(num_detections):
                box_coords_norm = boxes_xyxyn_all[i]
                cls_id_val = int(boxes_cls_all[i])
                conf_score_val = float(boxes_conf_all[i])
                
                # Bounding Box Processing from normalized coordinates
                x1_norm, y1_norm, x2_norm, y2_norm = box_coords_norm
                
                x_pct = x1_norm * 100.0
                y_pct = y1_norm * 100.0
                w_pct = (x2_norm - x1_norm) * 100.0
                h_pct = (y2_norm - y1_norm) * 100.0

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
                
                keypoints_for_this_detection_normalized_np = None
                keypoints_conf_for_this_detection_np = None

                if keypoints_xyn_all_detections_np is not None and i < keypoints_xyn_all_detections_np.shape[0]:
                    keypoints_for_this_detection_normalized_np = keypoints_xyn_all_detections_np[i]
                
                if keypoints_conf_all_detections_np is not None and i < keypoints_conf_all_detections_np.shape[0]:
                    keypoints_conf_for_this_detection_np = keypoints_conf_all_detections_np[i]

                if keypoints_for_this_detection_normalized_np is not None and defined_kp_labels_for_class:
                    # Iterate over rows of the (K,D) numpy array for this instance
                    for kp_idx, single_kp_coords_normalized_row in enumerate(keypoints_for_this_detection_normalized_np):
                        if kp_idx >= len(defined_kp_labels_for_class):
                            break # Processed all defined keypoints for this class
                        
                        if not (isinstance(single_kp_coords_normalized_row, (list, tuple)) or type(single_kp_coords_normalized_row).__name__ == 'ndarray') or len(single_kp_coords_normalized_row) < 2:
                            print(f"DEBUG YoloPoseBackend: Skipping malformed keypoint data for '{defined_kp_labels_for_class[kp_idx]}': {single_kp_coords_normalized_row}")
                            continue 

                        kp_x_norm = float(single_kp_coords_normalized_row[0])
                        kp_y_norm = float(single_kp_coords_normalized_row[1])
                        
                        # NEW Keypoint Filtering Logic
                        kp_label_name = defined_kp_labels_for_class[kp_idx]

                        if keypoints_conf_for_this_detection_np is not None and kp_idx < len(keypoints_conf_for_this_detection_np):
                            kp_confidence = float(keypoints_conf_for_this_detection_np[kp_idx])
                            print(f"DEBUG YoloPoseBackend: KP '{kp_label_name}' Coords: ({kp_x_norm:.3f}, {kp_y_norm:.3f}), Conf: {kp_confidence:.3f}")
                            if kp_confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
                                print(f"DEBUG YoloPoseBackend: Skipping KP '{kp_label_name}' due to low confidence ({kp_confidence:.3f} < {KEYPOINT_CONFIDENCE_THRESHOLD})")
                                continue
                        else:
                            # No confidence score available for this keypoint, fall back to (0,0) check
                            # This path might be taken if results.keypoints.conf was None for all detections,
                            # or if the conf tensor had an unexpected shape for this specific detection.
                            print(f"DEBUG YoloPoseBackend: No confidence for KP '{kp_label_name}'. Coords: ({kp_x_norm:.3f}, {kp_y_norm:.3f}). Checking for (0,0).")
                            if abs(kp_x_norm) < 1e-5 and abs(kp_y_norm) < 1e-5: # Stricter (0,0) check
                                print(f"DEBUG YoloPoseBackend: Skipping KP '{kp_label_name}' due to (0,0) location and no confidence score.")
                                continue
                        
                        current_kps_ls_items.append({
                            'from_name': kp_from_name, 
                            'to_name': 'img', 
                            'type': 'keypointlabels',
                            'original_width': orig_w,
                            'original_height': orig_h,
                            'image_rotation': 0,
                            'value': {
                                'x': kp_x_norm * 100.0, # Use normalized value directly
                                'y': kp_y_norm * 100.0, # Use normalized value directly
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

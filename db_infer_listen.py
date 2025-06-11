import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from typing import List, Any
import io # For in-memory byte streams
import json # To format results for the database
from PIL import Image # For handling image data; pip install Pillow
from ultralytics import YOLO # For YOLO model inference; pip install ultralytics
import torch # To check for CUDA availability
import time # For benchmarking

def get_unprocessed_inferences(conn, limit: int) -> List[Any]:
    """
    Retrieves a batch of unprocessed records (where infer_label is NULL)
    from the server_inference table.
    
    Args:
        conn: An active psycopg2 connection object.
        limit: The maximum number of records to retrieve.

    Returns:
        A list of DictRow objects.
    """

    records = []
    try:
        cur = conn.cursor()
        query = """
            SELECT * FROM server_inference 
            WHERE infer_label IS NULL 
            ORDER BY server_infer_id 
            LIMIT %s
        """
        cur.execute(query, (limit,))
        records = cur.fetchall()
        cur.close()
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching unprocessed records: {error}")
    return records

def update_inference_record(conn, server_id: int, model_name: str, infer_label: str, confidence: float):
    """
    Updates a record in the server_inference table with inference results
    using a provided database connection.
    
    Args:
        conn: An active psycopg2 connection object.
        server_id: The ID of the record to update.
        model_name: The name of the model file used.
        infer_label: The JSON string of the inference results.
        confidence: The calculated confidence score.
    """
    try:
        cur = conn.cursor()
        sql = """
            UPDATE server_inference
            SET model_file = %s, infer_label = %s, confidence = %s
            WHERE server_infer_id = %s
        """
        cur.execute(sql, (model_name, infer_label, confidence, server_id))
        cur.close()
        # The commit will be handled at the end of the batch in the main loop
    except (Exception, psycopg2.Error) as error:
        print(f"Error while updating record {server_id}: {error}")

BATCH_SIZE = 256 # Optimal batch size determined from benchmarking
MODEL_PATH = "./models/HB-eyes-1000_small.pt"
IMAGE_SIZE = (192, 128) # imgsz used during training (width, height)
CONF_THRESHOLD = 0.1 # Lowered confidence threshold for detection. Default is 0.25

def process_inference_results(conn, batch_ids, batch_results, model_obj):
    """
    Processes inference results, formats them for the DB, calls the
    update function, and prints a concise one-line summary.
    
    Args:
        conn: An active psycopg2 connection object.
        batch_ids: A list of server_infer_ids for the processed batch.
        batch_results: The results from the YOLO model.
        model_obj: The loaded YOLO model object.
    """
    model_name = os.path.basename(MODEL_PATH)
    
    # Define class-specific keypoint names
    keypoint_names = {
        0: ["left_pupil", "right_pupil", "nose_bridge"],  # face class
        1: ["spout_top", "spout_bottom"]                  # juice_tube class
    }

    for i, result in enumerate(batch_results):
        server_id = batch_ids[i]
        result_summary = ""

        # Check for both keypoints AND boxes to ensure we have a full detection
        if result.keypoints is not None and result.boxes is not None and result.boxes.data.numel() > 0:
            num_poses = result.keypoints.shape[0]
            
            # Group poses by class and keep only the most confident one for each class
            best_poses_by_class = {}
            highest_avg_confidence = 0.0

            for pose_idx in range(num_poses):
                # --- Bounding Box Data ---
                box_data = result.boxes[pose_idx]
                box_coords = box_data.xyxy[0].tolist() # [x1, y1, x2, y2]
                box_conf = box_data.conf[0].item()
                box_class_id = int(box_data.cls[0].item())
                
                # Skip if we already have a more confident detection for this class
                if box_class_id in best_poses_by_class and best_poses_by_class[box_class_id]['box_confidence'] >= box_conf:
                    continue
                
                box_info = {
                    "x1": round(box_coords[0], 2),
                    "y1": round(box_coords[1], 2),
                    "x2": round(box_coords[2], 2),
                    "y2": round(box_coords[3], 2),
                    "confidence": round(box_conf, 4),
                    "class_name": model_obj.names.get(box_class_id, 'unknown')
                }

                # --- Keypoint Data ---
                keypoints_xy = result.keypoints.xy[pose_idx]
                keypoints_conf = result.keypoints.conf[pose_idx] if result.keypoints.conf is not None else [0] * len(keypoints_xy)
                
                # Get the correct keypoint names for this class
                class_keypoint_names = keypoint_names.get(box_class_id, [])
                expected_keypoints = len(class_keypoint_names)
                
                keypoints_list = []
                confidences = []
                for kp_idx, (x, y) in enumerate(keypoints_xy):
                    # Skip keypoints beyond what this class should have
                    if kp_idx >= expected_keypoints:
                        continue
                        
                    conf = keypoints_conf[kp_idx].item()
                    confidences.append(conf)
                    keypoint_data = {
                        'name': class_keypoint_names[kp_idx],
                        'x': round(x.item(), 2),
                        'y': round(y.item(), 2),
                        'confidence': round(conf, 4)
                    }
                    keypoints_list.append(keypoint_data)
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                if avg_confidence > highest_avg_confidence:
                    highest_avg_confidence = avg_confidence

                # Store this as the best pose for this class
                best_poses_by_class[box_class_id] = {
                    'box_confidence': box_conf,
                    'pose_data': {
                        "box": box_info,
                        "keypoints": keypoints_list
                    }
                }

            # Extract only the pose data from the best poses
            all_poses_data = [pose_info['pose_data'] for pose_info in best_poses_by_class.values()]
            
            infer_label_json = json.dumps({"poses": all_poses_data}, indent=2)
            final_confidence = highest_avg_confidence
            
            result_summary = f"Detected {num_poses} pose(s) with confidence {final_confidence:.4f}."
            update_inference_record(conn, server_id, model_name, infer_label_json, final_confidence)

        else:
            infer_label_json = json.dumps({"poses": []})
            result_summary = "No poses detected."
            update_inference_record(conn, server_id, model_name, infer_label_json, 0.0)
        
        print(f"ID {server_id}: {result_summary}")

def main():
    """Main function to run the inference loop."""
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print("Error: PG_PASS not found in .env file or environment variables.")
        return

    # --- Hardware Check ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware check: PyTorch is using device: {device}")
    if device == 'cuda':
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: CUDA-enabled GPU not found. Falling back to CPU. Inference will be significantly slower.")
        print("  (To enable GPU, ensure NVIDIA drivers and CUDA are installed and that your PyTorch version supports CUDA.)")

    print(f"\nLoading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        # The model will automatically move to the detected device (GPU or CPU)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    conn = None
    total_processed = 0
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="base",
            user="postgres",
            password=db_password,
            cursor_factory=DictCursor
        )
        print("Database connection established.")

        while True:
            print(f"\nFetching new batch of up to {BATCH_SIZE} unprocessed records...")
            batch_data = get_unprocessed_inferences(conn, BATCH_SIZE)

            if not batch_data:
                print("No more unprocessed records found. All work is done.")
                break
            
            print(f"Found {len(batch_data)} records to process in this batch.")
            current_batch_pil_images = []
            current_batch_ids = []

            for record in batch_data:
                server_id = record['server_infer_id']
                mime_type = record.get('mime_type')
                input_data_bytes = record.get('input_data')

                if mime_type and input_data_bytes and mime_type.startswith('image/'):
                    try:
                        # --- FIX: Two-step correction for rotation AND distortion ---
                        image_stream = io.BytesIO(input_data_bytes)
                        unprocessed_image = Image.open(image_stream).convert("RGB")
                        
                        # Step 1: Resize to undo distortion
                        undistorted_dims = (unprocessed_image.height, unprocessed_image.width)
                        resized_image = unprocessed_image.resize(undistorted_dims)

                        # Step 2: Rotate to correct orientation
                        pil_image = resized_image.transpose(Image.Transpose.ROTATE_270)
                        
                        current_batch_pil_images.append(pil_image)
                        current_batch_ids.append(server_id)
                    except Exception as e:
                        print(f"ID {server_id}: Error preparing image, skipping. Error: {e}")
                else:
                    print(f"ID {server_id}: No valid image data, skipping.")

            if not current_batch_pil_images:
                print("No valid images found in the fetched batch. Committing any previous changes and fetching next batch.")
                conn.commit()
                continue
            
            print(f"Performing inference on batch of {len(current_batch_pil_images)} images...")
            try:
                start_time = time.perf_counter()
                batch_results = model(current_batch_pil_images, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False)
                end_time = time.perf_counter()
                
                total_time = end_time - start_time
                time_per_image = total_time / len(current_batch_pil_images) if current_batch_pil_images else 0

                print(f"Inference complete in {total_time:.4f}s ({time_per_image:.4f}s/image). Processing and updating results...")
                process_inference_results(conn, current_batch_ids, batch_results, model)
                total_processed += len(current_batch_ids)

            except Exception as e:
                print(f"Error during YOLO inference or results processing: {e}")
            finally:
                conn.commit()
                print("Batch processed and database changes committed.")

    except (Exception, psycopg2.Error) as error:
        print(f"A critical error occurred: {error}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")
        print(f"Total records processed in this run: {total_processed}")

if __name__ == "__main__":
    main()

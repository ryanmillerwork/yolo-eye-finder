import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from typing import List, Any
import io # For in-memory byte streams
import json # To format results for the database
from PIL import Image # For handling image data; pip install Pillow
from ultralytics import YOLO # For YOLO model inference; pip install ultralytics

def get_inferences_by_ids(server_infer_ids: List[int]) -> List[Any]:
    """
    Connects to the PostgreSQL database and retrieves records
    from the server_inference table by a list of server_infer_ids.
    Records are returned as dictionaries (DictRow).

    Args:
        server_infer_ids: A list of IDs of the inference records to retrieve.

    Returns:
        A list of DictRow objects, where each object represents a row.
        Returns an empty list if no records are found or in case of an error.
    """
    load_dotenv()  # Load environment variables from .env file

    if not server_infer_ids:
        print("No server_infer_ids provided.")
        return []

    db_host = "localhost"
    db_name = "base"
    db_user = "postgres"
    db_password = os.getenv("PG_PASS")

    if not db_password:
        print("Error: PG_PASS not found in .env file or environment variables.")
        return []

    conn = None
    records = []
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            cursor_factory=DictCursor
        )

        # Open a cursor to perform database operations
        cur = conn.cursor()

        query = "SELECT * FROM server_inference WHERE server_infer_id = ANY(%s)"
        cur.execute(query, (server_infer_ids,))

        records = cur.fetchall()

        return records

    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL or executing query: {error}")
        return []
    finally:
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

def get_most_recent_inferences(limit: int) -> List[Any]:
    """
    Connects to the PostgreSQL database and retrieves the most recent records
    from the server_inference table, ordered by server_infer_id descending.
    Records are returned as dictionaries (DictRow).

    Args:
        limit: The maximum number of records to retrieve.

    Returns:
        A list of DictRow objects, where each object represents a row.
        Returns an empty list if no records are found or in case of an error.
    """
    load_dotenv()  # Load environment variables from .env file

    db_host = "localhost"
    db_name = "base"
    db_user = "postgres"
    db_password = os.getenv("PG_PASS")

    if not db_password:
        print("Error: PG_PASS not found in .env file or environment variables.")
        return []

    conn = None
    records = []
    try:
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            cursor_factory=DictCursor
        )
        cur = conn.cursor()
        query = "SELECT * FROM server_inference ORDER BY server_infer_id DESC LIMIT %s"
        cur.execute(query, (limit,))
        records = cur.fetchall()
        return records

    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL or executing query: {error}")
        return []
    finally:
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

def update_inference_record(server_id: int, model_name: str, infer_label: str, confidence: float):
    """
    Connects to the PostgreSQL database and updates a record in the
    server_inference table with the results of an inference.
    """
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print(f"Error updating record {server_id}: PG_PASS not found.")
        return

    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="base",
            user="postgres",
            password=db_password
        )
        cur = conn.cursor()
        
        sql = """
            UPDATE server_inference
            SET model_file = %s, infer_label = %s, confidence = %s
            WHERE server_infer_id = %s
        """
        cur.execute(sql, (model_name, infer_label, confidence, server_id))
        conn.commit()
        # print(f"  Successfully updated record for server_infer_id: {server_id}") # Silenced for cleaner logs

    except (Exception, psycopg2.Error) as error:
        # Keep error logging
        print(f"Error while updating record {server_id}: {error}")
    finally:
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()

BATCH_SIZE = 64 # Optimal batch size determined from benchmarking
MODEL_PATH = "./models/HB-eyes-400_small.pt"
IMAGE_SIZE = (192, 128) # imgsz used during training (width, height)
CONF_THRESHOLD = 0.1 # Lowered confidence threshold for detection. Default is 0.25

def process_inference_results(batch_ids, batch_results, model_obj):
    """
    Processes inference results, formats them for the DB, calls the
    update function, and prints a concise one-line summary.
    """
    model_name = os.path.basename(MODEL_PATH)

    for i, result in enumerate(batch_results):
        server_id = batch_ids[i]
        result_summary = ""

        if result.keypoints is not None and result.keypoints.data.numel() > 0:
            num_poses = result.keypoints.shape[0]
            
            all_poses_data = []
            highest_avg_confidence = 0.0

            for pose_idx in range(num_poses):
                keypoints_xy = result.keypoints.xy[pose_idx]
                keypoints_conf = result.keypoints.conf[pose_idx] if result.keypoints.conf is not None else [0] * len(keypoints_xy)
                
                keypoints_list = []
                confidences = []
                for kp_idx, (x, y) in enumerate(keypoints_xy):
                    conf = keypoints_conf[kp_idx].item()
                    confidences.append(conf)
                    keypoint_data = {
                        'kp_index': kp_idx,
                        'name': model_obj.names.get(kp_idx, 'unknown'),
                        'x': round(x.item(), 2),
                        'y': round(y.item(), 2),
                        'confidence': round(conf, 4)
                    }
                    keypoints_list.append(keypoint_data)
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                if avg_confidence > highest_avg_confidence:
                    highest_avg_confidence = avg_confidence

                all_poses_data.append({
                    "pose_index": pose_idx,
                    "avg_confidence": round(avg_confidence, 4),
                    "keypoints": keypoints_list
                })

            infer_label_json = json.dumps({"poses": all_poses_data}, indent=2)
            final_confidence = highest_avg_confidence
            
            result_summary = f"Detected {num_poses} pose(s) with confidence {final_confidence:.4f}. DB updated."
            update_inference_record(server_id, model_name, infer_label_json, final_confidence)

        else:
            infer_label_json = json.dumps({"poses": []})
            result_summary = "No poses detected. DB updated."
            update_inference_record(server_id, model_name, infer_label_json, 0.0)
        
        print(f"ID {server_id}: {result_summary}")

if __name__ == "__main__":
    # Ensure ultralytics is installed: pip install ultralytics
    # Ensure Pillow is installed: pip install Pillow

    print(f"Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the model path is correct and ultralytics is installed correctly.")
        exit()

    print(f"Attempting to fetch the {BATCH_SIZE} most recent records for inference...")
    all_inference_data = get_most_recent_inferences(BATCH_SIZE)

    if not all_inference_data:
        print(f"No data found or an error occurred during DB fetch.")
    else:
        print(f"Found {len(all_inference_data)} record(s) to process.")
        all_inference_data.reverse()

        current_batch_pil_images = []
        current_batch_ids = []

        for i, record in enumerate(all_inference_data):
            server_id = record['server_infer_id']
            # print(f"Processing record ID: {server_id} ({i+1}/{len(all_inference_data)})") # Silenced for cleaner logs

            mime_type = record.get('mime_type')
            input_data_bytes = record.get('input_data')

            if mime_type and input_data_bytes and mime_type.startswith('image/'):
                try:
                    image_stream = io.BytesIO(input_data_bytes)
                    img = Image.open(image_stream).convert("RGB") 
                    current_batch_pil_images.append(img)
                    current_batch_ids.append(server_id)
                    # print(f"  Image from record {server_id} added to batch. Batch size: {len(current_batch_pil_images)}/{BATCH_SIZE}") # Silenced

                except Exception as e:
                    print(f"  Error processing image data for record ID {server_id}: {e}")
                    continue
            else:
                if not mime_type or not mime_type.startswith('image/'):
                    print(f"  Record ID {server_id}: Mime type {mime_type if mime_type else 'N/A'} is not an image. Skipping.")
                if not input_data_bytes:
                    print(f"  Record ID {server_id}: Input data is missing. Skipping.")
                continue

            # Perform inference if batch is full or it's the last record
            if len(current_batch_pil_images) == BATCH_SIZE or (i == len(all_inference_data) - 1 and len(current_batch_pil_images) > 0):
                if current_batch_pil_images: # Ensure batch is not empty
                    print(f"\nPerforming inference on batch of {len(current_batch_pil_images)} images...")
                    try:
                        batch_results = model(current_batch_pil_images, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False)
                        print(f"Inference complete. Processing and updating results...")
                        
                        process_inference_results(current_batch_ids, batch_results, model)

                    except Exception as e:
                        print(f"Error during YOLO inference or results processing: {e}")
                    finally:
                        # Clear the batch for the next set of images
                        current_batch_pil_images = []
                        current_batch_ids = []

        print("\nAll fetched records have been processed.")

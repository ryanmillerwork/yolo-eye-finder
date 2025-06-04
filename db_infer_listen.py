import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from typing import List, Any
import io # For in-memory byte streams
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

BATCH_SIZE = 8 # Adjust as needed based on your system memory and performance
MODEL_PATH = "./models/HB-eyes-400_small.pt"
IMAGE_SIZE = (192, 128) # imgsz used during training (width, height)

def process_inference_results(batch_ids, batch_results):
    """Placeholder function to process and print inference results."""
    for i, result in enumerate(batch_results):
        server_id = batch_ids[i]
        print(f"  Results for Record ID {server_id}:")
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            num_poses = len(result.keypoints.xy)
            print(f"    Detected {num_poses} pose(s).")
            # Example: Accessing keypoints for the first detected pose
            # keypoints_data = result.keypoints[0].data # Tensor of keypoints
            # print(f"    Keypoints for first pose (raw tensor): {keypoints_data}")
            # For more detailed keypoint processing, refer to Ultralytics documentation
            # on the structure of result.keypoints
        else:
            print("    No poses detected.")
        # print(result.summary()) # You can print a summary of detections

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

    test_ids = [185355]  # Example ID, replace or expand as needed
    # To test batching, ensure test_ids contains more IDs than BATCH_SIZE, or adjust BATCH_SIZE.
    # e.g., test_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"Attempting to fetch records for IDs: {test_ids}")
    all_inference_data = get_inferences_by_ids(test_ids)

    if not all_inference_data:
        print(f"No data found for server_infer_ids {test_ids} or an error occurred during DB fetch.")
    else:
        print(f"Found {len(all_inference_data)} record(s) in total.")

        current_batch_pil_images = []
        current_batch_ids = []

        for i, record in enumerate(all_inference_data):
            server_id = record['server_infer_id']
            print(f"Processing record ID: {server_id} ({i+1}/{len(all_inference_data)})")

            mime_type = record.get('mime_type')
            input_data_bytes = record.get('input_data')

            if mime_type and input_data_bytes and mime_type.startswith('image/'):
                try:
                    image_stream = io.BytesIO(input_data_bytes)
                    img = Image.open(image_stream)
                    # It's good practice to convert to RGB if not already, 
                    # as many models expect 3 channels.
                    img = img.convert("RGB") 
                    current_batch_pil_images.append(img)
                    current_batch_ids.append(server_id)
                    print(f"  Image from record {server_id} added to batch. Batch size: {len(current_batch_pil_images)}/{BATCH_SIZE}")

                except IOError as e:
                    print(f"  Error processing image data for record ID {server_id}: {e}")
                    continue # Skip to next record
                except Exception as e:
                    print(f"  An unexpected error occurred while preparing image for record ID {server_id}: {e}")
                    continue # Skip to next record
            else:
                if not mime_type or not mime_type.startswith('image/'):
                    print(f"  Record ID {server_id}: Mime type {mime_type if mime_type else 'N/A'} is not an image. Skipping.")
                if not input_data_bytes:
                    print(f"  Record ID {server_id}: Input data is missing. Skipping.")
                continue # Skip to next record

            # Perform inference if batch is full or it's the last record
            if len(current_batch_pil_images) == BATCH_SIZE or (i == len(all_inference_data) - 1 and len(current_batch_pil_images) > 0):
                print(f"Performing inference on batch of {len(current_batch_pil_images)} images...")
                try:
                    # Ultralytics will handle resizing to the model's expected input size if imgsz is specified.
                    # Your model was trained with imgsz=192,128
                    batch_results = model(current_batch_pil_images, imgsz=IMAGE_SIZE, verbose=False) # verbose=False to reduce console output
                    print(f"Inference complete for batch.")
                    
                    process_inference_results(current_batch_ids, batch_results)

                    # Placeholder for storing results back to DB
                    # for res_idx, result_obj in enumerate(batch_results):
                    #     original_server_id = current_batch_ids[res_idx]
                    #     # extracted_keypoints = ... # extract relevant data from result_obj.keypoints
                    #     # store_inference_results_to_db(original_server_id, extracted_keypoints)
                    # print(f"  (Placeholder) Inference results for batch would be stored.")

                except Exception as e:
                    print(f"Error during YOLO inference or results processing: {e}")
                finally:
                    # Clear the batch for the next set of images
                    current_batch_pil_images = []
                    current_batch_ids = []
        
        if len(current_batch_pil_images) > 0:
            print(f"Performing inference on the final batch of {len(current_batch_pil_images)} images...")
            try:
                batch_results = model(current_batch_pil_images, imgsz=IMAGE_SIZE, verbose=False)
                print(f"Inference complete for final batch.")
                process_inference_results(current_batch_ids, batch_results)
                # ... (placeholder for storing results as above)
            except Exception as e:
                print(f"Error during YOLO inference or results processing for final batch: {e}")

        print("All records processed.")

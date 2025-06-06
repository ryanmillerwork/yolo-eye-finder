import os
import sys
import argparse
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import io
from PIL import Image
import numpy as np # For robust image rotation
from ultralytics import YOLO
import cv2  # OpenCV is used for color conversion

# --- Configuration ---
MODEL_PATH = "./models/HB-eyes-400_small.pt"
CONF_THRESHOLD = 0.1  # Use a low threshold for debugging to see all possible detections
OUTPUT_FILENAME = "infer.png"

def get_inference_by_id(conn, server_id: int):
    """Retrieves a single record from the database by its ID."""
    try:
        with conn.cursor() as cur:
            query = "SELECT * FROM server_inference WHERE server_infer_id = %s"
            cur.execute(query, (server_id,))
            record = cur.fetchone()
            return record
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching record {server_id}: {error}")
        return None

def main():
    """Main function to fetch, infer, and save a labeled image."""
    parser = argparse.ArgumentParser(description="Fetch an image from the database, run YOLO inference, and save a labeled version.")
    parser.add_argument("server_infer_id", type=int, help="The server_infer_id of the record to process.")
    args = parser.parse_args()
    
    server_id_to_process = args.server_infer_id

    # --- Load Environment and Model ---
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print("Error: PG_PASS not found in .env file.", file=sys.stderr)
        return 1

    print(f"Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}", file=sys.stderr)
        return 1

    # --- Database and Inference ---
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost", database="base", user="postgres", password=db_password, cursor_factory=DictCursor
        )
        print(f"Database connection established. Fetching record for ID: {server_id_to_process}")
        
        record = get_inference_by_id(conn, server_id_to_process)

        if not record:
            print(f"No record found for server_infer_id: {server_id_to_process}", file=sys.stderr)
            return 1

        input_data_bytes = record.get('input_data')
        if not input_data_bytes:
            print(f"Record {server_id_to_process} has no image data (input_data is null).", file=sys.stderr)
            return 1

        print("Image data found. Preparing for inference...")
        try:
            image_stream = io.BytesIO(input_data_bytes)
            unprocessed_image = Image.open(image_stream).convert("RGB")
            print(f"  Dimensions as read from file metadata: {unprocessed_image.width}W x {unprocessed_image.height}H")
            
            # --- FIX: Two-step correction for rotation AND distortion ---
            # Step 1: Resize the image to its "pre-distortion" aspect ratio.
            # The distorted dimensions are (W, H), the original content dimensions were likely (H, W).
            print("  Step 1: Resizing to undo distortion...")
            undistorted_dims = (unprocessed_image.height, unprocessed_image.width)
            resized_image = unprocessed_image.resize(undistorted_dims)
            print(f"  Dimensions after resize: {resized_image.width}W x {resized_image.height}H")

            # Step 2: Rotate the now correctly-proportioned image to its proper orientation.
            # A 90-degree counter-clockwise rotation is needed to create the final portrait image.
            print("  Step 2: Rotating 90 degrees counter-clockwise to correct final orientation...")
            pil_image = resized_image.transpose(Image.Transpose.ROTATE_90)
            print(f"  Dimensions after final rotation: {pil_image.width}W x {pil_image.height}H")

        except Exception as e:
            print(f"Failed to load or process image from database bytes for ID {server_id_to_process}: {e}", file=sys.stderr)
            return 1
        
        # --- Run Inference and Plot Results ---
        print("Running inference...")
        results = model(pil_image, conf=CONF_THRESHOLD, verbose=False)
        
        if not results:
            print("Inference did not return any results object.", file=sys.stderr)
            return 1
            
        result = results[0] # Get the result for the first (and only) image
        print(f"Inference complete. Detected {len(result.keypoints) if result.keypoints else 0} poses.")
        
        # The plot() method returns a NumPy array (BGR format) with detections drawn on it
        annotated_image_bgr = result.plot() 
        
        # Convert from BGR (OpenCV's default) to RGB (Pillow's default)
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Create a Pillow Image from the annotated NumPy array
        final_image = Image.fromarray(annotated_image_rgb)
        
        # Log final dimensions before saving
        print(f"  Final image dimensions before saving: {final_image.width}W x {final_image.height}H")

        # Save the final image
        final_image.save(OUTPUT_FILENAME)
        print(f"Successfully saved annotated image to '{OUTPUT_FILENAME}'")

    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return 1
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    sys.exit(main()) 
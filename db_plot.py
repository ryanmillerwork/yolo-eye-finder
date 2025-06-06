import os
import sys
import argparse
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import io
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2  # For drawing

def get_record_by_id(conn, server_id: int):
    """Retrieves a single record from the database by its ID."""
    try:
        with conn.cursor() as cur:
            query = "SELECT input_data, infer_label FROM server_inference WHERE server_infer_id = %s"
            cur.execute(query, (server_id,))
            record = cur.fetchone()
            return record
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching record {server_id}: {error}", file=sys.stderr)
        return None

def draw_annotations(image, infer_label_json: str):
    """Draws keypoints from a JSON label onto an image."""
    try:
        data = json.loads(infer_label_json)
    except (json.JSONDecodeError, TypeError):
        print("Warning: Could not parse infer_label JSON or label is null. Returning original image.")
        return image

    # Convert PIL Image to NumPy array for OpenCV
    # OpenCV uses BGR, Pillow uses RGB, so we need to convert color formats
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    poses = data.get("poses", [])
    if not poses:
        print("No poses found in the label data.")
        return image

    for pose in poses:
        keypoints = pose.get("keypoints", [])
        for kp in keypoints:
            # Ensure keypoint has coordinates and confidence > 0
            if 'x' in kp and 'y' in kp and kp.get('confidence', 0) > 0:
                x, y = int(kp['x']), int(kp['y'])
                name = kp.get('name', 'unknown')
                conf = kp.get('confidence', 0)

                # Draw a circle for the keypoint
                # Color BGR, so (0, 255, 0) is bright green
                cv2.circle(cv2_image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                
                # Draw the label text
                # Color BGR, so (255, 255, 255) is white
                label = f"{name} ({conf:.2f})"
                cv2.putText(cv2_image, label, (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Convert back to RGB for saving with Pillow
    annotated_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_image_rgb)


def main():
    """Main function to fetch, parse, draw, and save."""
    parser = argparse.ArgumentParser(description="Fetch a record from the database, draw its stored infer_label, and save the image.")
    parser.add_argument("server_infer_id", type=int, help="The server_infer_id of the record to process.")
    parser.add_argument("output_path", type=str, help="The file path to save the annotated image (e.g., ./output.png).")
    args = parser.parse_args()

    # --- Load Environment ---
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print("Error: PG_PASS not found in .env file.", file=sys.stderr)
        return 1

    # --- Database and Image Processing ---
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost", database="base", user="postgres", password=db_password, cursor_factory=DictCursor
        )
        record = get_record_by_id(conn, args.server_infer_id)

        if not record:
            print(f"No record found for server_infer_id: {args.server_infer_id}", file=sys.stderr)
            return 1

        input_data_bytes = record.get('input_data')
        infer_label = record.get('infer_label')

        if not input_data_bytes:
            print(f"Record {args.server_infer_id} has no image data.", file=sys.stderr)
            return 1
            
        # --- Image Correction ---
        image_stream = io.BytesIO(input_data_bytes)
        unprocessed_image = Image.open(image_stream).convert("RGB")
        
        # Step 1: Resize to undo distortion
        undistorted_dims = (unprocessed_image.height, unprocessed_image.width)
        resized_image = unprocessed_image.resize(undistorted_dims)
        # Step 2: Rotate to correct orientation
        corrected_image = resized_image.transpose(Image.Transpose.ROTATE_270)
        print("Image loaded and orientation corrected.")

        # --- Draw Annotations ---
        print("Drawing annotations from database label...")
        annotated_image = draw_annotations(corrected_image, infer_label)

        # --- Save the final image ---
        annotated_image.save(args.output_path)
        print(f"Successfully saved annotated image to '{args.output_path}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return 1
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    sys.exit(main()) 
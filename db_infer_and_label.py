import os
import sys
import argparse
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import io
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np # For robust image rotation
from ultralytics import YOLO
import cv2  # OpenCV is used for color conversion

# --- Configuration ---
MODEL_PATH = "./models/HB-eyes-400_small.pt"
CONF_THRESHOLD = 0.1  # Use a low threshold for debugging to see all possible detections

# Output directories for different modes
BASE_OUTPUT_DIR = "/mnt/qpcs/db/db_infer_and_label"
OUTPUT_DIRS = {
    'inference': os.path.join(BASE_OUTPUT_DIR, "inference"),
    'save-only': os.path.join(BASE_OUTPUT_DIR, "save-only"),
    'plot-stored': os.path.join(BASE_OUTPUT_DIR, "plot-stored")
}

# Colors for plotting stored labels
COLORS = {
    'face_box': 'orange',
    'juice_tube_box': 'green',
    'left_pupil': 'red',
    'right_pupil': 'green',
    'nose_bridge': '#8A2BE2',  # blue-purple
    'spout_top': '#8B008B',    # red-purple
    'spout_bottom': 'lightblue'
}

def parse_id_specification(id_args):
    """
    Parse ID specification which can be:
    1. Manual list: [123, 456, 789]
    2. Range syntax: ["123456:20:134"] (start:step:count)
    3. Mixed: [123, 456, "789:5:10"]
    
    Args:
        id_args: List of arguments that can be integers or range strings
        
    Returns:
        List of integers (server_infer_ids)
    """
    result_ids = []
    
    for arg in id_args:
        arg_str = str(arg)
        
        # Check if it's a range specification (contains colons)
        if ':' in arg_str:
            try:
                parts = arg_str.split(':')
                if len(parts) != 3:
                    raise ValueError(f"Range syntax must be 'start:step:count', got: {arg_str}")
                
                start, step, count = map(int, parts)
                
                if count <= 0:
                    raise ValueError(f"Count must be positive, got: {count}")
                if step <= 0:
                    raise ValueError(f"Step must be positive, got: {step}")
                
                # Generate the sequence
                generated_ids = [start + i * step for i in range(count)]
                result_ids.extend(generated_ids)
                print(f"Generated {count} IDs from {start} with step {step}: {start} to {generated_ids[-1]}")
                
            except ValueError as e:
                print(f"Error parsing range '{arg_str}': {e}", file=sys.stderr)
                raise
        else:
            # It's a single integer
            try:
                result_ids.append(int(arg_str))
            except ValueError:
                print(f"Invalid ID specification: {arg_str}", file=sys.stderr)
                raise
    
    return result_ids

def ensure_output_dir_exists(mode):
    """Ensure the output directory for the given mode exists."""
    output_dir = OUTPUT_DIRS.get(mode)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        raise ValueError(f"Unknown mode: {mode}")

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

def correct_image_orientation(image):
    """
    Corrects image distortion and orientation in two steps:
    1. Resizes to undo distortion by swapping width/height dimensions
    2. Rotates 90 degrees clockwise to correct orientation
    
    Args:
        image (PIL.Image): The input image to correct
        
    Returns:
        PIL.Image: The corrected image
    """
    print("  Step 1: Resizing to undo distortion...")
    undistorted_dims = (image.height, image.width)
    resized_image = image.resize(undistorted_dims)
    print(f"  Dimensions after resize: {resized_image.width}W x {resized_image.height}H")

    print("  Step 2: Rotating 90 degrees clockwise to correct final orientation...")
    corrected_image = resized_image.transpose(Image.Transpose.ROTATE_270)
    print(f"  Dimensions after final rotation: {corrected_image.width}W x {corrected_image.height}H")
    
    return corrected_image

def draw_stored_labels(image, labels_json, confidence_threshold=0.5):
    """
    Draw stored labels (boxes and keypoints) on the image.
    
    Args:
        image (PIL.Image): The image to draw on
        labels_json (str or dict): JSON string or pre-parsed dict containing the stored labels
        confidence_threshold (float): Minimum confidence to draw elements
        
    Returns:
        PIL.Image: Image with labels drawn
    """
    print("--- Inside draw_stored_labels ---")
    
    # --- FONT LOADING ---
    try:
        # Using a smaller font size
        font = ImageFont.truetype("DejaVuSans.ttf", size=7)
        text_y_offset = 9
    except IOError:
        print("Default font not found, using PIL's default. Text will not be resized.")
        font = ImageFont.load_default()
        text_y_offset = 15
    
    labels_data = None
    if isinstance(labels_json, str):
        print(f"  labels_json is a string. Type: {type(labels_json)}")
        try:
            labels_data = json.loads(labels_json)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  Error parsing labels JSON string: {e}")
            return image
    elif isinstance(labels_json, dict):
        print(f"  labels_json is already a dict. Type: {type(labels_json)}")
        labels_data = labels_json
    else:
        print(f"  labels_json is of unexpected type: {type(labels_json)}. Content: {labels_json}")
        return image

    if not labels_data:
        print("  Could not load labels data.")
        return image
    
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    poses = labels_data.get('poses', [])
    print(f"  Found {len(poses)} poses in labels.")
    
    items_drawn = 0

    for i, pose in enumerate(poses):
        print(f"\n  Processing Pose #{i+1}")
        # Draw bounding box
        box = pose.get('box', {})
        box_confidence = box.get('confidence', 0)
        print(f"    Box confidence: {box_confidence:.4f} (threshold: >{confidence_threshold})")
        if box_confidence > confidence_threshold:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            class_name = box.get('class_name', '')
            
            # Choose color based on class
            if class_name == 'face':
                box_color = COLORS['face_box']
            elif class_name == 'juice_tube':
                box_color = COLORS['juice_tube_box']
            else:
                box_color = 'white'  # fallback
            
            # Draw bounding box rectangle with thinner lines
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
            
            # Draw class label and confidence
            label_text = f"{class_name} {box_confidence:.3f}"
            draw.text((x1, y1 - text_y_offset), label_text, fill=box_color, font=font)
            items_drawn += 1
            print(f"    -> DRAWN Box: {class_name} at [{x1},{y1},{x2},{y2}]")
        
        # Draw keypoints
        keypoints = pose.get('keypoints', [])
        print(f"    Found {len(keypoints)} keypoints.")
        for keypoint in keypoints:
            kp_confidence = keypoint.get('confidence', 0)
            name = keypoint.get('name', 'N/A')
            print(f"      Keypoint '{name}' confidence: {kp_confidence:.4f} (threshold: >{confidence_threshold})")
            if kp_confidence > confidence_threshold:
                x, y = keypoint['x'], keypoint['y']
                
                # Skip keypoints at (0,0) which usually means no detection
                if x == 0 and y == 0:
                    print("      -> SKIPPED Keypoint: (0,0)")
                    continue
                
                # Choose color based on keypoint name
                point_color = COLORS.get(name, 'white')
                
                # Draw keypoint as a smaller circle with no outline
                radius = 1
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=point_color)
                
                # Draw keypoint label
                label_text = f"{name} {kp_confidence:.3f}"
                draw.text((x + 5, y - text_y_offset), label_text, fill=point_color, font=font)
                items_drawn += 1
                print(f"      -> DRAWN Keypoint: {name} at ({x}, {y})")
    
    print(f"\n  Total items drawn: {items_drawn}")
    print("--- Exiting draw_stored_labels ---")
    return annotated_image

def process_single_image(conn, model, server_id, mode='inference'):
    """
    Process a single image according to the specified mode.
    
    Args:
        conn: Database connection
        model: YOLO model instance (can be None for non-inference modes)
        server_id (int): The server_infer_id to process
        mode (str): Processing mode - 'inference', 'save-only', or 'plot-stored'
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"\n--- Processing ID: {server_id} ---")
        record = get_inference_by_id(conn, server_id)

        if not record:
            print(f"No record found for server_infer_id: {server_id}", file=sys.stderr)
            return False

        input_data_bytes = record.get('input_data')
        if not input_data_bytes:
            print(f"Record {server_id} has no image data (input_data is null).", file=sys.stderr)
            return False

        print("Image data found. Preparing for inference...")
        try:
            image_stream = io.BytesIO(input_data_bytes)
            unprocessed_image = Image.open(image_stream).convert("RGB")
            print(f"  Dimensions as read from file metadata: {unprocessed_image.width}W x {unprocessed_image.height}H")
            
            # Apply image correction (distortion and orientation)
            pil_image = correct_image_orientation(unprocessed_image)

        except Exception as e:
            print(f"Failed to load or process image from database bytes for ID {server_id}: {e}", file=sys.stderr)
            return False
        
        # --- Process according to mode ---
        # Ensure output directory exists and get the full path
        try:
            output_dir = ensure_output_dir_exists(mode)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return False
            
        if mode == 'inference':
            # Run YOLO inference and save labeled image
            print("Running inference...")
            if model is None:
                print("Error: Model is required for inference mode", file=sys.stderr)
                return False
                
            results = model(pil_image, conf=CONF_THRESHOLD, verbose=False)
            
            if not results:
                print("Inference did not return any results object.", file=sys.stderr)
                return False
                
            result = results[0] # Get the result for the first (and only) image
            print(f"Inference complete. Detected {len(result.keypoints) if result.keypoints else 0} poses.")
            
            # The plot() method returns a NumPy array (BGR format) with detections drawn on it
            annotated_image_bgr = result.plot() 
            
            # Convert from BGR (OpenCV's default) to RGB (Pillow's default)
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Create a Pillow Image from the annotated NumPy array
            final_image = Image.fromarray(annotated_image_rgb)
            
            output_filename = os.path.join(output_dir, f"infer_{server_id}.png")
            
        elif mode == 'save-only':
            # Just save the corrected image without any inference
            print("Saving corrected image without inference...")
            final_image = pil_image
            output_filename = os.path.join(output_dir, f"image_{server_id}.png")
            
        elif mode == 'plot-stored':
            # Retrieve stored labels from database and plot them
            print("Plotting stored labels from database...")
            
            stored_labels = record.get('infer_label')
            if not stored_labels:
                print(f"Record {server_id} has no stored labels (infer_label is null).", file=sys.stderr)
                return False
            
            # Draw the labels on the image
            final_image = draw_stored_labels(pil_image, stored_labels, confidence_threshold=0.5)
            
            output_filename = os.path.join(output_dir, f"plot_{server_id}.png")
            
        else:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            return False
        
        # Log final dimensions before saving
        print(f"  Final image dimensions before saving: {final_image.width}W x {final_image.height}H")

        # Save the final image
        final_image.save(output_filename)
        print(f"Successfully saved image to '{output_filename}'")
        return True
        
    except Exception as e:
        print(f"Error processing ID {server_id}: {e}", file=sys.stderr)
        return False

def main():
    """Main function to fetch, infer, and save a labeled image."""
    parser = argparse.ArgumentParser(
        description="Fetch images from the database and process them according to the specified mode.",
        epilog="""
ID Specification Examples:
  Manual list:        123 456 789
  Range syntax:       123456:20:134  (start:step:count - start at 123456, every 20th ID, for 134 total)
  Mixed:              123 456 100:5:10
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("id_specs", nargs='+', 
                       help="Server infer IDs or range specifications. Use 'start:step:count' for ranges (e.g., '123456:20:134')")
    parser.add_argument("--mode", choices=['inference', 'save-only', 'plot-stored'], default='inference',
                       help="Processing mode: 'inference' (run YOLO and save labeled image), 'save-only' (just save corrected image), 'plot-stored' (plot stored labels from DB)")
    args = parser.parse_args()
    
    # Parse the ID specifications
    try:
        server_ids_to_process = parse_id_specification(args.id_specs)
    except (ValueError, TypeError) as e:
        print(f"Error parsing ID specifications: {e}", file=sys.stderr)
        return 1
    
    processing_mode = args.mode
    
    print(f"Total IDs to process: {len(server_ids_to_process)}")
    if len(server_ids_to_process) > 10:
        print(f"First 10 IDs: {server_ids_to_process[:10]}")
        print(f"Last 10 IDs: {server_ids_to_process[-10:]}")
    else:
        print(f"IDs: {server_ids_to_process}")

    # --- Load Environment and Model ---
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print("Error: PG_PASS not found in .env file.", file=sys.stderr)
        return 1

    # Only load model if we need it for inference
    model = None
    if processing_mode == 'inference':
        print(f"Loading YOLO model from: {MODEL_PATH}")
        try:
            model = YOLO(MODEL_PATH)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}", file=sys.stderr)
            return 1
    else:
        print(f"Mode: {processing_mode} - Skipping model loading")

    # --- Database and Inference ---
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost", database="base", user="postgres", password=db_password, cursor_factory=DictCursor
        )
        print(f"Database connection established. Processing {len(server_ids_to_process)} record(s)")
        
        # Process each ID
        successful_count = 0
        failed_count = 0
        
        for server_id in server_ids_to_process:
            success = process_single_image(conn, model, server_id, processing_mode)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        
        # Summary
        print(f"\n--- Processing Complete ---")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Total: {len(server_ids_to_process)}")
        
        # Return non-zero exit code if any failed
        if failed_count > 0:
            return 1

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
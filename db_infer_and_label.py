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
import cv2  # OpenCV is used for color conversion and video creation

# --- Configuration ---
MODEL_PATH = "./models/HB-eyes-1000_small.pt"
CONF_THRESHOLD = 0.5  # Confidence threshold for drawing detections

# Output directories for different modes
BASE_OUTPUT_DIR = "/mnt/qpcs/db/db_infer_and_label"
OUTPUT_DIRS = {
    'inference': os.path.join(BASE_OUTPUT_DIR, "inference"),
    'save-only': os.path.join(BASE_OUTPUT_DIR, "save-only"),
    'plot-stored': os.path.join(BASE_OUTPUT_DIR, "plot-stored"),
    'trial-video': os.path.join(BASE_OUTPUT_DIR, "trial-videos")
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

def get_inferences_by_trial_id(conn, trial_id: int):
    """Retrieves all records from the database for a given trial ID, ordered by client_time."""
    try:
        with conn.cursor() as cur:
            query = "SELECT * FROM server_inference WHERE server_trial_id = %s ORDER BY client_time"
            cur.execute(query, (trial_id,))
            records = cur.fetchall()
            return records
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching records for trial {trial_id}: {error}")
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

def draw_stored_labels(image, labels_json, confidence_threshold=None):
    """
    Draw stored labels (boxes and keypoints) on the image.
    
    Args:
        image (PIL.Image): The image to draw on
        labels_json (str or dict): JSON string or pre-parsed dict containing the stored labels
        confidence_threshold (float): Minimum confidence to draw elements (uses CONF_THRESHOLD if None)
        
    Returns:
        PIL.Image: Image with labels drawn
    """
    if confidence_threshold is None:
        confidence_threshold = CONF_THRESHOLD
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
    
    # Track drawn elements to ensure only 1 of each type
    drawn_boxes = set()
    drawn_keypoints = set()

    for i, pose in enumerate(poses):
        print(f"\n  Processing Pose #{i+1}")
        # Draw bounding box
        box = pose.get('box', {})
        box_confidence = box.get('confidence', 0)
        class_name = box.get('class_name', '')
        print(f"    Box confidence: {box_confidence:.4f} (threshold: >{confidence_threshold})")
        
        # Skip if below threshold or already drawn this class
        if box_confidence <= confidence_threshold or class_name in drawn_boxes:
            if class_name in drawn_boxes:
                print(f"    -> SKIPPED Box: {class_name} (already drawn)")
            continue
            
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        
        # Choose color based on class
        if class_name == 'face':
            box_color = COLORS['face_box']
        elif class_name == 'juice_tube':
            box_color = COLORS['juice_tube_box']
        else:
            box_color = 'white'  # fallback
        
        # Draw bounding box rectangle with thinner lines
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
        
        drawn_boxes.add(class_name)
        items_drawn += 1
        print(f"    -> DRAWN Box: {class_name} at [{x1},{y1},{x2},{y2}]")
        
        # Draw keypoints
        keypoints = pose.get('keypoints', [])
        print(f"    Found {len(keypoints)} keypoints.")
        for keypoint in keypoints:
            kp_confidence = keypoint.get('confidence', 0)
            name = keypoint.get('name', 'N/A')
            print(f"      Keypoint '{name}' confidence: {kp_confidence:.4f} (threshold: >{confidence_threshold})")
            
            # Skip if below threshold, already drawn, or at (0,0)
            if (kp_confidence <= confidence_threshold or 
                name in drawn_keypoints or
                (keypoint['x'] == 0 and keypoint['y'] == 0)):
                if name in drawn_keypoints:
                    print(f"      -> SKIPPED Keypoint: {name} (already drawn)")
                elif keypoint['x'] == 0 and keypoint['y'] == 0:
                    print("      -> SKIPPED Keypoint: (0,0)")
                continue
            
            x, y = keypoint['x'], keypoint['y']
            
            # Choose color based on keypoint name
            point_color = COLORS.get(name, 'white')
            
            # Draw keypoint as a smaller circle with no outline
            radius = 2
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                       fill=point_color)
            
            drawn_keypoints.add(name)
            items_drawn += 1
            print(f"      -> DRAWN Keypoint: {name} at ({x}, {y})")
    
    print(f"\n  Total items drawn: {items_drawn}")
    print("--- Exiting draw_stored_labels ---")
    return annotated_image

def create_trial_video(processed_images, output_path, fps=30):
    """
    Create a video from a list of processed images.
    
    Args:
        processed_images (list): List of PIL Image objects
        output_path (str): Path where the video should be saved
        fps (int): Frames per second for the output video
        
    Returns:
        bool: True if successful, False if failed
    """
    if not processed_images:
        print("No images to create video from")
        return False
    
    try:
        # Get dimensions from the first image
        first_image = processed_images[0]
        width, height = first_image.size
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating video with {len(processed_images)} frames at {fps} FPS")
        print(f"Video dimensions: {width}x{height}")
        
        for i, pil_image in enumerate(processed_images):
            # Convert PIL image to OpenCV format (BGR)
            rgb_array = np.array(pil_image)
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Write the frame
            video_writer.write(bgr_array)
            
            if (i + 1) % 50 == 0:  # Progress update every 50 frames
                print(f"  Processed {i + 1}/{len(processed_images)} frames")
        
        # Release the video writer
        video_writer.release()
        print(f"Video successfully saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

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
            
            # Use custom drawing function with consistent styling
            final_image = draw_yolo_results(pil_image, result)
            
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
            final_image = draw_stored_labels(pil_image, stored_labels)
            
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

def process_trial_video(conn, model, trial_id, fps=30):
    """
    Process all images for a trial and create a video with inference results.
    
    Args:
        conn: Database connection
        model: YOLO model instance
        trial_id (int): The trial_id to process
        fps (int): Frames per second for the output video
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"\n--- Processing Trial Video: {trial_id} ---")
        
        # Get all inference records for this trial
        records = get_inferences_by_trial_id(conn, trial_id)
        
        if not records:
            print(f"No records found for trial_id: {trial_id}", file=sys.stderr)
            return False
        
        print(f"Found {len(records)} images for trial {trial_id}")
        
        # Ensure output directory exists
        try:
            output_dir = ensure_output_dir_exists('trial-video')
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return False
        
        processed_images = []
        successful_count = 0
        failed_count = 0
        
        for i, record in enumerate(records):
            server_id = record.get('server_infer_id')
            print(f"\nProcessing frame {i+1}/{len(records)} (ID: {server_id})")
            
            input_data_bytes = record.get('input_data')
            if not input_data_bytes:
                print(f"Record {server_id} has no image data (input_data is null).", file=sys.stderr)
                failed_count += 1
                continue

            try:
                # Load and correct image
                image_stream = io.BytesIO(input_data_bytes)
                unprocessed_image = Image.open(image_stream).convert("RGB")
                pil_image = correct_image_orientation(unprocessed_image)
                
                # Run YOLO inference
                results = model(pil_image, conf=CONF_THRESHOLD, verbose=False)
                
                if not results:
                    print(f"Inference failed for ID {server_id}", file=sys.stderr)
                    failed_count += 1
                    continue
                    
                result = results[0]
                
                # Use custom drawing function with consistent styling
                final_image = draw_yolo_results(pil_image, result)
                
                processed_images.append(final_image)
                successful_count += 1
                
            except Exception as e:
                print(f"Failed to process image for ID {server_id}: {e}", file=sys.stderr)
                failed_count += 1
                continue
        
        if not processed_images:
            print(f"No images were successfully processed for trial {trial_id}", file=sys.stderr)
            return False
        
        # Create video
        video_filename = os.path.join(output_dir, f"trial_{trial_id}.mp4")
        video_success = create_trial_video(processed_images, video_filename, fps)
        
        # Summary
        print(f"\n--- Trial {trial_id} Processing Complete ---")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Total frames in video: {len(processed_images)}")
        
        return video_success
        
    except Exception as e:
        print(f"Error processing trial {trial_id}: {e}", file=sys.stderr)
        return False

def draw_yolo_results(image, results, confidence_threshold=None):
    """
    Draw YOLO inference results on the image using custom styling.
    
    Args:
        image (PIL.Image): The image to draw on
        results: YOLO results object
        confidence_threshold (float): Minimum confidence to draw elements (uses CONF_THRESHOLD if None)
        
    Returns:
        PIL.Image: Image with results drawn
    """
    if confidence_threshold is None:
        confidence_threshold = CONF_THRESHOLD
        
    print("--- Inside draw_yolo_results ---")
    
    # --- FONT LOADING ---
    try:
        # Using a smaller font size
        font = ImageFont.truetype("DejaVuSans.ttf", size=7)
        text_y_offset = 9
    except IOError:
        print("Default font not found, using PIL's default. Text will not be resized.")
        font = ImageFont.load_default()
        text_y_offset = 15
    
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    items_drawn = 0
    
    # Track drawn elements to ensure only 1 of each type
    drawn_boxes = set()
    drawn_keypoints = set()
    
    # Process bounding boxes
    if results.boxes is not None:
        boxes = results.boxes
        for i, box in enumerate(boxes):
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id] if results.names else f"class_{class_id}"
            
            print(f"  Box {i}: {class_name} confidence: {confidence:.4f} (threshold: >{confidence_threshold})")
            
            # Skip if below threshold or already drawn this class
            if confidence <= confidence_threshold or class_name in drawn_boxes:
                if class_name in drawn_boxes:
                    print(f"    -> SKIPPED Box: {class_name} (already drawn)")
                continue
                
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Choose color based on class
            if class_name == 'face':
                box_color = COLORS['face_box']
            elif class_name == 'juice_tube':
                box_color = COLORS['juice_tube_box']
            else:
                box_color = 'white'  # fallback
            
            # Draw bounding box rectangle with thin lines
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
            
            drawn_boxes.add(class_name)
            items_drawn += 1
            print(f"    -> DRAWN Box: {class_name} at [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
    
    # Process keypoints
    if results.keypoints is not None:
        keypoints = results.keypoints
        
        # Define keypoint names (adjust based on your model)
        keypoint_names = ['left_pupil', 'right_pupil', 'nose_bridge', 'spout_top', 'spout_bottom']
        
        for pose_idx, pose_keypoints in enumerate(keypoints):
            print(f"\n  Processing Pose #{pose_idx+1} keypoints")
            xy = pose_keypoints.xy[0]  # Get x,y coordinates
            conf = pose_keypoints.conf[0] if pose_keypoints.conf is not None else None
            
            for kp_idx, (x, y) in enumerate(xy):
                # Get keypoint name and confidence
                kp_name = keypoint_names[kp_idx] if kp_idx < len(keypoint_names) else f"keypoint_{kp_idx}"
                kp_confidence = float(conf[kp_idx]) if conf is not None else 1.0
                
                print(f"    Keypoint '{kp_name}' confidence: {kp_confidence:.4f} (threshold: >{confidence_threshold})")
                
                # Skip if below threshold, already drawn, or at (0,0)
                if (kp_confidence <= confidence_threshold or 
                    kp_name in drawn_keypoints or
                    (float(x) == 0 and float(y) == 0)):
                    if kp_name in drawn_keypoints:
                        print(f"      -> SKIPPED Keypoint: {kp_name} (already drawn)")
                    elif float(x) == 0 and float(y) == 0:
                        print(f"      -> SKIPPED Keypoint: {kp_name} (at origin)")
                    continue
                
                # Choose color based on keypoint name
                point_color = COLORS.get(kp_name, 'white')
                
                # Draw keypoint as a small circle
                radius = 2
                x_coord, y_coord = float(x), float(y)
                draw.ellipse([x_coord-radius, y_coord-radius, x_coord+radius, y_coord+radius], 
                           fill=point_color)
                
                drawn_keypoints.add(kp_name)
                items_drawn += 1
                print(f"      -> DRAWN Keypoint: {kp_name} at ({x_coord:.1f}, {y_coord:.1f})")
    
    print(f"\n  Total items drawn: {items_drawn}")
    print("--- Exiting draw_yolo_results ---")
    return annotated_image

def main():
    """Main function to fetch, infer, and save a labeled image."""
    parser = argparse.ArgumentParser(
        description="Fetch images from the database and process them according to the specified mode.",
        epilog="""
ID Specification Examples:
  Manual list:        123 456 789
  Range syntax:       123456:20:134  (start:step:count - start at 123456, every 20th ID, for 134 total)
  Mixed:              123 456 100:5:10

Trial Video Mode:
  --mode trial-video --trial-id 12345 --fps 30
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("id_specs", nargs='*', 
                       help="Server infer IDs or range specifications. Use 'start:step:count' for ranges (e.g., '123456:20:134'). Not used in trial-video mode.")
    parser.add_argument("--mode", choices=['inference', 'save-only', 'plot-stored', 'trial-video'], default='inference',
                       help="Processing mode: 'inference' (run YOLO and save labeled image), 'save-only' (just save corrected image), 'plot-stored' (plot stored labels from DB), 'trial-video' (create video from all images in a trial)")
    parser.add_argument("--trial-id", type=int,
                       help="Trial ID to process (required for trial-video mode)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for video output (default: 30)")
    args = parser.parse_args()
    
    processing_mode = args.mode
    
    # Validate arguments based on mode
    if processing_mode == 'trial-video':
        if not args.trial_id:
            print("Error: --trial-id is required for trial-video mode", file=sys.stderr)
            return 1
        trial_id = args.trial_id
        fps = args.fps
        print(f"Trial video mode: Processing trial {trial_id} at {fps} FPS")
    else:
        if not args.id_specs:
            print("Error: id_specs are required for non-trial-video modes", file=sys.stderr)
            return 1
        # Parse the ID specifications
        try:
            server_ids_to_process = parse_id_specification(args.id_specs)
        except (ValueError, TypeError) as e:
            print(f"Error parsing ID specifications: {e}", file=sys.stderr)
            return 1
        
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
    if processing_mode in ['inference', 'trial-video']:
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
        
        if processing_mode == 'trial-video':
            print(f"Database connection established. Processing trial {trial_id}")
            success = process_trial_video(conn, model, trial_id, fps)
            return 0 if success else 1
        else:
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
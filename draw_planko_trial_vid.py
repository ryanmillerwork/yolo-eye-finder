import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import json
import cv2
import numpy as np

# Hardcoded trial ID for testing
TRIAL_ID = 431370

# --- Video & World Settings ---
VIDEO_WIDTH = 1000
VIDEO_HEIGHT = 1000
FPS = 60
WORLD_RANGE = 20.0 # Game world coordinates are roughly -10 to 10

# --- Color Definitions (BGR for OpenCV) ---
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GREY = (128, 128, 128)

def world_to_pixels(world_coord):
    """Scales a single world coordinate to a pixel coordinate."""
    return int((world_coord + WORLD_RANGE / 2) * VIDEO_WIDTH / WORLD_RANGE)

def world_to_pixel_radius(world_radius):
    """Scales a world radius to a pixel radius."""
    return int(world_radius * VIDEO_WIDTH / WORLD_RANGE)

def generate_schematic_frames(trial_info):
    """Generates schematic video frames with a 250ms blank start, returning them as a list."""
    if not trial_info or 'stiminfo' not in trial_info:
        print("Invalid trial_info data.")
        return []

    stiminfo = trial_info.get('stiminfo', {})
    
    # --- Get all required data ---
    rt = trial_info.get('rt', 0)
    ball_t = stiminfo.get('ball_t')
    ball_x = stiminfo.get('ball_x')
    ball_y = stiminfo.get('ball_y')
    ball_radius = stiminfo.get('ball_radius')
    names = stiminfo.get('name', [])
    shapes = stiminfo.get('shape', [])
    sx = stiminfo.get('sx', [])
    sy = stiminfo.get('sy', [])
    tx = stiminfo.get('tx', [])
    ty = stiminfo.get('ty', [])
    angles = stiminfo.get('angle', [])
    contact_bodies = stiminfo.get('contact_bodies', [])
    contact_t = stiminfo.get('contact_t', [])
    is_correct = trial_info.get('status') == 1

    if not all([ball_t, ball_x, ball_y, ball_radius is not None]):
        print("Missing ball trajectory or radius data.")
        return []

    # --- Time Calculation (for the event part of the video) ---
    selection_time = (250 + rt) / 1000.0
    animation_duration = ball_t[-1]
    total_event_duration = selection_time + animation_duration

    # --- Pre-calculate Full Ball Trajectory ---
    total_event_frames = int(total_event_duration * FPS)
    selection_frame = int(selection_time * FPS)
    
    interp_x = np.full(total_event_frames, ball_x[0])
    interp_y = np.full(total_event_frames, ball_y[0])

    if total_event_frames > selection_frame:
        animation_frames = total_event_frames - selection_frame
        animation_frame_times = np.linspace(0, animation_duration, animation_frames, endpoint=True)
        interp_x[selection_frame:] = np.interp(animation_frame_times, ball_t, ball_x)
        interp_y[selection_frame:] = np.interp(animation_frame_times, ball_t, ball_y)

    # --- Separate and Pre-calculate Static Object Vertices ---
    plank_vertices, left_catcher_vertices, right_catcher_vertices = [], [], []
    for i, name in enumerate(names):
        if i < len(shapes) and shapes[i] == 'Box':
            px_x, px_y = world_to_pixels(tx[i]), VIDEO_HEIGHT - world_to_pixels(ty[i])
            px_w, px_h = world_to_pixel_radius(sx[i]), world_to_pixel_radius(sy[i])
            half_w, half_h = px_w / 2, px_h / 2
            
            rect_corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])
            cos_a, sin_a = np.cos(-angles[i]), np.sin(-angles[i])
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            translated_corners = (rot_matrix @ rect_corners.T).T + [px_x, px_y]
            vertices = translated_corners.astype(np.int32)
            
            if 'plank' in name:
                plank_vertices.append(vertices)
            elif 'catchl' in name:
                left_catcher_vertices.append(vertices)
            elif 'catchr' in name:
                right_catcher_vertices.append(vertices)

    # --- Determine Catcher Colors and Contact Time ---
    initial_left_color, initial_right_color = WHITE, WHITE
    final_left_color, final_right_color = WHITE, WHITE
    absolute_contact_time = float('inf')
    
    catcher_contact_indices = [i for i, body in enumerate(contact_bodies) if body.startswith('catch')]
    if catcher_contact_indices:
        # First, determine where the ball actually ended up.
        ball_landed_right = contact_bodies[catcher_contact_indices[-1]].startswith('catchr')

        # Now, determine which catcher was the "chosen" one based on the trial's outcome.
        # If correct, chosen is where ball landed. If incorrect, it's the other one.
        if is_correct:
            chosen_catcher_was_right = ball_landed_right
        else:
            chosen_catcher_was_right = not ball_landed_right

        # Get the timing of the first contact event.
        relative_contact_time = contact_t[catcher_contact_indices[0]]
        absolute_contact_time = selection_time + relative_contact_time
        
        # The final color depends only on the trial's outcome.
        final_color = GREEN if is_correct else RED
        
        # The "chosen" catcher gets the grey color initially, and the final color upon contact.
        if chosen_catcher_was_right:
            initial_right_color = GREY
            final_right_color = final_color
        else: # Left catcher was chosen
            initial_left_color = GREY
            final_left_color = final_color
            
    frame_times = np.linspace(0, total_event_duration, total_event_frames, endpoint=True)
    print(f"Generating {total_event_frames} schematic frames... Selection at {selection_time:.2f}s, Contact at {absolute_contact_time if absolute_contact_time != float('inf') else 'N/A'}s from scene start.")

    # --- Frame Generation for the event part of the video ---
    event_frames = []
    for i, frame_time in enumerate(frame_times):
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            
        if frame_time < selection_time:
            left_color, right_color = WHITE, WHITE
        elif frame_time < absolute_contact_time:
            left_color, right_color = initial_left_color, initial_right_color
        else:
            left_color, right_color = final_left_color, final_right_color

        for vertices in plank_vertices: cv2.drawContours(frame, [vertices], 0, WHITE, -1)
        for vertices in left_catcher_vertices: cv2.drawContours(frame, [vertices], 0, left_color, -1)
        for vertices in right_catcher_vertices: cv2.drawContours(frame, [vertices], 0, right_color, -1)
        
        px = world_to_pixels(interp_x[i])
        py = VIDEO_HEIGHT - world_to_pixels(interp_y[i]) 
        pr = world_to_pixel_radius(ball_radius)
        cv2.circle(frame, (px, py), pr, CYAN, -1)

        event_frames.append(frame)

    # --- Prepend Blank Frames ---
    blank_frames_count = int(0.250 * FPS)
    blank_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    blank_frames = [blank_frame] * blank_frames_count
    
    print(f"Prepending {blank_frames_count} blank frames (250ms).")
    
    return blank_frames + event_frames

def get_trial_info(conn, trial_id: int):
    """Retrieves the trialinfo for a given trial_id."""
    try:
        with conn.cursor() as cur:
            query = "SELECT trialinfo FROM server_trial WHERE server_trial_id = %s"
            cur.execute(query, (trial_id,))
            record = cur.fetchone()
            if record:
                return record['trialinfo']
            return None
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching record for trial {trial_id}: {error}")
        return None

def main():
    """Main function to fetch trial info and save a standalone schematic video."""
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print("Error: PG_PASS not found in .env file.")
        return 1

    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="base",
            user="postgres",
            password=db_password,
            cursor_factory=DictCursor
        )
        print(f"Database connection established. Fetching info for trial {TRIAL_ID}")

        trial_info = get_trial_info(conn, TRIAL_ID)

        if trial_info:
            frames = generate_schematic_frames(trial_info)
            if frames:
                # --- Video Setup ---
                output_dir = "/mnt/qpcs/db/db_infer_and_label/planko"
                os.makedirs(output_dir, exist_ok=True)
                video_path = os.path.join(output_dir, f"trial-{TRIAL_ID}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
                
                print(f"Saving {len(frames)} frames to {video_path}...")
                for frame in frames:
                    out.write(frame)
                
                out.release()
                print("Video saved successfully.")
        else:
            print(f"No trial info found for trial_id {TRIAL_ID}")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()

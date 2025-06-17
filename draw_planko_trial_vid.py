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

def create_ball_video(trial_info, trial_id):
    """Creates a video with initial static period, selection, and contact-based coloring."""
    if not trial_info or 'stiminfo' not in trial_info:
        print("Invalid trial_info data.")
        return

    stiminfo = trial_info.get('stiminfo', {})
    
    # --- Get all required data from stiminfo and trial_info ---
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
        return

    # --- Time Calculation ---
    selection_time = (250 + rt) / 1000.0  # Time before animation starts
    animation_duration = ball_t[-1]
    total_duration = selection_time + animation_duration

    # --- Video Setup ---
    output_dir = "/mnt/qpcs/db/db_infer_and_label/planko"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"trial-{trial_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # --- Pre-calculate Full Ball Trajectory ---
    total_frames = int(total_duration * FPS)
    selection_frame = int(selection_time * FPS)
    
    interp_x = np.full(total_frames, ball_x[0])
    interp_y = np.full(total_frames, ball_y[0])

    if total_frames > selection_frame:
        animation_frames = total_frames - selection_frame
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
        last_contact_body = contact_bodies[catcher_contact_indices[-1]]
        chosen_catcher_is_right = last_contact_body.startswith('catchr')
        
        relative_contact_time = contact_t[catcher_contact_indices[0]]
        absolute_contact_time = selection_time + relative_contact_time
        
        final_color = GREEN if is_correct else RED
        
        if chosen_catcher_is_right:
            initial_right_color = GREY
            final_right_color = final_color
        else: # Left catcher chosen
            initial_left_color = GREY
            final_left_color = final_color
            
    frame_times = np.linspace(0, total_duration, total_frames, endpoint=True)
    print(f"Generating {total_frames} frames... Selection at {selection_time:.2f}s, Contact at {absolute_contact_time if absolute_contact_time != float('inf') else 'N/A'}s")

    # --- Frame Generation ---
    for i, frame_time in enumerate(frame_times):
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Determine current catcher colors based on the frame time
        if frame_time < selection_time:
            left_color, right_color = WHITE, WHITE
        elif frame_time < absolute_contact_time:
            left_color, right_color = initial_left_color, initial_right_color
        else:
            left_color, right_color = final_left_color, final_right_color

        # Draw all static objects and catchers
        for vertices in plank_vertices: cv2.drawContours(frame, [vertices], 0, WHITE, -1)
        for vertices in left_catcher_vertices: cv2.drawContours(frame, [vertices], 0, left_color, -1)
        for vertices in right_catcher_vertices: cv2.drawContours(frame, [vertices], 0, right_color, -1)
        
        # Draw the ball at its pre-calculated position for this frame
        px = world_to_pixels(interp_x[i])
        py = VIDEO_HEIGHT - world_to_pixels(interp_y[i]) 
        pr = world_to_pixel_radius(ball_radius)
        cv2.circle(frame, (px, py), pr, CYAN, -1)

        out.write(frame)

    out.release()
    print(f"Video saved to {video_path}")

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
    """Main function to fetch and print trial info."""
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
            create_ball_video(trial_info, TRIAL_ID)
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

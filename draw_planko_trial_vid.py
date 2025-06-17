import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import json
import cv2
import numpy as np

# Hardcoded trial ID for testing
TRIAL_ID = 431370

# Video settings
VIDEO_WIDTH = 1000
VIDEO_HEIGHT = 1000
FPS = 60
# Game world coordinates are roughly -10 to 10 in x and y
WORLD_RANGE = 20.0

def world_to_pixels(world_coord):
    """Scales a single world coordinate to a pixel coordinate."""
    return int((world_coord + WORLD_RANGE / 2) * VIDEO_WIDTH / WORLD_RANGE)

def world_to_pixel_radius(world_radius):
    """Scales a world radius to a pixel radius."""
    return int(world_radius * VIDEO_WIDTH / WORLD_RANGE)

def create_ball_video(trial_info, trial_id):
    """Creates a video of the ball's trajectory and static planks."""
    if not trial_info or 'stiminfo' not in trial_info:
        print("Invalid trial_info data.")
        return

    stiminfo = trial_info.get('stiminfo', {})
    
    # --- Get Ball Data ---
    ball_t = stiminfo.get('ball_t')
    ball_x = stiminfo.get('ball_x')
    ball_y = stiminfo.get('ball_y')
    ball_radius = stiminfo.get('ball_radius')

    if not all([ball_t, ball_x, ball_y, ball_radius is not None]):
        print("Missing ball trajectory or radius data.")
        return

    # --- Get Plank Data ---
    names = stiminfo.get('name', [])
    shapes = stiminfo.get('shape', [])
    sx = stiminfo.get('sx', [])
    sy = stiminfo.get('sy', [])
    tx = stiminfo.get('tx', [])
    ty = stiminfo.get('ty', [])
    angles = stiminfo.get('angle', []) # Radians

    # --- Video Setup ---
    output_dir = "/mnt/qpcs/db/db_infer_and_label/planko"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"trial-{trial_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # --- Trajectory Interpolation for Ball ---
    duration = ball_t[-1]
    total_frames = int(duration * FPS)
    frame_times = np.linspace(0, duration, total_frames, endpoint=True)
    interp_x = np.interp(frame_times, ball_t, ball_x)
    interp_y = np.interp(frame_times, ball_t, ball_y)
    
    # --- Pre-calculate Plank Vertices in Pixel Coordinates ---
    plank_vertices = []
    for i, name in enumerate(names):
        if 'plank' in name and i < len(shapes) and shapes[i] == 'Box':
            # Convert world coordinates and dimensions to pixel values
            px_x = world_to_pixels(tx[i])
            px_y = VIDEO_HEIGHT - world_to_pixels(ty[i])
            px_w = world_to_pixel_radius(sx[i]) # Reusing this function as it just scales a length
            px_h = world_to_pixel_radius(sy[i])
            
            # Define corners of unrotated rectangle at origin
            half_w, half_h = px_w / 2, px_h / 2
            rect_corners = np.array([
                [-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]
            ])

            # Create rotation matrix. World Y is up, pixel Y is down, so negate angle.
            angle_rad = angles[i]
            cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Rotate corners around origin, then translate to final position
            rotated_corners = (rot_matrix @ rect_corners.T).T
            translated_corners = rotated_corners + [px_x, px_y]
            
            plank_vertices.append(translated_corners.astype(np.int32))

    print(f"Generating {total_frames} frames for a {duration:.2f}s video with {len(plank_vertices)} planks...")

    # --- Frame Generation ---
    for i in range(total_frames):
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Draw all static planks
        for vertices in plank_vertices:
            cv2.drawContours(frame, [vertices], 0, (255, 255, 255), -1)
        
        # Get interpolated ball position for the current frame
        x, y = interp_x[i], interp_y[i]
        
        # Convert world coordinates to pixel coordinates for drawing
        px = world_to_pixels(x)
        py = VIDEO_HEIGHT - world_to_pixels(y) 
        pr = world_to_pixel_radius(ball_radius)

        # Draw the ball (a white circle)
        cv2.circle(frame, (px, py), pr, (255, 255, 255), -1)

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

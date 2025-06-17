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
    """Creates a video of the ball's trajectory."""
    if not trial_info or 'stiminfo' not in trial_info:
        print("Invalid trial_info data.")
        return

    stiminfo = trial_info.get('stiminfo', {})
    
    # Get ball data from stiminfo
    ball_t = stiminfo.get('ball_t')
    ball_x = stiminfo.get('ball_x')
    ball_y = stiminfo.get('ball_y')
    ball_radius = stiminfo.get('ball_radius')

    if not all([ball_t, ball_x, ball_y, ball_radius is not None]):
        print("Missing ball trajectory or radius data.")
        return

    # --- Video Setup ---
    output_dir = "/mnt/qpcs/db/db_infer_and_label/planko"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"trial-{trial_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # --- Trajectory Interpolation ---
    duration = ball_t[-1]
    total_frames = int(duration * FPS)
    # Generate timestamps for each frame of the video
    frame_times = np.linspace(0, duration, total_frames, endpoint=True)
    
    # Interpolate ball positions for each frame time
    interp_x = np.interp(frame_times, ball_t, ball_x)
    interp_y = np.interp(frame_times, ball_t, ball_y)
    
    print(f"Generating {total_frames} frames for a {duration:.2f}s video...")

    # --- Frame Generation ---
    for i in range(total_frames):
        # Create a black background
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Get interpolated ball position for the current frame
        x = interp_x[i]
        y = interp_y[i]
        
        # Convert world coordinates to pixel coordinates for drawing
        # The y-axis is inverted in graphics (0 is at the top)
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

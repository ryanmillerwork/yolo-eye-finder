import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import json

# Hardcoded trial ID for testing
TRIAL_ID = 431370

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
            # Pretty print the JSON
            print(json.dumps(trial_info, indent=4))
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

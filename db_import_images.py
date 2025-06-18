import psycopg2
from psycopg2.extras import execute_values
import time
import os
from dotenv import load_dotenv
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Import image data from a client database to the server database.")
parser.add_argument("--source-db", required=True, help="IP address of the source (client) database.")
args = parser.parse_args()

# --- Load Environment Variables ---
load_dotenv()
db_password = os.getenv("PG_PASS")
if not db_password:
    raise ValueError("PG_PASS not found in .env file or environment variables")

# Configuration
client_db_config = {
    'host': args.source_db,
    'port': 5432,
    'dbname': 'base',
    'user': 'postgres',
    'password': db_password
}

server_db_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'base',
    'user': 'postgres',
    'password': db_password
}

TARGET_HOST = args.source_db
BATCH_SIZE = 1000

# Connect to databases
client_conn = psycopg2.connect(**client_db_config)
server_conn = psycopg2.connect(**server_db_config)
client_cur = client_conn.cursor()
server_cur = server_conn.cursor()

# Step 1: Count total rows to process
server_cur.execute("""
    SELECT COUNT(*) FROM server_inference
    WHERE input_data IS NULL AND host = %s
""", (TARGET_HOST,))
total_rows = server_cur.fetchone()[0]
print(f"Total rows missing input_data: {total_rows}")

# Step 2: Batched processing
offset = 0
total_updated = 0

while True:
    # Fetch a batch of infer_ids from the server
    server_cur.execute("""
        SELECT infer_id, host, client_time
        FROM server_inference
        WHERE input_data IS NULL AND host = %s
        ORDER BY infer_id
        LIMIT %s OFFSET %s
    """, (TARGET_HOST, BATCH_SIZE, offset))
    rows = server_cur.fetchall()
    if not rows:
        break

    # Create a list of tuples with the keys. This contains the original server client_time,
    # which is needed for the final UPDATE statement to be precise.
    keys = [(r[0], r[1], r[2]) for r in rows]

    # --- Fetch matching input_data from client DB ---
    # This uses a temporary table, which is more robust for composite keys with timestamps
    # than a large `IN` clause.
    client_cur.execute("""
        CREATE TEMP TABLE IF NOT EXISTS keys_to_fetch (
            infer_id INT,
            host INET,
            client_time TIMESTAMP WITHOUT TIME ZONE
        );
    """)
    client_cur.execute("TRUNCATE TABLE keys_to_fetch;")

    execute_values(client_cur,
        "INSERT INTO keys_to_fetch (infer_id, host, client_time) VALUES %s",
        keys
    )

    client_cur.execute("""
        SELECT i.infer_id, i.host, i.client_time, i.input_data
        FROM inference AS i
        JOIN keys_to_fetch AS t ON
            i.infer_id = t.infer_id AND
            i.host = t.host AND
            i.client_time = t.client_time;
    """)
    input_data_rows = client_cur.fetchall()

    # Index by composite key for fast lookup
    input_lookup = {(r[0], r[1], r[2]): r[3] for r in input_data_rows}

    # Perform update one by one
    for infer_id, host, client_time in keys:
        input_data = input_lookup.get((infer_id, host, client_time))
        if input_data:
            server_cur.execute("""
                UPDATE server_inference
                SET input_data = %s
                WHERE infer_id = %s AND host = %s AND client_time = %s
                      AND input_data IS NULL
            """, (input_data, infer_id, host, client_time))
            total_updated += server_cur.rowcount

    server_conn.commit()
    print(f"Batch complete: offset {offset} – updated so far: {total_updated}")
    offset += BATCH_SIZE

    time.sleep(0.1)  # short delay to avoid overloading either DB

# Cleanup
client_cur.close()
server_cur.close()
client_conn.close()
server_conn.close()

print(f"✅ Done. Total rows updated: {total_updated}")
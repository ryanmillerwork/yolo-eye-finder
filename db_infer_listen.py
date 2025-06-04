import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from typing import List, Any
import io # For in-memory byte streams
from PIL import Image # For handling image data; pip install Pillow

def get_inferences_by_ids(server_infer_ids: List[int]) -> List[Any]:
    """
    Connects to the PostgreSQL database and retrieves records
    from the server_inference table by a list of server_infer_ids.
    Records are returned as dictionaries (DictRow).

    Args:
        server_infer_ids: A list of IDs of the inference records to retrieve.

    Returns:
        A list of DictRow objects, where each object represents a row.
        Returns an empty list if no records are found or in case of an error.
    """
    load_dotenv()  # Load environment variables from .env file

    if not server_infer_ids:
        print("No server_infer_ids provided.")
        return []

    db_host = "localhost"
    db_name = "base"
    db_user = "postgres"
    db_password = os.getenv("PG_PASS")

    if not db_password:
        print("Error: PG_PASS not found in .env file or environment variables.")
        return []

    conn = None
    records = []
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            cursor_factory=DictCursor
        )

        # Open a cursor to perform database operations
        cur = conn.cursor()

        query = "SELECT * FROM server_inference WHERE server_infer_id = ANY(%s)"
        cur.execute(query, (server_infer_ids,))

        records = cur.fetchall()

        return records

    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL or executing query: {error}")
        return []
    finally:
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    test_ids = [185355]  # Example ID, replace as needed

    print(f"Attempting to fetch records for IDs: {test_ids}")
    inference_data_list = get_inferences_by_ids(test_ids)

    if inference_data_list:
        print(f"Found {len(inference_data_list)} record(s).")
        for record in inference_data_list:
            server_id = record['server_infer_id']
            print(f"Processing record ID: {server_id}")

            mime_type = record.get('mime_type')
            input_data_bytes = record.get('input_data') # This is likely memoryview or bytes

            if mime_type and input_data_bytes:
                if mime_type.startswith('image/'):
                    try:
                        # Convert byte data to an in-memory image object
                        image_stream = io.BytesIO(input_data_bytes)
                        img = Image.open(image_stream)
                        # You might want to convert to a specific mode, e.g., RGB
                        # img = img.convert("RGB") 
                        print(f"  Successfully loaded image data for record ID {server_id} into memory. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")

                        # 1. Perform YOLO inference here
                        # yolo_results = your_yolo_inference_function(img) # or pass input_data_bytes directly if your func supports it
                        # print(f"  (Placeholder) YOLO inference would be performed on image from record ID {server_id}.")
                        # print(f"  (Placeholder) YOLO results: {{yolo_results}}")

                        # 2. Store results back in PostgreSQL
                        # store_inference_results(server_id, yolo_results) # You'll need to define this function
                        # print(f"  (Placeholder) Inference results for record ID {server_id} would be stored.")

                    except IOError as e:
                        print(f"  Error processing image data for record ID {server_id}: {e}")
                    except Exception as e:
                        print(f"  An unexpected error occurred while processing image for record ID {server_id}: {e}")
                else:
                    print(f"  Record ID {server_id}: Mime type {mime_type} is not an image. Skipping image processing.")
            else:
                if not mime_type:
                    print(f"  Record ID {server_id}: Mime type is missing.")
                if not input_data_bytes:
                     print(f"  Record ID {server_id}: Input data is missing.")

    else:
        print(f"No data found for server_infer_ids {test_ids} or an error occurred.")

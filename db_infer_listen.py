import os
import psycopg2
from dotenv import load_dotenv
from typing import List, Tuple, Any

def get_inferences_by_ids(server_infer_ids: List[int]) -> List[Tuple[Any, ...]]:
    """
    Connects to the PostgreSQL database and retrieves records
    from the server_inference table by a list of server_infer_ids.

    Args:
        server_infer_ids: A list of IDs of the inference records to retrieve.

    Returns:
        A list of tuples, where each tuple represents a row.
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
            password=db_password
        )

        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Execute a query using ANY for a list of IDs
        # Note: For an empty list, this would cause an SQL error.
        # We've added a check above to return [] if server_infer_ids is empty.
        query = "SELECT * FROM server_inference WHERE server_infer_id = ANY(%s)"
        cur.execute(query, (server_infer_ids,))

        # Retrieve query results
        records = cur.fetchall()

        return records

    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL or executing query: {error}")
        return []  # Return empty list on error
    finally:
        # closing database connection.
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    # Example usage:
    # Make sure you have a .env file in the same directory as this script
    # with the line: PG_PASS='your_password'
    # Or that PG_PASS is set as an environment variable.

    test_ids = [185355]  # Replace with actual IDs you want to test
    # To test with a single ID, use a list: test_ids = [1]
    # To test with non-existent IDs: test_ids = [99999, 100000]

    print(f"Attempting to fetch records for IDs: {test_ids}")
    inference_data_list = get_inferences_by_ids(test_ids)

    if inference_data_list:
        print(f"Found {len(inference_data_list)} record(s):")
        for record in inference_data_list:
            print(record)
    else:
        print(f"No data found for server_infer_ids {test_ids} or an error occurred.")

    # Example with an empty list
    # print("\nAttempting to fetch records for an empty list of IDs:")
    # inference_data_empty = get_inferences_by_ids([])
    # if inference_data_empty:
    #     print(f"Found {len(inference_data_empty)} record(s):")
    #     for record in inference_data_empty:
    #         print(record)
    # else:
    #     print(f"No data found for empty ID list or an error occurred.")

    # Example with a non-existent ID along with potentially existing ones
    # test_ids_mixed = [1, 99999] # Assuming 1 might exist, 99999 might not
    # print(f"\nAttempting to fetch records for IDs: {test_ids_mixed}")
    # inference_data_mixed = get_inferences_by_ids(test_ids_mixed)
    # if inference_data_mixed:
    #     print(f"Found {len(inference_data_mixed)} record(s):")
    #     for record in inference_data_mixed:
    #         print(record)
    # else:
    #     print(f"No data found for server_infer_ids {test_ids_mixed} or an error occurred.")

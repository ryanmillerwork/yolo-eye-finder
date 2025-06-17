#!/usr/bin/env python3
"""
Script to assign trial IDs to server_inference records based on timing analysis.

This script analyzes the relationship between server_trial and server_inference tables
using client_time and trial_time to determine which inference records belong to which trials.
"""

import os
import sys
import argparse
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from datetime import datetime, timedelta

def get_trial_info(conn, trial_id):
    """
    Get the client_time (end time) and host for a specific trial.
    
    Args:
        conn: Database connection
        trial_id (int): The server_trial_id to look up
        
    Returns:
        tuple: (client_time, host) or (None, None) if not found
    """
    try:
        with conn.cursor() as cur:
            query = "SELECT client_time, host FROM server_trial WHERE server_trial_id = %s"
            cur.execute(query, (trial_id,))
            result = cur.fetchone()
            if result:
                return result['client_time'], result['host']
            else:
                print(f"No trial found with server_trial_id: {trial_id}")
                return None, None
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching trial {trial_id}: {error}")
        return None, None

def get_trials_for_date(conn, date_str):
    """
    Get all trial IDs for a specific date.
    
    Args:
        conn: Database connection
        date_str (str): Date in YYYY-MM-DD format
        
    Returns:
        list: List of server_trial_id values for that date
    """
    try:
        with conn.cursor() as cur:
            query = """
            SELECT server_trial_id, client_time, host 
            FROM server_trial 
            WHERE DATE(client_time) = %s
            ORDER BY client_time
            """
            cur.execute(query, (date_str,))
            results = cur.fetchall()
            
            print(f"Found {len(results)} trials for date {date_str}")
            if results:
                print(f"Trial IDs: {[r['server_trial_id'] for r in results]}")
                hosts = set(r['host'] for r in results)
                print(f"Hosts involved: {sorted(hosts)}")
            
            return [r['server_trial_id'] for r in results]
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching trials for date {date_str}: {error}")
        return []

def get_candidate_inferences(conn, trial_end_time, host):
    """
    Get server_inference records that might belong to the trial based on timing window and host.
    
    Args:
        conn: Database connection
        trial_end_time (datetime): The end time of the trial
        host (str): The host that ran the trial
        
    Returns:
        list: List of candidate inference records
    """
    try:
        # Define the time window: 12 seconds before to 2 seconds after trial end
        start_time = trial_end_time - timedelta(seconds=12)
        end_time = trial_end_time + timedelta(seconds=2)
        
        with conn.cursor() as cur:
            query = """
            SELECT server_infer_id, client_time, trial_time 
            FROM server_inference 
            WHERE client_time > %s AND client_time < %s AND host = %s
            ORDER BY client_time
            """
            cur.execute(query, (start_time, end_time, host))
            results = cur.fetchall()
            
            print(f"Found {len(results)} candidate inference records in time window for host '{host}'")
            print(f"Time window: {start_time} to {end_time}")
            
            return results
    except (Exception, psycopg2.Error) as error:
        print(f"Error fetching candidate inferences: {error}")
        return []

def find_trial_boundaries(candidate_records, trial_end_time):
    """
    Find the start and end boundaries of the trial within the candidate records.
    
    Args:
        candidate_records (list): List of candidate inference records
        trial_end_time (datetime): The end time of the trial
        
    Returns:
        tuple: (start_index, end_index) or (None, None) if boundaries not found
    """
    if not candidate_records:
        return None, None
    
    # DEBUG: Show trial_time progression for all records
    print(f"\nDEBUG: Trial time progression for all {len(candidate_records)} candidates:")
    for i, record in enumerate(candidate_records):
        time_diff = (record['client_time'] - trial_end_time).total_seconds()
        print(f"  {i:3d}: ID {record['server_infer_id']}, trial_time {record['trial_time']:5d}, "
              f"time_offset {time_diff:+7.3f}s")
        if i >= 20 and len(candidate_records) > 40:
            print(f"  ... (skipping {len(candidate_records) - 40} middle records)")
            break
    
    # Show last 20 if we skipped middle records
    if len(candidate_records) > 40:
        for i in range(max(21, len(candidate_records) - 20), len(candidate_records)):
            record = candidate_records[i]
            time_diff = (record['client_time'] - trial_end_time).total_seconds()
            print(f"  {i:3d}: ID {record['server_infer_id']}, trial_time {record['trial_time']:5d}, "
                  f"time_offset {time_diff:+7.3f}s")
    
    print()  # blank line
    
    # Find the sample time (500ms before trial end)
    sample_time = trial_end_time - timedelta(milliseconds=500)
    
    # Find the record closest to the sample time
    closest_index = 0
    min_diff = abs((candidate_records[0]['client_time'] - sample_time).total_seconds())
    
    for i, record in enumerate(candidate_records):
        diff = abs((record['client_time'] - sample_time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    
    print(f"Sample time: {sample_time}")
    print(f"Closest record index: {closest_index}, client_time: {candidate_records[closest_index]['client_time']}")
    print(f"Closest record trial_time: {candidate_records[closest_index]['trial_time']}")
    
    # Work backwards to find trial start
    start_index = closest_index
    print(f"\nDEBUG: Working backwards from index {closest_index}:")
    for i in range(closest_index - 1, max(-1, closest_index - 10), -1):
        if i < 0:
            break
        current_trial_time = candidate_records[i]['trial_time']
        next_trial_time = candidate_records[i + 1]['trial_time']
        
        print(f"  Index {i}: trial_time {current_trial_time} -> {next_trial_time} "
              f"(diff: {next_trial_time - current_trial_time:+d})")
        
        # If trial_time increases when going backwards, we've hit the previous trial
        if current_trial_time > next_trial_time:
            print(f"Found trial start boundary at index {i+1}")
            print(f"  Previous trial_time: {current_trial_time}, Current trial_time: {next_trial_time}")
            start_index = i + 1
            break
    else:
        # If we didn't break, check if we should look further back
        if closest_index >= 10:
            print(f"  (checking further back...)")
            for i in range(closest_index - 10, -1, -1):
                current_trial_time = candidate_records[i]['trial_time']
                next_trial_time = candidate_records[i + 1]['trial_time']
                
                if current_trial_time > next_trial_time:
                    print(f"Found trial start boundary at index {i+1}")
                    print(f"  Previous trial_time: {current_trial_time}, Current trial_time: {next_trial_time}")
                    start_index = i + 1
                    break
            else:
                # If we still didn't break, use the first record
                start_index = 0
                print(f"Used first record as trial start (index 0)")
        else:
            start_index = 0
            print(f"Used first record as trial start (index 0)")
    
    # Work forwards to find trial end
    end_index = closest_index
    print(f"\nDEBUG: Working forwards from index {closest_index}:")
    for i in range(closest_index + 1, min(len(candidate_records), closest_index + 10)):
        current_trial_time = candidate_records[i]['trial_time']
        previous_trial_time = candidate_records[i - 1]['trial_time']
        
        diff = current_trial_time - previous_trial_time
        print(f"  Index {i}: trial_time {previous_trial_time} -> {current_trial_time} "
              f"(diff: {diff:+d})")
        
        if diff == 0:
            record1 = candidate_records[i-1]
            record2 = candidate_records[i]
            time_between_dupes = (record2['client_time'] - record1['client_time']).total_seconds()
            print(f"  WARNING: Duplicate trial_time ({current_trial_time}) found for sequential records.")
            print(f"    - Record {i-1}: ID {record1['server_infer_id']}, client_time {record1['client_time']}")
            print(f"    - Record {i}:   ID {record2['server_infer_id']}, client_time {record2['client_time']}")
            print(f"    - Time between duplicate records: {time_between_dupes:.4f}s")
        
        # If trial_time drops significantly when going forwards, we've hit the next trial
        if current_trial_time < previous_trial_time and (previous_trial_time - current_trial_time) > 1000:  # 1 second drop
            print(f"Found trial end boundary at index {i-1}")
            print(f"  Previous trial_time: {previous_trial_time}, Next trial_time: {current_trial_time}")
            end_index = i - 1
            break
    else:
        # If we didn't break, check if we should look further forward
        if len(candidate_records) - closest_index > 10:
            print(f"  (checking further forward...)")
            for i in range(closest_index + 10, len(candidate_records)):
                current_trial_time = candidate_records[i]['trial_time']
                previous_trial_time = candidate_records[i - 1]['trial_time']
                
                if current_trial_time < previous_trial_time and (previous_trial_time - current_trial_time) > 1000:
                    print(f"Found trial end boundary at index {i-1}")
                    print(f"  Previous trial_time: {previous_trial_time}, Next trial_time: {current_trial_time}")
                    end_index = i - 1
                    break
            else:
                # If we still didn't break, use the last record
                end_index = len(candidate_records) - 1
                print(f"Used last record as trial end (index {end_index})")
        else:
            end_index = len(candidate_records) - 1
            print(f"Used last record as trial end (index {end_index})")
    
    return start_index, end_index

def assign_trial_ids_to_inferences(conn, trial_id, inference_records):
    """
    Update the server_trial_id column for the identified inference records.
    
    Args:
        conn: Database connection
        trial_id (int): The trial ID to assign
        inference_records (list): List of inference records to update
        
    Returns:
        int: Number of records updated
    """
    if not inference_records:
        return 0
    
    try:
        inference_ids = [record['server_infer_id'] for record in inference_records]
        
        with conn.cursor() as cur:
            # Use ANY to update multiple records efficiently
            query = """
            UPDATE server_inference 
            SET server_trial_id = %s 
            WHERE server_infer_id = ANY(%s)
            """
            cur.execute(query, (trial_id, inference_ids))
            updated_count = cur.rowcount
            
            # Commit the changes
            conn.commit()
            
            print(f"Updated {updated_count} inference records with trial_id {trial_id}")
            return updated_count
            
    except (Exception, psycopg2.Error) as error:
        print(f"Error updating inference records: {error}")
        conn.rollback()
        return 0

def process_trial(conn, trial_id, dry_run=False):
    """
    Process a single trial to assign inference record IDs.
    
    Args:
        conn: Database connection
        trial_id (int): The trial ID to process
        dry_run (bool): If True, don't actually update the database
        
    Returns:
        bool: True if successful, False if failed
    """
    print(f"\n--- Processing Trial {trial_id} ---")
    
    # Step 1: Get trial end time and host
    trial_end_time, host = get_trial_info(conn, trial_id)
    if not trial_end_time or not host:
        return False
    
    print(f"Trial end time: {trial_end_time}")
    print(f"Trial host: {host}")
    
    # Step 2: Get candidate inference records (filtered by host and time window)
    candidate_records = get_candidate_inferences(conn, trial_end_time, host)
    if not candidate_records:
        print("No candidate records found")
        return False
    
    # Step 3: Find trial boundaries
    start_index, end_index = find_trial_boundaries(candidate_records, trial_end_time)
    if start_index is None or end_index is None:
        print("Could not determine trial boundaries")
        return False
    
    # Step 4: Extract the trial's inference records
    trial_records = candidate_records[start_index:end_index + 1]
    
    print(f"Trial boundaries: index {start_index} to {end_index}")
    print(f"Trial contains {len(trial_records)} inference records")
    if trial_records:
        print(f"First record: ID {trial_records[0]['server_infer_id']}, trial_time {trial_records[0]['trial_time']}")
        print(f"Last record: ID {trial_records[-1]['server_infer_id']}, trial_time {trial_records[-1]['trial_time']}")
    
    # Step 5: Update database (unless dry run)
    if dry_run:
        print(f"DRY RUN: Would update {len(trial_records)} records with trial_id {trial_id}")
        return True
    else:
        updated_count = assign_trial_ids_to_inferences(conn, trial_id, trial_records)
        return updated_count > 0

def main():
    """Main function to assign trial IDs to inference records."""
    parser = argparse.ArgumentParser(
        description="Assign trial IDs to server_inference records based on timing analysis.",
        epilog="""
Examples:
  python db_assign_trial_ids.py 12345                    # Process single trial
  python db_assign_trial_ids.py 12345 12346 12347        # Process multiple trials
  python db_assign_trial_ids.py 12345 --dry-run          # Test without updating database
  python db_assign_trial_ids.py --date 2025-06-06        # Process all trials for a specific date
  python db_assign_trial_ids.py --date 2025-06-06 --dry-run  # Test date processing
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trial_ids", nargs='*', type=int,
                       help="Server trial IDs to process (not used with --date)")
    parser.add_argument("--date", type=str,
                       help="Process all trials for a specific date (format: YYYY-MM-DD)")
    parser.add_argument("--dry-run", action='store_true',
                       help="Show what would be updated without actually updating the database")
    args = parser.parse_args()
    
    # Validate arguments
    if args.date and args.trial_ids:
        print("Error: Cannot specify both --date and trial_ids", file=sys.stderr)
        return 1
    
    if not args.date and not args.trial_ids:
        print("Error: Must specify either --date or trial_ids", file=sys.stderr)
        return 1
    
    dry_run = args.dry_run
    
    if dry_run:
        print("DRY RUN MODE - No database changes will be made")
    
    # Load environment
    load_dotenv()
    db_password = os.getenv("PG_PASS")
    if not db_password:
        print("Error: PG_PASS not found in .env file.", file=sys.stderr)
        return 1
    
    # Database connection
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost", database="base", user="postgres", password=db_password, cursor_factory=DictCursor
        )
        print("Database connection established.")
        
        # Get trial IDs to process
        if args.date:
            print(f"Processing all trials for date: {args.date}")
            trial_ids = get_trials_for_date(conn, args.date)
            if not trial_ids:
                print(f"No trials found for date {args.date}")
                return 0
        else:
            trial_ids = args.trial_ids
            print(f"Processing {len(trial_ids)} specified trial(s): {trial_ids}")
        
        # Process each trial
        successful_count = 0
        failed_count = 0
        
        for trial_id in trial_ids:
            success = process_trial(conn, trial_id, dry_run)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        
        # Summary
        print(f"\n--- Processing Complete ---")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Total: {len(trial_ids)}")
        
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
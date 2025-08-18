#!/usr/bin/env python3
"""
Script to regenerate a map with fixed data processing
"""

import os
import sys
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

def regenerate_map(map_id):
    """Regenerate a map with the fixed data processing"""
    
    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "transparentsf"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    # Get the original map data
    cursor.execute("SELECT * FROM maps WHERE id = %s", [map_id])
    map_data = cursor.fetchone()
    
    if not map_data:
        print(f"Map {map_id} not found")
        return
    
    print(f"Regenerating map {map_id}: {map_data['title']}")
    
    # Parse the original location data
    if isinstance(map_data['location_data'], str):
        try:
            original_location_data = json.loads(map_data['location_data'])
        except:
            original_location_data = []
    else:
        original_location_data = map_data['location_data'] or []
    
    # Handle different data formats
    if isinstance(original_location_data, dict) and 'data' in original_location_data:
        original_location_data = original_location_data['data']
    elif not isinstance(original_location_data, list):
        original_location_data = []
    
    print(f"Original data has {len(original_location_data)} items")
    
    # Convert to DataFrame for processing
    import pandas as pd
    df = pd.DataFrame(original_location_data)
    
    # Process the data with the fixed function
    from tools.generate_map import process_dataset_for_map
    processed_data = process_dataset_for_map(df, map_data['type'])
    
    print(f"Processed data has {len(processed_data)} items")
    
    # Update the map in the database
    cursor.execute("""
        UPDATE maps 
        SET location_data = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
    """, (json.dumps(processed_data), map_id))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Map {map_id} regenerated successfully!")
    print(f"View at: http://localhost:8000/map/{map_id}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python regenerate_map.py <map_id>")
        sys.exit(1)
    
    map_id = sys.argv[1]
    regenerate_map(map_id)

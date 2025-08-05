#!/usr/bin/env python3
"""
Script to add evaluation tables to the database.
This should be run once to set up the new eval system.
"""

import os
import sys
from pathlib import Path

# Add parent directory to sys.path for module resolution
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from tools.db_utils import get_postgres_connection

def add_eval_tables():
    """Execute the SQL script to add eval tables."""
    try:
        # Read the SQL file
        sql_file_path = Path(__file__).parent / "add_eval_tables_simple.sql"
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()
        
        # Get database connection
        connection = get_postgres_connection()
        if not connection:
            print("Failed to establish database connection")
            return False
        
        # Execute the SQL
        cursor = connection.cursor()
        cursor.execute(sql_content)
        connection.commit()
        
        print("Successfully added eval tables to the database")
        return True
        
    except Exception as e:
        print(f"Error adding eval tables: {str(e)}")
        return False
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    success = add_eval_tables()
    if success:
        print("Eval tables setup completed successfully!")
    else:
        print("Failed to setup eval tables")
        sys.exit(1) 
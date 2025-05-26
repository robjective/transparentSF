#!/usr/bin/env python3
"""
Script to migrate the maps table from UUID to SERIAL (integer) primary key.
This makes map IDs smaller and more manageable.
"""

import os
import sys
import psycopg2
from pathlib import Path

def get_db_connection():
    """Get database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "transparentsf"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )

def run_migration():
    """Run the database migration."""
    try:
        # Read the migration SQL file
        migration_file = Path(__file__).parent / "migrate_map_id_to_integer.sql"
        
        if not migration_file.exists():
            print(f"Error: Migration file {migration_file} not found")
            return False
        
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        # Connect to database
        print("Connecting to database...")
        conn = get_db_connection()
        conn.autocommit = True  # Enable autocommit for DDL operations
        cursor = conn.cursor()
        
        # Check if maps table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'maps'
            );
        """)
        
        if not cursor.fetchone()[0]:
            print("Maps table does not exist. Creating with SERIAL id...")
            # If table doesn't exist, create it with SERIAL id directly
            cursor.execute("""
                CREATE TABLE maps (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    chart_id TEXT UNIQUE NOT NULL,
                    edit_url TEXT NOT NULL,
                    published_url TEXT NOT NULL,
                    location_data JSONB,
                    metadata JSONB,
                    metric_id TEXT,
                    group_field TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Maps table created successfully with SERIAL id.")
            return True
        
        # Check current id column type
        cursor.execute("""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = 'maps' AND column_name = 'id'
        """)
        
        current_type = cursor.fetchone()[0]
        print(f"Current id column type: {current_type}")
        
        if current_type == 'integer':
            print("Maps table already uses integer id. No migration needed.")
            return True
        
        if current_type != 'uuid':
            print(f"Warning: Unexpected id column type: {current_type}")
            response = input("Continue with migration? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Check if there are any existing records
        cursor.execute("SELECT COUNT(*) FROM maps")
        record_count = cursor.fetchone()[0]
        print(f"Found {record_count} existing records in maps table")
        
        if record_count > 0:
            print("Warning: This migration will change all existing map IDs!")
            print("Existing map references in reports may break.")
            response = input("Continue with migration? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Execute migration
        print("Running migration...")
        
        # Split the migration SQL into individual statements
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            if statement.startswith('--') or not statement:
                continue
            
            print(f"Executing statement {i+1}/{len(statements)}")
            try:
                cursor.execute(statement)
                print(f"✓ Statement {i+1} completed")
            except Exception as e:
                print(f"✗ Error in statement {i+1}: {e}")
                if "does not exist" in str(e) and "constraint" in str(e):
                    print("  (This error may be expected if constraint doesn't exist)")
                    continue
                else:
                    raise
        
        print("Migration completed successfully!")
        
        # Verify the migration
        cursor.execute("""
            SELECT data_type, column_default
            FROM information_schema.columns 
            WHERE table_name = 'maps' AND column_name = 'id'
        """)
        
        result = cursor.fetchone()
        print(f"New id column type: {result[0]}")
        print(f"New id column default: {result[1]}")
        
        # Show sample records
        cursor.execute("SELECT id, title, type FROM maps ORDER BY id LIMIT 5")
        records = cursor.fetchall()
        
        if records:
            print("\nSample records after migration:")
            for record in records:
                print(f"  ID: {record[0]}, Title: {record[1]}, Type: {record[2]}")
        else:
            print("No records found in maps table")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("Maps Table Migration: UUID to SERIAL")
    print("=" * 40)
    
    # Load environment variables if .env file exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("Loading environment variables from .env file...")
        from dotenv import load_dotenv
        load_dotenv()
    
    success = run_migration()
    
    if success:
        print("\n✓ Migration completed successfully!")
        print("Map IDs are now integers instead of UUIDs.")
        print("The generate_map function will now return map_id in its response.")
    else:
        print("\n✗ Migration failed!")
        sys.exit(1) 
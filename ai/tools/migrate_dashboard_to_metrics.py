#!/usr/bin/env python3
"""
Script to restore the metrics table from a backup file.
"""

import logging
import os
import re
import io
from ai.tools.db_utils import get_postgres_connection, execute_with_connection
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_sql_statements(sql_content):
    """Split SQL content into individual statements, handling comments and semicolons properly."""
    sql_content = re.sub(r'--.*$', '', sql_content, flags=re.MULTILINE)

    statements = []
    buffer = []
    in_string = False
    string_char = ''

    for char in sql_content:
        if char in ("'", '"'):
            if not in_string:
                in_string = True
                string_char = char
            elif string_char == char:
                in_string = False
        if char == ';' and not in_string:
            stmt = ''.join(buffer).strip()
            if stmt:
                statements.append(stmt)
            buffer = []
        else:
            buffer.append(char)

    # Capture any remaining statement after the loop
    stmt = ''.join(buffer).strip()
    if stmt:
        statements.append(stmt)

    return statements

def extract_copy_data(sql_content):
    """Extract the COPY data from the SQL content."""
    copy_match = re.search(r'COPY.*?FROM stdin;\n(.*?)\n\\.', sql_content, re.DOTALL)
    if copy_match:
        return copy_match.group(1)
    return None

def restore_metrics_from_backup(backup_file_path=None):
    """Restores the metrics table by handing the SQL file directly to psql."""
    logger.info("Starting direct restore of metrics table using psql.")

    # Use provided backup file path or default to metrics_backup.sql
    if backup_file_path:
        backup_file = backup_file_path
        logger.info(f"Using provided backup file: {backup_file}")
    else:
        backup_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'metrics_backup.sql')
        logger.info(f"Using default backup file: {backup_file}")

    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found at: {backup_file}")
        return False

    # Check if DATABASE_URL is available (common for managed services like Replit PostgreSQL)
    database_url = os.getenv("DATABASE_URL")
    
    # Set up environment for psql
    env = os.environ.copy()
    
    if database_url:
        # Use DATABASE_URL directly for psql
        logger.info("Using DATABASE_URL for psql restore")
        
        # Drop existing table first to avoid duplicate errors
        drop_cmd = [
            "psql",
            database_url,
            "-c", "DROP TABLE IF EXISTS public.metrics CASCADE;"
        ]
        
        # Pre-process the dump to remove ownership lines (not needed with DATABASE_URL)
        with open(backup_file, "r", encoding="utf-8") as f:
            dump_sql = f.read()
        
        # Remove ownership lines since they're not needed with DATABASE_URL
        fixed_sql = re.sub(r"OWNER TO\s+\w+;", "", dump_sql, flags=re.IGNORECASE)
        
        # Write to a temporary file for psql to consume
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_metrics_restore.sql")
        with open(tmp.name, "w", encoding="utf-8") as f:
            f.write(fixed_sql)
        
        restore_cmd = [
            "psql",
            database_url,
            "-v", "ON_ERROR_STOP=1",
            "-f", tmp.name
        ]
        
    else:
        # Use individual connection parameters (fallback for local setups)
        logger.info("Using individual connection parameters for psql restore")
        
        # Gather DB connection parameters from environment (fall back to defaults)
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "transparentsf")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Set up environment for psql (password via env var so no prompt)
        if db_password:
            env["PGPASSWORD"] = db_password
        
        # Drop existing table first to avoid duplicate errors
        drop_cmd = [
            "psql",
            "-h", db_host,
            "-p", str(db_port),
            "-U", db_user,
            "-d", db_name,
            "-c", "DROP TABLE IF EXISTS public.metrics CASCADE;"
        ]
        
        # Pre-process the dump so ownership lines reference the current user instead of hard-coded "postgres"
        with open(backup_file, "r", encoding="utf-8") as f:
            dump_sql = f.read()
        
        fixed_sql = re.sub(r"OWNER TO\s+postgres", f"OWNER TO {db_user}", dump_sql, flags=re.IGNORECASE)
        
        # Write to a temporary file for psql to consume
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_metrics_restore.sql")
        with open(tmp.name, "w", encoding="utf-8") as f:
            f.write(fixed_sql)
        
        restore_cmd = [
            "psql",
            "-h", db_host,
            "-p", str(db_port),
            "-U", db_user,
            "-d", db_name,
            "-v", "ON_ERROR_STOP=1",
            "-f", tmp.name
        ]

    try:
        logger.info("Dropping existing metrics table (if any)...")
        subprocess.run(drop_cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Running psql to restore metrics table from backup file...")
        subprocess.run(restore_cmd, check=True, env=env)
        logger.info("Metrics table restoration completed successfully via psql.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("psql command failed. Output:\n%s", e.stderr.decode() if e.stderr else str(e))
        return False

if __name__ == "__main__":
    success = restore_metrics_from_backup()
    if success:
        logger.info("Metrics table restoration process completed successfully.")
    else:
        logger.error("Metrics table restoration process failed.") 
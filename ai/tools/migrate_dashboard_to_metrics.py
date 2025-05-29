#!/usr/bin/env python3
"""
Script to restore the metrics table from a backup file.
"""

import logging
import os
import re
import io
from db_utils import get_postgres_connection, execute_with_connection

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

def restore_metrics_from_backup():
    """Restores the metrics table by handing the SQL file directly to psql."""
    logger.info("Starting direct restore of metrics table using psql.")

    # Path to the backup file
    backup_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'metrics_backup.sql')

    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found at: {backup_file}")
        return False

    # Gather DB connection parameters from environment (fall back to defaults)
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "transparentsf")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")

    # Set up environment for psql (password via env var so no prompt)
    env = os.environ.copy()
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

    restore_cmd = [
        "psql",
        "-h", db_host,
        "-p", str(db_port),
        "-U", db_user,
        "-d", db_name,
        "-v", "ON_ERROR_STOP=1",
        "-f", backup_file
    ]

    import subprocess

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
#!/usr/bin/env python3
"""
Script to restore the metrics table from a backup file.
"""

import logging
import os
from db_utils import get_postgres_connection, execute_with_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_metrics_from_backup():
    """Restores the metrics table from the backup file."""
    logger.info("Starting process to restore metrics table from backup file.")

    # Get the absolute path to the backup file
    backup_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'metrics_backup.sql')
    
    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found at: {backup_file}")
        return False

    def restore_operation(connection):
        cursor = connection.cursor()
        try:
            # Read the backup file
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_sql = f.read()
            
            # Split the SQL file into individual statements
            statements = backup_sql.split(';')
            
            # Execute each statement separately
            for statement in statements:
                # Skip empty statements
                if not statement.strip():
                    continue
                    
                # Add back the semicolon for execution
                statement = statement.strip() + ';'
                
                try:
                    cursor.execute(statement)
                except Exception as e:
                    logger.error(f"Error executing statement: {statement[:100]}... Error: {str(e)}")
                    raise
            
            connection.commit()
            logger.info("Successfully restored metrics table from backup file.")
            return True
        except Exception as e:
            connection.rollback()
            logger.error(f"Error restoring metrics table: {e}")
            raise
        finally:
            cursor.close()

    restore_result = execute_with_connection(operation=restore_operation)

    if restore_result["status"] == "success":
        logger.info("Metrics table restoration completed successfully.")
        return True
    else:
        logger.error(f"Metrics table restoration failed: {restore_result.get('message', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = restore_metrics_from_backup()
    if success:
        logger.info("Metrics table restoration process completed successfully.")
    else:
        logger.error("Metrics table restoration process failed.") 
import psycopg2
import logging
from typing import Optional, Dict, Any
import json
from datetime import date, datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle date and datetime objects."""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

def get_postgres_connection(
    host: str = None,
    port: int = None,
    dbname: str = None,
    user: str = None,
    password: str = None
) -> Optional[psycopg2.extensions.connection]:
    """
    Establish a connection to the PostgreSQL database.
    
    Args:
        host (str): Database host
        port (int): Database port
        dbname (str): Database name
        user (str): Database user
        password (str): Database password
        
    Returns:
        connection: PostgreSQL database connection or None if connection fails
    """
    # Check if DATABASE_URL is provided (common for managed services like Replit PostgreSQL)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            connection = psycopg2.connect(database_url)
            logging.info("Successfully connected to PostgreSQL database using DATABASE_URL")
            return connection
        except Exception as e:
            logging.error(f"Error connecting with DATABASE_URL: {e}")
            # Fall through to individual parameter connection
    
    # Use environment variables if parameters are not provided
    host = host or os.getenv("POSTGRES_HOST", "localhost")
    port = port or int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = dbname or os.getenv("POSTGRES_DB", "transparentsf")
    user = user or os.getenv("POSTGRES_USER", "postgres")
    password = password or os.getenv("POSTGRES_PASSWORD", "postgres")
    
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        logging.info("Successfully connected to PostgreSQL database using individual parameters")
        return connection
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL database: {e}")
        return None

def execute_with_connection(
    operation: callable,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None
) -> Dict[str, Any]:
    """
    Execute a database operation with proper connection handling.
    
    Args:
        operation: Function that takes a connection and returns a result
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        dict: Result with status and message
    """
    # Use environment variables if parameters are not provided
    db_host = db_host or os.getenv("POSTGRES_HOST", "localhost")
    db_port = db_port or int(os.getenv("POSTGRES_PORT", "5432"))
    db_name = db_name or os.getenv("POSTGRES_DB", "transparentsf")
    db_user = db_user or os.getenv("POSTGRES_USER", "postgres")
    db_password = db_password or os.getenv("POSTGRES_PASSWORD", "postgres")
    
    connection = None
    try:
        # Connect to database
        connection = get_postgres_connection(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        
        if connection is None:
            return {
                "status": "error",
                "message": "Failed to connect to database"
            }
        
        # Execute the operation
        result = operation(connection)
        
        return {
            "status": "success",
            "message": "Operation completed successfully",
            "result": result
        }
    
    except Exception as e:
        logging.error(f"Error in database operation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }
    
    finally:
        if connection:
            connection.close() 
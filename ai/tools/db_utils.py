import psycopg2
import logging
from typing import Optional, Dict, Any
import json
from datetime import date, datetime
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from contextlib import contextmanager

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global connection pool engine
_engine: Optional[Engine] = None

def get_connection_pool() -> Engine:
    """
    Get or create the global connection pool engine.
    This uses SQLAlchemy's QueuePool for efficient connection management.
    """
    global _engine
    
    if _engine is None:
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            # Build URL from individual components
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            dbname = os.getenv("POSTGRES_DB", "transparentsf")
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")
            
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
        # Create engine with connection pooling
        _engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,           # Number of connections to maintain in pool
            max_overflow=20,        # Additional connections that can be created
            pool_pre_ping=False,    # Disable pre-ping to avoid transaction conflicts
            pool_recycle=3600,      # Recycle connections after 1 hour
            pool_timeout=30,        # Timeout for getting connection from pool
            echo=False,             # Set to True for SQL debugging
            connect_args={
                "sslmode": "prefer",
                "connect_timeout": 30,
                "keepalives_idle": 600,
                "keepalives_interval": 30,
                "keepalives_count": 3
            }
        )
        
        logging.info("Created connection pool with SQLAlchemy")
    
    return _engine

@contextmanager
def get_pooled_connection():
    """
    Context manager for getting a connection from the pool.
    Automatically handles connection cleanup.
    
    Usage:
        with get_pooled_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics")
            result = cursor.fetchall()
    """
    engine = get_connection_pool()
    connection = None
    
    try:
        # Get connection from pool
        connection = engine.raw_connection()
        # Set autocommit to avoid transaction conflicts
        connection.autocommit = True
        yield connection
    except Exception as e:
        if connection:
            try:
                connection.rollback()
            except:
                pass  # Ignore rollback errors in autocommit mode
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass  # Ignore close errors

def execute_with_pool(operation, *args, **kwargs):
    """
    Execute a database operation using the connection pool.
    
    Args:
        operation: Function that takes a connection and returns a result
        *args, **kwargs: Arguments to pass to the operation function
        
    Returns:
        Result of the operation
    """
    with get_pooled_connection() as conn:
        return operation(conn, *args, **kwargs)

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
    password: str = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[psycopg2.extensions.connection]:
    """
    Establish a connection to the PostgreSQL database with retry logic.
    
    Args:
        host (str): Database host
        port (int): Database port
        dbname (str): Database name
        user (str): Database user
        password (str): Database password
        max_retries (int): Maximum number of connection retry attempts
        retry_delay (float): Delay between retry attempts in seconds
        
    Returns:
        connection: PostgreSQL database connection or None if connection fails
    """
    import time
    
    # Check if DATABASE_URL is provided (common for managed services like Replit PostgreSQL)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        for attempt in range(max_retries):
            try:
                # Add SSL and connection parameters for better reliability
                connection = psycopg2.connect(
                    database_url,
                    sslmode='prefer',  # Use SSL if available, but don't require it
                    connect_timeout=30,  # 30 second connection timeout
                    keepalives_idle=600,  # Send keepalive packets every 10 minutes
                    keepalives_interval=30,  # Wait 30 seconds for keepalive response
                    keepalives_count=3  # Close connection after 3 failed keepalives
                )
                logging.info("Successfully connected to PostgreSQL database using DATABASE_URL")
                return connection
            except Exception as e:
                logging.warning(f"Connection attempt {attempt + 1} failed with DATABASE_URL: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logging.error(f"All {max_retries} connection attempts failed with DATABASE_URL")
                    # Fall through to individual parameter connection
    
    # Use environment variables if parameters are not provided
    host = host or os.getenv("POSTGRES_HOST", "localhost")
    port = port or int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = dbname or os.getenv("POSTGRES_DB", "transparentsf")
    user = user or os.getenv("POSTGRES_USER", "postgres")
    password = password or os.getenv("POSTGRES_PASSWORD", "postgres")
    
    for attempt in range(max_retries):
        try:
            connection = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                sslmode='prefer',  # Use SSL if available, but don't require it
                connect_timeout=30,  # 30 second connection timeout
                keepalives_idle=600,  # Send keepalive packets every 10 minutes
                keepalives_interval=30,  # Wait 30 seconds for keepalive response
                keepalives_count=3  # Close connection after 3 failed keepalives
            )
            logging.info("Successfully connected to PostgreSQL database using individual parameters")
            return connection
        except Exception as e:
            logging.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logging.error(f"All {max_retries} connection attempts failed")
                return None

def execute_with_connection(
    operation: callable,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Execute a database operation with proper connection handling and retry logic.
    Now uses connection pool for better performance.
    
    Args:
        operation: Function that takes a connection and returns a result
        db_host: Database host (ignored when using connection pool)
        db_port: Database port (ignored when using connection pool)
        db_name: Database name (ignored when using connection pool)
        db_user: Database user (ignored when using connection pool)
        db_password: Database password (ignored when using connection pool)
        max_retries: Maximum number of retry attempts for connection failures
        
    Returns:
        dict: Result with status and message
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Use connection pool for better performance
            with get_pooled_connection() as connection:
                # Execute the operation
                result = operation(connection)
                
                return {
                    "status": "success",
                    "message": "Operation completed successfully",
                    "result": result
                }
        
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Check if this is a connection-related error that we should retry
            if any(keyword in error_msg for keyword in [
                'ssl connection has been closed',
                'connection already closed',
                'server closed the connection',
                'connection refused',
                'timeout expired',
                'connection reset'
            ]):
                logging.warning(f"Database connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Wait before retrying with exponential backoff
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying database operation in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"All {max_retries} database operation attempts failed")
            else:
                # Non-retryable error, fail immediately
                logging.error(f"Non-retryable database error: {e}")
                break
                
        except Exception as e:
            last_exception = e
            logging.error(f"Unexpected error in database operation: {e}")
            import traceback
            logging.error(traceback.format_exc())
            break
    
    # If we get here, all retries failed
    error_message = str(last_exception) if last_exception else "Unknown database error"
    return {
        "status": "error",
        "message": f"Database operation failed after {max_retries} attempts: {error_message}"
    }

def execute_with_pool_connection(operation: callable, max_retries: int = 3) -> Dict[str, Any]:
    """
    Execute a database operation using the connection pool with retry logic.
    This is the recommended way to execute database operations for better performance.
    
    Args:
        operation: Function that takes a connection and returns a result
        max_retries: Maximum number of retry attempts for connection failures
        
    Returns:
        dict: Result with status and message
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Use the connection pool
            result = execute_with_pool(operation)
            
            return {
                "status": "success",
                "message": "Operation completed successfully",
                "result": result
            }
            
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Check if this is a connection-related error that we should retry
            if any(keyword in error_msg for keyword in [
                'ssl connection has been closed',
                'connection already closed',
                'server closed the connection',
                'connection refused',
                'timeout expired',
                'connection reset',
                'pool is closed'
            ]):
                logging.warning(f"Database connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Wait before retrying with exponential backoff
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying database operation in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"All {max_retries} database operation attempts failed")
            else:
                # Non-retryable error, fail immediately
                logging.error(f"Non-retryable database error: {e}")
                break
    
    # If we get here, all retries failed
    error_message = str(last_exception) if last_exception else "Unknown database error"
    return {
        "status": "error",
        "message": f"Database operation failed after {max_retries} attempts: {error_message}"
    }

def check_database_health(
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None
) -> Dict[str, Any]:
    """
    Check the health of the database connection and return diagnostic information.
    
    Args:
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        dict: Health check results with status and diagnostic information
    """
    import time
    
    # Use environment variables if parameters are not provided
    db_host = db_host or os.getenv("POSTGRES_HOST", "localhost")
    db_port = db_port or int(os.getenv("POSTGRES_PORT", "5432"))
    db_name = db_name or os.getenv("POSTGRES_DB", "transparentsf")
    db_user = db_user or os.getenv("POSTGRES_USER", "postgres")
    db_password = db_password or os.getenv("POSTGRES_PASSWORD", "postgres")
    
    health_info = {
        "timestamp": time.time(),
        "connection_params": {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user
        },
        "ssl_info": {},
        "connection_test": {},
        "query_test": {}
    }
    
    connection = None
    try:
        # Test basic connection
        start_time = time.time()
        connection = get_postgres_connection(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            max_retries=1
        )
        connection_time = time.time() - start_time
        
        if connection is None:
            health_info["status"] = "error"
            health_info["message"] = "Failed to establish database connection"
            return health_info
        
        # Get SSL information
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT version(), ssl_is_used(), ssl_version()")
            ssl_info = cursor.fetchone()
            if ssl_info:
                health_info["ssl_info"] = {
                    "version": ssl_info[0],
                    "ssl_used": ssl_info[1] if len(ssl_info) > 1 else None,
                    "ssl_version": ssl_info[2] if len(ssl_info) > 2 else None
                }
            cursor.close()
        except Exception as ssl_error:
            health_info["ssl_info"]["error"] = str(ssl_error)
        
        # Test a simple query
        try:
            start_time = time.time()
            cursor = connection.cursor()
            cursor.execute("SELECT 1 as test_query")
            result = cursor.fetchone()
            query_time = time.time() - start_time
            cursor.close()
            
            health_info["query_test"] = {
                "success": True,
                "result": result[0] if result else None,
                "execution_time": query_time
            }
        except Exception as query_error:
            health_info["query_test"] = {
                "success": False,
                "error": str(query_error)
            }
        
        health_info["connection_test"] = {
            "success": True,
            "connection_time": connection_time,
            "connection_status": "open" if connection.closed == 0 else "closed"
        }
        
        health_info["status"] = "healthy"
        health_info["message"] = "Database connection is healthy"
        
    except Exception as e:
        health_info["status"] = "error"
        health_info["message"] = f"Database health check failed: {str(e)}"
        health_info["connection_test"] = {
            "success": False,
            "error": str(e)
        }
    
    finally:
        if connection:
            try:
                connection.close()
            except Exception as close_error:
                logging.warning(f"Error closing connection during health check: {close_error}")
    
    return health_info 
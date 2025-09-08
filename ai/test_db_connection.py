#!/usr/bin/env python3
"""
Database connection test script for TransparentSF.
This script helps diagnose database connection issues.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the ai directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.db_utils import check_database_health, get_postgres_connection, execute_with_connection
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_connection():
    """Test basic database connection."""
    logger.info("Testing basic database connection...")
    
    load_dotenv()
    
    # Get connection parameters
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = int(os.getenv("POSTGRES_PORT", "5432"))
    db_name = os.getenv("POSTGRES_DB", "transparentsf")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    logger.info(f"Connection parameters: {db_host}:{db_port}/{db_name} (user: {db_user})")
    
    # Test basic connection
    connection = get_postgres_connection(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    
    if connection:
        logger.info("✅ Basic connection successful")
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            logger.info(f"PostgreSQL version: {version[0] if version else 'Unknown'}")
            cursor.close()
        except Exception as e:
            logger.error(f"Error querying version: {e}")
        finally:
            connection.close()
    else:
        logger.error("❌ Basic connection failed")
        return False
    
    return True

def test_health_check():
    """Test database health check."""
    logger.info("Running database health check...")
    
    health_result = check_database_health()
    
    logger.info(f"Health check status: {health_result['status']}")
    logger.info(f"Health check message: {health_result['message']}")
    
    if health_result['status'] == 'healthy':
        logger.info("✅ Database health check passed")
        logger.info(f"Connection time: {health_result['connection_test'].get('connection_time', 'N/A')} seconds")
        logger.info(f"Query time: {health_result['query_test'].get('execution_time', 'N/A')} seconds")
        
        ssl_info = health_result.get('ssl_info', {})
        if ssl_info:
            logger.info(f"SSL used: {ssl_info.get('ssl_used', 'Unknown')}")
            logger.info(f"SSL version: {ssl_info.get('ssl_version', 'Unknown')}")
    else:
        logger.error("❌ Database health check failed")
        logger.error(f"Error details: {json.dumps(health_result, indent=2)}")
    
    return health_result['status'] == 'healthy'

def test_operation_with_retry():
    """Test database operation with retry logic."""
    logger.info("Testing database operation with retry logic...")
    
    def test_operation(connection):
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else 0
    
    result = execute_with_connection(
        operation=test_operation,
        max_retries=3
    )
    
    if result['status'] == 'success':
        table_count = result['result']
        logger.info(f"✅ Database operation successful - found {table_count} tables")
        return True
    else:
        logger.error(f"❌ Database operation failed: {result['message']}")
        return False

def main():
    """Main test function."""
    logger.info("Starting database connection tests...")
    
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Health Check", test_health_check),
        ("Operation with Retry", test_operation_with_retry)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! Database connection is working properly.")
    else:
        logger.error("\n⚠️  Some tests failed. Check the logs above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

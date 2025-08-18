#!/usr/bin/env python3
"""
Export Neighborhood Business Data to CSV
=======================================

This script pulls raw data for each of the 8 neighborhood business metrics (44-53)
and saves each into its own CSV file.

Metrics:
- 44: Castro/Upper Market Business Openings
- 45: Castro/Upper Market Business Closures  
- 46: North Beach - Business Openings
- 47: North Beach - Business Closings
- 48: Chinatown Business Closures
- 49: Chinatown Business Openings
- 50: West Portal - Business Openings
- 51: West Portal - Business Closings
- 52: Calle 24 - Business Openings
- 53: Calle 24 - Business Closings
"""

import os
import sys
import logging
import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Add the ai directory to the path so we can import our tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.data_fetcher import fetch_metric_data

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_metric_details(metric_id):
    """
    Get metric details from the database.
    
    Args:
        metric_id (int): The metric ID
        
    Returns:
        dict: Metric details or None if not found
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT * FROM metrics WHERE id = %s", [metric_id])
        metric = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return metric
        
    except Exception as e:
        logger.error(f"Error getting metric details for {metric_id}: {str(e)}")
        return None

def export_metric_to_csv(metric_id, output_dir):
    """
    Export a single metric's raw data to CSV.
    
    Args:
        metric_id (int): The metric ID
        output_dir (str): Output directory for Excel files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Processing metric {metric_id}...")
        
        # Get metric details
        metric = get_metric_details(metric_id)
        if not metric:
            logger.error(f"Metric {metric_id} not found")
            return False
            
        metric_name = metric['metric_name']
        logger.info(f"Found metric: {metric_name}")
        
        # Use the actual YTD query from the database for proper business logic
        from tools.data_fetcher import set_dataset
        
        # Get the YTD query which contains the correct business logic
        ytd_query = metric.get('ytd_query', '')
        endpoint = metric.get('endpoint', '')
        
        if not ytd_query:
            logger.error(f"No YTD query found for metric {metric_id}")
            return False
        
        # For export purposes, we want to get the raw data, not the aggregated YTD data
        # So we'll extract the WHERE clause from the YTD query and use it with SELECT *
        import re
        
        # Extract the WHERE clause from the YTD query
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP BY|\s*$)', ytd_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Replace date placeholders with actual dates for export
            # Use January 1, 2024 as start date to match the API query
            from datetime import date, timedelta
            end_date = date.today() - timedelta(days=1)
            start_date = date(2024, 1, 1)  # January 1, 2024
            
            where_clause = where_clause.replace('last_year_start', f"'{start_date.strftime('%Y-%m-%d')}'")
            where_clause = where_clause.replace('current_date', f"'{end_date.strftime('%Y-%m-%d')}'")
            where_clause = where_clause.replace('this_year_start', f"'{start_date.strftime('%Y-%m-%d')}'")
            where_clause = where_clause.replace('this_year_end', f"'{end_date.strftime('%Y-%m-%d')}'")
            where_clause = where_clause.replace('last_year_end', f"'{end_date.strftime('%Y-%m-%d')}'")
            
            query = f"SELECT * WHERE {where_clause} LIMIT 50000"
        else:
            # Fallback to simple query if we can't parse the WHERE clause
            query = f"SELECT * WHERE neighborhoods_analysis_boundaries = 'Castro/Upper Market' LIMIT 50000"
        
        if not query:
            logger.error(f"Failed to build query for metric {metric_id}")
            return False
        
        logger.info(f"Built export query: {query}")
        
        # Fetch data using the existing set_dataset function
        context_variables = {}
        result = set_dataset(context_variables, endpoint=endpoint, query=query)
        
        if 'error' in result:
            logger.error(f"Error fetching data for metric {metric_id}: {result['error']}")
            return False
            
        data = context_variables.get('dataset')
        if data is None or data.empty:
            logger.warning(f"No data returned for metric {metric_id}")
            return False
            
        logger.info(f"Retrieved {len(data)} records for metric {metric_id}")
        
        # Create safe filename
        safe_name = metric_name.replace('/', '_').replace(' ', '_').replace('🏳️‍🌈', 'Castro').replace('🇮🇹', 'NorthBeach').replace('🏮', 'Chinatown').replace('🏪', 'WestPortal').replace('🌮', 'Calle24')
        filename = f"metric_{metric_id:02d}_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Export to CSV
        data.to_csv(filepath, index=False)
        logger.info(f"Exported {len(data)} records to {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting metric {metric_id}: {str(e)}")
        return False

def main():
    """
    Main function to export all neighborhood business metrics.
    """
    # Define the metric IDs for neighborhood business metrics
    metric_ids = list(range(44, 54))  # 44-53
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "neighborhood_business_exports")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting export of {len(metric_ids)} neighborhood business metrics...")
    logger.info(f"Output directory: {output_dir}")
    
    # Track results
    successful_exports = []
    failed_exports = []
    
    # Export each metric
    for metric_id in metric_ids:
        success = export_metric_to_csv(metric_id, output_dir)
        if success:
            successful_exports.append(metric_id)
        else:
            failed_exports.append(metric_id)
    
    # Summary
    logger.info("=" * 50)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Successful exports: {len(successful_exports)}")
    logger.info(f"Failed exports: {len(failed_exports)}")
    
    if successful_exports:
        logger.info(f"Successfully exported metrics: {successful_exports}")
    
    if failed_exports:
        logger.error(f"Failed to export metrics: {failed_exports}")
    
    logger.info(f"Files saved to: {os.path.abspath(output_dir)}")
    
    return len(failed_exports) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

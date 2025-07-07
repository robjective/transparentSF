#!/usr/bin/env python3
"""
Script to regenerate YTD trend chart for chart ID 26240 with corrected data structure.
"""

import sys
import os
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta

# Add the ai directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.genChart import generate_ytd_trend_chart
from tools.store_time_series import store_time_series_in_db
from tools.db_utils import get_postgres_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_chart_data(chart_id):
    """Get the original chart data from the database."""
    try:
        conn = get_postgres_connection()
        if not conn:
            logger.error("Failed to connect to database")
            return None, None
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get chart metadata
        cursor.execute("""
            SELECT chart_id, object_name, field_name, period_type, district, metadata
            FROM time_series_metadata 
            WHERE chart_id = %s
        """, (chart_id,))
        
        metadata_result = cursor.fetchone()
        if not metadata_result:
            logger.error(f"Chart {chart_id} not found")
            return None, None
        
        # Get chart data points
        cursor.execute("""
            SELECT time_period, numeric_value
            FROM time_series_data 
            WHERE chart_id = %s
            ORDER BY time_period
        """, (chart_id,))
        
        data_results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert to trend_data format
        trend_data = {}
        for row in data_results:
            date_str = row['time_period'].strftime('%Y-%m-%d')
            trend_data[date_str] = float(row['numeric_value'])
        
        return trend_data, metadata_result
        
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return None, None

def regenerate_ytd_chart(chart_id):
    """Regenerate the YTD trend chart with corrected data structure."""
    logger.info(f"Regenerating YTD trend chart for chart ID {chart_id}")
    
    # Get original data
    trend_data, metadata_result = get_chart_data(chart_id)
    if not trend_data:
        logger.error("Failed to get chart data")
        return False
    
    logger.info(f"Retrieved {len(trend_data)} data points")
    
    # Prepare metadata for regeneration
    metadata = {
        "chart_title": metadata_result['object_name'],
        "y_axis_label": "Count",
        "object_type": "metric",
        "object_id": "unknown",
        "object_name": metadata_result['object_name'],
        "field_name": metadata_result['field_name'],
        "district": str(metadata_result['district']) if metadata_result['district'] else None
    }
    
    # Generate the chart with corrected data structure
    logger.info("Generating YTD trend chart with corrected data structure...")
    result = generate_ytd_trend_chart(
        trend_data=trend_data,
        metadata=metadata,
        district=metadata["district"],
        return_html=False,
        store_in_db=True
    )
    
    if result and not result.startswith("**Error**"):
        logger.info("✅ YTD trend chart regenerated successfully")
        return True
    else:
        logger.error(f"❌ Failed to regenerate chart: {result}")
        return False

def main():
    """Main function to regenerate chart ID 26240."""
    chart_id = 26240
    
    print(f"🔄 Regenerating YTD Trend Chart (ID: {chart_id})")
    print("=" * 60)
    
    success = regenerate_ytd_chart(chart_id)
    
    if success:
        print(f"✅ Successfully regenerated YTD trend chart for chart ID {chart_id}")
        print("The chart should now display as a dual-line chart with this year and last year data.")
        print("You can view it at: http://localhost:8000/backend/time-series-chart?chart_id=<new_chart_id>")
    else:
        print(f"❌ Failed to regenerate YTD trend chart for chart ID {chart_id}")

if __name__ == "__main__":
    main() 
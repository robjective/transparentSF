import os
import json
import logging
import traceback
import argparse
from datetime import datetime, date, timedelta
import pandas as pd
import re
from pathlib import Path
import copy
from typing import Dict, Any, List, Optional, Tuple
import textwrap
import sys

# Add current directory to path for direct script execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Handle imports for both direct execution and package import
try:
    # Try relative imports first (for package import)
    from .data_fetcher import set_dataset
    from .db_utils import get_postgres_connection, execute_with_connection
    from .genChart import generate_time_series_chart
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from data_fetcher import set_dataset
    from db_utils import get_postgres_connection, execute_with_connection
    from genChart import generate_time_series_chart

# Configure logging using the same approach as main.py
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Set output directory - one level up from tools directory
OUTPUT_DIR = os.path.join(script_dir, '..', 'output', 'records')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Get log level from environment variable
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_level_map.get(log_level_str, logging.INFO)

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Remove any existing handlers from root logger
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add file handler to root logger
root_file_handler = logging.FileHandler(os.path.join(logs_dir, 'detect_records.log'))
root_file_handler.setLevel(log_level)
root_file_handler.setFormatter(formatter)
root_logger.addHandler(root_file_handler)

# Add console handler to root logger
root_console_handler = logging.StreamHandler()
root_console_handler.setLevel(log_level)
root_console_handler.setFormatter(formatter)
root_logger.addHandler(root_console_handler)

# Now configure the module logger
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Log a message to confirm logging is set up
logger.info(f"Logging configured for detect_records.py with level: {log_level_str}")
root_logger.info(f"Root logger configured for detect_records.py with level: {log_level_str}")

def get_chart_data(chart_id: int) -> Optional[Dict[str, Any]]:
    """
    Get chart metadata and data from the database.
    
    Args:
        chart_id: The chart ID to retrieve
        
    Returns:
        Dictionary containing chart metadata and data, or None if not found
    """
    def get_chart_operation(connection):
        cursor = connection.cursor()
        
        # Get chart metadata
        cursor.execute("""
            SELECT 
                chart_id, 
                chart_title, 
                y_axis_label, 
                period_type, 
                object_type, 
                object_id, 
                object_name, 
                field_name, 
                district, 
                group_field,
                executed_query_url,
                metadata
            FROM time_series_metadata 
            WHERE chart_id = %s
        """, (chart_id,))
        
        metadata_result = cursor.fetchone()
        
        if not metadata_result:
            return None
        
        # Get chart data points
        cursor.execute("""
            SELECT 
                time_period, 
                group_value, 
                numeric_value
            FROM time_series_data 
            WHERE chart_id = %s
            ORDER BY time_period
        """, (chart_id,))
        
        data_results = cursor.fetchall()
        cursor.close()
        
        return {
            'metadata': metadata_result,
            'data': data_results
        }
    
    result = execute_with_connection(get_chart_operation)
    
    if result['status'] == 'success' and result['result']:
        return result['result']
    else:
        logger.error(f"Failed to get chart data: {result.get('message', 'Unknown error')}")
        return None

def get_metric_info(metric_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metric information from the database.
    
    Args:
        metric_id: The metric ID to retrieve
        
    Returns:
        Dictionary containing metric information, or None if not found
    """
    def get_metric_operation(connection):
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT 
                id,
                metric_name,
                metric_key,
                category,
                subcategory,
                endpoint,
                summary,
                definition,
                data_sf_url,
                ytd_query,
                metric_query,
                dataset_title,
                dataset_category,
                show_on_dash,
                item_noun,
                greendirection,
                location_fields,
                category_fields,
                metadata
            FROM metrics
            WHERE id = %s AND is_active = true
        """, (metric_id,))
        
        metric_row = cursor.fetchone()
        cursor.close()
        
        if not metric_row:
            return None
        
        return {
            'metric_id': str(metric_row[0]),
            'query_name': metric_row[1],
            'top_category': metric_row[3],
            'subcategory': metric_row[4],
            'endpoint': metric_row[5],
            'summary': metric_row[6],
            'definition': metric_row[7],
            'data_sf_url': metric_row[8],
            'category_fields': metric_row[16] or [],
            'location_fields': metric_row[15] or [],
            'query_data': {
                'ytd_query': metric_row[9],
                'metric_query': metric_row[10],
                'id': metric_row[0],
                'endpoint': metric_row[5]
            }
        }
    
    result = execute_with_connection(get_metric_operation)
    
    if result['status'] == 'success' and result['result']:
        return result['result']
    else:
        logger.error(f"Failed to get metric info: {result.get('message', 'Unknown error')}")
        return None

def analyze_series_for_records(series_data: List[Tuple], series_name: str = "Overall", 
                              metric_info: Dict[str, Any] = None, period_type: str = 'month', 
                              district: int = None) -> Dict[str, Any]:
    """
    Analyze a single series to detect record highs and lows.
    
    Args:
        series_data: List of (time_period, numeric_value) tuples
        series_name: Name of the series for logging
        metric_info: Metric information for historical data querying
        period_type: Period type for historical data querying
        
    Returns:
        Dictionary containing record analysis results
    """
    if not series_data:
        return {'error': f'No data available for {series_name}'}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(series_data, columns=['time_period', 'numeric_value'])
    df['time_period'] = pd.to_datetime(df['time_period'])
    df = df.sort_values('time_period')
    
    # Get the most recent value
    most_recent = df.iloc[-1]
    most_recent_period = most_recent['time_period']
    most_recent_value = most_recent['numeric_value']
    
    # Check if it's a record high
    historical_max = df['numeric_value'].max()
    is_record_high = most_recent_value >= historical_max
    
    # Check if it's a record low
    historical_min = df['numeric_value'].min()
    is_record_low = most_recent_value <= historical_min
    
    # Initialize historical data variables
    last_high_period = None
    years_since_last_high = None
    last_low_period = None
    years_since_last_low = None
    
    # Get the earliest date from the endpoint for accurate context
    earliest_date_from_endpoint = None
    if metric_info:
        earliest_date_from_endpoint = get_earliest_date_from_endpoint(metric_info, period_type, district)
        if earliest_date_from_endpoint:
            logger.info(f"Found earliest date from endpoint: {earliest_date_from_endpoint}")
        else:
            logger.warning("Could not get earliest date from endpoint, using chart data")
    
    # Query historical data if we have metric_info and it's a record
    if metric_info and (is_record_high or is_record_low):
        logger.info(f"Querying historical data for {series_name} (value: {most_recent_value})")
        
        # Extract group value from series name if it's not "Overall"
        group_value = None
        if series_name != "Overall":
            group_value = series_name
        
        # Get the current month being evaluated to exclude it from historical search
        exclude_month = None
        if period_type == 'month':
            # Format the most recent period as YYYY-MM for exclusion
            exclude_month = most_recent_period.strftime('%Y-%m')
            logger.info(f"Excluding current month from historical search: {exclude_month}")
        
        if is_record_high:
            # Query for historical high values
            historical_result = query_historical_data(
                metric_info, most_recent_value, True, period_type, group_value, exclude_month, most_recent_period, district
            )
            
            if historical_result and historical_result.get('last_occurrence'):
                last_high_period = historical_result['last_occurrence']
                years_since_last_high = historical_result['years_since_last']
                # Get the historical value from the dataset
                historical_value = None
                if historical_result.get('data') is not None and not historical_result['data'].empty:
                    historical_value = historical_result['data'].iloc[0]['value']
                if years_since_last_high is not None:
                    logger.info(f"Found historical high: {last_high_period} ({years_since_last_high:.1f} years ago) with value {historical_value}")
                else:
                    logger.info(f"Found historical high: {last_high_period} (could not calculate time difference) with value {historical_value}")
            else:
                logger.info("No historical high found - this is a new record")
                historical_value = None
        
        if is_record_low:
            # Query for historical low values
            historical_result = query_historical_data(
                metric_info, most_recent_value, False, period_type, group_value, exclude_month, most_recent_period, district
            )
            
            if historical_result and historical_result.get('last_occurrence'):
                last_low_period = historical_result['last_occurrence']
                years_since_last_low = historical_result['years_since_last']
                # Get the historical value from the dataset
                historical_value = None
                if historical_result.get('data') is not None and not historical_result['data'].empty:
                    historical_value = historical_result['data'].iloc[0]['value']
                if years_since_last_low is not None:
                    logger.info(f"Found historical low: {last_low_period} ({years_since_last_low:.1f} years ago) with value {historical_value}")
                else:
                    logger.info(f"Found historical low: {last_low_period} (could not calculate time difference) with value {historical_value}")
            else:
                logger.info("No historical low found - this is a new record")
                historical_value = None
    
    # Fallback to chart data analysis if no historical data available
    if is_record_high and last_high_period is None:
        # Find the last time it was this high before the current period
        previous_highs = df[(df['numeric_value'] >= most_recent_value) & 
                           (df['time_period'] < most_recent_period)]
        if not previous_highs.empty:
            last_high_period = previous_highs.iloc[-1]['time_period']
            years_since_last_high = (most_recent_period - last_high_period).days / 365.25
    
    if is_record_low and last_low_period is None:
        # Find the last time it was this low before the current period
        previous_lows = df[(df['numeric_value'] <= most_recent_value) & 
                          (df['time_period'] < most_recent_period)]
        if not previous_lows.empty:
            last_low_period = previous_lows.iloc[-1]['time_period']
            years_since_last_low = (most_recent_period - last_low_period).days / 365.25
    
    # Get the earliest date in the dataset
    earliest_date = df['time_period'].min()
    
    # Find the period when the historical minimum occurred
    historical_min_period = None
    if not df.empty:
        min_row = df[df['numeric_value'] == historical_min]
        if not min_row.empty:
            historical_min_period = min_row.iloc[0]['time_period']
    
    return {
        'series_name': series_name,
        'most_recent_period': most_recent_period,
        'most_recent_value': most_recent_value,
        'is_record_high': is_record_high,
        'is_record_low': is_record_low,
        'historical_max': historical_max,
        'historical_min': historical_min,
        'historical_min_period': historical_min_period,
        'last_high_period': last_high_period,
        'years_since_last_high': years_since_last_high,
        'last_high_value': historical_value if is_record_high else None,
        'last_low_period': last_low_period,
        'years_since_last_low': years_since_last_low,
        'last_low_value': historical_value if is_record_low else None,
        'total_periods': len(df),
        'earliest_date': earliest_date,
        'earliest_date_from_endpoint': earliest_date_from_endpoint
    }

def format_period_display(period, period_type: str) -> str:
    """
    Format a period for display based on period type.
    
    Args:
        period: The period datetime or string
        period_type: The period type (month, year, etc.)
        
    Returns:
        Formatted period string
    """
    # Handle string periods (from historical data queries)
    if isinstance(period, str):
        try:
            # First try to parse as ISO datetime string (most common from APIs)
            if 'T' in period:
                # Handle ISO datetime strings like "2018-01-01T00:00:00.000"
                dt = datetime.fromisoformat(period.replace('Z', '+00:00'))
                if period_type == 'month':
                    return dt.strftime('%B %Y')
                elif period_type == 'year':
                    return dt.strftime('%Y')
                else:
                    return dt.strftime('%B %d, %Y')
            
            # Handle other string formats
            if period_type == 'month':
                # Handle formats like "2023-05" or "2023-05-01"
                if len(period) == 7:  # "2023-05"
                    dt = datetime.strptime(period, '%Y-%m')
                else:  # "2023-05-01" or other formats
                    dt = datetime.strptime(period, '%Y-%m-%d')
                return dt.strftime('%B %Y')
            elif period_type == 'year':
                # Handle formats like "2023"
                dt = datetime.strptime(period, '%Y')
                return dt.strftime('%Y')
            else:
                # Try to parse as full date
                dt = datetime.strptime(period, '%Y-%m-%d')
                return dt.strftime('%B %d, %Y')
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse period string '{period}': {e}")
            return str(period)  # Return as-is if parsing fails
    
    # Handle datetime objects
    elif isinstance(period, datetime):
        if period_type == 'month':
            return period.strftime('%B %Y')
        elif period_type == 'year':
            return period.strftime('%Y')
        else:
            return period.strftime('%B %d, %Y')
    
    # Handle other types (like pandas Timestamp)
    else:
        try:
            # Convert to datetime if possible
            dt = pd.to_datetime(period)
            if period_type == 'month':
                return dt.strftime('%B %Y')
            elif period_type == 'year':
                return dt.strftime('%Y')
            else:
                return dt.strftime('%B %d, %Y')
        except Exception as e:
            logger.warning(f"Could not format period '{period}' of type {type(period)}: {e}")
            return str(period)  # Return as-is if all else fails

def generate_record_headline(analysis: Dict[str, Any], metric_name: str, district: int = 0) -> str:
    """
    Generate a headline for a record high or low.
    
    Args:
        analysis: The record analysis results
        metric_name: The name of the metric
        district: The district number (0 for citywide)
        
    Returns:
        Formatted headline
    """
    series_name = analysis['series_name']
    most_recent_period = analysis['most_recent_period']
    most_recent_value = analysis['most_recent_value']
    
    # Format the period for display
    period_display = format_period_display(most_recent_period, 'month')  # Assume monthly for now
    
    # Format the value (round to appropriate decimal places)
    if most_recent_value.is_integer():
        value_display = f"{int(most_recent_value):,}"
    else:
        value_display = f"{most_recent_value:,.1f}"
    
    # Determine district text
    if district == 0:
        district_text = "Citywide"
    else:
        district_text = f"District {district}"
    
    # Get the earliest date from the endpoint (more accurate than chart data)
    earliest_date = analysis.get('earliest_date_from_endpoint')
    if not earliest_date:
        # Fallback to chart data if endpoint query failed
        earliest_date = analysis.get('earliest_date')
    
    earliest_date_display = ""
    years_since_start = None
    if earliest_date:
        earliest_date_display = format_period_display(earliest_date, 'month')
        # Calculate years since start
        try:
            earliest_dt = datetime.fromisoformat(earliest_date.replace('Z', '+00:00'))
            current_dt = datetime.now()
            years_since_start = current_dt.year - earliest_dt.year
        except:
            years_since_start = 7  # Default to 7 years if we can't parse
    else:
        years_since_start = 7  # Default to 7 years
    
    # Generate the headline
    if analysis['is_record_high']:
        if analysis['years_since_last_high'] is None:
            # Never been this high before
            if earliest_date_display:
                headline = f"{metric_name} at highest level since {earliest_date_display}"
            else:
                headline = f"{metric_name} at highest level in recorded history"
        else:
            years = analysis['years_since_last_high']
            if years >= 1:
                headline = f"{metric_name} at highest level in {years:.1f} years"
            else:
                months = years * 12
                headline = f"{metric_name} at highest level in {months:.0f} months"
    elif analysis['is_record_low']:
        if analysis['years_since_last_low'] is None:
            # Never been this low before - this is the lowest on record
            if earliest_date_display:
                headline = f"{metric_name} at lowest level in {years_since_start} years, and the lowest on record"
            else:
                headline = f"{metric_name} at lowest level in recorded history"
        else:
            years = analysis['years_since_last_low']
            if years >= 1:
                headline = f"{metric_name} at lowest level in {years:.1f} years"
            else:
                months = years * 12
                headline = f"{metric_name} at lowest level in {months:.0f} months"
    else:
        # Not a record
        headline = f"{metric_name} at {value_display} in {period_display}"
    
    return f"{district_text} {headline}"

def generate_record_description(analysis: Dict[str, Any], metric_name: str, district: int = 0) -> str:
    """
    Generate a detailed description for a record high or low.
    
    Args:
        analysis: The record analysis results
        metric_name: The name of the metric
        district: The district number (0 for citywide)
        
    Returns:
        Formatted description
    """
    series_name = analysis['series_name']
    most_recent_period = analysis['most_recent_period']
    most_recent_value = analysis['most_recent_value']
    
    # Format the period for display
    period_display = format_period_display(most_recent_period, 'month')
    
    # Format the value
    if most_recent_value.is_integer():
        value_display = f"{int(most_recent_value):,}"
    else:
        value_display = f"{most_recent_value:,.1f}"
    
    # Determine district text
    if district == 0:
        district_text = "Citywide"
    else:
        district_text = f"District {district}"
    
    # Get the earliest date from the endpoint (more accurate than chart data)
    earliest_date = analysis.get('earliest_date_from_endpoint')
    if not earliest_date:
        # Fallback to chart data if endpoint query failed
        earliest_date = analysis.get('earliest_date')
    
    earliest_date_display = ""
    if earliest_date:
        earliest_date_display = format_period_display(earliest_date, 'month')
    
    description = f"In {period_display}, {district_text} {metric_name} were at {value_display}."
    
    if analysis['is_record_high']:
        if analysis['years_since_last_high'] is None:
            if earliest_date_display:
                description += f" That's the highest they have been since {earliest_date_display}."
            else:
                description += " That's the highest they have been in recorded history."
        else:
            years = analysis['years_since_last_high']
            last_high_period = analysis['last_high_period']
            last_high_display = format_period_display(last_high_period, 'month')
            
            # Use historical value if available, otherwise use current value
            historical_value = analysis.get('last_high_value', most_recent_value)
            if historical_value is not None:
                # Convert to float first, then check if it's an integer
                try:
                    historical_float = float(historical_value)
                    if historical_float.is_integer():
                        historical_display = f"{int(historical_float):,}"
                    else:
                        historical_display = f"{historical_float:,.1f}"
                except (ValueError, TypeError):
                    # If we can't convert to float, just use the string
                    historical_display = str(historical_value)
            else:
                historical_display = value_display
            
            if years >= 1:
                description += f" That's the highest they have been in more than {years:.1f} years, since {last_high_display} when they were {historical_display}."
            else:
                months = years * 12
                description += f" That's the highest they have been in {months:.0f} months, since {last_high_display} when they were {historical_display}."
    
    elif analysis['is_record_low']:
        if analysis['years_since_last_low'] is None:
            if earliest_date_display:
                description += f" That's the lowest they have been since {earliest_date_display}."
            else:
                description += " That's the lowest they have been in recorded history."
        else:
            years = analysis['years_since_last_low']
            last_low_period = analysis['last_low_period']
            last_low_display = format_period_display(last_low_period, 'month')
            
            # Use historical value if available, otherwise use current value
            historical_value = analysis.get('last_low_value', most_recent_value)
            if historical_value is not None:
                # Convert to float first, then check if it's an integer
                try:
                    historical_float = float(historical_value)
                    if historical_float.is_integer():
                        historical_display = f"{int(historical_float):,}"
                    else:
                        historical_display = f"{historical_float:,.1f}"
                except (ValueError, TypeError):
                    # If we can't convert to float, just use the string
                    historical_display = str(historical_value)
            else:
                historical_display = value_display
            
            if years >= 1:
                description += f" That's the lowest they have been in more than {years:.1f} years, since {last_low_display} when they were {historical_display}."
            else:
                months = years * 12
                description += f" That's the lowest they have been in {months:.0f} months, since {last_low_display} when they were {historical_display}."
    
    return description

def get_earliest_date_from_endpoint(metric_info: Dict[str, Any], period_type: str = 'month', 
                                   district: int = None) -> Optional[datetime]:
    """
    Query the endpoint to get the earliest date in the dataset.
    
    Args:
        metric_info: Metric information from database
        period_type: Period type (month, year)
        district: Optional district to filter by
        
    Returns:
        Earliest date as datetime object, or None if error
    """
    try:
        # Get the YTD query from metric_info
        original_query = None
        if isinstance(metric_info.get('query_data'), dict):
            original_query = metric_info['query_data'].get('ytd_query', '')
            if not original_query:
                original_query = metric_info['query_data'].get('metric_query', '')
        else:
            original_query = metric_info.get('ytd_query', '')
        
        if not original_query:
            logger.error(f"No query found for {metric_info.get('query_name', 'unknown')}")
            return None
        
        endpoint = metric_info['endpoint']
        context_variables = {}
        
        # Extract date field from the original query
        date_field_match = re.search(r'date_trunc_[ymd]+\(([^)]+)\)\s+as\s+date', original_query, re.IGNORECASE)
        if not date_field_match:
            logger.error(f"Could not extract date field from query: {original_query}")
            return None
        
        date_field_name = date_field_match.group(1).strip()
        
        # Determine the correct date_trunc level based on period_type
        if period_type == 'month':
            date_trunc_level = 'date_trunc_ym'
        elif period_type == 'year':
            date_trunc_level = 'date_trunc_y'
        else:  # day
            date_trunc_level = 'date_trunc_ymd'
        
        # Build query to get the earliest date
        earliest_date_query = f"SELECT MIN({date_trunc_level}({date_field_name})) as earliest_date"
        
        # Extract WHERE clause from original query, but remove date-related conditions
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|$)', original_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Remove date-related conditions
            conditions = []
            current_condition = ""
            in_quotes = False
            quote_char = None
            
            for char in where_clause:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                    current_condition += char
                elif char == ' ' and not in_quotes:
                    if current_condition.endswith(' AND'):
                        current_condition = current_condition[:-4].strip()
                        if current_condition:
                            conditions.append(current_condition)
                        current_condition = ""
                    else:
                        current_condition += char
                else:
                    current_condition += char
            
            if current_condition.strip():
                conditions.append(current_condition.strip())
            
            filtered_conditions = []
            
            for condition in conditions:
                condition = condition.strip()
                # Skip date-related conditions
                if any(keyword in condition.lower() for keyword in [
                    'date_trunc', 'current_date', 'this_year', '>=', '<=', '='
                ]) and any(date_field in condition.lower() for date_field in [
                    'date', 'datetime', 'time', 'period'
                ]):
                    continue
                else:
                    filtered_conditions.append(condition)
            
            # Reconstruct the WHERE clause
            if filtered_conditions:
                clean_where_clause = ' AND '.join(filtered_conditions)
                earliest_date_query += f" WHERE {clean_where_clause}"
        
        # Add district filter if specified
        if district is not None and district != 0:
            if 'WHERE' in earliest_date_query:
                earliest_date_query += f" AND supervisor_district = '{district}'"
            else:
                earliest_date_query += f" WHERE supervisor_district = '{district}'"
        
        logger.info(f"Earliest date query: {earliest_date_query}")
        
        # Execute the query
        result = set_dataset(context_variables, endpoint=endpoint, query=earliest_date_query)
        
        if 'error' in result:
            logger.error(f"Error querying earliest date: {result['error']}")
            return None
        
        dataset = context_variables.get('dataset')
        if dataset is None or dataset.empty:
            logger.warning("No data returned for earliest date query")
            return None
        
        # Get the earliest date
        earliest_date_str = dataset.iloc[0]['earliest_date']
        if earliest_date_str:
            try:
                # Parse the date string
                if 'T' in earliest_date_str:
                    earliest_date = datetime.fromisoformat(earliest_date_str.replace('Z', '+00:00'))
                else:
                    # Handle YYYY-MM format
                    if len(earliest_date_str) == 7:  # YYYY-MM
                        earliest_date = datetime.strptime(earliest_date_str, '%Y-%m')
                    else:
                        earliest_date = datetime.strptime(earliest_date_str, '%Y-%m-%d')
                
                logger.info(f"Found earliest date: {earliest_date}")
                return earliest_date
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing earliest date '{earliest_date_str}': {e}")
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting earliest date: {e}")
        return None

def query_historical_data(metric_info: Dict[str, Any], target_value: float, is_high: bool, 
                         period_type: str = 'month', group_value: str = None, 
                         exclude_month: str = None, current_period: datetime = None, 
                         district: int = None) -> Optional[Dict[str, Any]]:
    """
    Query historical data to find when the value was last this high/low.
    
    Args:
        metric_info: Metric information from database
        target_value: The value to search for
        is_high: True if searching for high values, False for low values
        period_type: Period type (month, year)
        group_value: Optional group value to filter by (for grouped charts)
        
    Returns:
        Dictionary with historical data results, or None if error
    """
    try:
        # Get the YTD query from metric_info
        original_query = None
        if isinstance(metric_info.get('query_data'), dict):
            original_query = metric_info['query_data'].get('ytd_query', '')
            if not original_query:
                original_query = metric_info['query_data'].get('metric_query', '')
        else:
            original_query = metric_info.get('ytd_query', '')
        
        if not original_query:
            logger.error(f"No query found for {metric_info.get('query_name', 'unknown')}")
            return None
        
        endpoint = metric_info['endpoint']
        context_variables = {}
        
        # Step 1: Extract date field and value field from the original query
        # Look for patterns like: date_trunc_ymd(Report_Datetime) as date, COUNT(*) as value
        date_field_match = re.search(r'date_trunc_[ymd]+\(([^)]+)\)\s+as\s+date', original_query, re.IGNORECASE)
        value_field_match = re.search(r'([^,]+)\s+as\s+value', original_query, re.IGNORECASE)
        
        if not date_field_match:
            logger.error(f"Could not extract date field from query: {original_query}")
            return None
        
        if not value_field_match:
            logger.error(f"Could not extract value field from query: {original_query}")
            return None
        
        date_field_name = date_field_match.group(1).strip()
        value_field_expression = value_field_match.group(1).strip()
        
        logger.info(f"Extracted date field: {date_field_name}")
        logger.info(f"Extracted value field: {value_field_expression}")
        
        # Step 2: Determine the correct date_trunc level based on period_type
        if period_type == 'month':
            date_trunc_level = 'date_trunc_ym'
        elif period_type == 'year':
            date_trunc_level = 'date_trunc_y'
        else:  # day
            date_trunc_level = 'date_trunc_ymd'
        
        # Step 3: Build the historical search query
        # Start with the SELECT clause
        historical_query = f"SELECT {date_trunc_level}({date_field_name}) as date, {value_field_expression} as value"
        
        # Extract WHERE clause from original query, but remove date-related conditions
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|$)', original_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            logger.info(f"Original WHERE clause: {where_clause}")
            
            # Remove date-related conditions more carefully
            # Split by AND to handle each condition separately, but be careful about AND inside quotes
            # Use a more sophisticated approach to split by AND outside of quotes
            conditions = []
            current_condition = ""
            in_quotes = False
            quote_char = None
            
            for char in where_clause:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                    current_condition += char
                elif char == ' ' and not in_quotes:
                    # Check if this is the start of "AND"
                    if current_condition.endswith(' AND'):
                        # Remove the "AND" and add the condition
                        current_condition = current_condition[:-4].strip()
                        if current_condition:
                            conditions.append(current_condition)
                        current_condition = ""
                    else:
                        current_condition += char
                else:
                    current_condition += char
            
            # Add the last condition
            if current_condition.strip():
                conditions.append(current_condition.strip())
            
            filtered_conditions = []
            
            for condition in conditions:
                condition = condition.strip()
                # Skip date-related conditions
                if any(keyword in condition.lower() for keyword in [
                    'date_trunc', 'current_date', 'this_year', '>=', '<=', '='
                ]) and any(date_field in condition.lower() for date_field in [
                    'date', 'datetime', 'time', 'period'
                ]):
                    logger.info(f"Skipping date condition: {condition}")
                    continue
                else:
                    filtered_conditions.append(condition)
            
            # Reconstruct the WHERE clause
            if filtered_conditions:
                clean_where_clause = ' AND '.join(filtered_conditions)
                logger.info(f"Clean WHERE clause: {clean_where_clause}")
                historical_query += f" WHERE {clean_where_clause}"
        
        
        # Add district filter if specified
        if district is not None and district != 0:  # Don't filter for district 0 (citywide)
            if 'WHERE' in historical_query:
                historical_query += f" AND supervisor_district = '{district}'"
            else:
                historical_query += f" WHERE supervisor_district = '{district}'"
            logger.info(f"Adding district filter: supervisor_district = '{district}'")
        
        # Add date filter to exclude the current month (likely incomplete)
        if exclude_month and period_type == 'month':
            # Add WHERE clause to exclude the current month
            if 'WHERE' in historical_query:
                historical_query += f" AND date < '{exclude_month}'"
            else:
                historical_query += f" WHERE date < '{exclude_month}'"
            logger.info(f"Excluding current month: {exclude_month}")
        
        # Add GROUP BY and HAVING clause
        operator = '>' if is_high else '<'
        historical_query += f" GROUP BY date HAVING value {operator} {target_value}"
        
        # Add ORDER BY and LIMIT
        historical_query += f" ORDER BY date DESC LIMIT 10"
        
        logger.info(f"Historical search query: {historical_query}")
        logger.info(f"Target value: {target_value}, Operator: {operator}, Is high: {is_high}")
        
        # Execute the historical search query
        result = set_dataset(context_variables, endpoint=endpoint, query=historical_query)
        
        # Log the executed query URL if available
        if 'queryURL' in result:
            logger.info(f"Historical data query URL: {result['queryURL']}")
            print(f"Query URL: {result['queryURL']}")
        
        if 'error' in result:
            logger.error(f"Error querying historical data: {result['error']}")
            return None
        
        dataset = context_variables.get('dataset')
        if dataset is None or dataset.empty:
            logger.info(f"No historical data found with value {operator} {target_value}")
            return {
                'status': 'success',
                'data': None,
                'last_occurrence': None,
                'years_since_last': None
            }
        
        # Get the most recent occurrence (first row since we ordered DESC)
        if not dataset.empty:
            last_occurrence = dataset.iloc[0]['date']
            
            # Calculate years since last occurrence
            if isinstance(last_occurrence, str):
                try:
                    # First try to parse as ISO datetime string
                    if 'T' in last_occurrence:
                        last_date = datetime.fromisoformat(last_occurrence.replace('Z', '+00:00'))
                    elif period_type == 'month':
                        # Try different month formats
                        if len(last_occurrence) == 7:  # "2020-04"
                            last_date = datetime.strptime(last_occurrence, '%Y-%m')
                        else:  # "2020-04-01" or other formats
                            last_date = datetime.strptime(last_occurrence, '%Y-%m-%d')
                    else:
                        last_date = datetime.strptime(last_occurrence, '%Y')
                except Exception as e:
                    # If we can't parse the date, return None
                    logger.warning(f"Could not parse last_occurrence date: {last_occurrence} - {e}")
                    return {
                        'status': 'success',
                        'data': dataset,
                        'last_occurrence': last_occurrence,
                        'years_since_last': None
                    }
            else:
                last_date = last_occurrence
            
            # Use the current_period for comparison if provided, otherwise use exclude_month
            if current_period:
                current_date = current_period
            elif exclude_month:
                try:
                    current_date = datetime.strptime(exclude_month, '%Y-%m')
                except:
                    current_date = datetime.now()
            else:
                current_date = datetime.now()
            
            years_since_last = (current_date - last_date).days / 365.25
            
            return {
                'status': 'success',
                'data': dataset,
                'last_occurrence': last_occurrence,
                'years_since_last': years_since_last
            }
        
        return {
            'status': 'success',
            'data': None,
            'last_occurrence': None,
            'years_since_last': None
        }
        
    except Exception as e:
        logger.error(f"Error querying historical data: {str(e)}")
        logger.error(traceback.format_exc())
        return None



def detect_records(chart_id: int) -> Dict[str, Any]:
    """
    Main function to detect record highs and lows for a given chart.
    
    Args:
        chart_id: The chart ID to analyze
        
    Returns:
        Dictionary containing record detection results
    """
    logger.info(f"Starting record detection for chart_id: {chart_id}")
    
    # Get chart data
    chart_data = get_chart_data(chart_id)
    if not chart_data:
        return {'error': f'Chart with ID {chart_id} not found'}
    
    metadata = chart_data['metadata']
    data_points = chart_data['data']
    
    # Extract metadata
    chart_title = metadata[1]
    period_type = metadata[3]
    object_id = metadata[5]
    object_name = metadata[6]
    district = metadata[8]
    group_field = metadata[9]
    executed_query_url = metadata[10]
    
    logger.info(f"Analyzing chart: {chart_title}")
    logger.info(f"Period type: {period_type}")
    logger.info(f"Object: {object_name} (ID: {object_id})")
    logger.info(f"District: {district}")
    logger.info(f"Group field: {group_field}")
    
    # Get metric information
    metric_info = get_metric_info(object_id)
    if not metric_info:
        logger.warning(f"Could not find metric info for object_id: {object_id}")
        metric_info = {
            'query_name': object_name,
            'endpoint': None,
            'query_data': {}
        }
    
    # Group data by series if there's a group field
    if group_field:
        # Group data by group_value
        series_groups = {}
        for time_period, group_value, numeric_value in data_points:
            if group_value not in series_groups:
                series_groups[group_value] = []
            series_groups[group_value].append((time_period, numeric_value))
        
        # Analyze each series
        all_series_results = []
        record_results = []
        record_charts = []
        for group_value, series_data in series_groups.items():
            logger.info(f"Analyzing series: {group_value}")
            analysis = analyze_series_for_records(series_data, str(group_value), metric_info, period_type, district)
            if 'error' not in analysis:
                # Generate headline and description
                analysis['headline'] = generate_record_headline(analysis, object_name, district)
                analysis['description'] = generate_record_description(analysis, object_name, district)
                all_series_results.append(analysis)
                # Only include if it's a record
                if analysis['is_record_high'] or analysis['is_record_low']:
                    record_results.append(analysis)
                    
                    # Generate chart for record-breaking series
                    logger.info(f"Generating chart for record-breaking series: {group_value}")
                    chart_result = generate_record_chart(analysis, metric_info, period_type)
                    if chart_result and 'error' not in chart_result:
                        logger.info(f"Successfully generated chart: {chart_result.get('chart_title', 'Unknown')}")
                        record_charts.append(chart_result)
                    else:
                        logger.warning(f"Failed to generate chart: {chart_result.get('error', 'Unknown error')}")
        
        return {
            'chart_id': chart_id,
            'chart_title': chart_title,
            'metric_name': object_name,
            'district': district,
            'period_type': period_type,
            'group_field': group_field,
            'series_count': len(series_groups),
            'series_results': all_series_results,
            'record_series': record_results,
            'total_records': len(record_results),
            'chart_metadata': {
                'district': district,
                'object_id': object_id,
                'object_name': object_name,
                'period_type': period_type,
                'group_field': group_field
            },
            'record_charts': record_charts
        }
    else:
        # Single series analysis
        series_data = [(time_period, numeric_value) for time_period, group_value, numeric_value in data_points]
        logger.info("Analyzing single series")
        
        analysis = analyze_series_for_records(series_data, "Overall", metric_info, period_type, district)
        if 'error' not in analysis:
            # Generate headline and description
            analysis['headline'] = generate_record_headline(analysis, object_name, district)
            analysis['description'] = generate_record_description(analysis, object_name, district)
            
            # Generate chart if this is a record
            chart_result = None
            if analysis['is_record_high'] or analysis['is_record_low']:
                logger.info(f"Generating chart for record-breaking metric: {object_name}")
                chart_result = generate_record_chart(analysis, metric_info, period_type)
                if chart_result and 'error' not in chart_result:
                    logger.info(f"Successfully generated chart: {chart_result.get('chart_title', 'Unknown')}")
                else:
                    logger.warning(f"Failed to generate chart: {chart_result.get('error', 'Unknown error')}")
            
            return {
                'chart_id': chart_id,
                'chart_title': chart_title,
                'metric_name': object_name,
                'district': district,
                'period_type': period_type,
                'group_field': None,
                'series_count': 1,
                'series_results': [analysis],
                'record_series': [analysis] if (analysis['is_record_high'] or analysis['is_record_low']) else [],
                'total_records': 1 if (analysis['is_record_high'] or analysis['is_record_low']) else 0,
                'chart_metadata': {
                    'district': district,
                    'object_id': object_id,
                    'object_name': object_name,
                    'period_type': period_type,
                    'group_field': None
                },
                'record_chart': chart_result
            }
        else:
            return analysis

def save_record_results(results: Dict[str, Any], output_dir: str = None) -> str:
    """
    Save record detection results to a file.
    
    Args:
        results: The record detection results
        output_dir: Output directory (defaults to OUTPUT_DIR)
        
    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create filename
    chart_id = results.get('chart_id', 'unknown')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"record_detection_chart_{chart_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved record detection results to: {filepath}")
    return filepath

def get_recent_charts_by_period_type(period_type: str = 'month', metric_id: str = None) -> List[Dict[str, Any]]:
    """
    Get the most recent active chart for each metric_id for a given period type.
    
    Args:
        period_type: Period type to filter by (month, year, etc.)
        metric_id: Optional specific metric ID to filter by
        
    Returns:
        List of dictionaries containing chart information
    """
    def get_charts_operation(connection):
        cursor = connection.cursor()
        
        if metric_id:
            # Query to get ALL active charts for a specific metric_id (including those with group fields)
            cursor.execute("""
                SELECT 
                    chart_id,
                    object_id,
                    object_name,
                    period_type,
                    district,
                    group_field,
                    created_at
                FROM time_series_metadata 
                WHERE is_active = true 
                AND period_type = %s
                AND object_id = %s
                AND group_field IS NULL
                ORDER BY created_at DESC
            """, (period_type, metric_id))
        else:
            # Query to get the most recent active chart for each metric_id
            cursor.execute("""
                SELECT DISTINCT ON (object_id) 
                    chart_id,
                    object_id,
                    object_name,
                    period_type,
                    district,
                    group_field,
                    created_at
                FROM time_series_metadata 
                WHERE is_active = true                 
                AND group_field IS NULL
                AND period_type = %s
                ORDER BY object_id, created_at DESC
            """, (period_type,))
        
        results = cursor.fetchall()
        cursor.close()
        
        return [
            {
                'chart_id': row[0],
                'object_id': row[1],
                'object_name': row[2],
                'period_type': row[3],
                'district': row[4],
                'group_field': row[5],
                'created_at': row[6]
            }
            for row in results
        ]
    
    result = execute_with_connection(get_charts_operation)
    
    if result['status'] == 'success':
        return result['result']
    else:
        logger.error(f"Failed to get recent charts: {result.get('message', 'Unknown error')}")
        return []

def run_detect_records_on_period_type(period_type: str = 'month', output_dir: str = None, metric_id: str = None) -> Dict[str, Any]:
    """
    Run detect_records on all recent charts for a given period type.
    
    Args:
        period_type: Period type to analyze (month, year, etc.)
        output_dir: Output directory for results
        metric_id: Optional specific metric ID to filter by
        
    Returns:
        Dictionary containing summary of all results
    """
    logger.info(f"Starting record detection for period type: {period_type}")
    if metric_id:
        logger.info(f"Filtering by metric_id: {metric_id}")
    
    # Get recent charts
    charts = get_recent_charts_by_period_type(period_type, metric_id)
    
    if not charts:
        logger.warning(f"No active charts found for period type: {period_type}")
        return {'error': f'No active charts found for period type: {period_type}'}
    
    logger.info(f"Found {len(charts)} charts to analyze")
    
    # Process each chart
    all_results = []
    successful_count = 0
    failed_count = 0
    
    for chart_info in charts:
        chart_id = chart_info['chart_id']
        object_name = chart_info['object_name']
        
        logger.info(f"Processing chart {chart_id}: {object_name}")
        
        try:
            # Run detect_records
            results = detect_records(chart_id)
            
            if 'error' not in results:
                all_results.append(results)
                successful_count += 1
                
                # Log summary
                logger.info(f"Chart {chart_id}: Processed {results['series_count']} series, {results['total_records']} records detected")
            else:
                logger.error(f"Chart {chart_id}: {results['error']}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing chart {chart_id}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_count += 1
    
    # Create summary
    summary = {
        'period_type': period_type,
        'metric_id': metric_id,
        'metric_name': charts[0]['object_name'] if charts else 'Unknown',
        'total_charts': len(charts),
        'successful_count': successful_count,
        'failed_count': failed_count,
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save summary with metric name
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create a clean metric name for the filename
    metric_name = charts[0]['object_name'] if charts else f"metric_{metric_id}"
    # Clean the metric name for filename (remove emojis and special characters)
    clean_metric_name = re.sub(r'[^\w\s-]', '', metric_name).strip()
    clean_metric_name = re.sub(r'\s+', '_', clean_metric_name)
    
    if metric_id:
        summary_filename = f"record_detection_{clean_metric_name}_{period_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        summary_filename = f"record_detection_summary_{period_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_filepath = os.path.join(output_dir, summary_filename)
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Summary saved to: {summary_filepath}")
    
    # Add the filepath to the summary for reference
    summary['summary_filepath'] = summary_filepath
    
    return summary

def list_available_charts(period_type: str = None, limit: int = 10, metric_id: str = None) -> List[Dict[str, Any]]:
    """
    List available charts for testing.
    
    Args:
        period_type: Optional period type filter
        limit: Maximum number of charts to return
        
    Returns:
        List of dictionaries containing chart information
    """
    def list_charts_operation(connection):
        cursor = connection.cursor()
        
        if metric_id:
            # Show all charts for a specific metric_id
            cursor.execute("""
                SELECT 
                    chart_id,
                    object_id,
                    object_name,
                    period_type,
                    district,
                    group_field,
                    created_at
                FROM time_series_metadata 
                WHERE is_active = true 
                AND object_id = %s
                AND group_field IS NULL
                ORDER BY created_at DESC
                LIMIT %s
            """, (metric_id, limit))
        elif period_type:
            cursor.execute("""
                SELECT 
                    chart_id,
                    object_id,
                    object_name,
                    period_type,
                    district,
                    group_field,
                    created_at
                FROM time_series_metadata 
                WHERE is_active = true 
                AND period_type = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (period_type, limit))
        else:
            cursor.execute("""
                SELECT 
                    chart_id,
                    object_id,
                    object_name,
                    period_type,
                    district,
                    group_field,
                    created_at
                FROM time_series_metadata 
                WHERE is_active = true 
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
        
        results = cursor.fetchall()
        cursor.close()
        
        return [
            {
                'chart_id': row[0],
                'object_id': row[1],
                'object_name': row[2],
                'period_type': row[3],
                'district': row[4],
                'group_field': row[5],
                'created_at': row[6]
            }
            for row in results
        ]
    
    result = execute_with_connection(list_charts_operation)
    
    if result['status'] == 'success':
        return result['result']
    else:
        logger.error(f"Failed to list charts: {result.get('message', 'Unknown error')}")
        return []

def create_llm_friendly_output(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an LLM-friendly output format that maximizes comprehension and explanation ability.
    
    Args:
        results: The original record detection results
        
    Returns:
        Dictionary with LLM-optimized structure and formatting
    """
    if 'error' in results:
        return {
            'status': 'error',
            'error_message': results['error'],
            'timestamp': datetime.now().isoformat()
        }
    
    # Extract basic information
    chart_id = results.get('chart_id')
    metric_name = results.get('metric_name', 'Unknown Metric')
    # Use object_name as chart_title if chart_title is None or empty
    chart_title = results.get('chart_title')
    if not chart_title or chart_title == 'None':
        chart_title = results.get('metric_name', 'Unknown Chart')
    period_type = results.get('period_type', 'month')
    district = results.get('district', 0)
    series_count = results.get('series_count', 0)
    total_records = results.get('total_records', 0)
    series_results = results.get('series_results', [])
    
    # Create LLM-friendly structure
    llm_output = {
        'analysis_summary': {
            'chart_id': chart_id,
            'metric_name': metric_name,
            'chart_title': chart_title,
            'period_type': period_type,
            'district': district,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_series_analyzed': series_count,
            'records_detected': total_records,
            'analysis_scope': f"Analyzed {series_count} series for {metric_name} ({period_type} data)"
        },
        'chart_metadata': results.get('chart_metadata', {}),
        
        'context_information': {
            'metric_description': f"This analysis examines {metric_name} data to identify record highs and lows",
            'period_context': f"Data is analyzed on a {period_type}ly basis",
            'district_context': f"Analysis covers {'all districts' if district == 0 else f'District {district}'}",
            'record_definition': "A record is defined as the highest (record high) or lowest (record low) value in the available historical data"
        },
        
        'series_analysis': [],
        
        'statistical_summary': {
            'record_highs': 0,
            'record_lows': 0,
            'normal_values': 0,
            'series_with_records': []
        },
        
        'key_findings': [],
        
        'recommendations': []
    }
    
    # Process each series
    for series in series_results:
        series_name = series.get('series_name', 'Unknown Series')
        most_recent_value = series.get('most_recent_value', 0)
        is_record_high = series.get('is_record_high', False)
        is_record_low = series.get('is_record_low', False)
        historical_max = series.get('historical_max', 0)
        historical_min = series.get('historical_min', 0)
        historical_min_period = series.get('historical_min_period')
        total_periods = series.get('total_periods', 0)
        
        # Determine status
        if is_record_high and is_record_low:
            status = "RECORD HIGH AND LOW"
        elif is_record_high:
            status = "RECORD HIGH"
        elif is_record_low:
            status = "RECORD LOW"
        else:
            status = "NORMAL"
        
        # Create series analysis entry
        series_analysis = {
            'series_name': series_name,
            'current_value': most_recent_value,
            'status': status,
            'historical_context': {
                'historical_maximum': historical_max,
                'historical_minimum': historical_min,
                'historical_minimum_period': format_period_display(historical_min_period, period_type) if historical_min_period else None,
                'data_points_analyzed': total_periods,
                'value_range': historical_max - historical_min
            },
            'record_details': {},
            'headline': series.get('headline', ''),  # Include headline from original series data
            'description': series.get('description', ''),  # Include description from original series data
            'earliest_date_from_endpoint': series.get('earliest_date_from_endpoint')  # Include earliest date for context
        }
        
        # Add record-specific details
        if is_record_high:
            last_high_period = series.get('last_high_period')
            years_since_last_high = series.get('years_since_last_high')
            last_high_value = series.get('last_high_value')
            
            series_analysis['record_details']['record_high'] = {
                'is_new_record': last_high_period is None,
                'previous_high_period': format_period_display(last_high_period, period_type) if last_high_period else None,
                'years_since_previous': years_since_last_high,
                'previous_high_value': last_high_value,
                'context': f"Previous high of {last_high_value} occurred {format_period_display(last_high_period, period_type)} ({years_since_last_high:.1f} years ago)" if last_high_period and years_since_last_high else "This appears to be a new record"
            }
            
            llm_output['statistical_summary']['record_highs'] += 1
            llm_output['statistical_summary']['series_with_records'].append(series_name)
        
        if is_record_low:
            last_low_period = series.get('last_low_period')
            years_since_last_low = series.get('years_since_last_low')
            last_low_value = series.get('last_low_value')
            earliest_date_from_endpoint = series.get('earliest_date_from_endpoint')
            
            # Determine the context message
            if last_low_period and years_since_last_low:
                context = f"Previous low of {last_low_value} occurred {format_period_display(last_low_period, period_type)} ({years_since_last_low:.1f} years ago)"
            elif earliest_date_from_endpoint:
                # Format the earliest date as MMM-YY
                earliest_date_str = earliest_date_from_endpoint.strftime('%b-%y')
                context = f"Lowest on record (dataset start date: {earliest_date_str})"
            else:
                context = "This appears to be a new record"
            
            series_analysis['record_details']['record_low'] = {
                'is_new_record': last_low_period is None,
                'previous_low_period': format_period_display(last_low_period, period_type) if last_low_period else None,
                'years_since_previous': years_since_last_low,
                'previous_low_value': last_low_value,
                'context': context
            }
            
            llm_output['statistical_summary']['record_lows'] += 1
            if series_name not in llm_output['statistical_summary']['series_with_records']:
                llm_output['statistical_summary']['series_with_records'].append(series_name)
        
        if not is_record_high and not is_record_low:
            llm_output['statistical_summary']['normal_values'] += 1
        
        # Add to series analysis
        llm_output['series_analysis'].append(series_analysis)
    
    # Generate key findings
    if total_records > 0:
        llm_output['key_findings'].append(f"Found {total_records} record{'s' if total_records > 1 else ''} across {len(llm_output['statistical_summary']['series_with_records'])} series")
        
        if llm_output['statistical_summary']['record_highs'] > 0:
            llm_output['key_findings'].append(f"{llm_output['statistical_summary']['record_highs']} series reached record high values")
        
        if llm_output['statistical_summary']['record_lows'] > 0:
            llm_output['key_findings'].append(f"{llm_output['statistical_summary']['record_lows']} series reached record low values")
    else:
        llm_output['key_findings'].append("No records detected - all values are within normal historical ranges")
    
    # Add recommendations based on findings
    if total_records > 0:
        llm_output['recommendations'].append("Monitor these record values closely for potential trends or anomalies")
        llm_output['recommendations'].append("Consider investigating the factors contributing to these record values")
    else:
        llm_output['recommendations'].append("Values are within normal ranges - continue regular monitoring")
    
    return llm_output

def llm_output_to_markdown(llm_output: dict) -> str:
    """
    Convert the LLM-friendly output dictionary to a Markdown report.
    """
    md = []
    summary = llm_output.get('analysis_summary', {})
    context = llm_output.get('context_information', {})
    stats = llm_output.get('statistical_summary', {})
    findings = llm_output.get('key_findings', [])
    recommendations = llm_output.get('recommendations', [])
    series = llm_output.get('series_analysis', [])

    # Title
    chart_title = summary.get('chart_title', summary.get('metric_name', 'Unknown Chart'))
    md.append(f"# Record Detection Analysis: {chart_title}")
    md.append("")
    md.append(f"**Period Type:** {summary.get('period_type')}")
    md.append(f"**District:** {summary.get('district')}")
    md.append(f"**Analysis Timestamp:** {summary.get('analysis_timestamp')}")
    md.append(f"**Total Series Analyzed:** {summary.get('total_series_analyzed')}")
    md.append(f"**Records Detected:** {summary.get('records_detected')}")
    # Add headline if present
    if series and series[0].get('headline'):
        md.append("")
        md.append(f"> **Headline:** {series[0]['headline']}")
    md.append("")

    # Context
    md.append("## Context")
    md.append(f"- {context.get('metric_description', '')}")
    md.append(f"- {context.get('period_context', '')}")
    md.append(f"- {context.get('district_context', '')}")
    md.append(f"- {context.get('record_definition', '')}")
    md.append("")

    # Statistical Summary
    md.append("## Statistical Summary")
    md.append(f"- Record Highs: {stats.get('record_highs', 0)}")
    md.append(f"- Record Lows: {stats.get('record_lows', 0)}")
    md.append(f"- Normal Values: {stats.get('normal_values', 0)}")
    md.append(f"- Series with Records: {len(stats.get('series_with_records', []))}")
    md.append("")

    # Key Findings
    md.append("## Key Findings")
    for finding in findings:
        md.append(f"- {finding}")
    md.append("")

    # Series Analysis
    md.append("## Series Analysis")
    for s in series:
        md.append(f"### {s.get('series_name', 'Unknown Series')}")
        if s.get('headline'):
            md.append(f"> {s['headline']}")
        md.append(f"- **Current Value:** {s.get('current_value')}")
        md.append(f"- **Status:** {s.get('status')}")
        hc = s.get('historical_context', {})
        # Historical Range
        md.append(f"- **Historical Range:** {hc.get('historical_minimum')} to {hc.get('historical_maximum')}")
        # Data Points
        md.append(f"- **Data Points Analyzed:** {hc.get('data_points_analyzed')}")
        # Record Details
        if s.get('record_details'):
            if 'record_high' in s['record_details']:
                high = s['record_details']['record_high']
                md.append(f"- **Record High Context:** {high.get('context')}")
            if 'record_low' in s['record_details']:
                low = s['record_details']['record_low']
                # Use the context that was already properly formatted in create_llm_friendly_output
                md.append(f"- **Record Low Context:** {low.get('context')}")
        
        # For non-record series, show historical minimum with month/year
        if not s.get('record_details') or 'record_low' not in s.get('record_details', {}):
            hc = s.get('historical_context', {})
            if hc.get('historical_minimum') is not None and hc.get('historical_minimum_period'):
                md.append(f"- **Historical Minimum:** {hc.get('historical_minimum')} in {hc.get('historical_minimum_period')}")
        
        md.append("")

    # Recommendations
    md.append("## Recommendations")
    for rec in recommendations:
        md.append(f"- {rec}")
    md.append("")

    return '\n'.join(md)

def generate_record_chart(analysis_result: Dict[str, Any], metric_info: Dict[str, Any], period_type: str = 'month') -> Dict[str, Any]:
    """
    Generate a chart for a record-breaking metric showing the entire time period since the last low point.
    
    Args:
        analysis_result: The record analysis result
        metric_info: Metric information for data fetching
        period_type: Period type (month, year, etc.)
        
    Returns:
        Dictionary with chart generation results
    """
    try:
        # Extract information from the analysis
        series_name = analysis_result.get('series_name', 'Overall')
        current_value = analysis_result.get('current_value', 0)
        is_record_low = analysis_result.get('is_record_low', False)
        is_record_high = analysis_result.get('is_record_high', False)
        
        # Get the last low/high period for chart time range
        last_period = None
        years_since_last = None
        
        if is_record_low and 'last_low_period' in analysis_result:
            last_period = analysis_result['last_low_period']
            years_since_last = analysis_result.get('years_since_last_low', 0)
        elif is_record_high and 'last_high_period' in analysis_result:
            last_period = analysis_result['last_high_period']
            years_since_last = analysis_result.get('years_since_last_high', 0)
        
        # If no historical period found, use a default range (7 years for monthly data)
        if not last_period:
            years_since_last = 7 if period_type == 'month' else 10
        
        # Calculate the start date for the chart
        if last_period:
            # Parse the last period date
            if period_type == 'month':
                try:
                    last_date = datetime.strptime(last_period, '%B %Y')
                    start_date = last_date - timedelta(days=30)  # Start a month before
                except:
                    # Fallback: use years_since_last
                    start_date = datetime.now() - timedelta(days=years_since_last * 365)
            else:  # year
                try:
                    last_date = datetime.strptime(last_period, '%Y')
                    start_date = datetime(last_date.year - 1, 1, 1)
                except:
                    start_date = datetime.now() - timedelta(days=years_since_last * 365)
        else:
            # Use years_since_last to calculate start date
            start_date = datetime.now() - timedelta(days=years_since_last * 365)
        
        # Format dates for the query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Generating chart for {series_name} from {start_date_str} to {end_date_str}")
        
        # Get the query from metric_info
        original_query = None
        if isinstance(metric_info.get('query_data'), dict):
            original_query = metric_info['query_data'].get('ytd_query', '')
            if not original_query:
                original_query = metric_info['query_data'].get('metric_query', '')
        else:
            original_query = metric_info.get('query_data', '')
        
        if not original_query:
            logger.error(f"No query found for chart generation")
            return {'error': 'No query available for chart generation'}
        
        # Transform the query for the extended time period
        # We need to modify the query to cover the entire period since the last record
        transformed_query = transform_query_for_extended_period(
            original_query, 
            start_date_str, 
            end_date_str, 
            period_type
        )
        
        # Set up context variables for chart generation
        context_variables = {}
        
        # Set the dataset using the transformed query
        result = set_dataset(
            context_variables=context_variables, 
            endpoint=metric_info.get('endpoint', ''), 
            query=transformed_query
        )
        
        if 'error' in result:
            logger.error(f"Error setting dataset for chart generation: {result['error']}")
            return {'error': f"Failed to fetch data for chart: {result['error']}"}
        
        # Get the dataset
        dataset = context_variables.get('dataset')
        if dataset is None or dataset.empty:
            logger.error("No dataset available for chart generation")
            return {'error': 'No data available for chart generation'}
        
        # Create chart title
        district_info = ""
        if analysis_result.get('district', 0) != 0:
            district_info = f" - District {analysis_result.get('district', 0)}"
        
        chart_title = f"{metric_info.get('query_name', 'Unknown Metric')}{district_info} - Record Analysis"
        
        # Set up context for chart generation
        chart_context = {
            'dataset': dataset,
            'chart_title': chart_title,
            'y_axis_label': metric_info.get('query_name', 'Value'),
            'noun': metric_info.get('query_name', 'Value')
        }
        
        # Determine the period field based on period_type
        period_field = f"{period_type}_period" if f"{period_type}_period" in dataset.columns else 'date'
        
        # Generate the chart
        chart_result = generate_time_series_chart(
            context_variables=chart_context,
            time_series_field=period_field,
            numeric_fields='value',
            aggregation_period=period_type,
            max_legend_items=10,
            show_average_line=True,
            return_html=True,
            output_dir='output/records/charts',
            store_in_db=True,
            object_type='record_analysis',
            object_id=metric_info.get('metric_id', 'unknown'),
            object_name=f"{metric_info.get('query_name', 'Unknown')} Record Analysis"
        )
        
        if isinstance(chart_result, tuple):
            markdown_content, html_content = chart_result
        else:
            markdown_content = str(chart_result)
            html_content = str(chart_result)
        
        return {
            'status': 'success',
            'markdown': markdown_content,
            'html': html_content,
            'chart_title': chart_title,
            'time_range': f"{start_date_str} to {end_date_str}",
            'period_type': period_type
        }
        
    except Exception as e:
        logger.error(f"Error generating record chart: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': f"Failed to generate chart: {str(e)}"}

def transform_query_for_extended_period(original_query: str, start_date: str, end_date: str, period_type: str) -> str:
    """
    Transform a query to cover an extended time period for record analysis charts.
    
    Args:
        original_query: The original SQL query
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period_type: Period type (month, year, etc.)
        
    Returns:
        Transformed SQL query
    """
    # Replace date placeholders in the original query
    modified_query = original_query
    
    # Replace common date placeholders
    replacements = {
        'this_year_start': f"'{start_date}'",
        'this_year_end': f"'{end_date}'",
        'last_year_start': f"'{start_date}'",
        'last_year_end': f"'{end_date}'",
        'start_date': f"'{start_date}'",
        'current_date': f"'{end_date}'"
    }
    
    # Apply replacements
    for placeholder, value in replacements.items():
        modified_query = re.sub(r'([=<>:\s]|^)' + re.escape(placeholder) + r'([=<>:\s]|$)', 
                                r'\1' + value + r'\2', 
                                modified_query)
    
    # Ensure consistent use of <= instead of < for date upper bounds
    modified_query = re.sub(r'<\s*current_date', f"<= '{end_date}'", modified_query)
    modified_query = re.sub(r'<\s*\'{end_date}\'', f"<= '{end_date}'", modified_query)
    
    # For YTD queries, we need to modify the date range
    if 'date_trunc_ymd' in modified_query or 'as date,' in modified_query:
        # Extract the date field from the query
        date_field_match = re.search(r'date_trunc_[ymd]+\((.*?)\)', modified_query)
        if date_field_match:
            date_field = date_field_match.group(1).strip()
            
            # Build a new query with proper date conditions
            # First, extract the SELECT part
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', modified_query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_part = select_match.group(1).strip()
                
                # Extract the FROM part
                from_match = re.search(r'FROM\s+(.*?)(?:\s+WHERE|\s+GROUP BY|\s+ORDER BY|\s+LIMIT|$)', 
                                     modified_query, re.IGNORECASE | re.DOTALL)
                if from_match:
                    from_part = from_match.group(1).strip()
                    
                    # Extract existing WHERE conditions (excluding date conditions)
                    where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP BY|\s+ORDER BY|\s+LIMIT|$)', 
                                          modified_query, re.IGNORECASE | re.DOTALL)
                    where_conditions = ""
                    if where_match:
                        where_text = where_match.group(1).strip()
                        # Remove date-related conditions
                        where_conditions = re.sub(r'AND?\s*(?:Report_Datetime|date|month_and_year_of_booking|arrest_date|disposition_date|received_datetime)\s*[<>=]+\s*[\'"]?[^\s\'"]+[\'"]?', '', where_text, flags=re.IGNORECASE)
                        where_conditions = re.sub(r'^\s*AND\s+', '', where_conditions)  # Remove leading AND
                        where_conditions = re.sub(r'\s+AND\s+$', '', where_conditions)  # Remove trailing AND
                    
                    # Build the new query
                    new_query = f"SELECT {select_part} FROM {from_part}"
                    if where_conditions:
                        new_query += f" WHERE {where_conditions} AND {date_field} >= '{start_date}' AND {date_field} <= '{end_date}'"
                    else:
                        new_query += f" WHERE {date_field} >= '{start_date}' AND {date_field} <= '{end_date}'"
                    
                    # Add GROUP BY, ORDER BY, LIMIT if they exist
                    group_match = re.search(r'GROUP BY\s+(.*?)(?:\s+ORDER BY|\s+LIMIT|$)', modified_query, re.IGNORECASE | re.DOTALL)
                    if group_match:
                        new_query += f" GROUP BY {group_match.group(1).strip()}"
                    
                    order_match = re.search(r'ORDER BY\s+(.*?)(?:\s+LIMIT|$)', modified_query, re.IGNORECASE | re.DOTALL)
                    if order_match:
                        new_query += f" ORDER BY {order_match.group(1).strip()}"
                    
                    limit_match = re.search(r'LIMIT\s+(.*?)$', modified_query, re.IGNORECASE | re.DOTALL)
                    if limit_match:
                        new_query += f" LIMIT {limit_match.group(1).strip()}"
                    
                    modified_query = new_query
    
    return modified_query

def main():
    """Main function to run record detection from command line."""
    parser = argparse.ArgumentParser(description='Detect record highs and lows for charts')
    parser.add_argument('chart_id', nargs='?', type=int, help='Chart ID to analyze (optional)')
    parser.add_argument('--period-type', '-p', choices=['month', 'year', 'week', 'day'], 
                        default='month', help='Period type to analyze (when not using specific chart_id)')
    parser.add_argument('--output-dir', '-o', help='Output directory for results')
    parser.add_argument('--all-metrics', '-a', action='store_true', 
                        help='Run on all metrics for the specified period type')
    parser.add_argument('--metric-id', '-m', type=str, help='Specific metric ID to analyze')
    parser.add_argument('--list-charts', '-l', action='store_true',
                        help='List available charts for testing')
    parser.add_argument('--debug-metric', type=str, help='Debug: show all charts for a specific metric_id')
    args = parser.parse_args()
    
    if args.debug_metric:
        # Debug: show all charts for a specific metric_id
        print(f"All charts for metric_id {args.debug_metric}:")
        charts = list_available_charts(metric_id=args.debug_metric, limit=50)
        
        if charts:
            for chart in charts:
                print(f"  Chart {chart['chart_id']}: {chart['object_name']} ({chart['period_type']}) - District {chart['district']} - Group: {chart['group_field']}")
        else:
            print("No charts found.")
        return
    
    if args.list_charts:
        # List available charts (silent - just return data)
        charts = list_available_charts(args.period_type, limit=20)
        return
    
    if args.chart_id:
        # Run on specific chart
        logger.info(f"Running record detection on specific chart: {args.chart_id}")
        results = detect_records(args.chart_id)
        
        if 'error' in results:
            logger.error(f"Record detection failed: {results['error']}")
            return
        
        # Create summary for single chart
        summary = {
            'chart_id': args.chart_id,
            'metric_id': results.get('metric_name', 'Unknown'),
            'metric_name': results['metric_name'],
            'period_type': results['period_type'],
            'total_charts': 1,
            'successful_count': 1 if 'error' not in results else 0,
            'failed_count': 0,
            'results': [results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary with metric name
        if args.output_dir is None:
            output_dir = OUTPUT_DIR
        else:
            output_dir = args.output_dir
        
        # Create a clean metric name for the filename
        metric_name = results['metric_name']
        clean_metric_name = re.sub(r'[^\w\s-]', '', metric_name).strip()
        clean_metric_name = re.sub(r'\s+', '_', clean_metric_name)
        
        summary_filename = f"record_detection_{clean_metric_name}_{results['period_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_filepath = os.path.join(output_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create LLM-friendly output
        llm_results = create_llm_friendly_output(results)
        
        # Print LLM-friendly summary
        print(f"\n=== RECORD DETECTION ANALYSIS ===")
        print(f"Chart ID: {llm_results['analysis_summary']['chart_id']}")
        print(f"Metric: {llm_results['analysis_summary']['metric_name']}")
        print(f"Analysis Scope: {llm_results['analysis_summary']['analysis_scope']}")
        print(f"Records Detected: {llm_results['analysis_summary']['records_detected']}")
        
        print(f"\n=== CONTEXT ===")
        print(f"• {llm_results['context_information']['metric_description']}")
        print(f"• {llm_results['context_information']['period_context']}")
        print(f"• {llm_results['context_information']['district_context']}")
        print(f"• {llm_results['context_information']['record_definition']}")
        
        print(f"\n=== STATISTICAL SUMMARY ===")
        stats = llm_results['statistical_summary']
        print(f"• Record Highs: {stats['record_highs']}")
        print(f"• Record Lows: {stats['record_lows']}")
        print(f"• Normal Values: {stats['normal_values']}")
        print(f"• Series with Records: {len(stats['series_with_records'])}")
        
        print(f"\n=== KEY FINDINGS ===")
        for finding in llm_results['key_findings']:
            print(f"• {finding}")
        
        print(f"\n=== SERIES ANALYSIS ===")
        for series in llm_results['series_analysis']:
            print(f"\n{series['series_name']}:")
            print(f"  Current Value: {series['current_value']}")
            print(f"  Status: {series['status']}")
            print(f"  Historical Range: {series['historical_context']['historical_minimum']} to {series['historical_context']['historical_maximum']}")
            print(f"  Data Points: {series['historical_context']['data_points_analyzed']}")
            
            if series['record_details']:
                if 'record_high' in series['record_details']:
                    high_details = series['record_details']['record_high']
                    print(f"  Record High Context: {high_details['context']}")
                if 'record_low' in series['record_details']:
                    low_details = series['record_details']['record_low']
                    print(f"  Record Low Context: {low_details['context']}")
        
        print(f"\n=== RECOMMENDATIONS ===")
        for rec in llm_results['recommendations']:
            print(f"• {rec}")
        
        print(f"\nResults saved to: {summary_filepath}")
        
        # Also save the LLM-friendly format
        llm_filename = f"llm_analysis_{clean_metric_name}_{results['period_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        llm_filepath = os.path.join(output_dir, llm_filename)
        
        with open(llm_filepath, 'w') as f:
            json.dump(llm_results, f, indent=2, default=str)
        
        print(f"LLM-friendly analysis saved to: {llm_filepath}")

        # Convert LLM-friendly JSON to Markdown
        markdown_report = llm_output_to_markdown(llm_results)
        md_filename = f"llm_analysis_{clean_metric_name}_{results['period_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        md_filepath = os.path.join(output_dir, md_filename)
        with open(md_filepath, 'w') as f:
            f.write(markdown_report)
        print(f"Markdown report saved to: {md_filepath}")
        
    elif args.all_metrics or args.metric_id:
        # Run on all metrics or specific metric for the period type
        if args.metric_id:
            logger.info(f"Running record detection on metric {args.metric_id} for period type: {args.period_type}")
        else:
            logger.info(f"Running record detection on all metrics for period type: {args.period_type}")
        summary = run_detect_records_on_period_type(args.period_type, args.output_dir, args.metric_id)
        
        if 'error' in summary:
            logger.error(f"Record detection failed: {summary['error']}")
            return
        
        # Create LLM-friendly summaries for all results
        llm_summaries = []
        total_records_found = 0
        total_series_analyzed = 0
        
        for result in summary['results']:
            llm_result = create_llm_friendly_output(result)
            llm_summaries.append(llm_result)
            total_records_found += llm_result['analysis_summary']['records_detected']
            total_series_analyzed += llm_result['analysis_summary']['total_series_analyzed']
        
        # Print comprehensive LLM-friendly summary
        if args.metric_id:
            print(f"\n=== RECORD DETECTION SUMMARY FOR METRIC {args.metric_id} ===")
        else:
            print(f"\n=== RECORD DETECTION SUMMARY FOR {args.period_type.upper()} PERIOD TYPE ===")
        
        print(f"Analysis Scope:")
        print(f"• Total Charts Analyzed: {summary['total_charts']}")
        print(f"• Successful Analyses: {summary['successful_count']}")
        print(f"• Failed Analyses: {summary['failed_count']}")
        print(f"• Total Series Analyzed: {total_series_analyzed}")
        print(f"• Total Records Found: {total_records_found}")
        
        if llm_summaries:
            print(f"\n=== DETAILED FINDINGS ===")
            for i, llm_result in enumerate(llm_summaries):
                if 'error' in llm_result:
                    continue
                    
                analysis = llm_result['analysis_summary']
                chart_title = analysis.get('chart_title', analysis.get('metric_name', 'Unknown Chart'))
                print(f"\n{chart_title}:")
                print(f"  • Records Detected: {analysis['records_detected']}")
                print(f"  • Series Analyzed: {analysis['total_series_analyzed']}")
                
                # Show record details
                for series in llm_result['series_analysis']:
                    if series['status'] != 'NORMAL':
                        print(f"  • {series['series_name']}: {series['current_value']} ({series['status']})")
                        if series.get('headline'):
                            print(f"    - Headline: {series['headline']}")
                        if series['record_details']:
                            if 'record_high' in series['record_details']:
                                high_details = series['record_details']['record_high']
                                print(f"    - Record High: {high_details['context']}")
                            if 'record_low' in series['record_details']:
                                low_details = series['record_details']['record_low']
                                print(f"    - Record Low: {low_details['context']}")
        else:
            print("\nNo series found across all charts.")
        

        
        # Generate comprehensive Markdown report
        comprehensive_md = []
        
        # Count total records found
        total_records = 0
        for llm_result in llm_summaries:
            if 'error' not in llm_result:
                for series in llm_result['series_analysis']:
                    if series['status'] != 'NORMAL':
                        total_records += 1
        
        # Get the metric name from the first result if available
        metric_name = "Unknown Metric"
        if llm_summaries:
            first_result = llm_summaries[0]
            if 'error' not in first_result:
                metric_name = first_result.get('analysis_summary', {}).get('metric_name', 'Unknown Metric')
        
        # Create dynamic title with record count and month
        current_month = datetime.now().strftime('%B')
        comprehensive_md.append(f"# {metric_name} - {total_records} New records detected in {current_month}")
        comprehensive_md.append("")
        
        # Collect all districts with their data for sorting
        district_data = []
        
        for llm_result in llm_summaries:
            if 'error' in llm_result:
                continue
                
            # Get district number from the chart metadata
            district_num = 0  # Default to citywide
            chart_metadata = llm_result.get('chart_metadata', {})
            if chart_metadata and 'district' in chart_metadata:
                district_value = chart_metadata['district']
                if district_value is not None:
                    try:
                        district_num = int(district_value)
                    except (ValueError, TypeError):
                        district_num = 0  # Default to citywide if can't parse
            else:
                # Fallback: try to extract from series name or chart title
                for series in llm_result['series_analysis']:
                    series_name = series.get('series_name', '')
                    if 'District' in series_name:
                        import re
                        district_match = re.search(r'District\s+(\d+)', series_name)
                        if district_match:
                            district_num = int(district_match.group(1))
                            break
                    else:
                        # Try to extract from chart title
                        chart_title = llm_result.get('analysis_summary', {}).get('chart_title', '')
                        district_match = re.search(r'District\s+(\d+)', chart_title)
                        if district_match:
                            district_num = int(district_match.group(1))
                            break
            
            # Process each series in this result
            for series in llm_result['series_analysis']:
                # Get the description and headline
                description = series.get('description', '')
                headline = series.get('headline', '')
                
                # For records, use the record context from create_llm_friendly_output
                record_details = series.get('record_details', {})
                if record_details:
                    if 'record_low' in record_details:
                        description = record_details['record_low']['context']
                    elif 'record_high' in record_details:
                        description = record_details['record_high']['context']
                
                # If no description, use the headline as fallback
                if not description:
                    description = series.get('headline', 'No description available')
                
                # Get historical context
                historical_context = series.get('historical_context', {})
                record_details = series.get('record_details', {})
                
                # Determine status and historical info
                status = series.get('status', 'NORMAL')
                current_value = series.get('current_value', 0)
                
                # Calculate years since earliest date for context
                earliest_date = historical_context.get('earliest_date', None)
                years_since_start = None
                if earliest_date:
                    try:
                        earliest_dt = datetime.fromisoformat(earliest_date.replace('Z', '+00:00'))
                        current_dt = datetime.now()
                        years_since_start = current_dt.year - earliest_dt.year
                    except:
                        years_since_start = 7  # Default to 7 years if we can't parse
                else:
                    years_since_start = 7  # Default to 7 years
                
                # Get historical context for non-record districts
                historical_info = ""
                if status == 'NORMAL':
                    # Show historical minimum with month/year for non-record districts
                    if historical_context:
                        historical_min = historical_context.get('historical_minimum', 0)
                        historical_min_period = historical_context.get('historical_minimum_period')
                        if historical_min_period:
                            historical_info = f" (Historical minimum: {historical_min} in {historical_min_period})"
                        elif current_value > historical_min:
                            historical_info = f" (Historical minimum: {historical_min})"
                
                district_data.append({
                    'district_num': district_num,
                    'description': description,
                    'headline': headline,
                    'status': status,
                    'current_value': current_value,
                    'historical_info': historical_info,
                    'years_since_start': years_since_start
                })
        
        # Sort by district number (0 for citywide, then 1-11)
        district_data.sort(key=lambda x: x['district_num'])
        
        # Generate sorted output
        for data in district_data:
            district_name = "Citywide" if data['district_num'] == 0 else f"District {data['district_num']}"
            
            if data['status'] != 'NORMAL':
                # Record set - show both headline and description
                comprehensive_md.append(f"- **{data['headline']}**")
                comprehensive_md.append(f"  {data['description']}")
            else:
                # No record - show current value and historical context
                comprehensive_md.append(f"- {district_name} 👮 Total Police Incidents: {data['current_value']}{data['historical_info']}")
        
        # Save comprehensive Markdown report
        comprehensive_md_filename = f"comprehensive_llm_analysis_{args.period_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        comprehensive_md_filepath = os.path.join(summary['summary_filepath'].rsplit('/', 1)[0], comprehensive_md_filename)
        
        with open(comprehensive_md_filepath, 'w') as f:
            f.write('\n'.join(comprehensive_md))
        
        print(f"Comprehensive Markdown report saved to: {comprehensive_md_filepath}")
        print(f"Original summary saved to: {summary['summary_filepath']}")
        
    else:
        # Show help if no arguments provided
        parser.print_help()

if __name__ == "__main__":
    main() 
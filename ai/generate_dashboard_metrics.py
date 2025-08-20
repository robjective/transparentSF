import os
import json
import logging
import traceback
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import sys
import pandas as pd
import re
import uuid
import math
from openai import OpenAI
import qdrant_client
from qdrant_client.http import models as rest
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import time
import argparse

# Add the ai directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.data_fetcher import set_dataset, fetch_data_from_api
from tools.db_utils import get_postgres_connection
from tools.genChart import generate_ytd_trend_chart

def clean_nan_values(obj):
    """
    Recursively clean NaN and infinity values from data structures to make them JSON-serializable.
    Replaces NaN/inf with None (which becomes null in JSON).
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):  # Handle pandas NaN types
        return None
    else:
        return obj
from tools.store_time_series import store_time_series_in_db

# Create logs directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Load environment variables to get LOG_LEVEL
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

# Configure logging with a single handler for both file and console
logger = logging.getLogger(__name__)
logger.setLevel(log_level)  # Use environment variable instead of hardcoded INFO
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add file handler
file_handler = logging.FileHandler(os.path.join(logs_dir, 'dashboard_metrics.log'))
file_handler.setLevel(log_level)  # Use environment variable instead of hardcoded INFO
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)  # Use environment variable instead of hardcoded INFO
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info(f"Dashboard metrics logging initialized with level: {log_level_str}")

def load_metrics_from_db():
    """Load metrics from the database instead of JSON files."""
    try:
        import psycopg2.extras
        
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to connect to database")
            return None
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query all active metrics from the database (both show_on_dash=true and false)
        cursor.execute("""
            SELECT 
                m.id,
                m.metric_name,
                m.metric_key,
                m.category,
                m.endpoint,
                m.summary,
                m.definition,
                m.data_sf_url,
                m.ytd_query,
                m.metric_query,
                m.location_fields,
                m.category_fields,
                m.city_id,
                m.display_order,
                m.is_active,
                m.show_on_dash,
                m.greendirection,
                d.title as dataset_title,
                d.category as dataset_category
            FROM metrics m
            LEFT JOIN datasets d ON m.endpoint = d.endpoint
            WHERE m.is_active = true
            ORDER BY m.display_order, m.metric_name
        """)
        
        metrics = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not metrics:
            logger.warning("No active metrics found in database")
            return {}
        
        # Convert to the expected dashboard queries format
        dashboard_queries = {}
        
        for metric in metrics:
            category = metric['category'] or 'uncategorized'
            
            # Initialize category if it doesn't exist
            if category not in dashboard_queries:
                dashboard_queries[category] = {}
            
            # Use the category as both top-level and subcategory for simplicity
            # This matches the existing structure where we have crime -> Crime -> queries
            subcategory = category.title()
            if subcategory not in dashboard_queries[category]:
                dashboard_queries[category][subcategory] = {
                    'endpoint': metric['endpoint'],
                    'queries': {}
                }
            
            # Create the query entry
            query_name = metric['metric_name']
            query_data = {
                'id': metric['id'],
                'endpoint': metric['endpoint'],
                'summary': metric['summary'] or '',
                'definition': metric['definition'] or '',
                'data_sf_url': metric['data_sf_url'] or '',
                'ytd_query': metric['ytd_query'] or '',
                'metric_query': metric['metric_query'] or '',
                'dataset_title': metric['dataset_title'] or '',
                'dataset_category': metric['dataset_category'] or '',
                'location_fields': metric['location_fields'] or [],
                'category_fields': metric['category_fields'] or [],
                'city_id': metric['city_id'],
                'display_order': metric['display_order'],
                'is_active': metric['is_active'],
                'show_on_dash': metric['show_on_dash'],
                'greendirection': metric['greendirection'] or 'up'
            }
            
            dashboard_queries[category][subcategory]['queries'][query_name] = query_data
        
        logger.info(f"Loaded {len(metrics)} metrics from database across {len(dashboard_queries)} categories")
        return dashboard_queries
        
    except Exception as e:
        logger.error(f"Error loading metrics from database: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_single_metric_from_db(metric_id):
    """Load a single metric from the database by ID."""
    try:
        import psycopg2.extras
        
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to connect to database")
            return None
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query the specific metric from the database
        # Handle both numeric and string IDs
        if str(metric_id).isdigit():
            # Numeric ID - search by id field
            cursor.execute("""
                SELECT 
                    m.id,
                    m.metric_name,
                    m.metric_key,
                    m.category,
                    m.endpoint,
                    m.summary,
                    m.definition,
                    m.data_sf_url,
                    m.ytd_query,
                    m.metric_query,
                    m.location_fields,
                    m.category_fields,
                    m.city_id,
                    m.display_order,
                    m.is_active,
                    m.show_on_dash,
                    m.greendirection,
                    d.title as dataset_title,
                    d.category as dataset_category
                FROM metrics m
                LEFT JOIN datasets d ON m.endpoint = d.endpoint
                WHERE m.id = %s AND m.is_active = true
            """, (int(metric_id),))
        else:
            # String ID - search by metric_key
            cursor.execute("""
                SELECT 
                    m.id,
                    m.metric_name,
                    m.metric_key,
                    m.category,
                    m.endpoint,
                    m.summary,
                    m.definition,
                    m.data_sf_url,
                    m.ytd_query,
                    m.metric_query,
                    m.location_fields,
                    m.category_fields,
                    m.city_id,
                    m.display_order,
                    m.is_active,
                    m.show_on_dash,
                    m.greendirection,
                    d.title as dataset_title,
                    d.category as dataset_category
                FROM metrics m
                LEFT JOIN datasets d ON m.endpoint = d.endpoint
                WHERE m.metric_key = %s AND m.is_active = true
            """, (str(metric_id),))
        
        metric = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not metric:
            logger.error(f"Metric with ID {metric_id} not found in database")
            return None
        
        # Convert to the expected dashboard queries format
        category = metric['category'] or 'uncategorized'
        subcategory = category.title()
        query_name = metric['metric_name']
        
        query_data = {
            'id': metric['id'],
            'endpoint': metric['endpoint'],
            'summary': metric['summary'] or '',
            'definition': metric['definition'] or '',
            'data_sf_url': metric['data_sf_url'] or '',
            'ytd_query': metric['ytd_query'] or '',
            'metric_query': metric['metric_query'] or '',
            'dataset_title': metric['dataset_title'] or '',
            'dataset_category': metric['dataset_category'] or '',
            'location_fields': metric['location_fields'] or [],
            'category_fields': metric['category_fields'] or [],
            'city_id': metric['city_id'],
            'display_order': metric['display_order'],
            'is_active': metric['is_active'],
            'show_on_dash': metric['show_on_dash'],
            'greendirection': metric['greendirection'] or 'up'
        }
        
        # Return in the expected format
        dashboard_queries = {
            category: {
                subcategory: {
                    'endpoint': metric['endpoint'],
                    'queries': {
                        query_name: query_data
                    }
                }
            }
        }
        
        logger.info(f"Loaded metric {metric_id} ({query_name}) from database")
        return dashboard_queries, category, subcategory, query_name, query_data
        
    except Exception as e:
        logger.error(f"Error loading metric {metric_id} from database: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_json_file(file_path):
    """Load a JSON file and return its contents. DEPRECATED - use database functions instead."""
    logger.warning(f"load_json_file is deprecated. Loading from {file_path} - consider migrating to database")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def get_date_ranges(target_date=None, query=None):
    """Calculate the date ranges for YTD comparisons."""
    # If no target_date provided, use yesterday
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    # This year's range
    this_year = target_date.year
    this_year_start = f"{this_year}-01-01"
    
    # Check if query contains date_trunc_ym in GROUP BY
    is_monthly_query = query and 'date_trunc_ym' in query and 'GROUP BY' in query
    
    if is_monthly_query:
        # For monthly queries, set end date to last day of previous month
        if target_date.month == 1:
            # If we're in January, use December of previous year
            end_year = target_date.year - 1
            end_month = 12
        else:
            end_year = target_date.year
            end_month = target_date.month - 1
            
        # Calculate last day of the month
        if end_month == 12:
            last_day = 31
        elif end_month in [4, 6, 9, 11]:
            last_day = 30
        elif end_month == 2:
            # Handle leap years
            if end_year % 4 == 0 and (end_year % 100 != 0 or end_year % 400 == 0):
                last_day = 29
            else:
                last_day = 28
        else:
            last_day = 31
            
        this_year_end = f"{end_year}-{end_month:02d}-{last_day}"
        logger.info(f"Using last day of previous month for end date: {this_year_end}")
    else:
        this_year_end = target_date.strftime('%Y-%m-%d')
    
    # Last year's range - use same day-of-year
    last_year = this_year - 1
    last_year_start = f"{last_year}-01-01"
    
    # For last year's end date, we need to handle leap years correctly
    # If this year is a leap year and today is Feb 29, use Feb 28 for last year
    if target_date.month == 2 and target_date.day == 29:
        last_year_end = f"{last_year}-02-28"
    else:
        # For last year's end date, use the same logic as this year's end date
        if is_monthly_query:
            # Use the same month/day as this_year_end but for last year
            last_year_end = f"{last_year}-{end_month:02d}-{last_day}"
        else:
            last_year_end = target_date.replace(year=last_year).strftime('%Y-%m-%d')
    
    # Ensure this_year_end is never later than yesterday
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    if this_year_end > yesterday_str:
        logger.info(f"Capping this_year_end date to yesterday: {this_year_end} -> {yesterday_str}")
        this_year_end = yesterday_str
        # Also adjust last_year_end to maintain the same day of year
        last_year_end = yesterday.replace(year=yesterday.year-1).strftime('%Y-%m-%d')
    
    # Calculate fiscal year ranges
    # San Francisco fiscal year runs from July 1 to June 30
    # If we're in the first half of the year (Jan-Jun), current fiscal year started last July
    # If we're in the second half of the year (Jul-Dec), current fiscal year started this July
    if target_date.month >= 7:  # July or later
        this_fiscal_year = str(this_year + 1)  # Fiscal year is next calendar year
        last_fiscal_year = str(this_year)
    else:  # January through June
        this_fiscal_year = str(this_year)
        last_fiscal_year = str(this_year - 1)
    
    logger.info(f"Date ranges: this_year={this_year_start} to {this_year_end}, last_year={last_year_start} to {last_year_end}")
    logger.info(f"Fiscal year ranges: this_fiscal_year={this_fiscal_year}, last_fiscal_year={last_fiscal_year}")
    
    return {
        'this_year_start': this_year_start,
        'this_year_end': this_year_end,
        'last_year_start': last_year_start,
        'last_year_end': last_year_end,
        'this_fiscal_year_start': this_fiscal_year,
        'this_fiscal_year_end': this_fiscal_year,
        'last_fiscal_year_start': last_fiscal_year,
        'last_fiscal_year_end': last_fiscal_year
    }

def debug_query(query, endpoint, date_ranges, query_name=None):
    """Debug a query by printing detailed information about the query and its execution."""
    logger.info("=" * 80)
    logger.info(f"DEBUG QUERY: {query_name}")
    logger.info("-" * 80)
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Date ranges: {date_ranges}")
    
    # Extract year from date ranges
    this_year = datetime.strptime(date_ranges['this_year_end'], '%Y-%m-%d').year
    last_year = this_year - 1
    
    # Check for hardcoded dates in the query
    this_year_pattern = re.compile(f"'{this_year}-\\d{{2}}-\\d{{2}}'")
    last_year_pattern = re.compile(f"'{last_year}-\\d{{2}}-\\d{{2}}'")
    
    this_year_dates = this_year_pattern.findall(query)
    last_year_dates = last_year_pattern.findall(query)
    
    if this_year_dates:
        logger.info(f"Found hardcoded this year dates: {this_year_dates}")
    if last_year_dates:
        logger.info(f"Found hardcoded last year dates: {last_year_dates}")
    
    # Check for date placeholders
    placeholders = ['this_year_start', 'this_year_end', 'last_year_start', 'last_year_end']
    for placeholder in placeholders:
        if placeholder in query:
            logger.info(f"Found placeholder: {placeholder} = {date_ranges.get(placeholder, 'Not in date_ranges')}")
    
    # Try to determine the actual last data date for this endpoint
    try:
        for date_col in ['date_issued', 'arrest_date', 'received_datetime', 'date']:
            last_date_query = f"SELECT max({date_col}) as last_data_date"
            context_variables = {}
            last_date_result = set_dataset(context_variables, endpoint=endpoint, query=last_date_query)
            
            if last_date_result.get('status') == 'success' and 'dataset' in context_variables and not context_variables['dataset'].empty:
                if not context_variables['dataset']['last_data_date'].iloc[0] is None:
                    actual_last_date = pd.to_datetime(context_variables['dataset']['last_data_date'].iloc[0]).date()
                    logger.info(f"Found actual last data date from {date_col}: {actual_last_date}")
                    break
    except Exception as e:
        logger.warning(f"Error determining last data date: {str(e)}")
    
    # Now modify the query with the updated date ranges
    modified_query = query
    
    # Replace date placeholders in the query
    for key, value in date_ranges.items():
        modified_query = modified_query.replace(key, f"'{value}'")
    
    # Check for hardcoded date patterns in the query and replace them
    this_year_pattern = re.compile(f"'{this_year}-\\d{{2}}-\\d{{2}}'")
    last_year_pattern = re.compile(f"'{last_year}-\\d{{2}}-\\d{{2}}'")
    
    # Find all hardcoded dates for this year and last year
    this_year_dates = this_year_pattern.findall(modified_query)
    last_year_dates = last_year_pattern.findall(modified_query)
    
    # Replace the latest this_year date with the actual max date
    if this_year_dates:
        latest_this_year_date = max(this_year_dates)
        modified_query = modified_query.replace(latest_this_year_date, f"'{date_ranges['this_year_end']}'")
        logger.info(f"Replaced hardcoded this year date {latest_this_year_date} with {date_ranges['this_year_end']}")
    
    # Replace the latest last_year date with the corresponding last year date
    if last_year_dates:
        # Filter out January 1st dates as these should remain as start dates
        jan_first = f"'{last_year}-01-01'"
        non_jan_first_dates = [date for date in last_year_dates if date != jan_first]
        
        if non_jan_first_dates:
            latest_last_year_date = max(non_jan_first_dates)
            modified_query = modified_query.replace(latest_last_year_date, f"'{date_ranges['last_year_end']}'")
            logger.info(f"Replaced hardcoded last year date {latest_last_year_date} with {date_ranges['last_year_end']}")
        else:
            logger.info(f"No non-January 1st last year dates to replace. Keeping {jan_first} as the start date.")
    
    logger.info(f"Modified query: {modified_query}")
    
    # Execute the query
    context_variables = {}
    result = set_dataset(context_variables, endpoint=endpoint, query=modified_query)
    logger.info(f"Query execution result status: {result.get('status')}")
    
    if result.get('status') == 'success' and 'dataset' in context_variables:
        df = context_variables['dataset']
        logger.info(f"Dataset retrieved successfully - Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Print the first few rows
        if not df.empty:
            logger.info("First few rows:")
            logger.info(df.head().to_string())
            
            # Get the max date
            for date_col in ['received_datetime', 'max_date', 'arrest_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    max_date = df[date_col].max()
                    if pd.notnull(max_date):
                        max_date = max_date.strftime('%Y-%m-%d')
                        logger.info(f"Max date determined from {date_col}: {max_date}")
                        break
            
            # Cap max_date to yesterday
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            if max_date and max_date > yesterday:
                logger.info(f"Capping max date from query results to yesterday: {max_date} -> {yesterday}")
                max_date = yesterday
    else:
        logger.error("Query failed or no data returned")
        if 'error' in result:
            logger.error(f"Error: {result['error']}")
        logger.error(f"Query URL: {result.get('queryURL')}")
    
    logger.info("=" * 80)
    return

def get_monthly_date_ranges(last_data_date):
    """Calculate monthly date ranges for comparison based on lastDataDate.
    
    Args:
        last_data_date (str): The last data date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary with monthly date ranges
    """
    try:
        # Parse the last data date
        last_date = datetime.strptime(last_data_date, '%Y-%m-%d').date()
        
        # Use today's date for monthly calculations, but cap it to the last data date
        today = datetime.now().date()
        if today > last_date:
            current_date = last_date
        else:
            current_date = today
        
        # Get the current month and year from current_date
        current_month = current_date.month
        current_year = current_date.year
        current_day = current_date.day
        
        # Calculate the start of current month
        current_month_start = date(current_year, current_month, 1)
        
        # Calculate the end of current month (up to last_data_date)
        current_month_end = last_date
        
        # Calculate the previous month
        if current_month == 1:
            # If we're in January, go to December of previous year
            previous_month = 12
            previous_year = current_year - 1
        else:
            previous_month = current_month - 1
            previous_year = current_year
        
        # Calculate the start of previous month
        previous_month_start = date(previous_year, previous_month, 1)
        
        # Calculate the end of previous month (same day as current month, but capped to month end)
        # First, try to use the same day
        try:
            previous_month_end = date(previous_year, previous_month, current_day)
        except ValueError:
            # If the day doesn't exist in the previous month (e.g., March 31 -> February 28/29),
            # use the last day of the previous month
            if previous_month == 12:
                next_month_start = date(previous_year + 1, 1, 1)
            else:
                next_month_start = date(previous_year, previous_month + 1, 1)
            previous_month_end = next_month_start - timedelta(days=1)
        
        # Cap the previous month end to the actual last day of that month
        if previous_month == 12:
            next_month_start = date(previous_year + 1, 1, 1)
        else:
            next_month_start = date(previous_year, previous_month + 1, 1)
        actual_previous_month_end = next_month_start - timedelta(days=1)
        
        if previous_month_end > actual_previous_month_end:
            previous_month_end = actual_previous_month_end
        
        monthly_ranges = {
            'this_month_start': current_month_start.strftime('%Y-%m-%d'),
            'this_month_end': current_month_end.strftime('%Y-%m-%d'),
            'last_month_start': previous_month_start.strftime('%Y-%m-%d'),
            'last_month_end': previous_month_end.strftime('%Y-%m-%d')
        }
        
        logger.info(f"Monthly date ranges for {last_data_date}: {monthly_ranges}")
        return monthly_ranges
        
    except Exception as e:
        logger.error(f"Error calculating monthly date ranges for {last_data_date}: {str(e)}")
        return None

def process_query_for_district(query, endpoint, date_ranges, query_name=None):
    """Process a single query and handle district-level aggregation from the same dataset."""
    try:
        logger.info(f"Processing query for endpoint {endpoint}, query_name: {query_name}")
        
        # Debug the query if it's for "Arrests Presented to DA"
        if query_name and "arrests presented" in query_name.lower():
            debug_query(query, endpoint, date_ranges, query_name)
        
        # Get date ranges if not provided
        if not date_ranges:
            date_ranges = get_date_ranges(query=query)
        
        # Now modify the query with the date ranges
        modified_query = query
        
        # Replace date placeholders in the query
        for key, value in date_ranges.items():
            modified_query = modified_query.replace(key, f"'{value}'")
        
        # Handle cases where the query uses direct year comparisons
        this_year = datetime.strptime(date_ranges['this_year_end'], '%Y-%m-%d').year
        last_year = this_year - 1
        
        logger.info(f"Processing years: this_year={this_year}, last_year={last_year}")
        
        # Check for hardcoded date patterns in the query and replace them
        # This is crucial for queries that have hardcoded dates like '2025-02-16'
        this_year_pattern = re.compile(f"'{this_year}-\\d{{2}}-\\d{{2}}'")
        last_year_pattern = re.compile(f"'{last_year}-\\d{{2}}-\\d{{2}}'")
        
        # Find all hardcoded dates for this year and last year
        this_year_dates = this_year_pattern.findall(modified_query)
        last_year_dates = last_year_pattern.findall(modified_query)
        
        # Replace the latest this_year date with the actual max date
        if this_year_dates:
            latest_this_year_date = max(this_year_dates)
            modified_query = modified_query.replace(latest_this_year_date, f"'{date_ranges['this_year_end']}'")
            logger.info(f"Replaced hardcoded this year date {latest_this_year_date} with {date_ranges['this_year_end']}")
        
        # Replace the latest last_year date with the corresponding last year date
        if last_year_dates:
            # Filter out January 1st dates as these should remain as start dates
            jan_first = f"'{last_year}-01-01'"
            non_jan_first_dates = [date for date in last_year_dates if date != jan_first]
            
            if non_jan_first_dates:
                latest_last_year_date = max(non_jan_first_dates)
                modified_query = modified_query.replace(latest_last_year_date, f"'{date_ranges['last_year_end']}'")
                logger.info(f"Replaced hardcoded last year date {latest_last_year_date} with {date_ranges['last_year_end']}")
            else:
                logger.info(f"No non-January 1st last year dates to replace. Keeping {jan_first} as the start date.")
        
        # Define all possible date patterns we need to fix
        date_patterns = [
            (f">= '{this_year}-01-01' AND < '{this_year}-01-01'",
             f">= '{this_year}-01-01' AND <= '{date_ranges['this_year_end']}'"),
            (f">= '{last_year}-01-01' AND < '{last_year}-01-01'",
             f">= '{last_year}-01-01' AND <= '{date_ranges['last_year_end']}'"),
            (f">= '{this_year}-01-01' AND <= '{this_year}-01-01'",
             f">= '{this_year}-01-01' AND <= '{date_ranges['this_year_end']}'"),
            (f">= '{last_year}-01-01' AND <= '{last_year}-01-01'",
             f">= '{last_year}-01-01' AND <= '{date_ranges['last_year_end']}'"),
            (f">= '{this_year}-01-01' AND date_issued < '{this_year}-01-01'",
             f">= '{this_year}-01-01' AND date_issued <= '{date_ranges['this_year_end']}'"),
            (f">= '{last_year}-01-01' AND date_issued < '{last_year}-01-01'",
             f">= '{last_year}-01-01' AND date_issued <= '{date_ranges['last_year_end']}'")
        ]
        
        # Apply all pattern replacements
        for pattern, replacement in date_patterns:
            if pattern in modified_query:
                logger.info(f"Replacing date pattern: {pattern} -> {replacement}")
                modified_query = modified_query.replace(pattern, replacement)
        
        logger.info(f"Modified query: {modified_query}")
        
        # Execute the query
        context_variables = {}
        result = set_dataset(context_variables, endpoint=endpoint, query=modified_query)
        logger.info(f"Query execution result status: {result.get('status')}")
        
        if result.get('status') == 'success' and 'dataset' in context_variables:
            query_info = {
                'original_query': query,
                'executed_query': modified_query
            }
            
            df = context_variables['dataset']
            logger.info(f"Dataset retrieved successfully - Shape: {df.shape}")
            
            results = {}
            has_district = 'supervisor_district' in df.columns
            logger.info(f"Query has district data: {has_district}")
            
            # Get the max date from the dataset
            max_date = None
            for date_col in ['max_date', 'received_datetime', 'arrest_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    max_date = df[date_col].max()
                    if pd.notnull(max_date):
                        max_date = max_date.strftime('%Y-%m-%d')
                        logger.info(f"Max date determined from {date_col}: {max_date}")
                        break
            
            # Cap max_date to yesterday
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            if max_date and max_date > yesterday:
                logger.info(f"Capping max date from query results to yesterday: {max_date} -> {yesterday}")
                max_date = yesterday
            
            # Calculate monthly date ranges for monthly comparison
            monthly_ranges = None
            if max_date:
                monthly_ranges = get_monthly_date_ranges(max_date)
            
            if has_district:
                # Check if this is an average metric by looking for AVG() in the query
                def detect_avg_aggregation(query):
                    """Detect if a query uses an AVG() aggregation function."""
                    if not query:
                        return False
                    import re
                    avg_pattern = r'AVG\s*\(([^)]+)\)'
                    avg_matches = re.findall(avg_pattern, query, re.IGNORECASE)
                    return len(avg_matches) > 0
                
                is_response_time = query_name and ('response time' in query_name.lower() or 'response (minutes)' in query_name.lower())
                is_avg_metric = detect_avg_aggregation(query)
                is_average_metric = is_response_time or is_avg_metric
                logger.info(f"Processing as response time metric: {is_response_time}")
                logger.info(f"Processing as average metric (AVG detected): {is_avg_metric}")
                logger.info(f"Processing as average metric (combined): {is_average_metric}")
                
                if is_average_metric:
                    if 'this_year' not in df.columns or 'last_year' not in df.columns:
                        logger.error(f"Required columns not found in dataset. Available columns: {df.columns.tolist()}")
                        return None
                    
                    df[['this_year', 'last_year']] = df[['this_year', 'last_year']].apply(pd.to_numeric, errors='coerce')
                    
                    # Calculate citywide and district averages
                    results['0'] = {
                        'lastYear': int(df['last_year'].mean()) if pd.notnull(df['last_year'].mean()) else 0,
                        'thisYear': int(df['this_year'].mean()) if pd.notnull(df['this_year'].mean()) else 0,
                        'lastDataDate': max_date
                    }
                    
                    for district in range(1, 12):
                        district_df = df[df['supervisor_district'] == str(district)]
                        if not district_df.empty:
                            results[str(district)] = {
                                'lastYear': int(district_df['last_year'].mean()) if pd.notnull(district_df['last_year'].mean()) else 0,
                                'thisYear': int(district_df['this_year'].mean()) if pd.notnull(district_df['this_year'].mean()) else 0,
                                'lastDataDate': max_date
                            }
                else:
                    # For non-response time metrics
                    if not df.empty and 'last_year' in df.columns and 'this_year' in df.columns:
                        df[['last_year', 'this_year']] = df[['last_year', 'this_year']].apply(pd.to_numeric, errors='coerce')
                        
                        results['0'] = {
                            'lastYear': int(df['last_year'].sum()),
                            'thisYear': int(df['this_year'].sum()),
                            'lastDataDate': max_date
                        }
                        
                        for district in range(1, 12):
                            district_df = df[df['supervisor_district'] == str(district)]
                            if not district_df.empty:
                                results[str(district)] = {
                                    'lastYear': int(district_df['last_year'].sum()),
                                    'thisYear': int(district_df['this_year'].sum()),
                                    'lastDataDate': max_date
                                }
            else:
                # For non-district queries, just return the total from first row
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    results['0'] = {
                        'lastYear': int(float(row.get('last_year', 0))),
                        'thisYear': int(float(row.get('this_year', 0))),
                        'lastDataDate': max_date
                    }
                    
                    # Monthly data processing removed - using MTD instead
            
            logger.info(f"Query processing completed successfully for {query_name}")
            
            # Print the final results if it's the specific metric we're debugging
            if query_name and "arrests presented" in query_name.lower():
                logger.info("=" * 80)
                logger.info(f"FINAL RESULTS FOR: {query_name}")
                logger.info("-" * 80)
                for district, district_results in results.items():
                    logger.info(f"District {district}: thisYear={district_results['thisYear']}, lastYear={district_results['lastYear']}, lastDataDate={district_results['lastDataDate']}")
                    if 'thisMonth' in district_results:
                        logger.info(f"  Monthly: thisMonth={district_results.get('thisMonth', 'N/A')}, lastMonth={district_results.get('lastMonth', 'N/A')}")
                logger.info("=" * 80)
            
            return {
                'results': results,
                'queries': query_info
            }
            
        else:
            logger.error("Query failed or no data returned")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
            logger.error(f"Query URL: {result.get('queryURL')}")
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return None

def process_query_for_district_optimized(query, endpoint, date_ranges, query_name=None, monthly_ranges=None, mtd_ranges=None):
    """Process a single query and handle district-level aggregation from the same dataset with optimized query count."""
    try:
        logger.info(f"Processing optimized query for endpoint {endpoint}, query_name: {query_name}")
        
        # Debug the query if it's for "Arrests Presented to DA"
        if query_name and "arrests presented" in query_name.lower():
            debug_query(query, endpoint, date_ranges, query_name)
        
        # Get date ranges if not provided
        if not date_ranges:
            date_ranges = get_date_ranges(query=query)
        
        # Now modify the query with the date ranges
        modified_query = query
        
        # Replace date placeholders in the query
        for key, value in date_ranges.items():
            modified_query = modified_query.replace(key, f"'{value}'")
        
        # Handle cases where the query uses direct year comparisons
        this_year = datetime.strptime(date_ranges['this_year_end'], '%Y-%m-%d').year
        last_year = this_year - 1
        
        logger.info(f"Processing years: this_year={this_year}, last_year={last_year}")
        
        # Check for hardcoded date patterns in the query and replace them
        # This is crucial for queries that have hardcoded dates like '2025-02-16'
        this_year_pattern = re.compile(f"'{this_year}-\\d{{2}}-\\d{{2}}'")
        last_year_pattern = re.compile(f"'{last_year}-\\d{{2}}-\\d{{2}}'")
        
        # Find all hardcoded dates for this year and last year
        this_year_dates = this_year_pattern.findall(modified_query)
        last_year_dates = last_year_pattern.findall(modified_query)
        
        # Replace the latest this_year date with the actual max date
        if this_year_dates:
            latest_this_year_date = max(this_year_dates)
            modified_query = modified_query.replace(latest_this_year_date, f"'{date_ranges['this_year_end']}'")
            logger.info(f"Replaced hardcoded this year date {latest_this_year_date} with {date_ranges['this_year_end']}")
        
        # Replace the latest last_year date with the corresponding last year date
        if last_year_dates:
            # Filter out January 1st dates as these should remain as start dates
            jan_first = f"'{last_year}-01-01'"
            non_jan_first_dates = [date for date in last_year_dates if date != jan_first]
            
            if non_jan_first_dates:
                latest_last_year_date = max(non_jan_first_dates)
                modified_query = modified_query.replace(latest_last_year_date, f"'{date_ranges['last_year_end']}'")
                logger.info(f"Replaced hardcoded last year date {latest_last_year_date} with {date_ranges['last_year_end']}")
            else:
                logger.info(f"No non-January 1st last year dates to replace. Keeping {jan_first} as the start date.")
        
        # Define all possible date patterns we need to fix
        date_patterns = [
            (f">= '{this_year}-01-01' AND < '{this_year}-01-01'",
             f">= '{this_year}-01-01' AND <= '{date_ranges['this_year_end']}'"),
            (f">= '{last_year}-01-01' AND < '{last_year}-01-01'",
             f">= '{last_year}-01-01' AND <= '{date_ranges['last_year_end']}'"),
            (f">= '{this_year}-01-01' AND <= '{this_year}-01-01'",
             f">= '{this_year}-01-01' AND <= '{date_ranges['this_year_end']}'"),
            (f">= '{last_year}-01-01' AND <= '{last_year}-01-01'",
             f">= '{last_year}-01-01' AND <= '{date_ranges['last_year_end']}'"),
            (f">= '{this_year}-01-01' AND date_issued < '{this_year}-01-01'",
             f">= '{this_year}-01-01' AND date_issued <= '{date_ranges['this_year_end']}'"),
            (f">= '{last_year}-01-01' AND date_issued < '{last_year}-01-01'",
             f">= '{last_year}-01-01' AND date_issued <= '{date_ranges['last_year_end']}'")
        ]
        
        # Apply all pattern replacements
        for pattern, replacement in date_patterns:
            if pattern in modified_query:
                logger.info(f"Replacing date pattern: {pattern} -> {replacement}")
                modified_query = modified_query.replace(pattern, replacement)
        
        logger.info(f"Modified query: {modified_query}")
        
        # Execute the query
        context_variables = {}
        result = set_dataset(context_variables, endpoint=endpoint, query=modified_query)
        logger.info(f"Query execution result status: {result.get('status')}")
        
        if result.get('status') == 'success' and 'dataset' in context_variables:
            query_info = {
                'original_query': query,
                'executed_query': modified_query
            }
            
            df = context_variables['dataset']
            logger.info(f"Dataset retrieved successfully - Shape: {df.shape}")
            
            results = {}
            has_district = 'supervisor_district' in df.columns
            logger.info(f"Query has district data: {has_district}")
            
            # Get the max date from the dataset
            max_date = None
            for date_col in ['max_date', 'received_datetime', 'arrest_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    max_date = df[date_col].max()
                    if pd.notnull(max_date):
                        max_date = max_date.strftime('%Y-%m-%d')
                        logger.info(f"Max date determined from {date_col}: {max_date}")
                        break
            
            # Cap max_date to yesterday
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            if max_date and max_date > yesterday:
                logger.info(f"Capping max date from query results to yesterday: {max_date} -> {yesterday}")
                max_date = yesterday
            
            # Use provided monthly and MTD ranges, or calculate them if not provided
            if monthly_ranges is None and max_date:
                monthly_ranges = get_monthly_date_ranges(max_date)
            
            if mtd_ranges is None and max_date:
                mtd_ranges = get_mtd_date_ranges(max_date)
            
            if has_district:
                # Check if this is an average metric by looking for AVG() in the query
                def detect_avg_aggregation(query):
                    """Detect if a query uses an AVG() aggregation function."""
                    if not query:
                        return False
                    import re
                    avg_pattern = r'AVG\s*\(([^)]+)\)'
                    avg_matches = re.findall(avg_pattern, query, re.IGNORECASE)
                    return len(avg_matches) > 0
                
                is_response_time = query_name and ('response time' in query_name.lower() or 'response (minutes)' in query_name.lower())
                is_avg_metric = detect_avg_aggregation(query)
                is_average_metric = is_response_time or is_avg_metric
                logger.info(f"Processing as response time metric: {is_response_time}")
                logger.info(f"Processing as average metric (AVG detected): {is_avg_metric}")
                logger.info(f"Processing as average metric (combined): {is_average_metric}")
                
                if is_average_metric:
                    if 'this_year' not in df.columns or 'last_year' not in df.columns:
                        logger.error(f"Required columns not found in dataset. Available columns: {df.columns.tolist()}")
                        return None
                    
                    df[['this_year', 'last_year']] = df[['this_year', 'last_year']].apply(pd.to_numeric, errors='coerce')
                    
                    # Calculate citywide average
                    results['0'] = {
                        'lastYear': int(df['last_year'].mean()) if pd.notnull(df['last_year'].mean()) else 0,
                        'thisYear': int(df['this_year'].mean()) if pd.notnull(df['this_year'].mean()) else 0,
                        'lastDataDate': max_date
                    }
                    
                    # Calculate district averages from the same dataset
                    for district in range(1, 12):
                        district_df = df[df['supervisor_district'] == str(district)]
                        if not district_df.empty:
                            results[str(district)] = {
                                'lastYear': int(district_df['last_year'].mean()) if pd.notnull(district_df['last_year'].mean()) else 0,
                                'thisYear': int(district_df['this_year'].mean()) if pd.notnull(district_df['this_year'].mean()) else 0,
                                'lastDataDate': max_date
                            }
                else:
                    # For non-response time metrics
                    if not df.empty and 'last_year' in df.columns and 'this_year' in df.columns:
                        df[['last_year', 'this_year']] = df[['last_year', 'this_year']].apply(pd.to_numeric, errors='coerce')
                        
                        # Calculate citywide total
                        results['0'] = {
                            'lastYear': int(df['last_year'].sum()),
                            'thisYear': int(df['this_year'].sum()),
                            'lastDataDate': max_date
                        }
                        
                        # Calculate district totals from the same dataset
                        for district in range(1, 12):
                            district_df = df[df['supervisor_district'] == str(district)]
                            if not district_df.empty:
                                results[str(district)] = {
                                    'lastYear': int(district_df['last_year'].sum()),
                                    'thisYear': int(district_df['this_year'].sum()),
                                    'lastDataDate': max_date
                                }
            else:
                # For non-district queries, just return the total from first row
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    results['0'] = {
                        'lastYear': int(float(row.get('last_year', 0))),
                        'thisYear': int(float(row.get('this_year', 0))),
                        'lastDataDate': max_date
                    }
            
            # Monthly query processing removed - using MTD instead
            
            # Process MTD data for all districts in one query if available
            if mtd_ranges:
                mtd_data_all = process_mtd_query_optimized(query, endpoint, mtd_ranges, date_ranges, query_name)
                if mtd_data_all:
                    # Add MTD data to all districts
                    for district_key in results.keys():
                        if district_key in mtd_data_all:
                            results[district_key].update(mtd_data_all[district_key])
                    # Add MTD query to query_info
                    if 'mtd_query' in mtd_data_all:
                        query_info['mtd_query'] = mtd_data_all['mtd_query']
                        query_info['executed_mtd_query'] = mtd_data_all['executed_mtd_query']
            
            logger.info(f"Optimized query processing completed successfully for {query_name}")
            
            # Print the final results if it's the specific metric we're debugging
            if query_name and "arrests presented" in query_name.lower():
                logger.info("=" * 80)
                logger.info(f"FINAL RESULTS FOR: {query_name}")
                logger.info("-" * 80)
                for district, district_results in results.items():
                    logger.info(f"District {district}: thisYear={district_results['thisYear']}, lastYear={district_results['lastYear']}, lastDataDate={district_results['lastDataDate']}")
                    if 'thisMonth' in district_results:
                        logger.info(f"  Monthly: thisMonth={district_results.get('thisMonth', 'N/A')}, lastMonth={district_results.get('lastMonth', 'N/A')}")
                    if 'thisMtd' in district_results:
                        logger.info(f"  MTD: thisMtd={district_results.get('thisMtd', 'N/A')}, lastMtd={district_results.get('lastMtd', 'N/A')}")
                logger.info("=" * 80)
            
            return {
                'results': results,
                'queries': query_info
            }
            
        else:
            logger.error("Query failed or no data returned")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
            logger.error(f"Query URL: {result.get('queryURL')}")
    except Exception as e:
        logger.error(f"Error executing optimized query: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return None

# Monthly query functions removed - using MTD instead

def get_mtd_date_ranges(last_data_date):
    """Calculate MTD (Month-to-Date) date ranges for comparison based on lastDataDate.
    
    Args:
        last_data_date (str): The last data date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary with MTD date ranges
    """
    try:
        # Parse the last data date
        last_date = datetime.strptime(last_data_date, '%Y-%m-%d').date()
        
        # Use today's date for MTD calculations, but cap it to the last data date
        today = datetime.now().date()
        if today > last_date:
            current_date = last_date
        else:
            current_date = today
        
        # Get the current month and year from current_date
        current_month = current_date.month
        current_year = current_date.year
        current_day = current_date.day
        
        # Calculate the start of current month
        current_month_start = date(current_year, current_month, 1)
        
        # Calculate the end of current month (up to last_data_date)
        current_month_end = last_date
        
        # Calculate the previous month
        if current_month == 1:
            # If we're in January, go to December of previous year
            previous_month = 12
            previous_year = current_year - 1
        else:
            previous_month = current_month - 1
            previous_year = current_year
        
        # Calculate the start of previous month
        previous_month_start = date(previous_year, previous_month, 1)
        
        # Calculate the end of previous month (same day as current month, but capped to month end)
        # For MTD, we want to compare the same day of the month
        try:
            previous_month_end = date(previous_year, previous_month, current_day)
        except ValueError:
            # If the day doesn't exist in the previous month (e.g., March 31 -> February 28/29),
            # use the last day of the previous month
            if previous_month == 12:
                next_month_start = date(previous_year + 1, 1, 1)
            else:
                next_month_start = date(previous_year, previous_month + 1, 1)
            previous_month_end = next_month_start - timedelta(days=1)
        
        # For MTD, we want to compare the same day of the month, not the end of the month
        # So we cap the previous month end to the same day as the current month end
        # But we also need to cap it to the actual last day of that month
        if previous_month == 12:
            next_month_start = date(previous_year + 1, 1, 1)
        else:
            next_month_start = date(previous_year, previous_month + 1, 1)
        actual_previous_month_end = next_month_start - timedelta(days=1)
        
        # Use the minimum of the same day and the actual month end
        previous_month_end = min(previous_month_end, actual_previous_month_end)
        
        mtd_ranges = {
            'this_mtd_start': current_month_start.strftime('%Y-%m-%d'),
            'this_mtd_end': current_month_end.strftime('%Y-%m-%d'),
            'last_mtd_start': previous_month_start.strftime('%Y-%m-%d'),
            'last_mtd_end': previous_month_end.strftime('%Y-%m-%d')
        }
        
        logger.info(f"MTD date ranges for {last_data_date}: {mtd_ranges}")
        return mtd_ranges
        
    except Exception as e:
        logger.error(f"Error calculating MTD date ranges for {last_data_date}: {str(e)}")
        return None

def process_mtd_query_optimized(query, endpoint, mtd_ranges, date_ranges, query_name=None):
    """Process a query to get MTD (Month-to-Date) comparison data for all districts in one query."""
    try:
        logger.info(f"Processing optimized MTD query for {query_name}")
        
        # Create a modified query for MTD comparison
        # Replace the year placeholders with MTD placeholders
        mtd_query = query.replace('this_year_start', 'this_mtd_start')
        mtd_query = mtd_query.replace('this_year_end', 'this_mtd_end')
        mtd_query = mtd_query.replace('last_year_start', 'last_mtd_start')
        mtd_query = mtd_query.replace('last_year_end', 'last_mtd_end')
        
        # Handle fiscal year variables - use the same fiscal year values from the main query
        # Replace fiscal year variables with actual values from date_ranges
        mtd_query = mtd_query.replace('this_fiscal_year_start', f"'{date_ranges['this_fiscal_year_start']}'")
        mtd_query = mtd_query.replace('this_fiscal_year_end', f"'{date_ranges['this_fiscal_year_end']}'")
        mtd_query = mtd_query.replace('last_fiscal_year_start', f"'{date_ranges['last_fiscal_year_start']}'")
        mtd_query = mtd_query.replace('last_fiscal_year_end', f"'{date_ranges['last_fiscal_year_end']}'")
        
        # Replace the column names in the SELECT clause
        mtd_query = mtd_query.replace('this_year', 'this_mtd')
        mtd_query = mtd_query.replace('last_year', 'last_mtd')
        
        # Replace date placeholders with actual MTD ranges
        for key, value in mtd_ranges.items():
            mtd_query = mtd_query.replace(key, f"'{value}'")
        
        # No hardcoded date replacement needed - queries use placeholders that are already replaced above
        
        logger.info(f"Optimized MTD query: {mtd_query}")
        
        # Execute the MTD query
        context_variables = {}
        result = set_dataset(context_variables, endpoint=endpoint, query=mtd_query)
        
        if result.get('status') == 'success' and 'dataset' in context_variables:
            df = context_variables['dataset']
            
            if not df.empty and 'this_mtd' in df.columns and 'last_mtd' in df.columns:
                df[['this_mtd', 'last_mtd']] = df[['this_mtd', 'last_mtd']].apply(pd.to_numeric, errors='coerce')
                
                # Check if this is an average metric by looking for AVG() in the query
                def detect_avg_aggregation(query):
                    """Detect if a query uses an AVG() aggregation function."""
                    if not query:
                        return False
                    import re
                    avg_pattern = r'AVG\s*\(([^)]+)\)'
                    avg_matches = re.findall(avg_pattern, query, re.IGNORECASE)
                    return len(avg_matches) > 0
                
                is_response_time = query_name and ('response time' in query_name.lower() or 'response (minutes)' in query_name.lower())
                is_avg_metric = detect_avg_aggregation(query)
                is_average_metric = is_response_time or is_avg_metric
                
                mtd_results = {}
                
                # Check if we have district data
                has_district = 'supervisor_district' in df.columns
                
                if has_district:
                    # Calculate citywide MTD data
                    if is_average_metric:
                        this_mtd_value = int(df['this_mtd'].mean()) if pd.notnull(df['this_mtd'].mean()) else 0
                        last_mtd_value = int(df['last_mtd'].mean()) if pd.notnull(df['last_mtd'].mean()) else 0
                    else:
                        this_mtd_value = int(df['this_mtd'].sum())
                        last_mtd_value = int(df['last_mtd'].sum())
                    
                    mtd_results['0'] = {
                        'thisMtd': this_mtd_value,
                        'lastMtd': last_mtd_value
                    }
                    
                    # Calculate district MTD data from the same dataset
                    for district in range(1, 12):
                        district_df = df[df['supervisor_district'] == str(district)]
                        if not district_df.empty:
                            if is_average_metric:
                                this_mtd_value = int(district_df['this_mtd'].mean()) if pd.notnull(district_df['this_mtd'].mean()) else 0
                                last_mtd_value = int(district_df['last_mtd'].mean()) if pd.notnull(district_df['last_mtd'].mean()) else 0
                            else:
                                this_mtd_value = int(district_df['this_mtd'].sum())
                                last_mtd_value = int(district_df['last_mtd'].sum())
                            
                            mtd_results[str(district)] = {
                                'thisMtd': this_mtd_value,
                                'lastMtd': last_mtd_value
                            }
                else:
                    # For non-district queries, just return the total
                    if is_average_metric:
                        this_mtd_value = int(df['this_mtd'].mean()) if pd.notnull(df['this_mtd'].mean()) else 0
                        last_mtd_value = int(df['last_mtd'].mean()) if pd.notnull(df['last_mtd'].mean()) else 0
                    else:
                        this_mtd_value = int(df['this_mtd'].sum())
                        last_mtd_value = int(df['last_mtd'].sum())
                    
                    mtd_results['0'] = {
                        'thisMtd': this_mtd_value,
                        'lastMtd': last_mtd_value
                    }
                
                logger.info(f"Optimized MTD data for {query_name}: {mtd_results}")
                # Add query information to the results
                mtd_results['mtd_query'] = query
                mtd_results['executed_mtd_query'] = mtd_query
                return mtd_results
            else:
                logger.warning(f"No MTD data columns found in dataset for {query_name}. Available columns: {df.columns.tolist()}")
        else:
            logger.error(f"Optimized MTD query failed for {query_name}")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
    
    except Exception as e:
        logger.error(f"Error processing optimized MTD query for {query_name}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

def process_ytd_trend_query(query, endpoint, date_ranges=None, target_date=None, query_name=None, district=None):
    """Process a YTD trend query to get historical daily counts."""
    try:
        logger.info(f"Processing YTD trend query for {query_name}" + (f" (District {district})" if district else ""))
        
        # Get date ranges if not provided
        if not date_ranges:
            date_ranges = get_date_ranges(target_date=target_date, query=query)
            
        # Replace date placeholders with actual dates
        modified_query = query.replace("date_trunc_y(date_sub_y(current_date, 1))", f"'{date_ranges['last_year_start']}'")
        modified_query = modified_query.replace("current_date", f"'{date_ranges['this_year_end']}'")
        modified_query = modified_query.replace("last_year_start", f"'{date_ranges['last_year_start']}'")
        
        # Handle fiscal year variables
        if 'this_fiscal_year_start' in date_ranges:
            modified_query = modified_query.replace("this_fiscal_year_start", f"'{date_ranges['this_fiscal_year_start']}'")
        if 'this_fiscal_year_end' in date_ranges:
            modified_query = modified_query.replace("this_fiscal_year_end", f"'{date_ranges['this_fiscal_year_end']}'")
        if 'last_fiscal_year_start' in date_ranges:
            modified_query = modified_query.replace("last_fiscal_year_start", f"'{date_ranges['last_fiscal_year_start']}'")
        if 'last_fiscal_year_end' in date_ranges:
            modified_query = modified_query.replace("last_fiscal_year_end", f"'{date_ranges['last_fiscal_year_end']}'")
        
        # Check for hardcoded date patterns in the query and replace them
        this_year = datetime.strptime(date_ranges['this_year_end'], '%Y-%m-%d').year
        last_year = this_year - 1
        
        # Use regex to find hardcoded dates
        this_year_pattern = re.compile(f"'{this_year}-\\d{{2}}-\\d{{2}}'")
        last_year_pattern = re.compile(f"'{last_year}-\\d{{2}}-\\d{{2}}'")
        
        # Find all hardcoded dates for this year and last year
        this_year_dates = this_year_pattern.findall(modified_query)
        last_year_dates = last_year_pattern.findall(modified_query)
        
        # Replace the latest this_year date with the actual max date
        if this_year_dates:
            latest_this_year_date = max(this_year_dates)
            modified_query = modified_query.replace(latest_this_year_date, f"'{date_ranges['this_year_end']}'")
            logger.info(f"Replaced hardcoded this year date {latest_this_year_date} with {date_ranges['this_year_end']}")
        
        # Replace the latest last_year date with the corresponding last year date
        if last_year_dates:
            # Filter out January 1st dates as these should remain as start dates
            jan_first = f"'{last_year}-01-01'"
            non_jan_first_dates = [date for date in last_year_dates if date != jan_first]
            
            if non_jan_first_dates:
                latest_last_year_date = max(non_jan_first_dates)
                modified_query = modified_query.replace(latest_last_year_date, f"'{date_ranges['last_year_end']}'")
                logger.info(f"Replaced hardcoded last year date {latest_last_year_date} with {date_ranges['last_year_end']}")
            else:
                logger.info(f"No non-January 1st last year dates to replace. Keeping {jan_first} as the start date.")
        
        # Modify query to filter by district if provided
        if district and district != '0':
            # Add district filter to the query
            if 'WHERE' in modified_query:
                modified_query = modified_query.replace('WHERE', f'WHERE supervisor_district = \'{district}\' AND')
            else:
                modified_query = modified_query.replace('GROUP BY', f'WHERE supervisor_district = \'{district}\' GROUP BY')
            logger.info(f"Added district filter for district {district}")
        
        logger.info(f"Modified trend query: {modified_query}")
        
        # Execute the query
        context_variables = {}
        result = set_dataset(context_variables, endpoint=endpoint, query=modified_query)
        
        if result.get('status') == 'success' and 'dataset' in context_variables:
            df = context_variables['dataset']
            df['date'] = pd.to_datetime(df['date'])
            
            # Get the last data date from the trend data
            last_data_date = df['date'].max()
            if pd.notnull(last_data_date):
                last_data_date_str = last_data_date.strftime('%Y-%m-%d')
                logger.info(f"Found last data date from trend data: {last_data_date_str}")
                
                # Ensure last_data_date is not later than yesterday
                yesterday = datetime.now() - timedelta(days=1)
                yesterday_str = yesterday.strftime('%Y-%m-%d')
                if last_data_date_str > yesterday_str:
                    logger.info(f"Capping last data date from trend data to yesterday: {last_data_date_str} -> {yesterday_str}")
                    last_data_date_str = yesterday_str
                    last_data_date = yesterday
                
                # Update date ranges if needed
                if last_data_date_str < date_ranges['this_year_end']:
                    logger.info(f"Updating date ranges to use last data date: {last_data_date_str}")
                    
                    # IMPORTANT FIX: For monthly data, we need to find the actual last data date
                    # Check if this is a monthly dataset (date_trunc_ym in the query)
                    is_monthly = 'date_trunc_ym' in query
                    if is_monthly:
                        # For monthly data, we need to find the actual last data date from the dataset
                        # This is to ensure we're using the actual last date, not just the first day of the month
                        try:
                            # Try to get the actual last data date from the endpoint
                            date_col = None
                            for possible_col in ['date_issued', 'arrest_date', 'received_datetime', 'date']:
                                if possible_col in endpoint:
                                    date_col = possible_col
                                    break
                            
                            if date_col:
                                actual_last_date_query = f"SELECT max({date_col}) as last_data_date"
                                actual_date_context = {}
                                actual_date_result = set_dataset(actual_date_context, endpoint=endpoint, query=actual_last_date_query)
                                
                                if actual_date_result.get('status') == 'success' and 'dataset' in actual_date_context and not actual_date_context['dataset'].empty:
                                    actual_last_date = pd.to_datetime(actual_date_context['dataset']['last_data_date'].iloc[0])
                                    if pd.notnull(actual_last_date):
                                        actual_last_date_str = actual_last_date.strftime('%Y-%m-%d')
                                        logger.info(f"Found actual last data date from endpoint: {actual_last_date_str}")
                                        
                                        # Use this actual date instead of the first day of the month
                                        last_data_date_str = actual_last_date_str
                                        last_data_date = actual_last_date
                        except Exception as e:
                            logger.warning(f"Error getting actual last data date: {str(e)}")
                            logger.warning("Falling back to trend data date")
                    
                    date_ranges.update({
                        'this_year_end': last_data_date_str,
                        'last_year_end': last_data_date.replace(year=last_data_date.year-1).strftime('%Y-%m-%d'),
                        'last_data_date': last_data_date_str
                    })
                    # Ensure last_year_start is always January 1st of the previous year
                    date_ranges['last_year_start'] = f"{last_data_date.year-1}-01-01"
                    logger.info(f"Ensuring last_year_start is January 1st: {date_ranges['last_year_start']}")
            
            # Sort by date and convert Timestamp keys to string dates
            trend_data = {
                date.strftime('%Y-%m-%d'): value 
                for date, value in df.sort_values('date').set_index('date')['value'].items()
            }
            
            logger.info(f"Processed {len(trend_data)} trend data points" + (f" for district {district}" if district else ""))
            
            # If date ranges were updated, re-execute the trend query with the updated date ranges
            if last_data_date_str and last_data_date_str < date_ranges.get('original_this_year_end', date_ranges['this_year_end']):
                logger.info(f"Re-executing trend query with updated date ranges: {last_data_date_str}")
                
                # Create updated query with the corrected date ranges
                updated_query = query.replace("date_trunc_y(date_sub_y(current_date, 1))", f"'{date_ranges['last_year_start']}'")
                updated_query = updated_query.replace("current_date", f"'{last_data_date_str}'")
                updated_query = updated_query.replace("last_year_start", f"'{date_ranges['last_year_start']}'")
                
                # Execute the updated query
                updated_result = set_dataset({}, endpoint=endpoint, query=updated_query)
                
                if updated_result.get('status') == 'success' and 'dataset' in updated_result and not updated_result['dataset'].empty:
                    updated_df = updated_result['dataset']
                    updated_df['date'] = pd.to_datetime(updated_df['date'])
                    
                    # Update trend data with the corrected results
                    updated_trend_data = {
                        date.strftime('%Y-%m-%d'): value 
                        for date, value in updated_df.sort_values('date').set_index('date')['value'].items()
                    }
                    
                    logger.info(f"Updated trend data with {len(updated_trend_data)} data points using corrected date ranges")
                    
                    return {
                        'trend_data': updated_trend_data,
                        'last_updated': updated_df['date'].max().strftime('%Y-%m-%d'),
                        'original_query': query,
                        'executed_query': updated_query
                    }
            
            return {
                'trend_data': trend_data,
                'last_updated': df['date'].max().strftime('%Y-%m-%d'),
                'original_query': query,
                'executed_query': modified_query
            }
            
        else:
            logger.error("YTD trend query failed or no data returned")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
            logger.error(f"Query URL: {result.get('queryURL')}")
            
    except Exception as e:
        logger.error(f"Error executing YTD trend query: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

def process_ytd_trend_query_optimized(query, endpoint, date_ranges=None, target_date=None, query_name=None):
    """Process a YTD trend query to get historical daily counts for all districts in one query."""
    try:
        logger.info(f"Processing optimized YTD trend query for {query_name}")
        
        # Get date ranges if not provided
        if not date_ranges:
            date_ranges = get_date_ranges(target_date=target_date, query=query)
            
        # Replace date placeholders with actual dates
        modified_query = query.replace("date_trunc_y(date_sub_y(current_date, 1))", f"'{date_ranges['last_year_start']}'")
        modified_query = modified_query.replace("current_date", f"'{date_ranges['this_year_end']}'")
        modified_query = modified_query.replace("last_year_start", f"'{date_ranges['last_year_start']}'")
        
        # Handle fiscal year variables
        if 'this_fiscal_year_start' in date_ranges:
            modified_query = modified_query.replace("this_fiscal_year_start", f"'{date_ranges['this_fiscal_year_start']}'")
        if 'this_fiscal_year_end' in date_ranges:
            modified_query = modified_query.replace("this_fiscal_year_end", f"'{date_ranges['this_fiscal_year_end']}'")
        if 'last_fiscal_year_start' in date_ranges:
            modified_query = modified_query.replace("last_fiscal_year_start", f"'{date_ranges['last_fiscal_year_start']}'")
        if 'last_fiscal_year_end' in date_ranges:
            modified_query = modified_query.replace("last_fiscal_year_end", f"'{date_ranges['last_fiscal_year_end']}'")
        
        # Check for hardcoded date patterns in the query and replace them
        this_year = datetime.strptime(date_ranges['this_year_end'], '%Y-%m-%d').year
        last_year = this_year - 1
        
        # Use regex to find hardcoded dates
        this_year_pattern = re.compile(f"'{this_year}-\\d{{2}}-\\d{{2}}'")
        last_year_pattern = re.compile(f"'{last_year}-\\d{{2}}-\\d{{2}}'")
        
        # Find all hardcoded dates for this year and last year
        this_year_dates = this_year_pattern.findall(modified_query)
        last_year_dates = last_year_pattern.findall(modified_query)
        
        # Replace the latest this_year date with the actual max date
        if this_year_dates:
            latest_this_year_date = max(this_year_dates)
            modified_query = modified_query.replace(latest_this_year_date, f"'{date_ranges['this_year_end']}'")
            logger.info(f"Replaced hardcoded this year date {latest_this_year_date} with {date_ranges['this_year_end']}")
        
        # Replace the latest last_year date with the corresponding last year date
        if last_year_dates:
            # Filter out January 1st dates as these should remain as start dates
            jan_first = f"'{last_year}-01-01'"
            non_jan_first_dates = [date for date in last_year_dates if date != jan_first]
            
            if non_jan_first_dates:
                latest_last_year_date = max(non_jan_first_dates)
                modified_query = modified_query.replace(latest_last_year_date, f"'{date_ranges['last_year_end']}'")
                logger.info(f"Replaced hardcoded last year date {latest_last_year_date} with {date_ranges['last_year_end']}")
            else:
                logger.info(f"No non-January 1st last year dates to replace. Keeping {jan_first} as the start date.")
        
        logger.info(f"Modified optimized trend query: {modified_query}")
        
        # Execute the query
        context_variables = {}
        result = set_dataset(context_variables, endpoint=endpoint, query=modified_query)
        
        if result.get('status') == 'success' and 'dataset' in context_variables:
            df = context_variables['dataset']
            df['date'] = pd.to_datetime(df['date'])
            
            # Get the last data date from the trend data
            last_data_date = df['date'].max()
            if pd.notnull(last_data_date):
                last_data_date_str = last_data_date.strftime('%Y-%m-%d')
                logger.info(f"Found last data date from trend data: {last_data_date_str}")
                
                # Ensure last_data_date is not later than yesterday
                yesterday = datetime.now() - timedelta(days=1)
                yesterday_str = yesterday.strftime('%Y-%m-%d')
                if last_data_date_str > yesterday_str:
                    logger.info(f"Capping last data date from trend data to yesterday: {last_data_date_str} -> {yesterday_str}")
                    last_data_date_str = yesterday_str
                    last_data_date = yesterday
                
                # Update date ranges if needed
                if last_data_date_str < date_ranges['this_year_end']:
                    logger.info(f"Updating date ranges to use last data date: {last_data_date_str}")
                    
                    # IMPORTANT FIX: For monthly data, we need to find the actual last data date
                    # Check if this is a monthly dataset (date_trunc_ym in the query)
                    is_monthly = 'date_trunc_ym' in query
                    if is_monthly:
                        # For monthly data, we need to find the actual last data date from the dataset
                        # This is to ensure we're using the actual last date, not just the first day of the month
                        try:
                            # Try to get the actual last data date from the endpoint
                            date_col = None
                            for possible_col in ['date_issued', 'arrest_date', 'received_datetime', 'date']:
                                if possible_col in endpoint:
                                    date_col = possible_col
                                    break
                            
                            if date_col:
                                actual_last_date_query = f"SELECT max({date_col}) as last_data_date"
                                actual_date_context = {}
                                actual_date_result = set_dataset(actual_date_context, endpoint=endpoint, query=actual_last_date_query)
                                
                                if actual_date_result.get('status') == 'success' and 'dataset' in actual_date_context and not actual_date_context['dataset'].empty:
                                    actual_last_date = pd.to_datetime(actual_date_context['dataset']['last_data_date'].iloc[0])
                                    if pd.notnull(actual_last_date):
                                        actual_last_date_str = actual_last_date.strftime('%Y-%m-%d')
                                        logger.info(f"Found actual last data date from endpoint: {actual_last_date_str}")
                                        
                                        # Use this actual date instead of the first day of the month
                                        last_data_date_str = actual_last_date_str
                                        last_data_date = actual_last_date
                        except Exception as e:
                            logger.warning(f"Error getting actual last data date: {str(e)}")
                            logger.warning("Falling back to trend data date")
                    
                    date_ranges.update({
                        'this_year_end': last_data_date_str,
                        'last_year_end': last_data_date.replace(year=last_data_date.year-1).strftime('%Y-%m-%d'),
                        'last_data_date': last_data_date_str
                    })
                    # Ensure last_year_start is always January 1st of the previous year
                    date_ranges['last_year_start'] = f"{last_data_date.year-1}-01-01"
                    logger.info(f"Ensuring last_year_start is January 1st: {date_ranges['last_year_start']}")
            
            # Check if we have district data
            has_district = 'supervisor_district' in df.columns
            
            if has_district:
                # Process trend data for all districts
                trend_data_all = {}
                
                # Citywide trend data (all districts combined)
                trend_data_all['0'] = {
                    date.strftime('%Y-%m-%d'): value 
                    for date, value in df.sort_values('date').set_index('date')['value'].items()
                }
                
                # District-specific trend data
                for district in range(1, 12):
                    district_df = df[df['supervisor_district'] == str(district)]
                    if not district_df.empty:
                        district_df['date'] = pd.to_datetime(district_df['date'])
                        trend_data_all[str(district)] = {
                            date.strftime('%Y-%m-%d'): value 
                            for date, value in district_df.sort_values('date').set_index('date')['value'].items()
                        }
                
                logger.info(f"Processed {len(trend_data_all)} district trend datasets" + 
                          f" with {len(trend_data_all.get('0', {}))} data points for citywide")
                
                return {
                    'trend_data_all': trend_data_all,
                    'last_updated': df['date'].max().strftime('%Y-%m-%d'),
                    'original_query': query,
                    'executed_query': modified_query
                }
            else:
                # For non-district queries, just return citywide trend data
                trend_data = {
                    date.strftime('%Y-%m-%d'): value 
                    for date, value in df.sort_values('date').set_index('date')['value'].items()
                }
                
                logger.info(f"Processed {len(trend_data)} trend data points for non-district query")
                
                return {
                    'trend_data_all': {'0': trend_data},
                    'last_updated': df['date'].max().strftime('%Y-%m-%d'),
                    'original_query': query,
                    'executed_query': modified_query
                }
            
        else:
            logger.error("Optimized YTD trend query failed or no data returned")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
            logger.error(f"Query URL: {result.get('queryURL')}")
            
    except Exception as e:
        logger.error(f"Error executing optimized YTD trend query: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

def generate_ytd_metrics(queries_data, output_dir, target_date=None):
    """Generate YTD metrics files for each district."""
    
    # Initialize the metrics structure
    metrics = {
        "districts": {
            "0": {
                "name": "Citywide",
                "categories": []
            }
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "data_as_of": None,  # Will be updated with actual data date
            "next_update": None  # Will be updated after processing
        }
    }
    
    # Process each top-level category (safety, economy, etc.)
    for top_category_name, top_category_data in queries_data.items():
        # Initialize category metrics for the top-level category
        top_category_metrics = {
            "category": top_category_name.title(),
            "metrics": []
        }
        
        # Process each subcategory
        for subcategory_name, subcategory_data in top_category_data.items():
            # Check if this is a valid subcategory with queries
            if isinstance(subcategory_data, dict) and 'queries' in subcategory_data:
                # Get the endpoint from the subcategory if it exists
                endpoint = subcategory_data.get('endpoint', None)
                
                # Format the endpoint if it exists
                if endpoint and not endpoint.startswith('http'):
                    endpoint = f"https://data.sfgov.org/resource/{endpoint}"
                if endpoint and not endpoint.endswith('.json'):
                    endpoint = f"{endpoint}.json"
                
                # Process each query in the subcategory
                for query_name, query_data in subcategory_data['queries'].items():
                    # Extract queries and metadata
                    if isinstance(query_data, str):
                        metric_query = query_data
                        ytd_query = None
                        # Get date ranges for definition
                        date_ranges_for_def = get_date_ranges(target_date=target_date, query=metric_query)
                        date_info = format_date_ranges_readable(date_ranges_for_def)
                        
                        metadata = {
                            "summary": "",
                            "definition": date_info,
                            "data_sf_url": "",
                            "ytd_query": ""
                        }
                        query_endpoint = endpoint
                    else:
                        metric_query = query_data.get('metric_query', '')
                        ytd_query = query_data.get('ytd_query', '')
                        # Get date ranges for definition
                        date_ranges_for_def = get_date_ranges(target_date=target_date, query=metric_query)
                        date_info = format_date_ranges_readable(date_ranges_for_def)
                        
                        # Add date info to existing definition
                        existing_definition = query_data.get('definition', '')
                        if existing_definition:
                            definition_with_dates = f"{existing_definition} {date_info}"
                        else:
                            definition_with_dates = date_info
                        
                        metadata = {
                            "summary": query_data.get('summary', ''),
                            "definition": definition_with_dates,
                            "data_sf_url": query_data.get('data_sf_url', ''),
                            "ytd_query": query_data.get('ytd_query', '')
                        }
                        # Check if this query has its own endpoint
                        if 'endpoint' in query_data:
                            query_endpoint = query_data['endpoint']
                            if query_endpoint and not query_endpoint.startswith('http'):
                                query_endpoint = f"https://data.sfgov.org/resource/{query_endpoint}"
                            if query_endpoint and not query_endpoint.endswith('.json'):
                                query_endpoint = f"{query_endpoint}.json"
                            logger.info(f"Using query-specific endpoint for {query_name}: {query_endpoint}")
                        else:
                            query_endpoint = endpoint
                    
                    # Skip this query if we don't have an endpoint
                    if not query_endpoint:
                        logger.warning(f"Skipping query {query_name} because no endpoint is defined")
                        continue
                    
                    # Get initial date ranges using yesterday as target
                    initial_date_ranges = get_date_ranges(target_date=target_date, query=metric_query)
                    
                    # Create a copy of initial date ranges for this metric
                    date_ranges = initial_date_ranges.copy()
                    trend_data = None
                    max_date = None  # Initialize max_date
                    
                    # First, process YTD trend query to get the actual last data date
                    if ytd_query:
                        trend_data = process_ytd_trend_query_optimized(ytd_query, query_endpoint, date_ranges=date_ranges, query_name=query_name)
                        if trend_data and 'last_updated' in trend_data:
                            max_date = trend_data['last_updated']
                            logger.info(f"Found max date from YTD trend query: {max_date}")
                            
                            # Update the most_recent_data_date in the database if we have a metric ID
                            if isinstance(query_data, dict) and "id" in query_data and isinstance(query_data["id"], int):
                                metric_id = query_data["id"]
                                update_metric_most_recent_data_date(metric_id, max_date)
                            
                            # Convert max_date to datetime for manipulation
                            max_date_dt = datetime.strptime(max_date, '%Y-%m-%d')
                            today = datetime.now()
                            yesterday = today - timedelta(days=1)
                            yesterday_str = yesterday.strftime('%Y-%m-%d')
                            
                            # Cap max_date to yesterday
                            if max_date > yesterday_str:
                                logger.info(f"Capping max date to yesterday: {max_date} -> {yesterday_str}")
                                max_date = yesterday_str
                                max_date_dt = yesterday
                                
                                # Update the database with the capped date as well
                                if isinstance(query_data, dict) and "id" in query_data and isinstance(query_data["id"], int):
                                    metric_id = query_data["id"]
                                    update_metric_most_recent_data_date(metric_id, max_date)
                            
                            # If max_date is in the future, set it to last day of previous month
                            if max_date_dt > today:
                                logger.info(f"Max date {max_date} is in the future, adjusting to last day of previous month")
                                if today.month == 1:
                                    # If we're in January, use December of previous year
                                    end_year = today.year - 1
                                    end_month = 12
                                else:
                                    end_year = today.year
                                    end_month = today.month - 1
                                    
                                # Calculate last day of the month
                                if end_month == 12:
                                    last_day = 31
                                elif end_month in [4, 6, 9, 11]:
                                    last_day = 30
                                elif end_month == 2:
                                    # Handle leap years
                                    if end_year % 4 == 0 and (end_year % 100 != 0 or end_year % 400 == 0):
                                        last_day = 29
                                    else:
                                        last_day = 28
                                else:
                                    last_day = 31
                                    
                                max_date = f"{end_year}-{end_month:02d}-{last_day}"
                                logger.info(f"Adjusted max date to last day of previous month: {max_date}")
                                
                                # Update the database with the adjusted date
                                if isinstance(query_data, dict) and "id" in query_data and isinstance(query_data["id"], int):
                                    metric_id = query_data["id"]
                                    update_metric_most_recent_data_date(metric_id, max_date)
                            
                            # Update date ranges with the max date
                            date_ranges['this_year_end'] = max_date
                            max_date_dt = datetime.strptime(max_date, '%Y-%m-%d')
                            
                            # For last year's end date, use the same day-of-month but in previous year
                            last_year_end = max_date_dt.replace(year=max_date_dt.year-1)
                            date_ranges['last_year_end'] = last_year_end.strftime('%Y-%m-%d')
                            
                            date_ranges['last_data_date'] = max_date
                            # Ensure last_year_start is always January 1st of the previous year
                            date_ranges['last_year_start'] = f"{max_date_dt.year-1}-01-01"
                            logger.info(f"Updated date ranges with max date: {date_ranges}")
                    
                    # Calculate monthly and MTD ranges using the updated date ranges
                    monthly_ranges = None
                    mtd_ranges = None
                    if max_date:
                        monthly_ranges = get_monthly_date_ranges(max_date)
                        mtd_ranges = get_mtd_date_ranges(max_date)
                        logger.info(f"Calculated monthly ranges: {monthly_ranges}")
                        logger.info(f"Calculated MTD ranges: {mtd_ranges}")
                    else:
                        # If no max_date from YTD query, use the end date from date_ranges
                        if 'this_year_end' in date_ranges:
                            max_date = date_ranges['this_year_end']
                            monthly_ranges = get_monthly_date_ranges(max_date)
                            mtd_ranges = get_mtd_date_ranges(max_date)
                            logger.info(f"Calculated monthly ranges using date_ranges: {monthly_ranges}")
                            logger.info(f"Calculated MTD ranges using date_ranges: {mtd_ranges}")
                    
                    # Process metric query with the adjusted date ranges (using optimized version)
                    query_results = process_query_for_district_optimized(metric_query, query_endpoint, date_ranges, query_name=query_name, monthly_ranges=monthly_ranges, mtd_ranges=mtd_ranges)
                    if query_results:
                        results = query_results['results']
                        queries = query_results['queries']
                        
                        # Extract just the endpoint identifier (remove URL and .json extension)
                        endpoint_identifier = query_endpoint
                        if endpoint_identifier:
                            # Remove the base URL if present
                            if endpoint_identifier.startswith('https://data.sfgov.org/resource/'):
                                endpoint_identifier = endpoint_identifier.replace('https://data.sfgov.org/resource/', '')
                            # Remove .json extension if present
                            if endpoint_identifier.endswith('.json'):
                                endpoint_identifier = endpoint_identifier[:-5]
                        
                        # Create metric object with metadata
                        metric_base = {
                            "name": query_name.replace(" YTD", ""),
                            "id": query_name.lower().replace(" ", "_").replace("-", "_").replace("_ytd", "") + "_ytd",
                            "metadata": metadata,
                            "queries": {
                                "metric_query": queries['original_query'],
                                "executed_query": queries['executed_query']
                            },
                            "endpoint": endpoint_identifier
                        }
                        
                        # Add MTD and monthly queries if available
                                # Monthly query references removed - using MTD instead
                        if 'mtd_query' in queries:
                            metric_base["queries"]["mtd_query"] = queries['mtd_query']
                        if 'executed_mtd_query' in queries:
                            metric_base["queries"]["executed_mtd_query"] = queries['executed_mtd_query']
                        
                        # Add location and category fields if available in the enhanced query data
                        if isinstance(query_data, dict):
                            if "location_fields" in query_data:
                                metric_base["location_fields"] = query_data.get("location_fields", [])
                            if "category_fields" in query_data:
                                metric_base["category_fields"] = query_data.get("category_fields", [])
                            # Also add the numeric ID if available
                            if "id" in query_data and isinstance(query_data["id"], int):
                                metric_base["numeric_id"] = query_data["id"]
                            # Add show flag based on show_on_dash setting
                            if "show_on_dash" in query_data:
                                metric_base["show"] = query_data["show_on_dash"]
                            else:
                                metric_base["show"] = True  # Default to showing if not specified
                            # Add greendirection field
                            if "greendirection" in query_data:
                                metric_base["greendirection"] = query_data["greendirection"]
                            else:
                                metric_base["greendirection"] = 'up'  # Default to up if not specified
                        
                        # Get the last data date from metric results if available
                        metric_last_data_date = None
                        if '0' in results and results['0'].get('lastDataDate'):
                            metric_last_data_date = results['0']['lastDataDate']
                        
                        # Add trend data if it was processed (citywide)
                        if trend_data and 'trend_data_all' in trend_data:
                            metric_base["trend_data"] = trend_data["trend_data_all"].get('0', {})
                            # Use metric's last data date if available (for monthly/truncated data),
                            # otherwise use trend's last updated date
                            metric_base["trend_last_updated"] = metric_last_data_date or trend_data["last_updated"]
                            metric_base["queries"]["ytd_query"] = trend_data["original_query"]
                            metric_base["queries"]["executed_ytd_query"] = trend_data["executed_query"]
                        
                        # Update metadata with the most recent data date
                        if metrics['metadata']['data_as_of'] is None or (metric_last_data_date and metric_last_data_date > metrics['metadata']['data_as_of']):
                            metrics['metadata']['data_as_of'] = metric_last_data_date
                        
                        # Ensure data_as_of is never later than yesterday
                        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        if metrics['metadata']['data_as_of'] and metrics['metadata']['data_as_of'] > yesterday:
                            logger.info(f"Capping data_as_of date to yesterday: {metrics['metadata']['data_as_of']} -> {yesterday}")
                            metrics['metadata']['data_as_of'] = yesterday
                        
                        # Update metadata definition to use actual date ranges from executed queries
                        actual_date_info = format_date_ranges_readable(date_ranges)
                        
                        # Update the definition in the metadata
                        if 'definition' in metric_base['metadata']:
                            # Replace the date info in the existing definition
                            existing_definition = metric_base['metadata']['definition']
                            # Remove any existing date comparison text
                            import re
                            # Remove both old and new date comparison patterns
                            # Old pattern: Comparing YYYY-MM-DD to YYYY-MM-DD vs YYYY-MM-DD to YYYY-MM-DD
                            # New pattern: Comparing (Month DD-Month DD YYYY) vs (Month DD-Month DD YYYY)
                            cleaned_definition = re.sub(r'Comparing \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2} vs \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}', '', existing_definition)
                            cleaned_definition = re.sub(r'Comparing \([^)]+ to [^)]+\) vs \([^)]+ to [^)]+\)', '', cleaned_definition)
                            cleaned_definition = re.sub(r'Comparing \([A-Za-z]{3} \d{1,2}-[A-Za-z]{3} \d{1,2} \d{4}\) vs \([A-Za-z]{3} \d{1,2}-[A-Za-z]{3} \d{1,2} \d{4}\)', '', cleaned_definition)
                            cleaned_definition = cleaned_definition.strip()
                            # Add the new date info
                            if cleaned_definition:
                                metric_base['metadata']['definition'] = f"{cleaned_definition} {actual_date_info}"
                            else:
                                metric_base['metadata']['definition'] = actual_date_info
                        
                        # Add citywide metric
                        if '0' in results:
                            citywide_metric = metric_base.copy()
                            citywide_metric.update({
                                "lastYear": results['0']['lastYear'],
                                "thisYear": results['0']['thisYear'],
                                "lastDataDate": metric_last_data_date or results['0'].get('lastDataDate')
                            })
                            
                            # Add monthly data if available
                            if 'thisMonth' in results['0']:
                                citywide_metric['thisMonth'] = results['0']['thisMonth']
                            if 'lastMonth' in results['0']:
                                citywide_metric['lastMonth'] = results['0']['lastMonth']
                            
                            # Add MTD data if available
                            if 'thisMtd' in results['0']:
                                citywide_metric['thisMtd'] = results['0']['thisMtd']
                            if 'lastMtd' in results['0']:
                                citywide_metric['lastMtd'] = results['0']['lastMtd']
                            
                            # Add date range information for headers
                            # Monthly date ranges removed - using MTD instead
                            
                            if mtd_ranges:
                                citywide_metric['mtd_date_ranges'] = {
                                    'this_mtd': f"{mtd_ranges['this_mtd_start']} to {mtd_ranges['this_mtd_end']}",
                                    'last_mtd': f"{mtd_ranges['last_mtd_start']} to {mtd_ranges['last_mtd_end']}"
                                }
                            
                            # Add MTD query information if available
                            if query_results and 'queries' in query_results:
                                queries_section = query_results['queries']
                                if 'mtd_query' in queries_section:
                                    citywide_metric['queries']['mtd_query'] = queries_section['mtd_query']
                                    citywide_metric['queries']['executed_mtd_query'] = queries_section['executed_mtd_query']
                                    logger.info(f"Added MTD query to citywide metric for {query_name}")
                                else:
                                    logger.warning(f"MTD query not found in queries section for {query_name}. Available keys: {list(queries_section.keys())}")
                                
                                # Monthly query references removed - using MTD instead
                            else:
                                logger.warning(f"No queries section found in query_results for {query_name}")
                            
                            top_category_metrics['metrics'].append(citywide_metric)

                            # Generate and store YTD trend chart for citywide (district 0)
                            if trend_data:
                                try:
                                    logger.info(f"Creating YTD trend chart for {query_name} - Citywide")

                                    chart_metadata = {
                                        "chart_title": f"{query_name} - YTD Trend",
                                        "y_axis_label": "Count" if "COUNT" in trend_data["original_query"].upper() else "Value",
                                        "description": f"Year-to-date trend for {query_name} in Citywide",
                                        "period_type": "ytd",
                                        "district": "0",
                                        "object_id": str(query_data.get("id", "unknown")),
                                        "object_type": "metric",
                                        "object_name": f"{query_name} - YTD Trend",
                                        "category": top_category_name.title()
                                    }

                                    chart_html = generate_ytd_trend_chart(
                                        trend_data=trend_data["trend_data_all"].get("0", {}),
                                        metadata=chart_metadata,
                                        district="0",
                                        return_html=True,
                                        store_in_db=True
                                    )

                                    if chart_html:
                                        logger.info(f"Successfully created YTD trend chart for {query_name} - Citywide")
                                    else:
                                        logger.warning(f"Failed to create YTD trend chart for {query_name} - Citywide")
                                except Exception as e:
                                    logger.error(f"Error creating YTD trend chart for {query_name} - Citywide: {str(e)}")
                                    logger.error(traceback.format_exc())
                        
                        # Add district metrics
                        for district_num in range(1, 12):
                            district_str = str(district_num)
                            if district_str in results:
                                # Initialize district if not exists BEFORE accessing it
                                if district_str not in metrics['districts']:
                                    metrics['districts'][district_str] = {
                                        "name": f"District {district_str}",
                                        "categories": []
                                    }
                                
                                district_data = metrics['districts'][district_str]
                                district_metric = metric_base.copy()
                                district_metric.update({
                                    "lastYear": results[district_str]['lastYear'],
                                    "thisYear": results[district_str]['thisYear'],
                                    "lastDataDate": metric_last_data_date or results[district_str].get('lastDataDate')
                                })
                                
                                # Add monthly data if available
                                if 'thisMonth' in results[district_str]:
                                    district_metric['thisMonth'] = results[district_str]['thisMonth']
                                if 'lastMonth' in results[district_str]:
                                    district_metric['lastMonth'] = results[district_str]['lastMonth']
                                
                                # Add MTD data if available
                                if 'thisMtd' in results[district_str]:
                                    district_metric['thisMtd'] = results[district_str]['thisMtd']
                                if 'lastMtd' in results[district_str]:
                                    district_metric['lastMtd'] = results[district_str]['lastMtd']
                                
                                # Add date range information for headers
                                # Monthly date ranges removed - using MTD instead
                                
                                if mtd_ranges:
                                    district_metric['mtd_date_ranges'] = {
                                        'this_mtd': f"{mtd_ranges['this_mtd_start']} to {mtd_ranges['this_mtd_end']}",
                                        'last_mtd': f"{mtd_ranges['last_mtd_start']} to {mtd_ranges['last_mtd_end']}"
                                    }
                                
                                # Add MTD query information if available
                                if query_results and 'queries' in query_results:
                                    queries_section = query_results['queries']
                                    if 'mtd_query' in queries_section:
                                        district_metric['queries']['mtd_query'] = queries_section['mtd_query']
                                        district_metric['queries']['executed_mtd_query'] = queries_section['executed_mtd_query']
                                    
                                    # Monthly query references removed - using MTD instead
                                
                                # Process district-specific trend data (using optimized version)
                                district_trend_data = None
                                if ytd_query and trend_data and 'trend_data_all' in trend_data:
                                    logger.info(f"Using optimized trend data for district {district_str}")
                                    district_trend_data = {
                                        "trend_data": trend_data["trend_data_all"].get(district_str, {}),
                                        "last_updated": trend_data["last_updated"],
                                        "original_query": trend_data["original_query"],
                                        "executed_query": trend_data["executed_query"]
                                    }
                                
                                # Add district-specific trend data
                                if district_trend_data and district_trend_data["trend_data"]:
                                    district_metric["trend_data"] = district_trend_data["trend_data"]
                                    district_metric["trend_last_updated"] = metric_last_data_date or district_trend_data["last_updated"]
                                    district_metric["queries"]["ytd_query"] = district_trend_data["original_query"]
                                    district_metric["queries"]["executed_ytd_query"] = district_trend_data["executed_query"]
                                    
                                    # Create and store YTD trend chart for this district
                                    try:
                                        logger.info(f"Creating YTD trend chart for {query_name} - District {district_str}")
                                        
                                        # Prepare chart metadata
                                        chart_metadata = {
                                            "chart_title": f"{query_name} - YTD Trend",
                                            "y_axis_label": "Count" if "COUNT" in district_trend_data["original_query"].upper() else "Value",
                                            "description": f"Year-to-date trend for {query_name} in {'Citywide' if district_str == '0' else f'District {district_str}'}",
                                            "period_type": "ytd",
                                            "district": district_str,
                                            "object_id": str(query_data.get("id", "unknown")),  # Use object_id instead of metric_id
                                            "object_type": "metric",
                                            "object_name": f"{query_name} - YTD Trend",
                                            "category": top_category_name.title()
                                        }
                                        
                                        # Create the chart using Plotly
                                        chart_html = generate_ytd_trend_chart(
                                            trend_data=district_trend_data["trend_data"],
                                            metadata=chart_metadata,
                                            district=district_str,
                                            return_html=True,
                                            store_in_db=True
                                        )
                                        
                                        if chart_html:
                                            logger.info(f"Successfully created YTD trend chart for {query_name} - District {district_str}")
                                        else:
                                            logger.warning(f"Failed to create YTD trend chart for {query_name} - District {district_str}")
                                            
                                    except Exception as e:
                                        logger.error(f"Error creating YTD trend chart for {query_name} - District {district_str}: {str(e)}")
                                        logger.error(traceback.format_exc())
                                
                                # Find or create category for this district
                                district_category = next(
                                    (cat for cat in metrics['districts'][district_str]['categories'] 
                                     if cat['category'] == top_category_name.title()),
                                    None
                                )
                                if district_category is None:
                                    district_category = {
                                        "category": top_category_name.title(),
                                        "metrics": []
                                    }
                                    metrics['districts'][district_str]['categories'].append(district_category)
                                district_category['metrics'].append(district_metric)
        
        if top_category_metrics['metrics']:
            metrics['districts']['0']['categories'].append(top_category_metrics)
    
    # Calculate the next update time (either 5 AM or 11 AM)
    now = datetime.now()
    next_5am = now.replace(hour=5, minute=0, second=0, microsecond=0)
    next_11am = now.replace(hour=11, minute=0, second=0, microsecond=0)
    
    # If we're past 11 AM, set targets to next day
    if now.hour >= 11:
        next_5am += timedelta(days=1)
        next_11am += timedelta(days=1)
    # If we're past 5 AM but before 11 AM, only adjust the 5 AM target
    elif now.hour >= 5:
        next_5am += timedelta(days=1)
    
    # Find the next closest update time
    next_update = min(next_5am, next_11am)
    metrics['metadata']['next_update'] = next_update.isoformat()
    
    # Create output directories
    dashboard_dir = output_dir  # Changed: Use output_dir directly as dashboard_dir
    history_dir = os.path.join(output_dir, 'history')
    os.makedirs(dashboard_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    # Save individual district files
    for district_num in range(12):  # 0-11 for citywide and districts 1-11
        district_str = str(district_num)
        if district_str in metrics['districts']:
            # Create district subfolder
            district_dir = os.path.join(dashboard_dir, district_str)
            os.makedirs(district_dir, exist_ok=True)
            
            # Create a copy of district data without trend data for top_level.json
            district_data = {
                "metadata": metrics['metadata'],
                "name": metrics['districts'][district_str]['name'],
                "categories": []
            }
            
            # Process each category
            for category in metrics['districts'][district_str]['categories']:
                category_copy = {
                    "category": category['category'],
                    "metrics": []
                }
                
                # Process each metric in the category
                for metric in category['metrics']:
                    # Create a copy of the metric without trend data
                    metric_copy = {
                        "name": metric['name'],
                        "id": metric['id'],
                        "lastYear": metric['lastYear'],
                        "thisYear": metric['thisYear'],
                        "lastDataDate": metric['lastDataDate'],
                        "endpoint": metric.get('endpoint', ''),
                        "queries": metric['queries'],
                        "metadata": metric.get('metadata', {})
                    }
                    
                    # Add greendirection to metadata
                    if 'greendirection' in metric:
                        metric_copy['metadata']['greendirection'] = metric['greendirection']
                    
                    # Add monthly data if available
                    if 'thisMonth' in metric:
                        metric_copy['thisMonth'] = metric['thisMonth']
                    if 'lastMonth' in metric:
                        metric_copy['lastMonth'] = metric['lastMonth']
                    
                    # Add MTD data if available
                    if 'thisMtd' in metric:
                        metric_copy['thisMtd'] = metric['thisMtd']
                    if 'lastMtd' in metric:
                        metric_copy['lastMtd'] = metric['lastMtd']
                    
                    # Add MTD date ranges if available
                    if 'mtd_date_ranges' in metric:
                        metric_copy['mtd_date_ranges'] = metric['mtd_date_ranges']
                    
                    # Add numeric_id if it exists
                    if 'numeric_id' in metric:
                        metric_copy['numeric_id'] = metric['numeric_id']
                    
                    # Add location_fields and category_fields if they exist
                    if 'location_fields' in metric:
                        metric_copy['location_fields'] = metric['location_fields']
                    if 'category_fields' in metric:
                        metric_copy['category_fields'] = metric['category_fields']
                    
                    # Add show flag if it exists
                    if 'show' in metric:
                        metric_copy['show'] = metric['show']
                    
                    category_copy['metrics'].append(metric_copy)
                    
                    # Save individual metric file with full data
                    file_id = metric.get('numeric_id', metric['id'])
                    metric_file = os.path.join(district_dir, f"{file_id}.json")
                    metric_data = {
                        "category": category['category'],
                        "metric_name": metric['name'],
                        "metric_id": metric['id'],
                        "lastYear": metric['lastYear'],
                        "thisYear": metric['thisYear'],
                        "lastDataDate": metric['lastDataDate'],
                        "endpoint": metric.get('endpoint', ''),
                        "queries": metric['queries'],
                        "metadata": metric.get('metadata', {})
                    }
                    
                    # Add monthly data if available
                    if 'thisMonth' in metric:
                        metric_data['thisMonth'] = metric['thisMonth']
                    if 'lastMonth' in metric:
                        metric_data['lastMonth'] = metric['lastMonth']
                    
                    # Add MTD data if available
                    if 'thisMtd' in metric:
                        metric_data['thisMtd'] = metric['thisMtd']
                    if 'lastMtd' in metric:
                        metric_data['lastMtd'] = metric['lastMtd']
                    
                    # Add MTD date ranges if available
                    if 'mtd_date_ranges' in metric:
                        metric_data['mtd_date_ranges'] = metric['mtd_date_ranges']
                    
                    # Add numeric_id if it exists
                    if 'numeric_id' in metric:
                        metric_data['numeric_id'] = metric['numeric_id']
                    
                    # Add location_fields and category_fields if they exist
                    if 'location_fields' in metric:
                        metric_data['location_fields'] = metric['location_fields']
                    if 'category_fields' in metric:
                        metric_data['category_fields'] = metric['category_fields']
                    
                    # Add show flag if it exists
                    if 'show' in metric:
                        metric_data['show'] = metric['show']
                    
                    # Note: trend_data is no longer added to individual metric files per user request
                    
                    # Add district breakdown for citywide metrics (district 0)
                    if district_str == '0':
                        district_breakdown = {}
                        # Look for this metric in each district's data
                        for d_num in range(1, 12):
                            d_str = str(d_num)
                            if d_str in metrics['districts']:
                                # Find matching metric in district data
                                for d_cat in metrics['districts'][d_str]['categories']:
                                    if d_cat['category'] == category['category']:
                                        for d_metric in d_cat['metrics']:
                                            if d_metric['id'] == metric['id']:
                                                district_breakdown[d_str] = {
                                                    "thisYear": d_metric['thisYear'],
                                                    "lastYear": d_metric['lastYear'],
                                                    "lastDataDate": d_metric['lastDataDate'],
                                                    "endpoint": d_metric.get('endpoint', '')
                                                }
                                                
                                                # Add monthly data to district breakdown if available
                                                if 'thisMonth' in d_metric:
                                                    district_breakdown[d_str]['thisMonth'] = d_metric['thisMonth']
                                                if 'lastMonth' in d_metric:
                                                    district_breakdown[d_str]['lastMonth'] = d_metric['lastMonth']
                                                
                                                # Add MTD data to district breakdown if available
                                                if 'thisMtd' in d_metric:
                                                    district_breakdown[d_str]['thisMtd'] = d_metric['thisMtd']
                                                if 'lastMtd' in d_metric:
                                                    district_breakdown[d_str]['lastMtd'] = d_metric['lastMtd']
                                                break
                        if district_breakdown:
                            metric_data["district_breakdown"] = district_breakdown

                    with open(metric_file, 'w', encoding='utf-8') as f:
                        clean_data = clean_nan_values(metric_data)
                        json.dump(clean_data, f, indent=2)
                    logger.info(f"Metric {file_id} (original id: {metric['id']}) saved to {metric_file}")
                
                district_data['categories'].append(category_copy)
            
            # Save top_level.json to district subfolder
            top_level_file = os.path.join(district_dir, 'top_level.json')
            
            with open(top_level_file, 'w', encoding='utf-8') as f:
                clean_data = clean_nan_values(district_data)
                json.dump(clean_data, f, indent=2)
            logger.info(f"District {district_str} top_level metrics saved to {top_level_file}")
            
            # Save to history directory with timestamp
            history_file = os.path.join(history_dir, f'district_{district_str}_{datetime.now().strftime("%Y%m%d")}.json')
            with open(history_file, 'w', encoding='utf-8') as f:
                clean_data = clean_nan_values(district_data)
                json.dump(clean_data, f, indent=2)
            logger.info(f"District {district_str} metrics history saved to {history_file}")
    
    return metrics

def create_ytd_vector_collection(metrics):
    """Create a vector collection for YTD metrics data."""
    try:
        collection_name = 'YTD'
        logger.info("Starting YTD vector collection creation")

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        
        client = OpenAI(api_key=openai_api_key)
        EMBEDDING_MODEL = "text-embedding-3-large"
        BATCH_SIZE = 100
        MAX_RETRIES = 3

        # Initialize Qdrant client
        qdrant = QdrantClient(host='localhost', port=6333)
        
        # Get sample embedding to determine vector size
        sample_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input="Sample text for vector size determination"
        )
        vector_size = len(sample_response.data[0].embedding)
        
        # Create or recreate collection
        def create_collection():
            if qdrant.collection_exists(collection_name):
                qdrant.delete_collection(collection_name)
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=vector_size
                )
            )
        
        # Attempt to create collection with retries
        for attempt in range(MAX_RETRIES):
            try:
                create_collection()
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Failed to create collection after {MAX_RETRIES} attempts")
                time.sleep(2 ** attempt)

        # Prepare data for embedding
        texts_to_embed = []
        metadata_batch = []
        
        # Process citywide metrics
        citywide_data = metrics['districts']['0']
        for category in citywide_data['categories']:
            category_name = category['category']
            for metric in category['metrics']:
                try:
                    # Validate required fields exist
                    required_fields = ['name', 'id', 'thisYear', 'lastYear', 'lastDataDate', 'queries']
                    missing_fields = [field for field in required_fields if field not in metric]
                    if missing_fields:
                        logger.warning(f"Skipping metric due to missing required fields: {missing_fields}")
                        continue
                        
                    metric_text = create_metric_text(category_name, metric, citywide_data['name'])
                    texts_to_embed.append(metric_text)
                    metadata_batch.append(create_metadata_dict(category_name, metric, '0', citywide_data['name']))
                except Exception as e:
                    logger.error(f"Error processing citywide metric {metric.get('name', 'unknown')}: {str(e)}")
                    continue

        # Process district metrics
        for district_num in range(1, 12):
            district_str = str(district_num)
            if district_str in metrics['districts']:
                district_data = metrics['districts'][district_str]
                for category in district_data['categories']:
                    category_name = category['category']
                    for metric in category['metrics']:
                        try:
                            # Validate required fields exist
                            required_fields = ['name', 'id', 'thisYear', 'lastYear', 'lastDataDate', 'queries']
                            missing_fields = [field for field in required_fields if field not in metric]
                            if missing_fields:
                                logger.warning(f"Skipping district {district_str} metric due to missing required fields: {missing_fields}")
                                continue
                                
                            metric_text = create_metric_text(category_name, metric, district_data['name'])
                            texts_to_embed.append(metric_text)
                            metadata_batch.append(create_metadata_dict(category_name, metric, district_str, district_data['name']))
                        except Exception as e:
                            logger.error(f"Error processing district {district_str} metric {metric.get('name', 'unknown')}: {str(e)}")
                            continue

        # Skip if no valid metrics were found
        if not texts_to_embed:
            logger.warning("No valid metrics found to embed. Skipping vector collection creation.")
            return

        # Process in batches
        total_points = len(texts_to_embed)
        points_to_upsert = []
        
        for i in range(0, total_points, BATCH_SIZE):
            batch_texts = texts_to_embed[i:i + BATCH_SIZE]
            batch_metadata = metadata_batch[i:i + BATCH_SIZE]
            
            # Get embeddings for the batch
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts
            )
            embeddings = [data.embedding for data in response.data]
            
            # Create points for the batch
            batch_points = [
                rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=metadata
                )
                for embedding, metadata in zip(embeddings, batch_metadata)
            ]
            points_to_upsert.extend(batch_points)
            
            # Upsert when batch is full or on last batch
            if len(points_to_upsert) >= BATCH_SIZE or i + BATCH_SIZE >= total_points:
                for upsert_attempt in range(MAX_RETRIES):
                    try:
                        if not qdrant.collection_exists(collection_name):
                            create_collection()
                        qdrant.upsert(
                            collection_name=collection_name,
                            points=points_to_upsert
                        )
                        points_to_upsert = []
                        break
                    except Exception as e:
                        logger.error(f"Upsert attempt {upsert_attempt + 1} failed: {str(e)}")
                        if upsert_attempt == MAX_RETRIES - 1:
                            raise
                        time.sleep(2 ** upsert_attempt)
        
        logger.info("Successfully created YTD vector collection")

    except Exception as e:
        logger.error(f"Error creating YTD vector collection: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't re-raise the exception to allow the script to continue with other operations

def create_metric_text(category_name, metric, district_name):
    """Create text representation for embedding."""
    category_descriptions = {
        'Safety': 'public safety, emergency response, and first responders',
        'Crime': 'law enforcement, police activity, and criminal statistics',
        'Economy': 'business and economic development'
    }
    
    category_help_text = {
        'Safety': 'This metric helps track emergency response, public safety incidents, and first responder activity.',
        'Crime': 'This metric helps track law enforcement activity, police incidents, and crime statistics.',
        'Economy': 'This metric helps track business and economic activity.'
    }
    
    # Safely access metadata fields with defaults if they don't exist
    metadata = metric.get('metadata', {})
    summary = metadata.get('summary', 'No summary available')
    definition = metadata.get('definition', 'No detailed definition available')
    data_sf_url = metadata.get('data_sf_url', 'No data source URL available')
    
    # Build the text representation
    text_parts = [
        f"Category: {category_name}",
        f"Metric Name: {metric['name']}",
        f"Description: {summary}",
        f"Detailed Definition: {definition}",
        f"This metric shows {metric['thisYear']} incidents in the current year compared to {metric['lastYear']} incidents last year for {district_name}."
    ]
    
    # Add monthly data if available
    if 'thisMonth' in metric and 'lastMonth' in metric:
        text_parts.append(f"For the current month, there were {metric['thisMonth']} incidents compared to {metric['lastMonth']} incidents in the previous month.")
    
    text_parts.extend([
        f"The data is current as of {metric['lastDataDate']}.",
        f"This is a {category_name.lower()} metric related to {category_descriptions.get(category_name, '')} in San Francisco.",
        f"{category_help_text.get(category_name, '')}",
        f"Data Source: {data_sf_url}",
        f"Query Context: {metric['queries'].get('metric_query', '')}",
        f"This data is specific to {district_name} in San Francisco."
    ])
    
    return "\n".join(text_parts).strip()

def create_metadata_dict(category_name, metric, district, district_name):
    """Create metadata dictionary for vector storage."""
    # Safely access metadata fields with defaults if they don't exist
    metadata = metric.get('metadata', {})
    summary = metadata.get('summary', 'No summary available')
    definition = metadata.get('definition', 'No detailed definition available')
    data_url = metadata.get('data_sf_url', 'No data source URL available')
    
    # Create the base metadata dictionary
    metadata_dict = {
        'category': category_name,
        'metric_name': metric['name'],
        'metric_id': metric['id'],
        'this_year': metric['thisYear'],
        'last_year': metric['lastYear'],
        'last_data_date': metric['lastDataDate'],
        'endpoint': metric.get('endpoint', ''),
        'summary': summary,
        'definition': definition,
        'data_url': data_url,
        'district': district,
        'district_name': district_name,
        'trend_data': metric.get('trend_data', {}),
        'queries': metric['queries']
    }
    
    # Add monthly data if available
    if 'thisMonth' in metric:
        metadata_dict['this_month'] = metric['thisMonth']
    if 'lastMonth' in metric:
        metadata_dict['last_month'] = metric['lastMonth']
    
    # Add location and category fields if available
    if 'location_fields' in metric:
        metadata_dict['location_fields'] = metric['location_fields']
    if 'category_fields' in metric:
        metadata_dict['category_fields'] = metric['category_fields']
    if 'numeric_id' in metric:
        metadata_dict['numeric_id'] = metric['numeric_id']
    if 'show' in metric:
        metadata_dict['show'] = metric['show']
    
    return metadata_dict

def setup_logging():
    """Configure logging for the dashboard metrics generation."""
    # Create logs directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure logging with a single handler for both file and console
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'dashboard_metrics.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def process_single_metric(metric_id, period_type='ytd'):
    """Process a single metric and generate its dashboard files.
    
    Args:
        metric_id (str or int): The ID of the metric to process (e.g., "arrests_presented_to_da_ytd" or 13)
        period_type (str): The type of period to process ('ytd', 'month', 'year', 'week')
    """
    # Load single metric from database
    result = load_single_metric_from_db(metric_id)
    if not result:
        logging.error(f"Failed to load metric {metric_id} from database")
        return None
    
    single_metric_queries, category_name, subcategory_name, query_name, metric_data = result
    
    logging.info(f"Loaded metric {metric_id} ({query_name}) from database")
    
    # Determine the output directory based on period_type
    script_dir = os.path.dirname(os.path.abspath(__file__))
    period_folder_map = {
        'ytd': 'dashboard',
        'month': 'monthly',
        'year': 'annual',
        'week': 'weekly'
    }
    
    if period_type not in period_folder_map:
        logging.error(f"Invalid period_type: {period_type}")
        return None
        
    period_folder = period_folder_map[period_type]
    output_dir = os.path.join(script_dir, 'output', period_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set target date to yesterday
    target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Process the single metric (using optimized version)
    metrics_result = generate_ytd_metrics(single_metric_queries, output_dir, target_date)
    
    if not metrics_result:
        return None
    
    # Now we need to recreate top_level.json files from all individual metric files
    for district_num in range(12):  # 0-11 for citywide and districts 1-11
        district_str = str(district_num)
        district_dir = os.path.join(output_dir, district_str)
        
        # Skip if this district wasn't processed
        if not os.path.exists(district_dir):
            continue
        
        # Initialize the district data structure
        district_data = {
            "metadata": metrics_result['metadata'],
            "name": f"{'Citywide' if district_str == '0' else f'District {district_str}'}",
            "categories": []
        }
        
        # Get all metric files in the district directory and sort them numerically
        metric_files = [f for f in os.listdir(district_dir) if f.endswith('.json') and f != 'top_level.json']
        
        # Sort files numerically by the numeric part of the filename
        def get_numeric_key(filename):
            # Remove .json extension and try to convert to int
            try:
                return int(filename.replace('.json', ''))
            except ValueError:
                # If conversion fails, return a large number to put non-numeric files at the end
                return float('inf')
        
        metric_files.sort(key=get_numeric_key)
        
        # Process each metric file in numeric order
        for metric_file in metric_files:
            try:
                with open(os.path.join(district_dir, metric_file), 'r', encoding='utf-8') as f:
                    metric_data = json.load(f)
                
                # Find or create category
                category = next(
                    (cat for cat in district_data['categories'] 
                     if cat['category'] == metric_data['category']),
                    None
                )
                
                if not category:
                    category = {
                        "category": metric_data['category'],
                        "metrics": []
                    }
                    district_data['categories'].append(category)
                
                # Create metric entry for top_level.json
                metric_entry = {
                    "name": metric_data['metric_name'],
                    "id": metric_data['metric_id'],
                    "lastYear": metric_data['lastYear'],
                    "thisYear": metric_data['thisYear'],
                    "lastDataDate": metric_data['lastDataDate'],
                    "endpoint": metric_data.get('endpoint', ''),
                    "metadata": metric_data['metadata']
                }
                
                # Add monthly data if available
                if 'thisMonth' in metric_data:
                    metric_entry['thisMonth'] = metric_data['thisMonth']
                if 'lastMonth' in metric_data:
                    metric_entry['lastMonth'] = metric_data['lastMonth']
                
                # Add MTD data if available
                if 'thisMtd' in metric_data:
                    metric_entry['thisMtd'] = metric_data['thisMtd']
                if 'lastMtd' in metric_data:
                    metric_entry['lastMtd'] = metric_data['lastMtd']
                
                # Add MTD date ranges if available
                if 'mtd_date_ranges' in metric_data:
                    metric_entry['mtd_date_ranges'] = metric_data['mtd_date_ranges']
                
                # Add numeric_id if it exists
                if 'numeric_id' in metric_data:
                    metric_entry['numeric_id'] = metric_data['numeric_id']
                
                # Add location_fields and category_fields if they exist
                if 'location_fields' in metric_data:
                    metric_entry['location_fields'] = metric_data['location_fields']
                if 'category_fields' in metric_data:
                    metric_entry['category_fields'] = metric_data['category_fields']
                
                # Add show flag if it exists
                if 'show' in metric_data:
                    metric_entry['show'] = metric_data['show']
                
                category['metrics'].append(metric_entry)
                
            except Exception as e:
                logging.error(f"Error processing metric file {metric_file}: {str(e)}")
                continue
        
        # Save top_level.json
        top_level_file = os.path.join(district_dir, 'top_level.json')
        with open(top_level_file, 'w', encoding='utf-8') as f:
            clean_data = clean_nan_values(district_data)
            json.dump(clean_data, f, indent=2)
        logging.info(f"Created new top_level.json for district {district_str}")
        
        # Save to history directory with timestamp
        history_dir = os.path.join(output_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f'district_{district_str}_{datetime.now().strftime("%Y%m%d")}.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            clean_data = clean_nan_values(district_data)
            json.dump(clean_data, f, indent=2)
        logging.info(f"District {district_str} metrics history saved to {history_file}")
    
    return metrics_result

def update_metric_most_recent_data_date(metric_id, most_recent_date):
    """Update the most_recent_data_date column for a specific metric.
    
    Args:
        metric_id (int): The ID of the metric to update
        most_recent_date (str): The most recent data date in YYYY-MM-DD format
    """
    try:
        import psycopg2
        
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to connect to database for updating most_recent_data_date")
            return False
        
        cursor = connection.cursor()
        
        # Update the most_recent_data_date column
        cursor.execute("""
            UPDATE metrics 
            SET most_recent_data_date = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (most_recent_date, metric_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Updated most_recent_data_date for metric {metric_id} to {most_recent_date}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating most_recent_data_date for metric {metric_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def format_date_ranges_readable(date_ranges):
    """
    Format date ranges in a more readable way with friendly date formats.
    
    Args:
        date_ranges (dict): Dictionary containing date range keys
        
    Returns:
        str: Formatted date range string
    """
    try:
        # Parse dates and format them in a friendly way
        this_year_start_dt = datetime.strptime(date_ranges['this_year_start'], '%Y-%m-%d')
        this_year_end_dt = datetime.strptime(date_ranges['this_year_end'], '%Y-%m-%d')
        last_year_start_dt = datetime.strptime(date_ranges['last_year_start'], '%Y-%m-%d')
        last_year_end_dt = datetime.strptime(date_ranges['last_year_end'], '%Y-%m-%d')
        
        # Format with month and day, year shown only once
        this_year_start_formatted = this_year_start_dt.strftime('%b %d')
        this_year_end_formatted = this_year_end_dt.strftime('%b %d')
        last_year_start_formatted = last_year_start_dt.strftime('%b %d')
        last_year_end_formatted = last_year_end_dt.strftime('%b %d')
        
        # Get the years
        this_year = this_year_start_dt.year
        last_year = last_year_start_dt.year
        
        # Format with parentheses around each range, year shown only once
        formatted_string = f"Comparing ({this_year_start_formatted}-{this_year_end_formatted} {this_year}) vs ({last_year_start_formatted}-{last_year_end_formatted} {last_year})"
        
        return formatted_string
    except Exception as e:
        logger.error(f"Error formatting date ranges: {str(e)}")
        # Fallback to original format if there's an error
        return f"Comparing {date_ranges['this_year_start']} to {date_ranges['this_year_end']} vs {date_ranges['last_year_start']} to {date_ranges['last_year_end']}"

def main():
    """Main function to generate dashboard metrics."""
    # Set up logging
    setup_logging()
    
    # Define output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "dashboard")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate dashboard metrics')
    parser.add_argument('--metric-id', help='ID of a single metric to process (e.g., "arrests_presented_to_da_ytd")')
    parser.add_argument('--period-type', help='Type of period to process (e.g., "ytd", "month", "year", "week")')
    parser.add_argument('--target-date', help='Target date for metrics calculation in YYYY-MM-DD format. Defaults to yesterday if not provided.')
    args = parser.parse_args()
    
    if args.metric_id:
        # Process single metric
        process_single_metric(args.metric_id, args.period_type)
    else:
        # Load dashboard queries from database
        logging.info("Loading metrics from database")
        dashboard_queries = load_metrics_from_db()
        
        if not dashboard_queries:
            logging.error("Failed to load metrics from database")
            return
        
        # Generate all metrics
        generate_ytd_metrics(dashboard_queries, output_dir, args.target_date)
    
    logging.info("Dashboard metrics generation complete")

if __name__ == '__main__':
    main() 
"""
Data Processing Module for Weekly Analysis

This module handles data loading, metric discovery, query transformation,
and field extraction for weekly analysis.
"""

import os
import logging
import traceback
import re
from datetime import date, timedelta
import psycopg2.extras
from dotenv import load_dotenv

from tools.db_utils import get_postgres_connection

# Load environment variables to get LOG_LEVEL
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create module logger
logger = logging.getLogger(__name__)

# Configure file handler for weekly analysis
weekly_log_file = os.path.join(logs_dir, 'weekly_metric_analysis.log')
file_handler = logging.FileHandler(weekly_log_file, mode='a', encoding='utf-8')
file_handler.setLevel(log_level)  # Use environment variable instead of hardcoded INFO

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger if it doesn't already have one
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.setLevel(log_level)  # Use environment variable instead of hardcoded INFO

logger.info(f"Weekly data processor logging initialized with level: {log_level_str}. Log file: {weekly_log_file}")

def find_metric_in_queries(queries_data, metric_id):
    """Find a specific metric in the database."""
    # Convert metric_id to string for consistent comparison
    metric_id_str = str(metric_id)
    logger.info(f"Searching for metric ID: {metric_id_str}")
    
    try:
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
                    m.subcategory,
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
                    m.most_recent_data_date,
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
                    m.subcategory,
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
                    m.most_recent_data_date,
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
            logger.error(f"Metric with ID '{metric_id_str}' not found in database")
            return None
        
        # Convert to the expected format
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
            'most_recent_data_date': metric['most_recent_data_date']
        }
        
        return {
            'top_category': metric['category'] or 'uncategorized',
            'subcategory': metric['subcategory'] or metric['category'] or 'uncategorized',
            'query_name': metric['metric_name'],
            'query_data': query_data,
            'endpoint': metric['endpoint'],
            'category_fields': metric['category_fields'] or [],
            'location_fields': metric['location_fields'] or [],
            'numeric_id': metric['id'],
            'id': str(metric['id']),
            'metric_id': str(metric['id']),
            'most_recent_data_date': metric['most_recent_data_date']
        }
        
    except Exception as e:
        logger.error(f"Error finding metric {metric_id} in database: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def detect_avg_aggregation(query):
    """
    Detect if a query uses an AVG() aggregation function for the main value field.
    
    Args:
        query (str): The SQL query to analyze
        
    Returns:
        bool: True if the query uses AVG() aggregation, False otherwise
    """
    if not query:
        return False
        
    # Look for AVG() function in the query
    avg_pattern = r'AVG\s*\(([^)]+)\)'
    avg_matches = re.findall(avg_pattern, query, re.IGNORECASE)
    
    # If we found any AVG() function, return True
    return len(avg_matches) > 0

def extract_date_field_from_query(query):
    """Extract the date field from a query."""
    # 1. Look for date_trunc_* pattern first – this gives us the **actual** column
    #    wrapped by the truncation function, which is what we want to display.
    date_trunc_match = re.search(r'date_trunc_[ymd]+ *\( *([^\)]+) *\)', query, re.IGNORECASE)
    if date_trunc_match:
        field = date_trunc_match.group(1).strip()
        logger.info(f"Found date field from date_trunc: {field}")
        return field

    # 2. If no date_trunc, check common date-like column names *excluding* the generic
    #    alias "date" – we'll only consider that as a very last resort.
    date_fields_to_check = [
        'incident_date', 'report_date', 'arrest_date', 'received_datetime',
        'Report_Datetime', 'disposition_date', 'dba_start_date', 'day', 'actual_date'
    ]

    for field in date_fields_to_check:
        if field in query:
            logger.info(f"Found date field in query: {field}")
            return field

    # 3. Finally, if we still haven't found anything, fall back to the alias 'date'
    if ' date ' in f" {query.lower()} ":
        logger.info("Falling back to generic 'date' alias")
        return 'date'

    return None

def transform_query_for_weekly(original_query, date_field, category_fields, recent_period, comparison_period, district=None):
    """
    Transform a query for weekly analysis by:
    1. Replacing date placeholders
    2. Using appropriate date ranges for recent and comparison periods
    3. Adding category fields to the SELECT clause
    4. Retrieving daily data for later aggregation by week
    5. Adding district filter if specified
    
    Args:
        original_query (str): The original SQL query
        date_field (str): The name of the date field
        category_fields (list): List of category fields
        recent_period (dict): Recent period date range
        comparison_period (dict): Comparison period date range
        district (int, optional): District number to filter by
        
    Returns:
        str: Transformed SQL query
    """
    # Ensure we have valid date ranges
    if not recent_period or not comparison_period:
        logger.error("Missing date periods in transform_query_for_weekly")
        # Create default periods
        today = date.today()
        if not recent_period:
            recent_period = {
                'start': today - timedelta(days=7),
                'end': today
            }
            logger.warning(f"Using default recent_period: {recent_period}")
        
        if not comparison_period:
            comparison_period = {
                'start': today - timedelta(days=364),  # 52 weeks ago
                'end': today - timedelta(days=8)
            }
            logger.warning(f"Using default comparison_period: {comparison_period}")
    
    # Verify the date fields exist in the period dictionaries
    for period, name in [(recent_period, 'recent_period'), (comparison_period, 'comparison_period')]:
        for field in ['start', 'end']:
            if field not in period or period[field] is None:
                period[field] = date.today() if field == 'end' else date.today() - timedelta(days=7)
                logger.warning(f"Missing {field} in {name}, using default: {period[field]}")
    
    # Format date strings for SQL - with error handling
    try:
        recent_start = recent_period['start'].isoformat()
        recent_end = recent_period['end'].isoformat()
        comparison_start = comparison_period['start'].isoformat()
        comparison_end = comparison_period['end'].isoformat()
    except (AttributeError, TypeError) as e:
        logger.error(f"Error formatting dates: {str(e)}")
        # Use string fallbacks
        recent_start = "2023-01-01" if not isinstance(recent_period.get('start'), date) else recent_period['start'].isoformat()
        recent_end = "2023-01-07" if not isinstance(recent_period.get('end'), date) else recent_period['end'].isoformat()
        comparison_start = "2022-12-01" if not isinstance(comparison_period.get('start'), date) else comparison_period['start'].isoformat()
        comparison_end = "2022-12-31" if not isinstance(comparison_period.get('end'), date) else comparison_period['end'].isoformat()
    
    # Replace any date placeholders in the original query
    modified_query = original_query
    
    # Get fiscal year values (similar to generate_dashboard_metrics.py)
    today = date.today()
    this_fiscal_year = str(today.year + 1)  # Fiscal year starts in July, so current fiscal year is next calendar year
    last_fiscal_year = str(today.year)
    
    replacements = {
        'this_year_start': f"'{recent_start}'",
        'this_year_end': f"'{recent_end}'",
        'last_year_start': f"'{comparison_start}'",
        'last_year_end': f"'{comparison_end}'",
        'start_date': f"'{comparison_start}'",
        'current_date': f"'{recent_end}'",
        # Add fiscal year variables
        'this_fiscal_year_start': f"'{this_fiscal_year}'",
        'this_fiscal_year_end': f"'{this_fiscal_year}'",
        'last_fiscal_year_start': f"'{last_fiscal_year}'",
        'last_fiscal_year_end': f"'{last_fiscal_year}'"
    }
    
    # Apply replacements - ensure we're not creating malformed field names
    for placeholder, value in replacements.items():
        # Use a more flexible replacement pattern that handles various contexts
        # Replace the placeholder when it's surrounded by spaces, operators, or at the start/end
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(placeholder) + r'(?![a-zA-Z0-9_])'
        modified_query = re.sub(pattern, value, modified_query)
        logger.info(f"Replaced {placeholder} with {value}")
    
    # Determine if it's a YTD query by checking format
    is_ytd_query = ('as date, COUNT(*)' in modified_query or 
                   'as date,' in modified_query or 
                   'date_trunc_ymd' in modified_query)
    
    # If it's a YTD query, we'll modify it to work with our weekly analysis
    if is_ytd_query:
        logger.info("Using YTD query format as basis for weekly analysis")
        
        # Extract the core table and WHERE conditions from the original query
        # This pattern looks for date_trunc, field selection, conditions
        ytd_pattern = r'SELECT\s+date_trunc_[ymd]+\((.*?)\)\s+as\s+date,\s+([^W]+)WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)'
        ytd_match = re.search(ytd_pattern, modified_query, re.IGNORECASE | re.DOTALL)
        
        if ytd_match:
            date_field_match = ytd_match.group(1).strip()
            value_part = ytd_match.group(2).strip()
            where_part = ytd_match.group(3).strip()
            
            # Remove current_date references and replace with our recent_end
            where_part = re.sub(r'<=\s*current_date', f"<= '{recent_end}'", where_part)
            
            # Keep the actual date instead of transforming to week
            date_select = f"{date_field_match} as actual_date"
            
            # Build the category fields part of the SELECT and GROUP BY
            category_select = ""
            group_by_fields = []
            
            for field in category_fields:
                if isinstance(field, dict):
                    field_name = field.get('fieldName', '')
                else:
                    field_name = field
                
                if field_name:
                    category_select += f", {field_name}"
                    group_by_fields.append(field_name)
            
            # Add period_type to distinguish recent from comparison - FIXED POSITION
            period_type_select = f", CASE WHEN {date_field_match} >= '{recent_start}' AND {date_field_match} <= '{recent_end}' THEN 'recent' ELSE 'comparison' END as period_type"
            
            # Build the complete transformed query
            group_by_clause = "GROUP BY actual_date, period_type"
            if group_by_fields:
                group_by_clause += ", " + ", ".join(group_by_fields)
                
            transformed_query = f"""
            SELECT 
                {date_select},
                {value_part}
                {period_type_select}
                {category_select}
            WHERE 
                {where_part} AND
                (
                    ({date_field_match} >= '{comparison_start}' AND {date_field_match} <= '{comparison_end}')
                    OR 
                    ({date_field_match} >= '{recent_start}' AND {date_field_match} <= '{recent_end}')
                )
            {group_by_clause}
            ORDER BY actual_date
            """
            
            return transformed_query
        else:
            # If we can't parse the YTD query, fall back to the regular transform
            logger.warning("Could not extract components from YTD query, falling back to standard transformation")

    # Try to extract the FROM clause
    from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', modified_query, re.IGNORECASE | re.DOTALL)
    
    # If FROM clause not found, try to infer it from the query
    if not from_match:
        # Check if there's a table name after SELECT
        table_match = re.search(r'SELECT.*?FROM\s+([^\s,]+)', modified_query, re.IGNORECASE)
        if table_match:
            from_clause = table_match.group(1).strip()
        else:
            logger.warning("Could not extract FROM clause from query, using modified query with replaced placeholders")
            return modified_query
    else:
        from_clause = from_match.group(1).strip()
    
    # Try to extract the WHERE clause from the modified query
    where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', modified_query, re.IGNORECASE | re.DOTALL)
    where_clause = ""
    if where_match:
        # Keep the original WHERE clause but add date filters for both periods
        original_where = where_match.group(1).strip()
        where_clause = f"""
        WHERE ({original_where}) AND (
            ({date_field} >= '{comparison_start}' AND {date_field} <= '{comparison_end}')
            OR 
            ({date_field} >= '{recent_start}' AND {date_field} <= '{recent_end}')
        )
        """
        
        # Add district filter if specified and we are not processing multiple districts
        if district is not None and isinstance(district, int) and district > 0 and 'supervisor_district' in modified_query:
            where_clause = where_clause.rstrip() + f" AND supervisor_district = '{district}'\n"
            logger.info(f"Added district filter to WHERE clause: supervisor_district = '{district}'")
    else:
        # Create a new WHERE clause with just date filters for both periods
        where_clause = f"""
        WHERE (
            ({date_field} >= '{comparison_start}' AND {date_field} <= '{comparison_end}')
            OR 
            ({date_field} >= '{recent_start}' AND {date_field} <= '{recent_end}')
        )
        """
        
        # Add district filter if specified and we are not processing multiple districts
        if district is not None and isinstance(district, int) and district > 0 and 'supervisor_district' in modified_query:
            where_clause = where_clause.rstrip() + f" AND supervisor_district = '{district}'\n"
            logger.info(f"Added district filter to new WHERE clause: supervisor_district = '{district}'")
    
    # If we have a valid FROM clause, proceed with transformation
    if from_clause:
        # Build the category fields part of the SELECT clause
        category_select = ""
        category_fields_list = []  # To track field names for GROUP BY
        
        for field in category_fields:
            if isinstance(field, dict):
                field_name = field.get('fieldName', '')
            else:
                field_name = field
            
            if field_name:
                category_select += f", {field_name}"
                category_fields_list.append(field_name)
        
        # Keep the actual date instead of transforming to week
        date_select = f"{date_field} as actual_date"

        # Build the GROUP BY clause with category fields (if any)
        group_by = "GROUP BY actual_date"
        for field_name in category_fields_list:
            group_by += f", {field_name}"
        
        # Add period_type to distinguish recent from comparison
        period_type_select = f", CASE WHEN {date_field} >= '{recent_start}' AND {date_field} <= '{recent_end}' THEN 'recent' ELSE 'comparison' END as period_type"
        
        # Build the complete transformed query with daily data
        transformed_query = f"""
        SELECT 
            {date_select},
            COUNT(*) as value
            {period_type_select}
            {category_select}
        FROM {from_clause}
        {where_clause}
        {group_by}, period_type
        ORDER BY actual_date
        """
        
        return transformed_query
    else:
        # If we couldn't extract or infer the FROM clause, return the modified query
        logger.warning("Could not determine FROM clause, using modified query with replaced placeholders")
        return modified_query 
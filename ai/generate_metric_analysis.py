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

from tools.data_fetcher import set_dataset
from tools.genChart import generate_time_series_chart
from tools.anomaly_detection import anomaly_detection
from tools.generate_map import generate_map
from tools.genAggregate import aggregate_data  # Import aggregate_data function

# Configure logging AFTER all imports to avoid being overridden by imported modules
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Set output directory
OUTPUT_DIR = os.path.join(script_dir, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for different period types
ANNUAL_DIR = os.path.join(OUTPUT_DIR, 'annual')
MONTHLY_DIR = os.path.join(OUTPUT_DIR, 'monthly')
os.makedirs(ANNUAL_DIR, exist_ok=True)
os.makedirs(MONTHLY_DIR, exist_ok=True)

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

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(log_level)  # Use environment variable instead of hardcoded INFO
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Remove any existing handlers from root logger
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add file handler to root logger
root_file_handler = logging.FileHandler(os.path.join(logs_dir, 'metric_analysis.log'))
root_file_handler.setLevel(log_level)  # Use environment variable instead of hardcoded INFO
root_file_handler.setFormatter(formatter)
root_logger.addHandler(root_file_handler)

# Add console handler to root logger
root_console_handler = logging.StreamHandler()
root_console_handler.setLevel(log_level)  # Use environment variable instead of hardcoded INFO
root_console_handler.setFormatter(formatter)
root_logger.addHandler(root_console_handler)

# Now configure the module logger
logger = logging.getLogger(__name__)
logger.setLevel(log_level)  # Use environment variable instead of hardcoded INFO

# The module logger will inherit handlers from the root logger
# so we don't need to add handlers to it

# Log a message to confirm logging is set up
logger.info(f"Logging configured for generate_metric_analysis.py with level: {log_level_str}")
root_logger.info(f"Root logger configured for generate_metric_analysis.py with level: {log_level_str}")

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None

def get_time_ranges(period_type):
    """
    Calculate recent and comparison periods based on period type.
    
    Args:
        period_type (str): One of 'year', 'month', 'day', or 'ytd'
    
    Returns:
        tuple: (recent_period, comparison_period) each containing start and end dates
    """
    today = date.today()
    
    if period_type == 'ytd':
        # Current year from Jan 1 to yesterday
        recent_period = {
            'start': date(today.year, 1, 1),
            'end': today - timedelta(days=1)
        }
        # Same days last year
        comparison_period = {
            'start': date(today.year - 1, 1, 1),
            'end': date(today.year - 1, today.month, today.day) - timedelta(days=1)
        }
    elif period_type == 'year':
        # Use previous year as the recent period
        previous_year = today.year - 1
        recent_period = {
            'start': date(previous_year, 1, 1),
            'end': date(previous_year, 12, 31)
        }
        # Compare to 6 years before that
        earliest_comparison_year = previous_year - 6
        comparison_period = {
            'start': date(earliest_comparison_year, 1, 1),
            'end': date(previous_year - 1, 12, 31)
        }
    elif period_type == 'month':
        # Use the previous complete month
        if today.month == 1:
            recent_month = 12
            recent_year = today.year - 1
        else:
            recent_month = today.month - 1
            recent_year = today.year
            
        # Calculate last day of the month
        if recent_month == 12:
            last_day = 31
        elif recent_month in [4, 6, 9, 11]:
            last_day = 30
        elif recent_month == 2:
            # Handle leap years
            if recent_year % 4 == 0 and (recent_year % 100 != 0 or recent_year % 400 == 0):
                last_day = 29
            else:
                last_day = 28
        else:
            last_day = 31
            
        recent_period = {
            'start': date(recent_year, recent_month, 1),
            'end': date(recent_year, recent_month, last_day)
        }
        
        # Compare to previous 24 months
        comparison_start_month = recent_month
        comparison_start_year = recent_year - 2
        
        comparison_period = {
            'start': date(comparison_start_year, comparison_start_month, 1),
            'end': date(recent_year, recent_month, 1) - timedelta(days=1)
        }
    else:  # day
        # Last complete day
        yesterday = today - timedelta(days=1)
        
        recent_period = {
            'start': yesterday,
            'end': yesterday
        }
        
        # Compare to same day in previous weeks
        comparison_start = yesterday - timedelta(days=28)  # 4 weeks ago
        comparison_end = yesterday - timedelta(days=1)  # yesterday
        
        comparison_period = {
            'start': comparison_start,
            'end': comparison_end
        }
    
    return recent_period, comparison_period

def find_metric_in_queries(queries_data, metric_id):
    """Find a specific metric in the dashboard queries data structure."""
    # Convert metric_id to string for consistent comparison
    metric_id_str = str(metric_id)
    logger.info(f"Searching for metric ID: {metric_id_str}")
    
    # Log the structure we're searching through
    logger.info(f"Searching through queries data structure: {list(queries_data.keys())}")
        
    for top_category_name, top_category_data in queries_data.items():
        logger.info(f"Checking top category: {top_category_name}")
        for subcategory_name, subcategory_data in top_category_data.items():
            logger.info(f"Checking subcategory: {subcategory_name}")
            if isinstance(subcategory_data, dict) and 'queries' in subcategory_data:
                for query_name, query_data in subcategory_data['queries'].items():
                    # Log the query we're checking
                    logger.info(f"Checking query: {query_name}")
                    
                    # Check for numeric ID match
                    if isinstance(query_data, dict):
                        query_id = query_data.get('id')
                        # Convert query_id to string for comparison
                        query_id_str = str(query_id) if query_id is not None else None
                        logger.info(f"Comparing metric ID {metric_id_str} with query ID {query_id_str}")
                        
                        if query_id_str == metric_id_str:
                            # Found a match by numeric ID
                            logger.info(f"Found match by numeric ID: {metric_id_str}")
                            # Check for endpoint at query level first, then fallback to subcategory level
                            endpoint = None
                            if 'endpoint' in query_data:
                                endpoint = query_data.get('endpoint')
                                logger.info(f"Using query-level endpoint: {endpoint}")
                            else:
                                endpoint = subcategory_data.get('endpoint', None)
                                logger.info(f"Using subcategory-level endpoint: {endpoint}")
                                
                            return {
                                'top_category': top_category_name,
                                'subcategory': subcategory_name,
                                'query_name': query_name,
                                'query_data': query_data,
                                'endpoint': endpoint,
                                'category_fields': query_data.get('category_fields', []) if isinstance(query_data, dict) else [],
                                'location_fields': query_data.get('location_fields', []) if isinstance(query_data, dict) else [],
                                'numeric_id': query_data.get('id', None) if isinstance(query_data, dict) else None,
                                'metric_id': metric_id_str  # Ensure we always have the metric_id from the search
                            }
                    
                    # For string IDs, try to match the query name
                    if isinstance(metric_id, str):
                        # Clean up the query name for comparison
                        clean_query_name = query_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("_ytd", "")
                        clean_metric_id = metric_id.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("_ytd", "")
                        
                        logger.info(f"Comparing string IDs - Query: {clean_query_name}, Metric: {clean_metric_id}")
                        
                        if clean_query_name == clean_metric_id:
                            logger.info(f"Found match by string ID: {metric_id}")
                            return {
                                'top_category': top_category_name,
                                'subcategory': subcategory_name,
                                'query_name': query_name,
                                'query_data': query_data,
                                'endpoint': query_data.get('endpoint', subcategory_data.get('endpoint')),
                                'category_fields': query_data.get('category_fields', []),
                                'location_fields': query_data.get('location_fields', []),
                                'numeric_id': query_data.get('id'),
                                'metric_id': metric_id
                            }
    
    logger.error(f"Metric with ID '{metric_id_str}' not found in dashboard queries")
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

def process_metric_analysis(metric_info, period_type='month', process_districts=False):
    """Process metric analysis for a given period type with optional district processing."""
    # Extract metric information
    metric_id = metric_info.get('metric_id', '')
    query_name = metric_info.get('query_name', metric_id)
    definition = metric_info.get('definition', '')
    summary = metric_info.get('summary', '')
    endpoint = metric_info.get('endpoint', '')
    data_sf_url = metric_info.get('data_sf_url', '')
    
    # Determine period description based on period_type
    period_desc = 'Monthly' if period_type == 'month' else 'Annual'
    
    # Get time ranges based on period type
    recent_period, comparison_period = get_time_ranges(period_type)
    
    # Create context variables and set the dataset
    context_variables = {}
    
    # Get the query from metric_info - USE YTD QUERY INSTEAD OF METRIC QUERY
    original_query = None
    if isinstance(metric_info.get('query_data'), dict):
        original_query = metric_info['query_data'].get('ytd_query', '')
        if not original_query:
            # Fall back to metric_query if ytd_query is not available
            original_query = metric_info['query_data'].get('metric_query', '')
    else:
        original_query = metric_info.get('query_data', '')
    
    if not original_query:
        logging.error(f"No query found for {query_name}")
        return None
    
    logging.info(f"Original query: {original_query}")
    
    # Check if the query uses AVG() aggregation
    uses_avg = detect_avg_aggregation(original_query)
    logging.info(f"Query uses AVG() aggregation: {uses_avg}")
    
    # Define value_field here, before using it in agg_functions
    value_field = 'value'
    
    # Extract category fields from metric_info
    category_fields = metric_info.get('category_fields', [])
    # Only use category fields that are explicitly defined - no default
    if not category_fields:
        category_fields = []
        logging.info("No category fields defined for this metric. Not using any default fields.")
    
    # Check if supervisor_district exists in category_fields
    has_district = False
    for field in category_fields:
        if (isinstance(field, dict) and field.get('fieldName') == 'supervisor_district') or field == 'supervisor_district':
            has_district = True
            break
    
    # If process_districts is True, make sure supervisor_district is used as a category field
    if process_districts and not has_district and 'supervisor_district' in original_query:
        # Add supervisor_district as a category field
        category_fields.append('supervisor_district')
        logging.info("Added supervisor_district to category fields for district processing")
        has_district = True
    
    # Determine the date field to use from the query
    date_field = extract_date_field_from_query(original_query)
    if not date_field:
        logging.warning(f"No date field found in query for {query_name}")
        date_field = 'date'  # Default to 'date'
    
    logging.info(f"Using date field: {date_field}")
    
    # Determine the appropriate date field name based on period type and query
    date_field_name = determine_date_field_name(original_query, date_field, period_type)
    
    # Set up filter conditions for date filtering
    filter_conditions = []
    if period_type == 'year':
        # For year periods, make sure we're comparing strings with strings
        year_end = str(recent_period['end'].year)
        year_start = str(comparison_period['start'].year)
        filter_conditions = [
            {'field': date_field_name, 'operator': '<=', 'value': year_end},
            {'field': date_field_name, 'operator': '>=', 'value': year_start},
        ]
        logging.info(f"Year filter conditions: {filter_conditions}")
    else:
        # For other period types, use the date objects
        filter_conditions = [
            {'field': date_field_name, 'operator': '<=', 'value': recent_period['end']},
            {'field': date_field_name, 'operator': '>=', 'value': comparison_period['start']},
        ]
        logging.info(f"Standard filter conditions: {filter_conditions}")
    
    # Transform the query for the specified period type
    transformed_query = transform_query_for_period(
        original_query, 
        date_field, 
        category_fields, 
        period_type, 
        recent_period, 
        comparison_period
    )
    
    logging.info(f"Transformed query: {transformed_query}")
    
    # Log the set_dataset call details
    logging.info(f"Calling set_dataset with endpoint: {endpoint}")
    logging.info(f"Query being used: {transformed_query}")
    
    # Create output file base name
    output_file_base = f"{metric_id}_{period_type}_analysis"
    logging.info(f"Base output file name: {output_file_base}")
    
    # Set the dataset using the endpoint and transformed query
    result = set_dataset(context_variables=context_variables, endpoint=endpoint, query=transformed_query)
    
    if 'error' in result:
        logging.error(f"Error setting dataset for {query_name}: {result['error']}")
        return None
    
    # Store the queryURL in context_variables if available
    if 'queryURL' in result:
        context_variables['executed_query_url'] = result['queryURL']
        logging.info(f"Stored executed_query_url in context: {result['queryURL']}")
    
    # Get the dataset from context_variables
    if 'dataset' not in context_variables:
        logging.error(f"No dataset found in context for {query_name}")
        return None
    
    dataset = context_variables['dataset']
    
    # Log available columns in dataset
    logging.info(f"Available columns in dataset: {dataset.columns.tolist()}")
    
    # Create or update value field if needed
    if value_field not in dataset.columns:
        if 'this_year' in dataset.columns:
            # Use this_year as the value field
            dataset[value_field] = dataset['this_year']
            logging.info(f"Created {value_field} from this_year column")
        elif dataset.select_dtypes(include=['number']).columns.tolist():
            # Use the first numeric column as the value field
            numeric_cols = dataset.select_dtypes(include=['number']).columns.tolist()
            # Filter out date-related columns
            numeric_cols = [col for col in numeric_cols if not any(date_term in col.lower() for date_term in ['year', 'month', 'day', 'date'])]
            
            if numeric_cols:
                dataset[value_field] = dataset[numeric_cols[0]]
                logging.info(f"Created {value_field} from {numeric_cols[0]} column")
            else:
                # If no suitable numeric column, use 1 as the value
                dataset[value_field] = 1
                logging.info(f"Created {value_field} with default value 1")
        else:
            # If no numeric columns, use 1 as the value
            dataset[value_field] = 1
            logging.info(f"Created {value_field} with default value 1")
    
    # Now define agg_functions using the value_field
    if uses_avg:
        agg_functions = {value_field: 'mean'}
        logging.info(f"Using 'mean' aggregation for field {value_field} based on AVG() detection in query")
    else:
        agg_functions = {value_field: 'sum'}
        logging.info(f"Using default 'sum' aggregation for field {value_field}")
    
    # Validate category fields - ensure they exist in the dataset
    valid_category_fields = []
    for field in category_fields:
        if isinstance(field, dict):
            field_name = field.get('fieldName', '')
        else:
            field_name = field
        
        if field_name and field_name in dataset.columns:
            valid_category_fields.append(field)
            logging.info(f"Validated category field: {field_name}")
        else:
            logging.warning(f"Category field {field_name} not found in dataset. Ignoring.")
    
    # Update category_fields with only valid fields
    category_fields = valid_category_fields
    
    # If no valid category fields, proceed without category-based analysis
    if not category_fields:
        logging.info("No valid category fields for this dataset. Proceeding with time-series analysis only.")
    
    # Create period field if it doesn't exist
    period_field = f"{period_type}_period"
    if period_field not in dataset.columns:
        if 'max_date' in dataset.columns:
            try:
                # Convert max_date to datetime
                dataset['max_date'] = pd.to_datetime(dataset['max_date'])
                
                # Create period field based on period_type
                if period_type == 'month':
                    # Format as YYYY-MM
                    dataset[period_field] = dataset['max_date'].dt.strftime('%Y-%m')
                else:  # year
                    # Format as YYYY
                    dataset[period_field] = dataset['max_date'].dt.strftime('%Y')
                
                logging.info(f"Created {period_field} from max_date column")
            except Exception as e:
                logging.error(f"Error creating {period_field} from max_date: {e}")
        else:
            # If no max_date column, use current date
            current_date = datetime.now()
            if period_type == 'month':
                period_value = current_date.strftime('%Y-%m')
            else:  # year
                period_value = current_date.strftime('%Y')
            
            dataset[period_field] = period_value
            logging.info(f"Created {period_field} with current date: {period_value}")
    
    # Update filter conditions to use period_field instead of date_field_name
    filter_conditions = [
        {'field': period_field, 'operator': '<=', 'value': recent_period['end'].isoformat(), 'is_date': False},
        {'field': period_field, 'operator': '>=', 'value': comparison_period['start'].isoformat(), 'is_date': False},
    ]
    
    # Log the updated filter conditions
    logging.info(f"Updated filter conditions: {filter_conditions}")
    
    # Create year field if it doesn't exist (needed for anomaly detection)
    if 'year' not in dataset.columns:
        if period_field in dataset.columns:
            try:
                if period_type == 'month':
                    # Extract year from month_period (format: YYYY-MM)
                    dataset['year'] = dataset[period_field].str[:4]
                else:  # year
                    # Year period is already in YYYY format
                    dataset['year'] = dataset[period_field]
                
                logging.info(f"Created 'year' field from {period_field}")
            except Exception as e:
                logging.error(f"Error creating 'year' field from {period_field}: {e}")
                # Default to current year
                dataset['year'] = datetime.now().year
                logging.info(f"Created 'year' field with current year: {datetime.now().year}")
        elif 'max_date' in dataset.columns:
            try:
                # Extract year from max_date
                dataset['year'] = pd.to_datetime(dataset['max_date']).dt.year
                logging.info("Created 'year' field from max_date column")
            except Exception as e:
                logging.error(f"Error creating 'year' field from max_date: {e}")
                # Default to current year
                dataset['year'] = datetime.now().year
                logging.info(f"Created 'year' field with current year: {datetime.now().year}")
        else:
            # Default to current year
            dataset['year'] = datetime.now().year
            logging.info(f"Created 'year' field with current year: {datetime.now().year}")
    
    # Update the dataset in context_variables
    context_variables['dataset'] = dataset
    
    # Initialize lists to store markdown and HTML content
    all_markdown_contents = []
    all_html_contents = []
    
    # First, process the overall (citywide) analysis as district 0
    process_analysis_result = process_single_analysis(
        context_variables=context_variables.copy(),
        category_fields=category_fields,
        period_type=period_type,
        period_field=period_field,
        filter_conditions=filter_conditions,
        query_name=f"{query_name} - Citywide",
        period_desc=f"{period_desc} - Citywide",
        value_field=value_field,
        recent_period=recent_period,
        comparison_period=comparison_period,
        uses_avg=uses_avg,
        agg_functions=agg_functions,
        district=0,  # Use 0 for citywide analysis
        metric_id=metric_id,
        base_metric_name=metric_info.get('query_name', metric_id)
    )
    
    if process_analysis_result:
        result = {
            'query_name': f"{query_name} - Citywide",
            'period_type': period_type,
            'markdown': process_analysis_result.get('markdown', ''),
            'html': process_analysis_result.get('html', ''),
            'metric_id': metric_id
        }
        
        # Save the analysis files for the citywide result (district 0)
        save_analysis_files(result, metric_id, period_type, district=0)
    
    # Generate choropleth maps for supervisor districts if they exist in the dataset
    if has_district and 'supervisor_district' in dataset.columns and period_type == 'month':
        logging.info("Found supervisor_district field. Generating choropleth maps...")
        
        # Create a dictionary to store map IDs for reference in markdown
        generated_maps = {
            'density_map_id': None,
            'density_map_url': None,
            'delta_map_id': None,
            'delta_map_url': None
        }
        
        try:
            # Get the most recent month's data
            if 'month_period' in dataset.columns:
                recent_months = dataset[dataset['period_type'] == 'recent']['month_period'].unique()
                if len(recent_months) > 0:
                    # Sort months and get the most recent one
                    recent_months = sorted(recent_months)
                    last_month = recent_months[-1]
                    
                    # For second_last_month, look in comparison period if not found in recent
                    second_last_month = recent_months[-2] if len(recent_months) > 1 else None
                    
                    # If we don't have a second recent month, look for the most recent comparison month
                    if second_last_month is None:
                        comparison_months = dataset[dataset['period_type'] == 'comparison']['month_period'].unique()
                        if len(comparison_months) > 0:
                            comparison_months = sorted(comparison_months)
                            second_last_month = comparison_months[-1]  # Most recent comparison month
                            logging.info(f"Using most recent comparison month as second_last_month: {second_last_month}")
                    
                    # Log the months we're using
                    logging.info(f"Recent months available: {recent_months}")
                    logging.info(f"Using last_month: {last_month}, second_last_month: {second_last_month}")
                    
                    # Format date values for display
                    last_month_display = last_month
                    second_last_month_display = second_last_month
                    
                    # Try to parse and format the date values
                    try:
                        # Check if last_month is an ISO date format string
                        if isinstance(last_month, str) and (last_month.startswith('20') or 'T' in last_month):
                            # Parse the date - try different formats
                            try:
                                if 'T' in last_month:
                                    # ISO format with time component
                                    dt = datetime.fromisoformat(last_month.replace('Z', '+00:00'))
                                else:
                                    # Just YYYY-MM format
                                    if '-' in last_month and len(last_month.split('-')) == 2:
                                        year, month = last_month.split('-')
                                        dt = datetime(int(year), int(month), 1)
                                    else:
                                        dt = datetime.fromisoformat(f"{last_month}-01")
                                
                                # Format as "Month Year"
                                last_month_display = dt.strftime('%B %Y')
                            except (ValueError, TypeError) as e:
                                logging.warning(f"Failed to parse last_month date {last_month}: {e}")
                                
                        # Do the same for second_last_month if it exists
                        if second_last_month and isinstance(second_last_month, str) and (second_last_month.startswith('20') or 'T' in second_last_month):
                            try:
                                if 'T' in second_last_month:
                                    dt = datetime.fromisoformat(second_last_month.replace('Z', '+00:00'))
                                else:
                                    if '-' in second_last_month and len(second_last_month.split('-')) == 2:
                                        year, month = second_last_month.split('-')
                                        dt = datetime(int(year), int(month), 1)
                                    else:
                                        dt = datetime.fromisoformat(f"{second_last_month}-01")
                                
                                second_last_month_display = dt.strftime('%B %Y')
                            except (ValueError, TypeError) as e:
                                logging.warning(f"Failed to parse second_last_month date {second_last_month}: {e}")
                    except Exception as e:
                        logging.warning(f"Error formatting date values: {e}")
                    
                    # Filter dataset for last month
                    last_month_data = dataset[(dataset['month_period'] == last_month) & 
                                            (dataset['period_type'] == 'recent')]
                    
                    # Prepare data for density map
                    if not last_month_data.empty:
                        density_map_data = []
                        
                        # Log the dataset structure
                        logging.info(f"Last month dataset columns: {last_month_data.columns.tolist()}")
                        logging.info(f"Last month dataset sample (first few rows): {last_month_data.head().to_dict()}")
                        logging.info(f"Value field: {value_field}, Aggregation function: {agg_functions[value_field]}")
                        
                        # Use direct pandas aggregation for spatial data rather than time series aggregation
                        # The genAggregate.aggregate_data function is designed for time series, not spatial data
                        
                        # First ensure value field is numeric
                        try:
                            last_month_data[value_field] = pd.to_numeric(last_month_data[value_field], errors='coerce')
                            logging.info(f"Converted {value_field} to numeric type")
                        except Exception as e:
                            logging.error(f"Error converting values to numeric: {e}")
                        
                        if agg_functions[value_field] == 'mean':
                            district_values = last_month_data.groupby('supervisor_district')[value_field].mean().reset_index()
                            logging.info(f"Using MEAN aggregation for district values")
                        else:
                            district_values = last_month_data.groupby('supervisor_district')[value_field].sum().reset_index()
                            logging.info(f"Using SUM aggregation for district values")
                        
                        logging.info(f"Aggregated district values: {district_values.head().to_dict()}")
                        
                        # Process results for the map
                        for _, row in district_values.iterrows():
                            if pd.isna(row['supervisor_district']):
                                logging.info(f"Skipping NaN district value")
                                continue
                            
                            try:
                                # Handle both numeric and text district values
                                district_value = row['supervisor_district']
                                if pd.api.types.is_numeric_dtype(type(district_value)) or (isinstance(district_value, str) and district_value.replace('.', '', 1).isdigit()):
                                    # It's a numeric value (or numeric as string) - convert to int if possible
                                    try:
                                        district = str(int(float(district_value)))
                                    except:
                                        district = str(district_value)
                                else:
                                    # It's a non-numeric string
                                    district = str(district_value)
                                    # If it's a string like "District 1", extract just the number
                                    district_match = re.search(r'district\s*(\d+)', district.lower())
                                    if district_match:
                                        district = district_match.group(1)
                                
                                value = float(row[value_field])
                                density_map_data.append({
                                    "district": district,
                                    "value": value
                                })
                                logging.info(f"Added district {district} with value {value} to map data")
                            except (ValueError, TypeError) as e:
                                logging.warning(f"Skipping invalid district value: {row['supervisor_district']} - {e}")
                        
                        # Log the final map data
                        logging.info(f"Final density map data: {density_map_data}")
                        
                        # Generate density map
                        if density_map_data:
                            map_title = f"{query_name} - {last_month_display} Values by District"
                            
                            # Convert density_map_data to CSV format for supervisor_district maps
                            csv_data = "district,value\n"
                            for item in density_map_data:
                                csv_data += f"{item['district']},{item['value']}\n"
                            
                            map_result = generate_map(
                                context_variables={},
                                map_title=map_title,
                                map_type="supervisor_district",
                                location_data=csv_data,
                                map_metadata={
                                    "period": last_month_display,
                                    "description": f"Values for {query_name} by supervisor district for {last_month_display}"
                                },
                                metric_id=metric_id,
                                group_field="supervisor_district"
                            )
                            
                            if map_result and map_result.get("map_id"):
                                map_id = map_result.get('map_id')
                                logging.info(f"Successfully generated density map with ID: {map_id}")
                                
                                # Store map ID and URL for markdown reference
                                generated_maps['density_map_id'] = map_id
                                generated_maps['density_map_url'] = map_result.get('publish_url')
                                
                                # Add map to HTML content
                                if "publish_url" in map_result:
                                    map_html = f"""
                                    <div class="map-container">
                                        <h3>District Map: {map_title}</h3>
                                        <iframe width="100%" height="500" src="{map_result['publish_url']}" 
                                            frameborder="0" allowfullscreen></iframe>
                                    </div>
                                    """
                                    all_html_contents.append(map_html)
                            else:
                                logging.error(f"Failed to generate density map: {map_result}")
                    
                    # Generate delta map showing percent change if we have two months
                    if second_last_month and not last_month_data.empty:
                        # Determine which period to look for second_last_month data
                        # If second_last_month is in recent_months, look in recent period, otherwise look in comparison
                        if second_last_month in recent_months:
                            second_last_month_data = dataset[(dataset['month_period'] == second_last_month) & 
                                                          (dataset['period_type'] == 'recent')]
                        else:
                            second_last_month_data = dataset[(dataset['month_period'] == second_last_month) & 
                                                          (dataset['period_type'] == 'comparison')]
                        
                        if not second_last_month_data.empty:
                            # Log data for debugging
                            logging.info(f"Second last month: {second_last_month}")
                            logging.info(f"Current month data size: {len(last_month_data)}")
                            logging.info(f"Previous month data size: {len(second_last_month_data)}")
                            
                            # Use direct pandas aggregation for spatial data
                            
                            # First ensure value field is numeric for both datasets
                            try:
                                last_month_data[value_field] = pd.to_numeric(last_month_data[value_field], errors='coerce')
                                second_last_month_data[value_field] = pd.to_numeric(second_last_month_data[value_field], errors='coerce')
                                logging.info(f"Converted {value_field} to numeric type for both months")
                            except Exception as e:
                                logging.error(f"Error converting values to numeric: {e}")
                            
                            # For current month
                            if agg_functions[value_field] == 'mean':
                                last_month_grouped = last_month_data.groupby('supervisor_district')[value_field].mean().reset_index()
                                second_last_month_grouped = second_last_month_data.groupby('supervisor_district')[value_field].mean().reset_index()
                                logging.info(f"Using MEAN aggregation for district percent change")
                            else:
                                last_month_grouped = last_month_data.groupby('supervisor_district')[value_field].sum().reset_index()
                                second_last_month_grouped = second_last_month_data.groupby('supervisor_district')[value_field].sum().reset_index()
                                logging.info(f"Using SUM aggregation for district percent change")
                            
                            logging.info(f"Current month aggregated values: {last_month_grouped.head().to_dict()}")
                            logging.info(f"Previous month aggregated values: {second_last_month_grouped.head().to_dict()}")
                            
                            # Merge the data to calculate percent change
                            merged_data = pd.merge(
                                last_month_grouped, 
                                second_last_month_grouped,
                                on='supervisor_district', 
                                suffixes=('_current', '_previous')
                            )
                            
                            logging.info(f"Merged data for percent change calculation: {merged_data.to_dict()}")
                            
                            delta_map_data = []
                            for _, row in merged_data.iterrows():
                                if pd.isna(row['supervisor_district']):
                                    logging.info(f"Skipping NaN district value")
                                    continue
                                
                                try:
                                    # Handle both numeric and text district values
                                    district_value = row['supervisor_district']
                                    if pd.api.types.is_numeric_dtype(type(district_value)) or (isinstance(district_value, str) and district_value.replace('.', '', 1).isdigit()):
                                        # It's a numeric value (or numeric as string) - convert to int if possible
                                        try:
                                            district = str(int(float(district_value)))
                                        except:
                                            district = str(district_value)
                                    else:
                                        # It's a non-numeric string
                                        district = str(district_value)
                                        # If it's a string like "District 1", extract just the number
                                        district_match = re.search(r'district\s*(\d+)', district.lower())
                                        if district_match:
                                            district = district_match.group(1)
                                    
                                    current_value = row[f"{value_field}_current"]
                                    previous_value = row[f"{value_field}_previous"]
                                    
                                    # Calculate absolute delta and percent change
                                    delta = current_value - previous_value
                                    if previous_value != 0:
                                        percent_change = ((current_value - previous_value) / previous_value)
                                    else:
                                        percent_change = 0 if current_value == 0 else 1.0
                                    
                                    # Store enhanced data format for delta maps
                                    delta_map_data.append({
                                        "district": district,
                                        "current_value": current_value,
                                        "previous_value": previous_value,
                                        "delta": delta,
                                        "percent_change": percent_change,
                                        "value": percent_change  # For backward compatibility and coloring
                                    })
                                    logging.info(f"District {district}: Current={current_value}, Previous={previous_value}, Delta={delta}, Change={percent_change:.1%}")
                                except (ValueError, TypeError) as e:
                                    logging.warning(f"Skipping invalid district for percent change: {row['supervisor_district']} - {e}")
                            
                            # Log the final delta map data
                            logging.info(f"Final delta map data: {delta_map_data}")
                            
                            # Generate delta map
                            if delta_map_data:
                                map_title = f"{query_name} - Changes by District from {second_last_month_display} to {last_month_display}"
                                
                                # Use enhanced data format for delta maps (not CSV conversion)
                                # This will trigger the enhanced tooltip format with current_value, previous_value, delta, percent_change
                                map_result = generate_map(
                                    context_variables={},
                                    map_title=map_title,
                                    map_type="supervisor_district",
                                    location_data=delta_map_data,  # Pass the enhanced list format directly
                                    map_metadata={
                                        "period": f"{second_last_month_display} to {last_month_display}",
                                        "description": f"Percent change in {query_name} by supervisor district",
                                        "map_type": "delta"
                                    },
                                    metric_id=metric_id,
                                    group_field="supervisor_district"
                                )
                                
                                if map_result and map_result.get("map_id"):
                                    map_id = map_result.get('map_id')
                                    logging.info(f"Successfully generated delta map with ID: {map_id}")
                                    
                                    # Store map ID and URL for markdown reference
                                    generated_maps['delta_map_id'] = map_id
                                    generated_maps['delta_map_url'] = map_result.get('publish_url')
                                    
                                    # Add map to HTML content
                                    if "publish_url" in map_result:
                                        map_html = f"""
                                        <div class="map-container">
                                            <h3>District Map: {map_title}</h3>
                                            <iframe width="100%" height="500" src="{map_result['publish_url']}" 
                                                frameborder="0" allowfullscreen></iframe>
                                        </div>
                                        """
                                        all_html_contents.append(map_html)
                                else:
                                    logging.error(f"Failed to generate delta map: {map_result}")
        except Exception as e:
            logging.error(f"Error generating choropleth maps: {str(e)}")
            logging.error(traceback.format_exc())
        
        # Add markdown reference for the maps
        map_markdown = ""
        if generated_maps['density_map_id']:
            map_markdown += f"\n\n### District Map References\n\n"
            map_markdown += f"* **Current Values Map**: Map ID: `{generated_maps['density_map_id']}`"
            if generated_maps['density_map_url']:
                map_markdown += f" - [View Map]({generated_maps['density_map_url']})"
            
            if generated_maps['delta_map_id']:
                map_markdown += f"\n* **Percent Change Map**: Map ID: `{generated_maps['delta_map_id']}`"
                if generated_maps['delta_map_url']:
                    map_markdown += f" - [View Map]({generated_maps['delta_map_url']})"
            
            all_markdown_contents.append(map_markdown)
    
    # Process district-specific analysis if needed
    if process_districts and has_district and 'supervisor_district' in dataset.columns:
        # Get list of districts in the dataset
        districts = dataset['supervisor_district'].dropna().unique()
        logging.info(f"Found {len(districts)} districts in dataset: {districts}")
        
        # Process each district
        for district in districts:
            try:
                # Skip non-numeric or invalid districts
                try:
                    district_num = int(district)
                    if district_num < 1 or district_num > 11:
                        logging.warning(f"Skipping invalid district number: {district}")
                        continue
                except (ValueError, TypeError):
                    logging.warning(f"Skipping non-numeric district: {district}")
                    continue
                
                logging.info(f"Processing district {district}")
                
                # Create district-specific filter conditions by copying the original ones
                district_filter_conditions = filter_conditions.copy()
                district_filter_conditions.append({
                    'field': 'supervisor_district',
                    'operator': '=',
                    'value': str(district)
                })
                
                # Process analysis for this district
                # For district 0 (citywide), keep supervisor_district as a category field
                if district == 0:
                    district_category_fields = category_fields
                else:
                    # For other districts, remove supervisor_district from category fields
                    district_category_fields = [f for f in category_fields if not ((isinstance(f, dict) and f.get('fieldName') == 'supervisor_district') 
                                    or f == 'supervisor_district')]
                
                district_result = process_single_analysis(
                    context_variables=context_variables.copy(),
                    category_fields=district_category_fields,
                    period_type=period_type,
                    period_field=period_field,
                    filter_conditions=district_filter_conditions,
                    query_name=f"{query_name} - District {district}",
                    period_desc=f"{period_desc} - District {district}",
                    value_field=value_field,
                    recent_period=recent_period,
                    comparison_period=comparison_period,
                    uses_avg=uses_avg,
                    agg_functions=agg_functions,
                    district=district,
                    metric_id=metric_id,
                    base_metric_name=metric_info.get('query_name', metric_id)
                )
                
                if district_result:
                    result = {
                        'query_name': f"{query_name} - District {district}",
                        'period_type': period_type,
                        'markdown': district_result.get('markdown', ''),
                        'html': district_result.get('html', ''),
                        'metric_id': metric_id
                    }
                    
                    # Save the analysis files for this district
                    save_analysis_files(result, metric_id, period_type, district=district)
            except Exception as e:
                logging.error(f"Error processing district {district}: {e}")
                logging.error(traceback.format_exc())
    
    return process_analysis_result

def clean_metric_name(query_name):
    """Clean the metric name by removing emojis and formatting it properly."""
    # Remove emojis and other special characters
    cleaned = re.sub(r'[^\w\s-]', '', query_name)
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Capitalize first letter of each word
    cleaned = ' '.join(word.capitalize() for word in cleaned.split())
    return cleaned

def process_single_analysis(context_variables, category_fields, period_type, period_field, 
                           filter_conditions, query_name, period_desc, value_field,
                           recent_period, comparison_period, uses_avg, agg_functions, district=None, metric_id=None, base_metric_name=None):
    """Process a single analysis with the given parameters."""
    # Get a copy of the dataset to avoid modifying the original
    dataset = context_variables['dataset'].copy()
    
    # Make a deep copy of context variables to preserve all nested data
    context = copy.deepcopy(context_variables)
    context['dataset'] = dataset
    
    # Clean the metric name for the y-axis label
    y_axis_label = clean_metric_name(query_name)
    context['y_axis_label'] = y_axis_label
    
    # Convert recent and comparison periods to string format for anomaly detection
    string_recent_period = {
        'start': recent_period['start'].strftime('%Y-%m-%d'),
        'end': recent_period['end'].strftime('%Y-%m-%d')
    }
    
    string_comparison_period = {
        'start': comparison_period['start'].strftime('%Y-%m-%d'),
        'end': comparison_period['end'].strftime('%Y-%m-%d')
    }
    
    # Initialize lists to store markdown and HTML content
    all_markdown_contents = []
    all_html_contents = []
    
    # Generate time series chart first
    logging.info(f"Generating main time series chart for {query_name}")
    main_chart_title = f'{query_name} <br> {period_desc} Trend'
    
    # Create a separate context for the main chart
    main_context = context.copy()
    main_context['chart_title'] = main_chart_title
    main_context['noun'] = query_name
    
    try:
        # Use period_field for time series chart
        chart_result = generate_time_series_chart(
            context_variables=main_context,
            time_series_field=period_field,
            numeric_fields=value_field,
            aggregation_period=period_type,
            max_legend_items=10,
            filter_conditions=filter_conditions,
            show_average_line=True,
            agg_functions=agg_functions,
            return_html=True,
            output_dir=ANNUAL_DIR if period_type == 'year' else MONTHLY_DIR,
            store_in_db=True,  # Enable database storage
            object_type='dashboard_metric',  # Add object type
            object_id=metric_id,  # Add object ID (metric_id)
            object_name=base_metric_name if base_metric_name else query_name # Use base name if available
        )
        
        logging.info(f"Successfully generated main time series chart for {query_name}")
        
        # Append chart HTML to content lists if available
        if chart_result:
            if isinstance(chart_result, tuple):
                markdown_content, html_content = chart_result
                all_markdown_contents.append(markdown_content)
                all_html_contents.append(html_content)
            elif isinstance(chart_result, dict) and 'html' in chart_result:
                all_html_contents.append(chart_result['html'])
            else:
                all_html_contents.append(str(chart_result))
        else:
            logging.warning(f"No main chart result returned for {query_name}")
    except Exception as e:
        logging.error(f"Error generating main time series chart for {query_name}: {str(e)}")
        logging.error(traceback.format_exc())
    
    # Process by category fields
    for category_field in category_fields:
        # Get the actual field name
        if isinstance(category_field, dict):
            category_field_name = category_field.get('fieldName', '')
            category_field_display = category_field.get('name', category_field_name)
        else:
            category_field_name = category_field
            category_field_display = category_field
        
        # Skip if category field is not in dataset
        if category_field_name not in dataset.columns:
            logging.warning(f"Category field '{category_field_name}' not found in dataset for {query_name}")
            continue
        
        # Skip supervisor_district if we're doing district-specific analysis (but not for district 0/citywide)
        if district is not None and district != 0 and category_field_name == 'supervisor_district':
            logging.info(f"Skipping supervisor_district category for district-specific analysis (district {district})")
            continue
        
        logging.info(f"Processing category field: {category_field_name} for {query_name}")
        
        try:
            # Generate time series chart for each category field
            chart_title = f"{query_name} <br> {value_field} by {period_desc} by {category_field_display}"
            category_context = context.copy()
            category_context['chart_title'] = chart_title
            
            # Generate the chart with grouping by category field
            cat_chart_result = generate_time_series_chart(
                context_variables=category_context,
                time_series_field=period_field,
                numeric_fields=value_field,
                aggregation_period=period_type,
                max_legend_items=10,
                group_field=category_field_name,
                filter_conditions=filter_conditions,
                show_average_line=False,
                agg_functions=agg_functions,
                return_html=True,
                output_dir=ANNUAL_DIR if period_type == 'year' else MONTHLY_DIR,
                store_in_db=True,  # Enable database storage
                object_type='dashboard_metric_category',  # Add object type with category suffix
                object_id=metric_id,  # Use just the metric ID
                object_name=f"{base_metric_name if base_metric_name else query_name} by {category_field_display}" # Use base name if available
            )
            
            logging.info(f"Successfully generated chart for {category_field_name} for {query_name}")
            
            # Append chart result to content lists
            if cat_chart_result:
                if isinstance(cat_chart_result, tuple):
                    markdown_content, html_content = cat_chart_result
                    all_markdown_contents.append(markdown_content)
                    all_html_contents.append(html_content)
                elif isinstance(cat_chart_result, dict) and 'html' in cat_chart_result:
                    all_html_contents.append(cat_chart_result['html'])
                else:
                    all_html_contents.append(str(cat_chart_result))
            else:
                logging.warning(f"No chart result returned for {category_field_name} for {query_name}")
        except Exception as e:
            logging.error(f"Error generating chart for {category_field_name} for {query_name}: {str(e)}")
            logging.error(traceback.format_exc())
        
        try:
            # Detect anomalies
            anomaly_results = anomaly_detection(
                context_variables=context,
                group_field=category_field_name,
                filter_conditions=filter_conditions,
                min_diff=2,
                recent_period=string_recent_period,
                comparison_period=string_comparison_period,
                date_field=period_field,
                numeric_field=value_field,
                y_axis_label=y_axis_label,
                title=f"{query_name} - {value_field} by {category_field_display}",
                period_type=period_type,
                agg_function='mean' if uses_avg else 'sum',
                output_dir='monthly' if period_type == 'month' else 'annual',
                object_type='dashboard_metric',
                object_id=metric_id,  # Ensure we're using just the metric_id
                object_name=query_name
            )
            
            # Get markdown and HTML content from anomaly results
            if anomaly_results:
                markdown_content = anomaly_results.get('markdown', anomaly_results.get('anomalies_markdown', 'No anomalies detected.'))
                html_content = anomaly_results.get('html', anomaly_results.get('anomalies', 'No anomalies detected.'))
                
                # Append content to lists
                all_markdown_contents.append(markdown_content)
                all_html_contents.append(html_content)
        except Exception as e:
            logging.error(f"Error detecting anomalies for {category_field_name} for {query_name}: {str(e)}")
            logging.error(traceback.format_exc())
    
    # Get endpoint and data_sf_url from context if available
    endpoint = context_variables.get('endpoint', '')
    data_sf_url = context_variables.get('data_sf_url', '')
    
    # Include metric_id in the title
    metric_title = metric_id if metric_id else query_name
    
    # Combine all markdown content
    query_string = ", ".join([f"{cond['field']} {cond['operator']} {cond['value']}" for cond in filter_conditions])
    combined_markdown = f"# {metric_title} - {query_name}\n\n**Analysis Type:** {period_desc}\n\n**Filters:** {query_string}\n\n{''.join(all_markdown_contents)}"
    
    # Combine all HTML content - ensure all items are strings
    html_content_strings = []
    for content in all_html_contents:
        if isinstance(content, str):
            html_content_strings.append(content)
        elif isinstance(content, tuple):
            html_content_strings.append(content[0] if content else "")
        else:
            html_content_strings.append(str(content) if content else "")
    
    combined_html = f"<h1>{metric_title} - {query_name}</h1>\n<p><strong>Analysis Type:</strong> {period_desc}</p>\n<p><strong>Filters:</strong> {query_string}</p>\n{''.join(html_content_strings)}"
    
    # Create result dictionary
    result = {
        'query_name': query_name,
        'period_type': period_type,
        'markdown': combined_markdown,
        'html': combined_html,
        'metric_id': metric_id
    }
    
    return result

def extract_date_field_from_query(query):
    """Extract the date field from a query."""
    date_fields_to_check = [
        'date', 'incident_date', 'report_date', 'arrest_date', 'received_datetime', 
        'Report_Datetime', 'disposition_date', 'dba_start_date'
    ]
    
    for field in date_fields_to_check:
        if field in query:
            logging.info(f"Found date field in query: {field}")
            return field
    
    # Try to find date_trunc patterns
    date_trunc_match = re.search(r'date_trunc_[ymd]+ *\( *([^\)]+) *\)', query)
    if date_trunc_match:
        field = date_trunc_match.group(1).strip()
        logging.info(f"Found date field from date_trunc: {field}")
        return field
    
    return None

def determine_date_field_name(query, date_field, period_type):
    """Determine the appropriate date field name based on query and period type."""
    if 'date_trunc_ymd' in query:
        return 'day'
    elif 'date_trunc_ym' in query:
        return 'month'
    elif 'date_trunc_y' in query:
        return 'year'
    elif 'as date' in query.lower():
        return 'date'
    elif date_field in ['year', 'month', 'day'] and date_field != period_type:
        return period_type
    else:
        return date_field

def transform_query_for_period(original_query, date_field, category_fields, period_type, recent_period, comparison_period, district=None):
    """
    Transform a query for monthly or annual analysis by:
    1. Replacing date placeholders
    2. Using appropriate date ranges for recent and comparison periods
    3. Adding category fields to GROUP BY
    4. Creating appropriate period fields
    5. Adding district filter if specified
    
    Args:
        original_query (str): The original SQL query
        date_field (str): The name of the date field
        category_fields (list): List of category fields
        period_type (str): 'month' or 'year'
        recent_period (dict): Recent period date range
        comparison_period (dict): Comparison period date range
        district (int, optional): District number to filter by
        
    Returns:
        str: Transformed SQL query
    """
    # Format date strings for SQL
    recent_start = recent_period['start'].isoformat()
    recent_end = recent_period['end'].isoformat()
    comparison_start = comparison_period['start'].isoformat()
    comparison_end = comparison_period['end'].isoformat()
    
    # Replace any date placeholders in the original query
    # We make a copy to avoid modifying the original query while iterating
    modified_query = original_query
    replacements = {
        'this_year_start': f"'{recent_start}'",
        'this_year_end': f"'{recent_end}'",
        'last_year_start': f"'{comparison_start}'",
        'last_year_end': f"'{comparison_end}'",
        'start_date': f"'{comparison_start}'",
        'current_date': f"'{recent_end}'"
    }
    
    # Apply replacements correctly - ensure we're not creating malformed field names
    for placeholder, value in replacements.items():
        # Make sure we're only replacing standalone instances of the placeholder
        # by checking for word boundaries or operators before/after
        modified_query = re.sub(r'([=<>:\s]|^)' + re.escape(placeholder) + r'([=<>:\s]|$)', 
                                r'\1' + value + r'\2', 
                                modified_query)
    
    # Ensure consistent use of <= instead of < for date upper bounds
    modified_query = re.sub(r'<\s*current_date', f"<= '{recent_end}'", modified_query)
    modified_query = re.sub(r'<\s*\'{recent_end}\'', f"<= '{recent_end}'", modified_query)
    
    # Determine if it's a YTD query by checking format
    is_ytd_query = ('as date, COUNT(*)' in modified_query or 
                   'as date,' in modified_query or 
                   'date_trunc_ymd' in modified_query)
    
    # If it's a YTD query, we'll modify it to work with our period types
    if is_ytd_query:
        logging.info("Using YTD query format as basis")
        
        # Extract the core table and WHERE conditions from the original query
        # This pattern looks for date_trunc, field selection, conditions
        ytd_pattern = r'SELECT\s+date_trunc_[ymd]+\((.*?)\)\s+as\s+date,\s+([^W]+)WHERE\s+(.*?)(?:GROUP BY|ORDER BY|$)'
        ytd_match = re.search(ytd_pattern, modified_query, re.IGNORECASE | re.DOTALL)
        
        if ytd_match:
            date_field_match = ytd_match.group(1).strip()
            value_part = ytd_match.group(2).strip()
            where_part = ytd_match.group(3).strip()
            
            # Remove current_date references and replace with our recent_end
            where_part = re.sub(r'<=\s*current_date', f"<= '{recent_end}'", where_part)
            # Also handle the case where "< current_date" is used instead of "<= current_date"
            where_part = re.sub(r'<\s*current_date', f"<= '{recent_end}'", where_part)
            
            # Generate appropriate date_trunc based on period_type
            if period_type == 'year':
                date_trunc = f"date_trunc_y({date_field_match})"
                period_field = "year_period"
            else:  # month
                date_trunc = f"date_trunc_ym({date_field_match})"
                period_field = "month_period"
            
            # Build the category fields part of the SELECT and GROUP BY
            category_select = ""
            group_by_fields = []
            
            for field in category_fields:
                if isinstance(field, dict):
                    field_name = field.get('fieldName', '')
                else:
                    field_name = field
                
                if field_name:
                    # Handle supervisor_district specially
                    if field_name == 'supervisor_district':
                        category_select += f", CASE WHEN supervisor_district IS NOT NULL THEN supervisor_district ELSE NULL END as supervisor_district"
                        group_by_fields.append("supervisor_district")
                    else:
                        category_select += f", {field_name}"
                        group_by_fields.append(field_name)
            
            # Add period_type to distinguish recent from comparison - FIXED POSITION
            period_type_select = f", CASE WHEN {date_field_match} >= '{recent_start}' AND {date_field_match} <= '{recent_end}' THEN 'recent' ELSE 'comparison' END as period_type"
            
            # Build the complete transformed query
            group_by_clause = f"GROUP BY {period_field}, period_type"
            if group_by_fields:
                group_by_clause += ", " + ", ".join(group_by_fields)
                
            transformed_query = f"""
            SELECT 
                {date_trunc} as {period_field},
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
            ORDER BY {period_field}
            """
            
            # Make sure we're using <= consistently by directly replacing potential leftover < operators
            transformed_query = transformed_query.replace(f"{date_field_match} < '{recent_end}'", f"{date_field_match} <= '{recent_end}'")
            
            return transformed_query
        else:
            # If we can't parse the YTD query, fall back to the modified query
            return modified_query
    
    # For non-YTD queries, check if the query already has a date_trunc function for the period
    if period_type == 'month' and 'date_trunc_ym' in modified_query:
        # Just use the modified query with replaced placeholders
        logging.info("Query already has date_trunc_ym, using original query with replaced placeholders")
        return modified_query
    elif period_type == 'year' and 'date_trunc_y' in modified_query:
        # Just use the modified query with replaced placeholders
        logging.info("Query already has date_trunc_y, using original query with replaced placeholders")
        return modified_query
    
    # Special case for the police incidents query
    if 'Report_Datetime' in modified_query and 'supervisor_district' in modified_query:
        # We know this is the police incidents query
        logging.info("Detected police incidents query format")
        
        # Build the category fields part of the SELECT clause
        category_select = ""
        category_fields_list = []  # To track field names for GROUP BY
        
        for field in category_fields:
            if isinstance(field, dict):
                field_name = field.get('fieldName', '')
            else:
                field_name = field
            
            if field_name:
                # Handle supervisor_district specially
                if field_name == 'supervisor_district':
                    category_select += f", CASE WHEN supervisor_district IS NOT NULL THEN supervisor_district ELSE NULL END as supervisor_district"
                    category_fields_list.append("supervisor_district")
                else:
                    category_select += f", {field_name}"
                    category_fields_list.append(field_name)
        
        # Build the date transformation part based on period_type
        if period_type == 'month':
            date_transform = f"date_trunc_ym(Report_Datetime) as month_period"
        else:  # year
            date_transform = f"date_trunc_y(Report_Datetime) as year_period"
        
        # Build the GROUP BY clause with category fields
        group_by = f"GROUP BY {period_type}_period"
        for field_name in category_fields_list:
            group_by += f", {field_name}"
        
        # Log the GROUP BY clause
        logging.info(f"GROUP BY clause: {group_by}")
        
        # Build the date range filter - include both recent and comparison periods
        # We need to include both periods but also add a flag to distinguish them
        date_range = f"""
        WHERE (
            (Report_Datetime >= '{comparison_start}' AND Report_Datetime <= '{comparison_end}')
            OR 
            (Report_Datetime >= '{recent_start}' AND Report_Datetime <= '{recent_end}')
        )
        """
        
        # Add district filter if specified
        if district is not None:
            date_range += f" AND supervisor_district = '{district}'"
            logging.info(f"Added district filter to query: supervisor_district = '{district}'")
        
        # Build the complete transformed query - simplified to just count records
        # Note: In Socrata API, we don't need a FROM clause
        transformed_query = f"""
        SELECT 
            {date_transform},
            COUNT(*) as value,
            CASE 
                WHEN Report_Datetime >= '{recent_start}' AND Report_Datetime <= '{recent_end}' THEN 'recent'
                ELSE 'comparison'
            END as period_type
            {category_select}
        {date_range}
        {group_by}, period_type
        ORDER BY {period_type}_period
        """
        
        return transformed_query
    
    # Try to extract the FROM clause
    from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', modified_query, re.IGNORECASE | re.DOTALL)
    
    # If FROM clause not found, try to infer it from the query
    if not from_match:
        # Check if there's a table name after SELECT
        table_match = re.search(r'SELECT.*?FROM\s+([^\s,]+)', modified_query, re.IGNORECASE)
        if table_match:
            from_clause = table_match.group(1).strip()
        else:
            logging.warning("Could not extract FROM clause from query, using modified query with replaced placeholders")
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
        
        # Add district filter if specified
        if district is not None and 'supervisor_district' in modified_query:
            where_clause = where_clause.rstrip() + f" AND supervisor_district = '{district}'\n"
            logging.info(f"Added district filter to WHERE clause: supervisor_district = '{district}'")
    else:
        # Create a new WHERE clause with just date filters for both periods
        where_clause = f"""
        WHERE (
            ({date_field} >= '{comparison_start}' AND {date_field} <= '{comparison_end}')
            OR 
            ({date_field} >= '{recent_start}' AND {date_field} <= '{recent_end}')
        )
        """
        
        # Add district filter if specified
        if district is not None and 'supervisor_district' in modified_query:
            where_clause = where_clause.rstrip() + f" AND supervisor_district = '{district}'\n"
            logging.info(f"Added district filter to new WHERE clause: supervisor_district = '{district}'")
    
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
                # Handle supervisor_district specially
                if field_name == 'supervisor_district':
                    category_select += f", CASE WHEN supervisor_district IS NOT NULL THEN supervisor_district ELSE NULL END as supervisor_district"
                    category_fields_list.append("supervisor_district")
                else:
                    category_select += f", {field_name}"
                    category_fields_list.append(field_name)
        
        # Build the date transformation part based on period_type
        if period_type == 'month':
            date_transform = f"date_trunc_ym({date_field}) as month_period"
        else:  # year
            date_transform = f"date_trunc_y({date_field}) as year_period"
        
        # Build the GROUP BY clause with category fields
        group_by = f"GROUP BY {period_type}_period"
        for field_name in category_fields_list:
            group_by += f", {field_name}"
        
        # Add period_type to distinguish recent from comparison
        period_type_select = f", CASE WHEN {date_field} >= '{recent_start}' AND {date_field} <= '{recent_end}' THEN 'recent' ELSE 'comparison' END as period_type"
        
        # Log the GROUP BY clause
        logging.info(f"GROUP BY clause: {group_by}, period_type")
        
        # Build the complete transformed query - simplified to just count records
        transformed_query = f"""
        SELECT 
            {date_transform},
            COUNT(*) as value
            {period_type_select}
            {category_select}
        FROM {from_clause}
        {where_clause}
        {group_by}, period_type
        ORDER BY {period_type}_period
        """
        
        return transformed_query
    else:
        # If we couldn't extract or infer the FROM clause, return the modified query
        logging.warning("Could not determine FROM clause, using modified query with replaced placeholders")
        return modified_query

def save_analysis_files(result, metric_id, period_type, output_dir=None, district=None):
    """Save analysis results to markdown files. HTML files are not saved."""
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create subdirectories for different period types - only use monthly and annual
    period_dirs = {
        'month': 'monthly',
        'year': 'annual'
    }
    
    for dir_name in period_dirs.values():
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    # Map period_type to directory name
    dir_name = period_dirs.get(period_type, 'other')
    
    # Use result's metric_id if available, otherwise fallback to the provided metric_id
    file_metric_id = result.get('metric_id', metric_id)
    
    # Ensure we have a valid metric_id
    if not file_metric_id or file_metric_id.strip() == '':
        # Generate a sanitized metric ID from the query name if no metric_id is available
        query_name = result.get('query_name', '')
        file_metric_id = query_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        logging.warning(f"Missing metric_id for {query_name}, using sanitized query name: {file_metric_id}")
    
    # Create district subfolder using the district number - 0 for citywide
    if district is not None:
        district_dir = os.path.join(output_dir, dir_name, f"{district}")
        os.makedirs(district_dir, exist_ok=True)
        output_path = district_dir
    else:
        # Default to folder 0 if no district specified (treating as citywide)
        district_dir = os.path.join(output_dir, dir_name, "0")
        os.makedirs(district_dir, exist_ok=True)
        output_path = district_dir
    
    # Generate filename - just metric_id.md
    md_filename = f"{file_metric_id}.md"
    md_path = os.path.join(output_path, md_filename)
    
    # Log the file path being used
    logging.info(f"Saving analysis to: {md_path}")
    
    # Get the markdown content
    markdown_content = result.get('markdown', '')
    
    # Write markdown file
    with open(md_path, 'w') as f:
        f.write(markdown_content)
    
    # Get district description based on district value
    if district == 0 or district is None:
        district_info = " for Citywide"
    else:
        district_info = f" for District {district}"
        
    logging.info(f"Saved {period_type} analysis for {file_metric_id}{district_info} to {md_path}")
    
    return {
        'md_path': md_path
    }

def main():
    """Main function to generate metric analysis."""
    parser = argparse.ArgumentParser(description='Generate analysis for a specific metric')
    parser.add_argument('metric_id', help='ID of the metric to analyze (e.g., "arrests_presented_to_da_ytd")')
    parser.add_argument('--period', '-p', choices=['monthly', 'annual', 'both'], default='both',
                        help='Period type for analysis: monthly (24 months lookback), annual (10 years lookback), or both (default)')
    parser.add_argument('--process-districts', action='store_true', 
                        help='Process and generate separate reports for each supervisor district if available')
    args = parser.parse_args()
    
    metric_id = args.metric_id
    period_choice = args.period
    process_districts = args.process_districts
    
    # Verify metric_id is not empty
    if not metric_id or metric_id.strip() == '':
        logger.error("metric_id cannot be empty")
        return
    
    # Log the metric ID we're processing
    logger.info(f"Processing metric ID: '{metric_id}'")
    
    # Create output directories - only create monthly and annual
    for dir_name in ['monthly', 'annual']:
        os.makedirs(os.path.join(OUTPUT_DIR, dir_name), exist_ok=True)
    
    # Load dashboard queries
    dashboard_queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dashboard", "dashboard_queries.json")
    enhanced_queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dashboard", "dashboard_queries_enhanced.json")
    
    # Log the paths we're trying to use
    logger.info(f"Looking for enhanced queries at: {enhanced_queries_path}")
    logger.info(f"Looking for standard queries at: {dashboard_queries_path}")
    
    # Try to load enhanced queries first, fall back to regular queries
    if os.path.exists(enhanced_queries_path):
        logger.info("Found enhanced dashboard queries file")
        dashboard_queries = load_json_file(enhanced_queries_path)
        if dashboard_queries:
            logger.info("Successfully loaded enhanced dashboard queries")
        else:
            logger.error("Failed to load enhanced dashboard queries")
    else:
        logger.warning(f"Enhanced queries file not found at {enhanced_queries_path}")
        logger.info("Using standard dashboard queries")
        dashboard_queries = load_json_file(dashboard_queries_path)
        if dashboard_queries:
            logger.info("Successfully loaded standard dashboard queries")
        else:
            logger.error("Failed to load standard dashboard queries")
    
    if not dashboard_queries:
        logger.error("Failed to load dashboard queries")
        return
    
    # Log the structure of the loaded queries
    logger.info(f"Loaded queries structure: {list(dashboard_queries.keys())}")
    
    # Find the metric in the queries
    metric_info = find_metric_in_queries(dashboard_queries, metric_id)
    if not metric_info:
        logger.error(f"Metric with ID '{metric_id}' not found in dashboard queries")
        return
    
    # Log the found metric info
    logger.info(f"Found metric: {metric_info.get('query_name')} with ID {metric_info.get('metric_id')} in category {metric_info.get('top_category')}")
    
    # Make sure metric_id is in the metric_info
    if 'metric_id' not in metric_info:
        metric_info['metric_id'] = metric_id
    
    # Log whether we're processing districts
    if process_districts:
        logger.info(f"Found metric: {metric_info['query_name']} with ID {metric_info['metric_id']} in category {metric_info['top_category']} - will process districts if available")
    else:
        logger.info(f"Found metric: {metric_info['query_name']} with ID {metric_info['metric_id']} in category {metric_info['top_category']}")
    
    # Process the metric based on the selected period type
    analysis_results = []
    
    # Monthly analysis (24 months lookback)
    if period_choice in ['monthly', 'both']:
        logger.info(f"Processing monthly analysis for metric ID: {metric_info['metric_id']}")
        monthly_analysis = process_metric_analysis(metric_info, period_type='month', process_districts=process_districts)
        if monthly_analysis:
            analysis_results.append(monthly_analysis)
    
    # Annual analysis (10 years lookback)
    if period_choice in ['annual', 'both']:
        logger.info(f"Processing annual analysis for metric ID: {metric_info['metric_id']}")
        annual_analysis = process_metric_analysis(metric_info, period_type='year', process_districts=process_districts)
        if annual_analysis:
            analysis_results.append(annual_analysis)
    
    # The save_analysis_files is now called within process_metric_analysis
    logger.info(f"Analysis complete for metric: {metric_info['metric_id']}")

if __name__ == "__main__":
    main() 
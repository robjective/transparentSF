"""
Analysis Engine Module for Weekly Analysis

This module contains the core analysis processing logic including
data processing, chart generation, and anomaly detection.
"""

import os
import logging
import traceback
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

from tools.data_fetcher import set_dataset
from tools.genChart import generate_time_series_chart
from tools.anomaly_detection import anomaly_detection

from .data_processor import (
    extract_date_field_from_query,
    transform_query_for_weekly,
    detect_avg_aggregation
)
from .time_utils import get_weekly_time_ranges

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

# Uvicorn will handle the logging configuration for this module
# No need to manually configure file handlers here
logger.info(f"Weekly analysis logging initialized with level: {log_level_str}")


def process_weekly_analysis(metric_info, process_districts=False):
    """
    Process weekly analysis for a given metric
    
    Args:
        metric_info (dict): Metric information including query data
        process_districts (bool): Whether to process district-level data
        
    Returns:
        dict: Analysis results
    """
    
    # Extract query data
    if not metric_info:
        logger.error("No metric info provided")
        return None
    
    # Get the query name from metric_info, checking both 'name' and 'query_name' fields
    query_name = metric_info.get('name', metric_info.get('query_name', 'Unknown Metric'))
    endpoint = metric_info.get('endpoint', '')
    district_field = metric_info.get('district_field', 'supervisor_district')
    
    # Initialize the context variables for storing data
    context_variables = {}
    
    # Define default metrics if needed
    if 'category_fields' in metric_info:
        category_fields = metric_info['category_fields']
    else:
        # Default category fields for analysis - use empty list since we get fields from database
        category_fields = []
    
    # Determine the reference date for time range calculations.  If the metric
    # includes a `most_recent_data_date` value, we want the **last fully
    # completed week _prior_ to that date**.  Otherwise we fall back to using
    # the current date.

    most_recent_str = metric_info.get('most_recent_data_date') or metric_info.get('query_data', {}).get('most_recent_data_date')

    if most_recent_str:
        try:
            most_recent_dt = datetime.strptime(str(most_recent_str), "%Y-%m-%d").date()
            reference_date = most_recent_dt - timedelta(days=1)  # ensure we look at week _before_ this date
            # Get the most recent ISO week (Monday-Sunday)
            recent_period, _ = get_weekly_time_ranges(reference_date)
            # Calculate the 12 weeks before the recent week
            comparison_start = recent_period['start'] - timedelta(weeks=12)
            comparison_end = recent_period['start'] - timedelta(days=1)
            comparison_period = {'start': comparison_start, 'end': comparison_end}
            logger.info(f"Using most_recent_data_date {most_recent_dt} to set time ranges: recent={recent_period['start']}–{recent_period['end']}, comparison={comparison_period['start']}–{comparison_period['end']}")
        except Exception as e:
            logger.warning(f"Failed to parse most_recent_data_date '{most_recent_str}' – falling back to today. Error: {e}")
            recent_period, _ = get_weekly_time_ranges()
            comparison_start = recent_period['start'] - timedelta(weeks=12)
            comparison_end = recent_period['start'] - timedelta(days=1)
            comparison_period = {'start': comparison_start, 'end': comparison_end}
    else:
        # Get date ranges for recent and comparison periods based on today's date
        recent_period, _ = get_weekly_time_ranges()
        comparison_start = recent_period['start'] - timedelta(weeks=12)
        comparison_end = recent_period['start'] - timedelta(days=1)
        comparison_period = {'start': comparison_start, 'end': comparison_end}
    
    # Extract query data based on the structure
    if 'query_data' in metric_info and isinstance(metric_info['query_data'], dict):
        # Try to find YTD query first as it's usually better for weekly analysis
        if 'ytd_query' in metric_info['query_data']:
            original_query = metric_info['query_data'].get('ytd_query', '')
            logger.info(f"Using YTD query as the basis for weekly analysis")
        # Also check for queries dictionary that might contain a YTD query
        elif 'queries' in metric_info['query_data'] and isinstance(metric_info['query_data']['queries'], dict):
            if 'ytd_query' in metric_info['query_data']['queries']:
                original_query = metric_info['query_data']['queries'].get('ytd_query', '')
                logger.info(f"Using YTD query from queries dictionary")
            elif 'executed_ytd_query' in metric_info['query_data']['queries']:
                original_query = metric_info['query_data']['queries'].get('executed_ytd_query', '')
                logger.info(f"Using executed YTD query from queries dictionary")
        
        # If no YTD query is found, fall back to the metric query
        if not original_query:
            original_query = metric_info['query_data'].get('metric_query', '')
            logger.info(f"No YTD query found, using regular metric query")
    else:
        original_query = metric_info.get('query_data', '')
        logger.info(f"Using provided query data directly")
    
    if not original_query:
        logger.error(f"No query found for {query_name}")
        return None
    
    logger.info(f"Original query: {original_query}")
    
    # Check if the query uses AVG() aggregation
    uses_avg = detect_avg_aggregation(original_query)
    logger.info(f"Query uses AVG() aggregation: {uses_avg}")
    
    # Define value_field
    value_field = 'value'  # Default
    
    # Handle category fields
    if not category_fields:
        category_fields = []
        logger.info("No category fields defined for this metric. Not using any default fields.")
    
    # Check if supervisor_district exists in category_fields
    has_district = False
    for field in category_fields:
        if isinstance(field, dict) and field.get('fieldName') == district_field:
            has_district = True
            break
        elif field == district_field:
            has_district = True
            break
    
    # If processing districts and supervisor_district not in category_fields, add it
    if process_districts and not has_district:
        # Add supervisor_district as a category field
        category_fields.append('supervisor_district')
        logger.info("Added supervisor_district to category fields for district processing")
        has_district = True
    
    # Determine the date field to use from the query
    date_field = extract_date_field_from_query(original_query)
    if not date_field:
        logger.warning(f"No date field found in query for {query_name}")
        date_field = 'date'  # Default to 'date'
    
    logger.info(f"Using date field: {date_field}")
    
    # Transform the query for weekly analysis
    # For the main chart, exclude supervisor_district from category fields to get all data
    # UNLESS we're processing districts, in which case we need supervisor_district in the main query
    if process_districts and has_district:
        # Include supervisor_district when processing districts
        main_chart_category_fields = category_fields
        logger.info("Including supervisor_district in main query for district processing")
    else:
        # Exclude supervisor_district for citywide-only analysis
        main_chart_category_fields = [field for field in category_fields if 
                                     (isinstance(field, str) and field.lower() != 'supervisor_district') or 
                                     (isinstance(field, dict) and field.get('fieldName', '').lower() != 'supervisor_district')]
        logger.info("Excluding supervisor_district from main query for citywide analysis")
    
    transformed_query = transform_query_for_weekly(
        original_query=original_query,
        date_field=date_field,
        category_fields=main_chart_category_fields,  # Use filtered category fields for main chart
        recent_period=recent_period,
        comparison_period=comparison_period,
        district=None  # We'll handle district filtering later
    )
    
    logger.info(f"Transformed query: {transformed_query}")
    
    # Log the set_dataset call details
    logger.info(f"Calling set_dataset with endpoint: {endpoint}")
    
    # Set the dataset using the endpoint and transformed query
    result = set_dataset(context_variables=context_variables, endpoint=endpoint, query=transformed_query)
    
    if 'error' in result:
        logger.error(f"Error setting dataset for {query_name}: {result['error']}")
        # If error contains "no-such-column: supervisor_district", we can proceed without filtering by district
        if 'supervisor_district' in str(result.get('error', '')).lower() and 'no-such-column' in str(result.get('error', '')).lower():
            logger.warning(f"Supervisor district field not found in dataset. Proceeding without district filtering.")
            # Try again with a modified query without the supervisor_district field
            # Remove supervisor_district from category fields
            cleaned_category_fields = [field for field in category_fields if (isinstance(field, str) and field != 'supervisor_district') or 
                                      (isinstance(field, dict) and field.get('fieldName') != 'supervisor_district')]
            
            # Transform the query again without supervisor_district
            transformed_query = transform_query_for_weekly(
                original_query=original_query,
                date_field=date_field,
                category_fields=cleaned_category_fields,
                recent_period=recent_period,
                comparison_period=comparison_period,
                district=None
            )
            
            logger.info(f"Retrying with modified query without supervisor_district: {transformed_query}")
            result = set_dataset(context_variables=context_variables, endpoint=endpoint, query=transformed_query)
            
            if 'error' in result:
                logger.error(f"Error setting dataset (second attempt) for {query_name}: {result['error']}")
                return None
        else:
            return None
    
    # Get the dataset from context_variables
    if 'dataset' not in context_variables:
        logger.error(f"No dataset found in context for {query_name}")
        return None
    
    dataset = context_variables['dataset']
    
    # Log available columns in dataset
    logger.info(f"Available columns in dataset: {dataset.columns.tolist()}")
    
    # Check if supervisor_district is actually in the dataset columns
    dataset_columns = [col.lower() for col in dataset.columns.tolist()]
    if 'supervisor_district' not in dataset_columns and district_field.lower() not in dataset_columns:
        logger.warning(f"supervisor_district field not found in dataset columns. Removing it from category fields.")
        # Remove supervisor_district from category fields
        category_fields = [field for field in category_fields if (isinstance(field, str) and field.lower() != 'supervisor_district') or 
                          (isinstance(field, dict) and field.get('fieldName', '').lower() != 'supervisor_district')]
        # Set process_districts to False since we don't have district information
        process_districts = False
        has_district = False
    
    # Create or update value field if needed
    if value_field not in dataset.columns:
        if 'this_year' in dataset.columns:
            # Use this_year as the value field
            dataset[value_field] = dataset['this_year']
            logger.info(f"Created {value_field} from this_year column")
        elif dataset.select_dtypes(include=['number']).columns.tolist():
            # Use the first numeric column as the value field
            numeric_cols = [col for col in dataset.columns if 
                            col not in ['actual_date', 'date', 'period_type', 'day', 'week'] and
                            pd.api.types.is_numeric_dtype(dataset[col])]
            if numeric_cols:
                dataset[value_field] = dataset[numeric_cols[0]]
                logger.info(f"Created {value_field} from {numeric_cols[0]} column")
            else:
                # If no suitable numeric column, use 1 as the value
                dataset[value_field] = 1
                logger.info(f"Created {value_field} with default value 1")
        else:
            # If no numeric columns, use 1 as the value
            dataset[value_field] = 1
            logger.info(f"Created {value_field} with default value 1")
    
    # Make sure actual_date field exists (our time series field)
    time_field = 'actual_date'
    if time_field not in dataset.columns:
        if 'day' in dataset.columns:
            # If the query returned 'day' instead of 'actual_date', use that
            time_field = 'day'
            logger.info(f"Using 'day' field instead of 'actual_date'")
        elif 'max_date' in dataset.columns:
            try:
                # Convert max_date to datetime if it's not already
                dataset['max_date'] = pd.to_datetime(dataset['max_date'])
                
                # Calculate start of week (Sunday)
                dataset[time_field] = dataset['max_date']
                logger.info(f"Created '{time_field}' field from max_date column")
            except Exception as e:
                logger.error(f"Error creating '{time_field}' field from max_date: {e}")
                # Use current date's week start
                today = datetime.now()
                dataset[time_field] = today
                logger.info(f"Created '{time_field}' field with current week start")
        else:
            # Try to create actual_date field from any date-like column
            date_columns = [col for col in dataset.columns if 'date' in col.lower() and col != 'period_type']
            if date_columns:
                try:
                    date_col = date_columns[0]
                    dataset[time_field] = pd.to_datetime(dataset[date_col])
                    logger.info(f"Created '{time_field}' field from {date_col} column")
                except Exception as e:
                    logger.error(f"Error creating '{time_field}' field from {date_col}: {e}")
                    # Use current date's week start
                    today = datetime.now()
                    dataset[time_field] = today
                    logger.info(f"Created '{time_field}' field with current week start")
            else:
                # Use current date
                today = datetime.now()
                dataset[time_field] = today
                logger.info(f"No date columns found. Created '{time_field}' field with current date")
    
    # Ensure time field is in datetime format for processing
    try:
        dataset[time_field] = pd.to_datetime(dataset[time_field])
        logger.info(f"Converted {time_field} to datetime format for processing")
    except Exception as e:
        logger.error(f"Error converting {time_field} to datetime: {e}")
    
    # Post-process daily data into weekly data
    logger.info("Processing daily data for weekly analysis")
    logger.info(f"Dataset columns: {dataset.columns.tolist()}")
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Sample of time_field values: {dataset[time_field].head().tolist()}")
    
    # Ensure the value field is numeric
    if value_field in dataset.columns:
        dataset[value_field] = pd.to_numeric(dataset[value_field], errors='coerce')
        logger.info(f"Converted {value_field} to numeric type")
        context_variables['dataset'] = dataset
    
    # Define aggregation functions based on whether the query uses AVG()
    agg_functions = {value_field: 'mean'} if uses_avg else {value_field: 'sum'}
    logger.info(f"Using '{list(agg_functions.values())[0]}' aggregation for field {value_field}")
    
    # Validate category fields - ensure they exist in the dataset
    valid_category_fields = []
    for field in category_fields:
        if isinstance(field, dict):
            field_name = field.get('fieldName', '')
        else:
            field_name = field
            
        if field_name and field_name in dataset.columns:
            valid_category_fields.append(field)
            logger.info(f"Validated category field: {field_name}")
        else:
            logger.warning(f"Category field {field_name} not found in dataset. Ignoring.")
    
    # Update category_fields with only valid fields
    category_fields = valid_category_fields
    
    # Initialize lists to store HTML contents for embedding in the final output
    all_html_contents = []
    
    if time_field in dataset.columns:
        try:
            if pd.api.types.is_datetime64_dtype(dataset[time_field]) or isinstance(dataset[time_field].iloc[0], (datetime, pd.Timestamp)):
                # If it's already a datetime, keep it that way
                pass
            else:
                # Otherwise try to convert to datetime
                dataset[time_field] = pd.to_datetime(dataset[time_field])
        except Exception as e:
            logger.error(f"Error standardizing {time_field} field format: {e}")
    
    # Set up filter conditions for date filtering (using the actual time field)
    filter_conditions = [
        {
            'field': time_field,
            'operator': '>=',
            'value': comparison_period['start'].isoformat(),
            'is_date': True
        },
        {
            'field': time_field,
            'operator': '<=', 
            'value': recent_period['end'].isoformat(),
            'is_date': True
        }
    ]
    
    # Generate main time series chart
    logger.info(f"Generating main time series chart for {query_name}")
    main_chart_title = f'{query_name} <br> Weekly Trend'
    
    try:
        # For the main chart, pass the raw dataset to genChart
        # genChart will handle the weekly aggregation properly
        main_chart_df = dataset.copy()
        
        # DEBUG: Log the dataset details before charting
        logger.info(f"[DEBUG] Main chart dataset details:")
        logger.info(f"  Shape: {main_chart_df.shape}")
        logger.info(f"  Columns: {main_chart_df.columns.tolist()}")
        logger.info(f"  Sample data:\n{main_chart_df.head(10).to_string()}")
        if 'period_type' in main_chart_df.columns:
            logger.info(f"  Period type distribution:\n{main_chart_df['period_type'].value_counts()}")
        
        # Log the first 10 rows and dtypes before charting
        logger.info(f"[Main Chart] Data sample before charting:\n{main_chart_df.head(10).to_string()}")
        logger.info(f"[Main Chart] Data types before charting:\n{main_chart_df.dtypes}")
        
        # Prepare context for the chart
        chart_context = context_variables.copy()
        chart_context['dataset'] = main_chart_df
        chart_context['y_axis_label'] = query_name
        chart_context['chart_title'] = main_chart_title
        chart_context['noun'] = query_name
        
        # Generate the time series chart - let genChart handle weekly aggregation
        chart_result = generate_time_series_chart(
            context_variables=chart_context,
            time_series_field=time_field,
            numeric_fields=value_field,
            aggregation_period='week',
            filter_conditions=filter_conditions,
            agg_functions=agg_functions,
            store_in_db=True,
            object_type='weekly_analysis',
            object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
            object_name=query_name,
            skip_aggregation=False  # Let genChart handle the aggregation
        )
        
        logger.info(f"Successfully generated main time series chart for {query_name}")
        
        # Debug: Log the dataset before chart generation
        logger.info(f"Dataset before chart generation:")
        logger.info(f"  Shape: {dataset.shape}")
        logger.info(f"  Columns: {dataset.columns.tolist()}")
        logger.info(f"  {value_field} dtype: {dataset[value_field].dtype}")
        logger.info(f"  {value_field} sample values: {dataset[value_field].head().tolist()}")
        logger.info(f"  {time_field} sample values: {dataset[time_field].head().tolist()}")
        
        # Debug: Log the chart result type and content
        logger.info(f"Chart result type: {type(chart_result)}")
        if chart_result:
            logger.info(f"Chart result length: {len(str(chart_result))}")
            logger.info(f"Chart result preview: {str(chart_result)[:200]}...")
        
        # Append chart HTML to content lists if available
        if chart_result:
            if isinstance(chart_result, dict) and 'chart_html' in chart_result:
                all_html_contents.append(str(chart_result['chart_html']))
            else:
                all_html_contents.append(str(chart_result))
        else:
            logger.warning(f"No main chart result returned for {query_name}")
            
    except Exception as e:
        logger.error(f"Error generating main time series chart for {query_name}: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Process each category field for charts
    charts = {}
    for field in category_fields:
        # Skip supervisor_district field if process_districts is already handling it
        if (isinstance(field, str) and field.lower() == 'supervisor_district') or \
           (isinstance(field, dict) and field.get('fieldName', '').lower() == 'supervisor_district'):
            if process_districts:
                logger.info(f"Skipping individual processing for supervisor_district field as it will be handled in district processing")
                continue
        
        field_name = field if isinstance(field, str) else field.get('fieldName', '')
        if not field_name or field_name.lower() not in [col.lower() for col in dataset.columns]:
            logger.warning(f"Field {field_name} not found in dataset columns. Skipping.")
            continue
        
        logger.info(f"Processing category field: {field_name} for {query_name}")
        
        try:
            # Log the first 3 rows and dtypes before charting
            logger.info(f"[Category Chart: {field_name}] Data sample before charting:\n{dataset.head(3).to_string()}")
            logger.info(f"[Category Chart: {field_name}] Data types before charting:\n{dataset.dtypes}")
            
            # For category charts, pass the raw dataset to genChart
            # genChart will handle the weekly aggregation and category grouping properly
            category_chart_df = dataset.copy()
            
            logger.info(f"[Category Chart: {field_name}] Using raw data sample:\n{category_chart_df.head(10).to_string()}")
            logger.info(f"[Category Chart: {field_name}] Data shape: {category_chart_df.shape}")
            
            # Generate chart for this category field
            cat_chart_context = context_variables.copy()
            cat_chart_context['dataset'] = category_chart_df
            cat_chart_context['chart_title'] = f"{query_name} <br> {value_field} by Week by {field_name}"
            cat_chart_result = generate_time_series_chart(
                context_variables=cat_chart_context,
                time_series_field=time_field,
                numeric_fields=value_field,
                aggregation_period='week',
                filter_conditions=filter_conditions,
                agg_functions=agg_functions,
                group_field=field_name,
                store_in_db=True,
                object_type='weekly_analysis',
                object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
                object_name=query_name,
                skip_aggregation=False  # Let genChart handle the aggregation
            )
            
            logger.info(f"Successfully generated chart for {field_name} for {query_name}")
            
            # Append chart result to content lists
            if cat_chart_result:
                if isinstance(cat_chart_result, dict) and 'chart_html' in cat_chart_result:
                    all_html_contents.append(str(cat_chart_result['chart_html']))
                else:
                    all_html_contents.append(str(cat_chart_result))
            else:
                logger.warning(f"No chart result returned for {field_name} for {query_name}")
        except Exception as e:
            logger.error(f"Error generating chart for {field_name} for {query_name}: {str(e)}")
            logger.error(traceback.format_exc())
        
        try:
            # Run anomaly detection
            anomaly_results = anomaly_detection(
                context_variables={'dataset': dataset},
                group_field=field_name,
                numeric_field=value_field,
                date_field=time_field,
                recent_period=recent_period,
                comparison_period=comparison_period,
                period_type='week',
                agg_function=list(agg_functions.values())[0],
                y_axis_label=query_name,
                title=f"{query_name} - {value_field} by {field_name}",
                object_type='weekly_analysis',
                object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
                object_name=query_name
            )
            
            # Add anomaly text to the content
            if anomaly_results:
                # Create a markdown version of the anomaly content
                markdown_content = f"### Anomalies by {field_name}\n\n"
                markdown_content += f"Recent Period: {recent_period['start']} to {recent_period['end']}\n\n"
                markdown_content += f"Comparison Period: {comparison_period['start']} to {comparison_period['end']}\n\n"
                
                # Handle different response types
                if isinstance(anomaly_results, dict):
                    if 'anomalies_markdown' in anomaly_results:
                        # Prioritize markdown content
                        markdown_content += anomaly_results['anomalies_markdown']
                    elif 'anomalies' in anomaly_results:
                        # If we have formatted anomalies, create a markdown table
                        if isinstance(anomaly_results['anomalies'], list):
                            # Add markdown table header
                            markdown_content += f"| {field_name} | Recent Period | Comparison Period | Change | % Change |\n"
                            markdown_content += "|" + "---|" * 5 + "\n"
                            
                            # Add rows for each anomaly
                            for anomaly in anomaly_results['anomalies']:
                                if isinstance(anomaly, dict):
                                    markdown_content += f"| {anomaly.get('category', 'N/A')} | {anomaly.get('recent', 0):.1f} | {anomaly.get('comparison', 0):.1f} | {anomaly.get('abs_change', 0):.1f} | {anomaly.get('pct_change', 0):.1f}% |\n"
                        else:
                            # Convert HTML to plain text if it's not a list
                            html_content = str(anomaly_results['anomalies'])
                            # Simple HTML to text conversion
                            import re
                            text_content = re.sub(r'<[^>]+>', '', html_content)
                            text_content = re.sub(r'\s+', ' ', text_content).strip()
                            markdown_content += text_content
                    else:
                        markdown_content += "No significant anomalies detected.\n"
                else:
                    # If the result is a string, use it directly
                    markdown_content += str(anomaly_results)
                
                # Add the markdown content
                all_html_contents.append(markdown_content)
        except Exception as e:
            logger.error(f"Error detecting anomalies for {field_name} for {query_name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Process district-level data if requested and if supervisor_district field exists
    if process_districts and has_district:
        # Check that supervisor_district column actually exists before proceeding
        district_col = None
        for col in dataset.columns:
            if col.lower() == 'supervisor_district' or col.lower() == district_field.lower():
                district_col = col
                break
                
        if not district_col:
            logger.warning(f"Cannot process districts: supervisor_district or {district_field} column not found in dataset")
        else:
            logger.info(f"Processing district-level data using column: {district_col}")
            
            # Get all unique district values
            districts = dataset[district_col].unique()
            logger.info(f"Found {len(districts)} unique districts to process: {districts}")
            
            # Process each district separately
            for district in districts:
                if district is None or (isinstance(district, str) and district.strip() == ''):
                    logger.info(f"Skipping empty/null district value")
                    continue
                    
                logger.info(f"Processing data for district: {district}")
                
                try:
                    # Convert district to integer if possible for consistent use throughout
                    try:
                        # Handle both string and numeric decimal values
                        if isinstance(district, str):
                            # Try to convert string to float first, then to int
                            district_num = int(float(district))
                        else:
                            # Handle numeric types (float, int, etc.)
                            district_num = int(float(district))
                    except (ValueError, TypeError):
                        district_num = str(district).replace(' ', '_')
                    
                    # Filter data for this district
                    district_dataset = dataset[dataset[district_col] == district].copy()
                    
                    if district_dataset.empty:
                        logger.warning(f"No data for district {district} - skipping")
                        continue
                    
                    # For district charts, pass the raw district dataset to genChart
                    # genChart will handle the weekly aggregation properly
                    district_chart_df = district_dataset.copy()
                    
                    logger.info(f"[District Chart: {district}] Using raw district data sample:\n{district_chart_df.head(10).to_string()}")
                    logger.info(f"[District Chart: {district}] Data shape: {district_chart_df.shape}")
                        
                    # Create a separate context with just this district's data
                    district_context = context_variables.copy()
                    district_context['dataset'] = district_chart_df
                    district_context['chart_title'] = f"{query_name} - District {district_num}"
                    
                    # Create a district-specific HTML content list
                    district_html_contents = []
                    
                    # Create district-specific filter conditions that include the district
                    district_filter_conditions = filter_conditions.copy()
                    
                    # Use the converted district number for filter condition
                    district_filter_conditions.append({
                        'field': district_col,
                        'operator': '=',
                        'value': district_num,
                        'is_date': False
                    })
                    
                    # Generate chart for this district
                    district_chart_result = generate_time_series_chart(
                        context_variables=district_context,
                        time_series_field=time_field,
                        numeric_fields=value_field,
                        aggregation_period='week',
                        filter_conditions=district_filter_conditions,  # Use district-specific filter conditions
                        agg_functions=agg_functions,
                        store_in_db=True,
                        object_type='weekly_analysis',
                        object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
                        object_name=query_name,
                        skip_aggregation=False  # Let genChart handle the aggregation
                    )
                    
                    logger.info(f"Successfully generated chart for district {district}")
                    
                    # Save district-specific analysis
                    if district_chart_result:
                        # Add district chart to district content list
                        district_html_contents.append(f"## District {district_num}\n\n")
                        if isinstance(district_chart_result, dict) and 'chart_html' in district_chart_result:
                            district_html_contents.append(str(district_chart_result['chart_html']))
                        else:
                            district_html_contents.append(str(district_chart_result))
                            
                        # Process category fields within this district (similar to generate_metric_analysis)
                        # Filter out supervisor_district from category fields for district-specific analysis
                        district_category_fields = [field for field in category_fields if 
                                                  (isinstance(field, str) and field.lower() != 'supervisor_district') or 
                                                  (isinstance(field, dict) and field.get('fieldName', '').lower() != 'supervisor_district')]
                        
                        logger.info(f"Processing {len(district_category_fields)} category fields within district {district}")
                        
                        # Process each category field within this district
                        for field in district_category_fields:
                            field_name = field if isinstance(field, str) else field.get('fieldName', '')
                            if not field_name or field_name.lower() not in [col.lower() for col in district_dataset.columns]:
                                logger.warning(f"Category field {field_name} not found in district {district} dataset. Skipping.")
                                continue
                            
                            logger.info(f"Processing category field: {field_name} for district {district}")
                            
                            try:
                                # Generate chart for this category field within the district
                                cat_chart_context = district_context.copy()
                                cat_chart_context['chart_title'] = f"{query_name} - District {district_num} <br> {value_field} by Week by {field_name}"
                                cat_chart_result = generate_time_series_chart(
                                    context_variables=cat_chart_context,
                                    time_series_field=time_field,
                                    numeric_fields=value_field,
                                    aggregation_period='week',
                                    filter_conditions=district_filter_conditions,  # Use district-specific filter conditions
                                    agg_functions=agg_functions,
                                    group_field=field_name,
                                    store_in_db=True,
                                    object_type='weekly_analysis',
                                    object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
                                    object_name=query_name,
                                    skip_aggregation=False  # Let genChart handle the aggregation
                                )
                                
                                logger.info(f"Successfully generated category chart for {field_name} in district {district}")
                                
                                # Append chart result to district content
                                if cat_chart_result:
                                    if isinstance(cat_chart_result, dict) and 'chart_html' in cat_chart_result:
                                        district_html_contents.append(str(cat_chart_result['chart_html']))
                                    else:
                                        district_html_contents.append(str(cat_chart_result))
                                else:
                                    logger.warning(f"No chart result returned for {field_name} in district {district}")
                            except Exception as e:
                                logger.error(f"Error generating chart for {field_name} in district {district}: {str(e)}")
                                logger.error(traceback.format_exc())
                            
                            try:
                                # Run anomaly detection for this category field within the district
                                district_cat_anomalies = anomaly_detection(
                                    context_variables={'dataset': district_dataset},
                                    group_field=field_name,
                                    numeric_field=value_field,
                                    date_field=time_field,
                                    recent_period=recent_period,
                                    comparison_period=comparison_period,
                                    period_type='week',
                                    agg_function=list(agg_functions.values())[0],
                                    y_axis_label=query_name,
                                    title=f"{query_name} - District {district_num} - {value_field} by {field_name}",
                                    object_type='weekly_analysis',
                                    object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
                                    object_name=query_name
                                )
                                
                                # Add anomaly text to the district content
                                if district_cat_anomalies:
                                    # Create a markdown version of the anomaly content
                                    markdown_content = f"### Anomalies by {field_name} for District {district_num}\n\n"
                                    markdown_content += f"Recent Period: {recent_period['start']} to {recent_period['end']}\n\n"
                                    markdown_content += f"Comparison Period: {comparison_period['start']} to {comparison_period['end']}\n\n"
                                    
                                    # Handle different response types
                                    if isinstance(district_cat_anomalies, dict):
                                        if 'anomalies_markdown' in district_cat_anomalies:
                                            # Prioritize markdown content
                                            markdown_content += district_cat_anomalies['anomalies_markdown']
                                        elif 'anomalies' in district_cat_anomalies:
                                            # If we have formatted anomalies, create a markdown table
                                            if isinstance(district_cat_anomalies['anomalies'], list):
                                                # Add markdown table header
                                                markdown_content += f"| {field_name} | Recent Period | Comparison Period | Change | % Change |\n"
                                                markdown_content += "|" + "---|" * 5 + "\n"
                                                
                                                # Add rows for each anomaly
                                                for anomaly in district_cat_anomalies['anomalies']:
                                                    if isinstance(anomaly, dict):
                                                        markdown_content += f"| {anomaly.get('category', 'N/A')} | {anomaly.get('recent', 0):.1f} | {anomaly.get('comparison', 0):.1f} | {anomaly.get('abs_change', 0):.1f} | {anomaly.get('pct_change', 0):.1f}% |\n"
                                            else:
                                                # Convert HTML to plain text if it's not a list
                                                html_content = str(district_cat_anomalies['anomalies'])
                                                # Simple HTML to text conversion
                                                import re
                                                text_content = re.sub(r'<[^>]+>', '', html_content)
                                                text_content = re.sub(r'\s+', ' ', text_content).strip()
                                                markdown_content += text_content
                                        else:
                                            markdown_content += "No significant anomalies detected.\n"
                                    else:
                                        # If the result is a string, use it directly
                                        markdown_content += str(district_cat_anomalies)
                                    
                                    # Add the markdown content
                                    district_html_contents.append(markdown_content)
                            except Exception as e:
                                logger.error(f"Error detecting anomalies for {field_name} in district {district}: {str(e)}")
                                logger.error(traceback.format_exc())
                            
                        # Try to detect anomalies for this district overall (existing code)
                        try:
                            district_anomalies = anomaly_detection(
                                context_variables={'dataset': district_dataset},
                                group_field=None,  # No need for category field since we're already filtering by district
                                numeric_field=value_field,
                                date_field=time_field,
                                recent_period=recent_period,
                                comparison_period=comparison_period,
                                period_type='week',
                                agg_function=list(agg_functions.values())[0],
                                y_axis_label=query_name,
                                title=f"{query_name} - District {district_num}",
                                object_type='weekly_analysis',
                                object_id=metric_info.get('metric_id', metric_info.get('id', 'unknown')),
                                object_name=query_name
                            )
                            
                            if district_anomalies:
                                # Convert HTML to markdown for district anomalies too
                                district_html_contents.append(f"### Anomalies for District {district_num}\n\n")
                                
                                # Handle different response types
                                if isinstance(district_anomalies, dict):
                                    if 'anomalies_markdown' in district_anomalies:
                                        # Prioritize markdown content
                                        markdown_content = district_anomalies['anomalies_markdown']
                                    elif 'anomalies' in district_anomalies:
                                        # Create markdown version
                                        markdown_content = f"Recent Period: {recent_period['start']} to {recent_period['end']}\n\n"
                                        markdown_content += f"Comparison Period: {comparison_period['start']} to {comparison_period['end']}\n\n"
                                        
                                        # If we have formatted anomalies, create a markdown table
                                        if isinstance(district_anomalies['anomalies'], list):
                                            # Add markdown table header
                                            markdown_content += "| Metric | Recent Period | Comparison Period | Change | % Change |\n"
                                            markdown_content += "|" + "---|" * 5 + "\n"
                                            
                                            # Add one row for the district level
                                            for anomaly in district_anomalies['anomalies']:
                                                if isinstance(anomaly, dict):
                                                    markdown_content += f"| District {district_num} | {anomaly.get('recent', 0):.1f} | {anomaly.get('comparison', 0):.1f} | {anomaly.get('abs_change', 0):.1f} | {anomaly.get('pct_change', 0):.1f}% |\n"
                                        else:
                                            # Convert HTML to plain text if it's not a list
                                            html_content = str(district_anomalies['anomalies'])
                                            # Simple HTML to text conversion
                                            import re
                                            text_content = re.sub(r'<[^>]+>', '', html_content)
                                            text_content = re.sub(r'\s+', ' ', text_content).strip()
                                            markdown_content += text_content
                                    else:
                                        markdown_content = "No significant anomalies detected for this district.\n"
                                else:
                                    # If the result is a string, use it directly
                                    markdown_content = str(district_anomalies)
                                
                                district_html_contents.append(markdown_content)
                        except Exception as e:
                            logger.error(f"Error detecting anomalies for district {district}: {str(e)}")
                            
                        # Create district-specific result and save it
                        district_content = "\n\n".join(district_html_contents)
                        district_result = {
                            'metric_id': metric_info.get('id', ''),
                            'name': f"{query_name} - District {district_num}",
                            'content': district_content,
                            'html_contents': district_html_contents,
                            'date_range': f"{recent_period['start']} to {recent_period['end']}"
                        }
                        
                        # Get numeric metric ID
                        numeric_id = metric_info.get('numeric_id') or metric_info.get('id')
                        if not numeric_id and 'metric_id' in metric_info:
                            numeric_id = metric_info['metric_id']
                        
                        # Save the district-specific result to files
                        try:
                            from .report_generator import save_weekly_analysis
                            save_weekly_analysis(district_result, numeric_id, district=district_num)
                            logger.info(f"Saved district-specific analysis for district {district}")
                        except Exception as e:
                            logger.error(f"Error saving district-specific analysis for district {district}: {str(e)}")
                            logger.error(traceback.format_exc())
                            
                        # Add a reference to this district's analysis in the main content
                        numeric_id = metric_info.get('numeric_id') or metric_info.get('id') 
                        if not numeric_id and 'metric_id' in metric_info:
                            numeric_id = metric_info['metric_id']
                            
                        if not numeric_id or (isinstance(numeric_id, str) and numeric_id.strip() == ''):
                            # Use sanitized query name if no ID available
                            metric_file_id = query_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                        else:
                            metric_file_id = numeric_id
                            
                        all_html_contents.append(f"## [District {district_num} Analysis](/{district_num}/{metric_file_id}.md)\n\n")
                    else:
                        logger.warning(f"No chart result returned for district {district}")
                        
                except Exception as e:
                    logger.error(f"Error processing district {district}: {str(e)}")
                    logger.error(traceback.format_exc())
    
    # Combine all markdown content
    all_content = "\n\n".join(all_html_contents)
    
    # Determine data current as-of value
    data_as_of_str = None
    try:
        # Prefer the metric's most_recent_data_date if available
        if most_recent_str:
            data_as_of_str = str(most_recent_dt)
        else:
            # Fallback to the maximum date seen in dataset for the chosen date_field
            if date_field in dataset.columns:
                data_as_of_str = str(pd.to_datetime(dataset[date_field]).max().date())
            elif 'actual_date' in dataset.columns:
                data_as_of_str = str(pd.to_datetime(dataset['actual_date']).max().date())
    except Exception as _:
        pass
    
    # Return the result object
    return {
        'metric_id': metric_info.get('id', ''),
        'name': query_name,
        'content': all_content,
        'html_contents': all_html_contents,
        'date_range': f"{recent_period['start']} to {recent_period['end']}",
        'data_as_of': data_as_of_str,
        'date_field': date_field
    }

def run_weekly_analysis(metrics_list=None, process_districts=False):
    """Run weekly analysis for specified metrics or all metrics if none specified."""
    start_time = datetime.now()
    logger.info(f"========== STARTING WEEKLY ANALYSIS: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    logger.info(f"Process districts: {process_districts}")
    
    if metrics_list:
        logger.info(f"Analyzing {len(metrics_list)} specified metrics: {', '.join(str(m) for m in metrics_list)}")
    else:
        logger.info("No metrics to analyze")
        return []
    
    # Process each metric
    all_results = []
    successful_metrics = []
    failed_metrics = []
    metrics_using_ytd = []  # Track metrics that used YTD queries
    
    # Log metric processing start
    logger.info(f"Beginning to process {len(metrics_list)} metrics")
    
    for i, metric_id in enumerate(metrics_list):
        metric_start_time = datetime.now()
        logger.info(f"[{i+1}/{len(metrics_list)}] Processing weekly analysis for metric: {metric_id} - Started at {metric_start_time.strftime('%H:%M:%S')}")
        
        # Find the metric in the database
        logger.info(f"Searching for metric '{metric_id}' in database...")
        from .data_processor import find_metric_in_queries
        metric_info = find_metric_in_queries(None, metric_id)  # Pass None since we're not using JSON files anymore
        if not metric_info:
            logger.error(f"Metric with ID '{metric_id}' not found in database")
            failed_metrics.append(metric_id)
            continue
        
        # Log metric details found
        query_name = metric_info.get('name', metric_info.get('query_name', 'Unknown Metric'))
        logger.info(f"Found metric '{metric_id}' - Query Name: '{query_name}'")
        logger.info(f"Category: {metric_info.get('top_category', 'Unknown')}/{metric_info.get('subcategory', 'Unknown')}")
        
        # Make sure metric_id is in the metric_info
        if 'metric_id' not in metric_info:
            metric_info['metric_id'] = metric_id
            logger.info(f"Added missing metric_id '{metric_id}' to metric_info")
        
        # Check if the metric has YTD queries
        has_ytd_query = False
        if isinstance(metric_info.get('query_data'), dict):
            if 'ytd_query' in metric_info['query_data']:
                has_ytd_query = True
                logger.info(f"Metric '{metric_id}' has a YTD query available")
            elif 'queries' in metric_info['query_data'] and isinstance(metric_info['query_data']['queries'], dict):
                if 'ytd_query' in metric_info['query_data']['queries'] or 'executed_ytd_query' in metric_info['query_data']['queries']:
                    has_ytd_query = True
                    logger.info(f"Metric '{metric_id}' has a YTD query in the queries dictionary")
        
        # Process the weekly analysis
        logger.info(f"Starting weekly analysis processing for '{query_name}'...")
        try:
            result = process_weekly_analysis(metric_info, process_districts=process_districts)
            if result:
                # Make sure numeric metric_id is set correctly in the result
                numeric_metric_id = metric_id  # This is the ID passed to find_metric_in_queries
                if not result.get('metric_id') or result.get('metric_id') == '':
                    result['metric_id'] = numeric_metric_id
                    logger.info(f"Set numeric metric_id in result: {numeric_metric_id}")
                
                # Save the analysis results to files
                from .report_generator import save_weekly_analysis
                saved_paths = save_weekly_analysis(result, numeric_metric_id, district=0)
                # Add file paths to the result object
                if saved_paths:
                    result['md_path'] = saved_paths.get('md_path')
                    result['json_path'] = saved_paths.get('json_path')
                
                all_results.append(result)
                successful_metrics.append(metric_id)
                metric_end_time = datetime.now()
                duration = (metric_end_time - metric_start_time).total_seconds()
                logger.info(f"Completed weekly analysis for {metric_id} - Duration: {duration:.2f} seconds")
                logger.info(f"Analysis saved to: {result.get('md_path', 'Unknown path')}")
                
                # Check if YTD query was used based on log messages
                if has_ytd_query:
                    metrics_using_ytd.append(metric_id)
                    logger.info(f"Successfully used YTD query for {metric_id}")
            else:
                logger.error(f"Failed to complete weekly analysis for {metric_id} - result was None")
                failed_metrics.append(metric_id)
        except Exception as e:
            logger.error(f"Exception while processing metric {metric_id}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_metrics.append(metric_id)
    
    # Log summary statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"========== WEEKLY ANALYSIS COMPLETE: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    logger.info(f"Total duration: {duration:.2f} seconds")
    logger.info(f"Metrics processed: {len(metrics_list)}")
    logger.info(f"Successful: {len(successful_metrics)} - {', '.join(str(m) for m in successful_metrics) if successful_metrics else 'None'}")
    logger.info(f"Failed: {len(failed_metrics)} - {', '.join(str(m) for m in failed_metrics) if failed_metrics else 'None'}")
    logger.info(f"Using YTD queries: {len(metrics_using_ytd)} - {', '.join(str(m) for m in metrics_using_ytd) if metrics_using_ytd else 'None'}")
    
    # Return results for potential newsletter generation
    return all_results 
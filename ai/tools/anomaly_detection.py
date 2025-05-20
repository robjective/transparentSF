import datetime
from datetime import date
import logging
import pandas as pd
from .generateAnomalyCharts import generate_anomalies_summary_with_charts 
from dateutil import parser  # Import this library for robust date parsing
import os
import psycopg2
from psycopg2.extras import Json
import json
import numpy as np
from .store_anomalies import store_anomaly_data, CustomJSONEncoder  # Import the store_anomalies module

# set logging level to INFO
logging.basicConfig(level=logging.INFO)

def find_key(item, field_name):
    if field_name is None:
        return None
    if field_name in item:
        return field_name
    for key in item.keys():
        if key.lower() == field_name.lower():
            return key
    return None

def get_item_value_case_insensitive(item, field_name):
    if field_name is None:
        return None
    for key in item.keys():
        if key.lower() == field_name.lower():
            return item[key]
    return None

def calculate_stats(values):
    n = len(values)
    if n == 0:
        return {'mean': None, 'stdDev': None}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = variance ** 0.5
    return {'mean': float(mean), 'stdDev': float(std_dev)}

def get_date_ranges():
    from datetime import date, timedelta

    today = date.today()

    # Calculate recent period: last full month (previous month)
    first_day_of_current_month = date(today.year, today.month, 1)
    recent_end = first_day_of_current_month - timedelta(days=1)  # Last day of previous month
    recent_start = date(recent_end.year, recent_end.month, 1)    # First day of previous month

    # Calculate comparison period: 12 months before recentStart
    comparison_end = recent_start - timedelta(days=1)            # Day before recentStart
    comparison_start = date(recent_start.year - 1, recent_start.month, 1)  # Same month and day, previous year

    print("Recent period:", recent_start, "to", recent_end)
    print("Comparison period:", comparison_start, "to", comparison_end)
    
    return {
        'recentPeriod': {
            'start': recent_start,
            'end': recent_end,
        },
        'comparisonPeriod': {
            'start': comparison_start,
            'end': comparison_end,
        },
    }

def generate_anomalies_summary(results):
    anomalies = []
    normal_items = []
    for item in results:
        group = item.get('group_value', 'Unknown Group')
        recent_mean = item.get('recent_mean')
        comparison_mean = item.get('comparison_mean')
        difference = item.get('difference')
        std_dev = item.get('stdDev', 0)
        out_of_bounds = item.get('out_of_bounds', False)

        # Determine the direction of the difference
        if difference > 0:
            direction = "higher"
        elif difference < 0:
            direction = "lower"
        else:
            direction = "the same as"

        # Absolute value of difference for reporting
        abs_difference = abs(difference)

        # Format the message based on whether it's an anomaly or normal
        if out_of_bounds:
            message = (
                f"Anomaly detected in '{group}': "
                f"Recent mean (RM: {recent_mean:.2f}) is {direction} than the comparison mean (CM: {comparison_mean:.2f}) "
                f"by {abs_difference:.2f} units. "
                f"Standard Deviation: {std_dev:.2f}."
            )
            anomalies.append(message)
        else:
            message = (
                f"'{group}': RM: {recent_mean:.2f}, CM: {comparison_mean:.2f}, Diff: {difference:.2f}"
            )
            normal_items.append(message)

    # Create the markdown summary
    markdown_summary = "## Anomalies Summary\n\n"
    
    if anomalies:
        markdown_summary += "### Anomalies Detected\n"
        for anomaly in anomalies:
            markdown_summary += f"- {anomaly}\n"
    else:
        markdown_summary += "No anomalies detected.\n"
    
    if normal_items:
        markdown_summary += "\n### Items Within Normal Range\n"
        for item in normal_items:
            markdown_summary += f"- {item}\n"

    # Convert to HTML for backward compatibility
    html_summary = f"<html><body>{markdown_summary}</body></html>"

    return {"anomalies": html_summary}

def group_data_by_field_and_date(data_array, group_field, numeric_field, date_field, period_type='month', agg_function='sum'):
    logging.info("=== Starting Data Grouping ===")
    logging.info(f"First 5 records of raw data:")
    for i, item in enumerate(data_array[:5]):
        logging.info(f"Record {i + 1}:")
        logging.info(f"  {group_field}: {item.get(group_field)}")
        logging.info(f"  {date_field}: {item.get(date_field)} (type: {type(item.get(date_field))})")
        logging.info(f"  {numeric_field}: {item.get(numeric_field)}")
        logging.info("---")

    # Create a set to track all unique dates we see
    all_dates = set()

    # Dictionary to track the sum and count for calculating mean if needed
    data_points = {}

    for item in data_array:
        group_value = item.get(group_field)
        date_obj = item.get(date_field)
        
        if not date_obj:
            logging.warning(f"Missing date for record: {item}")
            continue

        # Convert string dates to datetime objects
        if isinstance(date_obj, str):
            date_obj = custom_parse_date(date_obj, period_type)
            if not date_obj:
                logging.warning(f"Could not parse date for record: {item}")
                continue
        elif isinstance(date_obj, pd.Timestamp):
            date_obj = date_obj.to_pydatetime().date()
        elif isinstance(date_obj, datetime.datetime):
            date_obj = date_obj.date()
        elif not isinstance(date_obj, datetime.date):
            logging.warning(f"Unrecognized date type for record: {item}")
            continue

        # Use appropriate date format based on period_type
        if period_type == 'year':
            # For year period, always use YYYY format
            date_key = str(date_obj.year)
        elif period_type == 'month':
            # For month period, always use YYYY-MM format
            date_key = date_obj.strftime("%Y-%m")
        else:  # day
            date_key = date_obj.strftime("%Y-%m-%d")

        all_dates.add(date_key)

        # Initialize the data structure if needed
        if group_value not in data_points:
            data_points[group_value] = {}
        if date_key not in data_points[group_value]:
            data_points[group_value][date_key] = {'sum': 0.0, 'count': 0}

        # Convert numeric value to float, with error handling
        try:
            numeric_str = str(item.get(numeric_field, '0')).replace(',', '')  # Remove any commas
            numeric_value = float(numeric_str)
        except (ValueError, TypeError):
            logging.warning(f"Invalid numeric value for record: {item}")
            numeric_value = 0.0
            
        # Update the sum and count
        data_points[group_value][date_key]['sum'] += numeric_value
        data_points[group_value][date_key]['count'] += 1

    # Now apply the aggregation function to get the final grouped data
    grouped = {}
    for group_value, dates in data_points.items():
        grouped[group_value] = {}
        for date_key, values in dates.items():
            if agg_function == 'mean':
                # Avoid division by zero
                if values['count'] > 0:
                    grouped[group_value][date_key] = values['sum'] / values['count']
                else:
                    grouped[group_value][date_key] = 0.0
            else:  # Default to sum
                grouped[group_value][date_key] = values['sum']

    logging.info(f"Using aggregation function: {agg_function}")
    logging.info("=== Grouping Complete ===")
    logging.info(f"Total groups found: {len(grouped)}")
    logging.info(f"All unique dates found in data: {sorted(list(all_dates))}")
    for key in list(grouped.keys())[:3]:  # Show first 3 groups
        logging.info(f"Sample group '{key}' dates: {list(grouped[key].keys())}")

    return grouped

def custom_parse_date(date_str, period_type='month'):
    # Handle YYYY format for year period type
    if period_type == 'year' and isinstance(date_str, str):
        try:
            # For year period, always extract just the year regardless of input format
            if '-' in date_str:
                year = int(date_str.split('-')[0])
            else:
                year = int(date_str)
            return datetime.date(year, 1, 1)
        except ValueError:
            pass

    # Handle YYYY-MM format for month period type
    if isinstance(date_str, str) and len(date_str.split('-')) == 2:
        try:
            year, month = map(int, date_str.split('-'))
            return datetime.date(year, month, 1)
        except ValueError:
            pass

    # For other formats, try explicit parsing with known format
    date_formats = [
        "%Y-%m-%d",    # YYYY-MM-DD
        "%Y%m%d",      # YYYYMMDD
        "%m/%d/%Y",    # MM/DD/YYYY
        "%d/%m/%Y"     # DD/MM/YYYY
    ]
    
    for fmt in date_formats:
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Use dateutil.parser as last resort, with explicit dayfirst=False
    try:
        return parser.parse(date_str, dayfirst=False).date()
    except (ValueError, parser.ParserError):
        return None

def filter_data_by_date_and_conditions(data, filter_conditions, start_date=None, end_date=None, date_field=None, period_type='month'):
    # Initialize error counters
    error_counts = {
        'missing_date_field': 0,
        'unrecognized_date_type': 0,
        'invalid_date_format': 0,
        'date_outside_range': 0,
        'missing_condition_field': 0,
        'missing_value': 0,
        'invalid_numeric_date': 0,
        'non_numeric_comparison': 0,
        'comparison_error': 0
    }
    
    # Convert date strings to datetime.date objects
    def parse_filter_date(date_val):
        if isinstance(date_val, str):
            try:
                if len(date_val.split('-')) == 2:
                    # For YYYY-MM format
                    year, month = map(int, date_val.split('-'))
                    return datetime.date(year, month, 1)
                else:
                    # Try parsing as YYYY-MM-DD
                    return datetime.datetime.strptime(date_val, "%Y-%m-%d").date()
            except ValueError:
                pass
        return date_val

    # Function to format date based on period_type
    def format_date(date_obj, period_type):
        if period_type == 'year':
            return date_obj.strftime("%Y")
        elif period_type == 'month':
            return date_obj.strftime("%Y-%m")
        else:  # day
            return date_obj.strftime("%Y-%m-%d")

    if start_date:
        start_date = parse_filter_date(start_date)
    if end_date:
        end_date = parse_filter_date(end_date)

    # Update the date comparison in filter conditions
    for condition in filter_conditions:
        value = condition['value']
        if isinstance(value, (datetime.date, datetime.datetime)):
            condition['is_date'] = True
        elif isinstance(value, str):
            parsed_date = parse_filter_date(value)
            if isinstance(parsed_date, datetime.date):
                condition['value'] = parsed_date
                condition['is_date'] = True
            else:
                condition['is_date'] = False
        else:
            condition['is_date'] = False

    # Add logging for date range parameters
    logging.info(f"Date filtering parameters:")
    logging.info(f"  start_date: {start_date} ({type(start_date)})")
    logging.info(f"  end_date: {end_date} ({type(end_date)})")
    logging.info(f"  date_field: {date_field}")
    logging.info(f"  period_type: {period_type}")
    logging.info(f"  filter_conditions: {filter_conditions}")

    filtered_data = []
    for idx, item in enumerate(data):
        # Initialize meets_conditions to True
        meets_conditions = True

        # If date_field and date range are provided, perform date filtering
        if date_field and start_date and end_date:
            key_date_field = find_key(item, date_field)
            if key_date_field is None:
                error_counts['missing_date_field'] += 1
                continue
            date_value = item[key_date_field]
            
            # Preserve the original date value
            original_date = date_value
            
            # Convert date_field value to datetime.date if necessary
            if isinstance(date_value, str):
                if len(date_value.split('-')) == 2:
                    # If it's YYYY-MM format, keep it as is
                    item_date = date_value
                else:
                    item_date = custom_parse_date(date_value, period_type)
            elif isinstance(date_value, pd.Timestamp):
                # Format based on period_type
                item_date = format_date(date_value, period_type)
            elif isinstance(date_value, (datetime.date, datetime.datetime)):
                # Format based on period_type
                item_date = format_date(date_value, period_type)
            else:
                error_counts['unrecognized_date_type'] += 1
                continue

            # Update item with original date value
            item[date_field] = original_date

            # Convert date_field value to datetime.date if necessary
            if isinstance(date_value, str):
                item_date = custom_parse_date(date_value, period_type)
            elif isinstance(date_value, pd.Timestamp):
                item_date = date_value.date()
            elif isinstance(date_value, (datetime.date, datetime.datetime)):
                item_date = date_value.date() if isinstance(date_value, datetime.datetime) else date_value
            elif isinstance(date_value, (int, float)):
                try:
                    date_str = str(int(date_value))
                    item_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                except (ValueError, TypeError):
                    error_counts['invalid_numeric_date'] += 1
                    continue
            else:
                error_counts['unrecognized_date_type'] += 1
                continue

            if item_date is None:
                error_counts['invalid_date_format'] += 1
                continue
            
            # Convert start_date and end_date to first/last day of month if they're year-month dates
            if isinstance(start_date, str) and len(start_date.split('-')) == 2:
                start_date = datetime.datetime.strptime(f"{start_date}-01", "%Y-%m-%d").date()
            if isinstance(end_date, str) and len(end_date.split('-')) == 2:
                # Set to last day of month
                next_month = datetime.datetime.strptime(f"{end_date}-01", "%Y-%m-%d").date().replace(day=28) + datetime.timedelta(days=4)
                end_date = next_month - datetime.timedelta(days=next_month.day)

            # Ensure item_date is a datetime.date object for comparison
            if isinstance(item_date, str):
                if period_type == 'year':
                    try:
                        item_date = datetime.datetime.strptime(item_date, "%Y").date()
                    except ValueError:
                        error_counts['invalid_date_format'] += 1
                        continue
                elif period_type == 'month':
                    try:
                        item_date = datetime.datetime.strptime(f"{item_date}-01", "%Y-%m-%d").date()
                    except ValueError:
                        error_counts['invalid_date_format'] += 1
                        continue
                else:  # day
                    try:
                        item_date = datetime.datetime.strptime(item_date, "%Y-%m-%d").date()
                    except ValueError:
                        error_counts['invalid_date_format'] += 1
                        continue

            # Perform date filtering based on actual date objects
            try:
                if not (start_date <= item_date <= end_date):
                    error_counts['date_outside_range'] += 1
                    continue  # Skip this record
            except TypeError:
                error_counts['comparison_error'] += 1
                continue

            # Update item with formatted date string based on period_type
            item[date_field] = format_date(item_date, period_type)
        
        # Now, check the filter_conditions
        for condition in filter_conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            is_date = condition.get('is_date', False)
            key_field = find_key(item, field)
            
            if key_field is None:
                meets_conditions = False
                error_counts['missing_condition_field'] += 1
                break
            item_value = item[key_field]

            # If item_value is None, we can't compare
            if item_value is None:
                meets_conditions = False
                error_counts['missing_value'] += 1
                break

            # Parse item_value if condition value is a date
            if is_date:
                if isinstance(item_value, str):
                    item_value = custom_parse_date(item_value, period_type)
                elif isinstance(item_value, (datetime.date, datetime.datetime)):
                    item_value = item_value.date() if isinstance(item_value, datetime.datetime) else item_value
                elif isinstance(item_value, (int, float)):
                    try:
                        date_str = str(int(item_value))
                        item_value = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                    except (ValueError, TypeError):
                        meets_conditions = False
                        error_counts['invalid_numeric_date'] += 1
                        break
                else:
                    meets_conditions = False
                    error_counts['unrecognized_date_type'] += 1
                    break

                if item_value is None:
                    meets_conditions = False
                    error_counts['invalid_date_format'] += 1
                    break

                # Perform date comparison
                try:
                    if operator == '>' and not (item_value > value):
                        meets_conditions = False
                        break
                    elif operator == '<' and not (item_value < value):
                        meets_conditions = False
                        break
                    elif operator == '>=' and not (item_value >= value):
                        meets_conditions = False
                        break
                    elif operator == '<=' and not (item_value <= value):
                        meets_conditions = False
                        break
                    elif operator == '==' and not (item_value == value):
                        meets_conditions = False
                        break
                    elif operator == '!=' and not (item_value != value):
                        meets_conditions = False
                        break
                except Exception:
                    meets_conditions = False
                    error_counts['comparison_error'] += 1
                    break
            else:
                # Clean and convert values if numeric
                if isinstance(item_value, str):
                    item_value = item_value.replace(',', '')
                if isinstance(value, str):
                    value = value.replace(',', '')

                # Try numeric comparison first
                try:
                    item_value_float = float(item_value)
                    value_float = float(value)
                    is_numeric = True
                except (ValueError, TypeError):
                    is_numeric = False

                # Condition checks for non-date fields
                try:
                    if operator in ['=', '==']:
                        # For equality, use direct comparison without type conversion
                        if is_numeric:
                            meets_conditions = (item_value_float == value_float)
                        else:
                            meets_conditions = (str(item_value) == str(value))
                        if not meets_conditions:
                            break
                    elif operator == '!=':
                        if is_numeric:
                            meets_conditions = (item_value_float != value_float)
                        else:
                            meets_conditions = (str(item_value) != str(value))
                        if not meets_conditions:
                            break
                    elif operator in ['<', '<=', '>', '>=']:
                        if not is_numeric:
                            meets_conditions = False
                            error_counts['non_numeric_comparison'] += 1
                            break
                        if operator == '<' and not (item_value_float < value_float):
                            meets_conditions = False
                            break
                        elif operator == '<=' and not (item_value_float <= value_float):
                            meets_conditions = False
                            break
                        elif operator == '>' and not (item_value_float > value_float):
                            meets_conditions = False
                            break
                        elif operator == '>=' and not (item_value_float >= value_float):
                            meets_conditions = False
                            break
                except Exception:
                    meets_conditions = False
                    error_counts['comparison_error'] += 1
                    break

        if meets_conditions:
            filtered_data.append(item)

    # Add summary of date filtering
    if date_field and start_date and end_date:
        logging.info(f"Date filtering summary:")
        logging.info(f"  Total records before date filtering: {len(data)}")
        logging.info(f"  Records passing date filter: {len(filtered_data)}")
        logging.info(f"  Records filtered out by date: {len(data) - len(filtered_data)}")

    # Log error summary
    total_records = len(data)
    filtered_records = len(filtered_data)
    invalid_records = total_records - filtered_records
    logging.info(f"Total records processed: {total_records}")
    logging.info(f"Total records after filtering: {filtered_records}")
    logging.info(f"Total invalid records: {invalid_records}")
    
    # Log error counts
    if sum(error_counts.values()) > 0:
        logging.info("Error count summary:")
        for error_type, count in error_counts.items():
            if count > 0:
                logging.info(f"  {error_type}: {count}")
        
        # Log sample errors if there are any
        if error_counts['non_numeric_comparison'] > 0:
            logging.info("Sample non-numeric comparison errors may be caused by:")
            for condition in filter_conditions:
                logging.info(f"  Condition: {condition['field']} {condition['operator']} {condition['value']}")

    return filtered_data

def get_month_range(start_date, end_date):
    from dateutil.relativedelta import relativedelta
    months = []
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        # Only use YYYY-MM format consistently
        months.append(current_date.strftime("%Y-%m"))
        current_date += relativedelta(months=1)
    logging.info(f"Generated month range: {months}")
    return months

def parse_period_date(date_val):
    """
    Parse period date into standard format
    
    Args:
        date_val: Date value in various formats (string, datetime, date)
        
    Returns:
        dict: Year and month parts
    """
    # Handle datetime or date objects
    if isinstance(date_val, (datetime, date, pd.Timestamp)):
        return {
            'year': date_val.year,
            'month': date_val.month,
            'day': date_val.day
        }
    
    # Handle string format
    if isinstance(date_val, str):
        # Check if it's in YYYY-WXX format (week format)
        if 'W' in date_val:
            year = date_val.split('-W')[0]
            return {
                'year': int(year),
                'week': int(date_val.split('-W')[1])
            }
        
        # Handle ISO format date
        try:
            parts = date_val.split('-')
            if len(parts) >= 2:
                return {
                    'year': int(parts[0]),
                    'month': int(parts[1]),
                    'day': int(parts[2]) if len(parts) > 2 else 1
                }
        except (ValueError, IndexError):
            # If parsing fails, try to parse as datetime
            try:
                dt = pd.to_datetime(date_val)
                return {
                    'year': dt.year,
                    'month': dt.month,
                    'day': dt.day
                }
            except:
                # Return a default value if all parsing fails
                return {
                    'year': datetime.now().year,
                    'month': datetime.now().month,
                    'day': 1
                }
    
    # Default return if parsing fails
    return {
        'year': datetime.now().year,
        'month': datetime.now().month,
        'day': 1
    }

def anomaly_detection(
    context_variables,
    group_field=None,
    filter_conditions=[],
    min_diff=2,
    recent_period=None,
    comparison_period=None,
    date_field=None,
    numeric_field=None,
    y_axis_label=None,
    title=None,
    period_type='month',  # Add period_type parameter with default
    agg_function='sum',  # Add agg_function parameter with default 'sum'
    output_dir=None,  # Add output_dir parameter with default None
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None,
    store_in_db=True,
    object_type=None,   # Add object_type parameter
    object_id=None,     # Add object_id parameter
    object_name=None    # Add object_name parameter
):
    
    # Set default output_dir based on period_type if not provided
    if output_dir is None:
        output_dir = 'monthly' if period_type == 'month' else 'annual' if period_type == 'year' else period_type
    
    data = context_variables.get("dataset")
    if data is None:
        logging.error("Dataset is not available.")
        return {"error": "Dataset is not available."}

    data_records = data.to_dict('records')
    
    # Adjust date periods if date_field is provided
    if date_field and (recent_period is None or comparison_period is None):
        date_ranges = get_date_ranges()
        if recent_period is None:
            recent_period = date_ranges['recentPeriod']
        if comparison_period is None:
            comparison_period = date_ranges['comparisonPeriod']

    # Convert period dates to datetime.date objects, handling both YYYY-MM and YYYY-MM-DD formats
    def parse_period_date(date_val):
        if isinstance(date_val, str):
            if len(date_val.split('-')) == 2:
                # For YYYY-MM format
                year, month = map(int, date_val.split('-'))
                if 'start' in date_val:
                    return datetime.date(year, month, 1)
                else:
                    # For end date, get last day of month
                    if month == 12:
                        next_month = datetime.date(year + 1, 1, 1)
                    else:
                        next_month = datetime.date(year, month + 1, 1)
                    return next_month - datetime.timedelta(days=1)
            else:
                # For YYYY-MM-DD format
                return datetime.datetime.strptime(date_val, "%Y-%m-%d").date()
        return date_val

    if recent_period:
        recent_period = {
            'start': parse_period_date(recent_period['start']),
            'end': parse_period_date(recent_period['end'])
        }
    if comparison_period:
        comparison_period = {
            'start': parse_period_date(comparison_period['start']),
            'end': parse_period_date(comparison_period['end'])
        }

    # Apply filtering
    recent_data = filter_data_by_date_and_conditions(
        data_records,
        filter_conditions,
        start_date=comparison_period['start'] if comparison_period else None,
        end_date=recent_period['end'] if recent_period else None,
        date_field=date_field,
        period_type=period_type
    )

    # Group data with period_type and agg_function
    grouped_data = group_data_by_field_and_date(
        recent_data,
        group_field,
        numeric_field,
        date_field=date_field,
        period_type=period_type,
        agg_function=agg_function  # Pass the aggregation function
    )

    # Get the full list of periods between comparison_period['start'] and recent_period['end']
    if date_field:
        if period_type == 'year':  # Changed from date_field.lower() == 'year'
            # Get full years between start and end, format as YYYY only
            start_year = comparison_period['start'].year
            end_year = recent_period['end'].year
            full_months = [str(year) for year in range(start_year, end_year + 1)]
        else:
            # Get full months between start and end
            full_months = get_month_range(comparison_period['start'], recent_period['end'])
    else:
        full_months = ['All']  # When date_field is not provided

    results = []
    for group_value, data_points in grouped_data.items():
        # Get the first actual data point date for this group
        group_dates = sorted(data_points.keys())
        if not group_dates:
            continue  # Skip groups with no data
            
        first_group_date = group_dates[0]
        
        # Only fill zeros for months that are:
        # 1. After the first actual data point
        # 2. Within the comparison/recent periods
        # 3. Before the last actual data point
        filtered_full_months = [
            month for month in full_months 
            if month >= first_group_date
        ]

        # Now expand data_points only for relevant months
        for month in filtered_full_months:
            if month not in data_points:
                data_points[month] = 0

        dates = sorted(data_points.keys())
        counts = [data_points[date] for date in dates]

        comparison_counts = []
        recent_counts = []

        # Only consider dates after the first actual data point
        for date_key in dates:
            if date_key >= first_group_date:  # Only process dates after first actual data
                count = data_points[date_key]
                if date_field:
                    try:
                        # Use appropriate date format based on period_type
                        if period_type == 'year':
                            # For year period, just compare the years as strings
                            item_year = date_key
                            comp_start_year = str(comparison_period['start'].year)
                            comp_end_year = str(comparison_period['end'].year)
                            recent_start_year = str(recent_period['start'].year)
                            recent_end_year = str(recent_period['end'].year)
                            
                            if comp_start_year <= item_year <= comp_end_year:
                                comparison_counts.append(count)
                            elif recent_start_year <= item_year <= recent_end_year:
                                recent_counts.append(count)
                        else:
                            # For month/day periods, use full date comparison
                            if period_type == 'month':
                                item_date = datetime.datetime.strptime(date_key, "%Y-%m").date()
                            else:  # day
                                item_date = datetime.datetime.strptime(date_key, "%Y-%m-%d").date()
                            
                            if comparison_period['start'] <= item_date <= comparison_period['end']:
                                comparison_counts.append(count)
                            elif recent_period['start'] <= item_date <= recent_period['end']:
                                recent_counts.append(count)
                    except ValueError:
                        continue
                else:
                    comparison_counts = counts
                    recent_counts = counts
                    break

        # Initialize stats with default empty values
        comparison_stats = {'mean': None, 'stdDev': None}
        recent_stats = {'mean': None, 'stdDev': None}

        if comparison_counts:
            comparison_stats = calculate_stats(comparison_counts)
        if recent_counts:
            recent_stats = calculate_stats(recent_counts)

        if (
            comparison_stats['mean'] is not None and
            recent_stats['mean'] is not None and
            comparison_stats['stdDev'] is not None
        ):
            try:
                comparison_mean = float(comparison_stats['mean'])
                recent_mean = float(recent_stats['mean'])
                comparison_std_dev = float(comparison_stats['stdDev'])
                min_diff = float(min_diff)
            except ValueError:
                continue

            difference = recent_mean - comparison_mean

            if comparison_std_dev > 0:
                out_of_bounds = (
                    abs(difference) > comparison_std_dev * min_diff and
                    comparison_mean > 2 and
                    recent_mean > 2
                )

                if True:
                    results.append({
                        'group_value': group_value,
                        'comparison_mean': comparison_mean,
                        'recent_mean': recent_mean,
                        'difference': difference,
                        'stdDev': comparison_std_dev,
                        'dates': dates,
                        'counts': counts,
                        'out_of_bounds': out_of_bounds
                    })

    results.sort(key=lambda x: abs(x['difference']), reverse=True)
    
    metadata = {
        'recent_period': recent_period,
        'comparison_period': comparison_period,
        'group_field': group_field,
        'date_field': date_field,
        'y_axis_label': y_axis_label,
        'title': title or f"{numeric_field} by {group_field}",
        'filter_conditions': filter_conditions,
        'numeric_field': numeric_field,
        'period_type': period_type,
        'agg_function': agg_function,
        'object_type': object_type,   # Add object_type to metadata
        'object_id': object_id,       # Add object_id to metadata
        'object_name': object_name or title or f"{numeric_field} by {group_field}"    # Add object_name to metadata
    }

    # Add executed_query_url to metadata if it exists in context_variables
    if 'executed_query_url' in context_variables:
        metadata['executed_query_url'] = context_variables['executed_query_url']

    # Get script directory and set up output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, '..', 'output')
    
    # Create period-specific directory name
    period_name = 'annual' if period_type == 'year' else 'monthly'
    
    # Get district from filter conditions if it exists
    district = None
    if filter_conditions:
        for condition in filter_conditions:
            if condition.get('field', '').lower() in ['district', 'police_district']:
                district = condition.get('value')
                break
    
    # Default district to "0" if it's null
    if district is None:
        district = 0
    else:
        # Try to convert district to integer
        try:
            district = int(district)
        except (ValueError, TypeError):
            district = 0
    
    # Construct the full output path
    if district:
        output_dir = os.path.join(base_output_dir, period_name, str(district))
    else:
        output_dir = os.path.join(base_output_dir, period_name)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    html_content, markdown_content = generate_anomalies_summary_with_charts(results, metadata, output_dir=output_dir)
    
    # Store anomalies in PostgreSQL database if requested
    if store_in_db:
        try:
            # Use the new store_anomaly_data function from store_anomalies.py
            db_result = store_anomaly_data(
                results=results,
                metadata=metadata,
                db_host=db_host,
                db_port=db_port,
                db_name=db_name,
                db_user=db_user,
                db_password=db_password
            )
        except Exception as e:
            logging.error(f"Error storing anomalies in database: {e}")
    
    return {"anomalies":html_content, "anomalies_markdown":markdown_content}
import os
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import uuid
import logging
from typing import Dict, Any
import sys

# Add current directory to path for direct script execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Handle imports for both direct execution and package import
try:
    # Try relative imports first (for package import)
    from .genAggregate import aggregate_data
    from .anomaly_detection import filter_data_by_date_and_conditions
    from .store_time_series import store_time_series_in_db
    from .db_utils import execute_with_connection
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from genAggregate import aggregate_data
    from anomaly_detection import filter_data_by_date_and_conditions
    from store_time_series import store_time_series_in_db
    from db_utils import execute_with_connection

import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def store_chart_data(
    chart_data,
    metadata,
    db_host=None,
    db_port=None, 
    db_name=None,
    db_user=None,
    db_password=None
) -> Dict[str, Any]:
    """
    Main function to store chart data in the database.
    
    Args:
        chart_data: List of data points for the chart
        metadata: Dictionary with metadata about the chart
        db_host: Optional database host (defaults to env var)
        db_port: Optional database port (defaults to env var)
        db_name: Optional database name (defaults to env var)
        db_user: Optional database user (defaults to env var)
        db_password: Optional database password (defaults to env var)
        
    Returns:
        dict: Result with status and message
    """
    def store_operation(connection):
        try:
            return store_time_series_in_db(connection, chart_data, metadata)
        except Exception as e:
            logging.error(f"Failed to store time series data: {e}")
            return 0
    
    try:
        result = execute_with_connection(
            operation=store_operation,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )
        
        if result["status"] == "success" and result["result"] > 0:
            inserted_count = result["result"]
            return {
                "status": "success",
                "message": f"Successfully stored {inserted_count} data points in the database",
                "records_stored": inserted_count
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to store data in database: {result.get('error', 'Unknown error')}",
                "records_stored": 0
            }
    except Exception as e:
        logging.error(f"Error in store_chart_data: {e}")
        return {
            "status": "error",
            "message": f"Failed to store data in database: {str(e)}",
            "records_stored": 0
        }

def generate_time_series_chart(
    context_variables: dict,
    time_series_field: str,
    numeric_fields,
    aggregation_period: str = 'day',
    group_field: str = None,
    agg_functions: dict = None,
    max_legend_items: int = 10,
    filter_conditions: dict = None,
    null_group_label: str = 'NA',
    show_average_line: bool = False,
    y_axis_min: float = 0,
    y_axis_max: float = None,
    return_html: bool = False,
    output_dir: str = None,
    store_in_db: bool = False,
    object_type: str = None,
    object_id: str = None,
    object_name: str = None,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None,
    skip_aggregation: bool = False
) -> str:
    try:
        logging.info("Full context_variables: %s", context_variables)

        if isinstance(numeric_fields, str):
            numeric_fields = [numeric_fields]
        logging.debug("Numeric fields: %s", numeric_fields)

        logging.debug("Context Variables: %s", context_variables)

        # Retrieve title and y_axis_label from context_variables
        chart_title = context_variables.get("chart_title", "Time Series Chart")

        # Get y_axis_label from context_variables, with better fallback logic
        y_axis_label = context_variables.get("y_axis_label")
        if not y_axis_label:
            # Try to get it from the first numeric field name
            if numeric_fields and len(numeric_fields) > 0:
                field_name = numeric_fields[0].lower().replace('_', ' ')
                y_axis_label = "count" if field_name == "item count" else numeric_fields[0].capitalize()
                logging.info(f"No y_axis_label provided in context_variables, using derived label: {y_axis_label}")
            else:
                y_axis_label = "Value"
                logging.info(f"No y_axis_label or numeric fields provided, using default: {y_axis_label}")
        else:
            logging.info(f"Using provided y_axis_label from context_variables: {y_axis_label}")

        noun = context_variables.get("noun", y_axis_label)
        # Create a copy of the dataset to avoid modifying the original data
        original_df = context_variables.get("dataset")
        if original_df is None or original_df.empty:
            logging.error("Dataset is not available or is empty.")
            return "**Error**: Dataset is missing or empty. Please provide a valid dataset in 'context_variables'."

        df = original_df.copy()
        logging.debug("DataFrame copied to avoid in-place modifications.")

        # Store original column names for later reference
        original_columns = df.columns.tolist()
        column_mapping = {col.lower(): col for col in original_columns}
        logging.debug("Original column mapping: %s", column_mapping)
        
        # Convert columns to lowercase for case-insensitive comparison
        df.columns = df.columns.str.lower()
        logging.debug("Standardized DataFrame columns: %s", df.columns.tolist())
        
        # Convert input fields to lowercase for comparison
        time_series_field = time_series_field.lower()
        numeric_fields = [field.lower() for field in numeric_fields]
        if group_field:
            group_field = group_field.lower()
            logging.debug(f"Lowercased group_field: {group_field}")

        # Check for fields using case-insensitive comparison
        required_fields = [time_series_field] + numeric_fields
        if group_field:
            required_fields.append(group_field)
        missing_fields = [field for field in required_fields if field not in df.columns]

        if missing_fields:
            logging.error("Missing fields in DataFrame: %s. Available columns: %s", missing_fields, df.columns.tolist())
            return f"**Error**: Missing required fields in DataFrame: {missing_fields}. Please check the column names and ensure they are present."

        # Apply filter conditions if provided
        if filter_conditions:
            # Convert time_series_field to string if we have string year filters
            year_filters = [f for f in filter_conditions if f['field'].lower() == time_series_field.lower() and isinstance(f['value'], str) and f['value'].isdigit()]
            if year_filters and 'year' in time_series_field.lower():
                df[time_series_field] = df[time_series_field].astype(str)
                logging.info(f"Converted {time_series_field} to string type for year comparisons")
            
            data_records = df.to_dict('records')
            # Apply filter_data_by_date_and_conditions
            # Determine period_type based on time_series_field and aggregation_period
            period_type = 'month'  # default
            if 'year' in time_series_field.lower() or aggregation_period == 'year':
                period_type = 'year'
            elif aggregation_period == 'day':
                period_type = 'day'
            
            filtered_data = filter_data_by_date_and_conditions(
                data_records,
                filter_conditions,
                start_date=None,
                end_date=None,
                date_field=time_series_field,
                period_type=period_type
            )
            # Convert back to DataFrame
            df = pd.DataFrame(filtered_data)
            logging.info(f"Filtered data size: {len(df)} records after applying filters.")
            if df.empty:
                logging.error("No data available after applying filters.")
                return "**Error**: No data available after applying filters. Please adjust your filter conditions."

        # Convert numeric fields to proper numeric types
        for field in numeric_fields:
            if skip_aggregation:
                # Skip conversion if data is already processed (for weekly analysis)
                logging.info(f"Skipping numeric conversion for {field} due to skip_aggregation=True")
                continue
                
            pre_conversion_count = df[field].notna().sum()
            df[field] = df[field].astype(str).str.strip()
            df[field] = pd.to_numeric(df[field], errors='coerce')
            
            # Convert to integer if all values have no decimal places
            if df[field].notna().all() and (df[field] % 1 == 0).all():
                df[field] = df[field].astype(int)
                
            post_conversion_count = df[field].notna().sum()
            coerced_count = pre_conversion_count - post_conversion_count

            if coerced_count > 0:
                logging.info("Field '%s': %d values were coerced to NaN during conversion to numeric.", field, coerced_count)
                if 'id' in df.columns:
                    coerced_values = df[df[field].isna()][['id', field]].head(20)
                    logging.info("Sample of coerced values in field '%s' (ID and Value):\n%s", field, coerced_values)
                else:
                    coerced_values = df[df[field].isna()].head(20)
                    logging.info("Sample of coerced values in field '%s' (Full Row):\n%s", field, coerced_values)

        # Drop rows with NaN in numeric fields
        df.dropna(subset=numeric_fields, inplace=True)
        logging.info("Dropped NA values from DataFrame.")
        
        if group_field and null_group_label:
            df[group_field] = df[group_field].fillna(null_group_label)
            df[group_field] = df[group_field].replace('', null_group_label)
        
        if agg_functions is None:
            agg_functions = {
                field: 'mean' if any(field.endswith(suffix) for suffix in ['_avg', '_pct']) else 'sum'
                for field in numeric_fields
            }
        logging.info("Aggregation functions: %s", agg_functions)

        if skip_aggregation:
            # Skip aggregation - use data as-is (for weekly analysis where data is already aggregated)
            logging.info("Skipping aggregation - using data as-is")
            aggregated_df = df.copy()
            # Rename the time field to time_period for consistency
            if time_series_field in aggregated_df.columns:
                aggregated_df = aggregated_df.rename(columns={time_series_field: 'time_period'})
            else:
                logging.error(f"Time series field '{time_series_field}' not found in dataset")
                return f"**Error**: Time series field '{time_series_field}' not found in dataset"
            # Debug: Log data types and values when skipping aggregation
            logging.info(f"[genChart] Data sample after skip_aggregation:\n{aggregated_df.head(10).to_string()}")
            logging.info(f"[genChart] Data types after skip_aggregation:\n{aggregated_df.dtypes}")
            for col in aggregated_df.columns:
                if col in numeric_fields:
                    logging.info(f"  {col} sample values: {aggregated_df[col].head().tolist()}")
                    logging.info(f"  {col} unique values: {aggregated_df[col].unique()[:5]}")
        else:
            # Perform normal aggregation
            aggregated_df = aggregate_data(
                df=df,
                time_series_field=time_series_field,
                numeric_fields=numeric_fields,
                aggregation_period=aggregation_period,
                group_field=group_field,
                agg_functions=agg_functions
            )

        if 'time_period' not in aggregated_df.columns:
            logging.error("'time_period' column is missing after aggregation.")
            return "**Error**: The 'time_period' column is missing after aggregation. Check the 'aggregate_data' function for proper time grouping."

        # Handle time_period based on aggregation period
        if aggregation_period == 'week':
            # For weekly aggregation, check if time_period is in ISO week format (YYYY-Www)
            if aggregated_df['time_period'].dtype == 'object' and aggregated_df['time_period'].str.contains(r'^\d{4}-W\d{2}$', na=False).all():
                # ISO week format - keep as string, but create a datetime column for sorting
                logging.info("Detected ISO week format in time_period, keeping as string for display")
                # Create a datetime column for sorting (use the week_start if available, otherwise parse)
                if 'week_start' in aggregated_df.columns:
                    aggregated_df['time_period_dt'] = aggregated_df['week_start']
                else:
                    # Parse ISO week to get the Monday of that week for sorting
                    def parse_iso_week(iso_week_str):
                        try:
                            year, week = iso_week_str.split('-W')
                            year, week = int(year), int(week)
                            # Get the Monday of that ISO week
                            return date.fromisocalendar(year, week, 1)
                        except:
                            return pd.NaT
                    
                    aggregated_df['time_period_dt'] = aggregated_df['time_period'].apply(parse_iso_week)
                
                # Sort by the datetime column
                aggregated_df = aggregated_df.sort_values('time_period_dt')
            else:
                # Regular datetime format - convert to datetime
                aggregated_df['time_period'] = pd.to_datetime(aggregated_df['time_period'])
                aggregated_df['time_period_dt'] = aggregated_df['time_period']
        else:
            # For other aggregation periods, convert to datetime as before
            aggregated_df['time_period'] = pd.to_datetime(aggregated_df['time_period'])
            aggregated_df['time_period_dt'] = aggregated_df['time_period']
        
        logging.debug("Aggregated DataFrame sorted by 'time_period'.")
        
        # Compute values for the caption
        try:
            # Use time_period_dt for calculations, time_period for display
            time_period_col = 'time_period_dt' if 'time_period_dt' in aggregated_df.columns else 'time_period'
            display_col = 'time_period' if aggregation_period == 'week' and aggregated_df['time_period'].dtype == 'object' else 'time_period'
            
            last_time_period = aggregated_df[time_period_col].max()
            earliest_time_period = aggregated_df[time_period_col].min()

            # Get the appropriate time period name based on aggregation_period
            if aggregation_period == 'year':
                period_name = last_time_period.strftime('%Y')
            elif aggregation_period == 'month':
                period_name = last_time_period.strftime('%B %Y')
            elif aggregation_period == 'quarter':
                quarter = (last_time_period.month - 1) // 3 + 1
                period_name = f"Q{quarter} {last_time_period.year}"
            elif aggregation_period == 'week':
                # For ISO week format, get the display value
                if display_col == 'time_period' and aggregated_df['time_period'].dtype == 'object':
                    # Get the last ISO week string
                    last_iso_week = aggregated_df.loc[aggregated_df[time_period_col] == last_time_period, 'time_period'].iloc[0]
                    period_name = f"ISO Week {last_iso_week}"
                else:
                    period_name = f"Week {last_time_period.strftime('%U')} of {last_time_period.year}"
            elif aggregation_period == 'day':  # Specific handling for daily data
                period_name = last_time_period.strftime('%B %d, %Y')
            else:  # fallback for any other period
                period_name = last_time_period.strftime('%Y-%m-%d')

            logging.debug(f"Last time period: {last_time_period}, Period name: {period_name}")

            # Format numbers for caption
            def format_number(num):
                if num >= 1:
                    return f"{round(num):,}"
                return f"{num:.2f}"

            # Get the last period data using the datetime column for filtering
            last_period_mask = aggregated_df[time_period_col] == last_time_period
            total_latest = format_number(aggregated_df[last_period_mask][numeric_fields[0]].sum())
            logging.debug(f"Total value for last period: {total_latest}")

            # Exclude the last period for calculating the average of the rest across all groups
            rest_periods = aggregated_df[aggregated_df[time_period_col] < last_time_period]
            total_periods = 0  # Initialize with default value
            if rest_periods.empty:
                average_of_rest = 0
            else:
                # Sum over groups to get total per time period
                total_per_time_period = rest_periods.groupby(time_period_col)[numeric_fields[0]].sum()
                average_of_rest = total_per_time_period.mean()
                total_periods = len(rest_periods[time_period_col].unique())

            formatted_average = format_number(average_of_rest)
            logging.debug(f"Average value of rest periods: {formatted_average}")

            percentage_diff_total = ((float(total_latest.replace(',', '')) - average_of_rest) / average_of_rest) * 100 if average_of_rest != 0 else 0
            above_below_total = 'above' if float(total_latest.replace(',', '')) > average_of_rest else 'below'
            percentage_diff_total = abs(round(percentage_diff_total))
            y_axis_label_lower = y_axis_label.lower()

            caption_total = f"In {period_name}, {y_axis_label_lower} was {total_latest}, which is {percentage_diff_total}% {above_below_total} the {total_periods} {aggregation_period} average of {formatted_average}."
            logging.info(f"Caption for total: {caption_total}")

            # Caption for charts with a group_field
            caption_group = ""
            if group_field:
                try:
                    last_period_df = aggregated_df[last_period_mask]
                    numeric_values = last_period_df.groupby(group_field)[numeric_fields[0]].sum().to_dict()
                    logging.debug(f"Numeric values for last period by group: {numeric_values}")

                    # Calculate the average of the prior periods for each group
                    prior_periods = aggregated_df[aggregated_df[time_period_col] < last_time_period]
                    average_of_prior = prior_periods.groupby(group_field)[numeric_fields[0]].mean().to_dict()
                    logging.debug(f"Average values of prior periods by group: {average_of_prior}")

                    # Sort groups by their latest values to show most significant first
                    sorted_groups = sorted(numeric_values.items(), key=lambda x: x[1], reverse=True)
                    
                    captions_group = []
                    # Limit to top 5 groups to avoid overly long captions
                    for grp, value in sorted_groups[:5]:
                        if grp not in average_of_prior or average_of_prior[grp] == 0:
                            continue
                            
                        percentage_diff_group = ((value - average_of_prior[grp]) / average_of_prior[grp]) * 100
                        above_below_group = 'above' if value > average_of_prior[grp] else 'below'
                        percentage_diff_group = abs(round(percentage_diff_group))
                        
                        formatted_value = format_number(value)
                        formatted_avg = format_number(average_of_prior[grp])
                        
                        captions_group.append(
                            f"For {grp}, in {period_name}, there were {formatted_value} {y_axis_label_lower}, "
                            f"which is {percentage_diff_group}% {above_below_group} the {total_periods} {aggregation_period} average "
                            f"of {formatted_avg}."
                        )
                    
                    if len(sorted_groups) > 5:
                        captions_group.append(f"... and {len(sorted_groups) - 5} more groups.")
                        
                    caption_group = "<br>".join(captions_group)
                    logging.info(f"Generated group captions: {caption_group}")
                    
                except Exception as e:
                    logging.error(f"Error generating group captions: {e}")
                    caption_group = "Error generating group details."

            if group_field:
                caption = f"{caption_total}\n\n{caption_group}"
            else:
                caption = caption_total

            # ---------------------------------------------------------------------
            #  NEW LOGIC FOR MORE THAN 2 YEARS OF DATA
            # ---------------------------------------------------------------------
            # Fix the calculation of time span to handle pandas datetime objects correctly
            time_span_days = (last_time_period - earliest_time_period).total_seconds() / (60 * 60 * 24)
            time_span_years = time_span_days / 365
            
            if time_span_years > 2:
                try:
                    last_year_num = last_time_period.year
                    last_month_num = last_time_period.month
                    prior_year_num = last_year_num - 1

                    # For yearly charts, we compare full years without YTD label
                    is_yearly = time_series_field == 'year'

                    # Filter for current year, through last month in data
                    mask_current_year = (
                        aggregated_df['time_period'].dt.year == last_year_num
                    )
                    if not is_yearly:
                        mask_current_year &= (
                            aggregated_df['time_period'].dt.month <= last_month_num
                        )
                    current_year_df = aggregated_df[mask_current_year]

                    # Filter for prior year, through the same last month
                    mask_prior_year = (
                        aggregated_df['time_period'].dt.year == prior_year_num
                    )
                    if not is_yearly:
                        mask_prior_year &= (
                            aggregated_df['time_period'].dt.month <= last_month_num
                        )
                    prior_year_df = aggregated_df[mask_prior_year]

                    ytd_captions = []
                    
                    # Overall comparison
                    current_year_sum = current_year_df[numeric_fields[0]].sum()
                    prior_year_sum = prior_year_df[numeric_fields[0]].sum()

                    if prior_year_sum != 0:
                        ytd_diff_pct = ((current_year_sum - prior_year_sum) / prior_year_sum) * 100
                        ytd_diff_pct = round(ytd_diff_pct)
                        above_below_ytd = 'above' if current_year_sum > prior_year_sum else 'below'

                        # Adjust caption based on whether it's a yearly chart
                        if is_yearly:
                            ytd_captions.append(
                                f"In {last_year_num}, total {y_axis_label_lower} was {format_number(current_year_sum)}, "
                                f"which is {abs(ytd_diff_pct)}% {above_below_ytd} the {prior_year_num} total of {format_number(prior_year_sum)}."
                            )
                        else:
                            ytd_captions.append(
                                f"As of the end of {period_name}, YTD {last_year_num}, total {y_axis_label_lower} is {format_number(current_year_sum)}, "
                                f"which is {abs(ytd_diff_pct)}% {above_below_ytd} the YTD {prior_year_num} total of {format_number(prior_year_sum)}."
                            )

                    # Group-specific comparisons if group_field exists
                    if group_field:
                        # Get top groups by current year total
                        current_year_totals = current_year_df.groupby(group_field)[numeric_fields[0]].sum()
                        top_groups = current_year_totals.sort_values(ascending=False).head(5).index

                        for group in top_groups:
                            curr_group_sum = current_year_df[current_year_df[group_field] == group][numeric_fields[0]].sum()
                            prior_group_sum = prior_year_df[prior_year_df[group_field] == group][numeric_fields[0]].sum()

                            if prior_group_sum != 0:
                                group_ytd_diff_pct = ((curr_group_sum - prior_group_sum) / prior_group_sum) * 100
                                group_ytd_diff_pct = round(group_ytd_diff_pct)
                                group_above_below = 'above' if curr_group_sum > prior_group_sum else 'below'

                                # Adjust group caption based on whether it's a yearly chart
                                if is_yearly:
                                    ytd_captions.append(
                                        f"<br>For {group}, {last_year_num} {y_axis_label_lower} was {format_number(curr_group_sum)}, "
                                        f"which is {abs(group_ytd_diff_pct)}% {group_above_below} the {prior_year_num} total "
                                        f"of {format_number(prior_group_sum)}."
                                    )
                                else:
                                    ytd_captions.append(
                                        f"<br>For {group}, YTD {last_year_num} {y_axis_label_lower} is {format_number(curr_group_sum)}, "
                                        f"which is {abs(group_ytd_diff_pct)}% {group_above_below} the YTD {prior_year_num} total "
                                        f"of {format_number(prior_group_sum)}."
                                    )

                    ytd_caption = "\n".join(ytd_captions)
                    caption = f"{caption}\n\n{ytd_caption}"

                except Exception as ytd_err:
                    logging.warning(f"Failed to compute YTD comparison: {ytd_err}")

        except Exception as e:
            logging.error("Failed to compute caption values: %s", e)
            caption = ""
        
        # Store chart data in the database if requested
        if store_in_db:
            try:
                # Add executed_query_url and caption to metadata
                metadata = {
                    "chart_title": context_variables.get("chart_title", "Time Series Chart"),
                    "y_axis_label": context_variables.get("y_axis_label", numeric_fields[0]),
                    "aggregation_period": aggregation_period,
                    "filter_conditions": filter_conditions or [],
                    "object_type": object_type or context_variables.get("object_type", "unknown"),
                    "object_id": object_id or context_variables.get("object_id", "unknown"),
                    "object_name": object_name or context_variables.get("object_name", "unknown"),
                    "time_series_field": time_series_field,
                    "numeric_fields": numeric_fields,
                    "group_field": group_field,
                    "field_name": group_field or numeric_fields[0] if numeric_fields else "unknown",
                    "executed_query_url": context_variables.get("executed_query_url"),
                    "caption": caption,
                    "period_type": aggregation_period  # Add the period_type to metadata
                }
                
                # Prepare data points in the correct format for database storage
                chart_data = []
                for _, row in aggregated_df.iterrows():
                    data_point = {
                        'time_period': row['time_period'],
                        'value': row[numeric_fields[0]]  # Use the first numeric field as the value
                    }
                    
                    # Add group_value if group_field exists
                    if group_field:
                        data_point['group_value'] = row[group_field]
                        
                    chart_data.append(data_point)
                
                # Use the new store_chart_data function
                db_result = store_chart_data(
                    chart_data=chart_data,
                    metadata=metadata,
                    db_host=db_host,
                    db_port=db_port,
                    db_name=db_name,
                    db_user=db_user,
                    db_password=db_password
                )
                logging.info(f"Database storage result: {db_result['message']}")
            except Exception as db_error:
                logging.error(f"Database operation failed: {db_error}")
                return f"**Error**: Failed to store chart data: {str(db_error)}"
        
        # Limit legend to top max_legend_items
        if group_field:
            group_totals = aggregated_df.groupby(group_field)[numeric_fields].sum().sum(axis=1)
            top_groups = group_totals.sort_values(ascending=False).head(max_legend_items).index.tolist()
            logging.info("Top groups based on total values: %s", top_groups)

            # Create mask for top groups
            mask_top = aggregated_df[group_field].isin(top_groups)
            
            # Get the filtered data for top groups
            filtered_agg = aggregated_df[mask_top].copy()
            
            # Sum all other groups per time period
            others_df = (aggregated_df[~mask_top]
                        .groupby('time_period')[numeric_fields[0]]
                        .sum()
                        .reset_index())
            
            # Add the group field with "Others" value
            if not others_df.empty:
                others_df[group_field] = 'Others'
                filtered_agg = pd.concat([filtered_agg, others_df], ignore_index=True)
            
            aggregated_df = filtered_agg

        # Create output directory if needed
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if output_dir:
            chart_dir = output_dir
            os.makedirs(chart_dir, exist_ok=True)

        try:
            if group_field:
                group_field_original = column_mapping.get(group_field, group_field)
                logging.debug(f"Original group field name from mapping: {group_field_original}")
                fig = px.area(
                    aggregated_df,
                    x='time_period',  # Use the display column (string for ISO weeks)
                    y=numeric_fields[0],
                    color=group_field,
                    labels={
                        'time_period': time_series_field.capitalize(),
                        numeric_fields[0]: y_axis_label,
                        group_field: group_field_original.capitalize()
                    }
                )
                # Update area styling
                fig.update_traces(
                    line=dict(width=1),
                    opacity=0.8
                )
            else:
                fig = px.line(
                    aggregated_df,
                    x='time_period',  # Use the display column (string for ISO weeks)
                    y=numeric_fields[0],
                    labels={
                        'time_period': time_series_field.capitalize(),
                        numeric_fields[0]: y_axis_label
                    }
                )
                # Add markers to the single line
                fig.update_traces(
                    mode='lines+markers',
                    marker=dict(
                        size=6,
                        opacity=0.6,
                        line=dict(width=1)
                    )
                )

            # Calculate y-axis range
            y_min = y_axis_min  # This will be 0 by default from the function parameters
            if y_axis_max is None:
                if group_field:
                    # For stacked area charts, calculate the total height at each time period
                    total_by_period = aggregated_df.groupby('time_period')[numeric_fields[0]].sum()
                    y_max = total_by_period.max() * 1.1  # Add 10% padding to the maximum total
                else:
                    # For single line charts, use the raw maximum value
                    y_max = aggregated_df[numeric_fields[0]].max() * 1.1
            else:
                y_max = y_axis_max

            # Update the layout configuration for x-axis ticks and y-axis range
            fig.update_layout(
                yaxis=dict(
                    title=dict(
                        text=y_axis_label,
                        font=dict(size=14, family='Arial', color='black')
                    ),
                    tickfont=dict(size=10, family='Arial', color='black'),
                    range=[y_min, y_max],
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='lightgrey',
                    fixedrange=True
                ),
                xaxis=dict(
                    title=dict(font=dict(size=14, family='Arial', color='black')),
                    tickfont=dict(size=10, family='Arial', color='black'),
                    tickformat='%Y' if time_series_field == 'year' 
                              else '%m-%y' if aggregation_period == 'month'
                              else '%d-%m-%y' if aggregation_period == 'day'
                              else '%b %d' if aggregation_period == 'week'
                              else '%m-%y',
                    dtick="M12" if aggregation_period in ['year', 'month'] else "D7",  # One tick per year for year/month, weekly for days
                    tickangle=45 if aggregation_period == 'day' else 0,  # Angled labels for daily data
                    tickmode='array',
                    ticktext=[d.strftime('%Y' if time_series_field == 'year' 
                                       else '%m-%y' if aggregation_period == 'month'
                                       else '%d-%m-%y' if aggregation_period == 'day'
                                       else '%b %d' if aggregation_period == 'week'
                                       else '%m-%y') if hasattr(d, 'strftime') else str(d)
                             for d in aggregated_df['time_period'].unique()],
                    tickvals=aggregated_df['time_period'].unique()
                )
            )

            # Add average line if show_average_line is True
            if show_average_line:
                # Calculate average excluding the last month to match caption
                time_period_col = 'time_period_dt' if 'time_period_dt' in aggregated_df.columns else 'time_period'
                last_period = aggregated_df[time_period_col].max()
                prior_periods_df = aggregated_df[aggregated_df[time_period_col] < last_period]
                
                # Calculate average the same way as in caption
                total_per_time_period = prior_periods_df.groupby(time_period_col)[numeric_fields[0]].sum()
                average_value = total_per_time_period.mean()
                
                # Create a series for the average line - use display column for x-axis
                display_col = 'time_period'
                average_line = pd.Series(average_value, index=aggregated_df[display_col])
                
                # Format the average value
                formatted_avg = (
                    f"{average_value:,.0f}" if average_value >= 1 
                    else f"{average_value:.1f}"
                )
                
                fig.add_scatter(
                    x=average_line.index, 
                    y=average_line.values, 
                    mode='lines+text', 
                    name=f'Prior periods Average ({formatted_avg})',
                    line=dict(width=2, color='blue', dash='dash'),
                    text=[f"AVG: {formatted_avg}" if i == len(average_line)-1 else "" 
                          for i in range(len(average_line))],
                    textposition="middle right",
                    textfont=dict(size=10, color='blue')
                )

            # Calculate bottom margin based on whether we have a group field
            bottom_margin = 30  # Reduced margin since we're hiding legend for multi-series

            fig.update_layout(
                showlegend=not group_field,  # Hide legend for multi-series charts
                legend=dict(
                    orientation="h",    # Horizontal orientation
                    yanchor="bottom",
                    y=-0.3,            # Places legend further below the plot
                    xanchor="center",
                    x=0.5,             # Centers the legend horizontally
                    font=dict(size=8),
                    title=dict(
                        text=column_mapping.get(group_field, '').capitalize() if group_field else '',
                        side='left',  # Can be 'top', 'left', etc.
                        font=dict(size=8)  # Optional: control title font separately
                    )
                ),
                title={
                    'text': f"{chart_title} <BR>" if group_field else chart_title,
                    'y': 0.95,
                    'x': 0.5,
                    'font': dict(
                        family='Arial',
                        size=16,
                        color='black',
                        weight='bold'
                    )
                },
                xaxis=dict(
                    title=dict(font=dict(size=14, family='Arial', color='black')),
                    tickfont=dict(size=10, family='Arial', color='black')
                ),
                yaxis=dict(
                    title=dict(
                        text=y_axis_label,
                        font=dict(size=14, family='Arial', color='black')
                    ),
                    tickfont=dict(size=10, family='Arial', color='black')
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial", size=10, color="black"),
                autosize=True,
                margin=dict(l=50, r=50, t=80, b=bottom_margin)  # Dynamic bottom margin
            )

            # Highlight the last data point (only if no group_field)
            if not group_field and not aggregated_df.empty:
                last_point = aggregated_df.iloc[-1]
                last_x = last_point['time_period']  # Use display column
                last_y = last_point[numeric_fields[0]]
                
                # Use the same format as x-axis labels
                if hasattr(last_x, 'strftime'):
                    point_label = last_x.strftime('%Y' if time_series_field == 'year' 
                                                else '%m-%y' if aggregation_period == 'month'
                                                else '%b %d' if aggregation_period == 'week'
                                                else '%m-%y')
                else:
                    point_label = str(last_x)

                fig.add_scatter(
                    x=[last_x],
                    y=[last_y],
                    mode='markers',
                    name=point_label,
                    marker=dict(
                        size=12,
                        color='gold',
                        symbol='circle-open',
                        line=dict(width=2, color='gold')
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )

                fig.add_annotation(
                    x=last_x,
                    y=last_y,
                    text=f"{point_label}<br>{last_y:,} {y_axis_label}",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=12, family='Arial', color='#333'),
                    arrowcolor='gold',
                    arrowwidth=1,
                    bgcolor='rgba(255, 255, 0, 0.7)',
                    bordercolor='gold',
                    borderwidth=1,
                    ax=-60,
                    ay=-20,
                )

            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey')

            # Add filter conditions annotation
            if filter_conditions:
                filter_conditions_str = ', '.join([f"{cond['field']} {cond['operator']} {cond['value']}" for cond in filter_conditions])
            else:
                filter_conditions_str = "No filter conditions provided"
            fig.add_annotation(
                text=f"Filter Conditions: {filter_conditions_str}",
                xref="paper", yref="paper",
                x=0.0, y=-0.15,
                showarrow=False,
                font=dict(size=8, family='Arial', color='black'),
                xanchor='left'
            )

            # Generate a unique chart ID without saving an image
            chart_id = uuid.uuid4().hex[:6]
            logging.info("Not saving chart as image, only generating chart ID")

            # Prepare crosstabbed data
            if group_field:
                crosstab_df = aggregated_df.pivot_table(
                    index=group_field,
                    columns='time_period',
                    values=numeric_fields[0],
                    aggfunc='sum',
                    fill_value=0
                )
                # Format the date columns for better readability based on aggregation period
                crosstab_df.columns = [col.strftime('%Y' if aggregation_period == 'year'
                                                   else '%b %Y' if aggregation_period == 'month'
                                                   else '%d %b %Y' if aggregation_period == 'day'
                                                   else '%b %d' if aggregation_period == 'week'
                                                   else '%Y') if hasattr(col, 'strftime') else str(col)
                                     for col in crosstab_df.columns]
            else:
                crosstab_df = aggregated_df.set_index('time_period')[[numeric_fields[0]]].T
                crosstab_df.columns = [col.strftime('%Y' if aggregation_period == 'year'
                                                   else '%b %Y' if aggregation_period == 'month'
                                                   else '%d %b %Y' if aggregation_period == 'day'
                                                   else '%b %d' if aggregation_period == 'week'
                                                   else '%Y') if hasattr(col, 'strftime') else str(col)
                                     for col in crosstab_df.columns]
                crosstab_df.index = [y_axis_label]

            # Convert crosstab to HTML table
            html_table = crosstab_df.to_html(classes='data-table', index=True)

            # Create markdown content without image reference
            markdown_content = f""" 
{context_variables.get("chart_title", "Time Series Chart")}
Caption: {caption}

### Data Table
{crosstab_df.to_markdown()}

"""

            # Include crosstab data table with toggle in HTML content
            html_content = f'''
<div style="width:100%" id="chart_{chart_id}">
    <div style="width:100%; max-width:1200px;">
        {fig.to_html(full_html=False)}
    </div>
    <div> 
        {caption}
    </div>
    <p>
        <a href="javascript:void(0);" onclick="toggleDataTable('data_table_{chart_id}')">Show Data</a>
    </p>
    <div id="data_table_{chart_id}" style="display:none;">
        {html_table}
    </div>
    
</div>

<script>
function toggleDataTable(tableId) {{
    var table = document.getElementById(tableId);
    if (table.style.display === "none") {{
        table.style.display = "block";
    }} else {{
        table.style.display = "none";
    }}
}}
</script>

<style>
.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}}
.data-table th, .data-table td {{
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
}}
.data-table tr:nth-child(even) {{
    background-color: #f9f9f9;
}}
</style>
'''

            logging.info("Markdown content created with chart ID: %s", chart_id)

            if return_html:
                return markdown_content, html_content
            else:
                return markdown_content

        except Exception as e:
            logging.error("Failed to generate or save chart: %s", e)
            return f"**Error**: An unexpected error occurred while generating the chart: {e}"

    except ValueError as ve:
        logging.error("ValueError in generate_time_series_chart: %s", ve)
        return f"**Error**: {ve}"

    except Exception as e:
        logging.error("Unexpected error in generate_time_series_chart: %s", e)
        return f"**Error**: An unexpected error occurred: {e}"

def generate_ytd_trend_chart(
    trend_data: dict,
    metadata: dict,
    district: str = None,
    return_html: bool = False,
    output_dir: str = None,
    store_in_db: bool = False,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None
) -> str:
    """
    Generate a specialized YTD trend chart for district-specific trend data.
    Shows two lines: one for this year and one for last year for easy comparison.
    
    Args:
        trend_data: Dictionary with date keys and numeric values (e.g., {"2024-01-01": 10.5})
        metadata: Dictionary with chart metadata
        district: District identifier (e.g., "1", "2", etc., or None for citywide)
        return_html: Whether to return HTML or save to file
        output_dir: Directory to save the chart file
        store_in_db: Whether to store the chart data in the database
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        str: HTML content or file path
    """
    try:
        logging.info(f"Generating YTD trend chart for district {district or 'citywide'}")
        
        # Convert trend_data dictionary to DataFrame
        df = pd.DataFrame([
            {'time_period': date, 'value': float(value)} 
            for date, value in trend_data.items()
        ])
        
        if df.empty:
            logging.error("No trend data provided")
            return "**Error**: No trend data provided"
        
        # Convert time_period to datetime
        df['time_period'] = pd.to_datetime(df['time_period'])
        df = df.sort_values('time_period')
        
        # Extract year and day-of-year for comparison
        df['year'] = df['time_period'].dt.year
        df['day_of_year'] = df['time_period'].dt.dayofyear
        
        # Determine this year and last year
        this_year = df['year'].max()
        last_year = this_year - 1
        
        logging.info(f"Data spans years: {df['year'].min()} to {df['year'].max()}")
        logging.info(f"This year: {this_year}, Last year: {last_year}")
        
        # Create separate dataframes for each year
        this_year_data = df[df['year'] == this_year].copy()
        last_year_data = df[df['year'] == last_year].copy()
        
        logging.info(f"This year data points: {len(this_year_data)}")
        logging.info(f"Last year data points: {len(last_year_data)}")
        
        if not this_year_data.empty:
            logging.info(f"This year date range: {this_year_data['time_period'].min()} to {this_year_data['time_period'].max()}")
        if not last_year_data.empty:
            logging.info(f"Last year date range: {last_year_data['time_period'].min()} to {last_year_data['time_period'].max()}")
        
        # Extract metadata
        chart_title = metadata.get('chart_title', 'YTD Trend Chart')
        y_axis_label = metadata.get('y_axis_label', 'Value')
        object_type = metadata.get('object_type', 'metric')
        object_id = metadata.get('object_id', 'unknown')
        object_name = metadata.get('object_name', chart_title)
        field_name = metadata.get('field_name', 'trend_value')
        
        # Create district-specific title and description
        district_name = f"District {district}" if district and district != '0' else "Citywide"
        full_title = f"{chart_title} - {district_name}"
        
        # Create the chart using Plotly with two lines
        fig = go.Figure()
        
        # Add this year line
        if not this_year_data.empty:
            fig.add_trace(go.Scatter(
                x=this_year_data['day_of_year'],
                y=this_year_data['value'],
                mode='lines+markers',
                name=f'{this_year}',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
                hovertemplate='<b>Day %{x}</b><br>' +
                             f'{this_year}: %{{y:.2f}}<br>' +
                             '<extra></extra>'
            ))
        
        # Add last year line
        if not last_year_data.empty:
            fig.add_trace(go.Scatter(
                x=last_year_data['day_of_year'],
                y=last_year_data['value'],
                mode='lines+markers',
                name=f'{last_year}',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4),
                hovertemplate='<b>Day %{x}</b><br>' +
                             f'{last_year}: %{{y:.2f}}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=full_title,
            title_x=0.5,  # Center the title
            title_font_size=16,
            xaxis_title="Day of Year",
            yaxis_title=y_axis_label,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Customize axes
        fig.update_xaxes(
            gridcolor='lightgray',
            gridwidth=0.5,
            showgrid=True,
            range=[1, 366]  # Full year range
        )
        fig.update_yaxes(
            gridcolor='lightgray',
            gridwidth=0.5,
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        )
        
        # Store in database if requested
        if store_in_db:
            try:
                # Prepare metadata for database storage
                db_metadata = {
                    "chart_title": full_title,
                    "y_axis_label": y_axis_label,
                    "aggregation_period": "day",  # YTD trends are daily
                    "filter_conditions": [{"field": "district", "operator": "=", "value": district}] if district else [],
                    "object_type": object_type,
                    "object_id": object_id,
                    "object_name": object_name,
                    "time_series_field": "time_period",
                    "numeric_fields": ["value"],
                    "field_name": field_name,
                    "executed_query_url": metadata.get("executed_query_url"),
                    "caption": metadata.get("caption", ""),
                    "period_type": "day",
                    "group_field": "year"  # Add group_field to indicate this is grouped by year
                }
                
                # Prepare data points for database storage with proper year grouping
                chart_data = []
                
                # Add this year data with group_value
                for _, row in this_year_data.iterrows():
                    data_point = {
                        'time_period': row['time_period'].date(),
                        'value': row['value'],
                        'group_value': str(this_year)  # Use year as group_value
                    }
                    chart_data.append(data_point)
                
                # Add last year data with group_value
                for _, row in last_year_data.iterrows():
                    data_point = {
                        'time_period': row['time_period'].date(),
                        'value': row['value'],
                        'group_value': str(last_year)  # Use year as group_value
                    }
                    chart_data.append(data_point)
                
                # Store in database
                db_result = store_chart_data(
                    chart_data=chart_data,
                    metadata=db_metadata,
                    db_host=db_host,
                    db_port=db_port,
                    db_name=db_name,
                    db_user=db_user,
                    db_password=db_password
                )
                logging.info(f"YTD trend chart database storage result: {db_result['message']}")
            except Exception as db_error:
                logging.error(f"Database operation failed for YTD trend chart: {db_error}")
        
        # Return HTML or save to file
        if return_html:
            return fig.to_html(include_plotlyjs=True, full_html=True)
        elif output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"ytd_trend_{object_id}_{district or 'citywide'}.html"
            filepath = os.path.join(output_dir, filename)
            fig.write_html(filepath)
            logging.info(f"YTD trend chart saved to {filepath}")
            return filepath
        else:
            return fig.to_html(include_plotlyjs=True, full_html=True)
            
    except Exception as e:
        logging.error(f"Error generating YTD trend chart: {e}")
        return f"**Error**: Failed to generate YTD trend chart: {str(e)}"

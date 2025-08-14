# genaggregate.py

import pandas as pd
import logging
from typing import Union, Tuple

def generate_html_table(df: pd.DataFrame) -> str:
    """
    Generates an HTML table from a pandas DataFrame with basic styling.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to convert to HTML
    
    Returns:
    - str: HTML string containing the styled table
    """
    html_style = """
    <style>
        .dataframe {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .dataframe thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .dataframe th,
        .dataframe td {
            padding: 12px 15px;
            border: 1px solid #dddddd;
        }
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
    </style>
    """
    
    # Convert DataFrame to HTML with float format and classes
    html_table = df.to_html(
        float_format=lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else '',
        classes='dataframe',
        index=False
    )
    
    return html_style + html_table

def aggregate_data(
    df: pd.DataFrame,
    time_series_field: str,
    numeric_fields: list,
    aggregation_period: str = 'day',
    group_field: str = None,
    agg_functions: dict = None,
    return_html: bool = False
) -> Union[Tuple[pd.DataFrame, str], pd.DataFrame]:
    """
    Aggregates the DataFrame based on the specified time period and grouping field.

    Parameters:
    - df (pd.DataFrame): The input dataset.
    - time_series_field (str): The name of the date/time column.
    - numeric_fields (list): List of numeric columns to aggregate.
    - aggregation_period (str): The period to aggregate by ('day', 'week', 'month', 'quarter', 'year').
    - group_field (str, optional): Additional field to group by.
    - agg_functions (dict, optional): Dictionary specifying aggregation functions for each numeric field.
    - return_html (bool): If True, returns both DataFrame and HTML table. Default False.

    Returns:
    - If return_html=True: Tuple of (pd.DataFrame, str) containing aggregated data and HTML table
    - If return_html=False: pd.DataFrame of aggregated data
    """
    logging.debug("Starting aggregation process.")

    # Verify the time_series_field exists in the DataFrame
    if time_series_field not in df.columns:
        raise ValueError(f"Time series field '{time_series_field}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    # Convert time_series_field to datetime
    df = df.copy()  # Create a copy to avoid modifying the original
    df[time_series_field] = pd.to_datetime(df[time_series_field], errors='coerce')
    df = df.dropna(subset=[time_series_field])
    logging.debug("Converted '%s' to datetime and dropped NaT values.", time_series_field)

    # Special handling for weekly aggregation
    if aggregation_period.lower() == 'week':
        logging.debug("Using custom weekly aggregation logic (ISO week)")
        # Create ISO week-based columns for proper weekly aggregation
        df['iso_year'] = df[time_series_field].dt.isocalendar().year
        df['iso_week'] = df[time_series_field].dt.isocalendar().week
        # Create ISO week label (e.g., '2025-W25')
        df['time_period'] = df['iso_year'].astype(str) + '-W' + df['iso_week'].astype(str).str.zfill(2)
        
        # Prepare aggregation dictionary
        if agg_functions:
            agg_dict = agg_functions
        else:
            agg_dict = {
                field: 'mean' if any(field.endswith(suffix) for suffix in ['_avg', '_pct']) else 'sum' 
                for field in numeric_fields
            }
            logging.debug("Using 'mean' for fields ending in '_avg' or '_pct', 'sum' for others.")
        
        # Group by ISO year, ISO week, and additional fields
        # FIXED: Remove redundant grouping columns to prevent duplicates
        group_cols = ['iso_year', 'iso_week', 'time_period']
        if group_field:
            group_cols.append(group_field)
        logging.debug(f"Weekly aggregation grouping by: {group_cols}")
        
        # Perform weekly aggregation
        aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Drop helper columns that aren't needed for charting
        aggregated = aggregated.drop(['iso_year', 'iso_week'], axis=1)
        
        logging.debug("Weekly aggregation (ISO week) completed successfully.")
        return aggregated

    else:
        # Use standard pandas resampling for other periods
        # Set the time_series_field as the index for resampling
        df.set_index(time_series_field, inplace=True)
        logging.debug("Set '%s' as the DataFrame index for resampling.", time_series_field)

        # Define the resampling rule using the newer aliases
        resample_rule = {
            'day': 'D',
            'week': 'W',
            'month': 'ME',  # Changed from 'M' to 'ME'
            'quarter': 'Q',
            'year': 'YS'  # Use Year-Start so annual points land on Jan 1
        }.get(aggregation_period.lower(), 'D')

        logging.debug("Resampling with rule: %s", resample_rule)

        # Prepare aggregation dictionary
        if agg_functions:
            agg_dict = agg_functions
        else:
            agg_dict = {
                field: 'mean' if any(field.endswith(suffix) for suffix in ['_avg', '_pct']) else 'sum' 
                for field in numeric_fields
            }
            logging.debug("Using 'mean' for fields ending in '_avg' or '_pct', 'sum' for others.")

        # Perform resampling and aggregation
        if group_field:
            logging.debug("Grouping by additional field: %s", group_field)
            aggregated = df.groupby(group_field).resample(resample_rule).agg(agg_dict).reset_index()
        else:
            # Automatically group by 'agent' if it exists
            if 'agent' in df.columns:
                group_field = 'agent'
                logging.debug("'agent' field detected. Grouping by 'agent'.")
                aggregated = df.groupby(group_field).resample(resample_rule).agg(agg_dict).reset_index()
            else:
                aggregated = df.resample(resample_rule).agg(agg_dict).reset_index()
                logging.debug("No 'agent' field found. Aggregating without additional grouping.")

        # Rename the time column for clarity
        aggregated = aggregated.rename(columns={time_series_field: 'time_period'})
        logging.debug("Renamed time series field to 'time_period'.")

        logging.debug("Standard aggregation completed successfully.")

    logging.debug("Aggregation completed successfully.")
    
    if return_html:
        html_table = generate_html_table(aggregated)
        return aggregated, html_table
    
    return aggregated

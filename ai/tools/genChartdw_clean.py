import os
import requests
import json
import logging
from dotenv import load_dotenv
from pathlib import Path
import datetime

# Configure logging
logger = logging.getLogger(__name__)
# Remove explicit logging level setting - will use what's in .env
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Load environment variables
# Determine the project root based on the script's location
# Assumes genChartdw.py is in project_root/ai/tools/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent # Moves up two levels: tools -> ai -> project_root
ai_dir = script_dir.parent # This should be the 'ai' directory

possible_env_paths = [
    ai_dir / '.env',          # Check ai/.env first
    project_root / '.env',    # Then check project_root/.env (original logic)
    Path.home() / '.env'       # Finally, check home directory
]

loaded_env = False
for env_path in possible_env_paths:
    if env_path.exists():
        logger.info(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        loaded_env = True
        break

if not loaded_env:
    logger.warning("No .env file found in project root or home directory. Relying on environment variables being set.")

DATAWRAPPER_API_KEY = os.getenv("DATAWRAPPER_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000") # Default if not set
DW_API_BASE_URL = "https://api.datawrapper.de/v3"

if not DATAWRAPPER_API_KEY:
    logger.error("DATAWRAPPER_API_KEY not found in environment variables. Script cannot function.")
    # raise ValueError("DATAWRAPPER_API_KEY not set.") # Or handle more gracefully

def _make_dw_request(method, endpoint, headers=None, data=None, json_payload=None):
    """Helper function to make requests to Datawrapper API."""
    if not DATAWRAPPER_API_KEY:
        logger.error("Datawrapper API key is not configured.")
        return None

    url = f"{DW_API_BASE_URL}{endpoint}"
    
    default_headers = {
        "Authorization": f"Bearer {DATAWRAPPER_API_KEY}"
    }
    if headers:
        default_headers.update(headers)

    try:
        response = requests.request(method, url, headers=default_headers, data=data, json=json_payload)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        
        # For POST/PATCH creating/updating resources, Datawrapper often returns 200 or 201 with JSON
        # For PUT (data upload), it might be 204 No Content or 200 with JSON
        if response.content:
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.info(f"Response from {method} {url} was not JSON, returning raw content.")
                return response.text # or response.content for binary
        return None # For 204 No Content

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
    return None

def create_time_series_chart_from_data(chart_data, metadata):
    """
    Creates a Datawrapper line chart from provided time series data.
    
    Args:
        chart_data: List of data points with 'time_period', 'value', and optional 'group_value' keys
        metadata: Dictionary with chart metadata including title, object_name, field_name, etc.
        
    Returns:
        The public URL of the created and published Datawrapper chart, or None if failed.
    """
    # Use the title from metadata, with proper fallback chain
    # In the database, titles are typically stored as 'object_name'
    chart_title = (
        metadata.get('object_name') or  # First choice: object_name (how it's stored in DB)
        metadata.get('title') or        # Second choice: title
        metadata.get('chart_title') or  # Third choice: chart_title  
        f"Time Series Chart {metadata.get('chart_id', '')}"  # Fallback
    )
    logger.info(f"Starting Datawrapper chart creation from stored data, title: '{chart_title}'")

    if not DATAWRAPPER_API_KEY:
        logger.error("Cannot create chart: DATAWRAPPER_API_KEY is not set.")
        return None

    if not chart_data:
        logger.error("No chart data provided")
        return None

    # Convert date objects to strings for safe processing
    processed_chart_data = []
    for point in chart_data:
        processed_point = point.copy()
        
        # Handle time_period - convert date objects to strings
        time_period = processed_point.get('time_period')
        if isinstance(time_period, (datetime.date, datetime.datetime)):
            processed_point['time_period'] = time_period.strftime('%Y-%m-%d')
        elif time_period is not None:
            processed_point['time_period'] = str(time_period)
        
        # Ensure value is properly formatted
        value = processed_point.get('value')
        if value is not None:
            try:
                processed_point['value'] = float(value)
            except (ValueError, TypeError):
                processed_point['value'] = 0.0
        
        processed_chart_data.append(processed_point)

    # Log the first data point to inspect structure (now safe for JSON serialization)
    if processed_chart_data:
        try:
            logger.info(f"First data point: {json.dumps(processed_chart_data[0])}")
        except TypeError as e:
            logger.warning(f"Could not serialize first data point for logging: {e}")
        logger.info(f"Successfully received {len(processed_chart_data)} data points for chart '{chart_title}'.")

    # Check if we have multiple series (group_value field)
    # Fixed: Check not only if group_value key exists, but also if it has non-null values
    has_groups = any('group_value' in point and point['group_value'] is not None for point in processed_chart_data)
    
    if has_groups:
        # Multiple series - create CSV with separate columns for each group
        logger.info("Detected multiple series data, creating multi-line chart")
        
        # Get all unique groups and time periods
        groups = set()
        time_periods = set()
        
        for point in processed_chart_data:
            if 'group_value' in point and point['group_value'] is not None:
                groups.add(str(point['group_value']))
            if 'time_period' in point and point['time_period'] is not None:
                time_periods.add(point['time_period'])
        
        groups = sorted(list(groups))
        time_periods = sorted(list(time_periods))
        
        logger.info(f"Found {len(groups)} groups: {groups}")
        logger.info(f"Found {len(time_periods)} time periods")
        
        # Create CSV header with time_period and each group as a column
        csv_header = ["time_period"] + groups
        csv_data_lines = [",".join(csv_header)]
        
        # Create a dictionary to organize data by time_period and group
        data_matrix = {}
        for period in time_periods:
            data_matrix[period] = {}
            for group in groups:
                data_matrix[period][group] = None  # Default to None (empty)
        
        # Fill in the data
        for point in processed_chart_data:
            period = point.get('time_period')
            group = str(point.get('group_value', ''))
            value = point.get('value', 0)
            
            if period and group in groups:
                data_matrix[period][group] = value
        
        # Generate CSV rows
        for period in time_periods:
            row = [period]
            for group in groups:
                value = data_matrix[period][group]
                # Use empty string for None values (Datawrapper will treat as no data point)
                row.append(str(value) if value is not None else "")
            csv_data_lines.append(",".join(row))
        
        csv_data = "\n".join(csv_data_lines)
        
        # Configure chart for multiple lines
        lines_config = {}
        colors = ["#ad35fa", "#00bcd4", "#ff9800", "#4caf50", "#f44336", "#9c27b0", "#607d8b", "#1A365D", "#2D3748", "#4A5568", "#718096", "#A0AEC0", "#E2E8F0", "#F7FAFC", "#FF0080", "#00FF80", "#8000FF", "#FF8000", "#0080FF", "#80FF00", "#00FFFF", "#FF00FF", "#FFFF00", "#8B4513", "#228B22", "#FF1493", "#00CED1", "#FFD700", "#FF69B4", "#32CD32", "#FF4500", "#9370DB", "#20B2AA", "#FF6347", "#7B68EE", "#3CB371", "#FF7F50", "#6A5ACD", "#00FA9A", "#FFB6C1", "#4169E1", "#DC143C", "#00BFFF", "#FF8C00", "#9932CC", "#8FBC8F", "#FF69B4", "#00CED1", "#FFD700", "#FF1493", "#32CD32", "#FF4500", "#9370DB", "#20B2AA", "#FF6347", "#7B68EE", "#3CB371", "#FF7F50", "#6A5ACD", "#00FA9A", "#FFB6C1", "#4169E1", "#DC143C", "#00BFFF", "#FF8C00", "#9932CC"]  # Expanded color palette
        
        for i, group in enumerate(groups):
            color = colors[i % len(colors)]  # Cycle through colors if more groups than colors
            lines_config[group] = {
                "symbols": {
                    "on": "every",
                    "enabled": True
                },
                "colorKey": True,
                "directLabel": False,
                "valueLabels": {
                    "enabled": True
                },
                "name": group
            }
        
        custom_colors = {group: colors[i % len(colors)] for i, group in enumerate(groups)}
        values_config = groups  # Multiple value columns
        
    else:
        # Single series - use original format
        logger.info("Detected single series data, creating single-line chart")
        
        csv_data_lines = ["time_period,numeric_value"]  # CSV header
        for point in processed_chart_data:
            date_val = point.get('time_period', '')
            numeric_val = point.get('value', 0)
            csv_data_lines.append(f"{date_val},{numeric_val}")
        csv_data = "\n".join(csv_data_lines)
        
        # Configure chart for single line
        y_axis_label = metadata.get('field_name', metadata.get('object_name', 'Value'))
        lines_config = {
            "numeric_value": {
                "symbols": {
                    "on": "every", 
                    "enabled": True
                },
                "colorKey": True,
                "directLabel": False,
                "valueLabels": {
                    "enabled": True
                },
                "name": y_axis_label
            }
        }
        custom_colors = {"numeric_value": "#ad35fa"}
        values_config = "numeric_value"
    
    logger.debug(f"Prepared CSV data:\n{csv_data[:500]}...")  # Log a snippet

    # Create a new chart in Datawrapper
    logger.info("Creating new chart in Datawrapper...")
    
    # Use area chart for multi-line data, line chart for single series
    chart_type = "multiple-lines" if has_groups else "d3-lines" # Use multiple-lines for multi-series to get separate panels
    logger.info(f"Using chart type: {chart_type} (has_groups: {has_groups})")
    
    create_payload = {
        "title": chart_title,
        "type": chart_type,
    }
    created_chart_info = _make_dw_request("POST", "/charts", json_payload=create_payload)

    if not created_chart_info or "id" not in created_chart_info:
        logger.error("Failed to create Datawrapper chart.")
        return None
    
    chart_id = created_chart_info["id"]
    logger.info(f"Datawrapper chart created with ID: {chart_id}")

    # Upload data to the chart
    logger.info(f"Uploading data to chart ID: {chart_id}...")
    upload_headers = {"Content-Type": "text/csv"}
    upload_response = _make_dw_request("PUT", f"/charts/{chart_id}/data", headers=upload_headers, data=csv_data.encode('utf-8'))
    logger.info(f"Data upload process completed for chart ID: {chart_id}. Response: {type(upload_response)}")

    # Customize chart (metadata)
    logger.info(f"Customizing chart ID: {chart_id}...")
    
    # Get y-axis label from metadata
    y_axis_label = metadata.get('field_name', metadata.get('object_name', 'Value'))
    
    # Create description with district information if applicable
    description = ""
    district = metadata.get('district')
    if district and str(district) != "0":
        description = f"Data for District {district}. "
    
    # Add any existing caption/description
    existing_description = metadata.get('caption') or metadata.get('description', '')
    if existing_description:
        description += existing_description
    
    # Get executed_query_url from metadata if available
    executed_query_url = metadata.get("executed_query_url", "")
    
    # Determine appropriate date format based on period_type from metadata
    period_type = metadata.get('period_type', 'month')
    if period_type == "year":
        date_format = "YYYY"  # Annual format: just the year
    elif period_type == "quarter":
        date_format = "Q YYYY"  # Quarterly format: Q1 2023
    else:  # Default to monthly
        date_format = "MMM YYYY"  # Monthly format: Apr 2023
    
    customization_payload = {
        "metadata": {
            "describe": {
                "intro": description,
                "source-name": "DataSF",
                "source-url": executed_query_url,
                "byline": "Generated by TransparentSF"
            },
            "visualize": {
                "interpolation": "linear",
                "custom-colors": custom_colors,
                "y-grid": True,
                "x-grid": False,
                "lines": lines_config,
                "legend": {
                    "enabled": True,
                    "position": "top-right"
                },
                "custom-range-y": ["0", ""]  # Force y-axis to start at 0
            },
            "axes": {
                "keys": "time_period",  # X-axis data column
                "values": values_config,  # Y-axis data column(s)
                "y-grid": True,
                "y-label": y_axis_label,
                "dateFormat": date_format,  # Use conditional date format based on period_type
                "y-min": 0
            },
            "publish": {
                "embed-width": 700,
                "embed-height": 450,
                "autoDarkMode": True,
                "chart-height": 0
            }
        }
    }
    
    # Add grid layout configuration for multiple-lines charts
    if has_groups:
        # Ensure visualize key exists
        if "visualize" not in customization_payload:
            customization_payload["visualize"] = {}
            
        # Add grid layout settings
        customization_payload["visualize"].update({
            "gridLayout": "fixedCount",
            "gridColumnCount": 2,  # Show 2 charts per row
            "gridRowHeightFixed": 140,  # Fixed height for each panel
            "plotHeightFixed": 300,  # Fixed height for the plot area
            "independentYScales": True,  # Each panel gets its own Y scale
            "yGridLabelAllColumns": True,  # Show Y axis labels for all panels
            "xGridLabelAllColumns": True,  # Show X axis labels for all panels
            "gridColumnMinWidth": 200,  # Minimum width for each panel
            "gridColumnCountMobile": 1  # Single column on mobile
        })
    
    update_response = _make_dw_request("PATCH", f"/charts/{chart_id}", json_payload=customization_payload)
    if update_response:
        logger.info(f"Chart metadata updated successfully for chart ID: {chart_id}")
    else:
        logger.warning(f"Failed to update chart metadata for chart ID: {chart_id}")

    # Publish the chart
    logger.info(f"Publishing chart ID: {chart_id}...")
    publish_response = _make_dw_request("POST", f"/charts/{chart_id}/publish")

    # Handle publish response similar to the main function
    if not publish_response:
        # If no response content, try to get chart details directly
        logger.warning("No content returned from publish request, checking chart details...")
        chart_details = _make_dw_request("GET", f"/charts/{chart_id}")
        if chart_details:
            public_url = chart_details.get("publicUrl")
            if public_url:
                logger.warning(f"Publish command might have failed or returned no content, but found publicUrl: {public_url}")
                return public_url
        return None
    else:
        # Extract the public URL from the publish response
        public_url = publish_response.get("publicUrl")
        if public_url:
            logger.info(f"Successfully created and published Datawrapper chart: {public_url}")
            return public_url
        else:
            logger.error(f"Publish response did not contain publicUrl: {publish_response}")
            return None

    # If we get here, something went wrong
    logger.error(f"Failed to retrieve public URL for chart ID: {chart_id} after publishing.")
    return None

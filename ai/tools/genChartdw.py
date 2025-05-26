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

def create_datawrapper_chart(metric_id: str, intro: str = "", district: str = "0", period_type: str = "month", direct_data=None):
    """
    Creates a Datawrapper line chart for a given metric.

    Args:
        metric_id: The ID of the metric.
        intro: An introductory text/description for the chart.
        district: The district for which to fetch data.
        period_type: The period type for data aggregation (e.g., 'month', 'year').
        direct_data: Optional data dictionary to use instead of fetching from API. Used to avoid deadlocks.

    Returns:
        The public URL of the created and published Datawrapper chart, or None if failed.
    """
    # Create a default title based on the metric ID
    default_title = f"Metric {metric_id} - {district if district != '0' else 'Citywide'}"
    logger.info(f"Starting Datawrapper chart creation for metric_id: {metric_id}, title: '{default_title}'")

    if not DATAWRAPPER_API_KEY:
        logger.error("Cannot create chart: DATAWRAPPER_API_KEY is not set.")
        return None

    # 1. Get data either from the direct_data parameter or fetch from the backend
    if direct_data:
        logger.info(f"Using provided direct data for chart creation instead of making HTTP request")
        metric_data_response = direct_data
    else:
        # Fetch data from the backend via HTTP
        data_fetch_url = f"{API_BASE_URL}/backend/api/chart-by-metric?metric_id={metric_id}&district={district}&period_type={period_type}"
        logger.info(f"Fetching chart data from: {data_fetch_url}")
        try:
            response = requests.get(data_fetch_url)
            response.raise_for_status()
            metric_data_response = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for metric {metric_id}: {e}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from {data_fetch_url}")
            return None

    # Extract metadata and data from response
    chart_data_points = metric_data_response.get("data")
    metadata = metric_data_response.get("metadata", {})
    
    # Get chart title and y-axis label from metadata
    chart_title = metadata.get("chart_title", default_title)
    y_axis_label = metadata.get("y_axis_label", default_title)
    actual_description = metadata.get("caption", intro)
    
    if not chart_data_points:
        logger.error(f"No data points found in response data")
        return None
    
    # Log the first data point to inspect its structure
    if chart_data_points and len(chart_data_points) > 0:
        logger.info(f"First data point received: {json.dumps(chart_data_points[0])}")
    else:
        logger.info("chart_data_points is empty or None after fetching.")
        
    logger.info(f"Successfully received {len(chart_data_points)} data points for chart '{chart_title}'.")

    # 2. Convert data to CSV format for Datawrapper
    csv_data_lines = ["time_period,numeric_value"] # CSV header
    for point in chart_data_points:
        # Use the correct keys identified from the backend response
        date_val = point.get('time_period')
        numeric_val = point.get('numeric_value')
        csv_data_lines.append(f"{date_val},{numeric_val}")
    csv_data = "\n".join(csv_data_lines)
    logger.debug(f"Prepared CSV data with correct keys:\n{csv_data[:250]}...") # Log a snippet

    # 3. Create a new chart in Datawrapper
    logger.info("Creating new chart in Datawrapper...")
    create_payload = {
        "title": chart_title,  # Use the chart_title from the metadata
        "type": "d3-lines", # Line chart
        # "folderId": "YOUR_FOLDER_ID" # Optional: to organize charts
    }
    created_chart_info = _make_dw_request("POST", "/charts", json_payload=create_payload)

    if not created_chart_info or "id" not in created_chart_info:
        logger.error("Failed to create Datawrapper chart.")
        return None
    
    chart_id = created_chart_info["id"]
    logger.info(f"Datawrapper chart created with ID: {chart_id}")

    # 4. Upload data to the chart
    logger.info(f"Uploading data to chart ID: {chart_id}...")
    upload_headers = {"Content-Type": "text/csv"}
    # Datawrapper API for data upload expects raw CSV in the body, not as form data or JSON.
    upload_response = _make_dw_request("PUT", f"/charts/{chart_id}/data", headers=upload_headers, data=csv_data.encode('utf-8'))

    # A successful PUT to /data usually returns 204 No Content or 200 with chart metadata.
    # _make_dw_request handles this. If it returns None for 204, that's fine.
    # If it returns chart metadata (on 200), that's also fine.
    # We mainly care that it didn't raise an error.
    # Let's check if an error occurred by inspecting if upload_response is None AND an error was logged by _make_dw_request.
    # For now, we assume success if no exception was raised by _make_dw_request.
    logger.info(f"Data upload process completed for chart ID: {chart_id}. Response: {type(upload_response)}")


    # 5. Customize chart (metadata) - Crucial for making it look like time_series_chart.html
    logger.info(f"Customizing chart ID: {chart_id}...")
    
    # Mimic time_series_chart.html styling
    # - X-axis: date, Y-axis: value
    # - Date format for X-axis labels
    # - Grid lines
    # - Line color, thickness, interpolation (smooth)
    # - Tooltips/hover effects
    # - Title, description (intro)
    
    # Create description with district information if applicable
    description = ""
    if district and district != "0":
        description = f"Data for District {district}. "
    if actual_description:
        description += actual_description
    
    customization_payload = {
        "metadata": {
            "describe": {
                "intro": description,
                "source-name": "DataSF",
                "byline": "Generated by TransparentSF"
            },
            "visualize": {
                "interpolation": "linear", # Linear line type as requested
                "custom-colors": {
                    "numeric_value": "#ad35fa" # Purple color as requested
                },
                "y-grid": True,
                "x-grid": False,
                "lines": {
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
                        "name": y_axis_label # Use y_axis_label for the legend
                    }
                },
                "legend": {
                    "enabled": True,
                    "position": "top-right"
                },
                "custom-range-y": ["0", ""]  # This is the key setting that forces y-axis to start at 0
            },
            "axes": {
                "keys": "time_period", # X-axis data column
                "values": "numeric_value", # Y-axis data column
                "y-grid": True,
                "y-label": y_axis_label, # Use the y_axis_label from metadata
                "dateFormat": "MMM YYYY", # Format dates as "Apr 2023"
                "y-min": 0 # Keep this as backup
            },
            "publish": { # Ensure chart is embeddable
                "embed-width": 600, # Default or adjust
                "embed-height": 400 # Default or adjust
            }
        }
    }
    
    # Add specific line customization if 'value' is the recognized column
    # Datawrapper might automatically pick up the 'value' column if it's the only one besides 'date'.
    # The `metadata.visualize.lines` is more explicit.

    update_response = _make_dw_request("PATCH", f"/charts/{chart_id}", json_payload=customization_payload)
    if update_response is None and not logger.handlers[0].stream.getvalue().endswith("HTTP error occurred"): # A bit hacky check
        logger.info(f"Chart metadata updated successfully for chart ID: {chart_id}.")
    elif update_response:
         logger.info(f"Chart metadata updated successfully for chart ID: {chart_id}. Response: {type(update_response)}")
    else:
        logger.warning(f"Failed to update chart metadata or update returned no content for chart ID: {chart_id}.")
        # Continue to publish, might still work with defaults.

    # 6. Publish the chart
    logger.info(f"Publishing chart ID: {chart_id}...")
    publish_response = _make_dw_request("POST", f"/charts/{chart_id}/publish")

    if not publish_response: # This means error or 204 from POST
        logger.error(f"Failed to publish chart ID: {chart_id} or no response content. Check logs.")
        # Try to get chart info anyway to see if publicUrl exists
        chart_details = _make_dw_request("GET", f"/charts/{chart_id}")
        if chart_details and chart_details.get("publicUrl"):
            public_url = chart_details["publicUrl"]
            logger.warning(f"Publish command might have failed or returned no content, but found publicUrl: {public_url}")
            return public_url
        return None

    # The publish response usually contains the updated chart metadata, including publicUrl
    public_url = None
    if isinstance(publish_response, dict):
        public_url = publish_response.get("publicUrl")
        # Datawrapper often nests the URL like: publish_response['data'][0]['publicUrl'] or similar
        # Let's be more robust
        if not public_url:
            if "data" in publish_response and isinstance(publish_response["data"], list) and len(publish_response["data"]) > 0:
                public_url = publish_response["data"][0].get("publicUrl")
        
    if not public_url: # Fallback if structure is different or publicUrl not directly in response
        logger.info("Public URL not found directly in publish response, attempting to retrieve chart details...")
        chart_details = _make_dw_request("GET", f"/charts/{chart_id}")
        if chart_details:
            public_url = chart_details.get("publicUrl")

    if public_url:
        logger.info(f"Chart published successfully! Public URL: {public_url}")
        return public_url
    else:
        logger.error(f"Failed to retrieve public URL for chart ID: {chart_id} after publishing.")
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
        colors = ["#ad35fa", "#00bcd4", "#ff9800", "#4caf50", "#f44336", "#9c27b0", "#607d8b"]  # Multiple colors
        
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
    chart_type = "d3-area" if has_groups else "d3-lines"
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
    
    customization_payload = {
        "metadata": {
            "describe": {
                "intro": description,
                "source-name": "DataSF", 
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
                "dateFormat": "MMM YYYY",  # Format dates as "Apr 2023"
                "y-min": 0
            },
            "publish": {
                "embed-width": 700,
                "embed-height": 450
            }
        }
    }
    
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

    # Extract public URL from publish response
    public_url = None
    if isinstance(publish_response, dict):
        public_url = publish_response.get("publicUrl")
        # Try nested structure if not found directly
        if not public_url:
            if "data" in publish_response and isinstance(publish_response["data"], list) and len(publish_response["data"]) > 0:
                public_url = publish_response["data"][0].get("publicUrl")
        
    if not public_url:  # Fallback
        logger.info("Public URL not found in publish response, retrieving chart details...")
        chart_details = _make_dw_request("GET", f"/charts/{chart_id}")
        if chart_details:
            public_url = chart_details.get("publicUrl")

    if public_url:
        logger.info(f"Chart published successfully! Public URL: {public_url}")
        return public_url
    else:
        logger.error(f"Failed to retrieve public URL for chart ID: {chart_id} after publishing.")
        return None

if __name__ == '__main__':
    # Example Usage:
    # Ensure your .env file has DATAWRAPPER_API_KEY and API_BASE_URL (if not localhost:8000)
    # And your backend at API_BASE_URL has the /backend/api/chart-by-metric endpoint working.
    
    # Use logging configuration from environment instead of setting it explicitly here
    
    # Sample data for testing - directly providing data instead of fetching from API
    sample_data = {
        "metadata": {
            "chart_id": 5000,
            "chart_title": "💊 Drug Crime Incidents - District 11 <br> Monthly - District 11 Trend",
            "y_axis_label": "Drug Crime Incidents - District 11",
            "period_type": "monthly",
            "object_type": "dashboard_metric",
            "object_id": "4",
            "object_name": "💊 Drug Crime Incidents",
            "field_name": "unknown",
            "district": 11,
            "executed_query_url": "https://data.sfgov.org/resource/wg3w-h783.json?%24query=SELECT+date_trunc_ym%28Report_Datetime%29+as+month_period%2C+COUNT%28%2A%29+as+value+%2C+CASE+WHEN+Report_Datetime+%3E%3D+%272025-04-01%27+AND+Report_Datetime+%3C%3D+%272025-04-30%27+THEN+%27recent%27+ELSE+%27comparison%27+END+as+period_type+%2C+report_type_description%2C+incident_category%2C+incident_subcategory%2C+incident_description%2C+supervisor_district+WHERE+Report_Datetime+%3E%3D+%272023-04-01%27+AND+Report_Datetime+%3C%3D+%272025-04-30%27+AND+Incident_Category+IN+%28%27Drug+Offense%27%2C+%27Drug+Violation%27%29+AND+%28+%28Report_Datetime+%3E%3D+%272023-04-01%27+AND+Report_Datetime+%3C%3D+%272025-03-31%27%29+OR+%28Report_Datetime+%3E%3D+%272025-04-01%27+AND+Report_Datetime+%3C%3D+%272025-04-30%27%29+%29+GROUP+BY+month_period%2C+period_type%2C+report_type_description%2C+incident_category%2C+incident_subcategory%2C+incident_description%2C+supervisor_district+ORDER+BY+month_period+LIMIT+5000+OFFSET+0",
            "caption": "In April 2025, drug crime incidents - district 11 was 5, which is 60% above the 24 month average of 3.\n\nAs of the end of April 2025, YTD 2025, total drug crime incidents - district 11 is 19, which is 90% above the YTD 2024 total of 10.",
            "filter_conditions": [
                {"field": "month_period", "value": "2025-04-30", "is_date": True, "operator": "<="},
                {"field": "month_period", "value": "2023-04-01", "is_date": True, "operator": ">="},
                {"field": "supervisor_district", "value": "11", "is_date": False, "operator": "="}
            ]
        },
        "data": [
            {"time_period": "2023-04-30", "numeric_value": 2.0},
            {"time_period": "2023-05-31", "numeric_value": 8.0},
            {"time_period": "2023-06-30", "numeric_value": 0.0},
            {"time_period": "2023-07-31", "numeric_value": 3.0},
            {"time_period": "2023-08-31", "numeric_value": 0.0},
            {"time_period": "2023-09-30", "numeric_value": 2.0},
            {"time_period": "2023-10-31", "numeric_value": 3.0},
            {"time_period": "2023-11-30", "numeric_value": 1.0},
            {"time_period": "2023-12-31", "numeric_value": 2.0},
            {"time_period": "2024-01-31", "numeric_value": 3.0},
            {"time_period": "2024-02-29", "numeric_value": 0.0},
            {"time_period": "2024-03-31", "numeric_value": 2.0},
            {"time_period": "2024-04-30", "numeric_value": 5.0},
            {"time_period": "2024-05-31", "numeric_value": 2.0},
            {"time_period": "2024-06-30", "numeric_value": 0.0},
            {"time_period": "2024-07-31", "numeric_value": 12.0},
            {"time_period": "2024-08-31", "numeric_value": 4.0},
            {"time_period": "2024-09-30", "numeric_value": 3.0},
            {"time_period": "2024-10-31", "numeric_value": 0.0},
            {"time_period": "2024-11-30", "numeric_value": 6.0},
            {"time_period": "2024-12-31", "numeric_value": 3.0},
            {"time_period": "2025-01-31", "numeric_value": 7.0},
            {"time_period": "2025-02-28", "numeric_value": 4.0},
            {"time_period": "2025-03-31", "numeric_value": 3.0},
            {"time_period": "2025-04-30", "numeric_value": 5.0}
        ]
    }
    
    # Extract chart title and data from sample data
    chart_title = sample_data["metadata"]["chart_title"]
    chart_data = sample_data["data"]
    chart_caption = sample_data["metadata"]["caption"]
    y_axis_label = sample_data["metadata"]["y_axis_label"]
    
    # Create CSV data directly from the sample data
    csv_data_lines = ["time_period,numeric_value"]
    for point in chart_data:
        date_val = point.get('time_period')
        numeric_val = point.get('numeric_value')
        csv_data_lines.append(f"{date_val},{numeric_val}")
    csv_data = "\n".join(csv_data_lines)
    
    # Create a new chart in Datawrapper
    logger.info("Creating new chart in Datawrapper...")
    create_payload = {
        "title": chart_title,
        "type": "d3-lines", # Line chart
    }
    created_chart_info = _make_dw_request("POST", "/charts", json_payload=create_payload)

    if not created_chart_info or "id" not in created_chart_info:
        logger.error("Failed to create Datawrapper chart.")
    else:
        chart_id = created_chart_info["id"]
        logger.info(f"Datawrapper chart created with ID: {chart_id}")

        # Upload data to the chart
        logger.info(f"Uploading data to chart ID: {chart_id}...")
        upload_headers = {"Content-Type": "text/csv"}
        upload_response = _make_dw_request("PUT", f"/charts/{chart_id}/data", headers=upload_headers, data=csv_data.encode('utf-8'))
        logger.info(f"Data upload process completed for chart ID: {chart_id}. Response: {type(upload_response)}")

        # Customize chart with specific formatting requirements
        logger.info(f"Customizing chart ID: {chart_id}...")
        
        # Get first and last data points for labeling
        first_point_index = 0
        last_point_index = len(chart_data) - 1
        
        # Create description with district information if applicable
        description = ""
        district = sample_data["metadata"].get("district")
        if district and str(district) != "0":
            description = f"Data for District {district}. "
        if chart_caption:
            description += chart_caption
        
        customization_payload = {
            "metadata": {
                "describe": {
                    "intro": description,
                    "source-name": "DataSF",
                    "byline": "Generated by TransparentSF"
                },
                "visualize": {
                    "interpolation": "linear", # Linear line type as requested
                    "custom-colors": {
                        "numeric_value": "#ad35fa" # Purple color as requested
                    },
                    "y-grid": True,
                    "x-grid": False,
                    "lines": {
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
                            "name": y_axis_label # Use y_axis_label for the legend
                        }
                    },
                    "legend": {
                        "enabled": True,
                        "position": "top-right"
                    },
                    "custom-range-y": ["0", ""]  # This is the key setting that forces y-axis to start at 0
                },
                "axes": {
                    "keys": "time_period", # X-axis data column
                    "values": "numeric_value", # Y-axis data column
                    "y-grid": True,
                    "y-label": y_axis_label, # Use the y_axis_label from metadata
                    "dateFormat": "MMM YYYY", # Format dates as "Apr 2023"
                    "y-min": 0 # Keep this as backup
                },
                "publish": {
                    "embed-width": 700,
                    "embed-height": 450
                }
            }
        }
        
        update_response = _make_dw_request("PATCH", f"/charts/{chart_id}", json_payload=customization_payload)
        if update_response:
            logger.info(f"Chart metadata updated successfully for chart ID: {chart_id}. Response: {type(update_response)}")
        else:
            logger.warning(f"Failed to update chart metadata or update returned no content for chart ID: {chart_id}.")

        # Publish the chart
        logger.info(f"Publishing chart ID: {chart_id}...")
        publish_response = _make_dw_request("POST", f"/charts/{chart_id}/publish")

        # Get the public URL
        public_url = None
        if isinstance(publish_response, dict):
            public_url = publish_response.get("publicUrl")
            
        if not public_url:
            logger.info("Public URL not found directly in publish response, attempting to retrieve chart details...")
            chart_details = _make_dw_request("GET", f"/charts/{chart_id}")
            if chart_details:
                public_url = chart_details.get("publicUrl")

        if public_url:
            logger.info(f"Chart published successfully! Public URL: {public_url}")
            print(f"Successfully created chart. URL: {public_url}")
        else:
            logger.error(f"Failed to retrieve public URL for chart ID: {chart_id} after publishing.")
            print(f"Failed to create chart.") 
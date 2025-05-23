import os
import logging
import requests
import json
import datetime
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Load environment variables
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
ai_dir = script_dir.parent

possible_env_paths = [
    ai_dir / '.env',
    project_root / '.env',
    Path.home() / '.env'
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
DW_API_BASE_URL = "https://api.datawrapper.de/v3"

if not DATAWRAPPER_API_KEY:
    logger.error("DATAWRAPPER_API_KEY not found in environment variables. Script cannot function.")

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
        response.raise_for_status()
        
        if response.content:
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.info(f"Response from {method} {url} was not JSON, returning raw content.")
                return response.text
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
    return None

def generate_anomaly_chart_dw(item, chart_title, metadata, output_dir='static'):
    """
    Generates a Datawrapper chart for an anomaly and returns the chart URL.
    
    Parameters:
    - item (dict): Anomaly data containing dates, counts, comparison_mean, etc.
    - chart_title (str): Title for the chart.
    - metadata (dict): Contains comparison_period, recent_period, and other metadata.
    - output_dir (str): Directory to save chart details (not used for Datawrapper but kept for API compatibility).
    
    Returns:
    - str: URL of the published Datawrapper chart or None if failed.
    """
    logger.info(f"Starting Datawrapper anomaly chart creation: {chart_title}")
    
    if not DATAWRAPPER_API_KEY:
        logger.warning("DATAWRAPPER_API_KEY is not set. Returning a mock URL for testing purposes.")
        # Mock URL for testing when API key is not available
        return f"https://datawrapper.mock/chart/{uuid.uuid4()}"
    
    # Extract data from the item
    dates = item['dates']
    counts = item['counts']
    combined_data = []
    
    # Get period type from metadata, default to month if not specified
    period_type = metadata.get('period_type', 'month')
    
    # Process dates based on period type
    for idx, date_entry in enumerate(dates):
        if isinstance(date_entry, str):
            try:
                if period_type == 'year':
                    # For year period, expect YYYY format
                    year = int(date_entry)
                    date_obj = datetime.date(year, 1, 1)
                else:
                    # For monthly data, expect YYYY-MM format
                    if len(date_entry.split('-')) == 2:
                        year, month = map(int, date_entry.split('-'))
                        date_obj = datetime.date(year, month, 1)
                    else:
                        # If we somehow get just a year in monthly mode, use January
                        year = int(date_entry)
                        date_obj = datetime.date(year, 1, 1)
                        logger.warning(f"Found year-only format in monthly mode for date: {date_entry}")
            except ValueError as ve:
                logger.warning(f"Record {idx}: Invalid date format '{date_entry}' for {period_type} period. Skipping this date.")
                continue
        elif isinstance(date_entry, datetime.date):
            date_obj = date_entry
        else:
            logger.warning(f"Record {idx}: Unexpected date type '{type(date_entry)}'. Skipping this date.")
            continue
        
        # Store processed date and corresponding count
        combined_data.append((date_obj, counts[idx]))
    
    if not combined_data:
        logger.error("No valid dates available after processing.")
        return None
    
    # Define date ranges
    try:
        comparison_start = metadata['comparison_period']['start']
        comparison_end = metadata['comparison_period']['end']
        recent_start = metadata['recent_period']['start']
        recent_end = metadata['recent_period']['end']
    except (KeyError, TypeError) as e:
        logger.error(f"Error accessing metadata dates: {e}")
        return None
    
    # Filter data within the date range
    filtered_data = [(date, count) for date, count in combined_data if comparison_start <= date <= recent_end]
    
    # Split data into comparison and recent periods
    comparison_data = [(date, count) for date, count in filtered_data if comparison_start <= date <= comparison_end]
    recent_data = [(date, count) for date, count in filtered_data if recent_start <= date <= recent_end]
    
    # Prepare data for CSV format
    csv_data_lines = ["date,value,period,mean,upper_bound,lower_bound"]
    
    # Calculate statistics for reference lines
    comparison_mean = float(round(item['comparison_mean'], 1))
    std_dev = float(round(item['stdDev'], 1))
    upper_bound = comparison_mean + 2 * std_dev
    lower_bound = max(comparison_mean - 2 * std_dev, 0)
    
    # Add comparison period data
    for date, count in comparison_data:
        formatted_date = date.strftime('%Y-%m-%d')
        csv_data_lines.append(f"{formatted_date},{count},historical,{comparison_mean},{upper_bound},{lower_bound}")
    
    # Add recent period data
    for date, count in recent_data:
        formatted_date = date.strftime('%Y-%m-%d')
        csv_data_lines.append(f"{formatted_date},{count},recent,{comparison_mean},{upper_bound},{lower_bound}")
    
    csv_data = "\n".join(csv_data_lines)
    
    # Generate a caption
    y_axis_label = metadata.get('y_axis_label', 'Value')
    logger.info(f"Using y_axis_label in chart configuration: {y_axis_label}")
    percent_difference = abs((item['difference'] / item['comparison_mean']) * 100) if item['comparison_mean'] else 0
    action = 'increase' if item['difference'] > 0 else 'drop' if item['difference'] < 0 else 'no change'
    
    comparison_period_label = f"{comparison_start.strftime('%B %Y')} to {comparison_end.strftime('%B %Y')}"
    recent_period_label = f"{recent_start.strftime('%B %Y')}"
    
    # Use description from metadata if available, otherwise generate one
    if 'chart_description' in metadata:
        description = metadata['chart_description']
    else:
        caption = (
            f"In {recent_period_label}, there were {item['recent_mean']:,.0f} {item['group_value']} {y_axis_label.lower()} per month, "
            f"compared to an average of {comparison_mean:,.0f} per month over {comparison_period_label}, "
            f"a {percent_difference:.1f}% {action}."
        )
        
        # Create a description that explains the anomaly
        description = (
            f"{y_axis_label} - {action.capitalize()} of {percent_difference:.1f}% in {recent_period_label} "
            f"compared to the average over {comparison_period_label}."
        )
    
    # 1. Create a new chart in Datawrapper
    logger.info("Creating new anomaly chart in Datawrapper...")
    
    # Format the title to make sure the value after the colon is bold
    formatted_title = chart_title
    
    # Skip formatting if the title already contains HTML tags
    if ":" in chart_title and "<" not in chart_title:
        title_parts = chart_title.split(":", 1)
        formatted_title = f"{title_parts[0]}: <strong>{title_parts[1].strip()}</strong>"
        logger.info(f"Formatted title with bold: {formatted_title}")
    else:
        logger.info(f"Using title as-is (already contains formatting or no colon): {formatted_title}")
    
    create_payload = {
        "title": formatted_title,
        "type": "d3-lines",
    }
    
    created_chart_info = _make_dw_request("POST", "/charts", json_payload=create_payload)
    
    if not created_chart_info or "id" not in created_chart_info:
        logger.error("Failed to create Datawrapper chart.")
        return None
    
    chart_id = created_chart_info["id"]
    logger.info(f"Datawrapper chart created with ID: {chart_id}")
    
    # 2. Upload data to the chart
    logger.info(f"Uploading data to chart ID: {chart_id}...")
    upload_headers = {"Content-Type": "text/csv"}
    upload_response = _make_dw_request("PUT", f"/charts/{chart_id}/data", headers=upload_headers, data=csv_data.encode('utf-8'))
    
    logger.info(f"Data upload process completed for chart ID: {chart_id}. Response: {type(upload_response)}")
    
    # 3. Customize chart (metadata)
    logger.info(f"Customizing chart ID: {chart_id}...")
    
    # Determine chart format based on period_type
    date_format = "%Y" if period_type == 'year' else "%b %Y"
    
    customization_payload = {
        "metadata": {
            "describe": {
                "intro": description,
                "source-name": "DataSF",
                "byline": "Generated by TransparentSF",
                "aria-description": description,
                "notes": "Data from DataSF"
            },
            "visualize": {
                "dark-mode-invert": True,
                "x-grid": False,
                "y-grid": True,
                "interpolation": "natural",
                "connector-lines": True,
                "show-tooltips": True,
                "base-color": 2,
                "highlighted-series": [],
                "chart-type-set": True,
                "plotHeightMode": "fixed",
                "plotHeightFixed": 300,
                "plotHeightRatio": 0.5,
                "sharing": {
                    "enabled": True,
                    "auto": True
                },
                "color-category": {
                    "map": {
                        "mean": "#666666",
                        "value": "#ad35fa",
                        "lower_bound": "#cccccc",
                        "upper_bound": "#cccccc"
                    }
                },
                "custom-area-fills": [
                    {
                        "id": str(uuid.uuid4()).replace("-", "_"),
                        "from": "upper_bound",
                        "to": "lower_bound",
                        "color": "#e0e0e0",
                        "opacity": 0.2,
                        "interpolation": "linear"
                    }
                ],
                "lines": {
                    "value": {
                        "name": y_axis_label,
                        "symbols": {
                            "on": "every",
                            "enabled": True
                        },
                        "valueLabels": {
                            "enabled": True
                        },
                        "colorPalette": {
                            "recent": "#ad35fa",
                            "historical": "#666666"
                        },
                        "interpolation": "linear"
                    },
                    "mean": {
                        "name": "Average",
                        "stroke": "dash",
                        "strokeWidth": 2
                    },
                    "upper_bound": {
                        "name": "Upper Bound (2σ)",
                        "stroke": "dash",
                        "strokeWidth": 1
                    },
                    "lower_bound": {
                        "name": "Lower Bound (2σ)",
                        "stroke": "dash",
                        "strokeWidth": 1
                    }
                },
                "legend": {
                    "width": "auto",
                    "height": "auto",
                    "enabled": True,
                    "position": "top"
                }
            },
            "axes": {
                "keys": "date",
                "values": ["value", "mean", "upper_bound", "lower_bound"],
                "y-grid": True,
                "y-label": y_axis_label,
                "dateFormat": date_format,
                "y-min": 0
            },
            "publish": {
                "embed-width": 700,
                "embed-height": 600,
                "blocks": {
                    "logo": {
                        "enabled": False
                    },
                    "get-the-data": True,
                    "social-sharing": {
                        "enabled": True
                    },
                    "download-image": True,
                    "download-pdf": False,
                    "download-svg": False,
                    "embed": True
                },
                "social-sharing": {
                    "enabled": True,
                    "networks": ["twitter", "facebook", "linkedin", "email", "whatsapp"]
                },
                "responsive": {
                    "enabled": True,
                    "fallback-height": 600
                },
                "force-attribution": False,
                "autoDarkMode": False,
                "chart-height": 350
            }
        }
    }
    
    update_response = _make_dw_request("PATCH", f"/charts/{chart_id}", json_payload=customization_payload)
    if update_response:
        logger.info(f"Chart metadata updated successfully for chart ID: {chart_id}. Response: {type(update_response)}")
    else:
        logger.warning(f"Failed to update chart metadata or update returned no content for chart ID: {chart_id}.")
    
    # 4. Publish the chart
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
        return public_url
    else:
        logger.error(f"Failed to retrieve public URL for chart ID: {chart_id} after publishing.")
        return None

def generate_anomalies_summary_with_datawrapper(results, metadata, output_dir='static'):
    """
    Generates Datawrapper charts for each detected anomaly and returns a summary that 
    includes links to the charts.
    
    Parameters:
    - results (list of dict): List containing anomaly details.
    - metadata (dict): Metadata containing period information.
    - output_dir (str): Directory to save any local files (not used for Datawrapper charts).
    
    Returns:
    - tuple: (charts_info, markdown_summary)
        charts_info (list): List of dictionaries with chart details including URLs.
        markdown_summary (str): A concise Markdown summary of the anomalies with links to charts.
    """
    # Get the aggregation function from metadata
    agg_function = metadata.get('agg_function', 'sum')
    agg_function_display = 'Average' if agg_function == 'mean' else 'Total'
    
    # Initialize list to store chart information
    charts_info = []
    
    # Sort anomalies by 'out_of_bounds' status
    sorted_anomalies = sorted(results, key=lambda x: x['out_of_bounds'], reverse=True)
    
    # Generate charts for each anomaly
    for item in sorted_anomalies:
        if item['out_of_bounds']:
            # Create chart title
            chart_title = f"Anomaly in {agg_function_display} {metadata.get('y_axis_label', '')} in {item['group_value']}"
            
            try:
                # Generate Datawrapper chart
                chart_url = generate_anomaly_chart_dw(item, chart_title, metadata, output_dir)
                
                if chart_url:
                    # Store chart information
                    charts_info.append({
                        'group_value': item['group_value'],
                        'chart_title': chart_title,
                        'chart_url': chart_url,
                        'recent_mean': item['recent_mean'],
                        'comparison_mean': item['comparison_mean'],
                        'difference': item['difference'],
                        'percent_difference': (item['difference'] / item['comparison_mean']) * 100 if item['comparison_mean'] else 0
                    })
            except Exception as e:
                logger.error(f"Failed to generate chart for group {item.get('group_value', 'Unknown')}: {e}")
    
    # Generate Markdown summary
    markdown_summary = generate_markdown_summary(results, charts_info, metadata)
    
    return charts_info, markdown_summary

def generate_markdown_summary(results, charts_info, metadata):
    """
    Generates a Markdown summary of the anomalies with links to Datawrapper charts.
    
    Parameters:
    - results (list of dict): List containing all anomaly details.
    - charts_info (list of dict): List containing information about generated charts.
    - metadata (dict): Contains comparison_period and recent_period information.
    
    Returns:
    - str: Markdown summary of the anomalies with links to charts.
    """
    # Get the aggregation function from metadata
    agg_function = metadata.get('agg_function', 'sum')
    agg_function_display = 'Average' if agg_function == 'mean' else 'Total'
    
    summary = "## Anomaly Detection Summary\n\n"
    
    # Add metadata information
    title = metadata.get('title', 'Anomaly Detection Results')
    if title:
        summary += f"### {title} ({agg_function_display})\n\n"
    
    # Periods information
    summary += "**Period Information:**\n\n"
    
    # Format dates based on period_type
    period_type = metadata.get('period_type', 'month')
    if period_type == 'year':
        date_format = '%Y'
    else:  # month, day
        date_format = '%b %Y'
    
    if 'recent_period' in metadata and 'comparison_period' in metadata:
        recent_start = metadata['recent_period']['start'].strftime(date_format)
        recent_end = metadata['recent_period']['end'].strftime(date_format)
        comp_start = metadata['comparison_period']['start'].strftime(date_format)
        comp_end = metadata['comparison_period']['end'].strftime(date_format)
        
        summary += f"- Recent Period: {recent_start} to {recent_end}\n"
        summary += f"- Comparison Period: {comp_start} to {comp_end}\n\n"
    
    # Count anomalies
    anomalies = [row for row in results if row.get('out_of_bounds', False)]
    
    if anomalies:
        summary += f"**{len(anomalies)} Anomalies Detected:**\n\n"
        
        # Sort anomalies by percent difference
        anomalies.sort(key=lambda x: abs((x['difference'] / x['comparison_mean']) * 100 if x['comparison_mean'] else 0), reverse=True)
        
        # For each anomaly, find corresponding chart info
        for i, anomaly in enumerate(anomalies, 1):
            group = anomaly.get('group_value', 'Unknown')
            recent = anomaly.get('recent_mean', 0)
            comp = anomaly.get('comparison_mean', 0)
            pct_diff = (anomaly['difference'] / anomaly['comparison_mean']) * 100 if anomaly['comparison_mean'] else 0
            direction = "increase" if recent > comp else "decrease"
            
            # Find the chart URL if available
            chart_url = next((chart['chart_url'] for chart in charts_info if chart['group_value'] == group), None)
            
            # Add the anomaly description with chart link if available
            summary += f"{i}. **{group}**: {agg_function_display} {metadata.get('y_axis_label', 'Value')} {direction}d by **{abs(pct_diff):.1f}%** "
            summary += f"(from {comp:.1f} to {recent:.1f})"
            
            if chart_url:
                summary += f" - [View Chart]({chart_url})"
            
            summary += "\n\n"
    else:
        summary += "**No anomalies were detected.**\n\n"
    
    # Add summary table
    all_results = sorted(results, key=lambda x: abs((x['difference'] / x['comparison_mean']) * 100 if x['comparison_mean'] else 0), reverse=True)
    
    if all_results:
        summary += "### Summary Table\n\n"
        summary += "| Group | Recent | Comparison | % Change | Anomaly |\n"
        summary += "|-------|--------|------------|----------|--------|\n"
        
        for row in all_results:
            group = row.get('group_value', 'Unknown')
            recent = row.get('recent_mean', 0)
            comp = row.get('comparison_mean', 0)
            diff = row.get('difference', 0)
            pct_diff = (diff / comp) * 100 if comp else 0
            anomaly = "Yes" if row.get('out_of_bounds', False) else "No"
            
            # Find chart URL if available
            chart_url = next((chart['chart_url'] for chart in charts_info if chart['group_value'] == group), None)
            
            # Add hyperlink if chart exists
            if chart_url and row.get('out_of_bounds', False):
                summary += f"| [{group}]({chart_url}) | {recent:.1f} | {comp:.1f} | {pct_diff:.1f}% | {anomaly} |\n"
            else:
                summary += f"| {group} | {recent:.1f} | {comp:.1f} | {pct_diff:.1f}% | {anomaly} |\n"
    
    return summary

def test_with_crime_data(json_data=None):
    """Test generating an anomaly chart with San Francisco violent crime data.
    
    Args:
        json_data (dict, optional): Complete JSON response from the anomaly detection API.
            If None, uses default sample data.
    """
    logger.info("Starting test with SF violent crime data")
    
    if json_data is None:
        # Default sample crime anomaly data
        crime_data = {
            "id": 286846,
            "group_value": "Northern",
            "group_field_name": "police_district",
            "field_name": "value",
            "period_type": "month",
            "comparison_mean": 25.375,
            "recent_mean": 8.0,
            "difference": -17.375,
            "percent_change": -68.47,
            "std_dev": 7.204237757505416,
            "stdDev": 7.204237757505416,  # Ensure we have this field for compatibility
            "out_of_bounds": True,
            # Add chart data
            "dates": [
                "2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", 
                "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", 
                "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", 
                "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", 
                "2025-04"
            ],
            "counts": [
                18.0, 21.0, 18.0, 35.0, 25.0, 33.0, 32.0, 35.0, 36.0, 31.0, 18.0, 
                30.0, 29.0, 26.0, 25.0, 20.0, 24.0, 25.0, 40.0, 22.0, 17.0, 16.0, 
                13.0, 20.0, 8.0
            ]
        }
        
        # Set up metadata from the anomaly data
        metadata = {
            "title": "🚨 Violent Crime Incidents - District 3 - value by police_district",
            "y_axis_label": "Violent Crime Incidents - District 3",
            "period_type": "month",
            "agg_function": "sum",
            "comparison_period": {
                "start": datetime.date(2023, 4, 1),
                "end": datetime.date(2025, 3, 31)
            },
            "recent_period": {
                "start": datetime.date(2025, 4, 1),
                "end": datetime.date(2025, 4, 30)
            }
        }
    else:
        # Extract data from the JSON response
        if not isinstance(json_data, dict) or 'status' not in json_data or json_data['status'] != 'success':
            logger.error("Invalid JSON data format. Expected {'status': 'success', 'anomaly': {...}}")
            return None
        
        # Extract the anomaly data
        anomaly = json_data.get('anomaly', {})
        if not anomaly:
            logger.error("No anomaly data found in JSON")
            return None
        
        # Extract chart data
        chart_data = anomaly.get('chart_data', {})
        
        # Prepare crime_data dict with required fields
        crime_data = {
            "id": anomaly.get('id'),
            "group_value": anomaly.get('group_value'),
            "group_field_name": anomaly.get('group_field_name'),
            "field_name": anomaly.get('field_name'),
            "period_type": anomaly.get('period_type', 'month'),
            "comparison_mean": anomaly.get('comparison_mean'),
            "recent_mean": anomaly.get('recent_mean'),
            "difference": anomaly.get('difference'),
            "percent_change": anomaly.get('percent_change'),
            "std_dev": anomaly.get('std_dev'),
            "stdDev": anomaly.get('std_dev'),  # Copy to stdDev for compatibility
            "out_of_bounds": anomaly.get('out_of_bounds'),
            # Chart data
            "dates": chart_data.get('dates', []),
            "counts": chart_data.get('values', [])
        }
        
        # Extract metadata
        anomaly_metadata = anomaly.get('metadata', {})
        
        # Parse date strings to datetime objects
        recent_period = anomaly_metadata.get('recent_period', {})
        comparison_period = anomaly_metadata.get('comparison_period', {})
        
        # Convert date strings to datetime objects
        if recent_period and 'start' in recent_period and 'end' in recent_period:
            recent_start = datetime.datetime.strptime(recent_period['start'], '%Y-%m-%d').date()
            recent_end = datetime.datetime.strptime(recent_period['end'], '%Y-%m-%d').date()
        else:
            # Default to current month if not specified
            today = datetime.date.today()
            recent_start = datetime.date(today.year, today.month, 1)
            recent_end = today
        
        if comparison_period and 'start' in comparison_period and 'end' in comparison_period:
            comparison_start = datetime.datetime.strptime(comparison_period['start'], '%Y-%m-%d').date()
            comparison_end = datetime.datetime.strptime(comparison_period['end'], '%Y-%m-%d').date()
        else:
            # Default to previous year if not specified
            comparison_start = datetime.date(recent_start.year - 1, recent_start.month, 1)
            comparison_end = datetime.date(recent_end.year - 1, recent_end.month, recent_end.day)
        
        # Create metadata dict
        metadata = {
            "title": anomaly_metadata.get('title', f"Anomaly in {crime_data['group_value']}"),
            "y_axis_label": anomaly_metadata.get('y_axis_label', 'Value'),
            "period_type": anomaly_metadata.get('period_type', 'month'),
            "agg_function": anomaly_metadata.get('agg_function', 'sum'),
            "comparison_period": {
                "start": comparison_start,
                "end": comparison_end
            },
            "recent_period": {
                "start": recent_start,
                "end": recent_end
            }
        }
    
    # Generate chart title
    chart_title = f"Anomaly in {metadata.get('y_axis_label', 'Values')} in {crime_data['group_value']}"
    
    # Generate chart
    chart_url = generate_anomaly_chart_dw(crime_data, chart_title, metadata)
    
    if chart_url:
        print(f"Chart created successfully! URL: {chart_url}")
        
        # Format a Markdown description for the anomaly
        percent_diff = abs(crime_data['difference'] / crime_data['comparison_mean'] * 100) if crime_data['comparison_mean'] else 0
        direction = "decrease" if crime_data['difference'] < 0 else "increase"
        
        print("\n## Anomaly Description")
        print(f"**{crime_data['group_value']}**: {metadata.get('y_axis_label')} {direction}d by "
              f"**{percent_diff:.1f}%** (from {crime_data['comparison_mean']:.1f} to {crime_data['recent_mean']:.1f})")
        print(f"**Chart**: {chart_url}")
        
        return chart_url
    else:
        print("Failed to create chart.")
        return None

def generate_anomaly_chart_from_id(anomaly_id, chart_title=None, output_dir='static'):
    """
    Generate a Datawrapper chart for an anomaly using its ID.
    
    This function fetches the anomaly data using the anomaly ID from the API,
    formats it correctly, and then calls generate_anomaly_chart_dw to create the chart.
    
    Parameters:
    - anomaly_id (str): The ID of the anomaly to chart
    - chart_title (str, optional): Title for the chart. If not provided, a default title will be generated.
    - output_dir (str): Directory to save chart details (not used for Datawrapper but kept for API compatibility).
    
    Returns:
    - str: URL of the published Datawrapper chart or None if failed.
    """
    import requests
    import os
    
    logger.info(f"Generating Datawrapper chart for anomaly ID: {anomaly_id}")
    
    # Get API base URL from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    try:
        # Make a request to the API to get anomaly details
        anomaly_details_url = f"{api_base_url}/anomaly-analyzer/api/anomaly-details/{anomaly_id}"
        logger.info(f"Requesting anomaly data from: {anomaly_details_url}")
        
        response = requests.get(anomaly_details_url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch anomaly details: {response.status_code} - {response.text}")
            return None
            
        # Parse the response
        anomaly_response = response.json()
        
        # Check if the response was successful
        if anomaly_response.get("status") != "success":
            logger.error(f"API returned error: {anomaly_response.get('message', 'Unknown error')}")
            return None
            
        # Extract the anomaly data from the response
        anomaly_data = anomaly_response.get("anomaly", {})
        if not anomaly_data:
            logger.error("No anomaly data found in the response")
            return None
            
        # Check if we have the necessary data for charting
        if "chart_data" not in anomaly_data:
            logger.error("No chart_data found in the anomaly data")
            return None
            
        # Extract the needed data for the Datawrapper chart
        chart_data = anomaly_data.get("chart_data", {})
        
        # Prepare data for generate_anomaly_chart_dw
        item = {
            'dates': chart_data.get("dates", []),
            'counts': chart_data.get("values", []),
            'comparison_mean': anomaly_data.get("comparison_mean", 0),
            'recent_mean': anomaly_data.get("recent_mean", 0),
            'difference': anomaly_data.get("difference", 0),
            'stdDev': anomaly_data.get("std_dev", 0),
            'group_value': anomaly_data.get("group_value", "Unknown")
        }
        
        # Prepare metadata for generate_anomaly_chart_dw
        metadata = anomaly_data.get("metadata", {})
        
        # Ensure period dates are proper date objects, not strings
        import datetime
        
        # Convert comparison_period dates from strings to date objects if needed
        if "comparison_period" in metadata:
            comparison_period = metadata["comparison_period"]
            if isinstance(comparison_period.get("start"), str):
                comparison_period["start"] = datetime.datetime.strptime(comparison_period["start"], "%Y-%m-%d").date()
            if isinstance(comparison_period.get("end"), str):
                comparison_period["end"] = datetime.datetime.strptime(comparison_period["end"], "%Y-%m-%d").date()
            
        # Convert recent_period dates from strings to date objects if needed
        if "recent_period" in metadata:
            recent_period = metadata["recent_period"]
            if isinstance(recent_period.get("start"), str):
                recent_period["start"] = datetime.datetime.strptime(recent_period["start"], "%Y-%m-%d").date()
            if isinstance(recent_period.get("end"), str):
                recent_period["end"] = datetime.datetime.strptime(recent_period["end"], "%Y-%m-%d").date()
        
        # Add period information to metadata if not already present
        if "comparison_period" not in metadata or "recent_period" not in metadata:
            # Get dates from chart_data
            all_dates = [datetime.datetime.fromisoformat(d).date() if isinstance(d, str) else d for d in chart_data.get("dates", [])]
            periods = chart_data.get("periods", [])
            
            if all_dates and periods and len(all_dates) == len(periods):
                # Separate dates into comparison and recent periods
                comparison_dates = [d for i, d in enumerate(all_dates) if periods[i] == "historical"]
                recent_dates = [d for i, d in enumerate(all_dates) if periods[i] == "recent"]
                
                if comparison_dates and recent_dates:
                    metadata["comparison_period"] = {
                        "start": min(comparison_dates),
                        "end": max(comparison_dates)
                    }
                    metadata["recent_period"] = {
                        "start": min(recent_dates),
                        "end": max(recent_dates)
                    }
                else:
                    logger.warning("Could not extract comparison and recent dates from chart_data")
            else:
                logger.warning("Could not extract dates and periods from chart_data")
        
        # Use provided chart title or get it from metadata if available
        if not chart_title:
            # Calculate percentage change as integer (needed for description regardless of title source)
            percent_change = int(abs(item['difference'] / item['comparison_mean'] * 100)) if item['comparison_mean'] else 0
            
            # Get the item name from metadata (object_name) - needed for both title and y-axis label
            item_name = metadata.get("object_name", metadata.get("title", "Values"))
            
            # First check if a title is available in metadata
            if 'title' in metadata:
                chart_title = metadata['title']
                logger.info(f"Using title from metadata: {chart_title}")
            else:
                # If no title in metadata, generate one (old behavior)
                # Determine if this is a spike or drop
                trend_type = "Spike" if item['recent_mean'] > item['comparison_mean'] else "Drop"
                
                # Get the group field name and format it for display (replace underscores with spaces)
                group_field_display = anomaly_data.get("group_field_name", "group").replace("_", " ").title()
                
                # Get the group value
                group_value = item['group_value']
                
                # Generate a concise chart title without markdown bold formatting
                chart_title = f"{item_name} {percent_change}% {trend_type} in {group_field_display}: {group_value}"
                logger.info(f"Generated title: {chart_title}")
            
            # Format the description
            comp_start = metadata.get("comparison_period", {}).get("start")
            comp_end = metadata.get("comparison_period", {}).get("end")
            
            if comp_start and comp_end:
                # Format months and years
                start_formatted = comp_start.strftime('%B %Y')
                end_formatted = comp_end.strftime('%B %Y')
                
                direction = "Increase" if item['recent_mean'] > item['comparison_mean'] else "Decrease"
                description = f"{direction} of {percent_change}% vs the average between {start_formatted} and {end_formatted} of {item['comparison_mean']:.0f}."
                
                # Add the description to metadata for use in chart customization
                metadata["chart_description"] = description
            
            # Use object_name from metadata as the y-axis label in the metadata
            metadata["y_axis_label"] = item_name
            logger.info(f"Setting y_axis_label in metadata to: {item_name}")
        
        # Generate the Datawrapper chart
        return generate_anomaly_chart_dw(item, chart_title, metadata, output_dir)
        
    except Exception as e:
        logger.error(f"Error generating chart for anomaly ID {anomaly_id}: {str(e)}", exc_info=True)
        return None

if __name__ == '__main__':
    # Debug info about API key
    print(f"Datawrapper API key present: {bool(DATAWRAPPER_API_KEY)}")
    
    # Example usage - this would be replaced with actual data in production
    sample_anomaly = {
        'group_value': 'Sample Department',
        'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', 
                  '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06'],
        'counts': [100, 110, 105, 115, 95, 100, 120, 125, 130, 170, 180, 190],
        'comparison_mean': 104.17,
        'recent_mean': 152.5,
        'difference': 48.33,
        'stdDev': 7.36,
        'out_of_bounds': True
    }
    
    sample_metadata = {
        'title': 'Sample Anomaly Detection',
        'y_axis_label': 'Transactions',
        'period_type': 'month',
        'agg_function': 'mean',
        'comparison_period': {
            'start': datetime.date(2022, 1, 1),
            'end': datetime.date(2022, 6, 30)
        },
        'recent_period': {
            'start': datetime.date(2023, 1, 1),
            'end': datetime.date(2023, 6, 30)
        }
    }
    
    # Sample crime data in complete API response format
    crime_json_data = {
        "status": "success",
        "anomaly": {
            "id": 286846,
            "group_value": "Northern",
            "group_field_name": "police_district",
            "field_name": "value",
            "period_type": "month",
            "comparison_mean": 25.375,
            "recent_mean": 8.0,
            "difference": -17.375,
            "percent_change": -68.47,
            "std_dev": 7.204237757505416,
            "out_of_bounds": True,
            "created_at": "2025-05-12T15:47:58.663302",
            "metadata": {
                "title": "🚨 Violent Crime Incidents - District 3 - value by police_district",
                "object_id": "2",
                "date_field": "month_period",
                "group_field": "police_district",
                "object_name": "🚨 Violent Crime Incidents - District 3",
                "object_type": "dashboard_metric",
                "period_type": "month",
                "agg_function": "sum",
                "y_axis_label": "Violent Crime Incidents - District 3",
                "numeric_field": "value",
                "recent_period": {
                    "end": "2025-04-30",
                    "start": "2025-04-01"
                },
                "comparison_period": {
                    "end": "2025-03-31",
                    "start": "2023-04-01"
                }
            },
            "chart_data": {
                "dates": [
                    "2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", 
                    "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", 
                    "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", 
                    "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", 
                    "2025-04"
                ],
                "values": [
                    18.0, 21.0, 18.0, 35.0, 25.0, 33.0, 32.0, 35.0, 36.0, 31.0, 18.0, 
                    30.0, 29.0, 26.0, 25.0, 20.0, 24.0, 25.0, 40.0, 22.0, 17.0, 16.0, 
                    13.0, 20.0, 8.0
                ],
                "periods": [
                    "comparison", "comparison", "comparison", "comparison", "comparison",
                    "comparison", "comparison", "comparison", "comparison", "comparison",
                    "comparison", "comparison", "comparison", "comparison", "comparison",
                    "comparison", "comparison", "comparison", "comparison", "comparison",
                    "comparison", "comparison", "comparison", "comparison", "recent"
                ]
            }
        }
    }
    
    print("Choose which test to run:")
    print("1. Sample test (default)")
    print("2. Crime data test (hardcoded)")
    print("3. Crime data test (from JSON)")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip() or "1"
    
    if choice == "3":
        # Run the crime data test with JSON data
        test_with_crime_data(crime_json_data)
    elif choice == "2":
        # Run the crime data test with default data
        test_with_crime_data()
    else:
        # Run the original sample test
        chart_title = f"Anomaly in Average {sample_metadata['y_axis_label']} in {sample_anomaly['group_value']}"
        chart_url = generate_anomaly_chart_dw(sample_anomaly, chart_title, sample_metadata)
        
        if chart_url:
            print(f"Sample chart created successfully: {chart_url}")
        else:
            print("Failed to create sample chart.") 
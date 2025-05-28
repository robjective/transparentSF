import logging
import json
import os
import psycopg2
import psycopg2.extras
import uuid
import requests
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import time
import pandas as pd
import ast
import random
import string
from urllib.parse import quote

"""
Enhanced Map Generation with Series Support
==========================================

This module provides enhanced map generation capabilities with support for different colored series,
allowing for better data visualization and categorization on maps.

Key Features:
- Multiple colored series support for locator maps
- Predefined color palettes (categorical, status, priority, sequential)
- Custom color palette support
- Automatic legend generation for series
- Enhanced tooltips with series information

Series Support:
--------------
The enhanced map generation supports grouping markers into different colored series based on a field
in your data. This is particularly useful for:

1. **Categorical Data**: Different types of locations (e.g., Police Stations, Fire Stations, Hospitals)
2. **Status Data**: Different states (e.g., Active, Pending, Inactive)
3. **Priority Data**: Different priority levels (e.g., High, Medium, Low)
4. **Sequential Data**: Graduated values with color progression

Usage Examples:
--------------

1. Basic Series Map:
```python
location_data = [
    {"title": "Police Station", "lat": 37.7749, "lon": -122.4194, "series": "Police"},
    {"title": "Fire Station", "lat": 37.7849, "lon": -122.4094, "series": "Fire"},
    {"title": "Hospital", "lat": 37.7627, "lon": -122.4581, "series": "Medical"}
]

generate_map(
    context_variables={},
    map_title="SF Public Services",
    map_type="point",
    location_data=location_data,
    series_field="series",
    color_palette="categorical"  # Uses predefined categorical colors
)
```

2. Custom Color Palette:
```python
custom_colors = ["#FF0000", "#00FF00", "#0000FF"]  # Red, Green, Blue

generate_map(
    context_variables={},
    map_title="Custom Colors Map",
    map_type="point", 
    location_data=location_data,
    series_field="series",
    color_palette=custom_colors
)
```

3. Status-based Coloring:
```python
location_data = [
    {"title": "Active Site", "lat": 37.7749, "lon": -122.4194, "series": "Active"},
    {"title": "Pending Site", "lat": 37.7849, "lon": -122.4094, "series": "Pending"},
    {"title": "Inactive Site", "lat": 37.7627, "lon": -122.4581, "series": "Inactive"}
]

generate_map(
    context_variables={},
    map_title="Site Status Map",
    map_type="point",
    location_data=location_data,
    series_field="series",
    color_palette="status"  # Green, Amber, Red, Blue, Purple
)
```

Available Color Palettes:
------------------------
- **categorical**: 12 distinct colors for general categorization
- **status**: 5 colors for status indication (Green, Amber, Red, Blue, Purple)
- **priority**: 4 colors for priority levels (Red, Orange, Green, Grey)
- **sequential**: 9 colors for graduated/sequential data (light to dark progression)

The series functionality automatically:
- Assigns consistent colors to each series value
- Creates a legend showing all series with their colors
- Enhances tooltips to include series information
- Maintains color consistency across map updates

Supported Map Types for Series:
------------------------------
- point: Coordinate-based markers
- address: Address-based markers (geocoded automatically)
- intersection: Street intersection markers

Note: Series functionality is not available for district-based maps (supervisor_district, police_district)
as these use choropleth styling instead of individual markers.
"""

# Configure logging
logger = logging.getLogger(__name__)

# Determine the project root based on the script's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Moves up two levels: tools -> ai -> project_root
ai_dir = script_dir.parent  # This should be the 'ai' directory

# Load environment variables
possible_env_paths = [
    ai_dir / '.env',          # Check ai/.env first
    project_root / '.env',    # Then check project_root/.env
    Path.home() / '.env'      # Finally, check home directory
]

loaded_env = False
for env_path in possible_env_paths:
    if env_path.exists():
        logger.info(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        loaded_env = True
        break

if not loaded_env:
    logger.warning("No .env file found. Relying on environment variables being set.")

# Get Datawrapper API key
DATAWRAPPER_API_KEY = os.getenv("DATAWRAPPER_API_KEY")
if not DATAWRAPPER_API_KEY:
    logger.warning("DATAWRAPPER_API_KEY not found in environment variables. Datawrapper functionality will not work.")

# Default reference chart for district maps
DEFAULT_DISTRICT_CHART = os.getenv("DATAWRAPPER_REFERENCE_CHART", "j5vON")
# Default reference chart for locator maps
DEFAULT_LOCATOR_CHART = "dRKcH"

def _make_dw_request(method, endpoint, headers=None, data=None, json_payload=None):
    """Helper function to make requests to Datawrapper API."""
    if not DATAWRAPPER_API_KEY:
        logger.error("DATAWRAPPER_API_KEY not set. Cannot make API requests.")
        raise ValueError("DATAWRAPPER_API_KEY not set.")
        
    url = f"https://api.datawrapper.de/v3{endpoint}"
    
    default_headers = {
        "Authorization": f"Bearer {DATAWRAPPER_API_KEY}"
    }
    if headers:
        default_headers.update(headers)

    try:
        logger.info(f"Making {method} request to {url}")
        if data:
            pass
        if json_payload:
            pass
        
        response = requests.request(method, url, headers=default_headers, data=data, json=json_payload)
        
        logger.info(f"Response status code: {response.status_code}")
        
        response.raise_for_status()
        
        if response.content and response.headers.get('Content-Type', '').startswith('application/json'):
            try:
                json_response = response.json()
                return json_response
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e} for URL: {url}. Response text: {response.text}")
                return response.text 
        elif response.content:
                return response.text
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text} for URL: {url}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e} for URL: {url}")
        raise

def _prepare_locator_marker_data(location_data, series_field=None, color_palette=None):
    """
    Prepares marker data for Datawrapper locator maps from a list of location details.
    Enhanced to support different colored series for better data visualization.
    
    Args:
        location_data: A list of dictionaries, where each dictionary contains
                       details for a point (e.g., title, coordinates, tooltip, color).
                       Can handle both coordinate-based and address-based data.
                       Examples:
                       [
                           {
                               "title": "Point A",
                               "coordinates": [-122.4, 37.7],
                               "tooltip": "This is Point A",
                               "markerColor": "#ff0000", (optional)
                               "series": "Category 1", (optional - for colored series)
                           },
                           {
                               "title": "Point B",
                               "lat": 37.7,
                               "lon": -122.4,
                               "tooltip": "This is Point B",
                               "series": "Category 2", (optional)
                           },
                           {
                               "title": "Address C", 
                               "address": "123 Main St, San Francisco, CA",
                               "tooltip": "This is Address C",
                               "markerColor": "#0000ff", (optional)
                           }
                       ]
        series_field (str, optional): Field name to use for series grouping (e.g., 'series', 'category', 'type')
        color_palette (list, optional): List of colors to use for series. If not provided, uses default palette.
        
    Returns:
        A JSON string representing the marker data for Datawrapper API.
    """
    import uuid
    
    # Define default color palettes for series
    default_color_palettes = {
        'categorical': [
            '#ad35fa',  # Bright Purple (primary brand color)
            '#FF6B5A',  # Warm Coral
            '#4A7463',  # Spruce Green
            '#71B2CA',  # Sky Blue
            '#FFC107',  # Amber
            '#9C27B0',  # Purple
            '#2196F3',  # Light Blue
            '#E91E63',  # Pink
            '#4CAF50',  # Green
            '#FF5722',  # Deep Orange
            '#795548',  # Brown
            '#607D8B',  # Blue Grey
        ],
        'status': [
            '#4CAF50',  # Green - Good/Active
            '#FFC107',  # Amber - Warning/Pending
            '#F44336',  # Red - Error/Inactive
            '#2196F3',  # Blue - Info/Processing
            '#9C27B0',  # Purple - Special
        ],
        'priority': [
            '#F44336',  # Red - High Priority
            '#FF9800',  # Orange - Medium Priority
            '#4CAF50',  # Green - Low Priority
            '#9E9E9E',  # Grey - No Priority
        ],
        'sequential': [
            '#f7fcf0',  # Very Light Green
            '#e0f3db',  # Light Green
            '#ccebc5',  # Medium Light Green
            '#a8ddb5',  # Medium Green
            '#7bccc4',  # Teal Green
            '#4eb3d3',  # Light Blue
            '#2b8cbe',  # Medium Blue
            '#0868ac',  # Dark Blue
            '#084081',  # Very Dark Blue
        ]
    }
    
    # Use provided color palette or default categorical palette
    if color_palette is None:
        color_palette = default_color_palettes['categorical']
    elif isinstance(color_palette, str) and color_palette in default_color_palettes:
        color_palette = default_color_palettes[color_palette]
    
    def generate_unique_id():
        """Generate a unique ID similar to Datawrapper's format"""
        # Generate a 10-character random string similar to Datawrapper's IDs
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(10))
    
    def parse_coordinates(coord_value):
        """
        Parse coordinates from various formats into [longitude, latitude] format.
        
        Args:
            coord_value: Can be:
                - List/tuple: [-122.4, 37.7] (already in lon, lat format)
                - List/tuple: [37.7, -122.4] (lat, lon format - will be detected and swapped)
                - String: "37.7749,-122.4194" (lat,lon format)
                - String: "-122.4194,37.7749" (lon,lat format)
                
        Returns:
            [longitude, latitude] list or None if parsing fails
        """
        if not coord_value:
            return None
            
        try:
            if isinstance(coord_value, (list, tuple)) and len(coord_value) >= 2:
                lon, lat = float(coord_value[0]), float(coord_value[1])
                # Check if coordinates are in lat,lon format (lat should be between -90 and 90)
                # and lon should be between -180 and 180
                if -90 <= lon <= 90 and -180 <= lat <= 180:
                    # This looks like lat,lon format, swap them
                    return [lat, lon]  # Return as [longitude, latitude]
                else:
                    # Assume it's already in lon,lat format
                    return [lon, lat]
                    
            elif isinstance(coord_value, str):
                # Parse string coordinates
                parts = coord_value.split(',')
                if len(parts) >= 2:
                    coord1, coord2 = float(parts[0].strip()), float(parts[1].strip())
                    # Determine if this is lat,lon or lon,lat based on ranges
                    # Latitude is typically between -90 and 90
                    # Longitude is typically between -180 and 180
                    # For SF area: lat ~37.7, lon ~-122.4
                    if -90 <= coord1 <= 90 and -180 <= coord2 <= 180:
                        # First coordinate looks like latitude, second like longitude
                        return [coord2, coord1]  # Return as [longitude, latitude]
                    else:
                        # Assume first is longitude, second is latitude
                        return [coord1, coord2]
                        
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Failed to parse coordinates '{coord_value}': {e}")
            
        return None
    
    # If series_field is specified, collect all unique series values and assign colors
    series_color_map = {}
    if series_field:
        unique_series = set()
        for loc in location_data:
            series_value = loc.get(series_field)
            if series_value is not None:
                unique_series.add(str(series_value))
        
        # Sort series for consistent color assignment
        sorted_series = sorted(list(unique_series))
        
        # Assign colors to each series
        for i, series_value in enumerate(sorted_series):
            series_color_map[series_value] = color_palette[i % len(color_palette)]
        
        logger.info(f"Created series color mapping for {len(sorted_series)} series: {series_color_map}")
    
    markers = []
    for loc in location_data:
        # Check if we have coordinates in various formats
        parsed_coordinates = None
        has_lat_lon = "lat" in loc and "lon" in loc and loc["lat"] is not None and loc["lon"] is not None
        has_address = "address" in loc and loc["address"]
        
        # Try to parse coordinates first
        if "coordinates" in loc and loc["coordinates"]:
            parsed_coordinates = parse_coordinates(loc["coordinates"])
        
        if not loc.get("title"):
            logger.warning(f"Skipping location due to missing title: {loc}")
            continue
            
        if not parsed_coordinates and not has_lat_lon and not has_address:
            logger.warning(f"Skipping location due to missing valid coordinates, lat/lon, or address: {loc}")
            continue

        # Determine marker color based on series or explicit color
        marker_color = "#8b5cf6"  # Default purple
        
        if series_field and series_field in loc and loc[series_field] is not None:
            # Use series-based color
            series_value = str(loc[series_field])
            marker_color = series_color_map.get(series_value, marker_color)
            logger.debug(f"Using series color for '{loc['title']}' (series: {series_value}): {marker_color}")
        elif loc.get("markerColor"):
            # Use explicit marker color
            marker_color = loc["markerColor"]
        elif loc.get("color"):
            # Use explicit color
            marker_color = loc["color"]
        
        text_color = loc.get("textColor", "#333333")
        text_halo = loc.get("textHalo", "#f2f3f0")
        
        # Create marker in the exact format that works
        marker = {
            "type": "point",
            "title": loc["title"],
            "icon": {
                "id": loc.get("icon_id", "circle"),
                "path": loc.get("icon_path", "M1000 350a500 500 0 0 0-500-500 500 500 0 0 0-500 500 500 500 0 0 0 500 500 500 500 0 0 0 500-500z"),
                "height": loc.get("icon_height", 700),
                "width": loc.get("icon_width", 1000),
                "horiz-adv-x": 1000,
                "scale": 1
            },
            "scale": loc.get("scale", 0.6),  # Even smaller markers by default
            "markerColor": marker_color,
            "opacity": loc.get("opacity", 0.5),  # 50% opacity by default
            "text": {
                "color": text_color,
                "fontSize": loc.get("fontSize", 14),
                "halo": text_halo,
                "bold": loc.get("bold", False),
                "italic": loc.get("italic", False),
                "uppercase": loc.get("uppercase", False),
                "space": loc.get("space", False)
            },
            "id": generate_unique_id(),  # Generate unique ID like Datawrapper
            "markerSymbol": loc.get("markerSymbol", ""),
            "markerTextColor": text_color,
            "anchor": "bottom-center",  # Use bottom-center like working format
            "offsetY": loc.get("offsetY", 0),
            "offsetX": loc.get("offsetX", 0),
            "labelStyle": loc.get("labelStyle", "plain"),
            "class": loc.get("class", ""),
            "rotate": loc.get("rotate", 0),
            "visible": loc.get("visible", True),
            "locked": loc.get("locked", False),
            "preset": loc.get("preset", "-"),
            "alpha": loc.get("alpha", 0.5),  # Alternative opacity property
            "visibility": {
                "mobile": loc.get("mobile_visible", True),
                "desktop": loc.get("desktop_visible", True)
            },
            "connectorLine": {
                "enabled": loc.get("connector_enabled", False),
                "arrowHead": loc.get("arrow_head", "lines"),
                "type": loc.get("connector_type", "curveRight"),
                "targetPadding": loc.get("target_padding", 3),
                "stroke": loc.get("stroke", 1),
                "lineLength": loc.get("line_length", 0)
            }
        }
        
        # Add series information to marker for reference
        if series_field and series_field in loc:
            marker["series"] = str(loc[series_field])
        
        # Use parsed coordinates, lat/lon, or address for geocoding
        if parsed_coordinates:
            marker["coordinates"] = parsed_coordinates
            logger.debug(f"Using parsed coordinates for '{loc['title']}': {parsed_coordinates}")
        elif has_lat_lon:
            # Convert lat/lon to coordinates format [longitude, latitude]
            marker["coordinates"] = [float(loc["lon"]), float(loc["lat"])]
            logger.debug(f"Using lat/lon for '{loc['title']}': [{loc['lon']}, {loc['lat']}]")
        elif has_address:
            marker["address"] = loc["address"]
            logger.debug(f"Using address for '{loc['title']}': {loc['address']}")
        
        # Add tooltip if provided - enhance with series information
        tooltip_text = loc.get("tooltip") or loc.get("description", "")
        if series_field and series_field in loc and loc[series_field] is not None:
            series_info = f"Category: {loc[series_field]}"
            if tooltip_text:
                tooltip_text = f"{tooltip_text}<br>{series_info}"
            else:
                tooltip_text = series_info
        
        marker["tooltip"] = {
            "text": tooltip_text,
            "enabled": bool(tooltip_text)  # Enable tooltip only if there's text
        }
        
        markers.append(marker)

    logger.info(f"Prepared {len(markers)} markers for Datawrapper with series support")
    if series_color_map:
        logger.info(f"Series color mapping: {series_color_map}")
    
    return json.dumps({"markers": markers})

def _verify_and_fix_markers(chart_id, expected_markers_json, max_retries=3):
    """
    Verify that markers are properly loaded in the chart and fix if needed.
    
    Args:
        chart_id: The Datawrapper chart ID
        expected_markers_json: JSON string of expected marker data
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if markers are verified, False otherwise
    """
    import time
    
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        logger.info(f"Verification attempt {attempt + 1}/{max_retries} for chart {chart_id}")
        
        # Wait a bit for Datawrapper to process the data
        if attempt > 0:
            wait_time = 2 * attempt  # Progressive delay: 2s, 4s, 6s
            logger.info(f"Waiting {wait_time} seconds for Datawrapper to process markers...")
            time.sleep(wait_time)
        
        # Get current chart data
        chart_data_response = _make_dw_request("GET", f"/charts/{chart_id}/data")
        
        if chart_data_response:
            # Parse the expected markers
            expected_markers = json.loads(expected_markers_json)
            expected_count = len(expected_markers.get("markers", []))
            
            # Check if the response contains marker data
            if isinstance(chart_data_response, str):
                try:
                    current_data = json.loads(chart_data_response)
                except json.JSONDecodeError:
                    current_data = {}
            elif isinstance(chart_data_response, dict):
                current_data = chart_data_response
            else:
                current_data = {}
            
            current_markers = current_data.get("markers", [])
            current_count = len(current_markers)
            
            logger.info(f"Found {current_count} markers in chart data, expected {expected_count}")
            
            if current_count == expected_count and current_count > 0:
                logger.info(f"Markers verified successfully! Found all {current_count} expected markers.")
                return True
            elif current_count == 0:
                logger.warning(f"No markers found in chart data on attempt {attempt + 1}")
                # Re-upload the marker data
                logger.info(f"Re-uploading marker data for chart {chart_id}")
                upload_response = _make_dw_request("PUT", f"/charts/{chart_id}/data", headers={'Content-Type': 'application/json'}, data=expected_markers_json)
                if upload_response:
                    logger.info("Marker data re-uploaded successfully")
                else:
                    logger.error("Failed to re-upload marker data")
            else:
                logger.warning(f"Marker count mismatch: found {current_count}, expected {expected_count}")
        else:
            logger.error(f"Failed to retrieve chart data for verification on attempt {attempt + 1}")
    
    logger.error(f"Failed to verify markers after {max_retries} attempts")
    return False

def _refresh_chart_before_publish(chart_id):
    """
    Forces a refresh of the chart to ensure all data and metadata is properly processed
    before publishing.
    
    Args:
        chart_id (str): The chart ID to refresh
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the current chart data to force a refresh
        chart_data = _make_dw_request("GET", f"/charts/{chart_id}")
        if not chart_data:
            logger.error(f"Failed to refresh chart {chart_id}")
            return False
            
        # Also get the chart's data to ensure it's properly loaded
        chart_content = _make_dw_request("GET", f"/charts/{chart_id}/data")
        if not chart_content:
            logger.error(f"Failed to get chart data for {chart_id}")
            return False
            
        logger.info(f"Successfully refreshed chart {chart_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error refreshing chart {chart_id}: {str(e)}", exc_info=True)
        return False

def _fit_map_to_markers(chart_id, markers_json_string):
    """
    Fit the map view to show all markers optimally.
    
    Args:
        chart_id (str): The chart ID to update
        markers_json_string (str): JSON string containing marker data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Parse the markers to get coordinates
        markers_data = json.loads(markers_json_string)
        markers = markers_data.get("markers", [])
        
        if not markers:
            logger.warning("No markers found to fit map to")
            return False
        
        # Extract coordinates from markers
        coordinates = []
        for marker in markers:
            if marker.get("coordinates"):
                coordinates.append(marker["coordinates"])
        
        if not coordinates:
            logger.warning("No valid coordinates found in markers")
            return False
        
        # Calculate bounds
        lons = [coord[0] for coord in coordinates]
        lats = [coord[1] for coord in coordinates]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Calculate center
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        center_coords = [center_lon, center_lat]
        
        # Calculate appropriate zoom level based on bounds
        # This is a rough calculation - you might need to fine-tune
        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        max_diff = max(lon_diff, lat_diff)
        
        # Zoom level calculation (rough approximation)
        if max_diff > 0.5:
            zoom_level = 9
        elif max_diff > 0.2:
            zoom_level = 10
        elif max_diff > 0.1:
            zoom_level = 11
        elif max_diff > 0.05:
            zoom_level = 12
        elif max_diff > 0.02:
            zoom_level = 13
        elif max_diff > 0.01:
            zoom_level = 14
        else:
            zoom_level = 15
        
        # Add some padding by reducing zoom slightly
        zoom_level = max(8, zoom_level - 1)
        
        logger.info(f"Fitting map to markers: center={center_coords}, zoom={zoom_level}")
        
        # Update the chart view
        view_payload = {
            "metadata": {
                "visualize": {
                    "view": {
                        "center": center_coords,
                        "zoom": zoom_level,
                        "height": 75,
                        "pitch": 0,
                        "bearing": 0
                    }
                }
            }
        }
        
        _make_dw_request(
            "PATCH",
            f"/charts/{chart_id}",
            json_payload=view_payload
        )
        
        logger.info(f"Successfully fitted map {chart_id} to markers")
        return True
        
    except Exception as e:
        logger.error(f"Error fitting map to markers: {str(e)}", exc_info=True)
        return False

def _create_and_configure_locator_map(chart_title, markers_json_string, center_coords=None, zoom_level=None, series_info=None):
    """
    Creates an empty locator map, adds markers, and configures its view.

    Args:
        chart_title (str): The title for the new map.
        markers_json_string (str): A JSON string containing the marker data.
        center_coords (list, optional): [longitude, latitude] for map center.
        zoom_level (int/float, optional): Zoom level for the map.
        series_info (dict, optional): Information about series including color mapping for legend.

    Returns:
        str: The ID of the newly created and configured chart, or None on failure.
    """
    try:
        # 1. Create an empty locator map chart
        logger.info(f"Creating empty locator map with title: {chart_title}")
        create_chart_payload = {
            "title": chart_title,
            "type": "locator-map"
        }
        create_chart_response = _make_dw_request(
            "POST",
            "/charts",
            json_payload=create_chart_payload
        )
        if not create_chart_response or "id" not in create_chart_response:
            logger.error(f"Failed to create empty locator map. Response: {create_chart_response}")
            return None
        chart_id = create_chart_response["id"]
        logger.info(f"Successfully created empty locator map with ID: {chart_id}")

        # 2. Add markers to the map
        logger.info(f"Adding markers to map ID: {chart_id}")
        add_markers_response = _make_dw_request(
            "PUT",
            f"/charts/{chart_id}/data",
            headers={'Content-Type': 'application/json'}, # Ensure correct content type
            data=markers_json_string
        )
        # PUT /data returns 204 No Content on success, so response might be None or empty
        # We rely on raise_for_status in _make_dw_request to catch errors
        logger.info(f"Markers added to map ID: {chart_id}. Response status implies success if no error.")

        # 3. Set initial metadata and styling
        logger.info(f"Setting initial metadata for map ID: {chart_id}")
        
        initial_metadata_payload = {
            "metadata": {
                "data": {
                    "json": True
                },
                "visualize": {
                    "basemap": "osm",
                    "defaultMapSize": 600,
                    "visibility": {
                        "boundary_country": True,
                        "boundary_state": False,
                        "building": True,
                        "green": True,
                        "mountains": False,
                        "roads": True,
                        "urban": True,
                        "water": True,
                        "building3d": False
                    },
                    "mapLabel": True,
                    "scale": False,
                    "compass": False,
                    "miniMap": {
                        "enabled": False,
                        "bounds": []
                    },
                    "key": {
                        "enabled": bool(series_info and series_info.get('color_mapping')),
                        "title": series_info.get('legend_title', 'Categories') if series_info else "",
                        "items": []
                    },
                    "zoom": {
                        "enabled": True,
                        "position": "topright"
                    },
                    "source-name": "TransparentSF",
                    "source-url": "https://transparentsf.org",
                    "byline": "Generated by TransparentSF"
                }
            }
        }
        _make_dw_request(
            "PATCH",
            f"/charts/{chart_id}",
            json_payload=initial_metadata_payload
        )
        logger.info(f"Initial metadata set for map ID: {chart_id}")

        # Skip legend configuration for now to avoid breaking the map
        # TODO: Re-implement legend functionality once basic series coloring works
        if series_info and series_info.get('color_mapping'):
            logger.info(f"Series detected but skipping legend configuration to ensure map renders properly")
            logger.info(f"Color mapping: {series_info.get('color_mapping')}")

        # 4. Fit map to markers (this will override center_coords and zoom_level if not provided)
        if center_coords is None or zoom_level is None:
            logger.info(f"Fitting map to markers for optimal view")
            if not _fit_map_to_markers(chart_id, markers_json_string):
                logger.warning("Failed to fit map to markers, using fallback settings")
                # Fallback to manual calculation
                try:
                    markers_data = json.loads(markers_json_string)
                    if markers_data.get("markers"):
                        lats = [m["coordinates"][1] for m in markers_data["markers"] if m.get("coordinates")]
                        lons = [m["coordinates"][0] for m in markers_data["markers"] if m.get("coordinates")]
                        if lats and lons:
                            center_coords = [(min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2]
                            zoom_level = 11
                        else:
                            center_coords = [-122.44, 37.77] # SF default
                            zoom_level = 11
                    else:
                        center_coords = [-122.44, 37.77] # SF default
                        zoom_level = 11
                except Exception as e:
                    logger.error(f"Error calculating fallback center/zoom: {str(e)}")
                    center_coords = [-122.44, 37.77] # SF default
                    zoom_level = 11

        # 5. Set the custom view with center and zoom
        if center_coords and zoom_level:
            logger.info(f"Setting custom view for map ID: {chart_id}")
            custom_view_payload = {
                "metadata": {
                    "visualize": {
                        "customView": {
                            "center": {
                                "lat": center_coords[1],
                                "lon": center_coords[0]
                            },
                            "zoom": zoom_level,
                            "height": 75,
                            "pitch": 0,
                            "bearing": 0
                        }
                    }
                }
            }
            _make_dw_request(
                "PATCH",
                f"/charts/{chart_id}",
                json_payload=custom_view_payload
            )

        logger.info(f"Successfully created and configured locator map with ID: {chart_id}")
        return chart_id
    
    except Exception as e:
        logger.error(f"Error in _create_and_configure_locator_map: {str(e)}", exc_info=True)
        # Attempt to get chart_id if it was created before the error
        if 'chart_id' in locals() and chart_id:
            logger.error(f"An error occurred after chart {chart_id} was created. The chart might be in an incomplete state.")
            return chart_id # Return potentially incomplete chart ID
        return None

def process_location_data(location_data, map_type):
    """
    Process location data based on map type.
    
    Args:
        location_data: List of location data objects or JSON string
        map_type: Type of map (supervisor_district, police_district, intersection, point, address)
        
    Returns:
        Processed location data
    """
    processed_data = []
    
    # If location_data is a string, try to parse it as JSON
    if isinstance(location_data, str):
        try:
            location_data = json.loads(location_data)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse location_data as JSON: {location_data}")
            return []
    
    # Ensure location_data is a list
    if not isinstance(location_data, list):
        location_data = [location_data]
    
    # Process each item based on map type
    if map_type == "address":
        # For address maps, simply ensure the address is properly formatted
        for item in location_data:
            if isinstance(item, str):
                # Handle case where item is directly a string (assume it's an address)
                address = item
                processed_item = {"address": address}
            else:
                # Normal dictionary case
                processed_item = item.copy() if hasattr(item, 'copy') else {"address": str(item)}
                address = item.get("address") if hasattr(item, 'get') else str(item)
                
            # If we don't have an address, skip this item
            if not address:
                logger.warning("Address missing from location data item")
                continue
                
            # Ensure the address includes San Francisco for better geocoding
            if "address" in processed_item and processed_item["address"]:
                address = processed_item["address"]
                if "san francisco" not in address.lower() and "sf" not in address.lower():
                    processed_item["address"] = f"{address}, San Francisco, CA"
                
            # Add the item directly - Datawrapper will handle geocoding
            processed_data.append(processed_item)
    
    elif map_type == "intersection":
        # For intersection type, ensure proper formatting
        for item in location_data:
            if isinstance(item, str):
                # Assume this is an intersection name
                intersection = item
                processed_item = {"intersection": intersection}
                
                # Format the intersection as an address for Datawrapper
                if "san francisco" not in intersection.lower() and "sf" not in intersection.lower():
                    address = f"{intersection}, San Francisco, CA"
                else:
                    address = intersection
                    
                processed_item["address"] = address
            else:
                processed_item = item.copy() if hasattr(item, 'copy') else {"intersection": str(item)}
                intersection = item.get("intersection") if hasattr(item, 'get') else None
                
                # If it has lat/lon, use those directly
                if "lat" in processed_item and "lon" in processed_item:
                    processed_data.append(processed_item)
                    continue
                    
                # Otherwise, format the intersection as an address
                if intersection:
                    if "san francisco" not in intersection.lower() and "sf" not in intersection.lower():
                        address = f"{intersection}, San Francisco, CA"
                    else:
                        address = intersection
                        
                    processed_item["address"] = address
                    processed_data.append(processed_item)
            
    elif map_type in ["supervisor_district", "police_district"]:
        # For district maps, process district identifiers
        for item in location_data:
            if isinstance(item, str):
                # Assume this is a district identifier
                processed_item = {"district": item, "value": 0}
            else:
                processed_item = item.copy() if hasattr(item, 'copy') else {"district": str(item), "value": 0}
            
            # Ensure the district is a string
            if "district" in processed_item and not isinstance(processed_item["district"], str):
                processed_item["district"] = str(processed_item["district"])
                
            processed_data.append(processed_item)
    
    elif map_type == "point":
        # For point type, ensure we have lat/lon
        for item in location_data:
            if isinstance(item, dict):
                # Handle both lat/lon and latitude/longitude formats
                lat = item.get("lat") or item.get("latitude")
                lon = item.get("lon") or item.get("longitude")
                
                if lat is not None and lon is not None:
                    processed_item = {
                        "lat": float(lat),
                        "lon": float(lon),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "color": item.get("color", "#cc0000"),
                        "icon": item.get("icon", "circle"),
                        "scale": item.get("scale", "1")
                    }
                    processed_data.append(processed_item)
                else:
                    logger.warning(f"Point data missing lat/lon: {item}")
            else:
                logger.warning(f"Cannot process non-dictionary point data: {item}")
    
    return processed_data

def create_datawrapper_chart(chart_title, location_data, map_type="supervisor_district", reference_chart_id=None, center_coords=None, zoom_level=None, series_field=None, color_palette=None):
    """
    Creates a Datawrapper chart, either by cloning a reference chart or creating from scratch.

    For locator maps (map_type='point', 'address', 'intersection'), it uses the new creation logic.
    For other types, it attempts to clone and update data.
    
    Args:
        chart_title (str): Title for the chart.
        location_data (list or str): Data for the chart. For locator maps, this should be a list of point dicts.
                                     For other maps, it can be CSV data string or path.
        map_type (str): Type of map (e.g., 'supervisor_district', 'point', 'address', 'intersection').
        reference_chart_id (str, optional): ID of a Datawrapper chart to clone for non-locator maps.
        center_coords (list, optional): [longitude, latitude] for locator map center.
        zoom_level (int/float, optional): Zoom level for locator map.
        series_field (str, optional): Field name to use for series grouping (e.g., 'series', 'category', 'type').
        color_palette (list or str, optional): Color palette for series. Can be list of colors or palette name.
        
    Returns:
        str: The ID of the created/updated chart, or None on failure.
    """
    if not DATAWRAPPER_API_KEY:
        logger.error("DATAWRAPPER_API_KEY not set. Cannot create chart.")
        return None
    
    chart_id = None

    try:
        # Handle location_data that might be passed as JSON string from LLM
        if isinstance(location_data, str) and map_type in ["supervisor_district", "police_district"]:
            try:
                # Try to parse as JSON first
                parsed_data = json.loads(location_data)
                if isinstance(parsed_data, list):
                    location_data = parsed_data
                    logger.info(f"Successfully parsed JSON string location_data into list with {len(location_data)} items")
                else:
                    logger.warning(f"Parsed location_data is not a list: {type(parsed_data)}")
            except json.JSONDecodeError:
                # If it's not JSON, treat as CSV data and pass through
                logger.info(f"location_data appears to be CSV string, using as-is")
        
        # Handle location_data format conversion for district maps
        if map_type in ["supervisor_district", "police_district"] and isinstance(location_data, list):
            # Convert list of dictionaries to CSV format
            logger.info(f"Converting list format to CSV for {map_type} map")
            
            # Check if this is enhanced change data (with current, previous, delta fields)
            sample_item = location_data[0] if location_data else {}
            has_change_data = any(key in sample_item for key in ['current_value', 'previous_value', 'delta', 'percent_change'])
            
            # Ensure map_metadata is a dictionary and add the has_change_data flag
            if isinstance(map_metadata, str):
                try:
                    map_metadata = json.loads(map_metadata)
                except json.JSONDecodeError:
                    try:
                        map_metadata = ast.literal_eval(map_metadata)
                    except (ValueError, SyntaxError):
                        map_metadata = {}
            elif map_metadata is None:
                map_metadata = {}
            else:
                # Make a copy to avoid modifying the original
                map_metadata = dict(map_metadata)
            
            # Add the has_change_data flag to metadata
            map_metadata["has_change_data"] = has_change_data
            
            if has_change_data:
                # Enhanced CSV format for change/delta maps
                # For delta maps, put the percentage value first as the main 'value' column
                if map_metadata and map_metadata.get("map_type") == "delta":
                    csv_data = "district,value,current_value,previous_value,delta,percent_change\n"
                else:
                    csv_data = "district,current_value,previous_value,delta,percent_change,value\n"
                
                for item in location_data:
                    if isinstance(item, dict) and "district" in item:
                        district = item.get('district', '')
                        current = item.get('current_value', item.get('value', 0))
                        previous = item.get('previous_value', 0)
                        delta = item.get('delta', current - previous)
                        percent_change = item.get('percent_change', 0)
                        
                        # The 'value' column is what Datawrapper uses for coloring
                        # For delta maps, this should be the percent_change (not delta)
                        # For regular maps, this should be the current_value
                        # For delta maps, multiply by 100 to get percentage values for proper legend display
                        if map_metadata and map_metadata.get("map_type") == "delta":
                            # Convert to percentage and clamp to [-100, 100] so extreme outliers use edge colours
                            value_for_coloring = max(min(percent_change * 100, 100), -100)
                            # For delta maps, reorganize CSV to put percentage value first
                            csv_data += f"{district},{value_for_coloring},{current},{previous},{delta},{percent_change}\n"
                        else:
                            value_for_coloring = current
                            csv_data += f"{district},{current},{previous},{delta},{percent_change},{value_for_coloring}\n"
                    else:
                        logger.warning(f"Skipping invalid district data item: {item}")
            else:
                # Standard CSV format for backward compatibility
                csv_data = "district,value\n"
                for item in location_data:
                    if isinstance(item, dict) and "district" in item and "value" in item:
                        csv_data += f"{item['district']},{item['value']}\n"
                    else:
                        logger.warning(f"Skipping invalid district data item: {item}")
            
            location_data = csv_data
            logger.info(f"Converted to CSV format: {location_data}")
        
        # Handle locator maps (point, address, intersection) with the new logic
        if map_type in ["point", "address", "intersection"]:
            logger.info(f"Preparing locator map: {chart_title} (type: {map_type})")
            if not isinstance(location_data, list):
                logger.error(f"Location data for {map_type} map must be a list of location details.")
                try:
                    # Attempt to parse if it's a JSON string representing a list
                    parsed_location_data = json.loads(location_data)
                    if not isinstance(parsed_location_data, list):
                        raise ValueError("Parsed location data is not a list.")
                    location_data = parsed_location_data
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse location_data as list for {map_type} map: {e}")
                    return None
            
            # For address maps, geocode addresses to get coordinates
            if map_type == "address":
                logger.info(f"Geocoding {len(location_data)} addresses for more reliable positioning")
                geocoded_data = []
                
                for item in location_data:
                    if isinstance(item, dict) and item.get("address"):
                        address = item["address"]
                        
                        # Geocode the address
                        lat, lon = geocode_address(address)
                        
                        if lat is not None and lon is not None:
                            # Convert to point format with coordinates
                            geocoded_item = {
                                "coordinates": [lon, lat],  # Datawrapper expects [longitude, latitude]
                                "title": item.get("title", address),
                                "description": item.get("description", ""),
                                "color": item.get("color", "#8b5cf6"),
                                "scale": item.get("scale", 0.8)
                            }
                            geocoded_data.append(geocoded_item)
                            logger.debug(f"Successfully geocoded '{address}' to ({lat}, {lon})")
                        else:
                            # Fallback: keep as address for Datawrapper to try geocoding
                            logger.warning(f"Failed to geocode '{address}', falling back to Datawrapper geocoding")
                            geocoded_data.append(item)
                    else:
                        # Keep non-address items as-is
                        geocoded_data.append(item)
                
                location_data = geocoded_data
                successful_geocodes = sum(1 for item in location_data if "coordinates" in item)
                logger.info(f"Geocoded {successful_geocodes}/{len(location_data)} addresses successfully")
            
            # Process DataSF location format for point maps
            elif map_type == "point":
                logger.info(f"Processing {len(location_data)} point locations")
                processed_data = []
                
                for item in location_data:
                    if isinstance(item, dict):
                        # Handle DataSF location format: {"type": "Point", "coordinates": [lon, lat]}
                        if "location" in item and isinstance(item["location"], dict):
                            location_obj = item["location"]
                            if location_obj.get("type") == "Point" and "coordinates" in location_obj:
                                coords = location_obj["coordinates"]
                                if isinstance(coords, list) and len(coords) >= 2:
                                    lon, lat = float(coords[0]), float(coords[1])
                                    
                                    # Validate coordinates are in San Francisco area
                                    if 37.6 <= lat <= 37.9 and -122.6 <= lon <= -122.2:
                                        processed_item = {
                                            "coordinates": [lon, lat],  # Already in [longitude, latitude] format
                                            "title": item.get("title", item.get("dba_name", "Unknown Business")),
                                            "tooltip": item.get("tooltip", f"Industry: {item.get('naic_code_description', 'Unknown')}"),
                                            "industry": item.get("industry", item.get("naic_code_description", "Unknown")),
                                            "scale": 0.6,  # Smaller markers
                                            "opacity": 0.5,  # 50% opacity
                                            "alpha": 0.5  # Alternative opacity property
                                        }
                                        
                                        # Copy over any series field data
                                        if series_field and series_field in item:
                                            processed_item[series_field] = item[series_field]
                                        elif series_field == "naic_code_description" and "naic_code_description" in item:
                                            processed_item[series_field] = item["naic_code_description"]
                                        
                                        processed_data.append(processed_item)
                                        logger.debug(f"Processed DataSF location for '{processed_item['title']}': [{lon}, {lat}]")
                                    else:
                                        logger.warning(f"Coordinates for '{item.get('title', item.get('dba_name', 'Unknown'))}' are outside SF area: [{lon}, {lat}]")
                                else:
                                    logger.warning(f"Invalid coordinates format in location: {coords}")
                            else:
                                logger.warning(f"Invalid location object format: {location_obj}")
                        
                        # Handle direct coordinates format (already processed data)
                        elif "coordinates" in item or ("lat" in item and "lon" in item):
                            # Validate coordinates if they exist
                            coords_to_check = None
                            if "coordinates" in item:
                                coords_to_check = item["coordinates"]
                            elif "lat" in item and "lon" in item:
                                coords_to_check = [item["lon"], item["lat"]]
                            
                            if coords_to_check and len(coords_to_check) >= 2:
                                lon, lat = float(coords_to_check[0]), float(coords_to_check[1])
                                if 37.6 <= lat <= 37.9 and -122.6 <= lon <= -122.2:
                                    processed_data.append(item)
                                    logger.debug(f"Validated existing coordinates for '{item.get('title', 'Unknown')}': [{lon}, {lat}]")
                                else:
                                    logger.warning(f"Existing coordinates for '{item.get('title', 'Unknown')}' are outside SF area: [{lon}, {lat}]")
                            else:
                                processed_data.append(item)  # Let it through if we can't validate
                        
                        else:
                            logger.warning(f"No valid location data found in item: {item}")
                    else:
                        logger.warning(f"Invalid item format: {item}")
                
                location_data = processed_data
                logger.info(f"Successfully processed {len(location_data)} valid locations")
            
            # Prepare series information for legend
            series_info = None
            if series_field:
                # Extract unique series values and create color mapping
                unique_series = set()
                for item in location_data:
                    series_value = item.get(series_field)
                    if series_value is not None:
                        unique_series.add(str(series_value))
                
                if unique_series:
                    # Use default categorical palette if none specified
                    if color_palette is None:
                        color_palette = 'categorical'
                    
                    series_info = {
                        'series_field': series_field,
                        'legend_title': series_field.replace('_', ' ').title(),
                        'color_mapping': {}
                    }
                    
                    logger.info(f"Preparing series info for {len(unique_series)} series: {sorted(unique_series)}")

            markers_json_string = _prepare_locator_marker_data(location_data, series_field=series_field, color_palette=color_palette)
            if not markers_json_string or json.loads(markers_json_string).get("markers") == []:
                logger.error(f"No valid marker data prepared for locator map: {chart_title}")
                # For now, let's return None if no valid markers
                return None 
            
            # Extract color mapping from the prepared markers for legend
            if series_info:
                markers_data = json.loads(markers_json_string)
                color_mapping = {}
                for marker in markers_data.get("markers", []):
                    if "series" in marker:
                        color_mapping[marker["series"]] = marker["markerColor"]
                series_info['color_mapping'] = color_mapping
                logger.info(f"Extracted color mapping for legend: {color_mapping}")
            
            chart_id = _create_and_configure_locator_map(
                chart_title,
                markers_json_string,
                center_coords=center_coords, 
                zoom_level=zoom_level,
                series_info=series_info
            )
            if chart_id:
                logger.info(f"Successfully created and configured locator map '{chart_title}' with ID: {chart_id}")
                
                # Add a small delay to ensure all changes are processed
                time.sleep(1)
                
                # Force a refresh of the chart to ensure the published version will be correct
                try:
                    refresh_response = _make_dw_request("GET", f"/charts/{chart_id}")
                    if refresh_response:
                        logger.info(f"Chart {chart_id} refreshed successfully")
                except Exception as e:
                    logger.warning(f"Failed to refresh chart {chart_id}: {e}")
                    
            else:
                logger.error(f"Failed to create and configure locator map '{chart_title}'.")
            return chart_id # Return chart_id (or None if creation failed)

        # Existing logic for other map types (cloning, CSV upload)
        ref_chart_id = reference_chart_id or DEFAULT_DISTRICT_CHART
        if not ref_chart_id:
            logger.error("No reference chart ID provided for non-locator map type.")
            return None

        logger.info(f"Cloning chart from reference ID: {ref_chart_id} for new chart: {chart_title}")
        # Step 1: Copy the reference chart
        copy_response = _make_dw_request(
            "POST",
            f"/charts/{ref_chart_id}/copy"
        )
        if not copy_response or "id" not in copy_response:
            logger.error(f"Failed to copy reference chart {ref_chart_id}. Response: {copy_response}")
            return None
        chart_id = copy_response["id"]
        logger.info(f"Successfully copied chart. New chart ID: {chart_id}")

        # Step 2: Update the title of the new chart
        _make_dw_request(
            "PATCH",
            f"/charts/{chart_id}",
            json_payload={"title": chart_title}
        )
        logger.info(f"Updated title for chart ID {chart_id} to: {chart_title}")

        # Step 3: Upload new data to the chart
        # Check if location_data is a path or raw CSV string
        data_to_upload = ""
        if isinstance(location_data, str) and os.path.exists(location_data):
            with open(location_data, 'r') as f:
                data_to_upload = f.read()
            logger.info(f"Loaded data from file: {location_data}")
        elif isinstance(location_data, str):
            data_to_upload = location_data # Assume it's a CSV string
            logger.info("Using provided string as data for chart.")
        else:
            logger.error("location_data for non-locator maps must be a file path or CSV string.")
            # Attempt to delete the partially created chart
            if chart_id:
                logger.info(f"Attempting to delete partially created chart: {chart_id}")
                try:
                    _make_dw_request("DELETE", f"/charts/{chart_id}")
                    logger.info(f"Successfully deleted chart: {chart_id}")
                except Exception as del_e:
                    logger.error(f"Failed to delete chart {chart_id}: {del_e}")
            return None
        
        if not data_to_upload.strip():
            logger.warning(f"Data for chart {chart_id} is empty. Proceeding without data upload.")
        else:
            _make_dw_request(
                "PUT",
                f"/charts/{chart_id}/data",
                headers={"Content-Type": "text/csv"},
                data=data_to_upload
            )
            logger.info(f"Uploaded data to chart ID {chart_id}")

        return chart_id
    
    except Exception as e:
        logger.error(f"Error in create_datawrapper_chart for \'{chart_title}\': {str(e)}", exc_info=True)
        # If a chart was partially created before an error, attempt to delete it for non-locator maps
        if chart_id and map_type not in ["point", "address", "intersection"]:
            logger.warning(f"An error occurred after chart {chart_id} was cloned. Attempting to delete it.")
            try:
                _make_dw_request("DELETE", f"/charts/{chart_id}")
                logger.info(f"Successfully deleted partially created chart: {chart_id}")
            except Exception as del_e:
                logger.error(f"Failed to delete chart {chart_id} during cleanup: {del_e}")
        elif chart_id and map_type in ["point", "address", "intersection"]:
            logger.warning(f"An error occurred with locator map {chart_id}. It might be in an incomplete state.")
        return None

def _apply_custom_map_styling(chart_id, map_type, map_metadata=None):
    """
    Apply custom styling to a Datawrapper map based on the map type and metadata.
    
    Args:
        chart_id (str): The Datawrapper chart ID
        map_type (str): The type of map (e.g., 'supervisor_district')
        map_metadata (dict or str, optional): Metadata about the map, can be JSON string or dict
        
    Returns:
        bool: True if styling was applied successfully, False otherwise
    """
    try:
        logger.info(f"Applying custom styling to chart {chart_id}, type: {map_type}")
        
        # Ensure map_metadata is a dictionary
        if isinstance(map_metadata, str):
            try:
                map_metadata = json.loads(map_metadata)
            except json.JSONDecodeError:
                # Try parsing as Python dict string representation (e.g., "{'key': 'value'}")
                try:
                    map_metadata = ast.literal_eval(map_metadata)
                    logger.info(f"Successfully parsed map_metadata as Python dict for storage: {map_metadata}")
                except (ValueError, SyntaxError):
                    logger.warning(f"Invalid JSON and Python dict in map_metadata, treating as empty dict: {map_metadata}")
                    map_metadata = {}
        elif map_metadata is None:
            map_metadata = {}
        
        # Define color schemes for different map types
        density_color_scheme = [
            {"color": "#f0f9ff", "position": 0},     # Very light blue
            {"color": "#7dd3fc", "position": 0.25},  # Light blue
            {"color": "#0ea5e9", "position": 0.5},   # Medium blue
            {"color": "#0284c7", "position": 0.75},  # Dark blue
            {"color": "#0c4a6e", "position": 1}      # Very dark blue
        ]
        
        # Symmetric diverging palette with matched lightness on both sides
        # Endpoints chosen to have similar L* in Lab space so red & green feel balanced
        delta_color_scheme = [
            {"color": "#29c786", "position": 0},      # Green (≈L54)
            {"color": "#a3ebd1", "position": 0.25},   # Light green
            {"color": "#eeeeee", "position": 0.5},    # Neutral grey
            {"color": "#f7a3a3", "position": 0.75},   # Light red
            {"color": "#e24d4d", "position": 1.0}     # Red (≈L54)
        ]
        
        # Determine which color scheme to use
        if map_metadata.get("map_type") == "delta":
            colors = delta_color_scheme
            mode = "continuous"
            interpolation = "equidistant"
            legend_title = "% Change"
            legend_max = "+100%"
            legend_min = "-100%" 
            legend_center = "0%"
            legend_format = "0,0.[0]%"
            legend_labels = "custom"  # Use custom labels for fixed percentage range
            # Set explicit color scale range for percentage values (-100 to +100 = -100% to +100%)
            colorscale_min = -100.0
            colorscale_max = 100.0
            # Custom labels for the legend to show the full percentage range
            custom_labels = ["-100%", "-50%", "0%", "+50%", "+100%"]
            # Enhanced tooltip for delta maps - using correct Datawrapper format with double curly brackets
            tooltip_body = "Current: {{ current_value }}<br/>Previous: {{ previous_value }}<br/>Change: {{ delta }}<br/>% Change: {{ FORMAT(percent_change * 100, '0,0.[0]') }}%"
            tooltip_title = "District %REGION_NAME%"
        else:
            colors = density_color_scheme
            mode = "continuous"
            interpolation = "equidistant"
            legend_title = "VALUE"
            legend_max = "high"
            legend_min = "low"
            legend_center = "medium"
            legend_format = "0,0.[0]"  # Fixed: Remove % for actual values
            legend_labels = "ranges"  # Use auto-calculated ranges for density maps
            custom_labels = []
            # Use automatic scaling for density maps
            colorscale_min = None
            colorscale_max = None
            # Check if we have enhanced data available
            if map_metadata.get("has_change_data", False):
                tooltip_body = "Current: {{ current_value }}<br/>Previous: {{ previous_value }}<br/>Change: {{ delta }}"
                tooltip_title = "District %REGION_NAME%"
            else:
                # Standard tooltip for simple maps
                tooltip_body = "%REGION_VALUE%"
                tooltip_title = "District %REGION_NAME%"
        
        # Prepare colorscale configuration
        colorscale_config = {
            "colors": colors,
            "mode": mode,
            "interpolation": interpolation,
            "stops": "equidistant",
            "stopCount": 5,
            "palette": 0
        }
        
        # Add explicit min/max for delta maps to ensure proper percentage scaling
        if colorscale_min is not None and colorscale_max is not None:
            colorscale_config["min"] = colorscale_min
            colorscale_config["max"] = colorscale_max
            colorscale_config["stops"] = "custom"  # Use custom stops when min/max are specified
            # Define explicit custom stops for the -100% to +100% range
            colorscale_config["customStops"] = [
                {"position": 0.0, "color": "#29c786"},   # -100%
                {"position": 0.25, "color": "#a3ebd1"},  # -50%
                {"position": 0.5, "color": "#eeeeee"},   # 0%
                {"position": 0.75, "color": "#f7a3a3"},  # +50%
                {"position": 1.0, "color": "#e24d4d"}    # +100%
            ]
            # Explicitly fix min/max center so Datawrapper doesn't auto-rescale based on data range
            colorscale_config["rangeMin"] = -100
            colorscale_config["rangeMax"] = 100
            colorscale_config["rangeCenter"] = 0
        
        # Prepare legend configuration
        legend_config = {
            "size": 170,
            "title": legend_title,
            "labels": legend_labels,
            "enabled": True,
            "offsetX": 0,
            "offsetY": 0,
            "reverse": False,
            "labelMax": legend_max,
            "labelMin": legend_min,
            "position": "above",
            "hideItems": [],
            "interactive": True,
            "labelCenter": legend_center,
            "labelFormat": legend_format,
            "orientation": "horizontal",
            "titleEnabled": True
        }
        
        # Add custom labels for delta maps
        if custom_labels:
            legend_config["customLabels"] = custom_labels
        else:
            legend_config["customLabels"] = []
        
        # Prepare styling payload
        styling_payload = {
            "metadata": {
                "visualize": {
                    "colorscale": colorscale_config,
                    "tooltip": {
                        "body": tooltip_body,
                        "title": tooltip_title,
                        "sticky": True,
                        "enabled": True
                    },
                    "legends": {
                        "color": legend_config
                    }
                },
                "describe": {
                    "intro": map_metadata.get("description", ""),
                    "source-name": "DataSF",
                    "source-url": map_metadata.get("executed_url", ""),
                    "byline": "TransparentSF"
                }
            }
        }
        
        # Apply the styling
        _make_dw_request(
            "PATCH",
            f"/charts/{chart_id}",
            json_payload=styling_payload
        )
        
        logger.info(f"Applied custom styling to chart {chart_id} (type: {map_type})")
        return True
        
    except Exception as e:
        logger.error(f"Error applying custom styling to chart {chart_id}: {str(e)}", exc_info=True)
        return False

def get_db_connection():
    """Get a connection to the PostgreSQL database."""
    try:
        # Get database connection parameters from environment variables
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "transparentsf")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        logger.info(f"Connecting to database: {db_host}:{db_port}/{db_name}")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def generate_map(context_variables, map_title, map_type, location_data, map_metadata=None, reference_chart_id=None, metric_id=None, group_field=None, series_field=None, color_palette=None):
    """
    Generates a map using Datawrapper, stores its metadata, and returns relevant URLs.
    
    Args:
        context_variables (dict): Dictionary of context variables (e.g., for database connection).
        map_title (str): Desired title for the map.
        map_type (str): Type of map (e.g., 'supervisor_district', 'police_district', 'point', 'address', 'intersection').
                        'point', 'address', and 'intersection' will use the new locator map creation flow.
        location_data (list or str): Data for the map. 
                                   For locator maps ('point', 'address', 'intersection'): list of dicts with point/address details.
                                   For district maps: list of dicts [{"district": "1", "value": 120}] OR CSV string.
                                   For other maps: CSV string or path to CSV file.
        map_metadata (dict or str, optional): Additional metadata for the map (e.g., center, zoom for locator).
        reference_chart_id (str, optional): Datawrapper chart ID to use as a template for non-locator maps.
        metric_id (str, optional): Identifier for the metric being mapped (for storage).
        group_field (str, optional): Field used for grouping/aggregation (for storage).
        series_field (str, optional): Field name to use for series grouping (e.g., 'series', 'category', 'type').
        color_palette (list or str, optional): Color palette for series. Can be list of colors or palette name.
        
    Returns:
        dict: Contains map_id, edit_url, and publish_url, or None on failure.
    """
    logger.info(f"Generating map: '{map_title}' of type '{map_type}'")
    
    # Handle location_data that might be passed as JSON string from LLM
    if isinstance(location_data, str) and map_type in ["supervisor_district", "police_district"]:
        try:
            # Try to parse as JSON first
            parsed_data = json.loads(location_data)
            if isinstance(parsed_data, list):
                location_data = parsed_data
                logger.info(f"Successfully parsed JSON string location_data into list with {len(location_data)} items")
            else:
                logger.warning(f"Parsed location_data is not a list: {type(parsed_data)}")
        except json.JSONDecodeError:
            # If it's not JSON, treat as CSV data and pass through
            logger.info(f"location_data appears to be CSV string, using as-is")
    
    # Handle location_data format conversion for district maps
    if map_type in ["supervisor_district", "police_district"] and isinstance(location_data, list):
        # Convert list of dictionaries to CSV format
        logger.info(f"Converting list format to CSV for {map_type} map")
        
        # Check if this is enhanced change data (with current, previous, delta fields)
        sample_item = location_data[0] if location_data else {}
        has_change_data = any(key in sample_item for key in ['current_value', 'previous_value', 'delta', 'percent_change'])
        
        # Ensure map_metadata is a dictionary and add the has_change_data flag
        if isinstance(map_metadata, str):
            try:
                map_metadata = json.loads(map_metadata)
            except json.JSONDecodeError:
                try:
                    map_metadata = ast.literal_eval(map_metadata)
                except (ValueError, SyntaxError):
                    map_metadata = {}
        elif map_metadata is None:
            map_metadata = {}
        else:
            # Make a copy to avoid modifying the original
            map_metadata = dict(map_metadata)
            
        # Add the has_change_data flag to metadata
        map_metadata["has_change_data"] = has_change_data
        
        if has_change_data:
            # Enhanced CSV format for change/delta maps
            # For delta maps, put the percentage value first as the main 'value' column
            if map_metadata and map_metadata.get("map_type") == "delta":
                csv_data = "district,value,current_value,previous_value,delta,percent_change\n"
            else:
                csv_data = "district,current_value,previous_value,delta,percent_change,value\n"
            
            for item in location_data:
                if isinstance(item, dict) and "district" in item:
                    district = item.get('district', '')
                    current = item.get('current_value', item.get('value', 0))
                    previous = item.get('previous_value', 0)
                    delta = item.get('delta', current - previous)
                    percent_change = item.get('percent_change', 0)
                    
                    # The 'value' column is what Datawrapper uses for coloring
                    # For delta maps, this should be the percent_change (not delta)
                    # For regular maps, this should be the current_value
                    # For delta maps, multiply by 100 to get percentage values for proper legend display
                    if map_metadata and map_metadata.get("map_type") == "delta":
                        # Convert to percentage and clamp to [-100, 100] so extreme outliers use edge colours
                        value_for_coloring = max(min(percent_change * 100, 100), -100)
                        # For delta maps, reorganize CSV to put percentage value first
                        csv_data += f"{district},{value_for_coloring},{current},{previous},{delta},{percent_change}\n"
                    else:
                        value_for_coloring = current
                        csv_data += f"{district},{current},{previous},{delta},{percent_change},{value_for_coloring}\n"
                else:
                    logger.warning(f"Skipping invalid district data item: {item}")
        else:
            # Standard CSV format for backward compatibility
            csv_data = "district,value\n"
            for item in location_data:
                if isinstance(item, dict) and "district" in item and "value" in item:
                    csv_data += f"{item['district']},{item['value']}\n"
                else:
                    logger.warning(f"Skipping invalid district data item: {item}")
            
        location_data = csv_data
        logger.info(f"Converted to CSV format: {location_data}")
    
    # Prepare chart creation arguments
    chart_args = {
        "chart_title": map_title,
        "location_data": location_data,
        "map_type": map_type
    }

    # For locator maps, pass center and zoom if provided in map_metadata
    center_coords = None
    zoom_level = None
    if map_metadata and isinstance(map_metadata, dict):
        if map_type in ["point", "address", "intersection"]:
            center_coords = map_metadata.get("center")
            zoom_level = map_metadata.get("zoom")
            chart_args["center_coords"] = center_coords
            chart_args["zoom_level"] = zoom_level
        # For other map types, map_metadata might be used differently or passed as 'external_metadata'
        # For now, let's assume other map types don't use center/zoom from here directly for creation
    
    # Add series support for locator maps
    if map_type in ["point", "address", "intersection"]:
        if series_field:
            chart_args["series_field"] = series_field
        if color_palette:
            chart_args["color_palette"] = color_palette
    
    # For non-locator maps, pass reference_chart_id if provided
    if map_type not in ["point", "address", "intersection"] and reference_chart_id:
        chart_args["reference_chart_id"] = reference_chart_id

    chart_id = None
    try:
        chart_id = create_datawrapper_chart(**chart_args)

        if not chart_id:
            logger.error(f"Failed to create Datawrapper chart for '{map_title}'.")
            return None

        logger.info(f"Datawrapper chart created/updated with ID: {chart_id} for title '{map_title}'")

        # Apply custom styling before publishing
        logger.info(f"Applying custom styling to chart {chart_id}")
        
        # Get the executed query URL from context variables if available
        executed_url = context_variables.get('executed_query_url')
        if executed_url:
            if map_metadata is None:
                map_metadata = {}
            elif isinstance(map_metadata, str):
                try:
                    map_metadata = json.loads(map_metadata)
                except json.JSONDecodeError:
                    map_metadata = {}
            map_metadata['executed_url'] = executed_url
            
        _apply_custom_map_styling(chart_id, map_type, map_metadata)

        # Publish the chart once with all styling applied
        logger.info(f"Publishing chart {chart_id}")
        publish_response = _make_dw_request(
            "POST",
            f"/charts/{chart_id}/publish"
        )
        if not publish_response:
            logger.warning(f"Publish command for chart {chart_id} did not return a substantial response. Assuming success if no error.")
        
        logger.info(f"Chart {chart_id} published successfully.")

        edit_url = f"https://app.datawrapper.de/edit/{chart_id}"
        # Construct public URL (format may vary based on your Datawrapper setup/account)
        # Example: chart_data = _make_dw_request("GET", f"/charts/{chart_id}")
        # public_url = chart_data.get("publicUrl", f"https://datawrapper.dwcdn.net/{chart_id}/")
        # For simplicity, using the common DW CDN structure:
        public_url = f"https://datawrapper.dwcdn.net/{chart_id}/"

        # Store map metadata in the database
        map_id = None  # Initialize map_id
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create maps table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maps (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    chart_id TEXT UNIQUE NOT NULL,
                    edit_url TEXT NOT NULL,
                    published_url TEXT NOT NULL,
                    location_data JSONB,
                    metadata JSONB,
                    metric_id TEXT,
                    group_field TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Prepare location data for storage
            if isinstance(location_data, str):
                # For CSV data, store as text in a wrapper object
                location_data_json = {"csv_data": location_data, "type": "csv"}
            elif isinstance(location_data, list):
                # For list data, store directly
                location_data_json = {"data": location_data, "type": "list"}
            else:
                # For other types, convert to string
                location_data_json = {"data": str(location_data), "type": "other"}
            
            # Prepare metadata for storage
            # Ensure map_metadata is a dictionary
            if isinstance(map_metadata, str):
                try:
                    metadata_json = json.loads(map_metadata)
                except json.JSONDecodeError:
                    # Try parsing as Python dict string representation (e.g., "{'key': 'value'}")
                    try:
                        metadata_json = ast.literal_eval(map_metadata)
                        logger.info(f"Successfully parsed map_metadata as Python dict for storage: {metadata_json}")
                    except (ValueError, SyntaxError):
                        logger.warning(f"Invalid JSON and Python dict in map_metadata, treating as empty dict: {map_metadata}")
                        metadata_json = {}
            elif map_metadata is None:
                metadata_json = {}
            else:
                # If it's already a dict, make a copy to avoid modifying the original
                metadata_json = dict(map_metadata)
                
            # Add the executed query URL to metadata if available
            if executed_url:
                metadata_json['executed_query_url'] = executed_url
                
            metadata_json.update({
                "creation_timestamp": datetime.now().isoformat(),
                "datawrapper_chart_id": chart_id
            })
            
            # Determine map subtype for management purposes
            map_subtype = "density"  # Default to density
            if metadata_json.get("map_type") == "delta":
                map_subtype = "delta"
            
            # Before inserting, deactivate previous maps of the same type for this metric/group_field
            if metric_id and group_field:
                try:
                    # Deactivate previous maps of the same subtype (density or delta) for this metric/group_field
                    cursor.execute("""
                        UPDATE maps 
                        SET active = FALSE, updated_at = CURRENT_TIMESTAMP
                        WHERE metric_id = %s 
                        AND group_field = %s 
                        AND type = %s
                        AND (metadata->>'map_type' = %s OR (metadata->>'map_type' IS NULL AND %s = 'density'))
                        AND active = TRUE
                    """, (metric_id, group_field, map_type, map_subtype, map_subtype))
                    
                    deactivated_count = cursor.rowcount
                    if deactivated_count > 0:
                        logger.info(f"Deactivated {deactivated_count} previous {map_subtype} map(s) for metric_id={metric_id}, group_field={group_field}")
                except Exception as e:
                    logger.warning(f"Failed to deactivate previous maps: {str(e)}")
                    # Continue with insertion even if deactivation fails
            
            # Insert map record
            cursor.execute("""
                INSERT INTO maps (title, type, chart_id, edit_url, published_url, location_data, metadata, metric_id, group_field)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                map_title,
                map_type,
                chart_id,
                edit_url,
                public_url,
                json.dumps(location_data_json),
                json.dumps(metadata_json),
                metric_id,
                group_field
            ))
            
            map_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Stored map metadata in database with ID: {map_id}")
            
        except Exception as e:
            logger.error(f"Failed to store map metadata in database: {str(e)}", exc_info=True)
            # Don't fail the whole operation if database storage fails
            
        logger.info(f"Map generation complete for '{map_title}'. Chart ID: {chart_id}")
        logger.info(f"Edit URL: {edit_url}")
        logger.info(f"Public URL: {public_url}")

        return {
            "map_id": map_id,
            "edit_url": edit_url,
            "publish_url": public_url
        }
    
    except Exception as e:
        logger.error(f"An error occurred during map generation for '{map_title}': {str(e)}", exc_info=True)
        # If chart_id was created but subsequent steps failed, it might exist in Datawrapper
        if chart_id:
            logger.error(f"Chart {chart_id} may have been partially created in Datawrapper.")
        return None

def get_map_by_id(context_variables, map_id):
    """
    Retrieve a map from the database by its ID.
    
    Args:
        context_variables: Context variables from the chatbot
        map_id: The integer ID of the map to retrieve
        
    Returns:
        Dictionary containing the map data if found
    """
    logger.info(f"Retrieving map with ID: {map_id}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the map
        cursor.execute("SELECT * FROM maps WHERE id = %s", (map_id,))
        map_record = cursor.fetchone()
        
        # Close the connection
        cursor.close()
        conn.close()
        
        if map_record:
            # Convert to dictionary and process JSON fields
            map_data = dict(map_record)
            
            # Handle location_data - could be string or already parsed JSON
            if isinstance(map_data["location_data"], str):
                try:
                    map_data["location_data"] = json.loads(map_data["location_data"])
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in location_data: {map_data['location_data']}")
                    map_data["location_data"] = []
            
            # Handle metadata if it exists
            if map_data["metadata"]:
                if isinstance(map_data["metadata"], str):
                    try:
                        map_data["metadata"] = json.loads(map_data["metadata"])
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in metadata: {map_data['metadata']}")
                        map_data["metadata"] = {}
            
            logger.info(f"Map found: {map_data['title']}")
            return {
                "status": "success",
                "map": map_data
            }
        else:
            logger.warning(f"Map with ID {map_id} not found")
            return {
                "status": "error",
                "message": f"Map with ID {map_id} not found"
            }
    
    except Exception as e:
        logger.error(f"Error retrieving map: {str(e)}")
        return {"status": "error", "message": f"Error retrieving map: {str(e)}"}

def get_recent_maps(context_variables, limit=10, map_type=None):
    """
    Retrieve the most recent maps from the database.
    
    Args:
        context_variables: Context variables from the chatbot
        limit: Maximum number of maps to return (default: 10)
        map_type: Filter by map type (optional)
        
    Returns:
        Dictionary containing a list of maps
    """
    logger.info(f"Retrieving recent maps, limit: {limit}, type: {map_type}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Prepare the query
        if map_type:
            query = "SELECT * FROM maps WHERE type = %s ORDER BY created_at DESC LIMIT %s"
            cursor.execute(query, (map_type, limit))
        else:
            query = "SELECT * FROM maps ORDER BY created_at DESC LIMIT %s"
            cursor.execute(query, (limit,))
        
        maps = cursor.fetchall()
        
        # Close the connection
        cursor.close()
        conn.close()
        
        # Convert to list of dictionaries and process JSON fields
        map_list = []
        for map_record in maps:
            map_data = dict(map_record)
            
            # Handle location_data - could be string or already parsed JSON
            if isinstance(map_data["location_data"], str):
                try:
                    map_data["location_data"] = json.loads(map_data["location_data"])
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in location_data: {map_data['location_data']}")
                    map_data["location_data"] = []
            
            # Handle metadata if it exists
            if map_data["metadata"]:
                if isinstance(map_data["metadata"], str):
                    try:
                        map_data["metadata"] = json.loads(map_data["metadata"])
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in metadata: {map_data['metadata']}")
                        map_data["metadata"] = {}
            
            map_list.append(map_data)
        
        logger.info(f"Retrieved {len(map_list)} maps")
        return {
            "status": "success",
            "maps": map_list
        }
    
    except Exception as e:
        logger.error(f"Error retrieving maps: {str(e)}")
        return {"status": "error", "message": f"Error retrieving maps: {str(e)}"}

def get_active_maps_for_metric(context_variables, metric_id, group_field, map_subtype=None):
    """
    Retrieve active maps for a specific metric and group_field.
    
    Args:
        context_variables: Context variables from the chatbot
        metric_id: The metric ID to filter by
        group_field: The group field to filter by (e.g., 'supervisor_district')
        map_subtype: Optional filter by map subtype ('density' or 'delta')
        
    Returns:
        Dictionary containing a list of active maps
    """
    logger.info(f"Retrieving active maps for metric_id={metric_id}, group_field={group_field}, subtype={map_subtype}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Base query for active maps
        base_query = """
            SELECT * FROM maps 
            WHERE metric_id = %s 
            AND group_field = %s 
            AND active = TRUE
        """
        params = [metric_id, group_field]
        
        # Add subtype filter if specified
        if map_subtype:
            if map_subtype == "density":
                base_query += " AND (metadata->>'map_type' IS NULL OR metadata->>'map_type' != 'delta')"
            elif map_subtype == "delta":
                base_query += " AND metadata->>'map_type' = 'delta'"
        
        base_query += " ORDER BY created_at DESC"
        
        cursor.execute(base_query, params)
        maps = cursor.fetchall()
        
        # Close the connection
        cursor.close()
        conn.close()
        
        # Convert to list of dictionaries and process JSON fields
        map_list = []
        for map_record in maps:
            map_data = dict(map_record)
            
            # Handle location_data - could be string or already parsed JSON
            if isinstance(map_data["location_data"], str):
                try:
                    map_data["location_data"] = json.loads(map_data["location_data"])
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in location_data: {map_data['location_data']}")
                    map_data["location_data"] = []
            
            # Handle metadata if it exists
            if map_data["metadata"]:
                if isinstance(map_data["metadata"], str):
                    try:
                        map_data["metadata"] = json.loads(map_data["metadata"])
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in metadata: {map_data['metadata']}")
                        map_data["metadata"] = {}
            
            map_list.append(map_data)
        
        logger.info(f"Retrieved {len(map_list)} active maps")
        return {
            "status": "success",
            "maps": map_list
        }
    
    except Exception as e:
        logger.error(f"Error retrieving active maps: {str(e)}")
        return {"status": "error", "message": f"Error retrieving active maps: {str(e)}"}

def geocode_address(address, max_retries=3, delay=1):
    """
    Geocode an address to get latitude and longitude coordinates.
    Uses OpenStreetMap's Nominatim service (free, no API key required).
    
    Args:
        address (str): The address to geocode
        max_retries (int): Maximum number of retry attempts
        delay (float): Delay between requests in seconds
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if geocoding fails
    """
    if not address or not isinstance(address, str):
        logger.warning(f"Invalid address for geocoding: {address}")
        return None, None
    
    # Clean and prepare the address
    address = address.strip()
    if not address:
        return None, None
    
    # Ensure San Francisco is in the address for better results
    if "san francisco" not in address.lower() and "sf" not in address.lower():
        address = f"{address}, San Francisco, CA, USA"
    elif "usa" not in address.lower():
        address = f"{address}, USA"
    
    # URL encode the address
    encoded_address = quote(address)
    
    # Nominatim API endpoint (free OpenStreetMap geocoding service)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1&countrycodes=us"
    
    headers = {
        'User-Agent': 'TransparentSF/1.0 (https://transparentsf.org)'  # Required by Nominatim
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Geocoding attempt {attempt + 1} for address: {address}")
            
            # Add delay to respect rate limits (Nominatim allows 1 request per second)
            if attempt > 0:
                time.sleep(delay * attempt)
            else:
                time.sleep(delay)  # Always add a small delay
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Sanity check: ensure coordinates are roughly in San Francisco area
                # SF bounds: roughly 37.7-37.8 lat, -122.5 to -122.3 lon
                if 37.6 <= lat <= 37.9 and -122.6 <= lon <= -122.2:
                    logger.debug(f"Successfully geocoded '{address}' to ({lat}, {lon})")
                    return lat, lon
                else:
                    logger.warning(f"Geocoded coordinates for '{address}' ({lat}, {lon}) are outside SF area")
                    return None, None
            else:
                logger.warning(f"No geocoding results found for address: {address}")
                return None, None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Geocoding request failed for '{address}' (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to geocode address '{address}' after {max_retries} attempts")
                return None, None
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing geocoding response for '{address}': {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error geocoding '{address}': {e}")
            return None, None
    
    return None, None

def batch_geocode_addresses(addresses, batch_delay=1):
    """
    Geocode multiple addresses with rate limiting.
    
    Args:
        addresses (list): List of address strings
        batch_delay (float): Delay between each geocoding request
        
    Returns:
        dict: Dictionary mapping addresses to (lat, lon) tuples
    """
    results = {}
    
    logger.info(f"Starting batch geocoding of {len(addresses)} addresses")
    
    for i, address in enumerate(addresses):
        logger.info(f"Geocoding address {i+1}/{len(addresses)}: {address}")
        lat, lon = geocode_address(address, delay=batch_delay)
        results[address] = (lat, lon)
        
        # Progress logging
        if (i + 1) % 5 == 0:
            successful = sum(1 for lat, lon in results.values() if lat is not None)
            logger.info(f"Geocoded {i+1}/{len(addresses)} addresses, {successful} successful")
    
    successful = sum(1 for lat, lon in results.values() if lat is not None)
    logger.info(f"Batch geocoding complete: {successful}/{len(addresses)} addresses successfully geocoded")
    
    return results

def create_sample_series_data():
    """
    Create sample data with different series for testing the enhanced map functionality.
    
    Returns:
        list: Sample location data with series information
    """
    sample_data = [
        {
            "title": "Police Station - Central",
            "lat": 37.7749,
            "lon": -122.4194,
            "tooltip": "Central Police Station",
            "series": "Police Stations"
        },
        {
            "title": "Police Station - Mission",
            "lat": 37.7630,
            "lon": -122.4250,
            "tooltip": "Mission Police Station",
            "series": "Police Stations"
        },
        {
            "title": "Fire Station - Engine 1",
            "lat": 37.7849,
            "lon": -122.4094,
            "tooltip": "Fire Station Engine 1",
            "series": "Fire Stations"
        },
        {
            "title": "Fire Station - Engine 2",
            "lat": 37.7580,
            "lon": -122.4350,
            "tooltip": "Fire Station Engine 2",
            "series": "Fire Stations"
        },
        {
            "title": "Hospital - UCSF",
            "lat": 37.7627,
            "lon": -122.4581,
            "tooltip": "UCSF Medical Center",
            "series": "Hospitals"
        },
        {
            "title": "Hospital - CPMC",
            "lat": 37.7886,
            "lon": -122.4324,
            "tooltip": "California Pacific Medical Center",
            "series": "Hospitals"
        },
        {
            "title": "School - Lowell High",
            "lat": 37.7197,
            "lon": -122.4831,
            "tooltip": "Lowell High School",
            "series": "Schools"
        },
        {
            "title": "School - Washington High",
            "lat": 37.7749,
            "lon": -122.4194,
            "tooltip": "Washington High School",
            "series": "Schools"
        }
    ]
    return sample_data

# Testing
if __name__ == "__main__":
    # Initialize logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for a supervisor district map
    test_location_data_csv = """district,value
1,120
2,85
3,65
4,95
5,75
6,50
7,80
8,70
9,60
10,90
11,55"""
    
    test_metadata = {
        "description": "Test map showing values by supervisor district",
        "source": "Test data",
        "date_created": datetime.now().isoformat()
    }
    
    # Test generating a map
    result = generate_map(
        {},
        map_title="Test Supervisor District Map",
        map_type="supervisor_district",
        location_data=test_location_data_csv,
        map_metadata=test_metadata
    )
    
    print(f"Generate map result: {result}")
    
    if result and "map_id" in result:
        print("✅ Supervisor district map created successfully!")
        print(f"Map ID: {result['map_id']}")
        print(f"Edit URL: {result['edit_url']}")
        print(f"Public URL: {result['publish_url']}")
    else:
        print("❌ Failed to create supervisor district map")
        
    # Test enhanced change map with detailed tooltips
    enhanced_change_data = [
        {"district": "1", "current_value": 120, "previous_value": 100, "delta": 20, "percent_change": 0.20},
        {"district": "2", "current_value": 85, "previous_value": 90, "delta": -5, "percent_change": -0.056},
        {"district": "3", "current_value": 65, "previous_value": 70, "delta": -5, "percent_change": -0.071},
        {"district": "4", "current_value": 95, "previous_value": 80, "delta": 15, "percent_change": 0.188},
        {"district": "5", "current_value": 75, "previous_value": 75, "delta": 0, "percent_change": 0.0}
    ]
    
    enhanced_metadata = {
        "map_type": "delta",
        "description": "Change in incidents from previous month",
        "source": "Test enhanced data"
    }
    
    result = generate_map(
        {},
        map_title="Test Enhanced Change Map - District Crime Changes",
        map_type="supervisor_district",
        location_data=enhanced_change_data,
        map_metadata=enhanced_metadata
    )
    
    print(f"Generate enhanced change map result: {result}")
    
    if result and "map_id" in result:
        print("✅ Enhanced change map created successfully!")
        print(f"Map ID: {result['map_id']}")
        print(f"Edit URL: {result['edit_url']}")
        print(f"Public URL: {result['publish_url']}")
        print("🔍 This map should show detailed tooltips with current, previous, delta, and % change values")
    else:
        print("❌ Failed to create enhanced change map")
        
    # Test data for a locator map with points - single point with minimal styling
    test_point_data = [
        {
            "lat": 37.7749, 
            "lon": -122.4194, 
            "title": "Test Point", 
            "description": "Test Description"
        }
    ]
    
    # Test generating a point map
    result = generate_map(
        {},
        map_title="Test Single Point Map",
        map_type="point",
        location_data=test_point_data,
        map_metadata={"description": "Test single point map"}
    )
    
    print(f"Generate point map result: {result}")
    
    # Test data for a locator map with addresses - single address
    test_address_data = [
        {
            "address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA", 
            "title": "Test Address", 
            "description": "Test Description"
        }
    ]
    
    # Test generating an address map
    result = generate_map(
        {},
        map_title="Test Single Address Map",
        map_type="address",
        location_data=test_address_data,
        map_metadata={"description": "Test single address map"}
    )
    
    print(f"Generate address map result: {result}")
    
    # Test data for a locator map with series - multiple categories with different colors
    test_series_data = create_sample_series_data()
    
    # Test generating a series map with default categorical colors
    result = generate_map(
        {},
        map_title="Test Multi-Series Map - SF Public Services",
        map_type="point",
        location_data=test_series_data,
        map_metadata={"description": "Test multi-series map with different colored categories"},
        series_field="series",
        color_palette="categorical"
    )
    
    print(f"Generate series map result: {result}")
    
    if result and "map_id" in result:
        print("✅ Multi-series map created successfully!")
        print(f"Map ID: {result['map_id']}")
        print(f"Edit URL: {result['edit_url']}")
        print(f"Public URL: {result['publish_url']}")
        print("🎨 This map should show different colored markers for each series with a legend")
    else:
        print("❌ Failed to create multi-series map")
    
    # Test with custom color palette
    custom_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]  # Red, Green, Blue, Yellow
    result = generate_map(
        {},
        map_title="Test Custom Colors Map - SF Public Services",
        map_type="point",
        location_data=test_series_data,
        map_metadata={"description": "Test map with custom color palette"},
        series_field="series",
        color_palette=custom_colors
    )
    
    print(f"Generate custom colors map result: {result}")

    # Duplicate dRKcH but do not upload data
    duplicate_response = _make_dw_request(
        "POST",
        "/charts/dRKcH/copy"
    )
    print("Duplicate response:", duplicate_response)
    if duplicate_response and "id" in duplicate_response:
        chart_id = duplicate_response["id"]
        print("New chart ID:", chart_id)
        print("Edit URL:", f"https://app.datawrapper.de/map/{chart_id}/visualize") 

    # Test GeoJSON with SF landmarks
    test_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "title": "City Hall",
                    "description": "San Francisco City Hall",
                    "color": "#0066cc",
                    "icon": "building",
                    "scale": "1.5"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.4194, 37.7749]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "title": "Golden Gate Bridge",
                    "description": "Iconic suspension bridge",
                    "color": "#cc0000",
                    "icon": "bridge",
                    "scale": "1.2"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.4783, 37.8199]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "title": "Ferry Building",
                    "description": "Historic ferry terminal and marketplace",
                    "color": "#006600",
                    "icon": "building",
                    "scale": "1.3"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.3937, 37.7956]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "title": "Dolores Park",
                    "description": "Popular park in the Mission District",
                    "color": "#009900",
                    "icon": "park",
                    "scale": "1.4"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.4265, 37.7597]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "title": "Sutro Tower",
                    "description": "TV and radio transmission tower",
                    "color": "#666666",
                    "icon": "tower",
                    "scale": "1.2"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.4527, 37.7552]
                }
            }
        ]
    }
    
    print("\nTest GeoJSON for SF landmarks:")
    print(json.dumps(test_geojson, indent=2)) 
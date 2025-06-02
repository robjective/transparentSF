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
import math
import re

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
DEFAULT_LOCATOR_CHART = "cMIaQ"
# Default reference chart for symbol maps
DEFAULT_SYMBOL_CHART = "K8LoR"

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
        series_field (str, optional): Field name to use for series grouping
        color_palette (list, optional): List of colors to use for series
        
    Returns:
        A JSON string representing the marker data for Datawrapper API.
    """
    import uuid
    
    # Define default color palettes for series
    default_color_palettes = {
        'categorical': [
            '#ad35fa',  # Bright Purple
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
        ]
    }
    
    # Use provided color palette or default categorical palette
    if color_palette is None:
        color_palette = default_color_palettes['categorical']
    elif isinstance(color_palette, str) and color_palette in default_color_palettes:
        color_palette = default_color_palettes[color_palette]
    
    def parse_coordinates(coord_value):
        """Parse coordinates from various formats into [longitude, latitude] format."""
        if not coord_value:
            return None
            
        try:
            if isinstance(coord_value, (list, tuple)) and len(coord_value) >= 2:
                lon, lat = float(coord_value[0]), float(coord_value[1])
                if -90 <= lon <= 90 and -180 <= lat <= 180:
                    return [lat, lon]
                else:
                    return [lon, lat]
                    
            elif isinstance(coord_value, str):
                parts = coord_value.split(',')
                if len(parts) >= 2:
                    coord1, coord2 = float(parts[0].strip()), float(parts[1].strip())
                    if -90 <= coord1 <= 90 and -180 <= coord2 <= 180:
                        return [coord2, coord1]
                    else:
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
        
        sorted_series = sorted(list(unique_series))
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
        
        # Generate a title if none provided
        title = loc.get("title")
        if not title:
            title=""
        
        # Skip if no valid location data
        if not parsed_coordinates and not has_lat_lon and not has_address:
            logger.warning(f"Skipping location due to missing valid coordinates, lat/lon, or address: {loc}")
            continue

        # Create marker object with complete structure matching working locator maps
        marker = {
            "type": "point",
            "title": title,
            "icon": {
                "id": "circle",
                "path": "M1000 350a500 500 0 0 0-500-500 500 500 0 0 0-500 500 500 500 0 0 0 500 500 500 500 0 0 0 500-500z",
                "height": 700,
                "width": 1000,
                "horiz-adv-x": 1000,
                "scale": 1
            },
            "scale": loc.get("scale", 1),
            "markerColor": loc.get("markerColor", "#8b5cf6"),  # Use purple like working example
            "opacity": loc.get("opacity", 1),  # Keep higher opacity for visibility
            "text": {
                "color": "#333333",
                "fontSize": 14,
                "halo": "#f2f3f0",
                "bold": False,
                "italic": False,
                "uppercase": False,
                "space": False
            },
            "id": str(uuid.uuid4())[:10],  # Shorter ID like working example
            "markerSymbol": "",
            "markerTextColor": "#333333",
            "anchor": "bottom-center",
            "offsetY": 0,
            "offsetX": 0,
            "labelStyle": "plain",
            "class": "",
            "rotate": 0,
            "visible": True,
            "locked": False,
            "preset": "-",
            "alpha": loc.get("opacity",1),  # Use alpha property as well
            "visibility": {
                "mobile": True,
                "desktop": True
            },
            "connectorLine": {
                "enabled": False,
                "arrowHead": "lines",
                "type": "curveRight",
                "targetPadding": 3,
                "stroke": 1,
                "lineLength": 0
            },
            "coordinates": parsed_coordinates or [float(loc["lon"]), float(loc["lat"])],
            "tooltip": {
                "text": loc.get("tooltip", title),
                "enabled": True
            }
        }
        
        # Add series information if available
        if series_field and loc.get(series_field):
            series_value = str(loc[series_field])
            marker["markerColor"] = series_color_map.get(series_value, marker["markerColor"])
            marker["series"] = series_value
        
        markers.append(marker)
    
    # Create the final marker data structure
    marker_data = {
        "markers": markers
    }
    
    return json.dumps(marker_data)
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
    Fit the map view to show all markers optimally using the proper Datawrapper view structure.
    
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
        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        max_diff = max(lon_diff, lat_diff)
        
        # Zoom level calculation for SF area
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
        
        # Add some padding by reducing zoom slightly for better view
        zoom_level = max(10, zoom_level - 1)
        
        # Add padding to bounds for better visualization
        padding_factor = 0.1  # 10% padding
        lon_padding = lon_diff * padding_factor
        lat_padding = lat_diff * padding_factor
        
        # Create the proper view structure matching the working reference chart
        view_payload = {
            "metadata": {
                "visualize": {
                    "view": {
                        "fit": {
                            "top": [center_lon, max_lat + lat_padding],
                            "left": [min_lon - lon_padding, center_lat],
                            "right": [max_lon + lon_padding, center_lat],
                            "bottom": [center_lon, min_lat - lat_padding]
                        },
                        "zoom": zoom_level,
                        "pitch": 0,
                        "center": center_coords,
                        "height": 75,
                        "bearing": 0
                    }
                }
            }
        }
        
        logger.info(f"Fitting map to markers: center={center_coords}, zoom={zoom_level}")
        logger.info(f"Bounds: lon=[{min_lon:.4f}, {max_lon:.4f}], lat=[{min_lat:.4f}, {max_lat:.4f}]")
        
        _make_dw_request(
            "PATCH",
            f"/charts/{chart_id}",
            json_payload=view_payload
        )
        
        logger.info(f"Successfully fitted map {chart_id} to markers with proper view structure")
        return True
        
    except Exception as e:
        logger.error(f"Error fitting map to markers: {str(e)}", exc_info=True)
        return False

def _create_and_configure_locator_map(chart_title, markers_json_string, center_coords=None, zoom_level=None, series_info=None):
    """
    Creates and configures a Datawrapper locator map with markers.

    Args:
        chart_title (str): Title for the chart
        markers_json_string (str): JSON string containing marker data
        center_coords (list, optional): [longitude, latitude] for map center
        zoom_level (int/float, optional): Zoom level for the map
        series_info (dict, optional): Information about series for legend

    Returns:
        str: Chart ID if successful, None otherwise
    """
    try:
        # 1. Create empty locator map
        logger.info(f"Creating empty locator map with title: {chart_title}")
        create_response = _make_dw_request(
            "POST",
            "/charts",
            json_payload={
            "title": chart_title,
            "type": "locator-map"
        }
        )
        if not create_response or "id" not in create_response:
            logger.error("Failed to create empty locator map")
            return None
        chart_id = create_response["id"]
        logger.info(f"Successfully created empty locator map with ID: {chart_id}")

        # 2. Add markers to the map
        logger.info(f"Adding markers to map ID: {chart_id}")
        logger.info(f"Marker data being uploaded: {markers_json_string[:500]}...")  # Show first 500 chars
        
        _make_dw_request(
            "PUT",
            f"/charts/{chart_id}/data",
            headers={"Content-Type": "application/json"},
            data=markers_json_string
        )
        logger.info(f"Markers uploaded to map ID: {chart_id}")

        # Verify markers were uploaded correctly
        try:
            chart_data_response = _make_dw_request("GET", f"/charts/{chart_id}/data")
            if chart_data_response:
                if isinstance(chart_data_response, str):
                    try:
                        chart_data = json.loads(chart_data_response)
                        markers = chart_data.get("markers", [])
                        logger.info(f"Verification: Found {len(markers)} markers in chart data")
                        if markers:
                            logger.info(f"First marker: {markers[0]}")
                        else:
                            logger.warning("No markers found in chart data after upload!")
                    except json.JSONDecodeError:
                        logger.warning("Could not parse chart data response as JSON")
                else:
                    markers = chart_data_response.get("markers", [])
                    logger.info(f"Verification: Found {len(markers)} markers in chart data")
            else:
                logger.warning("No response when verifying chart data")
        except Exception as e:
            logger.warning(f"Error verifying markers: {e}")

        # 3. Set initial metadata and styling with comprehensive marker configuration
        initial_metadata_payload = {
            "metadata": {
                "data": {
                    "json": True
                },
                "visualize": {
                    "defaultMapSize": 600,
                    "mapLabel": True,
                    "scale": False,
                    "compass": False,
                    "style": "dw-light",
                    "visibility": {
                        "green": True,
                        "roads": True,
                        "urban": True,
                        "water": True,
                        "building": True,
                        "glaciers": False,
                        "mountains": False,
                        "building3d": False,
                        "boundary_state": False,
                        "boundary_country": True
                    },
                    "miniMap": {
                        "enabled": False
                    },
                    "upload": {
                        "maxSize": 2000000,
                        "maxMarkers": 100
                    }
                },
                "publish": {
                    "embed-width": 600,
                    "embed-height": 400,
                    "blocks": {
                        "logo": {"enabled": False},
                        "embed": False,
                        "download-pdf": False,
                        "download-svg": False,
                        "get-the-data": True,
                        "download-image": False
                    }
                },
                "describe": {
                    "source-name": "DataSF",
                    "source-url": "https://data.sfgov.org",
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

        # Force a republish to ensure markers and metadata are processed
        logger.info(f"Republishing chart {chart_id} to ensure markers are visible")
        _make_dw_request(
            "POST",
            f"/charts/{chart_id}/publish"
        )

        # 4. Fit map to markers (this will override center_coords and zoom_level if not provided)
        if center_coords is None or zoom_level is None:
            logger.info(f"Fitting map to markers for optimal view")
            success = _fit_map_to_markers(chart_id, markers_json_string)
            if not success:
                logger.warning("Failed to fit map to markers, using fallback settings")
                # Fallback to manual calculation
                try:
                    markers_data = json.loads(markers_json_string)
                    if markers_data.get("markers"):
                        lats = [m["coordinates"][1] for m in markers_data["markers"] if m.get("coordinates")]
                        lons = [m["coordinates"][0] for m in markers_data["markers"] if m.get("coordinates")]
                        if lats and lons:
                            center_coords = [(min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2]
                            zoom_level = 12
                        else:
                            center_coords = [-122.44, 37.77] # SF default
                            zoom_level = 12
                    else:
                        center_coords = [-122.44, 37.77] # SF default
                        zoom_level = 12
                except Exception as e:
                    logger.error(f"Error calculating fallback view: {e}")
                    center_coords = [-122.44, 37.77] # SF default if parsing fails
                    zoom_level = 12
            
                # Apply fallback view settings
                logger.info(f"Applying fallback view: center={center_coords}, zoom={zoom_level}")
                fallback_view_payload = {
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
                    json_payload=fallback_view_payload
                )
                logger.info(f"Applied fallback view settings to chart {chart_id}")
        else:
            # Use provided center_coords and zoom_level
            logger.info(f"Using provided center and zoom: {center_coords}, {zoom_level}")
            custom_view_payload = {
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
                json_payload=custom_view_payload
            )
            logger.info(f"Applied custom view settings to chart {chart_id}")

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
        map_type: Type of map (supervisor_district, police_district, intersection, point, address, symbol)
        
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
    if map_type == "symbol":
        # For symbol maps, process addresses and geocode them
        processed_data = process_symbol_map_data(location_data)
    
    elif map_type == "address":
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

def create_datawrapper_chart(chart_title, location_data, map_type="supervisor_district", reference_chart_id=None, center_coords=None, zoom_level=None, series_field=None, color_palette=None, map_metadata=None):
    """
    Creates a Datawrapper chart, either by cloning a reference chart or creating from scratch.

    For locator maps (map_type='point', 'address', 'intersection'), it uses the new creation logic.
    For other types, it attempts to clone and update data.
    
    Args:
        chart_title (str): Title for the chart.
        location_data (list or str): Data for the chart. For locator maps, this should be a list of point dicts.
                                     For other maps, it can be CSV data string or path.
        map_type (str): Type of map (e.g., 'supervisor_district', 'point', 'address', 'intersection', 'symbol').
        reference_chart_id (str, optional): ID of a Datawrapper chart to clone for non-locator maps.
        center_coords (list, optional): [longitude, latitude] for locator map center.
        zoom_level (int/float, optional): Zoom level for locator map.
        series_field (str, optional): Field name to use for series grouping (e.g., 'series', 'category', 'type').
        color_palette (list or str, optional): Color palette for series. Can be list of colors or palette name.
        map_metadata (dict, optional): Additional metadata for styling and configuration.
        
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
        
        # NEW: Attempt to parse string-formatted location_data for locator maps (point, address, intersection)
        if isinstance(location_data, str) and map_type in ["point", "address", "intersection"]:
            # First, try JSON parsing
            try:
                possible_list = json.loads(location_data)
                if isinstance(possible_list, list):
                    location_data = possible_list
                    logger.info(f"Successfully parsed JSON string location_data into list with {len(location_data)} items for {map_type} map")
                else:
                    raise ValueError("Parsed JSON is not a list")
            except (json.JSONDecodeError, ValueError):
                # Fallback to ast.literal_eval for Python literal strings (single quotes, etc.)
                try:
                    import ast
                    possible_list = ast.literal_eval(location_data)
                    if isinstance(possible_list, list):
                        location_data = possible_list
                        logger.info(f"Successfully parsed location_data using ast.literal_eval with {len(location_data)} items for {map_type} map")
                    else:
                        logger.error("ast.literal_eval output is not a list; cannot process location_data for locator map")
                        return None
                except (ValueError, SyntaxError) as parse_err:
                    logger.error(f"Failed to parse location_data string for locator map ({map_type}): {parse_err}")
        
        # Handle location_data format conversion for district maps and symbol maps
        if map_type in ["supervisor_district", "police_district"] and isinstance(location_data, list):
            # Convert list of dictionaries to CSV format for district-based maps
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
                            # Convert to percentage, round to nearest integer, and clamp to [-100, 100] so extreme outliers use edge colours
                            value_for_coloring = max(min(round(percent_change * 100), 100), -100)
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
        
        # Handle location_data format conversion for symbol maps (coordinate-based)
        elif map_type == "symbol":
            # Symbol maps are now handled as locator maps, so skip CSV conversion
            pass
        
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
                                            "title": item.get("title", ''),
                                            "lat": float(lat),
                                            "lon": float(lon),
                                            "value": item.get("value", 0),
                                            "description": item.get("description", ""),
                                            "series": item.get("series", "")
                                        }
                                        
                                        # Copy over any series field data
                                        if series_field and series_field in item:
                                            processed_item[series_field] = item[series_field]
                                        elif series_field == "naic_code_description" and "naic_code_description" in item:
                                            processed_item[series_field] = item["naic_code_description"]
                                        
                                        processed_data.append(processed_item)
                                        logger.debug(f"Processed DataSF location for '{processed_item['title']}': [{lon}, {lat}]")
                                    else:
                                        logger.warning(f"Coordinates for '{item.get('title', '')}' are outside SF area: [{lon}, {lat}]")
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
                                # Determine coordinate format and parse accordingly
                                coord1, coord2 = float(coords_to_check[0]), float(coords_to_check[1])
                                
                                # Check if coordinates are in [lat, lon] or [lon, lat] format
                                # Latitude should be between -90 and 90, longitude between -180 and 180
                                # For SF: lat ~37.7, lon ~-122.4
                                if -90 <= coord1 <= 90 and -180 <= coord2 <= 180:
                                    # First coordinate looks like latitude, second like longitude
                                    lat, lon = coord1, coord2
                                else:
                                    # Assume first is longitude, second is latitude
                                    lon, lat = coord1, coord2
                                
                                # Validate coordinates are in San Francisco area
                                if 37.6 <= lat <= 37.9 and -122.6 <= lon <= -122.2:
                                    processed_data.append(item)
                                    logger.debug(f"Validated existing coordinates for '{item.get('title', 'Unknown')}': lat={lat}, lon={lon}")
                                else:
                                    logger.warning(f"Existing coordinates for '{item.get('title', 'Unknown')}' are outside SF area: lat={lat}, lon={lon}")
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

        # Existing logic for all map types using cloning approach (district maps, symbol maps, etc.)
        if map_type == "symbol":
            ref_chart_id = reference_chart_id or DEFAULT_SYMBOL_CHART
            logger.info(f"Using symbol map reference chart: {ref_chart_id}")
        else:
            ref_chart_id = reference_chart_id or DEFAULT_DISTRICT_CHART
            logger.info(f"Using district map reference chart: {ref_chart_id}")
            
        if not ref_chart_id:
            logger.error(f"No reference chart ID provided for {map_type} map type.")
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
        elif isinstance(location_data, list) and map_type == "symbol":
            # Special handling for symbol maps: geocode addresses and convert to CSV
            logger.info(f"Processing {len(location_data)} symbol map locations with geocoding")
            
            # Geocode the data using our existing function
            geocoded_data = process_symbol_map_data(location_data)
            logger.info(f"Successfully processed {len(geocoded_data)} geocoded symbol locations")
            
            # Convert to CSV format for the symbol map reference chart
            csv_data = "latitude,longitude,title,value,description,series\n"
            for item in geocoded_data:
                lat = item.get('lat', 0)
                lon = item.get('lon', 0)
                title = item.get('title', '').replace(',', ';')  # Replace commas to avoid CSV issues
                value = item.get('value', 1)
                description = item.get('description', '').replace(',', ';')
                series = item.get('series', '')
                csv_data += f"{lat},{lon},{title},{value},{description},{series}\n"
            
            data_to_upload = csv_data
            logger.info(f"Converted {len(geocoded_data)} symbol locations to CSV format")
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

        # After creating the chart and before returning, apply custom styling for district maps
        if map_type in ["supervisor_district", "police_district"]:
            _apply_custom_map_styling(chart_id, map_type, map_metadata)

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
    Apply custom styling to a Datawrapper chart based on map type and metadata.
    
    Args:
        chart_id (str): The Datawrapper chart ID
        map_type (str): Type of map (e.g., 'supervisor_district', 'point', 'address', 'intersection', 'symbol')
        map_metadata (dict, optional): Additional metadata for styling
    """
    if not DATAWRAPPER_API_KEY:
        logger.error("DATAWRAPPER_API_KEY not set. Cannot apply styling.")
        return
    
    try:
        # Get the current chart data
        chart_url = f"https://api.datawrapper.de/v3/charts/{chart_id}"
        headers = {
            "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Get current chart data
        response = requests.get(chart_url, headers=headers)
        response.raise_for_status()
        chart_data = response.json()
        
        # Get metric info from metadata if available
        metric_info = map_metadata.get('metric_info') if map_metadata else None
        item_noun = metric_info.get('item_noun', 'Items') if metric_info else 'Items'
        greendirection = metric_info.get('greendirection', 'up') if metric_info else 'up'
        
        # Initialize styling payload
        styling_payload = {
            "metadata": {
                "visualize": {},
                "describe": {
                    "intro": map_metadata.get("description", "") if map_metadata else "",
                    "source-name": "DataSF",
                    "source-url": map_metadata.get("executed_url", "") if map_metadata else "",
                    "byline": "TransparentSF"
                }
            }
        }
        
        # Apply different styling based on map type
        if map_type in ["supervisor_district", "police_district"]:
            # Base configuration for all district maps
            base_config = {
                "basemap": "custom_upload",
                "basemapFilename": "districts_geojson.json",
                "basemapProjection": "geoAzimuthalEqualArea",
                "map-key-attr": "district",
                "map-type-set": True,
                "chart-type-set": True,
                "zoomable": True,
                "map-align": "center",
                "map-padding": 0,
                "hide-region-borders": True,
                "hide-empty-regions": False,
                "basemapRegions": "all",
                "max-map-height": 650,
                "min-label-zoom": 1,
                "zoom-button-pos": "br",
                "map-label-format": "0,0.[00]",
                "avoid-label-overlap": True,
                "mapViewCropPadding": 10,
                "basemapShowExtraOptions": False
            }
            
            # Check if this is a delta/change map
            is_delta_map = map_metadata and map_metadata.get("map_type") == "delta"
            
            if is_delta_map:
                # Delta map styling (for showing changes) - ALWAYS APPLY THIS FOR DELTA MAPS
                # Determine colors based on greendirection
                if greendirection == 'down':
                    # For metrics where decrease is good (e.g., crime)
                    colors = [
                        {"color": "#00dca6", "position": 0},      # Strong Green (for -100%)
                        {"color": "#a7e9d8", "position": 0.25},   # Light Green
                        {"color": "#eeeeee", "position": 0.5},    # Neutral Gray (for 0%)
                        {"color": "#f87171", "position": 0.75},   # Light Red
                        {"color": "#dc2626", "position": 1.0}     # Strong Red (for +100%)
                    ]
                else:
                    # For metrics where increase is good (e.g., housing units)
                    colors = [
                        {"color": "#dc2626", "position": 0},      # Strong Red (for -100%)
                        {"color": "#f87171", "position": 0.25},   # Light Red
                        {"color": "#eeeeee", "position": 0.5},    # Neutral Gray (for 0%)
                        {"color": "#a7e9d8", "position": 0.75},   # Light Teal/Green
                        {"color": "#00dca6", "position": 1.0}     # Strong Teal/Green (for +100%)
                    ]
                
                styling_payload["metadata"]["visualize"] = {
                    **base_config,
                    "colorscale": {
                        "mode": "continuous",
                        "stops": "equidistant",
                        "colors": colors,
                        "palette": 0,
                        "stopCount": 5,
                        "interpolation": "equidistant",
                        "min": -100,
                        "max": 100,
                        "domain": [-100, 100],
                        "rangeMin": -100,
                        "rangeMax": 100,
                        "rangeCenter": 0,
                        "customStops": [] # Clear any custom stops from template
                    },
                    "legends": {
                        "color": {
                            "size": 170,
                            "title": f"CHANGE IN {item_noun.upper()}",
                            "labels": "ranges",
                            "enabled": True,
                            "offsetX": 0,
                            "offsetY": 0,
                            "reverse": False,
                            "labelMax": "100%",  # Fixed maximum label
                            "labelMin": "-100%",  # Fixed minimum label
                            "position": "above",
                            "interactive": True,
                            "labelCenter": "medium",
                            "labelFormat": "0'%'",  # Format as integer percentage since values are already * 100
                            "orientation": "horizontal",
                            "titleEnabled": False,
                            "customLabels": []  # Clear any custom labels
                        }
                    },
                    "tooltip": {
                        "body": f"Current: {{{{current_value}}}} {item_noun}<br>Previous: {{{{previous_value}}}} {item_noun}<br>Change: {{{{delta}}}} {item_noun}<br>% Change: {{{{value}}}}%",
                        "title": "District {{ district }}",
                        "sticky": True,
                        "enabled": True
                    }
                }
            else:
                # Density map styling (for showing absolute values)
                styling_payload["metadata"]["visualize"] = {
                    **base_config,
                    "colorscale": {
                        "mode": "continuous",
                        "stops": "equidistant",
                        "colors": [
                            {"color": "#E9D8FA", "position": 0},      # Very Light Purple for low
                            {"color": "#ad35fa", "position": 1.0}     # Purple for high
                        ],
                        "palette": 0,
                        "stopCount": 2,
                        "interpolation": "linear"
                    },
                    "legends": {
                        "color": {
                            "size": 170,
                            "title": f"NUMBER OF {item_noun.upper()}",
                            "labels": "ranges",
                            "enabled": True,
                            "offsetX": 0,
                            "offsetY": 0,
                            "reverse": False,
                            "labelMax": "",  # Let Datawrapper auto-calculate from data
                            "labelMin": "",  # Let Datawrapper auto-calculate from data
                            "position": "above",
                            "interactive": True,
                            "labelCenter": "medium",
                            "labelFormat": "0,0.[00]",  # Format as number
                            "orientation": "horizontal",
                            "titleEnabled": False
                        }
                    },
                    "tooltip": {
                        "body": f"{{{{value}}}} {item_noun}",
                        "title": "District {{ district }}",
                        "sticky": True,
                        "enabled": True
                    }
                }
            
                # Add any custom colorscale if provided in metadata FOR NON-DELTA MAPS
                if map_metadata and "colorscale" in map_metadata and not is_delta_map:
                    styling_payload["metadata"]["visualize"]["colorscale"] = map_metadata["colorscale"]
        
        # Apply the styling
        _make_dw_request(
            "PATCH",
            f"/charts/{chart_id}",
            json_payload=styling_payload
        )
        
        logger.info(f"Applied custom styling to chart {chart_id} (type: {map_type})")
    except Exception as e:
        logger.error(f"Error applying custom styling to chart {chart_id}: {str(e)}", exc_info=True)

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

def generate_map(context_variables, map_title, map_type, location_data=None, map_metadata=None, reference_chart_id=None, metric_id=None, group_field=None, series_field=None, color_palette=None):
    """
    Generates a map using Datawrapper, stores its metadata, and returns relevant URLs.
    
    Args:
        context_variables (dict): Dictionary of context variables (e.g., for database connection).
        map_title (str): Desired title for the map.
        map_type (str): Type of map (e.g., 'supervisor_district', 'police_district', 'point', 'address', 'intersection', 'symbol').
        location_data (list or str, optional): Data for the map. 
                                   If None or "from_context", will use dataset from context_variables.
                                   For locator maps ('point', 'address', 'intersection'): list of dicts with point/address details.
                                   For district maps: list of dicts [{"district": "1", "value": 120}] OR CSV string.
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
    
    # Check if we should use dataset from context_variables
    if location_data is None or location_data == "from_context":
        dataset = context_variables.get("dataset")
        if dataset is not None and not dataset.empty:
            logger.info(f"Using dataset from context_variables with shape: {dataset.shape}")
            logger.info(f"Dataset columns: {dataset.columns.tolist()}")
            
            # Convert DataFrame to location_data based on map_type
            if map_type in ["point", "address", "intersection"]:
                # For locator maps, convert DataSF format to our format
                location_data = []
                for idx, row in dataset.iterrows():
                    if 'location' in row and row['location'] is not None:
                        location_obj = row['location']
                        if isinstance(location_obj, dict) and location_obj.get('type') == 'Point':
                            coords = location_obj.get('coordinates')
                            if coords and len(coords) >= 2:
                                item = {
                                    "location": location_obj,  # Keep original DataSF format
                                    "title": row.get('dba_name', row.get('business_name', row.get('name', ''))),
                                    "tooltip": f"Industry: {row.get('naic_code_description', 'Unknown')}"
                                }
                                # Copy over any series field data
                                if series_field and series_field in row:
                                    item[series_field] = row[series_field]
                                location_data.append(item)
                
                logger.info(f"Converted dataset to {len(location_data)} location points")
                
            elif map_type == "symbol":
                # For symbol maps, convert DataFrame to address-based location data
                logger.info(f"DEBUG: Starting symbol map conversion")
                location_data = []
                logger.info(f"DEBUG: Dataset has {len(dataset)} rows")
                for idx, row in dataset.iterrows():
                    logger.info(f"DEBUG: Processing row {idx}: {dict(row)}")
                    if 'address' in row and row['address'] is not None:
                        item = {
                            "title": row.get('title', ''),
                            "address": row['address'],
                            "value": row.get('value', 1),
                            "description": row.get('description', ''),
                        }
                        # Copy over any series field data
                        if series_field and series_field in row:
                            item[series_field] = row[series_field]
                        location_data.append(item)
                        logger.info(f"DEBUG: Added item: {item}")
                    else:
                        logger.warning(f"DEBUG: Row {idx} missing 'address' field or address is None")
                
                logger.info(f"Converted dataset to {len(location_data)} symbol map locations")
                
            elif map_type in ["supervisor_district", "police_district"]:
                # For district maps, we'd need aggregation logic here
                # This is more complex and might need the original district aggregation approach
                logger.warning("District maps from context dataset not yet implemented - falling back to original location_data")
                if not location_data:
                    return {"error": "District maps require aggregated location_data or manual data input"}
        else:
            logger.warning("No dataset found in context_variables or dataset is empty")
            if not location_data:
                return {"error": "No location_data provided and no dataset available in context_variables"}
    
    # Get metric information if metric_id is provided
    metric_info = None
    if metric_id:
        try:
            from tools.metrics_manager import get_metric_by_id
            metric_result = get_metric_by_id(metric_id)
            if metric_result["status"] == "success":
                metric_info = metric_result["metric"]
                logger.info(f"Retrieved metric info for ID {metric_id}")
        except Exception as e:
            logger.warning(f"Failed to get metric info for ID {metric_id}: {str(e)}")
    
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
    
    # NEW: Attempt to parse string-formatted location_data for locator maps (point, address, intersection)
    if isinstance(location_data, str) and map_type in ["point", "address", "intersection"]:
        # First, try JSON parsing
        try:
            possible_list = json.loads(location_data)
            if isinstance(possible_list, list):
                location_data = possible_list
                logger.info(f"Successfully parsed JSON string location_data into list with {len(location_data)} items for {map_type} map")
            else:
                raise ValueError("Parsed JSON is not a list")
        except (json.JSONDecodeError, ValueError):
            # Fallback to ast.literal_eval for Python literal strings (single quotes, etc.)
            try:
                import ast
                possible_list = ast.literal_eval(location_data)
                if isinstance(possible_list, list):
                    location_data = possible_list
                    logger.info(f"Successfully parsed location_data using ast.literal_eval with {len(location_data)} items for {map_type} map")
                else:
                    logger.error("ast.literal_eval output is not a list; cannot process location_data for locator map")
                    return None
            except (ValueError, SyntaxError) as parse_err:
                logger.error(f"Failed to parse location_data string for locator map ({map_type}): {parse_err}")
    
    # Handle location_data format conversion for district maps and symbol maps
    if map_type in ["supervisor_district", "police_district"] and isinstance(location_data, list):
        # Convert list of dictionaries to CSV format for district-based maps
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
                        # Convert to percentage, round to nearest integer, and clamp to [-100, 100] so extreme outliers use edge colours
                        value_for_coloring = max(min(round(percent_change * 100), 100), -100)
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
        "map_type": map_type,
        "map_metadata": map_metadata
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
        # For symbol maps, pass color and size settings
        elif map_type == "symbol":
            if "color" in map_metadata:
                chart_args["color_palette"] = map_metadata["color"]
    
    # Add series support for locator maps
    if map_type in ["point", "address", "intersection"]:
        if series_field:
            chart_args["series_field"] = series_field
        if color_palette:
            chart_args["color_palette"] = color_palette
    
    # For non-locator maps, pass reference_chart_id if provided
    if map_type not in ["point", "address", "intersection"]:
        chart_args["reference_chart_id"] = reference_chart_id

    chart_id = None
    try:
        chart_id = create_datawrapper_chart(**chart_args)

        if not chart_id:
            logger.error(f"Failed to create Datawrapper chart for '{map_title}'.")
            return None

        logger.info(f"Datawrapper chart created/updated with ID: {chart_id} for title '{map_title}'")

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

        # Add metric info to map_metadata if available
        if metric_info:
            if map_metadata is None:
                map_metadata = {}
            elif isinstance(map_metadata, str):
                try:
                    map_metadata = json.loads(map_metadata)
                except json.JSONDecodeError:
                    map_metadata = {}
            map_metadata['metric_info'] = metric_info
        
        # Apply custom styling before publishing
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
            elif map_type == "symbol":
                map_subtype = "symbol"
            
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
        # If chart_id exists, the chart may have been partially created in Datawrapper
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

def process_symbol_map_data(location_data):
    """
    Process location data for symbol maps, including geocoding addresses.
    
    Args:
        location_data: List of location data objects with addresses or coordinates
        
    Returns:
        Processed location data with coordinates
    """
    processed_data = []
    
    # Separate items that need geocoding from those that already have coordinates
    items_to_geocode = []
    items_with_coords = []
    
    for item in location_data:
        if isinstance(item, dict):
            # Check if item already has coordinates
            has_lat = item.get('lat') is not None or item.get('latitude') is not None
            has_lon = item.get('lon') is not None or item.get('longitude') is not None
            
            if has_lat and has_lon:
                # Item already has coordinates
                lat = item.get('lat') or item.get('latitude')
                lon = item.get('lon') or item.get('longitude')
                
                processed_item = {
                    "title": item.get('title', ''),
                    "lat": float(lat),
                    "lon": float(lon),
                    "value": item.get('value', 0),
                    "description": item.get('description', ''),
                    "series": item.get('series', '')
                }
                items_with_coords.append(processed_item)
                logger.info(f"Using existing coordinates for '{processed_item['title']}': ({lat}, {lon})")
            else:
                # Item needs geocoding
                address = item.get('address') or item.get('building_address')
                if address:
                    items_to_geocode.append(item)
    
    # Process items with existing coordinates
    processed_data.extend(items_with_coords)
    
    # Geocode items that need it
    if items_to_geocode:
        addresses = [item.get('address') or item.get('building_address') for item in items_to_geocode]
        geocoded_results = batch_geocode_addresses(addresses)
        
        # Combine original data with geocoded coordinates
        for item in items_to_geocode:
            address = item.get('address') or item.get('building_address')
            if address:
                lat, lon = geocoded_results.get(address, (None, None))
                
                if lat is not None and lon is not None:
                    processed_item = {
                        "title": item.get('title', ''),
                        "lat": lat,
                        "lon": lon,
                        "value": item.get('value', 0),
                        "description": item.get('description', ''),
                        "series": item.get('series', '')
                    }
                    processed_data.append(processed_item)
                    logger.info(f"Geocoded '{address}' to ({lat}, {lon})")
                else:
                    logger.warning(f"Failed to geocode address: {address}")
    
    # Calculate successfully geocoded count
    geocoded_count = 0
    if items_to_geocode:
        geocoded_count = len([item for item in items_to_geocode if geocoded_results.get(item.get('address') or item.get('building_address'), (None, None))[0] is not None])
    
    logger.info(f"Processed {len(processed_data)} total symbol locations ({len(items_with_coords)} with existing coords, {geocoded_count} geocoded)")
    
    return processed_data

# Testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from datetime import datetime

    def print_menu():
        print("\nMap Type Test Menu:")
        print("1. Supervisor District Map")
        print("2. Point (Locator) Map")
        print("3. Address Map")
        print("5. Symbol Map")
        print("6. Series/Legend Map (Colored Series)")
        print("7. Context Dataset Map")
        print("0. Exit")

    def test_supervisor_district():
        test_location_data = [
            {"district": str(i), "value": 100 - i * 5} for i in range(1, 12)
        ]
        test_metadata = {
            "description": "Test map showing values by supervisor district",
            "source": "Test data",
            "date_created": datetime.now().isoformat()
        }
        result = generate_map(
            {},
            map_title="Test Supervisor District Map",
            map_type="supervisor_district",
            location_data=test_location_data,
            map_metadata=test_metadata
        )
        print(f"Result: {result}")

    def test_point_locator():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_location_data = [
            {
                "title": "City Hall",
                "lat": 37.7793,
                "lon": -122.4193,
                "tooltip": "San Francisco City Hall",
                "markerColor": "#cc0000",
                "scale": 0.6,
                "opacity": 0.8
            },
            {
                "title": "Ferry Building",
                "lat": 37.7955,
                "lon": -122.3937,
                "tooltip": "Ferry Building",
                "markerColor": "#cc0000",
                "scale": 0.6,
                "opacity": 0.8
            },
            {
                "title": "Oracle Park",
                "lat": 37.7786,
                "lon": -122.3893,
                "tooltip": "Oracle Park",
                "markerColor": "#cc0000",
                "scale": 0.6,
                "opacity": 0.8
            }
        ]
        test_metadata = {
            "description": "Test locator map with points",
            "source": "Test data",
            "date_created": datetime.now().isoformat(),
            "center": [-122.4043, 37.7870],  # Center of the three points
            "zoom": 13  # Appropriate zoom level for SF
        }
        result = generate_map(
            {},
            map_title=f"Test Point Locator Map - {timestamp}",
            map_type="point",
            location_data=test_location_data,
            map_metadata=test_metadata
        )
        print(f"Result: {result}")

    def test_address_map():
        test_location_data = [
            {"address": "1 Dr Carlton B Goodlett Pl"},
            {"address": "Ferry Building, San Francisco"},
            {"address": "24 Willie Mays Plaza"}
        ]
        test_metadata = {
            "description": "Test address map",
            "source": "Test data",
            "date_created": datetime.now().isoformat()
        }
        result = generate_map(
            {},
            map_title="Test Address Map",
            map_type="address",
            location_data=test_location_data,
            map_metadata=test_metadata
        )
        print(f"Result: {result}")


    def test_symbol_map():
        test_location_data = [
            {
                "name": "District 1",
                "latitude": 37.7800,
                "longitude": -122.4800,
                "value": 50,
                "color": "#ad35fa",
                "description": "District 1: 50 units"
            },
            {
                "name": "District 2",
                "latitude": 37.7900,
                "longitude": -122.4200,
                "value": 80,
                "color": "#4A7463",
                "description": "District 2: 80 units"
            },
            {
                "name": "District 3",
                "latitude": 37.7600,
                "longitude": -122.4100,
                "value": 30,
                "color": "#FF6B5A",
                "description": "District 3: 30 units"
            }
        ]
        test_metadata = {
            "description": "Test symbol map",
            "source": "Test data",
            "date_created": datetime.now().isoformat(),
            "color": "#ad35fa"
        }
        result = generate_map(
            {},
            map_title="Test Symbol Map",
            map_type="symbol",
            location_data=test_location_data,
            map_metadata=test_metadata
        )
        print(f"Result: {result}")

    def test_series_legend_map():
        test_location_data = [
            {"title": "Police Station - Central", "lat": 37.7749, "lon": -122.4194, "tooltip": "Central Police Station", "series": "Police Stations"},
            {"title": "Fire Station - Engine 1", "lat": 37.7849, "lon": -122.4094, "tooltip": "Fire Station Engine 1", "series": "Fire Stations"},
            {"title": "Hospital - UCSF", "lat": 37.7627, "lon": -122.4581, "tooltip": "UCSF Medical Center", "series": "Hospitals"},
            {"title": "School - Lowell High", "lat": 37.7197, "lon": -122.4831, "tooltip": "Lowell High School", "series": "Schools"}
        ]
        test_metadata = {
            "description": "Test series/legend map",
            "source": "Test data",
            "date_created": datetime.now().isoformat()
        }
        result = generate_map(
            {},
            map_title="Test Series/Legend Map",
            map_type="point",
            location_data=test_location_data,
            map_metadata=test_metadata,
            series_field="series",
            color_palette="categorical"
        )
        print(f"Result: {result}")

    def test_context_dataset():
        """Test using dataset from context_variables"""
        import pandas as pd
        
        # Create a mock dataset similar to what would come from DataSF
        mock_data = [
            {
                "dba_name": "Sample Business 1",
                "naic_code_description": "Food Services",
                "location": {
                    "type": "Point",
                    "coordinates": [-122.4194, 37.7749]  # SF City Hall area
                },
                "markerColor": "#ff0000",
                "scale": 1.0,
                "opacity": 1.0
            },
            {
                "dba_name": "Sample Business 2", 
                "naic_code_description": "Retail Trade",
                "location": {
                    "type": "Point",
                    "coordinates": [-122.3937, 37.7955]  # Ferry Building area
                },
                "markerColor": "#ff0000",
                "scale": 1.0,
                "opacity": 1.0
            },
            {
                "dba_name": "Sample Business 3",
                "naic_code_description": "Professional Services", 
                "location": {
                    "type": "Point",
                    "coordinates": [-122.3893, 37.7786]  # Near Oracle Park
                },
                "markerColor": "#ff0000",
                "scale": 1.0,
                "opacity": 1.0
            }
        ]
        
        # Create DataFrame and context_variables like the real system would
        df = pd.DataFrame(mock_data)
        test_context_variables = {
            "dataset": df,
            "executed_query_url": "https://data.sfgov.org/test-query"
        }
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Testing context-based map generation...")
        print(f"Mock dataset shape: {df.shape}")
        print(f"Mock dataset columns: {df.columns.tolist()}")
        
        # Test the new approach - no location_data needed!
        result = generate_map(
            context_variables=test_context_variables,
            map_title=f"Test Context Dataset Map - {timestamp}",
            map_type="point",
            # location_data=None  # This should use context_variables automatically
        )
        print(f"Result: {result}")

    menu_functions = {
        "1": test_supervisor_district,
        "2": test_point_locator,
        "3": test_address_map,
        "5": test_symbol_map,
        "6": test_series_legend_map,
        "7": test_context_dataset
    }

    while True:
        print_menu()
        choice = input("Select a map type to test (0 to exit): ").strip()
        if choice == "0":
            print("Exiting.")
            break
        elif choice in menu_functions:
            menu_functions[choice]()
    else:
            print("Invalid choice. Please try again.")
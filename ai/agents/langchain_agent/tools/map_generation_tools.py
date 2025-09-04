"""
Map Generation Tools for LangChain Agent
========================================

This module provides tools for the LangChain explainer agent to generate maps
using the existing TransparentSF map generation system.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import pandas as pd

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

logger = logging.getLogger(__name__)

def generate_map_tool(context_variables: Dict[str, Any], map_title: str, map_type: str, 
                      map_metadata: Optional[Dict[str, Any]] = None, 
                      series_field: Optional[str] = None, 
                      color_palette: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a map visualization for geographic data using the TransparentSF map generation system.
    
    Args:
        context_variables: Context variables for the current analysis
        map_title: Descriptive title for the map
        map_type: Type of map to create. Must be one of:
            * "supervisor_district" - Map showing data by San Francisco supervisor district (1-11)
            * "police_district" - Map showing data by San Francisco police district
            * "intersection" - Map showing points at specific street intersections
            * "point" - Map showing points at specific lat/long coordinates
            * "address" - Map showing points at specific addresses (will be geocoded automatically)
            * "symbol" - Scaled-symbol map for points. Use this when you want marker size to represent a value
        map_metadata: Dictionary with additional information about the map
            * For change/delta maps, use: {"map_type": "delta", "description": "Change from previous period"}
            * For basic density maps, use: {"description": "Current values by district"}
            * For point/address/intersection maps, you can specify view settings:
              {"description": "Description", "zoom_level": 12, "center_lat": 37.7749, "center_lon": -122.4194}
        series_field: Optional field name for grouping markers into different colored series (only for point/address/intersection maps)
        color_palette: Optional color palette for series maps. Options:
            * "categorical" - Different colors for each series (default)
            * "status" - Green, Amber, Red, Blue, Purple for status data
            * "priority" - Red, Orange, Yellow, Green, Blue for priority levels
            * "sequential" - Graduated colors for sequential data
            * Custom list of hex colors: ["#FF0000", "#00FF00", "#0000FF"]
    
    Returns:
        Dictionary with map_id and URLs for editing and viewing the map
        
    Example:
        generate_map_tool(
            context_variables={},
            map_title="Crime Incidents by District",
            map_type="supervisor_district",
            map_metadata={"description": "Monthly crime incidents by supervisor district"}
        )
    """
    logger.info("=== Starting generate_map_tool ===")
    logger.info(f"Map title: {map_title}")
    logger.info(f"Map type: {map_type}")
    logger.info(f"Map metadata: {map_metadata}")
    logger.info(f"Series field: {series_field}")
    logger.info(f"Color palette: {color_palette}")
    
    try:
        # Import the map generation function from the main tools directory
        from ai.tools.generate_map import generate_map
        
        # Call the existing generate_map function
        result = generate_map(
            context_variables=context_variables,
            map_title=map_title,
            map_type=map_type,
            map_metadata=map_metadata or {},
            series_field=series_field,
            color_palette=color_palette
        )
        
        if result and "map_id" in result:
            logger.info(f"Map generated successfully with ID: {result['map_id']}")
            return {
                'status': 'success',
                'map_id': result['map_id'],
                'edit_url': result.get('edit_url'),
                'publish_url': result.get('publish_url'),
                'message': f'Map "{map_title}" created successfully'
            }
        else:
            logger.error("Map generation failed - no map_id returned")
            return {
                'status': 'error',
                'error': 'Map generation failed - no map_id returned',
                'result': result
            }
            
    except Exception as e:
        logger.exception(f"Error in generate_map_tool: {str(e)}")
        return {
            'status': 'error',
            'error': f'Failed to generate map: {str(e)}'
        }

def get_map_by_id_tool(context_variables: Dict[str, Any], map_id: int) -> Dict[str, Any]:
    """
    Retrieve a previously created map by ID.
    
    Args:
        context_variables: Context variables for the current analysis
        map_id: The ID of the map to retrieve
        
    Returns:
        Dictionary with map details including URLs and metadata
    """
    logger.info("=== Starting get_map_by_id_tool ===")
    logger.info(f"Map ID: {map_id}")
    
    try:
        # Import the function from the main tools directory
        from ai.tools.generate_map import get_map_by_id
        
        result = get_map_by_id(context_variables, map_id)
        
        if result and "map_id" in result:
            logger.info(f"Map {map_id} retrieved successfully")
            return {
                'status': 'success',
                'map': result
            }
        else:
            logger.warning(f"Map {map_id} not found or invalid")
            return {
                'status': 'error',
                'error': f'Map {map_id} not found or invalid'
            }
            
    except Exception as e:
        logger.exception(f"Error in get_map_by_id_tool: {str(e)}")
        return {
            'status': 'error',
            'error': f'Failed to retrieve map: {str(e)}'
        }

def get_recent_maps_tool(context_variables: Dict[str, Any], limit: int = 10, 
                         map_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a list of recently created maps.
    
    Args:
        context_variables: Context variables for the current analysis
        limit: Maximum number of maps to return (default: 10)
        map_type: Optional filter by map type (e.g., "supervisor_district", "point")
        
    Returns:
        Dictionary with list of recent maps
    """
    logger.info("=== Starting get_recent_maps_tool ===")
    logger.info(f"Limit: {limit}")
    logger.info(f"Map type filter: {map_type}")
    
    try:
        # Import the function from the main tools directory
        from ai.tools.generate_map import get_recent_maps
        
        result = get_recent_maps(context_variables, limit, map_type)
        
        if result and "maps" in result:
            logger.info(f"Retrieved {len(result['maps'])} recent maps")
            return {
                'status': 'success',
                'maps': result['maps'],
                'total_count': len(result['maps'])
            }
        else:
            logger.warning("No recent maps found")
            return {
                'status': 'success',
                'maps': [],
                'total_count': 0
            }
            
    except Exception as e:
        logger.exception(f"Error in get_recent_maps_tool: {str(e)}")
        return {
            'status': 'error',
            'error': f'Failed to retrieve recent maps: {str(e)}'
        }

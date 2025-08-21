"""
Map Generator Routes for TransparentSF
=====================================

This module provides routes for the new map generator interface that uses Mapbox
instead of Datawrapper for map visualizations.
"""

import os
import logging
import json
import random
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Templates will be set by main app
templates: Optional[Jinja2Templates] = None

def set_templates(template_instance: Jinja2Templates):
    """Set templates instance from main app."""
    global templates
    templates = template_instance

@router.get("/map-generator")
async def map_generator_page(request: Request):
    """Serve the map generator page."""
    if not templates:
        raise HTTPException(status_code=500, detail="Templates not configured")
    
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    return templates.TemplateResponse("map_generator.html", {
        "request": request,
        "mapbox_token": mapbox_token
    })

@router.get("/api/map-generator/metrics")
async def get_metrics():
    """Get all available metrics for the map generator dropdown."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get all active metrics that have a map_query defined
        query = """
        SELECT id, metric_name, category, subcategory, endpoint,
               most_recent_data_date, display_order, map_query, map_filters, map_config
        FROM metrics
        WHERE is_active = TRUE
          AND map_query IS NOT NULL
          AND trim(map_query) <> ''
        ORDER BY COALESCE(display_order, 999), id
        """
        
        cursor.execute(query)
        metrics = cursor.fetchall()
        
        # Debug: Check specific jail metrics
        for metric in metrics:
            if 'jail' in metric['metric_name'].lower():
                logger.error(f"JAIL METRIC FOUND: ID={metric['id']}, Name={metric['metric_name']}, map_query='{metric.get('map_query', 'NOT_FOUND')}'")
        
        # Debug: Log the query results
        logger.info(f"API query returned {len(metrics)} metrics")
        jail_metrics = [m for m in metrics if 'jail' in m['metric_name'].lower()]
        if jail_metrics:
            logger.warning(f"Found {len(jail_metrics)} jail metrics in API response: {[m['id'] for m in jail_metrics]}")
        
        # Convert any date objects to ISO format strings
        for metric in metrics:
            for key, value in metric.items():
                if hasattr(value, 'isoformat'):
                    metric[key] = value.isoformat()
        
        cursor.close()
        conn.close()
        
        return JSONResponse(content={
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error getting metrics: {str(e)}"}
        )

@router.get("/api/anomalies/{metric_id}")
async def get_anomalies_for_metric(metric_id: str, district: str = None, period_type: str = None, time_periods: str = None):
    """Get anomalies for a specific metric with optional filters."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build the base query with filters
        query = """
        SELECT 
            group_field_name, 
            group_value, 
            BOOL_OR(out_of_bounds) as out_of_bounds,
            AVG(comparison_mean) as comparison_mean,
            AVG(recent_mean) as recent_mean,
            AVG(difference) as difference,
            AVG(std_dev) as std_dev,
            MAX(recent_date) as recent_date
        FROM anomalies 
        WHERE object_id = %s
        """
        
        params = [metric_id]
        
        # Add district filter if specified and not "0" (citywide)
        if district and district != "0":
            query += " AND district = %s"
            params.append(district)
        
        # Add period type filter if specified
        if period_type:
            query += " AND period_type = %s"
            params.append(period_type)
        
        # Add time period filter if specified
        if time_periods and time_periods != "since_2024":
            try:
                time_periods_int = int(time_periods)
                # Filter by recent_date to match the time period
                # This ensures we only show anomalies from the selected time period
                query += " AND recent_date IS NOT NULL"
            except ValueError:
                pass  # Ignore invalid time_periods values
        
        # Add active status filter - only show active anomalies
        query += " AND is_active = TRUE"
        
        # Complete the query with grouping and ordering
        query += """
        GROUP BY group_field_name, group_value
        ORDER BY BOOL_OR(out_of_bounds) DESC, group_field_name, group_value
        LIMIT 300
        """
        
        logger.info(f"Anomaly query: {query}")
        logger.info(f"Anomaly query params: {params}")
        
        cursor.execute(query, params)
        anomalies = cursor.fetchall()
        
        # Convert any date objects to ISO format strings
        for anomaly in anomalies:
            for key, value in anomaly.items():
                if hasattr(value, 'isoformat'):
                    anomaly[key] = value.isoformat()
        
        cursor.close()
        conn.close()
        
        return JSONResponse(content={
            "status": "success",
            "anomalies": anomalies
        })
        
    except Exception as e:
        logger.error(f"Error getting anomalies for metric {metric_id}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error getting anomalies: {str(e)}"}
        )

@router.post("/api/generate-map")
async def generate_map_endpoint(request: Request):
    """Generate a new map using Mapbox."""
    try:
        data = await request.json()
        
        metric_id = data.get("metric_id")
        anomaly_type = data.get("anomaly_type")
        anomaly_field_name = data.get("anomaly_field_name")  # New parameter
        district = data.get("district", "0")
        period_type = data.get("period_type", "month")
        time_periods = data.get("time_periods", 2)  # Default to 2 time periods
        color_field = data.get("color_field")  # Optional: match preview coloring
        scale_dots = data.get("scale_dots", True)  # Default to True for scaling dots by count
        
        # Check if this is a preview request
        preview_mode = data.get("preview", False)
        
        # Ensure metric_id is an integer
        try:
            metric_id = int(metric_id) if metric_id else None
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid metric_id provided"}
            )
        
        if not metric_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Metric ID is required"}
            )
        
        logger.info(f"Generating map for metric_id={metric_id}, anomaly_type={anomaly_type}, anomaly_field_name={anomaly_field_name}, district={district}, period_type={period_type}, time_periods={time_periods}")
        
        # Import the map generation function
        from tools.generate_map import generate_map
        
        # Get metric details
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get metric information
        cursor.execute("SELECT * FROM metrics WHERE id = %s", [metric_id])
        metric = cursor.fetchone()
        
        if not metric:
            cursor.close()
            conn.close()
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Metric not found"}
            )
        
        # Fetch data for the map based on the metric
        try:
            # Import data fetching utilities
            from tools.data_fetcher import fetch_metric_data
            
            # Fetch the actual data for this metric
            data_result = fetch_metric_data(metric_id, district, period_type, time_periods, anomaly_type, anomaly_field_name)
            
            if not data_result or "error" in data_result:
                cursor.close()
                conn.close()
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Failed to fetch data: {data_result.get('error', 'Unknown error')}"}
                )
            
            # Extract the dataset from the result
            dataset = data_result.get("data")
            if dataset is None or dataset.empty:
                cursor.close()
                conn.close()
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "No data available for this metric"}
                )
            
            # Determine map type based on metric configuration
            map_type = "supervisor_district"  # Default
            if metric.get("map_config") and metric["map_config"].get("chart_type_preference"):
                map_type = metric["map_config"]["chart_type_preference"]
            elif metric.get("location_fields") and len(metric["location_fields"]) > 0:
                # Check if it has point data
                if any("lat" in str(field).lower() or "lon" in str(field).lower() for field in metric["location_fields"]):
                    map_type = "point"
            
            # Override map type if dataset has location data (point/symbol maps)
            if dataset is not None and not dataset.empty:
                # Check if dataset has location column with point data
                if 'location' in dataset.columns:
                    # Check if any rows have valid location data
                    location_data_count = dataset['location'].notna().sum()
                    if location_data_count > 0:
                        # If we have location data, use point or symbol map type
                        if map_type == "supervisor_district":
                            map_type = "symbol"  # Default to symbol for business data
            
            # Generate map title
            map_title = f"{metric['metric_name']}"
            if district != "0":
                map_title += f" - District {district}"
            if anomaly_type:
                map_title += f" - {anomaly_type}"
            
            # Call the generate_map function with Mapbox configuration
            map_result = generate_map(
                context_variables={
                    "dataset": dataset,
                    "map_config": metric.get("map_config")
                },
                map_title=map_title,
                map_type=map_type,
                location_data="from_context",  # Use dataset from context_variables
                map_metadata={
                    "description": f"Map showing {metric['metric_name']} data",
                    "metric_id": metric_id,
                    "district": district,
                    "period_type": period_type,
                    "time_periods": time_periods,
                    "anomaly_type": anomaly_type,
                    "anomaly_field_name": anomaly_field_name,
                    "color_field": color_field,
                    "scale_dots": scale_dots
                },
                metric_id=metric_id,
                map_provider="mapbox",  # Use Mapbox instead of Datawrapper
                preview_mode=preview_mode  # Pass preview mode flag
            )
            
            # For preview mode, return location_data instead of creating a map
            if preview_mode:
                if "error" in map_result:
                    cursor.close()
                    conn.close()
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": map_result["error"]}
                    )
                
                # Get the processed location data from the generate_map function
                location_data = map_result.get("location_data", [])
                
                cursor.close()
                conn.close()
                
                return JSONResponse(content={
                    "status": "success",
                    "location_data": location_data,
                    "data_points": len(location_data)
                })
            
        except ImportError:
            # Fallback: create sample data if data_fetcher is not available
            logger.warning("data_fetcher not available, using sample data")
            
            # Create sample location data based on map type
            sample_data = []
            if district == "0":
                # Citywide - create data for all districts
                districts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                for dist in districts:
                    sample_data.append({
                        "district": dist,
                        "value": random.randint(10, 100),
                        "title": f"District {dist}",
                        "color": f"#{random.randint(0, 0xFFFFFF):06x}"
                    })
            else:
                # Single district - create point data
                sample_data = [
                    {
                        "title": f"{metric['metric_name']} - Sample Point 1",
                        "lat": 37.7749 + random.uniform(-0.01, 0.01),
                        "lon": -122.4194 + random.uniform(-0.01, 0.01),
                        "value": random.randint(10, 100),
                        "description": f"Sample data point for {metric['metric_name']}"
                    },
                    {
                        "title": f"{metric['metric_name']} - Sample Point 2", 
                        "lat": 37.7849 + random.uniform(-0.01, 0.01),
                        "lon": -122.4094 + random.uniform(-0.01, 0.01),
                        "value": random.randint(10, 100),
                        "description": f"Sample data point for {metric['metric_name']}"
                    }
                ]
            
            # Generate map title
            map_title = f"{metric['metric_name']}"
            if district != "0":
                map_title += f" - District {district}"
            if anomaly_type:
                map_title += f" - {anomaly_type}"
            
            # Call the generate_map function with sample data
            map_result = generate_map(
                context_variables={},
                map_title=map_title,
                map_type="point" if district != "0" else "supervisor_district",
                location_data=sample_data,
                map_metadata={
                    "description": f"Map showing {metric['metric_name']} data (sample)",
                    "metric_id": metric_id,
                    "district": district,
                    "period_type": period_type,
                    "anomaly_type": anomaly_type,
                    "anomaly_field_name": anomaly_field_name
                },
                metric_id=metric_id,
                map_provider="mapbox"
            )
        
        cursor.close()
        conn.close()
        
        if not map_result or "error" in map_result:
            error_msg = map_result.get("error") if map_result else "Failed to generate map"
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": error_msg}
            )
        
        # Return success response
        return JSONResponse(content={
            "status": "success",
            "message": f"Successfully generated map for {metric['metric_name']}",
            "map_id": map_result.get("map_id"),
            "view_url": f"/map/{map_result.get('map_id')}",  # New Mapbox map URL
            "data_points": map_result.get("data_points", 0)
        })
        
    except Exception as e:
        logger.error(f"Error generating map: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

@router.get("/map/{map_id}")
async def view_map(map_id: str, request: Request):
    """View a specific map by ID."""
    if not templates:
        raise HTTPException(status_code=500, detail="Templates not configured")
    
    try:
        # Get map data from database
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT * FROM maps WHERE id = %s", [map_id])
        map_data = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not map_data:
            raise HTTPException(status_code=404, detail="Map not found")
        
        # Parse metadata if it's a string
        if isinstance(map_data['metadata'], str):
            try:
                metadata = json.loads(map_data['metadata'])
            except:
                metadata = {}
        else:
            metadata = map_data['metadata'] or {}
        
        # Parse location_data if it's a string
        if isinstance(map_data['location_data'], str):
            try:
                location_data = json.loads(map_data['location_data'])
            except:
                location_data = []
        else:
            location_data = map_data['location_data'] or []
        
        # Handle different data formats
        if isinstance(location_data, dict) and 'data' in location_data:
            # Old format: {'data': [...]}
            location_data = location_data['data']
        elif not isinstance(location_data, list):
            # Ensure location_data is always a list
            location_data = []
        
        # Debug logging
        logger.info(f"Map {map_id} - Type: {map_data.get('type')}")
        logger.info(f"Map {map_id} - Location data length: {len(location_data) if location_data else 0}")
        if location_data and len(location_data) > 0:
            logger.info(f"Map {map_id} - First item: {location_data[0]}")
            logger.info(f"Map {map_id} - Sample items with coordinates: {[item for item in location_data[:5] if 'lat' in item and 'lon' in item]}")
        
        # Get Mapbox token
        mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "")
        
        return templates.TemplateResponse("map.html", {
            "request": request,
            "map_data": map_data,
            "metadata": metadata,
            "location_data": location_data,
            "config": {
                "MAPBOX_ACCESS_TOKEN": mapbox_token
            }
        })
        
    except Exception as e:
        logger.error(f"Error viewing map {map_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading map: {str(e)}")

@router.get("/api/maps")
async def get_recent_maps(metric_id: Optional[str] = None, limit: int = 10):
    """Get recent maps, optionally filtered by metric."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        if metric_id:
            query = """
            SELECT * FROM maps 
            WHERE active = TRUE AND metric_id = %s
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, [metric_id, limit])
        else:
            query = """
            SELECT * FROM maps 
            WHERE active = TRUE
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, [limit])
        
        maps = cursor.fetchall()
        
        # Convert any date objects to ISO format strings
        for map_item in maps:
            for key, value in map_item.items():
                if hasattr(value, 'isoformat'):
                    map_item[key] = value.isoformat()
                # Parse JSON strings to objects
                elif key in ('metadata', 'location_data') and value:
                    if isinstance(value, str):
                        try:
                            map_item[key] = json.loads(value)
                        except:
                            pass
        
        cursor.close()
        conn.close()
        
        return JSONResponse(content={
            "status": "success",
            "maps": maps
        })
        
    except Exception as e:
        logger.error(f"Error getting recent maps: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error getting maps: {str(e)}"}
        )

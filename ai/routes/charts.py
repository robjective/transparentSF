#!/usr/bin/env python3
"""
Charts route module for handling chart-related API endpoints.
Moved from backend.py to reduce its size and improve organization.
"""

import os
import json
import logging
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize APIRouter
router = APIRouter()

# Templates will be set by the main app
templates = None

def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logging.info("Templates set in charts router")

# Configure logging
logger = logging.getLogger(__name__)

@router.get("/api/charts-for-review")
async def get_charts_for_review_api(
    limit: int = 10,  # Reduced from 20 to 10
    days_back: int = 14,  # Reduced from 30 to 14
    district_filter: Optional[str] = None,
    include_time_series: bool = True,
    include_anomalies: bool = True,
    include_maps: bool = True,
    only_active: bool = True,
    metric_filter: Optional[str] = None,
    object_type_filter: Optional[str] = None,
    metric_id: Optional[str] = None,
    only_recent: bool = True,  # New parameter
    max_total_charts: int = 25,  # New parameter
    include_metadata: bool = False,  # New parameter
    include_urls: bool = False,  # New parameter
    sort_by: str = "created_at"  # New parameter
):
    """Get charts from various sources for AI review and potential newsletter inclusion."""
    try:
        from tools.get_charts_for_review import get_charts_for_review
        
        context_variables = {}  # Add any context variables needed
        result = get_charts_for_review(
            context_variables=context_variables,
            limit=limit,
            days_back=days_back,
            district_filter=district_filter,
            include_time_series=include_time_series,
            include_anomalies=include_anomalies,
            include_maps=include_maps,
            only_active=only_active,
            metric_filter=metric_filter,
            object_type_filter=object_type_filter,
            metric_id=metric_id,
            only_recent=only_recent,
            max_total_charts=max_total_charts,
            include_metadata=include_metadata,
            include_urls=include_urls,
            sort_by=sort_by
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting charts for review: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "data": {
                    "time_series_charts": [],
                    "anomaly_charts": [],
                    "map_charts": [],
                    "summary": {"total_charts": 0, "error": str(e)}
                }
            }
        )

@router.get("/api/chart/{chart_id}")
async def get_chart_data(chart_id: str):
    """Get a specific chart by ID."""
    try:
        import psycopg2
        import psycopg2.extras
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            database=os.environ.get("POSTGRES_DB", "transparentsf"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query to get chart metadata
        cursor.execute("""
            SELECT 
                chart_id, 
                chart_title, 
                y_axis_label, 
                period_type, 
                object_type, 
                object_id, 
                object_name, 
                field_name, 
                district, 
                group_field,
                executed_query_url
            FROM time_series_metadata 
            WHERE chart_id = %s
        """, (chart_id,))
        
        metadata_result = cursor.fetchone()
        
        if not metadata_result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Chart with ID {chart_id} not found")
        
        # Query to get chart data points
        cursor.execute("""
            SELECT 
                time_period, 
                group_value, 
                numeric_value
            FROM time_series_data 
            WHERE chart_id = %s
            ORDER BY time_period
        """, (chart_id,))
        
        data_results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Map database period type to frontend period type for the response
        frontend_period_type_map = {
            'year': 'annual',
            'month': 'monthly',
            'week': 'weekly',
            'day': 'daily'
        }
        
        # Format the response according to the requested structure
        response = {
            "metadata": {
                "chart_id": metadata_result["chart_id"],
                "chart_title": metadata_result["chart_title"],
                "y_axis_label": metadata_result["y_axis_label"],
                "period_type": frontend_period_type_map.get(metadata_result["period_type"], metadata_result["period_type"]),
                "object_type": metadata_result["object_type"],
                "object_id": metadata_result["object_id"],
                "object_name": metadata_result["object_name"],
                "field_name": metadata_result["field_name"],
                "district": metadata_result["district"],
                "executed_query_url": metadata_result.get("executed_query_url", ""),
                "source-name": "DataSF",
                "byline": "Chart: TransparentSF"
            }
        }
        
        # Add group_field if it exists
        if metadata_result["group_field"]:
            response["metadata"]["group_field"] = metadata_result["group_field"]
            logger.info(f"Added group_field to response metadata: {metadata_result['group_field']}")
        
        # Format the data points
        response["data"] = []
        for row in data_results:
            # Clean the numeric value to handle NaN and infinite values
            numeric_value = row["numeric_value"]
            if numeric_value is None or (isinstance(numeric_value, float) and (math.isnan(numeric_value) or math.isinf(numeric_value))):
                continue
                
            data_point = {
                "time_period": row["time_period"].isoformat(),
                "numeric_value": numeric_value
            }
            
            # Add group_value if it exists
            if row["group_value"]:
                data_point["group_value"] = row["group_value"]
                
            response["data"].append(data_point)
        
        logger.info(f"Retrieved chart data for chart_id: {chart_id}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chart data: {str(e)}")

@router.get("/api/active-charts")
async def get_active_charts(
    limit: int = 50,
    chart_type: Optional[str] = None,
    district_filter: Optional[str] = None
):
    """Get active charts with optional filtering."""
    try:
        from tools.get_charts_for_review import get_charts_for_review
        
        # Determine which chart types to include
        include_time_series = chart_type is None or chart_type == "time_series"
        include_anomalies = chart_type is None or chart_type == "anomaly"
        include_maps = chart_type is None or chart_type == "map"
        
        result = get_charts_for_review(
            context_variables={},
            limit=limit,
            days_back=30,
            district_filter=district_filter,
            include_time_series=include_time_series,
            include_anomalies=include_anomalies,
            include_maps=include_maps,
            only_active=True,
            only_recent=False,
            max_total_charts=limit * 3,  # Allow more charts since we're filtering by type
            include_metadata=False,
            include_urls=False
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting active charts: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@router.get("/api/chart-by-metric")
async def get_chart_by_metric(
    metric_id: str,
    district: int = 0,
    chart_type: str = "time_series"
):
    """Get charts for a specific metric and district."""
    try:
        from tools.get_charts_for_review import get_charts_for_review
        
        # Determine which chart types to include based on chart_type parameter
        include_time_series = chart_type == "time_series"
        include_anomalies = chart_type == "anomaly"
        include_maps = chart_type == "map"
        
        result = get_charts_for_review(
            context_variables={},
            limit=10,
            days_back=90,  # Look back 3 months
            district_filter=str(district),
            metric_id=metric_id,
            include_time_series=include_time_series,
            include_anomalies=include_anomalies,
            include_maps=include_maps,
            only_active=True,
            only_recent=False,
            max_total_charts=20,
            include_metadata=True,
            include_urls=True
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting chart by metric: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@router.get("/api/recent-charts-summary")
async def get_recent_charts_summary(
    days_back: int = 7,
    only_active: bool = True,
    metric_id: Optional[str] = None
):
    """Get a summary of recently created charts."""
    try:
        from tools.get_charts_for_review import get_recent_charts_summary
        
        result = get_recent_charts_summary(
            context_variables={},
            days_back=days_back,
            only_active=only_active,
            metric_id=metric_id
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting recent charts summary: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@router.get("/api/chart/{chart_id}")
@router.get("/backend/api/chart/{chart_id}")
async def get_chart_data(chart_id: int):
    """
    Get chart data for a specific chart_id.
    Accessible via both /api/chart/{chart_id} and /backend/api/chart/{chart_id}
    
    Returns:
        ChartResponse with metadata and data points
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            database=os.environ.get("POSTGRES_DB", "transparentsf"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query to get chart metadata
        cursor.execute("""
            SELECT 
                chart_id, 
                chart_title, 
                y_axis_label, 
                period_type, 
                object_type, 
                object_id, 
                object_name, 
                field_name, 
                district, 
                group_field,
                executed_query_url
            FROM time_series_metadata 
            WHERE chart_id = %s
        """, (chart_id,))
        
        metadata_result = cursor.fetchone()
        
        if not metadata_result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Chart with ID {chart_id} not found")
        
        # Query to get chart data points
        cursor.execute("""
            SELECT 
                time_period, 
                group_value, 
                numeric_value
            FROM time_series_data 
            WHERE chart_id = %s
            ORDER BY time_period
        """, (chart_id,))
        
        data_results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Map database period type to frontend period type for the response
        frontend_period_type_map = {
            'year': 'annual',
            'month': 'monthly',
            'week': 'weekly',
            'day': 'daily'
        }
        
        # Format the response according to the requested structure
        response = {
            "metadata": {
                "chart_id": metadata_result["chart_id"],
                "chart_title": metadata_result["chart_title"],
                "y_axis_label": metadata_result["y_axis_label"],
                "period_type": frontend_period_type_map.get(metadata_result["period_type"], metadata_result["period_type"]),
                "object_type": metadata_result["object_type"],
                "object_id": metadata_result["object_id"],
                "object_name": metadata_result["object_name"],
                "field_name": metadata_result["field_name"],
                "district": metadata_result["district"],
                "executed_query_url": metadata_result.get("executed_query_url", ""),
                "source-name": "DataSF",
                "byline": "Chart: TransparentSF"
            }
        }
        
        # Add group_field if it exists
        if metadata_result["group_field"]:
            response["metadata"]["group_field"] = metadata_result["group_field"]
            logger.info(f"Added group_field to response metadata: {metadata_result['group_field']}")
        
        # Format the data points
        response["data"] = []
        for row in data_results:
            # Clean the numeric value to handle NaN and infinite values
            numeric_value = row["numeric_value"]
            if numeric_value is None or (isinstance(numeric_value, float) and (math.isnan(numeric_value) or math.isinf(numeric_value))):
                continue
                
            data_point = {
                "time_period": row["time_period"].isoformat(),
                "numeric_value": numeric_value
            }
            
            # Add group_value if it exists
            if row["group_value"]:
                data_point["group_value"] = row["group_value"]
                
            response["data"].append(data_point)
        
        logger.info(f"Retrieved chart data for chart_id: {chart_id}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chart data: {str(e)}")

@router.get("/api/active-charts")
@router.get("/backend/api/active-charts")
async def get_active_charts_legacy(metric_id: str, district: str = "0", period_type: str = "month"):
    """
    Get all active chart IDs for a given metric ID, district, and period type.
    Legacy endpoint for backward compatibility.
    
    Args:
        metric_id: The ID of the metric to get charts for
        district: The district ID (default: "0" for citywide)
        period_type: The period type (month, year, etc.)
        
    Returns:
        JSON with a list of active charts including their IDs and metadata
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info(f"Getting active charts for metric_id={metric_id}, district={district}, period_type={period_type}")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get all active charts for the specified metric, district, and period type
        query = """
        SELECT 
            chart_id, 
            object_name, 
            group_field, 
            object_id,
            district,
            period_type,
            is_active
        FROM time_series_metadata
        WHERE object_id = %s 
        AND district = %s 
        AND period_type = %s
        AND is_active = TRUE
        ORDER BY 
            CASE WHEN group_field IS NULL THEN 0 ELSE 1 END,
            group_field ASC
        """
        
        cursor.execute(query, [metric_id, district, period_type])
        charts = cursor.fetchall()
        
        # Convert any date objects to ISO format strings for JSON serialization
        for chart in charts:
            for key, value in chart.items():
                if hasattr(value, 'isoformat'):
                    chart[key] = value.isoformat()
        
        logger.info(f"Found {len(charts)} active charts for metric_id={metric_id}")
        
        cursor.close()
        conn.close()
        
        return JSONResponse(
            content={
                "status": "success",
                "metric_id": metric_id,
                "district": district,
                "period_type": period_type,
                "chart_count": len(charts),
                "charts": [dict(chart) for chart in charts]
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting active charts: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )

@router.get("/api/chart-by-metric")
@router.get("/backend/api/chart-by-metric")
async def get_chart_by_metric_legacy(
    metric_id: str,
    district: int = 0,
    period_type: str = 'year',
    group_field: str = None,
    groups: str = None
):
    """
    Get chart data for a specific metric, district, and period type.
    Legacy endpoint for backward compatibility.
    Accessible via both /api/chart-by-metric and /backend/api/chart-by-metric
    
    Parameters:
        metric_id: The ID of the metric (matches object_id in the database)
        district: The district ID (default: 0)
        period_type: The period type (default: 'year')
        group_field: Optional group field (default: null)
        groups: Comma-separated list of group values to include (default: null)
    
    Returns:
        ChartResponse with metadata and data points
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        # Map frontend period types to database period types if needed
        period_type_map = {
            'annual': 'year',
            'monthly': 'month',
            'weekly': 'week',
            'daily': 'day'
        }
        
        # Convert period type if it's one of the frontend values
        db_period_type = period_type_map.get(period_type, period_type)
        
        # Parse groups parameter if provided
        group_values = None
        if groups:
            if groups.strip() == '':
                # Empty string means no groups should be shown
                group_values = []
                logger.info("Empty groups parameter - will return no data")
            else:
                group_values = [g.strip() for g in groups.split(',')]
                logger.info(f"Using specified group values: {group_values}")
        
        logger.info(f"Looking for chart with object_id={metric_id}, district={district}, period_type={db_period_type}, group_field={group_field}, groups={groups}")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            database=os.environ.get("POSTGRES_DB", "transparentsf"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres")
        )
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Build the query with proper handling of group_field NULL vs value
        if group_field is None:
            group_field_condition = "AND group_field IS NULL"
            params = (metric_id, district, db_period_type)
        else:
            group_field_condition = "AND group_field = %s"
            params = (metric_id, district, db_period_type, group_field)
        
        # Query to find the chart
        query = f"""
            SELECT 
                chart_id, 
                period_type, 
                object_type, 
                object_id, 
                object_name, 
                field_name, 
                district, 
                group_field,
                executed_query_url,
                caption,
                metadata
            FROM time_series_metadata 
            WHERE object_id = %s AND district = %s 
            AND period_type = %s {group_field_condition}
            AND is_active = TRUE
        """
        
        cursor.execute(query, params)
        chart_metadata = cursor.fetchone()
        
        if not chart_metadata:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"No active chart found for metric_id={metric_id}, district={district}, period_type={db_period_type}, group_field={group_field}")
        
        # Get the chart data
        chart_id = chart_metadata["chart_id"]
        
        # Query to get chart data points
        data_query = """
            SELECT 
                time_period, 
                group_value, 
                numeric_value
            FROM time_series_data 
            WHERE chart_id = %s
        """
        
        data_params = [chart_id]
        
        # Add group filtering if specified
        if group_values is not None:
            if len(group_values) == 0:
                # Empty list means no groups should be shown
                data_query += " AND group_value IS NULL"
            else:
                placeholders = ','.join(['%s'] * len(group_values))
                data_query += f" AND group_value IN ({placeholders})"
                data_params.extend(group_values)
        
        data_query += " ORDER BY time_period"
        
        cursor.execute(data_query, data_params)
        data_results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Map database period type to frontend period type for the response
        frontend_period_type_map = {
            'year': 'annual',
            'month': 'monthly',
            'week': 'weekly',
            'day': 'daily'
        }
        
        # Format the response
        response = {
            "metadata": {
                "chart_id": chart_metadata["chart_id"],
                "period_type": frontend_period_type_map.get(chart_metadata["period_type"], chart_metadata["period_type"]),
                "object_type": chart_metadata["object_type"],
                "object_id": chart_metadata["object_id"],
                "object_name": chart_metadata["object_name"],
                "field_name": chart_metadata["field_name"],
                "district": chart_metadata["district"],
                "executed_query_url": chart_metadata.get("executed_query_url", ""),
                "source-name": "DataSF",
                "byline": "Chart: TransparentSF"
            }
        }
        
        # Add group_field if it exists
        if chart_metadata["group_field"]:
            response["metadata"]["group_field"] = chart_metadata["group_field"]
        
        # Format the data points
        response["data"] = []
        for row in data_results:
            # Clean the numeric value to handle NaN and infinite values
            numeric_value = row["numeric_value"]
            if numeric_value is None or (isinstance(numeric_value, float) and (math.isnan(numeric_value) or math.isinf(numeric_value))):
                continue
                
            data_point = {
                "time_period": row["time_period"].isoformat(),
                "numeric_value": numeric_value
            }
            
            # Add group_value if it exists
            if row["group_value"]:
                data_point["group_value"] = row["group_value"]
                
            response["data"].append(data_point)
        
        logger.info(f"Retrieved chart data for metric_id={metric_id}, district={district}, period_type={db_period_type}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart by metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chart data: {str(e)}")

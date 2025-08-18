#!/usr/bin/env python3
"""
Migration script to populate the new map_query, map_filters, and map_config columns
for existing metrics, especially complex ones like metric ID 46.
"""

import os
import sys
import json
import re
import logging
from typing import Dict, Any, List, Optional
from db_utils import get_postgres_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_geometry_filter(query: str) -> Optional[Dict[str, Any]]:
    """Extract geometry filter from a query."""
    # Look for within_polygon function calls
    polygon_match = re.search(r"within_polygon\(([^,]+),\s*'([^']+)'\)", query, re.IGNORECASE)
    if polygon_match:
        field = polygon_match.group(1).strip()
        geometry = polygon_match.group(2)
        return {
            "type": "within_polygon",
            "field": field,
            "value": geometry
        }
    return None

def extract_date_logic(query: str) -> Optional[Dict[str, Any]]:
    """Extract complex date logic from queries like metric ID 46."""
    # Look for CASE WHEN date logic
    case_when_match = re.search(
        r"CASE\s+WHEN\s+([^>=]+)\s*>=\s*'([^']+)'\s+THEN\s+([^ELSE]+)\s+ELSE\s+([^END]+)\s+END",
        query, re.IGNORECASE | re.DOTALL
    )
    
    if case_when_match:
        condition_field = case_when_match.group(1).strip()
        condition_date = case_when_match.group(2).strip()
        then_field = case_when_match.group(3).strip()
        else_field = case_when_match.group(4).strip()
        
        return {
            "field": then_field,
            "fallback_field": else_field,
            "fallback_condition": f"{condition_field} < '{condition_date}'"
        }
    
    # Simple date field detection
    date_fields = ["dba_start_date", "location_start_date", "incident_date", "requested_datetime", "created_date"]
    for field in date_fields:
        if field in query:
            return {"field": field}
    
    return None

def extract_static_filters(query: str) -> List[Dict[str, Any]]:
    """Extract static WHERE conditions that should always be applied."""
    filters = []
    
    # This is a simplified extraction - in practice you might want more sophisticated parsing
    # Look for simple equality conditions
    eq_matches = re.findall(r"(\w+)\s*=\s*'([^']+)'", query)
    for field, value in eq_matches:
        if field not in ["dba_start_date", "location_start_date", "incident_date"]:  # Skip date fields
            filters.append({
                "field": field,
                "operator": "=",
                "value": value
            })
    
    return filters

def create_map_query(endpoint: str, metric_query: str) -> str:
    """Create a simplified map query without complex transformations."""
    # Start with basic SELECT
    base_query = "SELECT *"
    
    # Add location field explicitly if we can detect it
    if "location" in metric_query and "SELECT *" not in base_query:
        base_query = "SELECT *, location"
    
    # Add endpoint reference (this will be filled in by the map generation code)
    return base_query

def get_map_config(metric: Dict[str, Any]) -> Dict[str, Any]:
    """Generate map configuration based on metric properties."""
    config = {
        "title_template": "{metric_name}",
        "supports_districts": True,
        "supports_time_periods": True,
        "data_point_threshold": 100
    }
    
    # Detect chart type preference based on data type
    if "business" in metric.get("subcategory", "").lower():
        config["chart_type_preference"] = "symbol"
    elif "incident" in metric.get("item_noun", "").lower():
        config["chart_type_preference"] = "point"
    else:
        config["chart_type_preference"] = "symbol"
    
    # Extract date field from ytd_query if available
    ytd_query = metric.get("ytd_query", "")
    date_logic = extract_date_logic(ytd_query)
    if date_logic:
        config["date_field"] = date_logic["field"]
        if "fallback_field" in date_logic:
            config["fallback_date_field"] = date_logic["fallback_field"]
            config["fallback_condition"] = date_logic["fallback_condition"]
    
    # Set location field
    config["location_field"] = "location"  # Standard for most DataSF datasets
    
    return config

def migrate_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a single metric to the new structure."""
    metric_id = metric["id"]
    metric_name = metric["metric_name"]
    endpoint = metric["endpoint"]
    metric_query = metric.get("metric_query", "")
    ytd_query = metric.get("ytd_query", "")
    
    logger.info(f"Migrating metric {metric_id}: {metric_name}")
    
    # Create map_query (simplified base query)
    map_query = create_map_query(endpoint, metric_query)
    
    # Extract and structure filters
    map_filters = {}
    
    # Extract geometry filter
    geometry_filter = extract_geometry_filter(metric_query)
    if geometry_filter:
        map_filters["geometry"] = geometry_filter
    
    # Extract date logic
    date_logic = extract_date_logic(metric_query)
    if date_logic:
        map_filters["date_range"] = date_logic
    
    # Extract static filters
    static_filters = extract_static_filters(metric_query)
    if static_filters:
        map_filters["static_filters"] = static_filters
    
    # Generate map config
    map_config = get_map_config(metric)
    
    return {
        "map_query": map_query,
        "map_filters": map_filters,
        "map_config": map_config
    }

def update_metric_in_db(conn, metric_id: int, map_data: Dict[str, Any]):
    """Update a metric with the new map data."""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE metrics 
            SET map_query = %s,
                map_filters = %s,
                map_config = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (
            map_data["map_query"],
            json.dumps(map_data["map_filters"]),
            json.dumps(map_data["map_config"]),
            metric_id
        ))
        
        cursor.close()
        logger.info(f"Updated metric {metric_id} with new map structure")
        
    except Exception as e:
        logger.error(f"Error updating metric {metric_id}: {str(e)}")
        raise

def main():
    """Main migration function."""
    # Connect to database
    conn = get_postgres_connection()
    if not conn:
        logger.error("Failed to connect to database")
        sys.exit(1)
    
    try:
        # Get all metrics that need migration
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, metric_name, metric_key, endpoint, metric_query, ytd_query, 
                   subcategory, item_noun, metadata
            FROM metrics 
            WHERE is_active = TRUE
            ORDER BY id
        """)
        
        metrics = cursor.fetchall()
        cursor.close()
        
        # Convert to list of dicts
        metric_dicts = []
        for metric in metrics:
            metric_dict = {
                "id": metric[0],
                "metric_name": metric[1],
                "metric_key": metric[2],
                "endpoint": metric[3],
                "metric_query": metric[4],
                "ytd_query": metric[5],
                "subcategory": metric[6],
                "item_noun": metric[7],
                "metadata": metric[8]
            }
            metric_dicts.append(metric_dict)
        
        logger.info(f"Found {len(metric_dicts)} metrics to migrate")
        
        # Process each metric
        for metric in metric_dicts:
            try:
                map_data = migrate_metric(metric)
                update_metric_in_db(conn, metric["id"], map_data)
            except Exception as e:
                logger.error(f"Failed to migrate metric {metric['id']}: {str(e)}")
                continue
        
        # Commit all changes
        conn.commit()
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        conn.rollback()
        sys.exit(1)
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()

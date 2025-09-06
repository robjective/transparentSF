#!/usr/bin/env python3
"""
Metrics management tool for TransparentSF.
Provides functions to query, edit, and add new metrics to the database.
"""

import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional, Union
from .db_utils import get_postgres_connection, execute_with_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_metrics(
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    active_only: bool = True,
    show_on_dash_only: bool = False
) -> Dict[str, Any]:
    """
    Retrieve all metrics from the database with optional filtering.
    
    Args:
        category: Filter by category (optional)
        subcategory: Filter by subcategory (optional)
        active_only: Only return active metrics (default: True)
        show_on_dash_only: Only return metrics shown on dashboard (default: False)
        
    Returns:
        dict: Result with status and metrics list
    """
    def get_metrics_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Build query with filters
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = %s"
            params.append(category)
            
        if subcategory:
            query += " AND subcategory = %s"
            params.append(subcategory)
            
        if active_only:
            query += " AND is_active = TRUE"
            
        if show_on_dash_only:
            query += " AND show_on_dash = TRUE"
            
        query += " ORDER BY category, subcategory, metric_name"
        
        cursor.execute(query, params)
        metrics = cursor.fetchall()
        
        # Convert to list of dictionaries
        result = []
        for metric in metrics:
            metric_dict = dict(metric)
            # Convert timestamps to strings
            if 'created_at' in metric_dict and metric_dict['created_at']:
                metric_dict['created_at'] = metric_dict['created_at'].isoformat()
            if 'updated_at' in metric_dict and metric_dict['updated_at']:
                metric_dict['updated_at'] = metric_dict['updated_at'].isoformat()
            result.append(metric_dict)
        
        cursor.close()
        return result
    
    result = execute_with_connection(operation=get_metrics_operation)
    
    if result["status"] == "success":
        return {
            "status": "success",
            "metrics": result["result"],
            "count": len(result["result"])
        }
    else:
        return result

def get_metric_by_id(metric_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific metric by ID.
    
    Args:
        metric_id: The ID of the metric to retrieve
        
    Returns:
        dict: Result with status and metric data
    """
    def get_metric_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM metrics WHERE id = %s", (metric_id,))
        metric = cursor.fetchone()
        
        if not metric:
            cursor.close()
            return None
            
        metric_dict = dict(metric)
        # Convert timestamps to strings
        if 'created_at' in metric_dict and metric_dict['created_at']:
            metric_dict['created_at'] = metric_dict['created_at'].isoformat()
        if 'updated_at' in metric_dict and metric_dict['updated_at']:
            metric_dict['updated_at'] = metric_dict['updated_at'].isoformat()
        
        cursor.close()
        return metric_dict
    
    result = execute_with_connection(operation=get_metric_operation)
    
    if result["status"] == "success":
        if result["result"] is None:
            return {
                "status": "error",
                "message": f"No metric found with ID {metric_id}"
            }
        return {
            "status": "success",
            "metric": result["result"]
        }
    else:
        return result

def get_metric_by_key(metric_key: str) -> Dict[str, Any]:
    """
    Retrieve a specific metric by its unique key.
    
    Args:
        metric_key: The unique key of the metric to retrieve
        
    Returns:
        dict: Result with status and metric data
    """
    def get_metric_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM metrics WHERE metric_key = %s", (metric_key,))
        metric = cursor.fetchone()
        
        if not metric:
            cursor.close()
            return None
            
        metric_dict = dict(metric)
        # Convert timestamps to strings
        if 'created_at' in metric_dict and metric_dict['created_at']:
            metric_dict['created_at'] = metric_dict['created_at'].isoformat()
        if 'updated_at' in metric_dict and metric_dict['updated_at']:
            metric_dict['updated_at'] = metric_dict['updated_at'].isoformat()
        
        cursor.close()
        return metric_dict
    
    result = execute_with_connection(operation=get_metric_operation)
    
    if result["status"] == "success":
        if result["result"] is None:
            return {
                "status": "error",
                "message": f"No metric found with key '{metric_key}'"
            }
        return {
            "status": "success",
            "metric": result["result"]
        }
    else:
        return result

def search_metrics(search_term: str) -> Dict[str, Any]:
    """
    Search metrics by name, summary, or definition.
    
    Args:
        search_term: Term to search for
        
    Returns:
        dict: Result with status and matching metrics
    """
    def search_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        search_pattern = f"%{search_term}%"
        query = """
            SELECT * FROM metrics 
            WHERE (metric_name ILIKE %s 
                   OR summary ILIKE %s 
                   OR definition ILIKE %s
                   OR category ILIKE %s
                   OR subcategory ILIKE %s)
            AND is_active = TRUE
            ORDER BY category, subcategory, metric_name
        """
        
        cursor.execute(query, (search_pattern, search_pattern, search_pattern, search_pattern, search_pattern))
        metrics = cursor.fetchall()
        
        # Convert to list of dictionaries
        result = []
        for metric in metrics:
            metric_dict = dict(metric)
            # Convert timestamps to strings
            if 'created_at' in metric_dict and metric_dict['created_at']:
                metric_dict['created_at'] = metric_dict['created_at'].isoformat()
            if 'updated_at' in metric_dict and metric_dict['updated_at']:
                metric_dict['updated_at'] = metric_dict['updated_at'].isoformat()
            result.append(metric_dict)
        
        cursor.close()
        return result
    
    result = execute_with_connection(operation=search_operation)
    
    if result["status"] == "success":
        return {
            "status": "success",
            "metrics": result["result"],
            "count": len(result["result"]),
            "search_term": search_term
        }
    else:
        return result

def add_metric(metric_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new metric to the database.
    
    Args:
        metric_data: Dictionary containing metric information
        
    Returns:
        dict: Result with status and new metric ID
    """
    required_fields = ['metric_name', 'metric_key', 'category', 'endpoint']
    
    # Validate required fields
    for field in required_fields:
        if field not in metric_data or not metric_data[field]:
            return {
                "status": "error",
                "message": f"Required field '{field}' is missing or empty"
            }
    
    def add_metric_operation(connection):
        cursor = connection.cursor()
        
        # Check if metric_key already exists
        cursor.execute("SELECT id FROM metrics WHERE metric_key = %s", (metric_data['metric_key'],))
        if cursor.fetchone():
            cursor.close()
            raise ValueError(f"Metric with key '{metric_data['metric_key']}' already exists")
        
        insert_query = """
            INSERT INTO metrics (
                metric_name, metric_key, category, subcategory, endpoint,
                summary, definition, data_sf_url, ytd_query, metric_query,
                dataset_title, dataset_category, show_on_dash, item_noun,
                greendirection, location_fields, category_fields, metadata, is_active
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """
        
        cursor.execute(insert_query, (
            metric_data['metric_name'],
            metric_data['metric_key'],
            metric_data['category'],
            metric_data.get('subcategory', ''),
            metric_data['endpoint'],
            metric_data.get('summary', ''),
            metric_data.get('definition', ''),
            metric_data.get('data_sf_url', ''),
            metric_data.get('ytd_query', ''),
            metric_data.get('metric_query', ''),
            metric_data.get('dataset_title', ''),
            metric_data.get('dataset_category', ''),
            metric_data.get('show_on_dash', True),
            metric_data.get('item_noun', 'Items'),
            metric_data.get('greendirection', 'up'),
            json.dumps(metric_data.get('location_fields', [])),
            json.dumps(metric_data.get('category_fields', [])),
            json.dumps(metric_data.get('metadata', {})),
            metric_data.get('is_active', True)
        ))
        
        new_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
        return new_id
    
    result = execute_with_connection(operation=add_metric_operation)
    
    if result["status"] == "success":
        return {
            "status": "success",
            "message": f"Successfully added metric '{metric_data['metric_name']}'",
            "metric_id": result["result"]
        }
    else:
        return result

def update_metric(metric_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing metric.
    
    Args:
        metric_id: ID of the metric to update
        updates: Dictionary containing fields to update
        
    Returns:
        dict: Result with status and message
    """
    if not updates:
        return {
            "status": "error",
            "message": "No updates provided"
        }
    
    def update_metric_operation(connection):
        cursor = connection.cursor()
        
        # Check if metric exists
        cursor.execute("SELECT id FROM metrics WHERE id = %s", (metric_id,))
        if not cursor.fetchone():
            cursor.close()
            raise ValueError(f"No metric found with ID {metric_id}")
        
        # Build update query dynamically
        set_clauses = []
        params = []
        
        for field, value in updates.items():
            # Map field names to actual DB columns or skip disallowed fields
            if field == 'name':
                set_clauses.append("metric_name = %s")
                params.append(value)
            elif field == 'key':
                # Do NOT allow changing the unique key
                continue
            elif field in ['location_fields', 'category_fields', 'metadata', 'map_filters', 'map_config']:
                set_clauses.append(f"{field} = %s")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{field} = %s")
                params.append(value)
        
        params.append(metric_id)
        
        update_query = f"""
            UPDATE metrics 
            SET {', '.join(set_clauses)}
            WHERE id = %s
        """
        
        cursor.execute(update_query, params)
        connection.commit()
        cursor.close()
        return cursor.rowcount
    
    result = execute_with_connection(operation=update_metric_operation)
    
    if result["status"] == "success":
        return {
            "status": "success",
            "message": f"Successfully updated metric with ID {metric_id}",
            "rows_affected": result["result"]
        }
    else:
        return result

def deactivate_metric(metric_id: int) -> Dict[str, Any]:
    """
    Deactivate a metric (soft delete).
    
    Args:
        metric_id: ID of the metric to deactivate
        
    Returns:
        dict: Result with status and message
    """
    return update_metric(metric_id, {"is_active": False})

def activate_metric(metric_id: int) -> Dict[str, Any]:
    """
    Activate a metric.
    
    Args:
        metric_id: ID of the metric to activate
        
    Returns:
        dict: Result with status and message
    """
    return update_metric(metric_id, {"is_active": True})

def get_categories() -> Dict[str, Any]:
    """
    Get all unique categories and subcategories.
    
    Returns:
        dict: Result with status and categories
    """
    def get_categories_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT category, subcategory, COUNT(*) as metric_count
            FROM metrics 
            WHERE is_active = TRUE
            GROUP BY category, subcategory
            ORDER BY category, subcategory
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Organize by category
        categories = {}
        for row in results:
            category = row['category']
            subcategory = row['subcategory']
            count = row['metric_count']
            
            if category not in categories:
                categories[category] = {
                    'subcategories': {},
                    'total_metrics': 0
                }
            
            categories[category]['subcategories'][subcategory] = count
            categories[category]['total_metrics'] += count
        
        cursor.close()
        return categories
    
    result = execute_with_connection(operation=get_categories_operation)
    
    if result["status"] == "success":
        return {
            "status": "success",
            "categories": result["result"]
        }
    else:
        return result

def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of metrics in the database.
    
    Returns:
        dict: Result with status and summary statistics
    """
    def get_summary_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Get total counts
        cursor.execute("SELECT COUNT(*) as total FROM metrics")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as active FROM metrics WHERE is_active = TRUE")
        active = cursor.fetchone()['active']
        
        cursor.execute("SELECT COUNT(*) as dashboard FROM metrics WHERE show_on_dash = TRUE AND is_active = TRUE")
        dashboard = cursor.fetchone()['dashboard']
        
        # Get category breakdown
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM metrics 
            WHERE is_active = TRUE
            GROUP BY category
            ORDER BY count DESC
        """)
        category_breakdown = cursor.fetchall()

        # Fetch basic detail list for each metric (name, endpoint, queries)
        cursor.execute(
            """
            SELECT 
                id,
                metric_name, 
                endpoint, 
                metric_query, 
                ytd_query
            FROM metrics
            WHERE is_active = TRUE
            ORDER BY metric_name
            """
        )
        metric_details = cursor.fetchall()

        cursor.close()
        return {
            'total_metrics': total,
            'active_metrics': active,
            'dashboard_metrics': dashboard,
            'inactive_metrics': total - active,
            'category_breakdown': [dict(row) for row in category_breakdown],
            'metric_details': [dict(row) for row in metric_details]
        }
    
    result = execute_with_connection(operation=get_summary_operation)
    
    if result["status"] == "success":
        return {
            "status": "success",
            "summary": result["result"]
        }
    else:
        return result

def search_metrics_by_endpoint(endpoint: str) -> Dict[str, Any]:
    """
    Search metrics by DataSF endpoint.
    
    Args:
        endpoint: The DataSF endpoint identifier to search for
        
    Returns:
        dict: Result with status and matching metrics
    """
    def search_operation(connection):
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT * FROM metrics 
            WHERE endpoint = %s 
            AND is_active = TRUE
            ORDER BY category, subcategory, metric_name
        """
        
        cursor.execute(query, (endpoint,))
        metrics = cursor.fetchall()
        
        # Convert to list of dicts
        metrics_list = [dict(metric) for metric in metrics]
        
        return {
            "status": "success",
            "count": len(metrics_list),
            "metrics": metrics_list
        }
    
    result = execute_with_connection(search_operation)
    
    if result["status"] == "success":
        return result["result"]
    else:
        return result

# Convenience functions for common operations
def list_all_metrics():
    """List all active metrics with basic info."""
    result = get_all_metrics(active_only=True)
    if result["status"] == "success":
        print(f"Found {result['count']} active metrics:")
        for metric in result["metrics"]:
            print(f"  {metric['id']}: {metric['metric_name']} ({metric['category']}/{metric['subcategory']})")
    else:
        print(f"Error: {result['message']}")

def show_metric_details(metric_id: int):
    """Show detailed information about a metric."""
    result = get_metric_by_id(metric_id)
    if result["status"] == "success":
        metric = result["metric"]
        print(f"Metric ID: {metric['id']}")
        print(f"Name: {metric['metric_name']}")
        print(f"Key: {metric['metric_key']}")
        print(f"Category: {metric['category']}/{metric['subcategory']}")
        print(f"Endpoint: {metric['endpoint']}")
        print(f"Summary: {metric['summary']}")
        print(f"Active: {metric['is_active']}")
        print(f"Show on Dashboard: {metric['show_on_dash']}")
    else:
        print(f"Error: {result['message']}")

if __name__ == "__main__":
    # Example usage
    print("Metrics Manager - Example Usage")
    print("=" * 40)
    
    # Get summary
    summary = get_metrics_summary()
    if summary["status"] == "success":
        print(f"Total metrics: {summary['summary']['total_metrics']}")
        print(f"Active metrics: {summary['summary']['active_metrics']}")
        print(f"Dashboard metrics: {summary['summary']['dashboard_metrics']}")
    
    # List categories
    categories = get_categories()
    if categories["status"] == "success":
        print("\nCategories:")
        for cat, data in categories["categories"].items():
            print(f"  {cat}: {data['total_metrics']} metrics") 
#!/usr/bin/env python3
"""
Simplified metrics tools for the explainer agent.
Provides easy-to-use functions for querying and managing metrics.
"""

import json
from typing import Dict, List, Any, Optional
from .metrics_manager import (
    get_all_metrics, get_metric_by_id, get_metric_by_key, search_metrics,
    add_metric, update_metric, deactivate_metric, activate_metric,
    get_categories, get_metrics_summary
)

def query_metrics(
    context_variables: Dict[str, Any],
    category: str = None,
    search_term: str = None,
    active_only: bool = True,
    dashboard_only: bool = False
) -> Dict[str, Any]:
    """
    Query metrics with various filters. This is the main function for finding metrics.
    
    Args:
        category: Filter by category (e.g., 'crime', 'safety', 'economy')
        search_term: Search in metric names, summaries, and definitions
        active_only: Only return active metrics (default: True)
        dashboard_only: Only return metrics shown on dashboard (default: False)
        
    Returns:
        dict: Result with status and list of matching metrics
        
    Example:
        # Find all crime metrics
        result = query_metrics(category="crime")
        
        # Search for police-related metrics
        result = query_metrics(search_term="police")
        
        # Get all dashboard metrics
        result = query_metrics(dashboard_only=True)
    """
    if search_term:
        return search_metrics(search_term)
    else:
        return get_all_metrics(
            category=category,
            active_only=active_only,
            show_on_dash_only=dashboard_only
        )

def get_metric_details(context_variables: Dict[str, Any], metric_identifier: Any) -> Dict[str, Any]:
    """
    Get detailed information about a specific metric.
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
        
    Returns:
        dict: Result with status and metric details
        
    Example:
        # Get metric by ID
        result = get_metric_details(1)
        
        # Get metric by key
        result = get_metric_details("total_police_incidents")
    """
    # Try to convert to int if it's a numeric string
    if isinstance(metric_identifier, str) and metric_identifier.isdigit():
        metric_identifier = int(metric_identifier)
    
    if isinstance(metric_identifier, int):
        return get_metric_by_id(metric_identifier)
    else:
        return get_metric_by_key(str(metric_identifier))

def list_categories(context_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get all available metric categories and subcategories.
    
    Returns:
        dict: Result with status and categories structure
        
    Example:
        result = list_categories()
        if result["status"] == "success":
            for category, data in result["categories"].items():
                print(f"{category}: {data['total_metrics']} metrics")
    """
    return get_categories()

def get_dashboard_metrics(context_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get all metrics that are shown on the dashboard.
    
    Returns:
        dict: Result with status and list of dashboard metrics
    """
    return get_all_metrics(active_only=True, show_on_dash_only=True)

def create_new_metric(
    context_variables: Dict[str, Any],
    name: str,
    key: str,
    category: str,
    endpoint: str,
    subcategory: str = "",
    summary: str = "",
    definition: str = "",
    data_sf_url: str = "",
    ytd_query: str = "",
    metric_query: str = "",
    dataset_title: str = "",
    dataset_category: str = "",
    show_on_dash: bool = True,
    item_noun: str = "Items",
    greendirection: str = "up",
    location_fields: List[Dict] = None,
    category_fields: List[Dict] = None,
    metadata: Dict = None
) -> Dict[str, Any]:
    """
    Create a new metric in the database.
    
    Args:
        name: Display name for the metric
        key: Unique identifier key (no spaces, lowercase recommended)
        category: Main category (e.g., 'crime', 'safety', 'economy')
        endpoint: DataSF endpoint ID
        subcategory: Subcategory (optional)
        summary: Brief description
        definition: Detailed definition
        data_sf_url: URL to the DataSF dataset
        ytd_query: Year-to-date query for time series
        metric_query: Query for metric calculations
        dataset_title: Title of the source dataset
        dataset_category: Category of the source dataset
        show_on_dash: Whether to show on dashboard (default: True)
        item_noun: Noun for items (e.g., "Cases", "Incidents")
        greendirection: Direction indicator ('up', 'down', 'neutral')
        location_fields: List of location field definitions
        category_fields: List of category field definitions
        metadata: Additional metadata
        
    Returns:
        dict: Result with status and new metric ID
        
    Example:
        result = create_new_metric(
            name="🚗 Vehicle Thefts",
            key="vehicle_thefts",
            category="crime",
            subcategory="property",
            endpoint="wg3w-h783",
            summary="Count of reported vehicle theft incidents",
            definition="Vehicle thefts include all reported incidents...",
            greendirection="down",
            show_on_dash=True
        )
    """
    metric_data = {
        'metric_name': name,
        'metric_key': key,
        'category': category,
        'subcategory': subcategory,
        'endpoint': endpoint,
        'summary': summary,
        'definition': definition,
        'data_sf_url': data_sf_url,
        'ytd_query': ytd_query,
        'metric_query': metric_query,
        'dataset_title': dataset_title,
        'dataset_category': dataset_category,
        'show_on_dash': show_on_dash,
        'item_noun': item_noun,
        'greendirection': greendirection,
        'location_fields': location_fields or [],
        'category_fields': category_fields or [],
        'metadata': metadata or {}
    }
    
    return add_metric(metric_data)

def edit_metric(
    context_variables: Dict[str, Any],
    metric_identifier: Any,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing metric.
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
        updates: Dictionary of fields to update, or JSON string representing dictionary
        
    Returns:
        dict: Result with status and message
        
    Example:
        # Update metric summary
        result = edit_metric(1, {"summary": "Updated summary text"})
        
        # Update multiple fields
        result = edit_metric("police_incidents", {
            "summary": "New summary",
            "show_on_dash": False,
            "definition": "Updated definition"
        })
    """
    # Handle case where updates is passed as a JSON string
    if isinstance(updates, str):
        try:
            updates = json.loads(updates)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Invalid JSON in updates parameter: {str(e)}"
            }
    
    # Ensure updates is a dictionary
    if not isinstance(updates, dict):
        return {
            "status": "error",
            "message": "Updates parameter must be a dictionary or valid JSON string"
        }
    
    # First get the metric ID if a key was provided
    if isinstance(metric_identifier, str):
        metric_result = get_metric_by_key(metric_identifier)
        if metric_result["status"] != "success":
            return metric_result
        metric_id = metric_result["metric"]["id"]
    else:
        metric_id = metric_identifier
    
    return update_metric(metric_id, updates)

def disable_metric(context_variables: Dict[str, Any], metric_identifier: Any) -> Dict[str, Any]:
    """
    Disable a metric (soft delete - keeps data but marks as inactive).
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
        
    Returns:
        dict: Result with status and message
        
    Example:
        result = disable_metric("old_metric_key")
    """
    # First get the metric ID if a key was provided
    if isinstance(metric_identifier, str):
        metric_result = get_metric_by_key(metric_identifier)
        if metric_result["status"] != "success":
            return metric_result
        metric_id = metric_result["metric"]["id"]
    else:
        metric_id = metric_identifier
    
    return deactivate_metric(metric_id)

def enable_metric(context_variables: Dict[str, Any], metric_identifier: Any) -> Dict[str, Any]:
    """
    Enable a previously disabled metric.
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
        
    Returns:
        dict: Result with status and message
        
    Example:
        result = enable_metric("metric_key")
    """
    # First get the metric ID if a key was provided
    if isinstance(metric_identifier, str):
        metric_result = get_metric_by_key(metric_identifier)
        if metric_result["status"] != "success":
            return metric_result
        metric_id = metric_result["metric"]["id"]
    else:
        metric_id = metric_identifier
    
    return activate_metric(metric_id)

def get_metrics_overview(context_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a high-level overview of all metrics in the system.
    
    Returns:
        dict: Result with status and summary statistics
        
    Example:
        result = get_metrics_overview()
        if result["status"] == "success":
            summary = result["summary"]
            print(f"Total metrics: {summary['total_metrics']}")
    """
    return get_metrics_summary()

def find_metrics_by_endpoint(context_variables: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """
    Find all metrics that use a specific DataSF endpoint.
    
    Args:
        endpoint: The DataSF endpoint ID
        
    Returns:
        dict: Result with status and list of matching metrics
        
    Example:
        result = find_metrics_by_endpoint("wg3w-h783")
    """
    all_metrics = get_all_metrics(active_only=False)
    if all_metrics["status"] != "success":
        return all_metrics
    
    matching_metrics = [
        metric for metric in all_metrics["metrics"]
        if metric.get("endpoint") == endpoint
    ]
    
    return {
        "status": "success",
        "metrics": matching_metrics,
        "count": len(matching_metrics),
        "endpoint": endpoint
    }

def get_crime_metrics(context_variables: Dict[str, Any]) -> Dict[str, Any]:
    """Get all crime-related metrics."""
    return query_metrics(context_variables, category="crime")

def get_safety_metrics(context_variables: Dict[str, Any]) -> Dict[str, Any]:
    """Get all safety-related metrics."""
    return query_metrics(context_variables, category="safety")

def get_economy_metrics(context_variables: Dict[str, Any]) -> Dict[str, Any]:
    """Get all economy-related metrics."""
    return query_metrics(context_variables, category="economy")

# Helper function to format metric information for display
def format_metric_info(metric: Dict[str, Any]) -> str:
    """
    Format metric information for easy reading.
    
    Args:
        metric: Metric dictionary
        
    Returns:
        str: Formatted metric information
    """
    info = f"ID: {metric['id']}\n"
    info += f"Name: {metric['metric_name']}\n"
    info += f"Key: {metric['metric_key']}\n"
    info += f"Category: {metric['category']}"
    if metric.get('subcategory'):
        info += f"/{metric['subcategory']}"
    info += f"\nEndpoint: {metric['endpoint']}\n"
    info += f"Summary: {metric.get('summary', 'No summary available')}\n"
    info += f"Active: {metric['is_active']}\n"
    info += f"Show on Dashboard: {metric['show_on_dash']}\n"
    if metric.get('data_sf_url'):
        info += f"DataSF URL: {metric['data_sf_url']}\n"
    
    return info

def display_metrics_list(metrics: List[Dict[str, Any]]) -> str:
    """
    Format a list of metrics for display.
    
    Args:
        metrics: List of metric dictionaries
        
    Returns:
        str: Formatted list of metrics
    """
    if not metrics:
        return "No metrics found."
    
    output = f"Found {len(metrics)} metrics:\n\n"
    for i, metric in enumerate(metrics, 1):
        output += f"{i}. {metric['metric_name']}\n"
        output += f"   ID: {metric['id']}, Key: {metric['metric_key']}\n"
        output += f"   Category: {metric['category']}"
        if metric.get('subcategory'):
            output += f"/{metric['subcategory']}"
        output += f"\n   Summary: {metric.get('summary', 'No summary')[:100]}...\n\n"
    
    return output

# Example usage and testing functions
if __name__ == "__main__":
    print("Explainer Metrics Tools - Example Usage")
    print("=" * 50)
    
    # Get overview
    overview = get_metrics_overview()
    if overview["status"] == "success":
        summary = overview["summary"]
        print(f"Total metrics: {summary['total_metrics']}")
        print(f"Active metrics: {summary['active_metrics']}")
        print(f"Dashboard metrics: {summary['dashboard_metrics']}")
    
    print("\nCategories:")
    categories = list_categories()
    if categories["status"] == "success":
        for cat, data in categories["categories"].items():
            print(f"  {cat}: {data['total_metrics']} metrics")
    
    print("\nCrime metrics:")
    crime_metrics = get_crime_metrics()
    if crime_metrics["status"] == "success":
        for metric in crime_metrics["metrics"][:3]:  # Show first 3
            print(f"  - {metric['metric_name']}") 
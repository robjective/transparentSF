#!/usr/bin/env python3
"""
Simplified metrics tools for the explainer agent, adapted for LangChain.
Provides easy-to-use functions for querying and managing metrics.
"""
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

# Add grandparent directory to sys.path for module resolution
current_dir = Path(__file__).parent
grandparent_dir = current_dir.parent.parent
sys.path.insert(0, str(grandparent_dir))

from tools.metrics_manager import (
    get_all_metrics, get_metric_by_id, get_metric_by_key, search_metrics,
    add_metric, update_metric, deactivate_metric, activate_metric,
    get_categories, get_metrics_summary
)

def query_metrics(
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
    """
    if search_term:
        return search_metrics(search_term)
    else:
        return get_all_metrics(
            category=category,
            active_only=active_only,
            show_on_dash_only=dashboard_only
        )

def get_metric_details(metric_identifier: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific metric.
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
        
    Returns:
        dict: Result with status and metric details
    """
    if isinstance(metric_identifier, str) and metric_identifier.isdigit():
        metric_identifier = int(metric_identifier)
    
    if isinstance(metric_identifier, int):
        return get_metric_by_id(metric_identifier)
    else:
        return get_metric_by_key(str(metric_identifier))



def get_dashboard_metrics() -> Dict[str, Any]:
    """
    Get all metrics that are shown on the dashboard.
    
    Returns:
        dict: Result with status and list of dashboard metrics
    """
    return get_all_metrics(active_only=True, show_on_dash_only=True)

def create_new_metric(
    name: str,
    key: str,
    category: str,
    endpoint: str,
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
    subcategory: str = ""
) -> Dict[str, Any]:
    """
    Create a new metric in the database.
    
    Args:
        name: Display name for the metric
        key: Unique identifier for the metric
        category: Category of the metric (e.g., 'crime', 'safety', 'economy')
        endpoint: DataSF endpoint identifier
        summary: Brief summary of the metric
        definition: Detailed definition of the metric
        data_sf_url: URL to the DataSF dataset
        ytd_query: Query for year-to-date calculations
        metric_query: Query for current period calculations
        dataset_title: Title of the dataset
        dataset_category: Category of the dataset
        show_on_dash: Whether to show on dashboard
        item_noun: Noun for the items being counted
        greendirection: Direction for green indicators ('up' or 'down')
        subcategory: Subcategory of the metric
        
    Returns:
        dict: Result with status and metric details
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
        'location_fields': [],
        'category_fields': [],
        'metadata': {}
    }
    
    return add_metric(metric_data)

def edit_metric(metric_identifier: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing metric.
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
        updates: Dictionary of fields to update, or JSON string representing dictionary
        
    Returns:
        dict: Result with status and message
    """
    if isinstance(updates, str):
        try:
            updates = json.loads(updates)
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Invalid JSON in updates: {e}"}
    
    if not isinstance(updates, dict):
        return {"status": "error", "message": "Updates must be a dictionary or JSON string"}
    
    if isinstance(metric_identifier, str) and metric_identifier.isdigit():
        metric_id = int(metric_identifier)
    elif isinstance(metric_identifier, int):
        metric_id = metric_identifier
    else:
        metric_result = get_metric_by_key(str(metric_identifier))
        if metric_result["status"] != "success":
            return metric_result
        metric_id = metric_result["metric"]["id"]
    
    return update_metric(metric_id, updates)

def disable_metric(metric_identifier: str) -> Dict[str, Any]:
    """
    Disable a metric (soft delete).
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
    """
    if isinstance(metric_identifier, str) and metric_identifier.isdigit():
        metric_id = int(metric_identifier)
    elif isinstance(metric_identifier, int):
        metric_id = metric_identifier
    else:
        metric_result = get_metric_by_key(str(metric_identifier))
        if metric_result["status"] != "success":
            return metric_result
        metric_id = metric_result["metric"]["id"]
        
    return deactivate_metric(metric_id)

def enable_metric(metric_identifier: str) -> Dict[str, Any]:
    """
    Enable a metric.
    
    Args:
        metric_identifier: Either metric ID (int) or metric key (str)
    """
    if isinstance(metric_identifier, str) and metric_identifier.isdigit():
        metric_id = int(metric_identifier)
    elif isinstance(metric_identifier, int):
        metric_id = metric_identifier
    else:
        metric_result = get_metric_by_key(str(metric_identifier))
        if metric_result["status"] != "success":
            return metric_result
        metric_id = metric_result["metric"]["id"]
        
    return activate_metric(metric_id)

def get_metrics_overview() -> Dict[str, Any]:
    """
    Get a summary overview of all metrics.
    """
    return get_metrics_summary()

def find_metrics_by_endpoint(endpoint: str) -> Dict[str, Any]:
    """
    Find all metrics using a specific DataSF endpoint.
    
    Args:
        endpoint: The DataSF endpoint identifier to search for
        
    Returns:
        dict: Result with status and list of matching metrics
    """
    return search_metrics_by_endpoint(endpoint)

 
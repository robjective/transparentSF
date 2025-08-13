"""
Clean, properly designed tools for the LangChain explainer agent.
These tools are designed to work with StructuredTool and use named arguments.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import pandas as pd
import requests
from urllib.parse import urljoin

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

logger = logging.getLogger(__name__)

def set_dataset_tool(endpoint: str, query: str) -> Dict[str, Any]:
    """
    Set dataset for analysis by querying DataSF.
    
    Args:
        endpoint: The dataset identifier WITHOUT the .json extension (e.g., 'ubvf-ztfx')
        query: The complete SoQL query string using standard SQL syntax
        
    Returns:
        Dictionary with status, data, and optional error message
    """
    logger.info("=== Starting set_dataset_tool ===")
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Query: {query}")

    try:
        # Validate required parameters
        if not endpoint:
            logger.error("Missing endpoint parameter")
            return {
                'status': 'error',
                'error': 'Endpoint is required', 
                'queryURL': None,
                'error_type': 'validation_error'
            }
        if not query:
            logger.error("Missing query parameter")
            return {
                'status': 'error',
                'error': 'Query is required', 
                'queryURL': None,
                'error_type': 'validation_error'
            }
            
        # Clean up endpoint - ensure it ends with .json
        if not endpoint.endswith('.json'):
            endpoint = f"{endpoint}.json"
            logger.info(f"Added .json to endpoint: {endpoint}")

        # Import the original function and call it
        from ai.tools.data_fetcher import fetch_data_from_api
        
        query_object = {'endpoint': endpoint, 'query': query}
        result = fetch_data_from_api(query_object)
        logger.info(f"API result status: {'success' if 'data' in result else 'error'}")
        
        if result and 'data' in result:
            data = result['data']
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Dataset successfully created with shape: {df.shape}")
                return {
                    'status': 'success', 
                    'data': df.to_dict('records'),
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'queryURL': result.get('queryURL')
                }
            else:
                logger.warning("API returned empty data")
                return {
                    'status': 'error',
                    'error': 'No data returned from the API', 
                    'queryURL': result.get('queryURL'),
                    'error_type': 'empty_data'
                }
        elif 'error' in result:
            logger.error(f"API returned error: {result['error']}")
            return {
                'status': 'error',
                'error': result['error'], 
                'queryURL': result.get('queryURL'),
                'error_type': 'api_error'
            }
        else:
            logger.error("Unexpected API response format")
            return {
                'status': 'error',
                'error': 'Unexpected API response format', 
                'queryURL': result.get('queryURL'),
                'error_type': 'unexpected_format'
            }
            
    except Exception as e:
        logger.exception("Unexpected error in set_dataset_tool")
        return {
            'status': 'error',
            'error': f'Unexpected error: {str(e)}', 
            'queryURL': None,
            'error_type': 'unexpected_error'
        }


def query_docs_tool(collection_name: str, query: str) -> Dict[str, Any]:
    """
    Search for additional context in documentation.
    
    Args:
        collection_name: The name of the document collection to search (e.g., "SFPublicData")
        query: The search query string describing what information you're looking for
        
    Returns:
        Dictionary with search results
    """
    logger.info("=== Starting query_docs_tool ===")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Query: {query}")
    
    try:
        # Import the original function and call it
        from ai.tools.vector_query import query_docs
        
        # Call the original function with the correct parameters
        # The original function expects (context_variables, collection_name, query)
        # Since we removed context_variables, we'll pass an empty dict
        result = query_docs({}, collection_name, query)
        
        logger.info(f"Query completed successfully")
        return {'status': 'success', 'results': result}
        
    except Exception as e:
        logger.exception("Error in query_docs_tool")
        return {'error': f'Error querying documentation: {str(e)}'}

def get_notes_tool() -> Dict[str, Any]:
    """
    Get summary of available analysis and documentation.
    
    Returns:
        Dictionary with notes information
    """
    logger.info("=== Starting get_notes_tool ===")
    
    try:
        # Import the original function and call it
        from ai.tools.notes_manager import get_notes
        
        # Call the original function with empty context
        result = get_notes({})
        
        logger.info("Notes retrieved successfully")
        return {'status': 'success', 'notes': result}
        
    except Exception as e:
        logger.exception("Error in get_notes_tool")
        return {'error': f'Error retrieving notes: {str(e)}'}

def get_dashboard_metric_tool(district_number: int = 0, metric_id: int = None) -> Dict[str, Any]:
    """
    Retrieve dashboard metric data containing anomalies.
    
    Args:
        district_number: District number (0 for citywide)
        metric_id: Metric ID to retrieve
        
    Returns:
        Dictionary with metric data
    """
    logger.info("=== Starting get_dashboard_metric_tool ===")
    logger.info(f"District: {district_number}")
    logger.info(f"Metric ID: {metric_id}")
    
    try:
        # Import the original function and call it
        from ai.tools.dashboard_metric_tool import get_dashboard_metric
        
        # Call the original function with empty context
        result = get_dashboard_metric({}, district_number=district_number, metric_id=metric_id)
        
        logger.info("Dashboard metric retrieved successfully")
        return {'status': 'success', 'metric_data': result}
        
    except Exception as e:
        logger.exception("Error in get_dashboard_metric_tool")
        return {'error': f'Error retrieving dashboard metric: {str(e)}'}

def query_anomalies_db_tool(query_type: str = 'by_metric_id', metric_id: int = None, district_filter: int = None, only_anomalies: bool = True) -> Dict[str, Any]:
    """
    Query anomalies directly from the PostgreSQL database.
    
    Args:
        query_type: Type of query ('by_metric_id', etc.)
        metric_id: Metric ID to query
        district_filter: District filter
        only_anomalies: Whether to return only anomalies
        
    Returns:
        Dictionary with anomaly data
    """
    logger.info("=== Starting query_anomalies_db_tool ===")
    logger.info(f"Query type: {query_type}")
    logger.info(f"Metric ID: {metric_id}")
    logger.info(f"District filter: {district_filter}")
    logger.info(f"Only anomalies: {only_anomalies}")
    
    try:
        # Import the original function and call it
        from ai.anomalyAnalyzer import query_anomalies_db
        
        # Call the original function with empty context
        result = query_anomalies_db({}, query_type=query_type, metric_id=metric_id, district_filter=district_filter, only_anomalies=only_anomalies)
        
        logger.info("Anomalies query completed successfully")
        return {'status': 'success', 'anomalies': result}
        
    except Exception as e:
        logger.exception("Error in query_anomalies_db_tool")
        return {'error': f'Error querying anomalies: {str(e)}'}

def get_anomaly_details_tool(anomaly_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific anomaly by ID.
    
    Args:
        anomaly_id: ID of the anomaly to retrieve
        
    Returns:
        Dictionary with anomaly details
    """
    logger.info("=== Starting get_anomaly_details_tool ===")
    logger.info(f"Anomaly ID: {anomaly_id}")
    
    try:
        # Import the original function and call it
        from ai.anomalyAnalyzer import get_anomaly_details
        
        # Call the original function with empty context
        result = get_anomaly_details({}, anomaly_id=anomaly_id)
        
        logger.info("Anomaly details retrieved successfully")
        return {'status': 'success', 'anomaly_details': result}
        
    except Exception as e:
        logger.exception("Error in get_anomaly_details_tool")
        return {'error': f'Error retrieving anomaly details: {str(e)}'}

def get_dataset_columns_tool(endpoint: str) -> Dict[str, Any]:
    """
    Get column information for a dataset endpoint.
    
    Args:
        endpoint: Dataset endpoint to query
        
    Returns:
        Dictionary with column information
    """
    logger.info("=== Starting get_dataset_columns_tool ===")
    logger.info(f"Endpoint: {endpoint}")
    
    try:
        # Import the original function and call it
        from ai.anomalyAnalyzer import get_dataset_columns
        
        # Call the original function with empty context
        result = get_dataset_columns({}, endpoint=endpoint)
        
        logger.info("Dataset columns retrieved successfully")
        return {'status': 'success', 'columns': result}
        
    except Exception as e:
        logger.exception("Error in get_dataset_columns_tool")
        return {'error': f'Error retrieving dataset columns: {str(e)}'}

def get_charts_for_review_tool(limit: int = 20, days_back: int = 30, district_filter: str = None, metric_id: str = None) -> Dict[str, Any]:
    """
    Get available charts for newsletter inclusion review.
    
    Args:
        limit: Maximum number of charts to return
        days_back: Number of days to look back
        district_filter: District filter
        metric_id: Filter by specific metric ID/object_id (optional)
        
    Returns:
        Dictionary with chart information
    """
    logger.info("=== Starting get_charts_for_review_tool ===")
    logger.info(f"Limit: {limit}")
    logger.info(f"Days back: {days_back}")
    logger.info(f"District filter: {district_filter}")
    logger.info(f"Metric ID: {metric_id}")
    
    try:
        # Import the original function and call it
        from ai.tools.get_charts_for_review import get_charts_for_review
        
        # Call the original function with empty context
        result = get_charts_for_review({}, limit=limit, days_back=days_back, district_filter=district_filter, metric_id=metric_id)
        
        logger.info("Charts for review retrieved successfully")
        return {'status': 'success', 'charts': result}
        
    except Exception as e:
        logger.exception("Error in get_charts_for_review_tool")
        return {'error': f'Error retrieving charts for review: {str(e)}'} 
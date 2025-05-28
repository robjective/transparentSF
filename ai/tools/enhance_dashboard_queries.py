#!/usr/bin/env python3
import json
import os
import re
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor
from tools.db_utils import get_postgres_connection

# Configure logging
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "enhance_dashboard_queries.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    logger.debug(f"Loading JSON file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logger.debug(f"Successfully loaded JSON file: {file_path}")
            return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def save_json_file(data, file_path):
    """Save data to a JSON file with pretty formatting."""
    logger.debug(f"Saving JSON file: {file_path}")
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            logger.debug(f"Successfully saved JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise

def get_dataset_info_from_db(endpoint_id):
    """Get dataset information from the PostgreSQL database."""
    logger.info(f"Looking for dataset in database: {endpoint_id}")
    
    try:
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to connect to database")
            return None
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Query the datasets table
        cursor.execute("""
            SELECT endpoint, title, category, description, columns
            FROM datasets 
            WHERE endpoint = %s AND is_active = true
        """, (endpoint_id,))
        
        dataset_row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not dataset_row:
            logger.warning(f"Dataset not found in database for endpoint {endpoint_id}")
            return None
        
        logger.debug(f"Loaded dataset info from database for {endpoint_id}")
        
        # Extract location and category fields
        location_fields = []
        category_fields = []
        
        columns = dataset_row['columns'] or []
        logger.debug(f"Found {len(columns)} columns in dataset {endpoint_id}")
        
        for column in columns:
            field_name = column.get("fieldName", "").lower() if column.get("fieldName") else ""
            description = column.get("description", "").lower() if column.get("description") else ""
            name = column.get("name", "")
            
            logger.debug(f"Processing column: {name} (fieldName: {field_name})")
            
            # Only include supervisor_district as a location field
            if field_name == "supervisor_district":
                logger.debug(f"Found location field: {name}")
                location_fields.append({
                    "name": name,
                    "fieldName": column.get("fieldName", ""),
                    "description": column.get("description", "")
                })
            
            # Identify category fields, excluding neighborhood/district fields
            if (any(term in field_name for term in ["category", "type", "class", "group", "status", "subcategory"]) or \
                "categor" in description or "classif" in description or "type of" in description) and \
                not any(term in field_name for term in ["neighborhood", "district"]):
                logger.debug(f"Found category field: {name}")
                category_fields.append({
                    "name": name,
                    "fieldName": column.get("fieldName", ""),
                    "description": column.get("description", "")
                })
        
        result = {
            "title": dataset_row['title'] or "",
            "category": dataset_row['category'] or "",
            "location_fields": location_fields,
            "category_fields": category_fields
        }
        logger.info(f"Dataset info for {endpoint_id}: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading dataset info from database for {endpoint_id}: {str(e)}")
        return None

def get_dataset_info(endpoint_id, datasets_dir=None):
    """Get dataset information from the PostgreSQL database (with fallback to files)."""
    # First try to get from database
    dataset_info = get_dataset_info_from_db(endpoint_id)
    
    if dataset_info:
        return dataset_info
    
    # Fallback to file-based approach if database fails and datasets_dir is provided
    if datasets_dir:
        logger.warning(f"Database lookup failed for {endpoint_id}, falling back to file-based approach")
        return get_dataset_info_from_files(endpoint_id, datasets_dir)
    
    logger.warning(f"No dataset info found for endpoint {endpoint_id}")
    return None

def get_dataset_info_from_files(endpoint_id, datasets_dir):
    """Get dataset information from the fixed datasets directory (legacy method)."""
    dataset_file = os.path.join(datasets_dir, f"{endpoint_id}.json")
    logger.info(f"Looking for dataset file: {dataset_file}")
    
    if not os.path.exists(dataset_file):
        logger.warning(f"Dataset file not found for endpoint {endpoint_id}")
        return None
    
    try:
        dataset_info = load_json_file(dataset_file)
        logger.debug(f"Loaded dataset info for {endpoint_id}")
        
        # Extract location and category fields
        location_fields = []
        category_fields = []
        
        columns = dataset_info.get("columns", [])
        logger.debug(f"Found {len(columns)} columns in dataset {endpoint_id}")
        
        for column in columns:
            field_name = column.get("fieldName", "").lower() if column.get("fieldName") else ""
            description = column.get("description", "").lower() if column.get("description") else ""
            name = column.get("name", "")
            
            logger.debug(f"Processing column: {name} (fieldName: {field_name})")
            
            # Only include supervisor_district as a location field
            if field_name == "supervisor_district":
                logger.debug(f"Found location field: {name}")
                location_fields.append({
                    "name": name,
                    "fieldName": column.get("fieldName", ""),
                    "description": column.get("description", "")
                })
            
            # Identify category fields, excluding neighborhood/district fields
            if (any(term in field_name for term in ["category", "type", "class", "group", "status", "subcategory"]) or \
                "categor" in description or "classif" in description or "type of" in description) and \
                not any(term in field_name for term in ["neighborhood", "district"]):
                logger.debug(f"Found category field: {name}")
                category_fields.append({
                    "name": name,
                    "fieldName": column.get("fieldName", ""),
                    "description": column.get("description", "")
                })
        
        result = {
            "title": dataset_info.get("title", ""),
            "category": dataset_info.get("category", ""),
            "location_fields": location_fields,
            "category_fields": category_fields
        }
        logger.info(f"Dataset info for {endpoint_id}: {json.dumps(result, indent=2)}")
        return result
    
    except Exception as e:
        logger.error(f"Error loading dataset info for {endpoint_id}: {str(e)}")
        return None

def enhance_dashboard_queries(queries_file, datasets_dir, output_file):
    """Enhance dashboard queries with IDs and category fields."""
    logger.info(f"Starting enhancement process")
    logger.info(f"Input files:")
    logger.info(f"  Queries file: {queries_file}")
    logger.info(f"  Datasets directory: {datasets_dir}")
    logger.info(f"  Output file: {output_file}")
    
    # Load the dashboard queries
    dashboard_queries = load_json_file(queries_file)
    logger.debug(f"Loaded dashboard queries with {len(dashboard_queries)} categories")
    
    # Check if existing enhanced queries file exists and load it to preserve category_fields
    existing_enhanced_queries = {}
    if os.path.exists(output_file):
        try:
            existing_enhanced_queries = load_json_file(output_file)
            logger.info(f"Loaded existing enhanced queries to preserve custom category_fields")
        except Exception as e:
            logger.warning(f"Could not load existing enhanced queries: {str(e)}")
    
    # Create a new enhanced structure
    enhanced_queries = {}
    
    # Initialize ID counter
    id_counter = 1
    
    # Create a mapping of metric names to IDs for reference
    id_mapping = {}
    
    # Process each category
    for category, subcategories in dashboard_queries.items():
        logger.info(f"Processing category: {category}")
        enhanced_queries[category] = {}
        
        # Process each subcategory
        for subcategory, subcategory_data in subcategories.items():
            logger.info(f"Processing subcategory: {subcategory}")
            enhanced_queries[category][subcategory] = {}
            
            # Copy existing subcategory data
            for key, value in subcategory_data.items():
                if key != "queries":
                    enhanced_queries[category][subcategory][key] = value
            
            # Add queries with enhancements
            enhanced_queries[category][subcategory]["queries"] = {}
            
            # Process each query
            if "queries" in subcategory_data:
                for metric_name, metric_data in subcategory_data["queries"].items():
                    logger.info(f"Processing metric: {metric_name}")
                    
                    # Assign a numeric ID
                    metric_id = id_counter
                    id_counter += 1
                    
                    # Store in mapping for reference
                    id_mapping[metric_id] = {
                        "name": metric_name,
                        "category": category,
                        "subcategory": subcategory
                    }
                    
                    # Create enhanced metric data
                    enhanced_metric = {
                        "id": metric_id,
                        **metric_data  # Copy all existing metric data
                    }
                    
                    # Get endpoint
                    endpoint_id = metric_data.get("endpoint")
                    
                    # Check if we already have custom category_fields for this endpoint in existing enhanced queries
                    existing_category_fields = None
                    if existing_enhanced_queries:
                        # Try to find the metric with this name or endpoint in existing enhanced queries
                        for ex_category in existing_enhanced_queries.values():
                            for ex_subcategory in ex_category.values():
                                for ex_metric_name, ex_metric_data in ex_subcategory.get("queries", {}).items():
                                    if (ex_metric_name == metric_name or 
                                        (endpoint_id and ex_metric_data.get("endpoint") == endpoint_id)):
                                        existing_category_fields = ex_metric_data.get("category_fields")
                                        if existing_category_fields:
                                            logger.info(f"Found existing category_fields for metric {metric_name}")
                                        break
                    
                    # If we have existing custom category_fields, use them
                    if existing_category_fields:
                        enhanced_metric["category_fields"] = existing_category_fields
                        logger.info(f"Using existing category_fields for metric {metric_name}")
                    else:
                        # Otherwise, get dataset info if endpoint is available in the metric data
                        dataset_info = None
                        if endpoint_id:
                            logger.info(f"No existing category_fields found, looking for endpoint: {endpoint_id}")
                            dataset_info = get_dataset_info(endpoint_id, datasets_dir)
                        else:
                            logger.warning(f"No endpoint found for metric: {metric_name}")
                        
                        # Add dataset info if available
                        if dataset_info:
                            logger.debug(f"Adding dataset info to metric {metric_name}")
                            enhanced_metric["dataset_title"] = dataset_info.get("title", "")
                            enhanced_metric["dataset_category"] = dataset_info.get("category", "")
                            enhanced_metric["location_fields"] = dataset_info.get("location_fields", [])
                            enhanced_metric["category_fields"] = dataset_info.get("category_fields", [])
                        else:
                            logger.warning(f"No dataset info available for metric {metric_name}")
                    
                    # Add to enhanced queries
                    enhanced_queries[category][subcategory]["queries"][metric_name] = enhanced_metric
    
    # Save the enhanced queries to the output file
    save_json_file(enhanced_queries, output_file)
    logger.info(f"Enhanced dashboard queries saved to {output_file}")
    
    # Save the ID mapping for reference
    mapping_file = os.path.join(os.path.dirname(output_file), "metric_id_mapping.json")
    save_json_file(id_mapping, mapping_file)
    logger.info(f"Metric ID mapping saved to {mapping_file}")

def main():
    # Define file paths - go up from tools directory to ai directory
    script_dir = Path(__file__).parent.parent  # Go up from tools to ai
    queries_file = script_dir / "data" / "dashboard" / "dashboard_queries.json"
    datasets_dir = script_dir / "data" / "datasets" / "fixed"
    output_file = script_dir / "data" / "dashboard" / "dashboard_queries_enhanced.json"
    
    logger.info("Starting main function")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Queries file: {queries_file}")
    logger.info(f"Datasets directory: {datasets_dir}")
    logger.info(f"Output file: {output_file}")
    
    # Enhance dashboard queries
    enhance_dashboard_queries(queries_file, datasets_dir, output_file)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Migration script to import dashboard queries from dashboard_queries_enhanced.json
into the new metrics table in PostgreSQL.
"""

import os
import json
import logging
from pathlib import Path
from db_utils import get_postgres_connection, execute_with_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dashboard_queries():
    """Load dashboard queries from the JSON file."""
    script_dir = Path(__file__).parent
    json_file = script_dir.parent / "data" / "dashboard" / "dashboard_queries_enhanced.json"
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dashboard queries: {e}")
        return None

def determine_greendirection(metric_name, category, subcategory):
    """
    Determine the greendirection value based on metric characteristics.
    
    Returns:
        str: 'up', 'down', or 'neutral'
    """
    metric_lower = metric_name.lower()
    
    # Crime metrics - increases are generally bad
    if category == 'crime':
        return 'down'
    
    # Safety metrics
    elif category == 'safety':
        # Response times - increases are bad
        if 'response' in metric_lower or 'minutes' in metric_lower:
            return 'down'
        # Fire incidents and fatalities - increases are bad
        elif 'incident' in metric_lower or 'fatalities' in metric_lower:
            return 'down'
        # 311 cases - increases could be neutral (more reporting) or bad (more problems)
        elif '311' in metric_lower:
            return 'neutral'
        # Encampment calls - increases are generally bad
        elif 'encampment' in metric_lower:
            return 'down'
        else:
            return 'neutral'
    
    # Economy metrics
    elif category == 'economy':
        # Housing units - increases are good
        if 'housing' in metric_lower and 'units' in metric_lower:
            return 'up'
        # Business registrations - increases are good
        elif 'registration' in metric_lower and 'business' in metric_lower:
            return 'up'
        elif 'registration' in metric_lower and 'retail' in metric_lower:
            return 'up'
        # Business closures - increases are bad
        elif 'closure' in metric_lower:
            return 'down'
        else:
            return 'neutral'
    
    # Default to neutral for unknown patterns
    else:
        return 'neutral'

def flatten_queries(data):
    """Flatten the nested query structure into a list of metrics."""
    metrics = []
    
    for category, category_data in data.items():
        for subcategory, subcategory_data in category_data.items():
            if 'queries' in subcategory_data:
                for metric_name, metric_data in subcategory_data['queries'].items():
                    # Create a unique key from the metric name
                    metric_key = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace(':', '').replace('🚨', 'violent_').replace('🏠', 'property_').replace('💊', 'drug_').replace('🔒', 'jail_').replace('🚓', 'arrests_').replace('⚖️', 'convictions_').replace('🚒', 'fire_').replace('💔', 'fatalities_').replace('🚑', 'response_').replace('📞', 'cases_').replace('⛺', 'encampment_').replace('🏠', 'housing_').replace('🏢', 'business_').replace('🛍️', 'retail_').replace('🏪', 'retail_closures_').replace('👮', 'police_').replace('🚫', 'closures_')
                    
                    # Clean up the key further
                    import re
                    metric_key = re.sub(r'[^\w_]', '', metric_key)
                    metric_key = re.sub(r'_+', '_', metric_key).strip('_')
                    
                    metric = {
                        'metric_name': metric_name,
                        'metric_key': metric_key,
                        'category': category,
                        'subcategory': subcategory,
                        'endpoint': metric_data.get('endpoint', ''),
                        'summary': metric_data.get('summary', ''),
                        'definition': metric_data.get('definition', ''),
                        'data_sf_url': metric_data.get('data_sf_url', ''),
                        'ytd_query': metric_data.get('ytd_query', ''),
                        'metric_query': metric_data.get('metric_query', ''),
                        'dataset_title': metric_data.get('dataset_title', ''),
                        'dataset_category': metric_data.get('dataset_category', ''),
                        'show_on_dash': metric_data.get('show_on_dash', 'yes') == 'yes',
                        'item_noun': metric_data.get('item_noun', 'Items'),
                        'location_fields': metric_data.get('location_fields', []),
                        'category_fields': metric_data.get('category_fields', []),
                        'metadata': {
                            'id': metric_data.get('id'),
                            'original_category': category,
                            'original_subcategory': subcategory
                        },
                        'greendirection': determine_greendirection(metric_name, category, subcategory)
                    }
                    metrics.append(metric)
    
    return metrics

def insert_metrics(connection, metrics):
    """Insert metrics into the database."""
    cursor = connection.cursor()
    inserted_count = 0
    
    # Clear existing metrics first
    cursor.execute("DELETE FROM metrics")
    logger.info("Cleared existing metrics")
    
    insert_query = """
        INSERT INTO metrics (
            metric_name, metric_key, category, subcategory, endpoint,
            summary, definition, data_sf_url, ytd_query, metric_query,
            dataset_title, dataset_category, show_on_dash, item_noun,
            location_fields, category_fields, metadata, greendirection
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    
    for metric in metrics:
        try:
            cursor.execute(insert_query, (
                metric['metric_name'],
                metric['metric_key'],
                metric['category'],
                metric['subcategory'],
                metric['endpoint'],
                metric['summary'],
                metric['definition'],
                metric['data_sf_url'],
                metric['ytd_query'],
                metric['metric_query'],
                metric['dataset_title'],
                metric['dataset_category'],
                metric['show_on_dash'],
                metric['item_noun'],
                json.dumps(metric['location_fields']),
                json.dumps(metric['category_fields']),
                json.dumps(metric['metadata']),
                metric['greendirection']
            ))
            inserted_count += 1
            logger.info(f"Inserted metric: {metric['metric_name']}")
        except Exception as e:
            logger.error(f"Error inserting metric {metric['metric_name']}: {e}")
    
    connection.commit()
    cursor.close()
    return inserted_count

def migrate_dashboard_to_metrics():
    """Main migration function."""
    logger.info("Starting dashboard queries migration to metrics table")
    
    # Load dashboard queries
    dashboard_data = load_dashboard_queries()
    if not dashboard_data:
        logger.error("Failed to load dashboard queries")
        return False
    
    # Flatten the queries
    metrics = flatten_queries(dashboard_data)
    logger.info(f"Found {len(metrics)} metrics to migrate")
    
    # Insert into database
    def migration_operation(connection):
        return insert_metrics(connection, metrics)
    
    result = execute_with_connection(operation=migration_operation)
    
    if result["status"] == "success":
        inserted_count = result["result"]
        logger.info(f"Successfully migrated {inserted_count} metrics to database")
        return True
    else:
        logger.error(f"Migration failed: {result['message']}")
        return False

if __name__ == "__main__":
    migrate_dashboard_to_metrics() 
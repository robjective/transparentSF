import os
import json
import re
import requests
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import logging
from tools.db_utils import get_postgres_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    """Sanitize the filename by removing or replacing invalid characters."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    sanitized = sanitized.strip()
    return sanitized

def scrape_dataset_metadata(dataset_url):
    """Retrieve dataset metadata including columns and descriptions via the Socrata API."""
    # Extract the dataset identifier from the URL
    match = re.search(r'/([a-z0-9]{4}-[a-z0-9]{4})(?:/|$)', dataset_url)
    if not match:
        logger.error(f"Could not extract dataset ID from URL: {dataset_url}")
        return None
    dataset_id = match.group(1)

    metadata_url = f'https://data.sfgov.org/api/views/{dataset_id}.json'

    try:
        response = requests.get(metadata_url)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve metadata from {metadata_url}: {e}")
        return None

    if response.status_code == 200:
        data = response.json()
        # Get the title and description
        title = data.get('name', 'Untitled')
        category = data.get('category', '')
        description = data.get('description', '')
        # Get the columns
        columns_info = []
        columns = data.get('columns', [])
        for column in columns:
            column_info = {
                'name': column.get('name'),
                'fieldName': column.get('fieldName'),
                'dataTypeName': column.get('dataTypeName'),
                'description': column.get('description'),
                'position': column.get('position'),
                'renderTypeName': column.get('renderTypeName'),
                'tableColumnId': column.get('id'),
            }
            columns_info.append(column_info)
        
        # Get publishing department
        publishing_department = data.get('metadata', {}).get('custom_fields', {}).get('Department Metrics', {}).get('Publishing Department', '')
        # Get most recent update date
        rows_updated_at = data.get('rowsUpdatedAt')
        if rows_updated_at:
            rows_updated_at = datetime.utcfromtimestamp(rows_updated_at)
        else:
            rows_updated_at = None

        dataset_info = {
            'category': category,
            'endpoint': dataset_id,
            'url': dataset_url,
            'title': title,
            'description': description,
            'columns': columns_info,
            'publishing_department': publishing_department,
            'rows_updated_at': rows_updated_at
        }
        return dataset_info
    else:
        logger.error(f"Failed to retrieve metadata: {response.status_code}")
        return None

def store_dataset_in_db(connection, dataset_info):
    """Store dataset metadata in the PostgreSQL database."""
    try:
        cursor = connection.cursor()
        
        # Check if dataset already exists
        cursor.execute("""
            SELECT id FROM datasets WHERE endpoint = %s
        """, (dataset_info['endpoint'],))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing dataset
            cursor.execute("""
                UPDATE datasets SET
                    url = %s,
                    title = %s,
                    category = %s,
                    description = %s,
                    publishing_department = %s,
                    rows_updated_at = %s,
                    columns = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE endpoint = %s
            """, (
                dataset_info['url'],
                dataset_info['title'],
                dataset_info['category'],
                dataset_info['description'],
                dataset_info['publishing_department'],
                dataset_info['rows_updated_at'],
                Json(dataset_info['columns']),
                dataset_info['endpoint']
            ))
            logger.info(f"Updated dataset {dataset_info['endpoint']}: {dataset_info['title']}")
        else:
            # Insert new dataset
            cursor.execute("""
                INSERT INTO datasets (
                    endpoint, url, title, category, description, 
                    publishing_department, rows_updated_at, columns
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                dataset_info['endpoint'],
                dataset_info['url'],
                dataset_info['title'],
                dataset_info['category'],
                dataset_info['description'],
                dataset_info['publishing_department'],
                dataset_info['rows_updated_at'],
                Json(dataset_info['columns'])
            ))
            logger.info(f"Inserted new dataset {dataset_info['endpoint']}: {dataset_info['title']}")
        
        connection.commit()
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"Error storing dataset {dataset_info['endpoint']} in database: {e}")
        connection.rollback()
        return False

def main():
    # Load all dataset URLs from the dataset_urls.json file
    data_folder = 'data'

    # Load the dataset URLs
    dataset_urls_file = os.path.join(data_folder, 'dataset_urls.json')
    try:
        with open(dataset_urls_file, 'r', encoding='utf-8') as f:
            dataset_urls = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset URLs file not found: {dataset_urls_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing dataset URLs file: {e}")
        return

    # Get database connection
    connection = get_postgres_connection()
    if not connection:
        logger.error("Failed to connect to database")
        return

    successful_count = 0
    failed_count = 0

    # Process each URL
    for dataset_url in dataset_urls:
        try:
            logger.info(f"Processing URL: {dataset_url}")
            dataset_info = scrape_dataset_metadata(dataset_url)
            if dataset_info:
                if store_dataset_in_db(connection, dataset_info):
                    successful_count += 1
                else:
                    failed_count += 1
            else:
                logger.warning(f"Failed to process dataset at {dataset_url}")
                failed_count += 1
        except Exception as e:
            logger.error(f"Error processing {dataset_url}: {e}")
            failed_count += 1

    # Close database connection
    connection.close()
    
    logger.info(f"Processing complete. Successfully processed: {successful_count}, Failed: {failed_count}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Functions to store anomaly detection results in PostgreSQL database.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import logging
import datetime
from datetime import date
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dateutil import parser
from tools.db_utils import get_postgres_connection, execute_with_connection, CustomJSONEncoder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_anomalies_table(connection):
    """
    Check if the anomalies table exists.
    
    Args:
        connection: PostgreSQL database connection
        
    Returns:
        bool: True if table exists, False otherwise
    """
    try:
        with connection.cursor() as cursor:
            # Check if the table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'anomalies'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                # Create the table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE anomalies (
                        id SERIAL PRIMARY KEY,
                        group_value TEXT,
                        group_field_name TEXT,
                        period_type TEXT,
                        comparison_mean FLOAT,
                        recent_mean FLOAT,
                        difference FLOAT,
                        std_dev FLOAT,
                        out_of_bounds BOOLEAN,
                        recent_date DATE,
                        comparison_dates JSONB,
                        comparison_counts JSONB,
                        recent_dates JSONB,
                        recent_counts JSONB,
                        metadata JSONB,
                        field_name TEXT,
                        object_type TEXT,
                        object_id TEXT, 
                        object_name TEXT,
                        recent_data JSONB,
                        comparison_data JSONB,
                        district INTEGER,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX idx_anomalies_created_at ON anomalies (created_at);
                    CREATE INDEX idx_anomalies_object_id ON anomalies (object_id);
                    CREATE INDEX idx_anomalies_is_active ON anomalies (is_active);
                """)
                
                connection.commit()
                logging.info("Created anomalies table")
                return True
            
            # Check if is_active column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'anomalies' AND column_name = 'is_active'
                );
            """)
            column_exists = cursor.fetchone()[0]
            
            if not column_exists:
                # Add is_active column if it doesn't exist
                cursor.execute("ALTER TABLE anomalies ADD COLUMN is_active BOOLEAN DEFAULT TRUE")
                cursor.execute("CREATE INDEX idx_anomalies_is_active ON anomalies (is_active)")
                connection.commit()
                logging.info("Added is_active column to anomalies table")
                
            # Check if required indexes exist
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE tablename = 'anomalies' AND indexname = 'idx_anomalies_created_at'
                );
            """)
            index_exists = cursor.fetchone()[0]
            
            if not index_exists:
                logging.warning("Some indexes on anomalies table may be missing. Performance might be affected.")
            
            return True
    except Exception as e:
        logging.error(f"Error checking anomalies table: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def store_anomalies_in_db(connection, results, metadata):
    """
    Store detected anomalies in the PostgreSQL database.
    
    Args:
        connection: PostgreSQL database connection
        results: List of anomaly results
        metadata: Metadata about the anomaly detection run
    
    Returns:
        int: Number of anomalies stored
    """
    if connection is None:
        logging.error("No database connection available")
        return 0
    
    try:
        # Check if the table exists
        if not create_anomalies_table(connection):
            logging.error("Cannot store anomalies - table does not exist")
            return 0
        
        # Use the custom JSON encoder to serialize the entire results and metadata
        serializable_metadata = json.loads(json.dumps(metadata, cls=CustomJSONEncoder))
        
        # Extract object information from metadata
        object_type = metadata.get('object_type', 'unknown')
        object_id = metadata.get('object_id', 'unknown')
        object_name = metadata.get('object_name', metadata.get('title', 'unknown'))
        field_name = metadata.get('numeric_field', 'unknown')
        group_field_name = metadata.get('group_field', 'unknown')
        period_type = metadata.get('period_type', 'month')
        
        # Ensure title is set in metadata
        if 'title' not in serializable_metadata:
            if object_name != 'unknown':
                serializable_metadata['title'] = object_name
            else:
                # Generate a title based on available information
                title_parts = []
                if field_name != 'unknown':
                    title_parts.append(field_name)
                if group_field_name != 'unknown':
                    title_parts.append(f"by {group_field_name}")
                serializable_metadata['title'] = " - ".join(title_parts) if title_parts else "Anomaly Analysis"
        
        # Ensure object_name is set if not present
        if 'object_name' not in serializable_metadata and 'title' in serializable_metadata:
            serializable_metadata['object_name'] = serializable_metadata['title']
        
        # Get district from filter conditions if it exists
        district = None
        if 'filter_conditions' in metadata:
            for condition in metadata['filter_conditions']:
                field = condition.get('field', '').lower()
                if field in ['district', 'police_district', 'supervisor_district']:
                    district = condition.get('value')
                    break
        
        # Default district to 0 if it's null
        if district is None:
            district = 0
        else:
            # Try to convert district to integer
            try:
                district = int(district)
            except (ValueError, TypeError):
                district = 0
        
        # Check for recent_period_end and get it if available
        recent_period_end = None
        if 'recent_period' in metadata and 'end' in metadata['recent_period']:
            recent_period_end = metadata['recent_period']['end']
            if isinstance(recent_period_end, str):
                try:
                    recent_period_end = parser.parse(recent_period_end).date()
                except:
                    pass
        
        # First, deactivate any existing active anomalies with the same parameters
        with connection.cursor() as cursor:
            cursor.execute("""
                UPDATE anomalies 
                SET is_active = FALSE
                WHERE object_type = %s 
                AND object_id = %s
                AND object_name = %s
                AND group_field_name = %s
                AND period_type = %s
                AND district::TEXT = %s
                AND is_active = TRUE
            """, (object_type, object_id, object_name, group_field_name, period_type, str(district)))
            
        inserted_count = 0
        
        with connection.cursor() as cursor:
            for result in results:
                # Create a fully serialized copy of the result using our custom encoder
                serializable_result = json.loads(json.dumps(result, cls=CustomJSONEncoder))
                
                # Extract dates and counts from results
                all_dates = result.get('dates', [])
                all_counts = result.get('counts', [])
                
                # Initialize arrays for comparison and recent data
                comparison_dates = []
                comparison_counts = []
                recent_dates = []
                recent_counts = []
                recent_date = recent_period_end  # Default to end of recent period
                
                # Organize data into comparison and recent periods
                if 'recent_period' in metadata and 'comparison_period' in metadata:
                    recent_start = metadata['recent_period']['start']
                    recent_end = metadata['recent_period']['end']
                    comp_start = metadata['comparison_period']['start']
                    comp_end = metadata['comparison_period']['end']
                    
                    # Parse dates if they're strings
                    if isinstance(recent_start, str):
                        recent_start = parser.parse(recent_start).date()
                    if isinstance(recent_end, str):
                        recent_end = parser.parse(recent_end).date()
                    if isinstance(comp_start, str):
                        comp_start = parser.parse(comp_start).date()
                    if isinstance(comp_end, str):
                        comp_end = parser.parse(comp_end).date()
                    
                    # Use recent_end as the recent_date field value
                    recent_date = recent_end
                    
                    # Group data by period based on date string
                    for i, date_str in enumerate(all_dates):
                        try:
                            # Parse the date based on period_type
                            if period_type == 'year':
                                date_obj = datetime.datetime.strptime(date_str, "%Y").date()
                            elif period_type == 'month':
                                date_obj = datetime.datetime.strptime(f"{date_str}-01", "%Y-%m-%d").date()
                            else:
                                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                            
                            # Add to appropriate arrays
                            if comp_start <= date_obj <= comp_end:
                                comparison_dates.append(date_str)
                                comparison_counts.append(all_counts[i])
                            elif recent_start <= date_obj <= recent_end:
                                recent_dates.append(date_str)
                                recent_counts.append(all_counts[i])
                        except (ValueError, TypeError):
                            continue
                
                # Add comparison and recent data to the serializable result
                serializable_result['comparison_dates'] = comparison_dates
                serializable_result['comparison_counts'] = comparison_counts
                serializable_result['recent_dates'] = recent_dates
                serializable_result['recent_counts'] = recent_counts
                serializable_result['recent_date'] = recent_date
                
                # Extract recent and comparison data for individual points
                recent_data = {}
                comparison_data = {}
                
                # Store data points in their respective dictionaries
                for i, date_str in enumerate(comparison_dates):
                    comparison_data[date_str] = comparison_counts[i]
                
                for i, date_str in enumerate(recent_dates):
                    recent_data[date_str] = recent_counts[i]
                
                # Generate caption for the anomaly
                percent_difference = abs((result['difference'] / result['comparison_mean']) * 100) if result['comparison_mean'] else 0
                action = 'increase' if result['difference'] > 0 else 'drop' if result['difference'] < 0 else 'no change'
                y_axis_label = metadata.get('y_axis_label', 'Value').lower()
                
                comparison_period_label = f"{comp_start.strftime('%B %Y')} to {comp_end.strftime('%B %Y')}"
                recent_period_label = f"{recent_start.strftime('%B %Y')}"
                
                caption = (
                    f"In {recent_period_label}, there were {result['recent_mean']:,.0f} {result['group_value']} {y_axis_label} per month, "
                    f"compared to an average of {result['comparison_mean']:,.0f} per month over {comparison_period_label}, "
                    f"a {percent_difference:.1f}% {action}."
                )
                
                # Add caption to metadata
                if isinstance(serializable_metadata, dict):
                    serializable_metadata['caption'] = caption
                
                cursor.execute("""
                    INSERT INTO anomalies 
                    (group_value, group_field_name, period_type, comparison_mean, recent_mean, 
                     difference, std_dev, out_of_bounds, recent_date, comparison_dates, 
                     comparison_counts, recent_dates, recent_counts, metadata, field_name, 
                     object_type, object_id, object_name, recent_data, comparison_data, district, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                """, (
                    result['group_value'],
                    group_field_name,
                    period_type,
                    result['comparison_mean'],
                    result['recent_mean'],
                    result['difference'],
                    result.get('stdDev', 0),
                    result.get('out_of_bounds', False),
                    recent_date,
                    Json(serializable_result['comparison_dates']),
                    Json(serializable_result['comparison_counts']),
                    Json(serializable_result['recent_dates']),
                    Json(serializable_result['recent_counts']),
                    Json(serializable_metadata),
                    field_name,
                    object_type,
                    object_id,
                    object_name,
                    Json(recent_data),
                    Json(comparison_data),
                    str(district)
                ))
                inserted_count += 1
        
        connection.commit()
        logger.info("Successfully stored anomaly detection results in database")
        return inserted_count
    except Exception as e:
        logging.error(f"Error storing anomalies in database: {e}")
        connection.rollback()
        return 0

def store_anomaly_data(
    results,
    metadata,
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None
) -> Dict[str, Any]:
    """
    Main function to store anomaly data in the database.
    
    Args:
        results: List of anomaly results
        metadata: Dictionary with metadata about the anomalies
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        dict: Result with status and message
    """
    try:
        # Extract metrics_id and period_type from metadata
        metrics_id = metadata.get('object_id', 'unknown')
        period_type = metadata.get('period_type', 'month')
        
        def store_operation(connection):
            return store_anomalies_in_db(connection, results, metadata)
        
        result = execute_with_connection(
            operation=store_operation,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )
        
        if result["status"] == "success":
            inserted_count = result["result"]
            return {
                "status": "success",
                "message": f"Successfully stored {inserted_count} anomalies in the database",
                "anomalies_stored": inserted_count,
            }
        else:
            return result
    
    except Exception as e:
        logging.error(f"Error in store_anomaly_data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }

def get_anomalies(
    query_type='recent',
    limit=30, 
    group_filter=None,
    date_start=None,
    date_end=None,
    only_anomalies=True,
    metric_name=None,
    district_filter=None,
    metric_id=None,
    period_type=None,
    only_active=True,
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None
) -> Dict[str, Any]:
    """
    Query the database for anomalies based on various filters.
    
    Args:
        query_type: Type of query ('recent', 'by_group', 'by_date', 'by_anomaly_severity', 'by_district')
        limit: Maximum number of results to return
        group_filter: Filter by group value
        date_start: Filter by start date
        date_end: Filter by end date
        only_anomalies: Only return records where out_of_bounds is true
        metric_name: Filter by metric name
        district_filter: Filter by district
        metric_id: Filter by object_id (metric ID)
        period_type: Filter by period type (year, month, week, day)
        only_active: Only return active anomalies (default: True)
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        dict: Query results
    """
    def query_operation(connection):
        # Create cursor with dictionary-like results
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Start building the query
        query = "SELECT id, group_field_name, group_value, comparison_mean, recent_mean, difference, std_dev, out_of_bounds, created_at, metadata, district, object_id, period_type, is_active "
        query += "FROM anomalies WHERE 1=1 "
        params = []
        
        # Apply filters based on parameters
        if only_anomalies:
            query += "AND out_of_bounds = true "
            
        if only_active:
            query += "AND is_active = true "
            
        if group_filter:
            query += "AND group_value ILIKE %s "
            params.append(f"%{group_filter}%")
            
        if date_start:
            query += "AND created_at >= %s "
            params.append(date_start)
            
        if date_end:
            query += "AND created_at <= %s "
            params.append(date_end)
            
        if metric_name:
            # Filter by metric name in the metadata JSON
            query += "AND metadata->>'object_name' ILIKE %s "
            params.append(f"%{metric_name}%")
        
        if district_filter:
            # Filter by district - ensure both sides are treated as text
            query += "AND district::TEXT = %s "
            params.append(str(district_filter))
            
        if metric_id:
            # Filter by object_id
            query += "AND object_id::TEXT = %s "
            params.append(str(metric_id))
        
        if period_type:
            # Filter by period_type
            query += "AND period_type = %s "
            params.append(period_type)
        
        # Different query types
        if query_type == 'recent':
            query += "ORDER BY created_at DESC "
        elif query_type == 'by_group':
            query += "ORDER BY group_value ASC, created_at DESC "
        elif query_type == 'by_date':
            query += "ORDER BY created_at ASC "
        elif query_type == 'by_anomaly_severity':
            query += "ORDER BY ABS(difference) DESC "
        elif query_type == 'by_district':
            query += "ORDER BY district ASC, created_at DESC "
        
        # Add limit
        query += "LIMIT %s"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Convert to list of dictionaries and handle datetime objects
        result_list = []
        for row in results:
            row_dict = dict(row)
            # Format created_at for display
            if row_dict.get('created_at') and isinstance(row_dict['created_at'], datetime.datetime):
                row_dict['created_at'] = row_dict['created_at'].isoformat()
            result_list.append(row_dict)
        
        cursor.close()
        return result_list
    
    result = execute_with_connection(
        operation=query_operation,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    )
    
    if result["status"] == "success":
        return {
            "status": "success",
            "count": len(result["result"]),
            "results": result["result"]
        }
    else:
        return result

def get_anomaly_details(
    anomaly_id,
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None
) -> Dict[str, Any]:
    """
    Get detailed information about a specific anomaly by ID.
    
    Args:
        anomaly_id: The ID of the anomaly to retrieve
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        dict: Detailed anomaly information
    """
    def get_details_operation(connection):
        # Create cursor with dictionary-like results
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Query for the specific anomaly
        query = "SELECT * FROM anomalies WHERE id = %s"
        cursor.execute(query, (anomaly_id,))
        anomaly = cursor.fetchone()
        
        if not anomaly:
            cursor.close()
            return None
        
        # Convert to dict and handle datetime objects
        item = dict(anomaly)
        if 'created_at' in item and isinstance(item['created_at'], datetime.datetime):
            item['created_at'] = item['created_at'].isoformat()
        
        cursor.close()
        return item
    
    result = execute_with_connection(
        operation=get_details_operation,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    )
    
    if result["status"] == "success":
        if result["result"] is None:
            return {
                "status": "error",
                "message": f"No anomaly found with ID {anomaly_id}"
            }
        return {
            "status": "success",
            "anomaly": result["result"]
        }
    else:
        return result

def clear_anomalies_by_metrics_id(
    metrics_id,
    period_type,
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None
) -> Dict[str, Any]:
    """
    Mark anomalies as inactive for a specific metrics_id and period_type.
    
    Args:
        metrics_id: The metrics ID to clear anomalies for
        period_type: The period type to clear (year, month, week, day)
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        
    Returns:
        dict: Result with status and message
    """
    def clear_operation(connection):
        # Create cursor
        cursor = connection.cursor()
        
        # Mark anomalies as inactive instead of deleting them
        cursor.execute(
            "UPDATE anomalies SET is_active = FALSE WHERE object_id = %s AND period_type = %s AND is_active = TRUE",
            (str(metrics_id), period_type)
        )
        
        # Get number of rows updated
        updated_count = cursor.rowcount
        
        # Commit the transaction
        connection.commit()
        
        cursor.close()
        return updated_count
    
    result = execute_with_connection(
        operation=clear_operation,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    )
    
    if result["status"] == "success":
        updated_count = result["result"]
        return {
            "status": "success",
            "message": f"Successfully marked {updated_count} anomalies as inactive",
            "updated_count": updated_count
        }
    else:
        return result 
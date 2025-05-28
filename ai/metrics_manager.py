"""
Metrics Manager - Database-only endpoints for metric control functionality
"""

import json
import logging
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import psycopg2.extras

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/api/enhanced-queries")
async def get_enhanced_queries_db():
    """Serve enhanced dashboard queries built entirely from database."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query all active metrics from the database
        cursor.execute("""
            SELECT 
                m.id,
                m.metric_name,
                m.metric_key,
                m.category,
                m.subcategory,
                m.endpoint,
                m.summary,
                m.definition,
                m.data_sf_url,
                m.ytd_query,
                m.metric_query,
                m.dataset_title,
                m.dataset_category,
                m.show_on_dash,
                m.item_noun,
                m.greendirection,
                m.location_fields,
                m.category_fields,
                m.metadata,
                d.title as dataset_title_from_datasets,
                d.category as dataset_category_from_datasets,
                d.columns as dataset_columns
            FROM metrics m
            LEFT JOIN datasets d ON m.endpoint = d.endpoint AND d.is_active = true
            WHERE m.is_active = true
            ORDER BY m.category, m.subcategory, m.id
        """)
        
        metrics_rows = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Build the enhanced queries structure
        enhanced_queries = {}
        
        for row in metrics_rows:
            category = row['category']
            subcategory = row['subcategory'] or category
            metric_name = row['metric_name']
            
            # Initialize category if not exists
            if category not in enhanced_queries:
                enhanced_queries[category] = {}
            
            # Initialize subcategory if not exists
            if subcategory not in enhanced_queries[category]:
                enhanced_queries[category][subcategory] = {
                    "queries": {}
                }
            
            # Use dataset info from datasets table if available, otherwise from metrics table
            dataset_title = row['dataset_title_from_datasets'] or row['dataset_title'] or ""
            dataset_category = row['dataset_category_from_datasets'] or row['dataset_category'] or ""
            
            # Build the metric data
            metric_data = {
                "id": row['id'],
                "endpoint": row['endpoint'],
                "summary": row['summary'] or "",
                "definition": row['definition'] or "",
                "data_sf_url": row['data_sf_url'] or "",
                "show_on_dash": "yes" if row['show_on_dash'] else "no",
                "item_noun": row['item_noun'] or "Items",
                "ytd_query": row['ytd_query'] or "",
                "metric_query": row['metric_query'] or "",
                "dataset_title": dataset_title,
                "dataset_category": dataset_category,
                "greendirection": row['greendirection'] or "up",
                "location_fields": row['location_fields'] or [],
                "category_fields": row['category_fields'] or []
            }
            
            # Add metadata if available
            if row['metadata']:
                metric_data["metadata"] = row['metadata']
            
            # Add the metric to the enhanced queries
            enhanced_queries[category][subcategory]["queries"][metric_name] = metric_data
        
        logger.info(f"Built enhanced queries from database with {len(metrics_rows)} metrics")
        return JSONResponse(content=enhanced_queries)
        
    except Exception as e:
        logger.error(f"Error building enhanced queries from database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building enhanced queries: {str(e)}")


@router.get("/backend/get-endpoint-columns/{endpoint}")
async def get_endpoint_columns_db(endpoint: str):
    """Get available columns for an endpoint from database only."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query the datasets table for this endpoint
        cursor.execute("""
            SELECT columns
            FROM datasets 
            WHERE endpoint = %s AND is_active = true
        """, (endpoint,))
        
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if row and row['columns']:
            # Extract column names from the database
            columns = [col["fieldName"] for col in row['columns'] if col.get("fieldName")]
            
            return JSONResponse({
                "status": "success",
                "columns": columns
            })
        else:
            logger.warning(f"No columns found in database for endpoint: {endpoint}")
            raise HTTPException(status_code=404, detail=f"No columns found for endpoint: {endpoint}")
            
    except Exception as e:
        logger.error(f"Error getting columns for endpoint '{endpoint}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting columns: {str(e)}")


@router.get("/backend/get-selected-columns/{endpoint}")
async def get_selected_columns_db(endpoint: str, metric_id: str = None):
    """Get currently selected columns for an endpoint and optionally a specific metric ID from database."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query the metrics table for this endpoint and metric_id
        if metric_id:
            cursor.execute("""
                SELECT category_fields
                FROM metrics 
                WHERE endpoint = %s AND id = %s AND is_active = true
            """, (endpoint, int(metric_id)))
        else:
            # If no metric_id provided, get the first metric with this endpoint
            cursor.execute("""
                SELECT category_fields
                FROM metrics 
                WHERE endpoint = %s AND is_active = true
                LIMIT 1
            """, (endpoint,))
        
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if row and row['category_fields']:
            # Extract field names from category_fields
            selected_columns = [field["fieldName"] for field in row['category_fields'] if field.get("fieldName")]
            
            logger.info(f"Found selected columns for endpoint {endpoint}" + 
                       (f" and metric_id {metric_id}" if metric_id else "") + 
                       f": {selected_columns}")
            
            return JSONResponse({
                "status": "success",
                "columns": selected_columns
            })
        else:
            logger.info(f"No selected columns found for endpoint {endpoint}" + 
                       (f" and metric_id {metric_id}" if metric_id else ""))
            return JSONResponse({
                "status": "success",
                "columns": []
            })
            
    except Exception as e:
        logger.error(f"Error getting selected columns for endpoint '{endpoint}'" + 
                    (f" and metric_id '{metric_id}'" if metric_id else "") + 
                    f": {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting selected columns: {str(e)}")


@router.post("/backend/update-selected-columns")
async def update_selected_columns_db(request: Request):
    """Update the selected columns for a metric in the database."""
    try:
        from tools.db_utils import get_postgres_connection
        
        # Get request data
        data = await request.json()
        endpoint = data.get("endpoint")
        columns = data.get("columns", [])
        metric_id = data.get("metric_id")
        
        if not endpoint or not metric_id:
            raise HTTPException(status_code=400, detail="Both endpoint and metric_id are required")
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build the category_fields JSON structure
        category_fields = [
            {
                "name": col,
                "fieldName": col,
                "description": f"Selected column for metric analysis"
            }
            for col in columns
        ]
        
        # Update the metric in the database
        cursor.execute("""
            UPDATE metrics 
            SET category_fields = %s, updated_at = CURRENT_TIMESTAMP
            WHERE endpoint = %s AND id = %s AND is_active = true
        """, (json.dumps(category_fields), endpoint, int(metric_id)))
        
        if cursor.rowcount == 0:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"No metric found with endpoint: {endpoint} and id: {metric_id}")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully updated category_fields for metric {metric_id} with endpoint {endpoint}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Updated category_fields for metric {metric_id} with endpoint {endpoint}"
        })
        
    except Exception as e:
        logger.error(f"Error updating category_fields: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating selected columns: {str(e)}")


@router.get("/backend/run_specific_metric")
async def run_specific_metric_db(metric_id: int, district_id: int = 0, period_type: str = 'year'):
    """Run analysis for a specific metric using database-only approach."""
    logger.info(f"Run specific metric called for metric_id: {metric_id}, district_id: {district_id}, period_type: {period_type}")
    
    # Validate period_type
    period_folder_map = {
        'year': 'annual',
        'month': 'monthly',
        'day': 'daily',
        'ytd': 'ytd',
        'week': 'weekly'
    }
    
    if period_type not in period_folder_map:
        logger.error(f"Invalid period_type: {period_type}")
        return JSONResponse({
            "status": "error",
            "message": f"Invalid period_type: {period_type}. Must be one of: {', '.join(period_folder_map.keys())}"
        }, status_code=400)
    
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get the metric details from the database
        cursor.execute("""
            SELECT 
                m.id,
                m.metric_name,
                m.metric_key,
                m.category,
                m.subcategory,
                m.endpoint,
                m.summary,
                m.definition,
                m.data_sf_url,
                m.ytd_query,
                m.metric_query,
                m.dataset_title,
                m.dataset_category,
                m.show_on_dash,
                m.item_noun,
                m.greendirection,
                m.location_fields,
                m.category_fields,
                m.metadata
            FROM metrics m
            WHERE m.id = %s AND m.is_active = true
        """, (metric_id,))
        
        metric_row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not metric_row:
            logger.error(f"Metric ID {metric_id} not found in database")
            return JSONResponse({
                "status": "error",
                "message": f"Metric ID {metric_id} not found"
            }, status_code=404)
        
        # Build metric_info structure for the analysis functions
        metric_info = {
            'metric_id': str(metric_row['id']),
            'query_name': metric_row['metric_name'],
            'top_category': metric_row['category'],
            'subcategory': metric_row['subcategory'],
            'endpoint': metric_row['endpoint'],
            'summary': metric_row['summary'],
            'definition': metric_row['definition'],
            'data_sf_url': metric_row['data_sf_url'],
            'category_fields': metric_row['category_fields'] or [],
            'location_fields': metric_row['location_fields'] or [],
            'query_data': {
                'ytd_query': metric_row['ytd_query'],
                'metric_query': metric_row['metric_query'],
                'id': metric_row['id'],
                'endpoint': metric_row['endpoint']
            }
        }
        
        # Create logs directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Determine which script to run based on period_type
        if period_type == 'ytd':
            # For YTD metrics, use generate_dashboard_metrics.py
            logger.info(f"Running YTD dashboard metrics generation for metric ID {metric_id}")
            
            try:
                from generate_dashboard_metrics import process_single_metric
                
                if callable(process_single_metric):
                    result = process_single_metric(metric_id=metric_id, period_type=period_type)
                    logger.info(f"YTD metrics generation completed for metric ID {metric_id}")
                else:
                    from generate_dashboard_metrics import main as generate_all_metrics
                    generate_all_metrics()
                    logger.info(f"All YTD metrics generated (including metric ID {metric_id})")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import specific metric generation function: {str(e)}")
                from generate_dashboard_metrics import main as generate_all_metrics
                generate_all_metrics()
                logger.info(f"All YTD metrics generated (including metric ID {metric_id})")
        
        elif period_type == 'week':
            # For weekly analysis, use generate_weekly_analysis.py
            logger.info(f"Running weekly analysis for metric ID {metric_id}")
            
            try:
                from generate_weekly_analysis import run_weekly_analysis, generate_weekly_newsletter
                
                results = run_weekly_analysis(
                    metrics_list=[str(metric_id)],
                    process_districts=(district_id == 0)
                )
                
                newsletter_path = generate_weekly_newsletter(results)
                logger.info(f"Weekly analysis completed for metric ID {metric_id}")
                
            except ImportError as e:
                logger.error(f"Could not import generate_weekly_analysis module: {str(e)}")
                return JSONResponse({
                    "status": "error",
                    "message": f"Missing required module: {str(e)}"
                }, status_code=500)
            except Exception as e:
                logger.error(f"Error in generate_weekly_analysis: {str(e)}")
                return JSONResponse({
                    "status": "error",
                    "message": f"Error generating weekly analysis: {str(e)}"
                }, status_code=500)
                
        else:
            # For monthly/annual analysis, use generate_metric_analysis.py
            logger.info(f"Running {period_type} analysis for metric ID {metric_id}")
            
            try:
                from generate_metric_analysis import process_metric_analysis
                
                result = process_metric_analysis(
                    metric_info=metric_info,
                    period_type=period_type,
                    process_districts=True
                )
                logger.info(f"{period_type.capitalize()} analysis completed for metric ID {metric_id}")
                
            except ImportError as e:
                logger.error(f"Could not import generate_metric_analysis module: {str(e)}")
                return JSONResponse({
                    "status": "error",
                    "message": f"Missing required module: {str(e)}"
                }, status_code=500)
            except Exception as e:
                logger.error(f"Error in generate_metric_analysis: {str(e)}")
                return JSONResponse({
                    "status": "error",
                    "message": f"Error generating metric analysis: {str(e)}"
                }, status_code=500)
        
        return JSONResponse({
            "status": "success",
            "message": f"Successfully completed {period_type} analysis for metric {metric_id}"
        })
        
    except Exception as e:
        logger.error(f"Error running specific metric analysis: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error running metric analysis: {str(e)}"
        }, status_code=500)


@router.post("/api/metrics")
async def create_metric_db(request: Request):
    """Create a new metric in the database."""
    try:
        from tools.db_utils import get_postgres_connection
        
        data = await request.json()
        
        # Validate required fields
        required_fields = ['metric_name', 'category', 'endpoint']
        for field in required_fields:
            if not data.get(field):
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Generate metric_key from metric_name
        metric_key = data['metric_name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        # Insert the new metric
        cursor.execute("""
            INSERT INTO metrics (
                metric_name, metric_key, category, subcategory, endpoint,
                summary, definition, data_sf_url, ytd_query, metric_query,
                dataset_title, dataset_category, show_on_dash, item_noun,
                location_fields, category_fields, metadata, greendirection
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            data['metric_name'],
            metric_key,
            data['category'],
            data.get('subcategory', data['category']),
            data['endpoint'],
            data.get('summary', ''),
            data.get('definition', ''),
            data.get('data_sf_url', ''),
            data.get('ytd_query', ''),
            data.get('metric_query', ''),
            data.get('dataset_title', ''),
            data.get('dataset_category', ''),
            data.get('show_on_dash', True),
            data.get('item_noun', 'Items'),
            json.dumps(data.get('location_fields', [])),
            json.dumps(data.get('category_fields', [])),
            json.dumps(data.get('metadata', {})),
            data.get('greendirection', 'up')
        ))
        
        new_metric_id = cursor.fetchone()['id']
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Created new metric with ID {new_metric_id}: {data['metric_name']}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Created metric: {data['metric_name']}",
            "metric_id": new_metric_id
        })
        
    except Exception as e:
        logger.error(f"Error creating metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating metric: {str(e)}")


@router.delete("/api/metrics/{metric_id}")
async def delete_metric(metric_id: int):
    """Delete a metric from the database by ID."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Check if metric exists
        cursor.execute("SELECT id, metric_name FROM metrics WHERE id = %s", (metric_id,))
        metric = cursor.fetchone()
        
        if not metric:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"Metric with ID {metric_id} not found")
        
        metric_name = metric['metric_name']
        
        # Delete the metric (soft delete by setting is_active to false)
        cursor.execute("""
            UPDATE metrics 
            SET is_active = false, updated_at = CURRENT_TIMESTAMP 
            WHERE id = %s
        """, (metric_id,))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully deleted metric {metric_id}: {metric_name}")
        
        return {
            "status": "success",
            "message": f"Metric '{metric_name}' (ID: {metric_id}) has been deleted",
            "metric_id": metric_id,
            "metric_name": metric_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting metric {metric_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting metric: {str(e)}")


@router.delete("/api/metrics/{metric_id}/hard")
async def hard_delete_metric(metric_id: int):
    """Permanently delete a metric from the database by ID (hard delete)."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Check if metric exists
        cursor.execute("SELECT id, metric_name FROM metrics WHERE id = %s", (metric_id,))
        metric = cursor.fetchone()
        
        if not metric:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"Metric with ID {metric_id} not found")
        
        metric_name = metric['metric_name']
        
        # Delete related data first (to avoid foreign key constraints)
        # Delete time series data
        cursor.execute("""
            DELETE FROM time_series_data 
            WHERE chart_id IN (
                SELECT chart_id FROM time_series_metadata 
                WHERE object_id = %s AND object_type LIKE '%%metric%%'
            )
        """, (str(metric_id),))
        
        # Delete time series metadata
        cursor.execute("""
            DELETE FROM time_series_metadata 
            WHERE object_id = %s AND object_type LIKE '%%metric%%'
        """, (str(metric_id),))
        
        # Delete anomalies
        cursor.execute("""
            DELETE FROM anomalies 
            WHERE object_id = %s AND object_type LIKE '%%metric%%'
        """, (str(metric_id),))
        
        # Delete the metric itself
        cursor.execute("DELETE FROM metrics WHERE id = %s", (metric_id,))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully hard deleted metric {metric_id}: {metric_name}")
        
        return {
            "status": "success",
            "message": f"Metric '{metric_name}' (ID: {metric_id}) has been permanently deleted",
            "metric_id": metric_id,
            "metric_name": metric_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error hard deleting metric {metric_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error hard deleting metric: {str(e)}") 
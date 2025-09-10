"""
Metrics Manager - Database-only endpoints for metric control functionality
"""

import json
import logging
import os
import tempfile
from fastapi import APIRouter, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import psycopg2.extras

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/api/enhanced-queries")
async def get_enhanced_queries_db():
    """Serve enhanced dashboard queries built entirely from database."""
    try:
        from tools.db_utils import get_pooled_connection
        
        with get_pooled_connection() as connection:
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
                    m.is_active,
                    m.display_order,
                    m.most_recent_data_date,
                    d.title as dataset_title_from_datasets,
                    d.category as dataset_category_from_datasets,
                    d.columns as dataset_columns
                FROM metrics m
                LEFT JOIN datasets d ON m.endpoint = d.endpoint AND d.is_active = true
                WHERE m.is_active = true
                ORDER BY m.display_order NULLS LAST, m.id
            """)
            
            metrics_rows = cursor.fetchall()
        
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
                "category_fields": row['category_fields'] or [],
                "is_active": row['is_active'] if row['is_active'] is not None else True,
                "display_order": row['display_order'],
                "most_recent_data_date": row['most_recent_data_date'].isoformat() if row['most_recent_data_date'] else None,
                "metadata": row['metadata']
            }
            
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
        import json
        
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
            # Parse the JSON string if it's a string, otherwise use as-is
            category_fields = row['category_fields']
            if isinstance(category_fields, str):
                try:
                    category_fields = json.loads(category_fields)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing category_fields JSON for endpoint '{endpoint}': {e}")
                    return JSONResponse({
                        "status": "success",
                        "columns": []
                    })
            
            # Extract field names from category_fields
            selected_columns = [field["fieldName"] for field in category_fields if field.get("fieldName")]
            
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
        
        # Import asyncio for background processing
        import asyncio
        
        # Determine which script to run based on period_type
        if period_type == 'ytd':
            # For YTD metrics, use generate_dashboard_metrics.py
            logger.info(f"Running YTD dashboard metrics generation for metric ID {metric_id}")
            
            try:
                from generate_dashboard_metrics import process_single_metric
                
                if callable(process_single_metric):
                    # Run in background thread to prevent blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, process_single_metric, metric_id, period_type)
                    logger.info(f"YTD metrics generation completed for metric ID {metric_id}")
                else:
                    from generate_dashboard_metrics import main as generate_all_metrics
                    # Run in background thread to prevent blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, generate_all_metrics)
                    logger.info(f"All YTD metrics generated (including metric ID {metric_id})")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import specific metric generation function: {str(e)}")
                from generate_dashboard_metrics import main as generate_all_metrics
                # Run in background thread to prevent blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, generate_all_metrics)
                logger.info(f"All YTD metrics generated (including metric ID {metric_id})")
        
        elif period_type == 'week':
            # For weekly analysis, use generate_weekly_analysis.py
            logger.info(f"Running weekly analysis for metric ID {metric_id}")
            
            try:
                from tools.analysis.weekly import run_weekly_analysis
                # from tools.analysis.weekly import generate_weekly_newsletter
                
                # Run in background thread to prevent blocking
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, 
                    run_weekly_analysis,
                    [str(metric_id)],
                    (district_id == 0)
                )
                
                # Newsletter generation temporarily disabled
                # newsletter_path = generate_weekly_newsletter(results)
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
                
                # Run in background thread to prevent blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    process_metric_analysis,
                    metric_info,
                    period_type,
                    True  # process_districts
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


@router.post("/run_specific_metric_async")
async def run_specific_metric_async(request: Request):
    """Run analysis for a specific metric asynchronously with job tracking."""
    try:
        body = await request.json()
        metric_id = body.get('metric_id')
        district_id = body.get('district_id', 0)
        period_type = body.get('period_type', 'year')
        
        if not metric_id:
            return JSONResponse({
                "status": "error",
                "message": "metric_id is required"
            }, status_code=400)
        
        logger.info(f"Run specific metric async called for metric_id: {metric_id}, district_id: {district_id}, period_type: {period_type}")
        
        # Import the background job manager
        from background_jobs import job_manager
        
        # Create a background job
        job_id = job_manager.create_job(
            "metric_analysis", 
            f"Analysis for metric {metric_id} ({period_type})"
        )
        
        # Define the function to run in the background
        def run_metric_analysis():
            # This will be the same logic as the sync version but without the async/await
            from tools.db_utils import get_postgres_connection
            import psycopg2.extras
            
            connection = get_postgres_connection()
            if not connection:
                raise Exception("Database connection failed")
            
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
                raise Exception(f"Metric ID {metric_id} not found")
            
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
            
            # Run the appropriate analysis based on period_type
            if period_type == 'ytd':
                from generate_dashboard_metrics import process_single_metric
                if callable(process_single_metric):
                    return process_single_metric(metric_id=metric_id, period_type=period_type)
                else:
                    from generate_dashboard_metrics import main as generate_all_metrics
                    return generate_all_metrics()
            elif period_type == 'week':
                from tools.analysis.weekly import run_weekly_analysis
                return run_weekly_analysis(
                    metrics_list=[str(metric_id)],
                    process_districts=True  # Always process districts for weekly analysis
                )
            else:
                from generate_metric_analysis import process_metric_analysis
                return process_metric_analysis(
                    metric_info=metric_info,
                    period_type=period_type,
                    process_districts=True
                )
        
        # Start the job in the background
        import asyncio
        asyncio.create_task(job_manager.run_job(job_id, run_metric_analysis))
        
        return JSONResponse({
            "status": "success",
            "message": f"Started {period_type} analysis for metric {metric_id}",
            "job_id": job_id
        })
        
    except Exception as e:
        logger.error(f"Error starting specific metric analysis: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error starting metric analysis: {str(e)}"
        }, status_code=500)


@router.post("/api/metrics")
async def create_new_metric(request: Request):
    """Create a new metric in the database."""
    try:
        from tools.db_utils import get_postgres_connection
        
        data = await request.json()
        
        # Validate required fields
        required_fields = ['name', 'key', 'category', 'endpoint']
        for field in required_fields:
            if not data.get(field):
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Insert the new metric
        cursor.execute("""
            INSERT INTO metrics (
                metric_name, metric_key, category, subcategory, endpoint,
                summary, definition, data_sf_url, ytd_query, metric_query,
                dataset_title, dataset_category, show_on_dash, item_noun,
                location_fields, category_fields, metadata, greendirection, is_active,
                map_query, map_filters, map_config
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            data['name'],
            data['key'],
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
            data.get('greendirection', 'up'),
            True,
            data.get('map_query', ''),
            json.dumps(data.get('map_filters', {})),
            json.dumps(data.get('map_config', {}))
        ))
        
        new_metric_id = cursor.fetchone()['id']
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Created new metric with ID {new_metric_id}: {data['name']}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Created metric: {data['name']}",
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


@router.post("/api/migrate-dashboard-to-metrics")
async def migrate_dashboard_to_metrics_api():
    """Run the dashboard to metrics migration script."""
    try:
        logger.info("Starting dashboard to metrics migration via API")
        
        # Import and run the migration function
        import sys
        import os
        
        # Add the tools directory to the Python path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tools_dir = os.path.join(script_dir, 'tools')
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        
        try:
            from migrate_dashboard_to_metrics import migrate_dashboard_to_metrics
            
            # Run the migration
            success = migrate_dashboard_to_metrics()
            
            if success:
                logger.info("Dashboard migration completed successfully")
                return JSONResponse({
                    "status": "success",
                    "message": "Dashboard queries successfully migrated to metrics table"
                })
            else:
                logger.error("Dashboard migration failed")
                return JSONResponse({
                    "status": "error",
                    "message": "Dashboard migration failed - check logs for details"
                }, status_code=500)
                
        except ImportError as e:
            logger.error(f"Could not import migration script: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Could not import migration script: {str(e)}"
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error during dashboard migration: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error during migration: {str(e)}"
        }, status_code=500)


@router.put("/api/metrics/{metric_identifier}")
async def edit_metric(metric_identifier: str, request: Request):
    """Update an existing metric."""
    try:
        from tools.db_utils import get_postgres_connection
        
        data = await request.json()
        logger.info(f"Updating metric {metric_identifier} with data: {data}")
        logger.info(f"Data types: {[(k, type(v)) for k, v in data.items()]}")
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Try to convert to integer for ID lookup
        try:
            metric_id = int(metric_identifier)
            cursor.execute("SELECT id, metric_name FROM metrics WHERE id = %s", (metric_id,))
        except ValueError:
            # If not an integer, treat as metric_key
            cursor.execute("SELECT id, metric_name FROM metrics WHERE metric_key = %s", (metric_identifier,))
        
        metric = cursor.fetchone()
        if not metric:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_identifier}")
        
        # Build update query dynamically based on provided fields
        update_fields = []
        params = []
        
        logger.info(f"Processing update fields: {list(data.keys())}")
        logger.info(f"Raw data: {data}")
        logger.info(f"Raw data types: {[(k, type(v), str(v)[:100]) for k, v in data.items()]}")
        
        for field, value in data.items():
            logger.info(f"Processing field: {field} = {value} (type: {type(value)})")
            if field in ['name', 'category', 'subcategory', 'endpoint', 'summary', 'definition',
                        'data_sf_url', 'ytd_query', 'metric_query', 'dataset_title', 'dataset_category',
                        'show_on_dash', 'item_noun', 'greendirection', 'is_active', 'map_query']:
                if field == 'name':
                    update_fields.append("metric_name = %s")
                    params.append(value)
                    logger.info(f"Added metric_name = %s with value: {value}")
                elif field == 'id':
                    logger.info(f"Skipping field 'id' as it's not allowed to be updated")
                    continue  # Don't allow updating the ID
                elif field == 'key':
                    logger.info(f"Skipping field 'key' as it's not allowed to be updated")
                    continue  # Don't allow updating the key
                elif field in ['location_fields', 'category_fields', 'metadata', 'map_filters', 'map_config']:
                    update_fields.append(f"{field} = %s")
                    logger.info(f"Processing JSON field {field} with value type: {type(value)}")
                    logger.info(f"JSON field {field} value: {value}")
                    json_value = json.dumps(value)
                    params.append(json_value)
                    logger.info(f"Added {field} = %s with JSON value: {json_value}")
                else:
                    update_fields.append(f"{field} = %s")
                    params.append(value)
                    logger.info(f"Added {field} = %s with value: {value}")
            else:
                logger.warning(f"Ignoring unknown field: {field}")
        
        logger.info(f"Update fields: {update_fields}")
        logger.info(f"Parameters before adding metric ID: {params}")
        logger.info(f"Number of parameters before adding metric ID: {len(params)}")
        
        if not update_fields:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        logger.info(f"Added updated_at = CURRENT_TIMESTAMP (no parameter needed)")
        
        query = f"""
            UPDATE metrics 
            SET {', '.join(update_fields)}
            WHERE id = %s
            RETURNING id, metric_name
        """
        
        # Add the metric ID as the last parameter for the WHERE clause
        params.append(metric['id'])
        
        logger.info(f"Final query: {query}")
        logger.info(f"Final parameters: {params}")
        logger.info(f"Number of %s placeholders in query: {query.count('%s')}")
        logger.info(f"Number of parameters: {len(params)}")
        
        # Verify parameter count matches placeholder count
        placeholder_count = query.count('%s')
        if len(params) != placeholder_count:
            raise HTTPException(
                status_code=500, 
                detail=f"Parameter count mismatch: {len(params)} parameters for {placeholder_count} placeholders"
            )
        
        try:
            cursor.execute(query, params)
            logger.info("Query executed successfully")
        except Exception as e:
            logger.error(f"Database error during execute: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {params}")
            raise
        
        updated_metric = cursor.fetchone()
        
        if not updated_metric:
            connection.rollback()
            cursor.close()
            connection.close()
            raise HTTPException(status_code=500, detail="Failed to update metric")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully updated metric {metric_identifier}: {updated_metric['metric_name']}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Updated metric: {updated_metric['metric_name']}",
            "metric_id": updated_metric['id']
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating metric {metric_identifier}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating metric: {str(e)}")


@router.post("/api/metrics/{metric_identifier}/disable")
async def disable_metric(metric_identifier: str):
    """Deactivate a metric (soft delete)."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Try to convert to integer for ID lookup
        try:
            metric_id = int(metric_identifier)
            cursor.execute("SELECT id, metric_name FROM metrics WHERE id = %s", (metric_id,))
        except ValueError:
            # If not an integer, treat as metric_key
            cursor.execute("SELECT id, metric_name FROM metrics WHERE metric_key = %s", (metric_identifier,))
        
        metric = cursor.fetchone()
        if not metric:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_identifier}")
        
        cursor.execute("""
            UPDATE metrics 
            SET is_active = false, updated_at = CURRENT_TIMESTAMP 
            WHERE id = %s
            RETURNING id, metric_name
        """, (metric['id'],))
        
        updated_metric = cursor.fetchone()
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully disabled metric {metric_identifier}: {updated_metric['metric_name']}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Disabled metric: {updated_metric['metric_name']}",
            "metric_id": updated_metric['id']
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling metric {metric_identifier}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error disabling metric: {str(e)}")


@router.post("/api/metrics/{metric_identifier}/enable")
async def enable_metric(metric_identifier: str):
    """Reactivate a previously disabled metric."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Try to convert to integer for ID lookup
        try:
            metric_id = int(metric_identifier)
            cursor.execute("SELECT id, metric_name FROM metrics WHERE id = %s", (metric_id,))
        except ValueError:
            # If not an integer, treat as metric_key
            cursor.execute("SELECT id, metric_name FROM metrics WHERE metric_key = %s", (metric_identifier,))
        
        metric = cursor.fetchone()
        if not metric:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_identifier}")
        
        cursor.execute("""
            UPDATE metrics 
            SET is_active = true, updated_at = CURRENT_TIMESTAMP 
            WHERE id = %s
            RETURNING id, metric_name
        """, (metric['id'],))
        
        updated_metric = cursor.fetchone()
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully enabled metric {metric_identifier}: {updated_metric['metric_name']}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Enabled metric: {updated_metric['metric_name']}",
            "metric_id": updated_metric['id']
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling metric {metric_identifier}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enabling metric: {str(e)}")


@router.get("/api/metrics/by-endpoint/{endpoint}")
async def find_metrics_by_endpoint(endpoint: str):
    """Find all metrics using a specific DataSF endpoint."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                id, metric_name, metric_key, category, subcategory,
                summary, definition, show_on_dash, is_active
            FROM metrics 
            WHERE endpoint = %s
            ORDER BY category, subcategory, id
        """, (endpoint,))
        
        metrics = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return JSONResponse({
            "status": "success",
            "endpoint": endpoint,
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error finding metrics by endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding metrics by endpoint: {str(e)}")


@router.get("/api/metrics")
async def query_metrics(
    category: str = None,
    search_term: str = None,
    active_only: bool = True,
    dashboard_only: bool = False
):
    """Search and filter metrics in the database."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build the query
        query = """
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
                m.is_active,
                m.most_recent_data_date
            FROM metrics m
            WHERE 1=1
        """
        params = []
        
        if category:
            query += " AND m.category = %s"
            params.append(category)
        
        if search_term:
            query += """ AND (
                m.metric_name ILIKE %s OR
                m.metric_key ILIKE %s OR
                m.summary ILIKE %s OR
                m.definition ILIKE %s
            )"""
            search_pattern = f"%{search_term}%"
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern])
        
        if active_only:
            query += " AND m.is_active = true"
        
        if dashboard_only:
            query += " AND m.show_on_dash = true"
        
        query += " ORDER BY m.category, m.subcategory, m.id"
        
        cursor.execute(query, params)
        metrics = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Convert date objects to strings for JSON serialization
        for metric in metrics:
            if metric.get('most_recent_data_date'):
                metric['most_recent_data_date'] = metric['most_recent_data_date'].isoformat()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error querying metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying metrics: {str(e)}")


@router.get("/api/metric/{metric_identifier}")
async def get_metric_details(metric_identifier: str):
    """Get detailed information about a specific metric by ID or key."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Try to convert to integer for ID lookup
        try:
            metric_id = int(metric_identifier)
            cursor.execute("""
                SELECT * FROM metrics 
                WHERE id = %s
            """, (metric_id,))
        except ValueError:
            # If not an integer, treat as metric_key
            cursor.execute("""
                SELECT * FROM metrics 
                WHERE metric_key = %s
            """, (metric_identifier,))
        
        metric = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not metric:
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_identifier}")
        
        return JSONResponse({
            "status": "success",
            "metric": metric
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metric details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metric details: {str(e)}")


@router.get("/api/metrics/categories")
async def list_categories():
    """Get all available metric categories and subcategories."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT category, subcategory 
            FROM metrics 
            WHERE is_active = true 
            ORDER BY category, subcategory
        """)
        
        categories = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Group by category
        category_map = {}
        for row in categories:
            if row['category'] not in category_map:
                category_map[row['category']] = []
            if row['subcategory'] and row['subcategory'] != row['category']:
                category_map[row['category']].append(row['subcategory'])
        
        return JSONResponse({
            "status": "success",
            "categories": category_map
        })
        
    except Exception as e:
        logger.error(f"Error listing categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing categories: {str(e)}")


@router.get("/api/metrics/overview")
async def get_metrics_overview():
    """Get summary statistics about the metrics system."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # First gather high-level counts
        cursor.execute(
            """
            SELECT 
                COUNT(*)                                 AS total_metrics,
                COUNT(*) FILTER (WHERE is_active = true) AS active_metrics,
                COUNT(*) FILTER (WHERE show_on_dash = true) AS dashboard_metrics,
                COUNT(DISTINCT category)                 AS total_categories,
                COUNT(DISTINCT endpoint)                 AS total_endpoints
            FROM metrics
            """
        )

        overview = cursor.fetchone()

        # Next, fetch a concise list of each metric's key details
        cursor.execute(
            """
            SELECT 
                metric_name,
                endpoint,
                metric_query,
                ytd_query
            FROM metrics
            WHERE is_active = true
            ORDER BY metric_name
            """
        )

        metric_details = cursor.fetchall()

        cursor.close()
        connection.close()

        return JSONResponse({
            "status": "success",
            "overview": overview,
            "metrics": metric_details
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics overview: {str(e)}")


@router.post("/api/restore-metrics-from-backup")
async def restore_metrics_from_backup_api(backup_file: UploadFile = File(...)):
    """Run the metrics table restore script with uploaded file."""
    try:
        logger.info(f"Starting metrics table restore via API with file: {backup_file.filename}")
        
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sql")
        try:
            content = await backup_file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file.close()
            
            logger.info(f"Saved uploaded file to temporary location: {temp_file.name}")
            
            # Import and run the restore function
            import sys
            import os
            
            # Add the tools directory to the Python path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tools_dir = os.path.join(script_dir, 'tools')
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            
            try:
                from migrate_dashboard_to_metrics import restore_metrics_from_backup
                
                # Run the restore with the uploaded file
                success = restore_metrics_from_backup(temp_file.name)
                
                if success:
                    logger.info("Metrics table restore completed successfully")
                    return JSONResponse({
                        "status": "success",
                        "message": f"Metrics table successfully restored from {backup_file.filename}"
                    })
                else:
                    logger.error("Metrics table restore failed")
                    return JSONResponse({
                        "status": "error",
                        "message": "Metrics table restore failed - check logs for details"
                    }, status_code=500)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                    logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file.name}: {e}")
                    
        except Exception as e:
            # Make sure to close temp file on error
            try:
                temp_file.close()
                os.unlink(temp_file.name)
            except:
                pass
            raise e
                
        except ImportError as e:
            logger.error(f"Could not import restore script: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Could not import restore script: {str(e)}"
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error during metrics table restore: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error during restore: {str(e)}"
        }, status_code=500)


@router.post("/api/backup-metrics-table")
async def backup_metrics_table_api():
    """Create a backup of the metrics table, moving any existing backup to history."""
    try:
        import subprocess
        import os
        import shutil
        from datetime import datetime
        
        logger.info("Starting metrics table backup via API")
        
        # Get project root directory (3 levels up from this file)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Define paths
        backup_file = os.path.join(project_root, 'metrics_backup.sql')
        history_dir = os.path.join(project_root, 'backup_history')
        
        # Create history directory if it doesn't exist
        os.makedirs(history_dir, exist_ok=True)
        
        # Move existing backup to history if it exists
        old_backup_moved = False
        old_backup_location = None
        if os.path.exists(backup_file):
            # Get the modification time of the existing backup
            mod_time = datetime.fromtimestamp(os.path.getmtime(backup_file))
            timestamp = mod_time.strftime("%Y%m%d_%H%M%S")
            
            # Move to history with timestamp
            old_backup_location = os.path.join(history_dir, f'metrics_backup_{timestamp}.sql')
            shutil.move(backup_file, old_backup_location)
            old_backup_moved = True
            logger.info(f"Moved existing backup to: {old_backup_location}")
        
        # Check if DATABASE_URL is provided (common for managed services like Replit PostgreSQL)
        database_url = os.getenv("DATABASE_URL")
        
        # Set up environment for pg_dump
        env = os.environ.copy()
        
        if database_url:
            # Use DATABASE_URL for managed database
            logger.info("Using DATABASE_URL for backup")
            dump_cmd = [
                "pg_dump",
                database_url,
                "--table=public.metrics",
                "--inserts",
                "--column-inserts",
                "--no-owner",
                "-f", backup_file
            ]
        else:
            # Fall back to individual parameters for local development
            logger.info("Using individual database parameters for backup")
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            db_name = os.getenv("POSTGRES_DB", "transparentsf")
            db_user = os.getenv("POSTGRES_USER", "postgres")
            db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
            
            if db_password:
                env["PGPASSWORD"] = db_password
            
            # Create the backup using pg_dump with individual parameters
            dump_cmd = [
                "pg_dump",
                "-h", db_host,
                "-p", str(db_port),
                "-U", db_user,
                "-d", db_name,
                "--table=public.metrics",
                "--inserts",
                "--column-inserts",
                "--no-owner",
                "-f", backup_file
            ]
        
        logger.info("Creating new backup of metrics table...")
        
        # Try pg_dump first
        try:
            result = subprocess.run(dump_cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            # pg_dump failed or is not available, try Python backup
            logger.warning(f"pg_dump failed or not available: {str(e)}")
            logger.info("Falling back to Python-based metrics backup")
            
            try:
                # Create Python-based backup for metrics table
                connection = get_postgres_connection()
                if not connection:
                    raise Exception("Could not connect to database")
                
                cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                with open(backup_file, 'w', encoding='utf-8') as backup_f:
                    # Write header
                    backup_f.write("-- Metrics table backup created with Python/psycopg2\n")
                    backup_f.write(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Get table structure
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = 'metrics'
                        ORDER BY ordinal_position;
                    """)
                    columns = cursor.fetchall()
                    
                    # Write DROP TABLE statement
                    backup_f.write("DROP TABLE IF EXISTS \"metrics\" CASCADE;\n")
                    
                    # Write CREATE TABLE statement
                    backup_f.write("CREATE TABLE \"metrics\" (\n")
                    
                    column_definitions = []
                    for col in columns:
                        col_def = f'    "{col["column_name"]}" {col["data_type"]}'
                        if col["is_nullable"] == "NO":
                            col_def += " NOT NULL"
                        if col["column_default"]:
                            col_def += f" DEFAULT {col['column_default']}"
                        column_definitions.append(col_def)
                    
                    backup_f.write(",\n".join(column_definitions))
                    backup_f.write("\n);\n\n")
                    
                    # Get and write table data
                    cursor.execute('SELECT * FROM "metrics"')
                    rows = cursor.fetchall()
                    
                    if rows:
                        # Get column names
                        column_names = [col['column_name'] for col in columns]
                        
                        # Write INSERT statements
                        for row in rows:
                            values = []
                            for col_name in column_names:
                                value = row[col_name]
                                if value is None:
                                    values.append('NULL')
                                elif isinstance(value, str):
                                    # Escape single quotes in strings
                                    escaped_value = value.replace("'", "''")
                                    values.append(f"'{escaped_value}'")
                                elif isinstance(value, (int, float)):
                                    values.append(str(value))
                                elif isinstance(value, bool):
                                    values.append('TRUE' if value else 'FALSE')
                                else:
                                    # For other types, convert to string and escape
                                    escaped_value = str(value).replace("'", "''")
                                    values.append(f"'{escaped_value}'")
                            
                            column_list = ', '.join([f'"{name}"' for name in column_names])
                            value_list = ', '.join(values)
                            backup_f.write(f'INSERT INTO "metrics" ({column_list}) VALUES ({value_list});\n')
                        
                        backup_f.write("\n")
                    
                    # Get and write indexes
                    cursor.execute("""
                        SELECT indexname, indexdef
                        FROM pg_indexes 
                        WHERE schemaname = 'public' 
                        AND tablename = 'metrics'
                        AND indexname NOT LIKE '%_pkey';
                    """)
                    indexes = cursor.fetchall()
                    
                    for index in indexes:
                        backup_f.write(f"{index['indexdef']};\n")
                    
                    backup_f.write("\n-- Backup completed\n")
                
                cursor.close()
                connection.close()
                logger.info(f"Python metrics backup completed: {backup_file}")
                
            except Exception as python_backup_error:
                logger.error(f"Python metrics backup also failed: {str(python_backup_error)}")
                return JSONResponse({
                    "status": "error",
                    "message": f"Both pg_dump and Python backup failed. pg_dump error: {str(e)}, Python backup error: {str(python_backup_error)}"
                }, status_code=500)
        
        # Verify the backup file was created
        if not os.path.exists(backup_file):
            logger.error("Backup file was not created")
            return JSONResponse({
                "status": "error",
                "message": "Backup file was not created"
            }, status_code=500)
        
        # Get file size for confirmation
        file_size = os.path.getsize(backup_file)
        logger.info(f"Backup created successfully: {backup_file} ({file_size} bytes)")
        
        response_data = {
            "status": "success",
            "message": "Metrics table backup created successfully",
            "backup_file": backup_file,
            "backup_size_bytes": file_size,
            "old_backup_moved": old_backup_moved,
            "filename": "metrics_backup.sql",
            "download_url": "/api/download-metrics-backup"
        }
        
        if old_backup_moved:
            response_data["old_backup_location"] = old_backup_location
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error during metrics table backup: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error during backup: {str(e)}"
        }, status_code=500)


@router.get("/api/download-metrics-backup")
async def download_metrics_backup():
    """Download the metrics backup file."""
    logger.debug("Download metrics backup called")
    
    try:
        # Get project root directory (3 levels up from this file)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Define backup file path
        backup_file = os.path.join(project_root, 'metrics_backup.sql')
        
        if not os.path.exists(backup_file):
            raise HTTPException(status_code=404, detail="Metrics backup file not found")
        
        return FileResponse(
            backup_file,
            media_type='application/octet-stream',
            filename='metrics_backup.sql'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading metrics backup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metrics-with-order")
async def get_metrics_with_order():
    """Get all metrics with their current display_order for reordering interface."""
    try:
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query all metrics with their display_order, including inactive ones for reordering
        cursor.execute("""
            SELECT 
                id,
                metric_name as name,
                category,
                subcategory,
                endpoint,
                is_active,
                display_order,
                most_recent_data_date
            FROM metrics 
            ORDER BY COALESCE(display_order, 999), id
        """)
        
        metrics = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Convert date objects to strings for JSON serialization
        for metric in metrics:
            if metric.get('most_recent_data_date'):
                metric['most_recent_data_date'] = metric['most_recent_data_date'].isoformat()
        
        return JSONResponse({
            "status": "success",
            "metrics": [dict(metric) for metric in metrics]
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics with order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics with order: {str(e)}")


@router.post("/api/update-metric-order")
async def update_metric_order(request: Request):
    """Update the display_order for multiple metrics."""
    try:
        from tools.db_utils import get_postgres_connection
        
        data = await request.json()
        metrics = data.get('metrics', [])
        
        if not metrics:
            raise HTTPException(status_code=400, detail="No metrics provided")
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        
        # Update each metric's display_order
        for metric in metrics:
            metric_id = metric.get('id')
            display_order = metric.get('display_order')
            
            if metric_id is None or display_order is None:
                continue
                
            cursor.execute("""
                UPDATE metrics 
                SET display_order = %s 
                WHERE id = %s
            """, (display_order, metric_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return JSONResponse({
            "status": "success",
            "message": f"Updated display order for {len(metrics)} metrics"
        })
        
    except Exception as e:
        logger.error(f"Error updating metric order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating metric order: {str(e)}") 
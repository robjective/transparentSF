from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import logging
from typing import Optional

# Import centralized database utilities
from ai.tools.db_utils import get_postgres_connection, execute_with_connection

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate-dw-time-series")
async def generate_dw_time_series_chart(chart_id: int = Query(..., description="Chart ID to generate DW version for")):
    """
    Generate a Datawrapper version of a time series chart using the chart_id.
    This mimics the behavior in monthly_report.py expand_chart_references function.
    """
    try:
        logger.info(f"Generating DW time series chart for chart_id: {chart_id}")
        
        # Import the tools we need (same as in monthly_report.py)
        from tools.store_time_series import get_time_series_metadata, get_time_series_data
        from tools.genChartdw import create_time_series_chart_from_data
        
        # Get the metadata for this chart
        metadata_df = get_time_series_metadata(chart_id=int(chart_id))
        if metadata_df.empty:
            logger.error(f"No metadata found for time_series_id: {chart_id}")
            raise HTTPException(status_code=404, detail=f"No metadata found for time_series_id: {chart_id}")
        
        metadata_row = metadata_df.iloc[0]
        
        # Get the time series data
        data_df = get_time_series_data(chart_id=int(chart_id))
        if data_df.empty:
            logger.error(f"No data found for time_series_id: {chart_id}")
            raise HTTPException(status_code=404, detail=f"No data found for time_series_id: {chart_id}")
        
        # Extract metadata information (same logic as in monthly_report.py)
        metadata_json = metadata_row.get('metadata', {})
        if isinstance(metadata_json, str):
            try:
                import json
                metadata_json = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata_json = {}
        
        # Try multiple sources for the chart title with proper fallback
        chart_title = (
            metadata_json.get('title') or         # First: title from metadata JSONB
            metadata_json.get('chart_title') or   # Second: chart_title from metadata JSONB  
            metadata_row.get('object_name') or    # Third: object_name from table
            f"Time Series Chart {chart_id}"       # Fallback
        )
        
        object_name = metadata_row.get('object_name', 'Unknown')
        field_name = metadata_row.get('field_name', 'Value')
        period_type = metadata_row.get('period_type', 'month')
        district = metadata_row.get('district', 0)
        
        # Prepare chart data in the format expected by create_time_series_chart_from_data
        chart_data = []
        for _, row in data_df.iterrows():
            chart_data.append({
                'time_period': row['time_period'],
                'value': row['numeric_value'],
                'group_value': row.get('group_value')
            })
        
        # Create chart metadata
        chart_metadata = {
            'title': chart_title,
            'object_name': object_name,
            'field_name': field_name,
            'period_type': period_type,
            'district': district,
            'chart_id': chart_id,
            'executed_query_url': metadata_row.get('executed_query_url', ''),
            'source-name': 'DataSF',
            'byline': 'Chart: TransparentSF'
        }
        
        # Generate the Datawrapper chart
        dw_chart_url = create_time_series_chart_from_data(
            chart_data=chart_data,
            metadata=chart_metadata
        )
        
        if dw_chart_url:
            logger.info(f"Successfully generated DW time series chart for chart_id {chart_id}: {dw_chart_url}")
            return JSONResponse({
                "status": "success",
                "message": f"Successfully generated DW time series chart for chart_id {chart_id}",
                "chart_url": dw_chart_url,
                "chart_id": chart_id
            })
        else:
            logger.error(f"Failed to generate DW time series chart for chart_id {chart_id}")
            raise HTTPException(status_code=500, detail=f"Failed to generate DW time series chart for chart_id {chart_id}")
            
    except Exception as e:
        logger.error(f"Error generating DW time series chart for chart_id {chart_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating DW time series chart: {str(e)}")


@router.post("/generate-dw-anomaly")
async def generate_dw_anomaly_chart(anomaly_id: int = Query(..., description="Anomaly ID to generate DW version for")):
    """
    Generate a Datawrapper version of an anomaly chart using the anomaly_id.
    This mimics the behavior in monthly_report.py expand_chart_references function.
    """
    try:
        logger.info(f"Generating DW anomaly chart for anomaly_id: {anomaly_id}")
        
        # Use the same helper function as in monthly_report.py
        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
        
        chart_url = generate_anomaly_chart_from_id(anomaly_id)
        
        if chart_url:
            logger.info(f"Successfully generated DW anomaly chart for anomaly_id {anomaly_id}: {chart_url}")
            return JSONResponse({
                "status": "success",
                "message": f"Successfully generated DW anomaly chart for anomaly_id {anomaly_id}",
                "chart_url": chart_url,
                "anomaly_id": anomaly_id,
                "source-name": "DataSF",
                "byline": "Chart: TransparentSF"
            })
        else:
            logger.error(f"Failed to generate DW anomaly chart for anomaly_id {anomaly_id}")
            raise HTTPException(status_code=500, detail=f"Failed to generate DW anomaly chart for anomaly_id {anomaly_id}")
            
    except Exception as e:
        logger.error(f"Error generating DW anomaly chart for anomaly_id {anomaly_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating DW anomaly chart: {str(e)}")


@router.post("/generate-dw-charts-batch")
async def generate_dw_charts_batch(
    metric_id: int = Query(..., description="Metric ID to generate charts for"),
    district_id: str = Query("0", description="District ID"),
    period_type: str = Query("month", description="Period type (month/year)"),
    chart_types: str = Query("time-series,anomaly", description="Comma-separated chart types to generate")
):
    """
    Generate multiple DW charts for a metric in batch.
    This is a convenience endpoint that can generate both time series and anomaly charts.
    """
    try:
        logger.info(f"Generating DW charts batch for metric_id: {metric_id}, district: {district_id}, period: {period_type}")
        
        chart_types_list = [ct.strip() for ct in chart_types.split(",")]
        results = {
            "status": "success",
            "metric_id": metric_id,
            "district_id": district_id,
            "period_type": period_type,
            "generated_charts": []
        }
        
        # Generate time series charts if requested
        if "time-series" in chart_types_list:
            try:
                # Get active charts for this metric
                import requests
                import os
                
                api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
                charts_response = requests.get(f"{api_base_url}/backend/api/active-charts?metric_id={metric_id}&district={district_id}&period_type={period_type}")
                
                if charts_response.ok:
                    charts_data = charts_response.json()
                    charts = charts_data.get("charts", [])
                    
                    for chart in charts:
                        chart_id = chart.get("chart_id")
                        if chart_id:
                            try:
                                # Call our own endpoint to generate the DW chart
                                dw_response = await generate_dw_time_series_chart(chart_id)
                                if dw_response.status_code == 200:
                                    response_data = dw_response.body.decode()
                                    import json
                                    chart_result = json.loads(response_data)
                                    results["generated_charts"].append({
                                        "type": "time-series",
                                        "chart_id": chart_id,
                                        "url": chart_result.get("chart_url"),
                                        "status": "success"
                                    })
                                else:
                                    results["generated_charts"].append({
                                        "type": "time-series",
                                        "chart_id": chart_id,
                                        "status": "failed",
                                        "error": "Failed to generate DW chart"
                                    })
                            except Exception as e:
                                logger.error(f"Error generating DW chart for chart_id {chart_id}: {str(e)}")
                                results["generated_charts"].append({
                                    "type": "time-series",
                                    "chart_id": chart_id,
                                    "status": "failed",
                                    "error": str(e)
                                })
                else:
                    logger.warning(f"Failed to fetch active charts for metric {metric_id}")
                    
            except Exception as e:
                logger.error(f"Error processing time series charts: {str(e)}")
        
        # Generate anomaly charts if requested
        if "anomaly" in chart_types_list:
            try:
                # Get anomalies for this metric
                import requests
                import os
                
                api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
                anomalies_response = requests.get(f"{api_base_url}/anomaly-analyzer/api/query-anomalies?object_id={metric_id}&district={district_id}&period_type={period_type}&only_anomalies=true&only_active=true&limit=10")
                
                if anomalies_response.ok:
                    anomalies_data = anomalies_response.json()
                    anomalies = anomalies_data.get("anomalies", [])
                    
                    for anomaly in anomalies:
                        anomaly_id = anomaly.get("id")
                        if anomaly_id:
                            try:
                                # Call our own endpoint to generate the DW chart
                                dw_response = await generate_dw_anomaly_chart(anomaly_id)
                                if dw_response.status_code == 200:
                                    response_data = dw_response.body.decode()
                                    import json
                                    chart_result = json.loads(response_data)
                                    results["generated_charts"].append({
                                        "type": "anomaly",
                                        "anomaly_id": anomaly_id,
                                        "url": chart_result.get("chart_url"),
                                        "status": "success"
                                    })
                                else:
                                    results["generated_charts"].append({
                                        "type": "anomaly",
                                        "anomaly_id": anomaly_id,
                                        "status": "failed",
                                        "error": "Failed to generate DW chart"
                                    })
                            except Exception as e:
                                logger.error(f"Error generating DW chart for anomaly_id {anomaly_id}: {str(e)}")
                                results["generated_charts"].append({
                                    "type": "anomaly",
                                    "anomaly_id": anomaly_id,
                                    "status": "failed",
                                    "error": str(e)
                                })
                else:
                    logger.warning(f"Failed to fetch anomalies for metric {metric_id}")
                    
            except Exception as e:
                logger.error(f"Error processing anomaly charts: {str(e)}")
        
        # Check if any charts were generated
        successful_charts = [c for c in results["generated_charts"] if c.get("status") == "success"]
        if successful_charts:
            results["message"] = f"Successfully generated {len(successful_charts)} DW chart(s)"
        else:
            results["status"] = "warning"
            results["message"] = "No charts were successfully generated"
        
        return JSONResponse(results)
        
    except Exception as e:
        logger.error(f"Error in batch DW chart generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in batch DW chart generation: {str(e)}")


@router.post("/update-chart-metadata")
async def update_chart_metadata(request: Request):
    """
    Update chart metadata with DW URL.
    """
    try:
        data = await request.json()
        chart_id = data.get("chart_id")
        chart_type = data.get("chart_type")
        dw_url = data.get("dw_url")
        
        if not chart_id or not dw_url:
            raise HTTPException(status_code=400, detail="chart_id and dw_url are required")
        
        logger.info(f"Updating chart metadata for chart_id: {chart_id}, type: {chart_type}, dw_url: {dw_url}")
        
        # Import database utilities
        import psycopg2
        import psycopg2.extras
        import os
        import json
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get current metadata
        cursor.execute("SELECT metadata FROM time_series_metadata WHERE chart_id = %s", (chart_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Chart with ID {chart_id} not found")
        
        # Update metadata with DW URL
        current_metadata = result['metadata'] or {}
        if isinstance(current_metadata, str):
            try:
                current_metadata = json.loads(current_metadata)
            except json.JSONDecodeError:
                current_metadata = {}
        
        current_metadata['dw_url'] = dw_url
        
        # Update the database
        cursor.execute("""
            UPDATE time_series_metadata 
            SET metadata = %s 
            WHERE chart_id = %s
        """, (json.dumps(current_metadata), chart_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully updated chart metadata for chart_id: {chart_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Updated chart metadata for chart_id: {chart_id}",
            "chart_id": chart_id,
            "dw_url": dw_url
        })
        
    except Exception as e:
        logger.error(f"Error updating chart metadata: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating chart metadata: {str(e)}")


@router.post("/update-anomaly-metadata")
async def update_anomaly_metadata(request: Request):
    """
    Update anomaly metadata with DW URL.
    """
    try:
        data = await request.json()
        anomaly_id = data.get("anomaly_id")
        dw_url = data.get("dw_url")
        
        if not anomaly_id or not dw_url:
            raise HTTPException(status_code=400, detail="anomaly_id and dw_url are required")
        
        logger.info(f"Updating anomaly metadata for anomaly_id: {anomaly_id}, dw_url: {dw_url}")
        
        # Import database utilities
        import psycopg2
        import psycopg2.extras
        import os
        import json
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get current metadata from anomalies table
        cursor.execute("SELECT metadata FROM anomalies WHERE id = %s", (anomaly_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Anomaly with ID {anomaly_id} not found")
        
        # Update metadata with DW URL
        current_metadata = result['metadata'] or {}
        if isinstance(current_metadata, str):
            try:
                current_metadata = json.loads(current_metadata)
            except json.JSONDecodeError:
                current_metadata = {}
        
        current_metadata['dw_url'] = dw_url
        
        # Update the database
        cursor.execute("""
            UPDATE anomalies 
            SET metadata = %s 
            WHERE id = %s
        """, (json.dumps(current_metadata), anomaly_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully updated anomaly metadata for anomaly_id: {anomaly_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Updated anomaly metadata for anomaly_id: {anomaly_id}",
            "anomaly_id": anomaly_id,
            "dw_url": dw_url
        })
        
    except Exception as e:
        logger.error(f"Error updating anomaly metadata: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating anomaly metadata: {str(e)}")


@router.get("/api/chart-metadata/{chart_id}")
async def get_chart_metadata(chart_id: int):
    """
    Get chart metadata including DW URL if available.
    """
    try:
        logger.info(f"Getting chart metadata for chart_id: {chart_id}")
        
        # Import database utilities
        import psycopg2
        import psycopg2.extras
        import os
        import json
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get chart metadata
        cursor.execute("""
            SELECT chart_id, metadata, object_name, field_name, period_type, district
            FROM time_series_metadata 
            WHERE chart_id = %s
        """, (chart_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Chart with ID {chart_id} not found")
        
        # Parse metadata
        metadata = result['metadata'] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        return JSONResponse({
            "status": "success",
            "chart_id": chart_id,
            "metadata": metadata,
            "object_name": result['object_name'],
            "field_name": result['field_name'],
            "period_type": result['period_type'],
            "district": result['district']
        })
        
    except Exception as e:
        logger.error(f"Error getting chart metadata: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting chart metadata: {str(e)}")


@router.get("/api/anomaly-metadata/{anomaly_id}")
async def get_anomaly_metadata(anomaly_id: int):
    """
    Get anomaly metadata including DW URL if available.
    """
    try:
        logger.info(f"Getting anomaly metadata for anomaly_id: {anomaly_id}")
        
        # Import database utilities
        import psycopg2
        import psycopg2.extras
        import os
        import json
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get anomaly metadata
        cursor.execute("""
            SELECT id, metadata, object_name, field_name, period_type, district
            FROM anomalies 
            WHERE id = %s
        """, (anomaly_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Anomaly with ID {anomaly_id} not found")
        
        # Parse metadata
        metadata = result['metadata'] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        return JSONResponse({
            "status": "success",
            "anomaly_id": anomaly_id,
            "metadata": metadata,
            "object_name": result['object_name'],
            "field_name": result['field_name'],
            "period_type": result['period_type'],
            "district": result['district']
        })
        
    except Exception as e:
        logger.error(f"Error getting anomaly metadata: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting anomaly metadata: {str(e)}") 
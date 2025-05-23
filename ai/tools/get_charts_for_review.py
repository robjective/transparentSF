#!/usr/bin/env python3
"""
Tool to get charts from various sources for AI review and newsletter inclusion.
Queries time_series_metadata, anomalies, and maps tables to provide chart options.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

try:
    from .db_utils import execute_with_connection
except ImportError:
    # Try absolute import for testing
    from db_utils import execute_with_connection

logger = logging.getLogger(__name__)

def get_charts_for_review(
    context_variables: Dict[str, Any],
    limit: int = 20,
    days_back: int = 30,
    district_filter: Optional[str] = None,
    chart_types: Optional[List[str]] = None,
    include_time_series: bool = True,
    include_anomalies: bool = True,
    include_maps: bool = True,
    only_active: bool = True,
    metric_filter: Optional[str] = None,
    object_type_filter: Optional[str] = None,
    metric_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get charts from various sources for AI review and potential newsletter inclusion.
    
    Args:
        context_variables: Context variables dictionary
        limit: Maximum number of charts to return per type (default: 20)
        days_back: Number of days back to look for charts (default: 30)
        district_filter: Filter by specific district (e.g., '0' for citywide, '1'-'11' for districts)
        chart_types: List of chart types to include ['time_series', 'anomaly', 'map']
        include_time_series: Whether to include time series charts (default: True)
        include_anomalies: Whether to include anomaly charts (default: True)
        include_maps: Whether to include maps (default: True)
        only_active: Whether to only return active charts (default: True)
        metric_filter: Filter by specific metric/object name (optional)
        object_type_filter: Filter by specific object type (optional)
        metric_id: Filter by specific metric ID/object_id (optional)
        
    Returns:
        Dictionary containing chart information organized by type
    """
    try:
        # Set default chart types if not provided
        if chart_types is None:
            chart_types = []
            if include_time_series:
                chart_types.append('time_series')
            if include_anomalies:
                chart_types.append('anomaly')
            if include_maps:
                chart_types.append('map')
        
        # Calculate date cutoff
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
        
        results = {
            'time_series_charts': [],
            'anomaly_charts': [],
            'map_charts': [],
            'summary': {
                'total_charts': 0,
                'days_back': days_back,
                'district_filter': district_filter,
                'only_active': only_active,
                'metric_filter': metric_filter,
                'object_type_filter': object_type_filter,
                'metric_id': metric_id,
                'query_date': datetime.now().isoformat()
            }
        }
        
        # Query time series charts
        if 'time_series' in chart_types:
            def get_time_series_operation(connection):
                cursor = connection.cursor()
                
                time_series_query = """
                    SELECT 
                        chart_id,
                        chart_title,
                        caption,
                        object_type,
                        object_name,
                        field_name,
                        y_axis_label,
                        period_type,
                        district,
                        created_at,
                        metadata,
                        executed_query_url,
                        is_active
                    FROM time_series_metadata 
                    WHERE created_at >= %s
                """
                
                params = [cutoff_date_str]
                
                if only_active:
                    time_series_query += " AND is_active = true"
                
                if district_filter is not None:
                    time_series_query += " AND district = %s"
                    params.append(district_filter)
                
                if metric_filter is not None:
                    time_series_query += " AND object_name ILIKE %s"
                    params.append(f"%{metric_filter}%")
                
                if object_type_filter is not None:
                    time_series_query += " AND object_type = %s"
                    params.append(object_type_filter)
                
                if metric_id is not None:
                    time_series_query += " AND object_id = %s"
                    params.append(str(metric_id))
                
                time_series_query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(time_series_query, params)
                rows = cursor.fetchall()
                cursor.close()
                return rows
            
            time_series_result = execute_with_connection(get_time_series_operation)
            
            if time_series_result['status'] == 'success':
                for row in time_series_result['result']:
                    chart_info = {
                        'chart_id': row[0],
                        'title': row[1] or f"Time Series Chart {row[0]}",
                        'caption': row[2] or "No caption available",
                        'object_type': row[3],
                        'object_name': row[4],
                        'field_name': row[5],
                        'y_axis_label': row[6],
                        'period_type': row[7],
                        'district': row[8],
                        'created_at': row[9].isoformat() if row[9] else None,
                        'metadata': row[10],
                        'executed_query_url': row[11],
                        'chart_type': 'time_series',
                        'chart_reference': f"[CHART:time_series_id:{row[0]}]"
                    }
                    results['time_series_charts'].append(chart_info)
        
        # Query anomaly charts
        if 'anomaly' in chart_types:
            def get_anomalies_operation(connection):
                cursor = connection.cursor()
                
                anomaly_query = """
                    SELECT 
                        id,
                        caption,
                        group_value,
                        group_field_name,
                        period_type,
                        object_type,
                        object_name,
                        field_name,
                        district,
                        created_at,
                        metadata,
                        executed_query_url,
                        recent_mean,
                        comparison_mean,
                        difference,
                        out_of_bounds
                    FROM anomalies 
                    WHERE created_at >= %s
                """
                
                params = [cutoff_date_str]
                
                if only_active:
                    anomaly_query += " AND is_active = true"
                
                if district_filter is not None:
                    anomaly_query += " AND district = %s"
                    params.append(district_filter)
                
                if metric_filter is not None:
                    anomaly_query += " AND object_name ILIKE %s"
                    params.append(f"%{metric_filter}%")
                
                if object_type_filter is not None:
                    anomaly_query += " AND object_type = %s"
                    params.append(object_type_filter)
                
                if metric_id is not None:
                    anomaly_query += " AND object_id = %s"
                    params.append(str(metric_id))
                
                anomaly_query += " ORDER BY created_at DESC, out_of_bounds DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(anomaly_query, params)
                rows = cursor.fetchall()
                cursor.close()
                return rows
            
            anomaly_result = execute_with_connection(get_anomalies_operation)
            
            if anomaly_result['status'] == 'success':
                for row in anomaly_result['result']:
                    chart_info = {
                        'chart_id': row[0],
                        'title': f"Anomaly Detection: {row[6] or 'Unknown'} - {row[7] or 'Unknown Field'}",
                        'caption': row[1] or "No caption available", 
                        'group_value': row[2],
                        'group_field_name': row[3],
                        'period_type': row[4],
                        'object_type': row[5],
                        'object_name': row[6],
                        'field_name': row[7],
                        'district': row[8],
                        'created_at': row[9].isoformat() if row[9] else None,
                        'metadata': row[10],
                        'executed_query_url': row[11],
                        'recent_mean': row[12],
                        'comparison_mean': row[13],
                        'difference': row[14],
                        'out_of_bounds': row[15],
                        'chart_type': 'anomaly',
                        'chart_reference': f"[CHART:anomaly:{row[0]}]"
                    }
                    results['anomaly_charts'].append(chart_info)
        
        # Query maps (check if maps table exists first)
        if 'map' in chart_types:
            def check_maps_table_operation(connection):
                cursor = connection.cursor()
                check_query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'maps'
                    );
                """
                cursor.execute(check_query)
                exists = cursor.fetchone()[0]
                cursor.close()
                return exists
            
            maps_exist_result = execute_with_connection(check_maps_table_operation)
            
            if maps_exist_result['status'] == 'success' and maps_exist_result['result']:
                def get_maps_operation(connection):
                    cursor = connection.cursor()
                    
                    maps_query = """
                        SELECT 
                            id,
                            title,
                            type,
                            location_data,
                            metadata,
                            created_at,
                            published_url,
                            chart_id,
                            metric_id,
                            group_field,
                            active
                        FROM maps 
                        WHERE created_at >= %s
                    """
                    
                    params = [cutoff_date_str]
                    
                    if only_active:
                        maps_query += " AND active = true"
                    
                    if metric_filter is not None:
                        maps_query += " AND (title ILIKE %s OR metric_id ILIKE %s)"
                        params.append(f"%{metric_filter}%")
                        params.append(f"%{metric_filter}%")
                    
                    if metric_id is not None:
                        maps_query += " AND metric_id = %s"
                        params.append(str(metric_id))
                    
                    maps_query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(maps_query, params)
                    rows = cursor.fetchall()
                    cursor.close()
                    return rows
                
                maps_result = execute_with_connection(get_maps_operation)
                
                if maps_result['status'] == 'success':
                    for row in maps_result['result']:
                        chart_info = {
                            'chart_id': row[0],  # id (uuid)
                            'title': row[1] or f"Map {row[0]}",  # title
                            'caption': "No description available",  # Default since no description column
                            'map_type': row[2],  # type
                            'district': None,  # No district column in maps table
                            'created_at': row[5].isoformat() if row[5] else None,  # created_at
                            'metadata': row[4],  # metadata
                            'published_url': row[6],  # published_url
                            'location_data': row[3],  # location_data
                            'metric_id': row[8],  # metric_id
                            'group_field': row[9],  # group_field
                            'active': row[10],  # active
                            'chart_type': 'map',
                            'chart_reference': f"[CHART:map:{row[0]}]"
                        }
                        results['map_charts'].append(chart_info)
            else:
                logger.info("Maps table does not exist, skipping map charts")
        
        # Calculate summary statistics
        total_charts = (
            len(results['time_series_charts']) + 
            len(results['anomaly_charts']) + 
            len(results['map_charts'])
        )
        results['summary']['total_charts'] = total_charts
        results['summary']['time_series_count'] = len(results['time_series_charts'])
        results['summary']['anomaly_count'] = len(results['anomaly_charts'])
        results['summary']['map_count'] = len(results['map_charts'])
        
        logger.info(f"Retrieved {total_charts} charts for review")
        
        return {
            'success': True,
            'data': results,
            'message': f"Retrieved {total_charts} charts from the last {days_back} days"
        }
        
    except Exception as e:
        error_msg = f"Error retrieving charts for review: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'data': {
                'time_series_charts': [],
                'anomaly_charts': [],
                'map_charts': [],
                'summary': {'total_charts': 0, 'error': error_msg}
            }
        }


def get_recent_charts_summary(
    context_variables: Dict[str, Any],
    days_back: int = 7,
    only_active: bool = True,
    metric_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a quick summary of recently created charts.
    
    Args:
        context_variables: Context variables dictionary
        days_back: Number of days back to look (default: 7)
        only_active: Whether to only count active charts (default: True)
        metric_id: Filter by specific metric ID/object_id (optional)
        
    Returns:
        Dictionary with chart count summary
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
        
        # Count time series charts
        def count_time_series_operation(connection):
            cursor = connection.cursor()
            query = "SELECT COUNT(*) FROM time_series_metadata WHERE created_at >= %s"
            params = [cutoff_date_str]
            
            if only_active:
                query += " AND is_active = true"
            
            if metric_id is not None:
                query += " AND object_id = %s"
                params.append(str(metric_id))
            
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        
        ts_result = execute_with_connection(count_time_series_operation)
        ts_count = ts_result['result'] if ts_result['status'] == 'success' else 0
        
        # Count anomaly charts
        def count_anomalies_operation(connection):
            cursor = connection.cursor()
            query = "SELECT COUNT(*) FROM anomalies WHERE created_at >= %s"
            params = [cutoff_date_str]
            
            if only_active:
                query += " AND is_active = true"
            
            if metric_id is not None:
                query += " AND object_id = %s"
                params.append(str(metric_id))
            
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        
        anom_result = execute_with_connection(count_anomalies_operation)
        anom_count = anom_result['result'] if anom_result['status'] == 'success' else 0
        
        # Count maps (if table exists)
        map_count = 0
        
        def check_maps_operation(connection):
            cursor = connection.cursor()
            check_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'maps'
                );
            """
            cursor.execute(check_query)
            exists = cursor.fetchone()[0]
            cursor.close()
            return exists
        
        maps_exist_result = execute_with_connection(check_maps_operation)
        
        if maps_exist_result['status'] == 'success' and maps_exist_result['result']:
            def count_maps_operation(connection):
                cursor = connection.cursor()
                query = "SELECT COUNT(*) FROM maps WHERE created_at >= %s"
                params = [cutoff_date_str]
                
                if only_active:
                    query += " AND active = true"
                
                if metric_id is not None:
                    query += " AND metric_id = %s"
                    params.append(str(metric_id))
                
                cursor.execute(query, params)
                count = cursor.fetchone()[0]
                cursor.close()
                return count
            
            map_result = execute_with_connection(count_maps_operation)
            map_count = map_result['result'] if map_result['status'] == 'success' else 0
        
        summary = {
            'days_back': days_back,
            'time_series_charts': ts_count,
            'anomaly_charts': anom_count,
            'map_charts': map_count,
            'total_charts': ts_count + anom_count + map_count,
            'query_date': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'data': summary,
            'message': f"Found {summary['total_charts']} charts in the last {days_back} days"
        }
        
    except Exception as e:
        error_msg = f"Error getting charts summary: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'data': {'total_charts': 0, 'error': error_msg}
        } 
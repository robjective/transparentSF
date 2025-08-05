#!/usr/bin/env python3

import psycopg2
import psycopg2.extras
import json
from datetime import datetime, date, timedelta

def test_stale_data_detection():
    # Connect to database
    conn = psycopg2.connect(
        dbname="transparentsf",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    # Test with metric_id = 2
    object_id = "2"
    
    # Get the most_recent_data_date from metrics table
    cursor.execute("""
        SELECT most_recent_data_date 
        FROM metrics 
        WHERE id = %s
        LIMIT 1
    """, [object_id])
    metric_result = cursor.fetchone()
    
    print(f"Metric {object_id} most_recent_data_date: {metric_result['most_recent_data_date'] if metric_result else 'None'}")
    
    # Get charts for this metric
    cursor.execute("""
        SELECT chart_id, object_name, metadata 
        FROM time_series_metadata 
        WHERE object_id = %s
        LIMIT 1
    """, [object_id])
    chart_result = cursor.fetchone()
    
    if chart_result and chart_result['metadata']:
        metadata = chart_result['metadata']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        print(f"Chart {chart_result['chart_id']}: {chart_result['object_name']}")
        
        # Look for the end date in filter_conditions (the condition with operator "<=")
        filter_conditions = metadata.get('filter_conditions', [])
        expected_end_date = None
        
        for condition in filter_conditions:
            if condition.get('operator') == '<=' and condition.get('is_date', False):
                end_date_str = condition.get('value')
                if end_date_str:
                    try:
                        if 'T' in end_date_str:
                            start_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                        else:
                            start_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                        
                        # Calculate the last day of the month from the start date
                        if start_date.day == 1:  # If it's the first day of the month
                            # Get the last day of the month
                            if start_date.month == 12:
                                next_month = start_date.replace(year=start_date.year + 1, month=1, day=1)
                            else:
                                next_month = start_date.replace(month=start_date.month + 1, day=1)
                            expected_end_date = next_month - timedelta(days=1)
                        else:
                            expected_end_date = start_date
                        break
                    except ValueError:
                        try:
                            start_date = datetime.strptime(end_date_str, '%Y-%m').date()
                            # Calculate the last day of the month
                            if start_date.month == 12:
                                next_month = start_date.replace(year=start_date.year + 1, month=1, day=1)
                            else:
                                next_month = start_date.replace(month=start_date.month + 1, day=1)
                            expected_end_date = next_month - timedelta(days=1)
                            break
                        except ValueError:
                            continue
        
        print(f"Expected end date: {expected_end_date}")
        
        # Compare dates
        if expected_end_date and metric_result and metric_result['most_recent_data_date']:
            most_recent = metric_result['most_recent_data_date']
            if isinstance(most_recent, str):
                most_recent = datetime.strptime(most_recent, '%Y-%m-%d').date()
            
            print(f"Most recent data date: {most_recent}")
            print(f"Expected end date: {expected_end_date}")
            print(f"Is stale? {most_recent < expected_end_date}")
            
            if most_recent < expected_end_date:
                days_stale = (expected_end_date - most_recent).days
                print(f"Days stale: {days_stale}")
                stale_data_warning = f"Data is {days_stale} days stale (most recent: {most_recent}, expected: {expected_end_date})"
                print(f"Stale data warning: {stale_data_warning}")
            else:
                print("Data is NOT stale")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    test_stale_data_detection() 
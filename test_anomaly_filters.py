#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from ai.tools.db_utils import get_postgres_connection

def test_anomaly_filter_extraction():
    """Test how anomaly filters are extracted"""
    
    # Simulate the anomaly data extraction
    anomaly_id = 444610
    
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        # Get anomaly details
        cursor.execute("""
            SELECT id, group_value, group_field_name, district, recent_date, period_type
            FROM anomalies 
            WHERE id = %s
        """, (anomaly_id,))
        
        result = cursor.fetchone()
        if result:
            anomaly_id, group_value, group_field_name, district, recent_date, period_type = result
            
            print(f"Raw anomaly data:")
            print(f"  group_value: {group_value}")
            print(f"  group_field_name: {group_field_name}")
            print(f"  district: {district}")
            print(f"  recent_date: {recent_date}")
            print(f"  period_type: {period_type}")
            
            # Simulate the filter extraction logic
            anomaly_filters = {
                "district": district,
                "group_field": group_field_name,
                "field_value": group_value,
                "time_period": recent_date,
                "period_type": period_type
            }
            
            # Remove None/empty values
            anomaly_filters = {k: v for k, v in anomaly_filters.items() if v is not None and v != "" and v != "unknown"}
            
            print(f"\nExtracted anomaly filters: {anomaly_filters}")
            
            # Show what filters would be applied
            if anomaly_filters.get("district") and anomaly_filters["district"] != "0":
                print(f"  Will filter by district: {anomaly_filters['district']}")
            if anomaly_filters.get("group_field") and anomaly_filters.get("field_value"):
                print(f"  Will filter by {anomaly_filters['group_field']} = {anomaly_filters['field_value']}")
            if anomaly_filters.get("time_period"):
                print(f"  Will filter by time period: {anomaly_filters['time_period']}")
                
        else:
            print(f"Anomaly {anomaly_id} not found")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_anomaly_filter_extraction() 
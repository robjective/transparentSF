#!/usr/bin/env python3
"""
Debug script for generate map functionality
Tests the query transformation logic with anomaly_id 429496
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

from urllib.parse import unquote
import re
from datetime import datetime, timedelta
import calendar
import json

def test_query_transformation():
    """Test the query transformation logic"""
    
    # Get the anomaly data
    print("=== Testing Query Transformation Logic ===")
    
    try:
        from tools.db_utils import get_postgres_connection
        from tools.store_anomalies import get_anomaly_details
        
        # Get anomaly details
        anomaly_result = get_anomaly_details(429496)
        if anomaly_result["status"] != "success":
            print(f"Error getting anomaly: {anomaly_result}")
            return
        
        anomaly_data = anomaly_result["anomaly"]
        metadata = anomaly_data.get("metadata", {})
        query_url = metadata.get("executed_query_url")
        
        print(f"Anomaly ID: 429496")
        print(f"Metric Name: {metadata.get('object_name', 'Unknown')}")
        print(f"Date Field: {metadata.get('date_field', 'Unknown')}")
        print(f"Object ID: {metadata.get('object_id', 'Unknown')}")
        print()
        
        if not query_url:
            print("No query URL found in anomaly metadata")
            return
        
        print("=== Original Query URL ===")
        print(query_url)
        print()
        
        # Decode the URL
        decoded_url = unquote(query_url)
        print("=== Decoded URL ===")
        print(decoded_url)
        print()
        
        # Extract the query part
        if "?" in decoded_url:
            query_part = decoded_url.split("?")[1]
            if query_part.startswith("$query="):
                original_query = query_part[7:]
            else:
                original_query = query_part
        else:
            original_query = ""
        
        # Decode the query
        original_query = unquote(original_query)
        print("=== Original Query ===")
        print(original_query)
        print()
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', original_query, re.IGNORECASE | re.DOTALL)
        original_where_conditions = ""
        if where_match:
            original_where_conditions = where_match.group(1).strip()
            print("=== Extracted WHERE Conditions ===")
            print(original_where_conditions)
            print()
        else:
            print("=== No WHERE clause found ===")
            print()
        
        # Extract the actual date field from the query URL (not the aggregated field)
        # Look for date_trunc patterns to find the actual date field
        date_field_match = re.search(r'date_trunc[^)]*\(\s*[\'"]?(\w+)[\'"]?\s*\)', original_query, re.IGNORECASE)
        if date_field_match:
            # Extract the actual date field from date_trunc
            date_field = date_field_match.group(1)
            print(f"Extracted date field from date_trunc: {date_field}")
        else:
            # Try to find common date field names in the query
            common_date_fields = ['requested_datetime', 'report_datetime', 'incident_datetime', 'date', 'datetime', 'created_date', 'incident_date', 'report_date']
            date_field = None
            for field in common_date_fields:
                if field in original_query.lower():
                    date_field = field
                    break
            
            if not date_field:
                print("Could not determine date field from query")
                return
        
        print(f"=== Date Field: {date_field} ===")
        
        # Test our date filter generation
        period_type = "month"
        now = datetime.now()
        
        if period_type == "month":
            last_month_start = (now - timedelta(days=30)).replace(day=1)
            last_month_end = (last_month_start.replace(day=calendar.monthrange(last_month_start.year, last_month_start.month)[1]))
            current_month_start = now.replace(day=1)
            current_month_end = (current_month_start.replace(day=calendar.monthrange(current_month_start.year, current_month_start.month)[1]))
            
            date_filter = f"({date_field} >= '{last_month_start.strftime('%Y-%m-%d')}' AND {date_field} <= '{last_month_end.strftime('%Y-%m-%d')}') OR ({date_field} >= '{current_month_start.strftime('%Y-%m-%d')}' AND {date_field} <= '{current_month_end.strftime('%Y-%m-%d')}')"
        
        print("=== Generated Date Filter ===")
        print(date_filter)
        print()
        
        # Test WHERE condition cleaning
        if original_where_conditions:
            print("=== Cleaning WHERE Conditions ===")
            
            # Remove date conditions that use the same date field
            cleaned_where = re.sub(rf'{date_field}\s*[><=]+\s*[\'"][^\'"]*[\'"]', '', original_where_conditions)
            print(f"After removing {date_field} conditions:")
            print(cleaned_where)
            print()
            
            cleaned_where = re.sub(rf'date_trunc[^)]*{date_field}[^)]*\)\s*[><=]+\s*[\'"][^\'"]*[\'"]', '', cleaned_where)
            print(f"After removing date_trunc conditions:")
            print(cleaned_where)
            print()
            
            # Clean up any double ANDs or ORs
            cleaned_where = re.sub(r'\s+AND\s+AND\s+', ' AND ', cleaned_where)
            cleaned_where = re.sub(r'\s+OR\s+OR\s+', ' OR ', cleaned_where)
            cleaned_where = cleaned_where.strip()
            
            print(f"After cleaning double operators:")
            print(cleaned_where)
            print()
            
            # Create final query
            if cleaned_where:
                detailed_query = f"SELECT * WHERE {date_filter} AND ({cleaned_where})"
            else:
                detailed_query = f"SELECT * WHERE {date_filter}"
        else:
            detailed_query = f"SELECT * WHERE {date_filter}"
        
        print("=== Final Generated Query ===")
        print(detailed_query)
        print()
        
        # Test the API endpoint
        print("=== Testing API Endpoint ===")
        import requests
        
        url = "http://localhost:8000/anomaly-analyzer/api/generate-map"
        payload = {
            "metric_id": metadata.get('object_id', '1'),
            "district": "0",
            "period_type": "month",
            "anomaly_id": "429496"
        }
        
        print(f"Request payload: {json.dumps(payload, indent=2)}")
        print()
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        except Exception as e:
            print(f"Error calling API: {e}")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query_transformation() 
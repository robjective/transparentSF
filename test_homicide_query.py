#!/usr/bin/env python3
"""
Test script to debug the homicide query WHERE clause issue.
"""

import re

def test_homicide_query_extraction():
    """Test the WHERE clause extraction logic with the actual homicide query."""
    
    # This is the actual homicide metric query from the database
    homicide_query = "SELECT date_trunc_ymd(Report_Datetime) as date, COUNT(*) as value WHERE Report_Datetime >= last_year_start AND Report_Datetime <= current_date AND Incident_Category = 'Homicide' GROUP BY date ORDER BY date"
    
    print(f"Original homicide query: {homicide_query}")
    
    # Extract WHERE clause using the same logic as the fix
    original_where_clause = ""
    if "WHERE" in homicide_query.upper():
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', homicide_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            original_where_clause = where_match.group(1).strip()
            print(f"Extracted WHERE clause: {original_where_clause}")
        else:
            print("No WHERE clause found")
    else:
        print("No WHERE clause in query")
    
    # Test the new query building logic
    new_query = "SELECT *"
    if original_where_clause:
        new_query += f" WHERE {original_where_clause}"
        print(f"New detail query: {new_query}")
    else:
        print(f"New detail query: {new_query}")
    
    # Test if the homicide filter is preserved
    if "Incident_Category = 'Homicide'" in original_where_clause:
        print("✓ Homicide filter is preserved in the WHERE clause")
    else:
        print("✗ Homicide filter is NOT preserved in the WHERE clause")
    
    # Test if date filters are preserved
    if "Report_Datetime >= last_year_start" in original_where_clause and "Report_Datetime <= current_date" in original_where_clause:
        print("✓ Date filters are preserved in the WHERE clause")
    else:
        print("✗ Date filters are NOT preserved in the WHERE clause")

def test_anomaly_filter_application():
    """Test how anomaly filters would be applied to the homicide query."""
    
    original_where = "Report_Datetime >= last_year_start AND Report_Datetime <= current_date AND Incident_Category = 'Homicide'"
    
    # Simulate the anomaly filter logic
    anomaly_filters = {
        "district": "0",  # Citywide
        "group_field": "incident_category",
        "field_value": "Homicide"
    }
    
    print(f"\nTesting anomaly filter application with WHERE clause: {original_where}")
    print(f"Anomaly filters: {anomaly_filters}")
    
    # Test district filter (should be skipped for district 0)
    if anomaly_filters.get("district") and anomaly_filters["district"] != "0":
        district_filter = f"supervisor_district = '{anomaly_filters['district']}'"
        if district_filter not in original_where:
            print(f"Would add district filter: {district_filter}")
        else:
            print(f"District filter already present: {district_filter}")
    else:
        print("Skipping district filter for district 0 (citywide)")
    
    # Test group field filter
    if anomaly_filters.get("group_field") and anomaly_filters.get("field_value"):
        group_field = anomaly_filters["group_field"]
        field_value = anomaly_filters["field_value"]
        field_filter = f"{group_field} = '{field_value}'"
        
        if field_filter not in original_where:
            print(f"Would add group field filter: {field_filter}")
        else:
            print(f"Group field filter already present: {field_filter}")
    
    # Test if the homicide filter is already in the original WHERE clause
    homicide_filter = "Incident_Category = 'Homicide'"
    if homicide_filter in original_where:
        print(f"✓ Homicide filter already present: {homicide_filter}")
    else:
        print(f"✗ Homicide filter NOT present: {homicide_filter}")

if __name__ == "__main__":
    print("Testing homicide query WHERE clause extraction...")
    test_homicide_query_extraction()
    
    print("\nTesting anomaly filter application...")
    test_anomaly_filter_application()
    
    print("\nTest completed!") 
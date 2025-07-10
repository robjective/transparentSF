#!/usr/bin/env python3
"""
Script to debug the citywide aggregation issue.
"""

import os
import sys
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

# Add the ai directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

# Load environment variables
load_dotenv()

def test_citywide_vs_district_aggregation():
    """Test how citywide vs district data is being aggregated."""
    
    # Import the anomaly detection function
    from tools.anomaly_detection import anomaly_detection, group_data_by_field_and_date, filter_data_by_date_and_conditions
    from tools.data_fetcher import set_dataset
    
    print("🔍 Testing Citywide vs District Aggregation")
    print("=" * 60)
    
    # Test with 311 cases data
    endpoint = "vw6y-z8j6"  # 311 Cases dataset
    query = """
    SELECT 
        service_request_id,
        service_name,
        service_subtype,
        supervisor_district,
        date_trunc_ym(requested_datetime) as month,
        COUNT(*) as case_count
    WHERE requested_datetime >= '2024-01-01'
    GROUP BY service_request_id, service_name, service_subtype, supervisor_district, month
    ORDER BY month DESC
    LIMIT 1000
    """
    
    print(f"📊 Testing with 311 Cases data")
    print(f"Endpoint: {endpoint}")
    print(f"Query: {query}")
    
    # Create context variables
    context_variables = {}
    
    # Get the data
    result = set_dataset(context_variables, endpoint=endpoint, query=query)
    
    if 'error' in result:
        print(f"❌ Error fetching data: {result['error']}")
        return
    
    dataset = context_variables.get('dataset')
    if dataset is None or dataset.empty:
        print("❌ No data returned")
        return
    
    print(f"✅ Data fetched successfully")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
    print(f"Sample data:")
    print(dataset.head())
    
    # Check supervisor_district values
    if 'supervisor_district' in dataset.columns:
        districts = dataset['supervisor_district'].dropna().unique()
        print(f"\n🏛️ Districts in data: {sorted(districts)}")
        
        # Check counts by district
        district_counts = dataset['supervisor_district'].value_counts()
        print(f"\n📈 Counts by district:")
        for district, count in district_counts.items():
            print(f"  District {district}: {count} records")
    
    # Test 1: Citywide aggregation (no district filter)
    print(f"\n🧪 Test 1: Citywide aggregation (no district filter)")
    print("-" * 50)
    
    # Convert to records for anomaly detection
    data_records = dataset.to_dict('records')
    
    # Test grouping by service_subtype for citywide
    grouped_citywide = group_data_by_field_and_date(
        data_records,
        group_field='service_subtype',
        numeric_field='case_count',
        date_field='month',
        period_type='month',
        agg_function='sum'
    )
    
    print(f"Citywide groups: {len(grouped_citywide)}")
    for group, data in list(grouped_citywide.items())[:3]:
        print(f"  {group}: {data}")
    
    # Test 2: District-specific aggregation
    print(f"\n🧪 Test 2: District-specific aggregation")
    print("-" * 50)
    
    # Filter for a specific district
    district_1_data = [record for record in data_records if record.get('supervisor_district') == 1]
    print(f"District 1 records: {len(district_1_data)}")
    
    if district_1_data:
        grouped_district_1 = group_data_by_field_and_date(
            district_1_data,
            group_field='service_subtype',
            numeric_field='case_count',
            date_field='month',
            period_type='month',
            agg_function='sum'
        )
        
        print(f"District 1 groups: {len(grouped_district_1)}")
        for group, data in list(grouped_district_1.items())[:3]:
            print(f"  {group}: {data}")
    
    # Test 3: Compare totals
    print(f"\n🧪 Test 3: Compare citywide vs district totals")
    print("-" * 50)
    
    # Calculate citywide totals
    citywide_totals = {}
    for group, data in grouped_citywide.items():
        citywide_totals[group] = sum(data.values())
    
    # Calculate district totals
    district_totals = {}
    for district in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        district_data = [record for record in data_records if record.get('supervisor_district') == district]
        if district_data:
            grouped_district = group_data_by_field_and_date(
                district_data,
                group_field='service_subtype',
                numeric_field='case_count',
                date_field='month',
                period_type='month',
                agg_function='sum'
            )
            
            for group, data in grouped_district.items():
                if group not in district_totals:
                    district_totals[group] = 0
                district_totals[group] += sum(data.values())
    
    # Compare
    print(f"Citywide totals (top 5):")
    for group, total in sorted(citywide_totals.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {group}: {total}")
    
    print(f"\nSum of district totals (top 5):")
    for group, total in sorted(district_totals.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {group}: {total}")
    
    # Check for discrepancies
    print(f"\n🔍 Checking for discrepancies:")
    for group in set(citywide_totals.keys()) | set(district_totals.keys()):
        citywide = citywide_totals.get(group, 0)
        district_sum = district_totals.get(group, 0)
        if abs(citywide - district_sum) > 1:  # Allow for small rounding differences
            print(f"  ❌ {group}: Citywide={citywide}, District Sum={district_sum}, Diff={citywide - district_sum}")
        else:
            print(f"  ✅ {group}: Citywide={citywide}, District Sum={district_sum}")

if __name__ == "__main__":
    test_citywide_vs_district_aggregation() 
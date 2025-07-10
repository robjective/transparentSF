#!/usr/bin/env python3
"""
Script to demonstrate the citywide vs district aggregation issue.
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

def test_citywide_vs_district_data():
    """Test how citywide vs district data is being processed."""
    
    # Import the anomaly detection function
    from tools.anomaly_detection import anomaly_detection, group_data_by_field_and_date, filter_data_by_date_and_conditions
    from tools.data_fetcher import set_dataset
    
    print("🔍 Testing Citywide vs District Data Processing")
    print("=" * 60)
    
    # Test with 311 calls data
    endpoint = "https://data.sfgov.org/resource/vw6y-z8j6.json"
    query = """
    SELECT 
        date_trunc_ym(requested_datetime) as month_period,
        COUNT(*) as value,
        CASE 
            WHEN requested_datetime >= '2024-01-01' AND requested_datetime <= '2024-12-31' THEN 'recent'
            ELSE 'comparison'
        END as period_type,
        CASE WHEN supervisor_district IS NOT NULL THEN supervisor_district ELSE NULL END as supervisor_district
    WHERE 
        (requested_datetime >= '2022-01-01' AND requested_datetime <= '2023-12-31')
        OR 
        (requested_datetime >= '2024-01-01' AND requested_datetime <= '2024-12-31')
    GROUP BY month_period, period_type, supervisor_district
    ORDER BY month_period
    """
    
    print(f"📊 Testing with 311 calls data")
    print(f"Query: {query}")
    
    # Set the dataset
    context_variables = {}
    result = set_dataset(context_variables=context_variables, endpoint=endpoint, query=query)
    
    if 'error' in result:
        print(f"❌ Error setting dataset: {result['error']}")
        return
    
    dataset = context_variables['dataset']
    print(f"✅ Dataset loaded with {len(dataset)} rows")
    print(f"📋 Columns: {dataset.columns.tolist()}")
    
    # Show sample data
    print(f"\n📊 Sample data (first 10 rows):")
    print(dataset.head(10).to_string())
    
    # Check district distribution
    print(f"\n🏛️ District distribution:")
    district_counts = dataset['supervisor_district'].value_counts().sort_index()
    print(district_counts)
    
    # Check recent vs comparison period distribution
    print(f"\n📅 Period distribution:")
    period_counts = dataset['period_type'].value_counts()
    print(period_counts)
    
    # Test citywide aggregation (no district filter)
    print(f"\n🌆 Citywide aggregation (no district filter):")
    citywide_data = dataset.copy()
    citywide_recent = citywide_data[citywide_data['period_type'] == 'recent']
    citywide_comparison = citywide_data[citywide_data['period_type'] == 'comparison']
    
    print(f"Recent period total: {citywide_recent['value'].sum()}")
    print(f"Comparison period total: {citywide_comparison['value'].sum()}")
    
    # Test district 1 aggregation (with district filter)
    print(f"\n🏛️ District 1 aggregation (with district filter):")
    district1_data = dataset[dataset['supervisor_district'] == 1].copy()
    district1_recent = district1_data[district1_data['period_type'] == 'recent']
    district1_comparison = district1_data[district1_data['period_type'] == 'comparison']
    
    print(f"District 1 recent period total: {district1_recent['value'].sum()}")
    print(f"District 1 comparison period total: {district1_comparison['value'].sum()}")
    
    # Test district 2 aggregation (with district filter)
    print(f"\n🏛️ District 2 aggregation (with district filter):")
    district2_data = dataset[dataset['supervisor_district'] == 2].copy()
    district2_recent = district2_data[district2_data['period_type'] == 'recent']
    district2_comparison = district2_data[district2_data['period_type'] == 'comparison']
    
    print(f"District 2 recent period total: {district2_recent['value'].sum()}")
    print(f"District 2 comparison period total: {district2_comparison['value'].sum()}")
    
    # Verify that citywide = sum of all districts
    print(f"\n🔍 Verification:")
    all_districts_recent = 0
    all_districts_comparison = 0
    
    for district in sorted(dataset['supervisor_district'].dropna().unique()):
        district_data = dataset[dataset['supervisor_district'] == district]
        district_recent = district_data[district_data['period_type'] == 'recent']['value'].sum()
        district_comparison = district_data[district_data['period_type'] == 'comparison']['value'].sum()
        all_districts_recent += district_recent
        all_districts_comparison += district_comparison
        print(f"District {district}: Recent={district_recent}, Comparison={district_comparison}")
    
    print(f"\nSum of all districts recent: {all_districts_recent}")
    print(f"Sum of all districts comparison: {all_districts_comparison}")
    print(f"Citywide recent: {citywide_recent['value'].sum()}")
    print(f"Citywide comparison: {citywide_comparison['value'].sum()}")
    
    if all_districts_recent == citywide_recent['value'].sum():
        print("✅ Citywide aggregation is correct!")
    else:
        print("❌ Citywide aggregation is incorrect!")
        print(f"Difference: {citywide_recent['value'].sum() - all_districts_recent}")

if __name__ == "__main__":
    test_citywide_vs_district_data() 
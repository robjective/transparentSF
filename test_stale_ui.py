#!/usr/bin/env python3
"""
Test script to verify stale data detection UI elements.
"""

import asyncio
import json
from anomalyAnalyzer import get_top_metric_changes

async def test_stale_data_ui():
    """Test stale data detection UI elements."""
    
    print("Testing stale data detection UI...")
    
    # Test with metric ID 10 (known to have stale data)
    result = await get_top_metric_changes(period_type='month', limit=5, object_id='10')
    data = json.loads(result.body.decode())
    
    print(f"API Response Status: {data['status']}")
    print(f"Total Results: {data['count']}")
    print(f"Stale Data Warnings Count: {data.get('stale_data_warnings', 0)}")
    
    # Check if stale data warning is in individual results
    all_items = data.get('positive_changes', []) + data.get('negative_changes', [])
    stale_items = [item for item in all_items if item.get('stale_data_warning')]
    
    print(f"Items with stale data warnings: {len(stale_items)}")
    
    for item in stale_items:
        print(f"\nMetric: {item['object_name']}")
        print(f"Stale Data Warning: {item['stale_data_warning']}")
        print(f"Recent Value: {item['recent_value']}")
        print(f"Previous Value: {item['previous_value']}")
        print(f"Delta: {item['delta']}")
    
    # Test with multiple metrics to see if stale data warnings are detected
    print("\n" + "="*50)
    print("Testing with multiple metrics...")
    
    result_all = await get_top_metric_changes(period_type='month', limit=10)
    data_all = json.loads(result_all.body.decode())
    
    print(f"Total Results: {data_all['count']}")
    print(f"Stale Data Warnings Count: {data_all.get('stale_data_warnings', 0)}")
    
    all_items_all = data_all.get('positive_changes', []) + data_all.get('negative_changes', [])
    stale_items_all = [item for item in all_items_all if item.get('stale_data_warning')]
    
    print(f"Items with stale data warnings: {len(stale_items_all)}")
    
    if stale_items_all:
        print("\nMetrics with stale data:")
        for item in stale_items_all:
            print(f"- {item['object_name']}: {item['stale_data_warning']}")
    else:
        print("\nNo stale data detected in the sample.")

if __name__ == "__main__":
    asyncio.run(test_stale_data_ui()) 
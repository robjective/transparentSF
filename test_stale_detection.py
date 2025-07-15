#!/usr/bin/env python3
"""
Test script to verify stale data detection works correctly.
"""

import asyncio
import json
from anomalyAnalyzer import get_top_metric_changes

async def test_stale_detection():
    """Test stale data detection for multiple metrics."""
    
    print("Testing stale data detection...")
    
    # Test with multiple metrics
    result = await get_top_metric_changes(period_type='month', limit=10)
    data = json.loads(result.body.decode())
    
    print(f"Total results: {data['count']}")
    print(f"Stale data warnings: {data.get('stale_data_warnings', 0)}")
    
    # Find items with stale data warnings
    all_items = data.get('positive_changes', []) + data.get('negative_changes', [])
    stale_items = [item for item in all_items if item.get('stale_data_warning')]
    
    print(f"Items with stale data warnings: {len(stale_items)}")
    
    for item in stale_items:
        print(f"- {item['object_name']}: {item['stale_data_warning']}")
    
    # Test with specific metric ID 10
    print("\nTesting specific metric ID 10:")
    result_10 = await get_top_metric_changes(period_type='month', limit=5, object_id='10')
    data_10 = json.loads(result_10.body.decode())
    
    print(f"Metric 10 results: {data_10['count']}")
    print(f"Metric 10 stale warnings: {data_10.get('stale_data_warnings', 0)}")
    
    if data_10.get('positive_changes'):
        item = data_10['positive_changes'][0]
        if item.get('stale_data_warning'):
            print(f"Metric 10 stale data: {item['stale_data_warning']}")
        else:
            print("Metric 10: No stale data warning found")

if __name__ == "__main__":
    asyncio.run(test_stale_detection()) 
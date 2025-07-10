#!/usr/bin/env python3
"""
Test script to verify group filter functionality.
"""

import requests
import json

def test_group_filter():
    """Test the group filter functionality."""
    
    # Test URL - replace with your actual server URL
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "name": "No groups parameter",
            "params": {"metric_id": "test_metric", "district": "0", "period_type": "month"},
            "expected": "Should return all groups"
        },
        {
            "name": "Empty groups parameter",
            "params": {"metric_id": "test_metric", "district": "0", "period_type": "month", "groups": ""},
            "expected": "Should return no data"
        },
        {
            "name": "Specific groups",
            "params": {"metric_id": "test_metric", "district": "0", "period_type": "month", "groups": "group1,group2"},
            "expected": "Should return only specified groups"
        }
    ]
    
    print("🧪 Testing Group Filter Functionality")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        print(f"Expected: {test_case['expected']}")
        
        try:
            # Make the API request
            response = requests.get(f"{base_url}/api/chart-by-metric", params=test_case['params'])
            
            if response.status_code == 200:
                data = response.json()
                data_count = len(data.get('data', []))
                print(f"✅ Status: {response.status_code}")
                print(f"📊 Data points returned: {data_count}")
                
                if data_count == 0:
                    print("📝 Result: No data returned (as expected for empty groups)")
                else:
                    print("📝 Result: Data returned")
                    
                    # Show group values if present
                    groups = set()
                    for item in data.get('data', []):
                        if 'group_value' in item:
                            groups.add(item['group_value'])
                    
                    if groups:
                        print(f"🏷️ Groups found: {sorted(groups)}")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")

if __name__ == "__main__":
    test_group_filter() 
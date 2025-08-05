#!/usr/bin/env python3
"""
Test script for the generate map functionality.
This script tests the new API endpoint for generating maps from anomaly data.
"""

import requests
import json
import sys
import os

# Add the ai directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_generate_map_api():
    """Test the generate map API endpoint."""
    
    # Test data - you'll need to replace these with actual metric IDs from your database
    test_cases = [
        {
            "name": "Test with valid metric ID",
            "data": {
                "metric_id": "1",  # Replace with actual metric ID
                "district": "0",
                "period_type": "month"
            }
        },
        {
            "name": "Test with district filter",
            "data": {
                "metric_id": "1",  # Replace with actual metric ID
                "district": "1",
                "period_type": "month"
            }
        },
        {
            "name": "Test with weekly period",
            "data": {
                "metric_id": "1",  # Replace with actual metric ID
                "district": "0",
                "period_type": "week"
            }
        }
    ]
    
    base_url = "http://localhost:8000"  # Adjust if your server runs on a different port
    
    print("Testing Generate Map API Endpoint")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            # Make the API request
            response = requests.post(
                f"{base_url}/anomaly-analyzer/api/generate-map",
                headers={"Content-Type": "application/json"},
                json=test_case["data"],
                timeout=30  # 30 second timeout
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result.get('message', 'No message')}")
                print(f"Map ID: {result.get('map_id', 'N/A')}")
                print(f"Data Points: {result.get('data_points', 'N/A')}")
                print(f"Map Type: {result.get('map_type', 'N/A')}")
                if result.get('publish_url'):
                    print(f"Publish URL: {result['publish_url']}")
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server. Make sure the server is running.")
        except requests.exceptions.Timeout:
            print("Error: Request timed out.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test completed.")

def test_metric_retrieval():
    """Test metric retrieval functionality."""
    
    print("\nTesting Metric Retrieval")
    print("=" * 30)
    
    try:
        from tools.metrics_manager import get_metric_by_id
        
        # Test with a sample metric ID
        metric_id = 1  # Replace with actual metric ID
        result = get_metric_by_id(metric_id)
        
        if result["status"] == "success":
            metric = result["metric"]
            print(f"Successfully retrieved metric: {metric['metric_name']}")
            print(f"Query URL: {metric.get('query_url', 'N/A')}")
            print(f"Category: {metric.get('category', 'N/A')}")
        else:
            print(f"Failed to retrieve metric: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error testing metric retrieval: {str(e)}")

if __name__ == "__main__":
    print("Generate Map API Test Suite")
    print("=" * 40)
    
    # Test metric retrieval first
    test_metric_retrieval()
    
    # Test the generate map API
    test_generate_map_api()
    
    print("\nTo run this test:")
    print("1. Make sure your FastAPI server is running")
    print("2. Update the metric IDs in the test cases with actual IDs from your database")
    print("3. Run: python test_generate_map.py") 
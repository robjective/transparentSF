#!/usr/bin/env python3

import sys
import os
sys.path.append('ai/tools')
from generate_map import generate_map
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_symbol_map():
    """Test symbol map creation using clone approach"""
    
    # Test data for a symbol map
    test_symbol_data = [
        {'district': '1', 'value': 450},
        {'district': '2', 'value': 380},
        {'district': '3', 'value': 620},
        {'district': '4', 'value': 290},
        {'district': '5', 'value': 510}
    ]

    test_metadata = {
        'description': 'Test symbol map using clone approach',
        'color': '#0066cc',
        'min_size': 5,
        'max_size': 30
    }

    print('Testing symbol map with clone approach...')
    result = generate_map(
        {},
        map_title='Test Symbol Map - Clone Approach',
        map_type='symbol',
        location_data=test_symbol_data,
        map_metadata=test_metadata
    )

    print(f'Result: {result}')
    if result and 'map_id' in result:
        print('✅ Symbol map created successfully!')
        print(f'Map ID: {result["map_id"]}')
        print(f'Edit URL: {result["edit_url"]}')
        print(f'Public URL: {result["publish_url"]}')
        return True
    else:
        print('❌ Failed to create symbol map')
        return False

if __name__ == "__main__":
    success = test_symbol_map()
    sys.exit(0 if success else 1) 
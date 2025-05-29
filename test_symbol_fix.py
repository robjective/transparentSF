from ai.tools.generate_map import generate_map

print("=== Testing NEW Symbol Map Approach (Choropleth → Symbol Conversion) ===")

# Test symbol map with choropleth base conversion
housing_data = [
    {"district": "1", "value": 450},
    {"district": "2", "value": 380},
    {"district": "3", "value": 620},
    {"district": "4", "value": 290},
    {"district": "5", "value": 510}
]

housing_metadata = {
    "description": "Test housing units by district using choropleth base",
    "color": "#0066cc"
}

result = generate_map(
    {},
    map_title='Symbol Map - Choropleth Base Conversion',
    map_type='symbol',
    location_data=housing_data,
    map_metadata=housing_metadata
)

print('Symbol map conversion result:', result)
if result and result['map_id']:
    print('✅ Symbol map created successfully!')
    print(f'Map ID: {result["map_id"]}')
    print(f'Edit URL: {result["edit_url"]}')
    print(f'Public URL: {result["publish_url"]}')
    print()
    print('🔧 This should now show:')
    print('1. Proper SF district boundaries (not white)')
    print('2. Symbol circles sized by value')
    print('3. No 500 error when editing')
    print('4. Working legend showing value ranges')
else:
    print('❌ Failed to create symbol map') 
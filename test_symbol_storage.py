from ai.tools.generate_map import generate_map

# Test symbol map with database storage
housing_data = [
    {'district': '1', 'value': 450},
    {'district': '2', 'value': 380},
    {'district': '3', 'value': 620}
]

housing_metadata = {
    'description': 'Test housing units by district',
    'color': '#0066cc'
}

result = generate_map(
    {},
    map_title='Test Symbol Map - Database Storage',
    map_type='symbol',
    location_data=housing_data,
    map_metadata=housing_metadata
)

print('Symbol map with database storage result:', result)
if result and result['map_id']:
    print('✅ Symbol map successfully stored in database with ID:', result['map_id'])
    print('Edit URL:', result['edit_url'])
    print('Public URL:', result['publish_url'])
else:
    print('❌ Failed to store symbol map in database') 
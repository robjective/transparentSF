import json
from ai.tools.generate_map import generate_map

# Test data for a symbol map with SF landmarks
test_symbol_data = [
    {
        "title": "City Hall",
        "lat": 37.7749,
        "lon": -122.4194,
        "tooltip": "San Francisco City Hall - Civic Center",
        "icon": "building",
        "scale": 1.5,
        "color": "#0066cc"
    },
    {
        "title": "Golden Gate Bridge",
        "lat": 37.8199,
        "lon": -122.4783,
        "tooltip": "Iconic suspension bridge connecting SF to Marin",
        "icon": "bridge",
        "scale": 1.2,
        "color": "#cc0000"
    },
    {
        "title": "Ferry Building",
        "lat": 37.7956,
        "lon": -122.3937,
        "tooltip": "Historic ferry terminal and marketplace",
        "icon": "building",
        "scale": 1.3,
        "color": "#006600"
    },
    {
        "title": "Dolores Park",
        "lat": 37.7597,
        "lon": -122.4265,
        "tooltip": "Popular park in the Mission District",
        "icon": "park",
        "scale": 1.4,
        "color": "#009900"
    },
    {
        "title": "Sutro Tower",
        "lat": 37.7552,
        "lon": -122.4527,
        "tooltip": "TV and radio transmission tower",
        "icon": "tower",
        "scale": 1.2,
        "color": "#666666"
    }
]

# Generate the map
result = generate_map(
    {},
    map_title="San Francisco Landmarks",
    map_type="point",
    location_data=test_symbol_data,
    map_metadata={
        "description": "Map showing notable landmarks in San Francisco",
        "center": [-122.44, 37.77],  # SF center coordinates
        "zoom": 12
    }
)

# Print results
if result and "map_id" in result:
    print("✅ Symbol map created successfully!")
    print(f"Map ID: {result['map_id']}")
    print(f"Edit URL: {result['edit_url']}")
    print(f"Public URL: {result['publish_url']}")
else:
    print("❌ Failed to create symbol map") 
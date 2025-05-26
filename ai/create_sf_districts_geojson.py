#!/usr/bin/env python3
"""
Create a GeoJSON file of San Francisco supervisorial districts
from the SF Open Data API district officials dataset.
"""

import requests
import json
from pathlib import Path

def fetch_sf_districts():
    """Fetch all SF supervisorial districts with their polygon geometry."""
    
    print("Fetching SF supervisorial district data...")
    
    # SF Open Data API endpoint for district officials
    url = "https://data.sfgov.org/resource/f2zs-jevy.json"
    
    # Query to get all districts with their polygons
    params = {
        "$query": "select sup_dist, sup_dist_name, sup_name, polygon order by sup_dist"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        districts_data = response.json()
        print(f"✅ Fetched {len(districts_data)} districts")
        
        return districts_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching district data: {e}")
        return None

def create_geojson(districts_data):
    """Convert the district data to GeoJSON format."""
    
    print("Converting to GeoJSON format...")
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for district in districts_data:
        # Extract polygon geometry
        polygon_geom = district.get("polygon")
        if not polygon_geom:
            print(f"⚠️ No polygon data for district {district.get('sup_dist', 'unknown')}")
            continue
            
        # Create feature
        feature = {
            "type": "Feature",
            "properties": {
                "district": district.get("sup_dist", ""),
                "district_name": district.get("sup_dist_name", ""),
                "supervisor": district.get("sup_name", ""),
                "district_id": district.get("sup_dist", "")  # For compatibility
            },
            "geometry": polygon_geom
        }
        
        geojson["features"].append(feature)
        print(f"✅ Added District {district.get('sup_dist')}: {district.get('sup_name')}")
    
    print(f"✅ Created GeoJSON with {len(geojson['features'])} district features")
    return geojson

def save_geojson(geojson_data, filename="sf_supervisorial_districts.geojson"):
    """Save the GeoJSON data to a file."""
    
    output_path = Path(filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ GeoJSON saved to: {output_path.absolute()}")
        print(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"❌ Error saving GeoJSON: {e}")
        return None

def validate_geojson(geojson_data):
    """Basic validation of the GeoJSON structure."""
    
    print("Validating GeoJSON structure...")
    
    if geojson_data.get("type") != "FeatureCollection":
        print("❌ Invalid GeoJSON: not a FeatureCollection")
        return False
        
    features = geojson_data.get("features", [])
    if not features:
        print("❌ Invalid GeoJSON: no features found")
        return False
        
    # Check each feature
    for i, feature in enumerate(features):
        if feature.get("type") != "Feature":
            print(f"❌ Invalid feature {i}: not a Feature")
            return False
            
        if not feature.get("geometry"):
            print(f"❌ Invalid feature {i}: no geometry")
            return False
            
        if not feature.get("properties"):
            print(f"❌ Invalid feature {i}: no properties")
            return False
            
        # Check if district property exists
        district = feature["properties"].get("district")
        if not district:
            print(f"❌ Invalid feature {i}: no district property")
            return False
    
    print(f"✅ GeoJSON validation passed: {len(features)} valid features")
    return True

def create_district_summary(geojson_data):
    """Create a summary of the districts for reference."""
    
    print("\n" + "="*50)
    print("SF SUPERVISORIAL DISTRICTS SUMMARY")
    print("="*50)
    
    features = geojson_data.get("features", [])
    
    for feature in sorted(features, key=lambda x: int(x["properties"]["district"])):
        props = feature["properties"]
        district_num = props.get("district", "?")
        district_name = props.get("district_name", "Unknown")
        supervisor = props.get("supervisor", "Unknown")
        
        print(f"District {district_num:2s}: {supervisor:<20} ({district_name})")
    
    print(f"\nTotal Districts: {len(features)}")

def main():
    """Main function to create SF districts GeoJSON."""
    
    print("SF Supervisorial Districts GeoJSON Creator")
    print("=" * 50)
    
    # Step 1: Fetch district data from SF Open Data API
    districts_data = fetch_sf_districts()
    if not districts_data:
        print("❌ Failed to fetch district data")
        return
    
    # Step 2: Convert to GeoJSON
    geojson_data = create_geojson(districts_data)
    if not geojson_data:
        print("❌ Failed to create GeoJSON")
        return
    
    # Step 3: Validate the GeoJSON
    if not validate_geojson(geojson_data):
        print("❌ GeoJSON validation failed")
        return
    
    # Step 4: Save to file
    output_file = save_geojson(geojson_data)
    if not output_file:
        print("❌ Failed to save GeoJSON")
        return
    
    # Step 5: Create summary
    create_district_summary(geojson_data)
    
    print(f"\n✅ SUCCESS!")
    print(f"GeoJSON file created: {output_file}")
    print(f"This file can be used for:")
    print(f"- Datawrapper custom map uploads")
    print(f"- District boundary visualization")
    print(f"- Spatial analysis of SF data")

if __name__ == "__main__":
    main() 
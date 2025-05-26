#!/usr/bin/env python3
"""
Debug script to check how business coordinates are being processed
"""

# Sample DataSF business data (what we should be getting)
correct_datasf_format = [
    {
        "dba_name": "William Miller",
        "location": {
            "type": "Point",
            "coordinates": [-122.458041, 37.774566]
        },
        "dba_start_date": "2025-05-24T00:00:00.000",
        "naic_code_description": "Accommodations"
    },
    {
        "dba_name": "Teamkat LLC/Dba Kash Consulting",
        "location": {
            "type": "Point",
            "coordinates": [-122.4496215, 37.78128]
        },
        "dba_start_date": "2025-05-24T00:00:00.000",
        "naic_code_description": "Administrative and Support Services"
    },
    {
        "dba_name": "Ann And Edgar Degiovanni",
        "location": {
            "type": "Point",
            "coordinates": [-122.455242, 37.755405]
        },
        "dba_start_date": "2025-05-24T00:00:00.000",
        "naic_code_description": "Accommodations"
    }
]

# What the explainer agent seems to be generating (clustered coordinates)
clustered_format = [
    {
        "coordinates": [-122.4516375, 37.800693],
        "title": "Ann And Edgar Degiovanni",
        "tooltip": "Arts, Entertainment, and Recreation"
    },
    {
        "coordinates": [-122.4516355, 37.800696],
        "title": "William Miller",
        "tooltip": "Other Services (except Public Administration)"
    },
    {
        "coordinates": [-122.4516295, 37.800672],
        "title": "Teamkat LLC/Dba Kash Consulting",
        "tooltip": "Professional, Scientific, and Technical Services"
    }
]

def analyze_coordinates():
    print("🔍 Analyzing Business Coordinate Processing\n")
    
    print("✅ CORRECT DataSF Format:")
    for business in correct_datasf_format:
        coords = business["location"]["coordinates"]
        print(f"  {business['dba_name']}: [{coords[0]}, {coords[1]}]")
    
    print(f"\n📊 Coordinate Range (Correct):")
    lons = [b["location"]["coordinates"][0] for b in correct_datasf_format]
    lats = [b["location"]["coordinates"][1] for b in correct_datasf_format]
    print(f"  Longitude range: {min(lons):.6f} to {max(lons):.6f} (spread: {max(lons) - min(lons):.6f})")
    print(f"  Latitude range: {min(lats):.6f} to {max(lats):.6f} (spread: {max(lats) - min(lats):.6f})")
    
    print("\n❌ CLUSTERED Format (What we're getting):")
    for business in clustered_format:
        coords = business["coordinates"]
        print(f"  {business['title']}: [{coords[0]}, {coords[1]}]")
    
    print(f"\n📊 Coordinate Range (Clustered):")
    lons = [b["coordinates"][0] for b in clustered_format]
    lats = [b["coordinates"][1] for b in clustered_format]
    print(f"  Longitude range: {min(lons):.6f} to {max(lons):.6f} (spread: {max(lons) - min(lons):.6f})")
    print(f"  Latitude range: {min(lats):.6f} to {max(lats):.6f} (spread: {max(lats) - min(lats):.6f})")
    
    print("\n🎯 DIAGNOSIS:")
    print("The issue is that the explainer agent is not properly extracting coordinates")
    print("from the DataSF location field. Instead of getting diverse coordinates")
    print("across San Francisco, all businesses are getting nearly identical coordinates.")
    print("\nThis suggests the coordinate extraction logic in the explainer agent")
    print("needs to be fixed to properly handle the DataSF location format:")
    print("  business['location']['coordinates'] -> [longitude, latitude]")

def show_correct_extraction():
    print("\n🔧 CORRECT EXTRACTION EXAMPLE:")
    print("```python")
    print("# When processing DataSF business data:")
    print("for business in business_data:")
    print("    if business.get('location') and business['location'].get('coordinates'):")
    print("        coords = business['location']['coordinates']  # [lon, lat]")
    print("        business_locations.append({")
    print("            'coordinates': coords,  # Use the actual coordinates")
    print("            'title': business.get('dba_name', 'Unknown'),")
    print("            'tooltip': f\"Industry: {business.get('naic_code_description', 'Unknown')}\",")
    print("            'industry': business.get('naic_code_description', 'Unknown')")
    print("        })")
    print("```")

if __name__ == "__main__":
    analyze_coordinates()
    show_correct_extraction() 
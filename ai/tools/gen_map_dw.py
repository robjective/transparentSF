import os
import requests
import json
import logging
from dotenv import load_dotenv
from pathlib import Path
import psycopg2
import psycopg2.extras

# Configure logging
logger = logging.getLogger(__name__)
# Remove explicit logging level setting - will use what's in .env
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Load environment variables
# Determine the project root based on the script's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Moves up two levels: tools -> ai -> project_root
ai_dir = script_dir.parent  # This should be the 'ai' directory

possible_env_paths = [
    ai_dir / '.env',          # Check ai/.env first
    project_root / '.env',    # Then check project_root/.env (original logic)
    Path.home() / '.env'      # Finally, check home directory
]

loaded_env = False
for env_path in possible_env_paths:
    if env_path.exists():
        logger.info(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        loaded_env = True
        break

if not loaded_env:
    logger.warning("No .env file found in project root or home directory. Relying on environment variables being set.")

DATAWRAPPER_API_KEY = os.getenv("DATAWRAPPER_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # Default if not set
DW_API_BASE_URL = "https://api.datawrapper.de/v3"

if not DATAWRAPPER_API_KEY:
    logger.error("DATAWRAPPER_API_KEY not found in environment variables. Script cannot function.")

def _make_dw_request(method, endpoint, headers=None, data=None, json_payload=None):
    """Helper function to make requests to Datawrapper API."""
    if not DATAWRAPPER_API_KEY:
        logger.error("Datawrapper API key is not configured.")
        return None

    url = f"{DW_API_BASE_URL}{endpoint}"
    
    default_headers = {
        "Authorization": f"Bearer {DATAWRAPPER_API_KEY}"
    }
    if headers:
        default_headers.update(headers)

    try:
        response = requests.request(method, url, headers=default_headers, data=data, json=json_payload)
        response.raise_for_status()
        
        if response.content:
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.info(f"Response from {method} {url} was not JSON, returning raw content.")
                return response.text
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
    return None

def get_db_connection():
    """Get a connection to the PostgreSQL database."""
    try:
        # Get database connection parameters from environment variables
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "transparentsf")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        logger.info(f"Connecting to database: {db_host}:{db_port}/{db_name}")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def get_district_shapes(district_type="supervisor", district_ids=None):
    """
    Fetch district shape data from the city's API endpoint.
    
    Args:
        district_type: Type of district ("supervisor" or "police")
        district_ids: List of district IDs to fetch (optional, will fetch all if None)
        
    Returns:
        Dictionary mapping district IDs to GeoJSON polygon data
    """
    logger.info(f"Fetching {district_type} district shapes")
    
    try:
        # Set up endpoint ID and field name based on district type
        if district_type.lower() == "supervisor":
            endpoint_id = "f2zs-jevy"  # Supervisor districts endpoint (2022)
            id_field = "sup_dist"
        elif district_type.lower() == "police":
            endpoint_id = "wkhw-cjsf"  # Police districts endpoint
            id_field = "district"
        else:
            logger.error(f"Unsupported district type: {district_type}")
            return None
        
        # Import the set_dataset function
        from tools.data_fetcher import set_dataset
        
        # Build the SoQL query - polygon is the column that contains the geometry data
        query = "SELECT *"  # Don't explicitly include 'polygon' as it's already part of *
        if district_ids:
            # Format list of district IDs for query
            district_filter = " OR ".join([f"{id_field}='{dist_id}'" for dist_id in district_ids])
            query += f" WHERE {district_filter}"
        
        # Create context variables dictionary
        context_variables = {}
        
        # Use set_dataset to fetch the data
        result = set_dataset(context_variables, endpoint=endpoint_id, query=query)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Error fetching district shapes: {result['error']}")
            return None
        
        # Check if we got data
        dataset = context_variables.get("dataset")
        if dataset is None or dataset.empty:
            logger.error("No district data retrieved")
            return None
        
        # Process the data into a dictionary mapping district IDs to GeoJSON
        district_shapes = {}
        for _, row in dataset.iterrows():
            district_id = str(row.get(id_field))
            if not district_id:
                logger.warning(f"Missing district ID in data row")
                continue
                
            # Extract the geometry data - check polygon column
            if "polygon" in row and row["polygon"]:
                geometry_data = row["polygon"]
                # Parse the geometry data if it's a string
                geometry = json.loads(geometry_data) if isinstance(geometry_data, str) else geometry_data
                district_shapes[district_id] = geometry
            else:
                logger.warning(f"No polygon data found for district {district_id}")
        
        return district_shapes
    
    except Exception as e:
        logger.error(f"Error fetching district shapes: {str(e)}")
        return None

def create_datawrapper_map(map_id):
    """
    Creates a Datawrapper map for the specified map ID, or returns existing published URL if available.
    
    Args:
        map_id: The ID of the map to generate
        
    Returns:
        The public URL of the existing or newly created Datawrapper map, or None if failed
    """
    logger.info(f"Creating Datawrapper map for map_id: {map_id}")
    
    if not DATAWRAPPER_API_KEY:
        logger.error("Cannot create map: DATAWRAPPER_API_KEY is not set.")
        return None
    
    try:
        # Fetch map data from database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the map
        cursor.execute("SELECT * FROM maps WHERE id = %s", (map_id,))
        map_record = cursor.fetchone()
        
        if not map_record:
            logger.error(f"Map with ID {map_id} not found in database")
            cursor.close()
            conn.close()
            return None
        
        # Convert to dictionary and process JSON fields
        map_data = dict(map_record)
        
        # Check if map already has a published URL
        if map_data.get("published_url"):
            logger.info(f"Map {map_id} already has published URL: {map_data['published_url']}")
            cursor.close()
            conn.close()
            return map_data["published_url"]
        
        # If no published URL, proceed with creating a new map
        logger.info(f"No published URL found for map {map_id}, creating new Datawrapper chart")
        
        map_data["location_data"] = json.loads(map_data["location_data"]) if isinstance(map_data["location_data"], str) else map_data["location_data"]
        map_data["metadata"] = json.loads(map_data["metadata"]) if map_data["metadata"] and isinstance(map_data["metadata"], str) else map_data["metadata"] or {}
        
        map_title = map_data["title"]
        map_type = map_data["type"]
        location_data = map_data["location_data"]
        metadata = map_data["metadata"]
        
        logger.info(f"Processing map: {map_title}, type: {map_type}")
        
        # If location_data is stored as a wrapper dict with csv_data, unpack it
        if isinstance(location_data, dict) and location_data.get("type") == "csv":
            csv_str = location_data.get("csv_data", "").strip()
            if csv_str:
                import csv
                import io
                reader = csv.DictReader(io.StringIO(csv_str))
                location_data = [row for row in reader]
                # ensure numeric where appropriate (district kept as str)
                for row in location_data:
                    for k in row:
                        try:
                            # attempt float conversion
                            row[k] = float(row[k]) if row[k] not in ["", None] and k != "district" else row[k]
                        except ValueError:
                            pass
            else:
                logger.error("location_data csv_data is empty – cannot create map")
                return None
        
        chart_id = None
        create_url = "https://api.datawrapper.de/v3/charts"
        create_headers = {
            "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Handle different map types with different approaches
        if map_type == "supervisor_district":
            # Since we don't have a SF-specific built-in map for supervisorial districts,
            # we'll use a point map with labeled locations instead of a choropleth
            create_payload = {
                "title": map_title,
                "type": "d3-maps-symbols",  # Use symbol map instead of choropleth
                "metadata": {
                    "describe": {
                        "intro": metadata.get("intro", f"Map of San Francisco supervisor districts"),
                        "byline": metadata.get("byline", "TransparentSF"),
                        "source-name": metadata.get("source", "SF Open Data")
                    }
                }
            }
            
            # Create the chart
            create_response = requests.post(create_url, headers=create_headers, json=create_payload)
            create_response.raise_for_status()
            chart_data = create_response.json()
            chart_id = chart_data["id"]
            logger.info(f"Supervisor district map created with ID: {chart_id}")
            
            # Get district center coordinates - these are approximate
            district_centers = {
                "1": {"lat": 37.7850, "lon": -122.4874},  # Richmond
                "2": {"lat": 37.7991, "lon": -122.4367},  # Marina
                "3": {"lat": 37.7977, "lon": -122.4142},  # North Beach
                "4": {"lat": 37.7818, "lon": -122.4608},  # Sunset
                "5": {"lat": 37.7734, "lon": -122.4400},  # Haight/Western Addition
                "6": {"lat": 37.7762, "lon": -122.4121},  # SOMA/Tenderloin
                "7": {"lat": 37.7293, "lon": -122.4539},  # West of Twin Peaks
                "8": {"lat": 37.7536, "lon": -122.4300},  # Mission
                "9": {"lat": 37.7447, "lon": -122.4198},  # Bernal Heights
                "10": {"lat": 37.7308, "lon": -122.3998},  # Bayview
                "11": {"lat": 37.7096, "lon": -122.4483}   # Oceanview/Ingleside
            }
            
            # Prepare CSV with district points
            csv_rows = ["name,latitude,longitude,value,color,label"]
            for item in location_data:
                district_id = str(item["district"])
                district_name = f"District {district_id}"
                value = item.get("value", 0)
                color = item.get("color", "#de2d26")
                
                # Use district center coordinates
                if district_id in district_centers:
                    lat = district_centers[district_id]["lat"]
                    lon = district_centers[district_id]["lon"]
                    csv_rows.append(f"{district_name},{lat},{lon},{value},{color},{district_id}")
            
            csv_data = "\n".join(csv_rows)
            
            # Upload CSV data
            upload_url = f"https://api.datawrapper.de/v3/charts/{chart_id}/data"
            upload_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
                "Content-Type": "text/csv"
            }
            
            upload_response = requests.put(upload_url, headers=upload_headers, data=csv_data)
            upload_response.raise_for_status()
            logger.info(f"CSV data uploaded to map {chart_id}")
            
            # Customize the map
            customize_url = f"https://api.datawrapper.de/v3/charts/{chart_id}"
            customize_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Get colors from metadata if provided
            colors = metadata.get("colors", ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"])
            
            customize_payload = {
                "metadata": {
                    "visualize": {
                        "basemap": "streets",  # Use streets basemap
                        "map-type": "symbol",
                        "symbol-type": "circle",
                        "symbol-size": "value",
                        "symbol-min-size": 20,
                        "symbol-max-size": 50,
                        "symbol-color": "color",
                        "tooltip": {
                            "enabled": True,
                            "template": "<b>District {{label}}</b><br>Value: {{value}}"
                        },
                        "labels": {
                            "enabled": True,
                            "column": "label"
                        },
                        "zoom-level": 12,
                        "initial-lat": 37.7749,
                        "initial-lon": -122.4194
                    },
                    "axes": {
                        "keys": ["longitude", "latitude"]
                    }
                }
            }
            
            customize_response = requests.patch(customize_url, headers=customize_headers, json=customize_payload)
            customize_response.raise_for_status()
            logger.info(f"Map {chart_id} customized")
            
        elif map_type == "police_district":
            # Similar approach for police districts - use points instead of choropleth
            create_payload = {
                "title": map_title,
                "type": "d3-maps-symbols",  # Use symbol map instead of choropleth
                "metadata": {
                    "describe": {
                        "intro": metadata.get("intro", f"Map of San Francisco police districts"),
                        "byline": metadata.get("byline", "TransparentSF"),
                        "source-name": metadata.get("source", "SF Open Data")
                    }
                }
            }
            
            # Create the chart
            create_response = requests.post(create_url, headers=create_headers, json=create_payload)
            create_response.raise_for_status()
            chart_data = create_response.json()
            chart_id = chart_data["id"]
            logger.info(f"Police district map created with ID: {chart_id}")
            
            # Police district center coordinates (approximate)
            district_centers = {
                "Bayview": {"lat": 37.7336, "lon": -122.3918},
                "Central": {"lat": 37.7981, "lon": -122.4071},
                "Ingleside": {"lat": 37.7249, "lon": -122.4482},
                "Mission": {"lat": 37.7597, "lon": -122.4214},
                "Northern": {"lat": 37.7816, "lon": -122.4298},
                "Park": {"lat": 37.7673, "lon": -122.4570},
                "Richmond": {"lat": 37.7778, "lon": -122.4650},
                "Southern": {"lat": 37.7815, "lon": -122.3975},
                "Taraval": {"lat": 37.7432, "lon": -122.4781},
                "Tenderloin": {"lat": 37.7838, "lon": -122.4141}
            }
            
            # Prepare CSV with district points
            csv_rows = ["name,latitude,longitude,value,color"]
            for item in location_data:
                district_id = str(item["district"])
                district_name = district_id
                value = item.get("value", 0)
                color = item.get("color", "#3182bd")
                
                # Use district center coordinates
                if district_id in district_centers:
                    lat = district_centers[district_id]["lat"]
                    lon = district_centers[district_id]["lon"]
                    csv_rows.append(f"{district_name},{lat},{lon},{value},{color}")
            
            csv_data = "\n".join(csv_rows)
            
            # Upload CSV data
            upload_url = f"https://api.datawrapper.de/v3/charts/{chart_id}/data"
            upload_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
                "Content-Type": "text/csv"
            }
            
            upload_response = requests.put(upload_url, headers=upload_headers, data=csv_data)
            upload_response.raise_for_status()
            logger.info(f"CSV data uploaded to map {chart_id}")
            
            # Customize the map
            customize_url = f"https://api.datawrapper.de/v3/charts/{chart_id}"
            customize_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Get colors from metadata if provided
            colors = metadata.get("colors", ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"])
            
            customize_payload = {
                "metadata": {
                    "visualize": {
                        "basemap": "streets",  # Use streets basemap
                        "map-type": "symbol",
                        "symbol-type": "circle",
                        "symbol-size": "value",
                        "symbol-min-size": 20,
                        "symbol-max-size": 50,
                        "symbol-color": "color",
                        "tooltip": {
                            "enabled": True,
                            "template": "<b>{{name}}</b><br>Value: {{value}}"
                        },
                        "labels": {
                            "enabled": True,
                            "column": "name"
                        },
                        "zoom-level": 12,
                        "initial-lat": 37.7749,
                        "initial-lon": -122.4194
                    },
                    "axes": {
                        "keys": ["longitude", "latitude"]
                    }
                }
            }
            
            customize_response = requests.patch(customize_url, headers=customize_headers, json=customize_payload)
            customize_response.raise_for_status()
            logger.info(f"Map {chart_id} customized")
            
        elif map_type in ["intersection", "point"]:
            # Point maps work as expected - no major changes needed
            create_payload = {
                "title": map_title,
                "type": "d3-maps-symbols",
                "metadata": {
                    "describe": {
                        "intro": metadata.get("intro", f"Map of San Francisco locations"),
                        "byline": metadata.get("byline", "TransparentSF"),
                        "source-name": metadata.get("source", "SF Open Data")
                    }
                }
            }
            
            # Create the chart
            create_response = requests.post(create_url, headers=create_headers, json=create_payload)
            create_response.raise_for_status()
            chart_data = create_response.json()
            chart_id = chart_data["id"]
            logger.info(f"Point/intersection map created with ID: {chart_id}")
            
            # Prepare CSV data with latitude, longitude, and values
            csv_rows = ["name,latitude,longitude,value,color"]
            for item in location_data:
                name = item.get("name", "Location")
                lat = str(item.get("latitude"))
                lon = str(item.get("longitude"))
                value = str(item.get("value", 100))
                color = item.get("color", "#e41a1c")
                csv_rows.append(f"{name},{lat},{lon},{value},{color}")
            
            csv_data = "\n".join(csv_rows)
            
            # Upload CSV data
            upload_url = f"https://api.datawrapper.de/v3/charts/{chart_id}/data"
            upload_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
                "Content-Type": "text/csv"
            }
            
            upload_response = requests.put(upload_url, headers=upload_headers, data=csv_data)
            upload_response.raise_for_status()
            logger.info(f"CSV data uploaded to map {chart_id}")
            
            # Customize the map
            customize_url = f"https://api.datawrapper.de/v3/charts/{chart_id}"
            customize_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            customize_payload = {
                "metadata": {
                    "visualize": {
                        "basemap": "streets",
                        "map-type": "symbol",
                        "symbol-type": "circle",
                        "symbol-size": "value",
                        "symbol-min-size": metadata.get("min_size", 5),
                        "symbol-max-size": metadata.get("max_size", 30),
                        "symbol-color": "color",
                        "tooltip": {
                            "enabled": True,
                            "template": "<b>{{name}}</b><br>Value: {{value}}"
                        },
                        "zoom-level": metadata.get("zoom", 13),
                        "initial-lat": metadata.get("initial_lat", 37.7749),
                        "initial-lon": metadata.get("initial_lon", -122.4194)
                    },
                    "axes": {
                        "keys": ["longitude", "latitude"]
                    }
                }
            }
            
            customize_response = requests.patch(customize_url, headers=customize_headers, json=customize_payload)
            customize_response.raise_for_status()
            logger.info(f"Map {chart_id} customized")
            
        else:
            logger.error(f"Unsupported map type: {map_type}")
            cursor.close()
            conn.close()
            return None
        
        # Publish the map
        if chart_id:
            publish_url = f"https://api.datawrapper.de/v3/charts/{chart_id}/publish"
            publish_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}"
            }
            
            publish_response = requests.post(publish_url, headers=publish_headers)
            publish_response.raise_for_status()
            logger.info(f"Map {chart_id} published")
            
            # Get the published URL
            get_url = f"https://api.datawrapper.de/v3/charts/{chart_id}"
            get_headers = {
                "Authorization": f"Bearer {DATAWRAPPER_API_KEY}"
            }
            
            get_response = requests.get(get_url, headers=get_headers)
            get_response.raise_for_status()
            chart_info = get_response.json()
            
            if "publicUrl" in chart_info:
                public_url = chart_info["publicUrl"]
                
                # Update the map record in the database
                cursor.execute(
                    "UPDATE maps SET published_url = %s, chart_id = %s WHERE id = %s",
                    (public_url, chart_id, map_id)
                )
                conn.commit()
                
                logger.info(f"Map {map_id} updated with publicUrl: {public_url} and chart_id: {chart_id}")
                cursor.close()
                conn.close()
                
                return public_url
            else:
                logger.error("Failed to get public URL")
                cursor.close()
                conn.close()
                return None
        else:
            logger.error("Failed to create chart_id")
            cursor.close()
            conn.close()
            return None
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error creating Datawrapper map: {str(e)}")
        return None

if __name__ == "__main__":
    # Example Usage:
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test fetching district shapes
    district_shapes = get_district_shapes("supervisor")
    if district_shapes:
        logger.info(f"Successfully fetched shapes for {len(district_shapes)} supervisor districts")
        
    # To test map creation, you need a valid map ID from the database
    # Uncomment and update with a valid map ID to test
    # map_url = create_datawrapper_map("your-map-id-here")
    # if map_url:
    #     logger.info(f"Map created and published at: {map_url}") 
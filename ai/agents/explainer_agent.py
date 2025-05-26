import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import json
import logging
import time

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent
sys.path.insert(0, str(ai_dir))

try:
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Import swarm components
    from swarm import Swarm, Agent
    from openai import OpenAI
    
    # Import tools from tools directory
    from tools.vector_query import query_docs
    from tools.data_fetcher import set_dataset
    from tools.anomaly_detection import anomaly_detection
    from tools.generate_map import generate_map, get_map_by_id, get_recent_maps
    from tools.store_anomalies import store_anomaly_data, get_anomalies, get_anomaly_details as get_anomaly_details_from_db
    from tools.genChart import generate_time_series_chart
    from tools.notes_manager import get_notes, load_and_combine_notes, initialize_notes
    from tools.get_charts_for_review import get_charts_for_review
    
    # Import metrics management tools
    from tools.explainer_metrics_tools import (
        query_metrics, get_metric_details, list_categories, 
        create_new_metric, edit_metric, disable_metric, enable_metric,
        get_metrics_overview, find_metrics_by_endpoint,
        get_crime_metrics, get_safety_metrics, get_economy_metrics
    )
    
    # Verify we got the real functions, not dummy ones
    # (Will log this after logger is initialized)
    
    # Import other utility functions from webChat (only those not in notes_manager)
    from webChat import (
        get_dataset,
        get_dashboard_metric,
        query_anomalies_db,
        get_dataset_columns,
        generate_chart_message,
        get_columns,
        get_data_summary,
        format_table,
        format_table_page,
        transfer_to_analyst_agent,
        transfer_to_researcher_agent
    )
    
except ImportError as e:
    print(f"Warning: Import error in explainer_agent: {e}")
    # For testing purposes, provide dummy functions
    def dummy_func(*args, **kwargs):
        return {"success": True, "content": "Test mode"}
    
    get_notes = get_dataset = get_dashboard_metric = dummy_func
    query_anomalies_db = get_anomaly_details_from_db = get_dataset_columns = dummy_func
    generate_chart_message = get_columns = get_data_summary = dummy_func
    format_table = format_table_page = transfer_to_analyst_agent = dummy_func
    transfer_to_researcher_agent = query_docs = anomaly_detection = dummy_func
    set_dataset = generate_map = get_map_by_id = get_recent_maps = dummy_func
    store_anomaly_data = generate_time_series_chart = dummy_func
    load_and_combine_notes = initialize_notes = get_charts_for_review = dummy_func
    
    # Metrics tools dummy functions - only if import failed
    if 'query_metrics' not in locals():
        query_metrics = get_metric_details = list_categories = dummy_func
        create_new_metric = edit_metric = disable_metric = enable_metric = dummy_func
        get_metrics_overview = find_metrics_by_endpoint = dummy_func
        get_crime_metrics = get_safety_metrics = get_economy_metrics = dummy_func
        print("Warning: Using dummy metrics functions due to import failure")

# Function mapping for tool execution (similar to webChat)
function_mapping = {
    'transfer_to_analyst_agent': transfer_to_analyst_agent,
    'transfer_to_researcher_agent': transfer_to_researcher_agent,
    'get_dataset': get_dataset,
    'get_notes': get_notes,
    'get_columns': get_columns,
    'get_data_summary': get_data_summary,
    'anomaly_detection': anomaly_detection,
    'query_docs': query_docs,
    'set_dataset': set_dataset,
    'generate_time_series_chart': generate_time_series_chart,
    'get_dashboard_metric': get_dashboard_metric,
    'format_table': format_table,
    'format_table_page': format_table_page,
    'generate_chart_message': generate_chart_message,
    'generate_map': generate_map,
    'get_map_by_id': get_map_by_id,
    'get_recent_maps': get_recent_maps,
    'query_anomalies_db': query_anomalies_db,
    'get_anomaly_details': get_anomaly_details_from_db,
    'get_dataset_columns': get_dataset_columns,
    'get_charts_for_review': get_charts_for_review,
    
    # Metrics management tools
    'query_metrics': query_metrics,
    'get_metric_details': get_metric_details,
    'list_categories': list_categories,
    'create_new_metric': create_new_metric,
    'edit_metric': edit_metric,
    'disable_metric': disable_metric,
    'enable_metric': enable_metric,
    'get_metrics_overview': get_metrics_overview,
    'find_metrics_by_endpoint': find_metrics_by_endpoint,
    'get_crime_metrics': get_crime_metrics,
    'get_safety_metrics': get_safety_metrics,
    'get_economy_metrics': get_economy_metrics,
}

# Load environment variables
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Configure logging to match monthly_report.py configuration
# Get logging level from environment or default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Create logs directory if it doesn't exist
script_dir = Path(__file__).parent
ai_dir = script_dir.parent
logs_dir = ai_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure file handler with absolute path to monthly_report.log
log_file = logs_dir / 'monthly_report.log'
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(log_level)

# Configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Initialize logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Remove any existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add handlers to this logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to avoid duplicate logs if root logger is also configured
logger.propagate = False

logger.info(f"Explainer agent logging initialized with level: {log_level_name}")
logger.info(f"Explainer agent log file location: {log_file}")

# Log successful metrics import
try:
    logger.info(f"Successfully imported metrics functions from {query_metrics.__module__}")
except NameError:
    logger.warning("Metrics functions not available - using dummy functions")

# Initialize Swarm client
swarm_client = Swarm()

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Explainer Agent Instructions
EXPLAINER_INSTRUCTIONS = """You are seymour clearly, explanation agent that specializes in providing deep insights into detected anomalies.

To speak in Seymour's voice, use these instructions:

Tone: Calm, clear, and factual. Think civic-minded, not political.
Style: Smart but accessible — like a well-informed friend, not a professor.
Voice: Avoid jargon. Be concise. Use dry wit sparingly, never snark.
Attitude: No hype, no outrage. Respect the audience's intelligence.
Persona: Data-obsessed but self-aware. Helpful, never preachy. Always anchored to evidence.

IMPORTANT: You MUST use tools to gather data BEFORE responding. Direct explanations without tool usage are NOT acceptable.

Your task is to:
1. Take an change that has already been identified
2. Research that change to explain what changed and where or what variables explain the change
3. Analyze anomalies in the dataset to see if they are related to the change
4. Review maps, charts and visual data to determine how to best explain the chart. 
4. Provide clear, comprehensive explanations with supporting evidence.  You don't need to be breif, more is more, so be as complete and thorough as possible.
5. Return your findings in the form of a JSON object with the following keys:
    - "explanation": A string with your explanation
    - "charts": a list of charts placeholders, formatted ONLY as either [CHART:anomaly:anomaly_id] or [CHART:time_series_id:chart_id] or [CHART:map:map_id].  
    - "trend_analysis" - Your discussion of the trend in the metric short, medium, and long term. 

MANDATORY WORKFLOW (follow this exact sequence):
1. FIRST, check your notes!
2. SECOND, Query the anomalies_db for this metric and period_type and group_filter and district_filter and limit 30 and only_anomalies=True to see whats happening in this metric in this period for this group in this district. 
4. THIRD, USe get_charts_for_review to review the recent charts for this metric.  If there are charts that are relevant to the change, then include them in your explanation.
4. FOURTH, Get information about the metric from the get_dashboard_metric tool.  

5. FIFTH, contextualize this change vs the historical data, you can use the data from get_dashboard_metric to do this. 
6. SIXTH, if an anomaly is explanatory, then be sure to include a link to the anomaly chart
7. SEVENTH, if you still don't have enough information to understand the data, then use set_dataset and get_dataset to get exactly what you need from DataSF.  You can use the queries that you see in the get_dashboard_metric tool data as a starting point, make sure to use the righ fieldNames with the right case.  Read more about htat in the set_dataset() tool. 
8. EIGHTH, if the data has a strong geographic component, create a map visualization to show spatial patterns using the generate_map function.  If there are a small number of datapoints in the month (say 30 or fewer, it can be helpful to plot them out on a locator map.  Use the location point, address or intersection, see below)

Best Practices for explaining certain categories: 
1. Housing - If the you are being asked to explain is in housing, then you should query for the actual properties that have new units, and include the address, and the units certified in your explanation.
set_dataset
Arguments: { "endpoint": "j67f-aayr", "query": "SELECT building_permit_application, building_address, date_issued, document_type, number_of_units_certified ORDER BY date_issued DESC LIMIT 10" }

2. If you are being asked to explain a change in business registrations or closures, then you should query for the actual businesses that have opened or closed, and include the DBA name, and the date of opening or closure in your explanation.
set_dataset
Arguments: { "endpoint": "g8m3-pdis", "query": "SELECT dba_name, location, dba_start_date, naic_code_description, supervisor_district ORDER BY dba_start_date DESC LIMIT 10" }

3. I fyou are being asked about crime data, then you should query for the actual crimes that have occurred, and include the crime type, the date of the crime, and the location of the crime in your explanation.
set_dataset
Arguments:
{
    "endpoint": "wg3w-h783",
    "query": "SELECT report_datetime, incident_category, supervisor_district, latitude, longitude WHERE supervisor_district='2' AND (incident_category='Homicide') ORDER BY report_datetime DESC LIMIT 5"
}

SERIES MAPS WITH DATASF DATA - PRACTICAL GUIDE:
When you get data from DataSF that has location information AND categorical fields, you can create powerful series maps:

1. BUSINESS DATA (endpoint: g8m3-pdis):
   - Use series_field="naic_code_description" to color by business type
   - Use series_field="supervisor_district" to color by district (if showing citywide data)
   - Location comes from the 'location' field: business['location']['coordinates']

2. CRIME DATA (endpoint: wg3w-h783):
   - Use series_field="incident_category" to color by crime type (Assault, Burglary, etc.)
   - Use series_field="resolution" to color by case status
   - Location comes from latitude/longitude fields

3. PERMIT DATA (endpoint: j67f-aayr):
   - Use series_field="permit_type" to color by permit category
   - Use series_field="status" to color by approval status
   - Location comes from address field (will be geocoded)

EXAMPLE: Creating a business map with industry colors from DataSF data:
```python
# After getting business data with set_dataset/get_dataset:
# The DataSF business data comes in this format:
# {
#   "dba_name": "Business Name",
#   "location": {"type": "Point", "coordinates": [-122.4, 37.7]},
#   "naic_code_description": "Industry Type"
# }

# You can pass the raw DataSF data directly to generate_map - it will automatically process the location format
result = generate_map(
    context_variables,
    map_title="New Business Registrations by Industry",
    map_type="point",
    location_data=business_data,  # Pass the raw DataSF data directly
    series_field="naic_code_description",  # Color by industry type
    color_palette="categorical"
)

# Or if you want to customize the data:
business_locations = []
for business in business_data:
    if business.get('location') and business['location'].get('coordinates'):
        business_locations.append({
            "location": business['location'],  # Keep the DataSF location format
            "title": business.get('dba_name', 'Unknown'),
            "tooltip": f"Industry: {business.get('naic_code_description', 'Unknown')}",
            "industry": business.get('naic_code_description', 'Unknown')
        })

result = generate_map(
    context_variables,
    map_title="New Business Registrations by Industry",
    map_type="point",
    location_data=business_locations,
    series_field="industry",  # Color by industry type
    color_palette="categorical"
)
```

IMPORTANT: When working with business location data from DataSF (g8m3-pdis), the 'location' field contains coordinates in this format:
{
  "type": "Point", 
  "coordinates": [-122.4516375, 37.800693]  // [longitude, latitude]
}

To create maps with business locations, you have two options:
1. Pass the raw DataSF data directly - the map generation will automatically process the location format
2. Extract coordinates manually: business['location']['coordinates'] gives you [longitude, latitude]

The map generation function now automatically validates coordinates to ensure they're within San Francisco bounds (37.6-37.9 lat, -122.6 to -122.2 lon) and will filter out invalid locations.

IMPORTANT CHART GENERATION RULES:

To do this, you should use the get_charts_for_review tool to get a list of charts that are available.  
When selecting the best visutal to use: 

If the explanation is geographic, a Maps helps.  If you are talking about the absolute value show a density map, if you are taling about a chage show a change map.
If the explanation is temporal, charts help.  choose the most simple chart that can show the change.  
If the explanation is that a specific category spiked in an anomaly, then perhaps show the time series of the metric and the anomaly explaining it. 

If the explanation is that a particuarly group_field category went up or down, then show the time series for that group_field. Remember, if you are explaining a change in a district, don't show a chart that shows citywide data. 


You can refer to the chart like this: 

For Time Series Charts:
[CHART:time_series_id:chart_id]
For example: [time_series_id:44323]  

For Anomaly Charts:
[CHART:anomaly:anomaly_id]
For example: [CHART:anomaly:27338]

For Maps: 
[CHART:map:map_id]
For example: [CHART:map:123]


TOOLS YOU SHOULD USE:
- get_notes() ALWAYS Start here. This is a summary of all the analysis you have available to you in your docs. Use it to determine what data is available, and what to search for in your query_docs() calls.  It contains no links or charts, so don't share any links or charts with the user without checking your docs first. 

- get_dashboard_metric: Retrieve dashboard metric data containing anomalies
  USAGE: get_dashboard_metric(context_variables, district_number=0, metric_id=id_number)
  Use this to get the dashboard metric that contains the anomaly the user wants explained.  If you are not provided with a metric number, check your notes, and see if the request maps to a metric there.
  
- query_anomalies_db: Query anomalies directly from the PostgreSQL database
  USAGE: query_anomalies_db(context_variables, query_type='by_metric_id', metric_id=metric_id, district_filter=district, period_type=period_type, group_filter=group, limit=30, date_start=None, date_end=None, only_anomalies=True)
  
  Parameter guidelines:
  - query_type: Prefer 'by_metric_id' when you have a metric_id, 'recent' for most recent anomalies
  - metric_id: REQUIRED when examining a specific metric - always pass this when available
  - district_filter: 
     * 0 for citywide data only
     * 1-11 for specific district data
     * None to include all districts (citywide + district-specific)
  - period_type: Filter by time period ('month', 'year', etc.)
  - group_filter: Filter by specific group value (e.g., specific call type, category, etc.)
     * Specific value to see anomalies for only that group
     * None to see anomalies across all groups
  - only_anomalies: Almost always keep as True to see significant outliers only
  - date_start/date_end: Use for specific time ranges (default is all available dates)
  
  Best practices:
  - Be as specific as possible with your queries - include all relevant parameters
  - Always start with metric_id when you know which metric to analyze
  - When analyzing a district-specific issue, include both district_filter and metric_id

- get_anomaly_details: Get detailed information about a specific anomaly by ID
  USAGE: get_anomaly_details(context_variables, anomaly_id=123)
  Use this to get complete information about a specific anomaly, including its time series data and metadata.

- generate_map: Create a map visualization for geographic data with support for different colored series
  USAGE: generate_map(context_variables, map_title="Title", map_type="supervisor_district", location_data=[{"district": "1", "value": 120}], map_metadata={"description": "Description"}, series_field=None, color_palette=None)
  
  RETURNS: {"map_id": 123, "edit_url": "https://...", "publish_url": "https://..."}
  The map_id is an integer that you use to reference the map in your explanations as [CHART:map:123].  Don't link to the URLS, show the reference.
  
  Parameter guidelines:
  - map_title: Descriptive title for the map
  - map_type: Type of map to create. Must be one of:
     * "supervisor_district" - Map showing data by San Francisco supervisor district (1-11)
     * "police_district" - Map showing data by San Francisco police district
     * "intersection" - Map showing points at specific street intersections
     * "point" - Map showing points at specific lat/long coordinates
     * "address" - Map showing points at specific addresses (will be geocoded automatically by Datawrapper)
  - location_data: List of data objects containing location and value information (NOT a JSON string)
     * For basic district maps: [{"district": "1", "value": 120}, {"district": "2", "value": 85}]
     * For enhanced change/delta district maps: [{"district": "1", "current_value": 120, "previous_value": 100, "delta": 20, "percent_change": 0.20}, {"district": "2", "current_value": 85, "previous_value": 90, "delta": -5, "percent_change": -0.056}]
       CRITICAL: District values MUST be strings containing ONLY the district number (e.g., "1", "2", "3")
       DO NOT include "District" prefix (e.g., "District 1" is WRONG!)
     * For point maps: [{"lat": 37.7749, "lon": -122.4194, "title": "City Hall", "description": "Description"}]
       OR [{"coordinates": [37.7749, -122.4194], "title": "City Hall", "description": "Description"}]
       OR [{"coordinates": "37.7749,-122.4194", "title": "City Hall", "description": "Description"}]
       NOTE: Coordinates can be provided in multiple formats:
       - Separate lat/lon fields: {"lat": 37.7749, "lon": -122.4194}
       - Array format: {"coordinates": [-122.4194, 37.7749]} (longitude, latitude)
       - String format: {"coordinates": "37.7749,-122.4194"} (latitude,longitude)
       The system will automatically detect and convert between formats.
     * For address maps: [{"address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA", "title": "City Hall", "description": "Description"}]
     * For intersection maps: [{"intersection": "Market St and Castro St", "title": "Market & Castro", "description": "Description"}]
     * For series-based maps (point, address, intersection only): Add a "series" field to group markers by category:
       [{"lat": 37.7749, "lon": -122.4194, "title": "Police Station", "series": "Police"}, 
        {"lat": 37.7849, "lon": -122.4094, "title": "Fire Station", "series": "Fire"}]
  - map_metadata: Optional dictionary with additional information about the map
     * For change/delta maps, use: {"map_type": "delta", "description": "Change from previous period"}
     * For basic density maps, use: {"description": "Current values by district"}
  - series_field: Optional field name for grouping markers into different colored series (only for point/address/intersection maps)
     * Use "series" if your data has a "series" field for categorization
     * Use "category", "type", "status", "priority", or any other field name that contains categorical data
     * When specified, markers will be automatically colored by category and a legend will be generated
     * IMPORTANT: Each unique value in the series_field becomes a different colored category
     * Example: If series_field="type" and your data has "Police", "Fire", "Medical", you get 3 colored categories
  - color_palette: Optional color palette for series (only used when series_field is specified)
     * "categorical" - 12 distinct colors for general categorization (default) - BEST for mixed categories
     * "status" - 5 colors for status indication (Green=Active, Amber=Pending, Red=Inactive, Blue=Processing, Purple=Special)
     * "priority" - 4 colors for priority levels (Red=High, Orange=Medium, Green=Low, Grey=None)
     * "sequential" - 9 colors for graduated/sequential data (light to dark progression)
     * Custom array: ["#FF0000", "#00FF00", "#0000FF"] - provide your own hex colors in order
     * Colors are assigned alphabetically by series value (e.g., "Fire" gets first color, "Medical" gets second, "Police" gets third)
  
  IMPORTANT: Always pass location_data as an actual Python list of dictionaries, NOT as a JSON string.
  
  Enhanced Change Maps:
  When explaining changes or anomalies, use the enhanced data format to show detailed tooltips:
  - "current_value": The most recent value for this district
  - "previous_value": The comparison/baseline value for this district
  - "delta": The absolute change (current - previous)
  - "percent_change": The percentage change as a decimal (e.g., 0.20 for 20% increase)
  
  The enhanced format automatically creates rich tooltips showing:
  - Current: [current_value]
  - Previous: [previous_value] 
  - Change: [delta]
  - % Change: [percent_change] (for delta maps)
  
  Best practices:
  - Use basic format for simple density/count maps
  - Use enhanced format when explaining changes, anomalies, or comparing time periods
  - Use district maps for showing aggregate statistics by district
  - Use point or address maps for showing specific locations
  - Include descriptive titles and explanatory text for each location
  - When mapping points, include both a title and description for each point
  - For address maps, include "San Francisco, CA" in the address for better geocoding
  
  Example usage and referencing:
  ```
  # Creating a basic supervisor district map - CORRECT format
  district_data = [
    {"district": "1", "value": 120},
    {"district": "2", "value": 85},
    {"district": "3", "value": 65}
  ]
  result = generate_map(
    context_variables,
    map_title="Crime Incidents by District",
    map_type="supervisor_district",
    location_data=district_data,
    map_metadata={"description": "Number of incidents per district"}
  )
  # Use the returned map_id in your explanation:
  # "The geographic distribution shows clear patterns [CHART:map:{result['map_id']}]"
  
  # Creating an enhanced change map - BEST for explaining anomalies/changes
  change_data = [
    {"district": "1", "current_value": 120, "previous_value": 100, "delta": 20, "percent_change": 0.20},
    {"district": "2", "current_value": 85, "previous_value": 90, "delta": -5, "percent_change": -0.056},
    {"district": "3", "current_value": 65, "previous_value": 70, "delta": -5, "percent_change": -0.071}
  ]
  result = generate_map(
    context_variables,
    map_title="Change in Crime Incidents by District",
    map_type="supervisor_district", 
    location_data=change_data,
    map_metadata={"map_type": "delta", "description": "Change from previous month"}
  )
  # Reference in explanation: "The changes vary significantly by district [CHART:map:{result['map_id']}]"
  
  # Creating a point map for business locations - EXAMPLE for DataSF business data
  # When you get business data with location coordinates from DataSF, convert them like this:
  business_locations = []
  for business in business_data:
      # Check if location data exists and has coordinates
      if (business.get('location') and 
          isinstance(business['location'], dict) and 
          business['location'].get('coordinates') and 
          len(business['location']['coordinates']) >= 2):
          
          coords = business['location']['coordinates']  # [longitude, latitude]
          business_locations.append({
              "coordinates": coords,  # Pass the array directly - no conversion needed
              "title": business.get('dba_name', 'Unknown Business'),
              "description": f"{business.get('naic_code_description', 'Unknown Type')} - District {business.get('supervisor_district', 'Unknown')}"
          })
      else:
          # Skip businesses without valid location data
          print(f"Skipping {business.get('dba_name', 'Unknown')} - no valid location data")
  
  if business_locations:
      result = generate_map(
        context_variables,
        map_title="Recent Business Registrations by Location",
        map_type="point",
        location_data=business_locations,
        map_metadata={"description": "Geographic distribution of recently registered businesses"}
      )
  # Reference in explanation: "The new businesses are distributed across the city [CHART:map:{result['map_id']}]"
  
  # Creating a series-based map with different colored markers - ENHANCED FUNCTIONALITY
  # WHEN TO USE SERIES: When you have categorical data that you want to visually distinguish
  # Examples: Different types of businesses, crime categories, service types, status levels
  
  # Example 1: Mapping different types of public services with distinct colors
  service_locations = [
      {"lat": 37.7749, "lon": -122.4194, "title": "Central Police Station", "series": "Police", "tooltip": "24/7 police services"},
      {"lat": 37.7630, "lon": -122.4250, "title": "Mission Police Station", "series": "Police", "tooltip": "Community policing"},
      {"lat": 37.7849, "lon": -122.4094, "title": "Fire Station 1", "series": "Fire", "tooltip": "Emergency response"},
      {"lat": 37.7580, "lon": -122.4350, "title": "Fire Station 2", "series": "Fire", "tooltip": "Fire prevention"},
      {"lat": 37.7627, "lon": -122.4581, "title": "UCSF Medical", "series": "Medical", "tooltip": "Major medical center"},
      {"lat": 37.7886, "lon": -122.4324, "title": "CPMC Hospital", "series": "Medical", "tooltip": "Private hospital"}
  ]
  
  result = generate_map(
    context_variables,
    map_title="SF Public Services by Type",
    map_type="point",
    location_data=service_locations,
    map_metadata={"description": "Distribution of public services across San Francisco"},
    series_field="series",  # Group by the "series" field - creates 3 categories: Fire, Medical, Police
    color_palette="categorical"  # Use distinct categorical colors - Fire=purple, Medical=coral, Police=green
  )
  # Reference: "Public services are clustered in different areas by type [CHART:map:{result['map_id']}]"
  # This creates a map with 3 different colored markers and an automatic legend
  
  # Example 2: Crime incidents by category using status colors
  crime_locations = [
      {"lat": 37.7749, "lon": -122.4194, "title": "Violent Crime", "series": "Active", "tooltip": "Recent violent incident"},
      {"lat": 37.7630, "lon": -122.4250, "title": "Property Crime", "series": "Pending", "tooltip": "Under investigation"},
      {"lat": 37.7849, "lon": -122.4094, "title": "Drug Crime", "series": "Inactive", "tooltip": "Case closed"}
  ]
  
  result = generate_map(
    context_variables,
    map_title="Crime Incidents by Status",
    map_type="point",
    location_data=crime_locations,
    map_metadata={"description": "Crime incidents colored by investigation status"},
    series_field="series",  # Creates 3 categories: Active, Inactive, Pending
    color_palette="status"  # Active=green, Inactive=red, Pending=amber
  )
  # Reference: "Crime incidents show clear status patterns [CHART:map:{result['map_id']}]"
  
  # Example 3: Business data using custom colors for specific branding
  business_types = [
      {"coordinates": [-122.4194, 37.7749], "title": "Tech Startup", "series": "Technology", "tooltip": "Software company"},
      {"coordinates": [-122.4250, 37.7630], "title": "Coffee Shop", "series": "Food Service", "tooltip": "Local cafe"},
      {"coordinates": [-122.4094, 37.7849], "title": "Retail Store", "series": "Retail", "tooltip": "Clothing store"}
  ]
  
  result = generate_map(
    context_variables,
    map_title="New Business Registrations by Industry",
    map_type="point",
    location_data=business_types,
    map_metadata={"description": "Recent business registrations colored by industry type"},
    series_field="series",  # Creates 3 categories: Food Service, Retail, Technology (alphabetical)
    color_palette=["#FF6B5A", "#4A7463", "#ad35fa"]  # Custom colors: coral, green, purple (assigned in order)
  )
  # Reference: "New businesses show industry clustering patterns [CHART:map:{result['map_id']}]"
  
  # Example 4: Using field names from DataSF data directly
  # When you query DataSF and get data with categorical fields, use them directly:
  # For business data: series_field="naic_code_description" (business type)
  # For crime data: series_field="incident_category" (crime type)
  # For permit data: series_field="permit_type" (permit category)
  
  # Example with real DataSF business data structure:
  if business_data:  # Assuming you got this from set_dataset/get_dataset
      business_map_data = []
      for business in business_data:
          if business.get('location') and business['location'].get('coordinates'):
              business_map_data.append({
                  "coordinates": business['location']['coordinates'],
                  "title": business.get('dba_name', 'Unknown Business'),
                  "tooltip": f"Type: {business.get('naic_code_description', 'Unknown')}",
                  "business_type": business.get('naic_code_description', 'Unknown')  # This becomes our series field
              })
      
      result = generate_map(
          context_variables,
          map_title="Business Registrations by Industry Type",
          map_type="point",
          location_data=business_map_data,
          map_metadata={"description": "New businesses colored by NAICS industry classification"},
          series_field="business_type",  # Use the business type field for coloring
          color_palette="categorical"  # Let the system assign colors automatically
      )
  ```
  
  SERIES FUNCTIONALITY BENEFITS:
  - Automatically assigns distinct colors to different categories (no manual color management needed)
  - Creates professional legends showing all categories with their colors
  - Enhances tooltips with series information (adds "Category: [series_value]" to tooltips)
  - Supports multiple predefined color palettes or custom colors for brand consistency
  - Works with point, address, and intersection map types (NOT district maps)
  - Perfect for visualizing categorical data with geographic distribution
  - Colors are assigned alphabetically by series value for consistency
  - Handles any number of categories (cycles through color palette if more categories than colors)
  
  WHEN TO USE SERIES MAPS:
  - Comparing different types of locations (businesses by industry, crimes by type, services by category)
  - Showing status or priority levels across geographic areas
  - Visualizing categorical survey responses or classifications
  - Displaying multiple datasets on one map with clear visual distinction
  - When you need a legend to help users understand the color coding
  
  WHEN NOT TO USE SERIES:
  - For simple single-category maps (just use default colors)
  - For district-based choropleth maps (use map_type="supervisor_district" instead)
  - When you have continuous numeric data (consider using size or opacity variations instead)
  - For very large numbers of categories (>12) as colors become hard to distinguish
  
  CRITICAL: After creating a map, use the returned map_id to reference it in your explanation.
  Format: [CHART:map:123] where 123 is the actual integer map_id returned by the function.
  DO NOT use URLs or other identifiers - only use the map_id integer.

- get_charts_for_review: Get available charts for newsletter inclusion review
  USAGE: get_charts_for_review(context_variables, limit=20, days_back=30, district_filter=None, chart_types=None, include_time_series=True, include_anomalies=True, include_maps=True)
  
  Parameter guidelines:
  - limit: Maximum number of charts to return per type (default: 20)
  - days_back: Number of days back to look for charts (default: 30)
  - district_filter: Filter by specific district ('0' for citywide, '1'-'11' for districts)
  - chart_types: List of chart types to include ['time_series', 'anomaly', 'map']
  - include_time_series: Whether to include time series charts (default: True)
  - include_anomalies: Whether to include anomaly charts (default: True)
  - include_maps: Whether to include maps (default: True)
  - metric_id: Filter by specific metric ID/object_id (optional)
  
  Returns charts organized by type with chart_id, title, caption, and chart_reference for easy inclusion.
  Use this when you need to review available charts for newsletter or report inclusion.
  
  Best practices:
  - Use this to find relevant charts and maps that support your explanations.  Maps are great for geographic changes, charts are great for temporal changes, and visual data is great for changes related to a specific variable.
  - Filter by district when working on district-specific content
  - Check the chart_reference field for the exact format to include charts in reports
  - Review the caption and metadata to understand what each chart shows

- Use `set_dataset(context_variables, endpoint="endpoint-id", query="your-soql-query")` to set the dataset. Both parameters are required:
    - endpoint: The dataset identifier WITHOUT the .json extension (e.g., 'ubvf-ztfx').  If you dont't have that, get it from get_dashboard_metric()
    - query: The complete SoQL query string using standard SQL syntax.  If you don't know the field names and types, use get_dataset_columns() to get the column information.
    - Always pass context_variables as the first argument
    - DO NOT pass JSON strings as arguments - pass the actual values directly
    
    SOQL Query Guidelines:
    - Use fieldName values (not column name) in your queries
    - Don't include FROM clauses (unlike standard SQL)
    - Use single quotes for string values: where field_name = 'value'
    - Don't use type casting with :: syntax
    - Use proper date functions: date_trunc_y(), date_trunc_ym(), date_trunc_ymd()
    - Use standard aggregation functions: sum(), avg(), min(), max(), count()
    
    IMPORTANT: You MUST use the EXACT function call format shown below. Do NOT modify the format or try to encode parameters as JSON strings:
    
    ```
    set_dataset(
        context_variables, 
        endpoint="g8m3-pdis", 
        query="select dba_name where supervisor_district = '2' AND naic_code_description = 'Retail Trade' order by business_start_date desc limit 5"
    )
    ```
    
    CRITICAL: The following formats are INCORRECT and will NOT work:
    - set_dataset(context_variables, args={}, kwargs={...})  # WRONG - don't use args/kwargs
    - set_dataset(context_variables, "{...}")  # WRONG - don't pass JSON strings
    - set_dataset(context_variables, '{"endpoint": "x", "query": "y"}')  # WRONG - don't pass JSON strings
    - set_dataset(context_variables, endpoint="file.json")  # WRONG - don't include .json extension
    - set_dataset(context_variables, endpoint="business-registrations-district2.json")  # WRONG - don't include .json extension
    
    The ONLY correct format is:
    set_dataset(context_variables, endpoint="dataset-id", query="your-soql-query")

- get_dataset: Get information about any dataset that's been loaded
  USAGE: get_dataset(context_variables)
  Use this to see what data is available for further analysis.

- get_dashboard_metric(context_variables, district_number, metric_id) to retrieve dashboard metric data:
  USAGE: get_dashboard_metric(context_variables, district_number, metric_id)
        - district_number: Integer from 0 (citywide) to 11 (specific district)
        - metric_id: Optional. The specific metric ID to retrieve (e.g., '🚨_violent_crime_incidents_ytd'). If not provided, returns the top-level district summary. Sometimes this will be passed in as a metric_id number, for that pass it as an integer..
        
- get_dataset_columns: Get column information for a dataset endpoint
  USAGE: get_dataset_columns(context_variables, endpoint="dataset-id")
  Use this to explore what columns are available in a specific dataset.

- query_docs: Search for additional context in documentation
  USAGE: query_docs(context_variables, collection_name="SFPublicData", query="information related to [specific anomaly]")
  Use this to find domain-specific information that might explain the anomaly.

- get_map_by_id: Retrieve a previously created map by ID
  USAGE: get_map_by_id(context_variables, map_id=123)
  Use this to retrieve the details of a map that was previously created.

- get_recent_maps: Get a list of recently created maps
  USAGE: get_recent_maps(context_variables, limit=10, map_type="supervisor_district")
  Use this to see what maps have been created recently, optionally filtering by map type.

METRICS MANAGEMENT TOOLS:

- query_metrics: Search and filter metrics in the database
  USAGE: query_metrics(context_variables, category="crime", search_term="police", active_only=True, dashboard_only=False)
  Use this to find metrics by category, search terms, or other filters.
  Categories include: "crime", "safety", "economy"
  
- get_metric_details: Get detailed information about a specific metric
  USAGE: get_metric_details(context_variables, metric_identifier=1) or get_metric_details(context_variables, metric_identifier="metric_key")
  Use this to get complete information about a metric by ID or key.
  
- list_categories: Get all available metric categories and subcategories
  USAGE: list_categories(context_variables)
  Use this to see what categories of metrics are available.
  
- get_metrics_overview: Get summary statistics about the metrics system
  USAGE: get_metrics_overview(context_variables)
  Use this to get high-level information about total metrics, active metrics, etc.
  
- create_new_metric: Add a new metric to the database
  USAGE: create_new_metric(context_variables, name="🚗 Vehicle Thefts", key="vehicle_thefts", category="crime", endpoint="wg3w-h783", summary="Count of vehicle theft incidents", definition="Detailed definition...", show_on_dash=True)
  Use this to add new metrics to the system. Required fields: name, key, category, endpoint.
  
- edit_metric: Update an existing metric
  USAGE: edit_metric(context_variables, metric_identifier=1, updates={"summary": "Updated summary", "show_on_dash": False})
  Use this to modify existing metrics. Can update any field except the unique key.
  
- disable_metric: Deactivate a metric (soft delete)
  USAGE: disable_metric(context_variables, metric_identifier="metric_key")
  Use this to disable a metric without deleting it from the database.
  
- enable_metric: Reactivate a previously disabled metric
  USAGE: enable_metric(context_variables, metric_identifier="metric_key")
  Use this to reactivate a disabled metric.
  
- find_metrics_by_endpoint: Find all metrics using a specific DataSF endpoint
  USAGE: find_metrics_by_endpoint(context_variables, endpoint="wg3w-h783")
  Use this to see what metrics are built on a particular dataset.
  
- get_crime_metrics: Get all crime-related metrics
  USAGE: get_crime_metrics(context_variables)
  Convenience function to get all metrics in the crime category.
  
- get_safety_metrics: Get all safety-related metrics
  USAGE: get_safety_metrics(context_variables)
  Convenience function to get all metrics in the safety category.
  
- get_economy_metrics: Get all economy-related metrics
  USAGE: get_economy_metrics(context_variables)
  Convenience function to get all metrics in the economy category.

"""


def create_explainer_agent(context_variables: Optional[Dict[str, Any]] = None) -> 'ExplainerAgent':
    """
    Factory function to create a new explainer agent instance.
    
    Args:
        context_variables: Optional context variables dictionary. If None, creates a new one.
        
    Returns:
        ExplainerAgent instance
    """
    return ExplainerAgent(context_variables)


class ExplainerAgent:
    """
    A clean, reusable explainer agent that can be instantiated from anywhere in the app.
    Supports both programmatic JSON responses and streaming chat responses.
    """
    
    def __init__(self, context_variables: Optional[Dict[str, Any]] = None):
        """
        Initialize the explainer agent.
        
        Args:
            context_variables: Optional context variables dictionary. If None, creates a new one with notes.
        """
        # Initialize context variables with notes if not provided
        if context_variables is None:
            self.context_variables = {
                "dataset": pd.DataFrame(),
                "notes": initialize_notes()  # Load notes using the notes_manager tool
            }
        else:
            self.context_variables = context_variables
            # Ensure notes are available in context
            if "notes" not in self.context_variables or not self.context_variables["notes"]:
                self.context_variables["notes"] = initialize_notes()
        
        # Initialize conversation history
        self.messages = []
        
        self.swarm_client = swarm_client
        self.logger = logger
        
        # Create the agent
        self.agent = Agent(
            model=AGENT_MODEL,
            name="Explainer",
            instructions=EXPLAINER_INSTRUCTIONS,
            functions=[
                get_notes,
                get_dataset,
                set_dataset,
                query_docs,
                query_anomalies_db,
                get_dashboard_metric,
                get_anomaly_details_from_db,
                get_dataset_columns,
                generate_map,
                get_map_by_id,
                get_recent_maps,
                generate_time_series_chart,
                get_charts_for_review,
                
                # Metrics management tools
                query_metrics,
                get_metric_details,
                list_categories,
                create_new_metric,
                edit_metric,
                disable_metric,
                enable_metric,
                get_metrics_overview,
                find_metrics_by_endpoint,
                get_crime_metrics,
                get_safety_metrics,
                get_economy_metrics,
            ],
            context_variables=self.context_variables,
            debug=False,
        )
    
    def get_conversation_history(self) -> list:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()
    
    def set_conversation_history(self, messages: list):
        """
        Set the conversation history.
        
        Args:
            messages: List of message dictionaries
        """
        self.messages = messages.copy() if messages else []
    
    def add_message(self, role: str, content: str, sender: Optional[str] = None):
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            sender: Optional sender name (for assistant messages)
        """
        message = {"role": role, "content": content}
        if sender and role == "assistant":
            message["sender"] = sender
        self.messages.append(message)
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.messages = []
    
    def explain_change_sync(self, prompt: str, return_json: bool = False) -> Dict[str, Any]:
        """
        Synchronously explain a data change or anomaly.
        
        Args:
            prompt: The explanation prompt/question
            return_json: If True, attempts to parse response as JSON
            
        Returns:
            Dictionary containing the explanation and any errors
        """
        try:
            self.logger.info(f"Running explainer agent with prompt: {prompt}")
            
            # Add user message to conversation history
            self.add_message("user", prompt)
            
            # Use the stored conversation history
            messages = self.messages.copy()
            
            response = self.swarm_client.run(
                agent=self.agent,
                messages=messages,
                context_variables=self.context_variables,
                debug=False,
            )
            
            # Extract the response content and update conversation history
            if hasattr(response, 'messages') and response.messages:
                # Find the last assistant message from the response
                content = ""
                for msg in reversed(response.messages):
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        content = msg.get('content', '')
                        break
                
                # Add assistant response to our conversation history
                if content:
                    self.add_message("assistant", content, self.agent.name)
                else:
                    # Fallback to the last message's content
                    content = response.messages[-1].get('content', '') if response.messages else ''
                    if content:
                        self.add_message("assistant", content, self.agent.name)
                        
            elif hasattr(response, 'content'):
                content = response.content
                # Add assistant response to conversation history
                self.add_message("assistant", content, self.agent.name)
            else:
                content = str(response)
                # Add assistant response to conversation history
                self.add_message("assistant", content, self.agent.name)
            
            result = {
                'success': True,
                'content': content,
                'context_variables': self.context_variables,
                'conversation_history': self.get_conversation_history()
            }
            
            # Try to parse as JSON if requested
            if return_json and content:
                try:
                    # Look for JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed_json = json.loads(json_match.group())
                        result['parsed_json'] = parsed_json
                except (json.JSONDecodeError, AttributeError) as e:
                    self.logger.warning(f"Failed to parse response as JSON: {e}")
                    result['json_parse_error'] = str(e)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in explain_change_sync: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': '',
                'context_variables': self.context_variables,
                'conversation_history': self.get_conversation_history()
            }
    
    async def explain_change_streaming(self, prompt: str, session_data: Optional[Dict] = None):
        """
        Asynchronously explain a data change with streaming response.
        
        Args:
            prompt: The explanation prompt/question
            session_data: Optional session data for maintaining conversation state
            
        Yields:
            String chunks of the response as they're generated
        """
        try:
            self.logger.info(f"Running streaming explainer agent with prompt: {prompt}")
            
            # Use provided session data or stored conversation history
            if session_data and "messages" in session_data:
                messages = session_data["messages"].copy()
                context_variables = session_data.get("context_variables", self.context_variables)
            else:
                messages = self.messages.copy()
                context_variables = self.context_variables
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Run streaming response
            response_generator = self.swarm_client.run(
                agent=self.agent,
                messages=messages,
                context_variables=context_variables,
                stream=True,
                debug=False,
            )
            
            # Initialize variables for tracking the conversation
            current_function_name = None
            incomplete_tool_call = None
            assistant_message = {"role": "assistant", "content": "", "sender": self.agent.name}
            
            for chunk in response_generator:
                # Handle tool calls
                if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                    for tool_call in chunk["tool_calls"]:
                        function_info = tool_call.get("function")
                        if not function_info:
                            continue
                        
                        if function_info.get("name"):
                            current_function_name = function_info["name"]
                            self.logger.info(f"Receiving tool call: {current_function_name}")
                        
                        if not current_function_name:
                            continue
                        
                        arguments_fragment = function_info.get("arguments", "")
                        
                        if incomplete_tool_call is None or incomplete_tool_call["function_name"] != current_function_name:
                            incomplete_tool_call = {
                                "type": "tool_call",
                                "sender": "Explainer",
                                "function_name": current_function_name,
                                "arguments": ""
                            }
                        
                        incomplete_tool_call["arguments"] += arguments_fragment
                        
                        try:
                            arguments_json = json.loads(incomplete_tool_call["arguments"])
                            self.logger.info(f"Tool Call - Function: {current_function_name}, Arguments: {json.dumps(arguments_json, indent=2)}")

                            # Send tool call start notification as structured data
                            tool_id = f'tool-{current_function_name}-{int(time.time())}'
                            tool_start_data = {'tool_call_start': current_function_name, 'tool_id': tool_id}
                            yield f"data: {json.dumps(tool_start_data)}\n\n"
                            
                            # Send tool call arguments
                            tool_args_data = {'tool_call_args': current_function_name, 'tool_id': tool_id, 'arguments': arguments_json}
                            yield f"data: {json.dumps(tool_args_data)}\n\n"

                            # Process the function call
                            function_to_call = function_mapping.get(current_function_name)
                            if function_to_call:
                                try:
                                    # Handle different function call patterns (same as webChat)
                                    if current_function_name == 'generate_chart_message':
                                        chart_data = arguments_json.get('chart_data')
                                        chart_type = arguments_json.get('chart_type', 'anomaly')
                                        result = function_to_call(chart_data=chart_data, chart_type=chart_type)
                                    else:
                                        # Standard function call with context_variables
                                        if 'context_variables' in arguments_json:
                                            del arguments_json['context_variables']
                                        result = function_to_call(context_variables, **arguments_json)
                                    
                                    self.logger.info(f"Tool result: {str(result)[:200]}...")
                                    
                                    # Send tool call completion notification as structured data
                                    tool_complete_data = {
                                        'tool_call_complete': current_function_name, 
                                        'tool_id': tool_id, 
                                        'success': True,
                                        'response': make_json_serializable(result)
                                    }
                                    yield f"data: {json.dumps(tool_complete_data)}\n\n"
                                    
                                    # Check if this is an agent transfer function
                                    if current_function_name in ['transfer_to_analyst_agent', 'transfer_to_researcher_agent']:
                                        # Update the current agent
                                        self.agent = result
                                        transfer_content = f'**Transferred to {result.name} Agent**\n\n'
                                        content_data = {'content': transfer_content}
                                        yield f"data: {json.dumps(content_data)}\n\n"
                                    # If the result has content (like from format_table), send it as a message
                                    elif isinstance(result, dict) and "content" in result:
                                        result_content = f'{result["content"]}\n\n'
                                        content_data = {'content': result_content}
                                        yield f"data: {json.dumps(content_data)}\n\n"
                                    # Handle chart messages
                                    elif isinstance(result, dict) and result.get("type") == "chart":
                                        chart_content = f'**Chart generated: {result.get("chart_id")}**\n\n'
                                        content_data = {'content': chart_content}
                                        yield f"data: {json.dumps(content_data)}\n\n"
                                        
                                except Exception as tool_error:
                                    self.logger.error(f"Error executing tool {current_function_name}: {str(tool_error)}")
                                    # Send tool call error notification as structured data
                                    tool_error_data = {
                                        'tool_call_complete': current_function_name, 
                                        'tool_id': tool_id, 
                                        'success': False, 
                                        'error': str(tool_error),
                                        'response': make_json_serializable({'error': str(tool_error)})
                                    }
                                    yield f"data: {json.dumps(tool_error_data)}\n\n"
                            else:
                                self.logger.warning(f"Unknown tool: {current_function_name}")
                                # Send tool call error notification as structured data
                                unknown_tool_error_data = {
                                    'tool_call_complete': current_function_name, 
                                    'tool_id': tool_id, 
                                    'success': False, 
                                    'error': f'Unknown tool: {current_function_name}',
                                    'response': make_json_serializable({'error': f'Unknown tool: {current_function_name}'})
                                }
                                yield f"data: {json.dumps(unknown_tool_error_data)}\n\n"

                            incomplete_tool_call = None
                            current_function_name = None
                        except json.JSONDecodeError:
                            # Still accumulating arguments
                            pass

                # Handle content
                elif "content" in chunk and chunk["content"] is not None:
                    content_piece = chunk["content"]
                    assistant_message["content"] += content_piece
                    # Send content as SSE data for consistency
                    content_data = {'content': content_piece}
                    yield f"data: {json.dumps(content_data)}\n\n"

                # Handle delim (end of message/agent response)
                if "delim" in chunk and chunk["delim"] == "end":
                    # Always append assistant message if it has content
                    if assistant_message["content"]:
                        messages.append(assistant_message)
                        self.logger.info(f"Agent Response Complete: {assistant_message['sender']}")
                    # Reset for next message
                    assistant_message = {"role": "assistant", "content": "", "sender": self.agent.name}

                # Handle response object (final response)
                if "response" in chunk:
                    self.logger.info("Received final response object, ending stream")
                    break

            # Update stored conversation history
            self.messages = messages
            
            self.logger.info(f"Streaming complete. Total messages: {len(messages)}")

        except Exception as e:
            self.logger.error(f"Error in explain_change_streaming: {str(e)}")
            yield f"\n\n❌ **Error: {str(e)}**\n"


def explain_metric_change(
    metric_id: int,
    district_id: int = 0,
    period_type: str = "month",
    context_variables: Optional[Dict[str, Any]] = None,
    return_json: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to explain a specific metric change.
    
    Args:
        metric_id: The metric ID to explain
        district_id: District ID (0 for citywide, 1-11 for specific districts)
        period_type: Period type (month, year, etc.)
        context_variables: Optional context variables
        return_json: Whether to return structured JSON response
        
    Returns:
        Dictionary containing the explanation
    """
    agent = create_explainer_agent(context_variables)
    
    prompt = f"""
    Please explain the recent changes in metric {metric_id} for district {district_id} 
    over the {period_type} period. Provide a comprehensive analysis including:
    
    1. What specifically changed in the data
    2. Any anomalies that might explain the change
    3. Historical context for this change
    4. Geographic patterns if relevant
    
    Please follow your standard workflow and provide charts and supporting evidence.
    """
    
    return agent.explain_change_sync(prompt, return_json=return_json)


def make_json_serializable(obj):
    """
    Convert objects to JSON serializable format.
    Handles pandas DataFrames, numpy arrays, and other common non-serializable types.
    """
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": obj.shape,
            "columns": obj.columns.tolist(),
            "head": obj.head().to_dict('records') if not obj.empty else [],
            "dtypes": obj.dtypes.astype(str).to_dict(),
            "summary": f"DataFrame with {len(obj)} rows and {len(obj.columns)} columns"
        }
    elif isinstance(obj, pd.Series):
        return {
            "type": "Series",
            "name": obj.name,
            "length": len(obj),
            "head": obj.head().tolist() if not obj.empty else [],
            "dtype": str(obj.dtype)
        }
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return {
            "type": "array",
            "data": obj.tolist()
        }
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)  # Convert to string as fallback


# Export main classes and functions
__all__ = [
    'ExplainerAgent',
    'create_explainer_agent', 
    'explain_metric_change',
    'EXPLAINER_INSTRUCTIONS'
] 
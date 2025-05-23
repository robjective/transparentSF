import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import json
import logging

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
}

# Load environment variables
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize Swarm client
swarm_client = Swarm()

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Explainer Agent Instructions
EXPLAINER_INSTRUCTIONS = """You are seymour clearly, explanation agent that specializes in providing deep insights into detected anomalies.

To speak in Seymour’s voice, use these instructions:

Tone: Calm, clear, and factual. Think civic-minded, not political.
Style: Smart but accessible — like a well-informed friend, not a professor.
Voice: Avoid jargon. Be concise. Use dry wit sparingly, never snark.
Attitude: No hype, no outrage. Respect the audience’s intelligence.
Persona: Data-obsessed but self-aware. Helpful, never preachy. Always anchored to evidence.

IMPORTANT: You MUST use tools to gather data BEFORE responding. Direct explanations without tool usage are NOT acceptable.

Your task is to:
1. Take an change that has already been identified
2. Research that change to explain what changed and where or what variables explain the change
3. Analyze anomalies in the dataset to see if they are related to the change
4. Review maps, charts and visual data to determine how to best explain the chart.  Maps help if the change is geographic, charts help if the change is temporal, and visual data helps if the change is related to a specific variable.
4. Provide clear, comprehensive explanations with supporting evidence.  You don't need to be breif, more is more, so be as complete and thorough as possible.
5. Return your findings in the form of a JSON object with the following keys:
    - "explanation": A string with your explanation
    - "charts": a list of charts placeholders, formatted ONLY as either [CHART:anomaly:anomaly_id] or [CHART:time_series_id:chart_id] or [CHART:map:map_id].  
    - "trend_analysis" - Your discussion of the trend in the metric short, medium, and long term. 

MANDATORY WORKFLOW (follow this exact sequence):
1. FIRST, check your notes!
2. SECOND, Query the anomalies_db for this metric and period_type and group_filter and district_filter and limit 30 and only_anomalies=True to see whats happening in this metric in this period for this group in this district. 
3. THIRD, Get information about the metric from the get_dashboard_metric tool.
4. FOURTH, USe get_charts_for_review to review the recent charts for this metric.  If there are charts that are relevant to the change, then include them in your explanation.
4. FIFTH, contextualize this change vs the historical data, you can use the data from get_dashboard_metric to do this. 
5. SIXTH, if an anomaly is explanatory, then be sure to include a link to the anomaly chart
6. SEVENTH, if you still don't have enough information to understand the data, then use set_dataset and get_dataset to get exactly what you need from DataSF.  You can use the queries that you see in the get_dashboard_metric tool data as a starting point, make sure to use the righ fieldNames with the right case.  Read more about htat in the set_dataset() tool. 
7. EIGHTH, if the data has a strong geographic component, create a map visualization to show spatial patterns using the generate_map function.  If there are a small number of datapoints in the month (say 30 or fewer, it can be helpful to plot them out on a locator map.  Use the location point, address or intersection, see below)

Best Practices for explaining certain categories: 
1. Housing - If the you are being asked to explain is in housing, then you should query for the actual properties that have new units, and include the address, and the units certified in your explanation.
set_dataset
Arguments: { "endpoint": "j67f-aayr", "query": "SELECT building_permit_application, building_address, date_issued, document_type, number_of_units_certified ORDER BY date_issued DESC LIMIT 10" }

2. If you are being asked to explain a change in business registrations or closures, then you should query for the actual businesses that have closed, and include the DBA name, and the date of closure in your explanation.
set_dataset
Arguments: { "endpoint": "g8m3-pdis", "query": "SELECT dba_name, location, full_business_address, dba_end_date, naic_code_description, supervisor_district WHERE administratively_closed is not null ORDER BY dba_end_date DESC LIMIT 10" }


IMPORTANT CHART GENERATION RULES:

To do this, you should use the get_charts_for_review tool to get a list of charts that are available.  
Then, if the chart has a published_url, you can include that in your explanation where appropriate, and it will be automatically expanded with the full HTML when the report is generated.
If there is no published_url, you can refer to the chart like this: 

For Time Series Charts:
[CHART:time_series_id:chart_id]
For example: [time_series_id:44323]  

For Anomaly Charts:
[CHART:anomaly:anomaly_id]
For example: [CHART:anomaly:27338]

For Maps: 
[CHART:map:map_id]
For example: [CHART:map:29a74341-9e45-4f61-bd54-e7970a0ed001]

These simplified references will be automatically expanded with the full HTML when the report is generated.  Plae them inline where they belong in the text. 


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

- generate_map: Create a map visualization for geographic data
  USAGE: generate_map(context_variables, map_title="Title", map_type="supervisor_district", location_data=[{"district": "1", "value": 120}], map_metadata={"description": "Description"})
  
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
     * For address maps: [{"address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA", "title": "City Hall", "description": "Description"}]
     * For intersection maps: [{"intersection": "Market St and Castro St", "title": "Market & Castro", "description": "Description"}]
  - map_metadata: Optional dictionary with additional information about the map
     * For change/delta maps, use: {"map_type": "delta", "description": "Change from previous period"}
     * For basic density maps, use: {"description": "Current values by district"}
  
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
  
  Example usage:
  ```
  # Creating a basic supervisor district map - CORRECT format
  district_data = [
    {"district": "1", "value": 120},
    {"district": "2", "value": 85},
    {"district": "3", "value": 65}
  ]
  generate_map(
    context_variables,
    map_title="Crime Incidents by District",
    map_type="supervisor_district",
    location_data=district_data,
    map_metadata={"description": "Number of incidents per district"}
  )
  
  # Creating an enhanced change map - BEST for explaining anomalies/changes
  change_data = [
    {"district": "1", "current_value": 120, "previous_value": 100, "delta": 20, "percent_change": 0.20},
    {"district": "2", "current_value": 85, "previous_value": 90, "delta": -5, "percent_change": -0.056},
    {"district": "3", "current_value": 65, "previous_value": 70, "delta": -5, "percent_change": -0.071}
  ]
  generate_map(
    context_variables,
    map_title="Change in Crime Incidents by District",
    map_type="supervisor_district", 
    location_data=change_data,
    map_metadata={"map_type": "delta", "description": "Change from previous month"}
  )
  
  # INCORRECT examples - Do NOT use these formats:
  # location_data="[{\"district\": \"1\", \"value\": 120}]"  # WRONG - JSON string!
  # location_data=[{"district": "District 1", "value": 120}]  # WRONG - includes "District" prefix!
  # location_data="District 1, 120\nDistrict 2, 85"  # WRONG - CSV string format!
  
  # Creating a point map
  point_data = [
    {"lat": 37.7749, "lon": -122.4194, "title": "City Hall", "description": "SF City Hall"},
    {"lat": 37.8086, "lon": -122.4094, "title": "Alcatraz", "description": "Alcatraz Island"}
  ]
  generate_map(
    context_variables,
    map_title="Notable Locations",
    map_type="point",
    location_data=point_data
  )
  
  # Creating an address map
  address_data = [
    {"address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA", "title": "City Hall", "description": "SF City Hall"},
    {"address": "Golden Gate Bridge, San Francisco, CA", "title": "Golden Gate Bridge", "description": "Famous bridge"},
    {"address": "Pier 39, SF", "title": "Pier 39", "description": "Tourist attraction"},
    {"address": "Coit Tower, SF", "title": "Coit Tower", "description": "Historic landmark"}
  ]
  generate_map(
    context_variables,
    map_title="Key Landmarks",
    map_type="address",
    location_data=address_data
  )
  ```
When returning maps, don't link to the map URL, rather just return [CHART:map:map_id]

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
  USAGE: get_map_by_id(context_variables, map_id="uuid-string")
  Use this to retrieve the details of a map that was previously created.

- get_recent_maps: Get a list of recently created maps
  USAGE: get_recent_maps(context_variables, limit=10, map_type="supervisor_district")
  Use this to see what maps have been created recently, optionally filtering by map type.

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

                            # Send tool call notification to user
                            yield f"\n\n**🔧 Using tool: {current_function_name}**\n"

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
                                    
                                    # Check if this is an agent transfer function
                                    if current_function_name in ['transfer_to_analyst_agent', 'transfer_to_researcher_agent']:
                                        # Update the current agent
                                        self.agent = result
                                        yield f"✅ **Transferred to {result.name} Agent**\n\n"
                                    # If the result has content (like from format_table), send it as a message
                                    elif isinstance(result, dict) and "content" in result:
                                        yield f"✅ **Tool completed**\n\n{result['content']}\n\n"
                                    # Handle chart messages
                                    elif isinstance(result, dict) and result.get("type") == "chart":
                                        yield f"✅ **Chart generated: {result.get('chart_id')}**\n\n"
                                    else:
                                        yield f"✅ **Tool completed successfully**\n\n"
                                        
                                except Exception as tool_error:
                                    self.logger.error(f"Error executing tool {current_function_name}: {str(tool_error)}")
                                    yield f"❌ **Tool error: {str(tool_error)}**\n\n"
                            else:
                                self.logger.warning(f"Unknown tool: {current_function_name}")
                                yield f"❌ **Unknown tool: {current_function_name}**\n\n"

                            incomplete_tool_call = None
                            current_function_name = None
                        except json.JSONDecodeError:
                            # Still accumulating arguments
                            pass

                # Handle content
                elif "content" in chunk and chunk["content"] is not None:
                    content_piece = chunk["content"]
                    assistant_message["content"] += content_piece
                    yield content_piece

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


# Export main classes and functions
__all__ = [
    'ExplainerAgent',
    'create_explainer_agent', 
    'explain_metric_change',
    'EXPLAINER_INSTRUCTIONS'
] 
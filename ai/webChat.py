import os
import json
import qdrant_client
from openai import OpenAI
import openai
from swarm import Swarm, Agent
from tools.anomaly_detection import anomaly_detection
import pandas as pd
from dotenv import load_dotenv
import logging
from tools.data_fetcher import set_dataset
from tools.vector_query import query_docs  #
from tools.genChart import generate_time_series_chart
from tools.retirementdata import read_csv_with_encoding

from tools.generate_map import generate_map, get_map_by_id, get_recent_maps
from tools.gen_map_dw import create_datawrapper_map
from tools.notes_manager import get_notes, load_and_combine_notes, initialize_notes
from tools.dashboard_metric_tool import get_dashboard_metric
from pathlib import Path
# Import FastAPI and related modules
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid
import datetime
import sys
import asyncio
import math
import psycopg2
import psycopg2.extras
import json
from datetime import datetime, date, time as dt_time
import time as time_module
from decimal import Decimal
import traceback
# Add import for store_anomalies function
from tools.store_anomalies import get_anomaly_details as get_anomaly_details_from_db, get_anomalies
# Import the generate_chart_message function
from chart_message import generate_chart_message, generate_anomaly_chart_html
import re

# ------------------------------
# Configuration and Setup
# ------------------------------

load_dotenv()
# Confirm that there is an openai_api_key set
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize APIRouter
router = APIRouter()

# Serve static files and templates
# Create static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")
router.mount("/static", StaticFiles(directory="static"), name="static")
# Mount the static directory
templates = Jinja2Templates(directory="templates")

# Configure logging with more detailed format
# Get absolute path for logs directory
current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(current_dir, 'logs')
log_file = os.path.join(logs_dir, 'webchat.log')

# Create logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Test write access and log startup
try:
    with open(log_file, 'a', encoding='utf-8') as f:
        startup_message = f"\n{'='*50}\nLog started at {datetime.now()}\nLog file: {log_file}\nPython path: {sys.executable}\n{'='*50}\n"
        f.write(startup_message)
except Exception as e:
    print(f"Error writing to log file: {e}")
    raise

# Configure our module's logger
logger = logging.getLogger(__name__)

# Test the logger
logger.info(f"WebChat logging initialized at {datetime.now()}")
logger.info(f"Log file location: {log_file}")

# Initialize connections
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize OpenAI client for direct API calls
openai.api_key = openai_api_key

# Initialize the Swarm client
swarm_client = Swarm()
client = OpenAI()

qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)

# Set embedding model
EMBEDDING_MODEL = "text-embedding-3-large"
# AGENT_MODEL = "gpt-3.5-turbo-16k"
# AGENT_MODEL = "gpt-3.5-turbo-16k"
AGENT_MODEL = "gpt-5"

# Set Qdrant collection
collection_name = "SFPublicData"

# Session management (simple in-memory store)
sessions = {}
# Load and combine the data
data_folder = './data'  # Replace with the actual path

# Initialize context_variables with an empty DataFrame
combined_df = {"dataset": pd.DataFrame()}

# Add environment check at the top with other configurations
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Load the combined notes using the new tool
combined_notes = initialize_notes()

def save_notes_to_file(notes_text, filename="combined_notes.txt"):
    """
    Saves the combined notes to a file in the output/notes directory.
    """
    logger = logging.getLogger(__name__)
    
    script_dir = Path(__file__).parent
    notes_dir = script_dir / 'output' / 'notes'
    
    # Create notes directory if it doesn't exist
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = notes_dir / filename
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(notes_text)
        logger.info(f"Successfully saved notes to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving notes to file: {e}")
        return False

def load_and_combine_climate_data():
    data_folder = 'data/climate'
    vera_file = os.path.join(data_folder, 'vcus-2025.csv')
    
    # Load the CSV file with detected encoding
    try:
        vera_df = pd.read_csv(vera_file)
        logger = logging.getLogger(__name__)
        logger.info(f"Successfully loaded climate data from {vera_file}")
        logger.info(f"DataFrame shape: {vera_df.shape}")
        return vera_df
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading climate data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Load the climate data
combined_df = {"dataset": pd.DataFrame()}

context_variables = {
    "dataset": combined_df,  # Store the actual DataFrame here
    "notes": combined_notes  # Store the notes string here
}

# Define the maximum number of messages to keep in context
MAX_HISTORY = 10
SUMMARY_INTERVAL = 10

def get_dataset(context_variables, *args, **kwargs):
    """
    Returns the dataset for analysis.
    """
    dataset = context_variables.get("dataset")
    if dataset is not None and not dataset.empty:
        return dataset
    else:
        return {'error': 'Dataset is not available or is empty.'}

def get_notes(context_variables, *args, **kwargs):
    """
    Returns the notes from context variables, with length checking and logging.
    """
    logger = logging.getLogger(__name__)
    
    try:
        notes = context_variables.get("notes", "").strip()
        total_length = len(notes)
        
        logger.info(f"""
=== get_notes called ===
Total length: {total_length} characters
Approximate tokens: {len(notes.split())}
First 100 chars: {notes[:100]}
Number of lines: {len(notes.splitlines())}
""")
        
        # If notes exceed OpenAI's limit, truncate them
        if total_length > MAX_MESSAGE_LENGTH:
            logger.warning(f"""
Notes exceed maximum length:
Current length: {total_length}
Maximum allowed: {MAX_MESSAGE_LENGTH}
Difference: {total_length - MAX_MESSAGE_LENGTH}
""")
            # Keep the first part and last part with a message in between
            keep_length = (MAX_MESSAGE_LENGTH // 2) - 100  # Leave room for the truncation message
            truncation_message = "\n\n[CONTENT TRUNCATED DUE TO LENGTH]\n\n"
            notes = notes[:keep_length] + truncation_message + notes[-keep_length:]
            logger.info(f"""
Notes truncated:
New length: {len(notes)}
Truncation point: {keep_length}
""")
        
        if notes:
            return {"notes": notes}
        else:
            logger.error("No notes found or notes are empty")
            return {"error": "No notes found or notes are empty"}
            
    except Exception as e:
        logger.error(f"Error in get_notes: {str(e)}")
        return {"error": f"Error processing notes: {str(e)}"}

def get_columns(context_variables, *args, **kwargs):

    """
    Returns the list of columns in the dataset.
    """
    dataset = context_variables.get("dataset")
    if dataset is not None and not dataset.empty:
        return {"columns": dataset.columns.tolist()}
    else:
        return {"error": "Dataset is not available or is empty."}

def set_columns(context_variabls, *args, **kwargs):
    """
    Sets the columns for the query.
    """
    columns = kwargs.get("columns")
    if columns:
        return {"columns": columns}
    else:
        return {"error": "No columns provided."}

def get_data_summary(context_variables,*args, **kwargs):
    """
    Returns a statistical summary of the dataset.
    """
    dataset = context_variables.get("dataset")
    if dataset is not None:
        summary = dataset.describe(include='all').to_dict()
        return {"summary": summary}
    else:
        return {"error": "Dataset is not available."}     

def transfer_to_analyst_agent(context_variables, *args, **kwargs):
    """
    Transfers the conversation to the Anomaly Finder Agent.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Transferring to Analyst Agent. Context variables: {list(context_variables.keys())}")
    
    # Create a new instance of the analyst agent with the current context variables
    new_analyst_agent = Agent(
        model=AGENT_MODEL,
        name="Analyst",
        instructions="""
        **Function Usage:**

        - Use `query_docs(context_variables, "SFPublicData", query)` to search for datasets. The `query` parameter is a string describing the data the user is interested in. always pass the context_variables and the collection name is allways "SFPublicData"
        - Use the `transfer_to_researcher_agent` function (without any parameters) to transfer to the researcher agent. 
        - Use `set_dataset(context_variables, endpoint="endpoint-id", query="your-soql-query")` to set the dataset. Both parameters are required:
            - endpoint: The dataset identifier WITHOUT the .json extension (e.g., 'ubvf-ztfx').  If you dont't have that, get it from get_dashboard_metric()
            - query: The complete SoQL query string using standard SQL syntax
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
            
        - Use `get_dataset(context_variables)` to retrieve the current dataset stored in context_variables:
            - This function takes only the context_variables parameter
            - Returns the pandas DataFrame if available, or an error dictionary if the dataset is unavailable or empty
            - Use this to check if a dataset is loaded or to perform custom analysis on the raw data
            - Example usage: `dataset = get_dataset(context_variables)`
            
        - Use `generate_time_series_chart(context_variables, column_name, start_date, end_date, aggregation_period, return_html=False)` to generate a time series chart. 
        - Use `get_dashboard_metric(context_variables, district_number, metric_id)` to retrieve dashboard metric data:
            - district_number: Integer from 0 (citywide) to 11 (specific district)
            - metric_id: Optional. The specific metric ID to retrieve (e.g., '🚨_violent_crime_incidents_ytd'). If not provided, returns the top-level district summary. Sometimes this will be passed in as a metric_id number, for that pass it as an integer..
        
        """,
        functions=[query_docs, set_dataset, get_dataset, set_columns, get_data_summary, anomaly_detection, generate_time_series_chart, get_dashboard_metric, transfer_to_researcher_agent],
        context_variables=context_variables,
        debug=True,
    )
    
    return new_analyst_agent

def transfer_to_researcher_agent(context_variables, *args, **kwargs):
    """
    Transfers the conversation to the Data Agent.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Transferring to Researcher Agent. Context variables: {list(context_variables.keys())}")
    
    # Create a new instance of the researcher agent with the current context variables
    new_researcher_agent = Agent(
        model=AGENT_MODEL,
        name="Researcher",
        instructions=Researcher_agent.instructions,
        functions=Researcher_agent.functions,
        context_variables=context_variables,
        debug=True
    )
    
    return new_researcher_agent

def get_dashboard_metric(context_variables, district_number=0, metric_id=None):
    """
    Retrieves dashboard metric data for a specific district and metric.
    
    Args:
        context_variables: The context variables dictionary
        district_number: The district number (0 for citywide, 1-11 for specific districts). Can be int or string.
        metric_id: The specific metric ID to retrieve. If None, returns the top_level.json file.
    
    Returns:
        A dictionary containing the metric data or error message
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Convert district_number to int if it's a string
        try:
            district_number = int(district_number)
        except (TypeError, ValueError):
            return {"error": f"Invalid district number format: {district_number}. Must be a number between 0 and 11."}
        
        # Validate district number range
        if district_number < 0 or district_number > 11:
            return {"error": f"Invalid district number: {district_number}. Must be between 0 (citywide) and 11."}
        
        # Construct the base path - looking one level up from the script
        script_dir = Path(__file__).parent
        dashboard_dir = script_dir / 'output' / 'dashboard'
        
        logger.info(f"Looking for dashboard data in: {dashboard_dir}")
        
        # If metric_id is None, return the top-level district summary
        if metric_id is None:
            file_path = dashboard_dir / f"district_{district_number}.json"
            logger.info(f"Fetching top-level dashboard data for district {district_number} from {file_path}")
        else:
            # Metric ID is provided, look in the district-specific folder
            district_folder = dashboard_dir / str(district_number)
            
            # If metric_id doesn't end with .json, add it
            if not metric_id.endswith('.json'):
                metric_id = f"{metric_id}.json"
                
            file_path = district_folder / metric_id
            logger.info(f"Fetching specific metric '{metric_id}' for district {district_number} from {file_path}")
        
        # Check if the file exists
        if not file_path.exists():
            # If specific metric file doesn't exist, list available metrics
            if metric_id is not None:
                available_metrics = []
                district_folder = dashboard_dir / str(district_number)
                if district_folder.exists():
                    available_metrics = [f.name for f in district_folder.glob('*.json')]
                
                return {
                    "error": f"Metric '{metric_id}' not found for district {district_number}",
                    "available_metrics": available_metrics
                }
            else:
                return {"error": f"Dashboard data not found for district {district_number}"}
        
        # Read and parse the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add metadata about the source
        result = {
            "source": str(file_path),
            "district": district_number,
            "metric_id": metric_id,
            "data": data
        }
        
        # Add endpoint from metric data if available
        if isinstance(data, dict) and "endpoint" in data:
            result["endpoint"] = data["endpoint"]
            logger.info(f"Found endpoint in metric data: {data['endpoint']}")
        
        # NEW CODE: Try to find and add corresponding analysis files
        analysis_content = {}
        total_analysis_length = 0
        max_tokens = 100000  # Approximate max tokens we want to allow
        
        # Get metric ID for analysis files
        metric_id_number = None
        
        # Clean the metric_id to use as base filename
        if metric_id:
            base_metric_name = metric_id.replace('.json', '')
            
            # Try to extract the ID number from the data if we have a named metric
            if isinstance(data, dict) and "id" in data:
                # If the metric has an ID in its data, use that for analysis files
                metric_id_number = str(data["id"])
                logger.info(f"Found metric ID number {metric_id_number} from data")
            elif base_metric_name.isdigit():
                # If the metric_id itself is a number, use it directly
                metric_id_number = base_metric_name
                logger.info(f"Using metric_id as a number: {metric_id_number}")
            else:
                # Try to find the ID from the dashboard_queries_enhanced.json file
                queries_file = script_dir / 'data' / 'dashboard' / 'dashboard_queries_enhanced.json'
                if queries_file.exists():
                    try:
                        with open(queries_file, 'r', encoding='utf-8') as f:
                            queries_data = json.load(f)
                        
                        # Look for the metric name in the queries data
                        for category in queries_data.values():
                            for subcategory in category.values():
                                if "queries" in subcategory:
                                    for query_name, query_data in subcategory["queries"].items():
                                        # Check if the name matches our metric name (with or without emoji)
                                        item_name = query_name.lower()
                                        clean_metric_name = base_metric_name.lower()
                                        
                                        # Remove emojis and special characters for comparison
                                        import re
                                        clean_item_name = re.sub(r'[^\w\s]', '', item_name).strip()
                                        clean_base_name = re.sub(r'[^\w\s]', '', clean_metric_name).strip()
                                        
                                        if clean_item_name == clean_base_name or item_name == clean_metric_name:
                                            metric_id_number = str(query_data["id"])
                                            # Add query information to result
                                            result["endpoint"] = query_data.get("endpoint")
                                            result["ytd_query"] = query_data.get("ytd_query")
                                            result["metric_query"] = query_data.get("metric_query")
                                            logger.info(f"Found metric ID {metric_id_number} from dashboard_queries_enhanced.json")
                                            break
                        
                        if not metric_id_number:
                            logger.info(f"Could not find metric ID for '{base_metric_name}' in dashboard_queries_enhanced.json")
                    except Exception as e:
                        logger.error(f"Error reading dashboard_queries_enhanced.json: {str(e)}")
                else:
                    logger.info(f"dashboard_queries_enhanced.json not found at {queries_file}")
                
                if not metric_id_number:
                    logger.info(f"Could not determine metric ID number from {base_metric_name}")
            
            # Look for analysis files in monthly and annual folders
            monthly_dir = script_dir / 'output' / 'monthly'
            annual_dir = script_dir / 'output' / 'annual'
            
            # Only proceed if we have a metric ID number
            if metric_id_number:
                # Paths for analysis files using the ID number
                monthly_analysis_path = monthly_dir / f"{district_number}/{metric_id_number}.md"
                annual_analysis_path = annual_dir / f"{district_number}/{metric_id_number}.md"
                
                logger.info(f"Looking for monthly analysis at: {monthly_analysis_path}")
                logger.info(f"Looking for annual analysis at: {annual_analysis_path}")
                
                # Read monthly analysis if it exists
                if monthly_analysis_path.exists():
                    try:
                        with open(monthly_analysis_path, 'r', encoding='utf-8') as f:
                            monthly_content = f.read()
                            total_analysis_length += len(monthly_content.split())
                            analysis_content["monthly_analysis"] = monthly_content
                            logger.info(f"Found monthly analysis file ({len(monthly_content)} chars)")
                    except Exception as e:
                        logger.error(f"Error reading monthly analysis: {str(e)}")
                
                # Read annual analysis if it exists
                if annual_analysis_path.exists():
                    try:
                        with open(annual_analysis_path, 'r', encoding='utf-8') as f:
                            annual_content = f.read()
                            total_analysis_length += len(annual_content.split())
                            analysis_content["annual_analysis"] = annual_content
                            logger.info(f"Found annual analysis file ({len(annual_content)} chars)")
                    except Exception as e:
                        logger.error(f"Error reading annual analysis: {str(e)}")
            else:
                logger.info("No metric ID number available for finding analysis files")
            
            # Check if we need to summarize (approximating tokens as words/0.75)
            estimated_tokens = total_analysis_length / 0.75
            if estimated_tokens > max_tokens and analysis_content:
                logger.info(f"Analysis content too large (~{estimated_tokens:.0f} tokens). Summarizing...")
                
                # Create a simple summary by truncating and adding a note
                for key in analysis_content:
                    original_length = len(analysis_content[key])
                    # Calculate how much to keep (proportional to original length)
                    proportion = len(analysis_content[key]) / total_analysis_length
                    max_chars = int((max_tokens * 0.75) * proportion * 4)  # Rough char estimate
                    
                    if len(analysis_content[key]) > max_chars:
                        analysis_content[key] = (
                            f"{analysis_content[key][:max_chars]}\n\n"
                            f"[Note: Analysis truncated due to length. Original size: {original_length} characters]"
                        )
                        logger.info(f"Truncated {key} from {original_length} to {len(analysis_content[key])} chars")
            
            # Add analysis content to result if any was found
            if analysis_content:
                result["analysis"] = analysis_content
                logger.info(f"Added analysis content with keys: {list(analysis_content.keys())}")
        
        logger.info(f"Successfully retrieved dashboard metric data from {file_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard metric: {str(e)}", exc_info=True)
        return {"error": f"Error retrieving dashboard metric: {str(e)}"}


def get_dataset_columns(context_variables, endpoint=None):
    """
    Tool to get the columns available in a particular dataset endpoint.
    
    Args:
        context_variables: The context variables dictionary
        endpoint: The dataset identifier (e.g., 'ubvf-ztfx')
        
    Returns:
        Dictionary with columns information or error
    """
    try:
        logger.info(f"""
=== get_dataset_columns called ===
Endpoint: {endpoint}
Context variables keys: {list(context_variables.keys())}
""")
        
        if not endpoint:
            logger.error("No endpoint provided")
            return {"error": "No endpoint provided"}
        
        # Clean the endpoint parameter (remove .json if present)
        endpoint = endpoint.replace('.json', '')
        logger.info(f"Cleaned endpoint: {endpoint}")
        
        # Import database utilities
        from tools.db_utils import get_postgres_connection
        import psycopg2.extras
        
        # Get database connection
        connection = get_postgres_connection()
        if not connection:
            logger.error("Database connection failed")
            return {"error": "Database connection failed"}
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query the datasets table for this endpoint
        cursor.execute("""
            SELECT columns, title, description, category
            FROM datasets 
            WHERE endpoint = %s AND is_active = true
        """, (endpoint,))
        
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not row:
            logger.error(f"No dataset found in database for endpoint: {endpoint}")
            return {"error": f"No dataset found for endpoint: {endpoint}"}
        
        if not row['columns']:
            logger.error(f"No columns found in database for endpoint: {endpoint}")
            return {"error": f"No columns found for endpoint: {endpoint}"}
        
        # Extract columns information from the database
        columns = row['columns']
        logger.info(f"Found {len(columns)} columns in database for endpoint {endpoint}")
        
        # Return simplified column info for agent use
        column_info = []
        for col in columns:
            if isinstance(col, dict):
                info = {
                    "name": col.get("fieldName", col.get("name", "unknown")),
                    "type": col.get("dataTypeName", col.get("type", "unknown")),
                    "description": col.get("description", "")
                }
                column_info.append(info)
            else:
                logger.warning(f"Unexpected column type: {type(col)}")
        
        logger.info(f"""
=== Column Information Summary ===
Total columns found: {len(column_info)}
First few columns: {column_info[:3] if column_info else 'None'}
""")
        
        result = {
            "endpoint": endpoint,
            "column_count": len(column_info),
            "columns": column_info,
            "dataset_title": row.get('title', ''),
            "dataset_description": row.get('description', ''),
            "dataset_category": row.get('category', '')
        }
        
        logger.info(f"""
=== Final Result ===
Endpoint: {result['endpoint']}
Column count: {result['column_count']}
Columns: {[col['name'] for col in result['columns']][:5]}... (showing first 5)
Dataset title: {result['dataset_title']}
""")
        
        return result
        
    except Exception as e:
        logger.error(f"""
=== Error in get_dataset_columns ===
Error type: {type(e).__name__}
Error message: {str(e)}
Stack trace:
{traceback.format_exc()}
""")
        return {"error": f"Failed to get dataset columns: {str(e)}"}

def query_anomalies_db(context_variables, query_type='recent', limit=10, group_filter=None, date_start=None, date_end=None, only_anomalies=True, metric_name=None, district_filter=None, metric_id=None, period_type=None):
    """
    Query the PostgreSQL database for anomalies based on the specified parameters.
    
    Args:
        context_variables: Context variables from the chatbot
        query_type: Type of query to execute (recent, top, bottom, by_group, by_metric, by_metric_id)
        limit: Maximum number of results to return
        group_filter: Filter by specific group value
        date_start: Start date for filtering
        date_end: End date for filtering
        only_anomalies: If True, only return results where out_of_bounds is True
        metric_name: Filter by specific field_name (metric name)
        district_filter: Filter by specific district
        metric_id: Filter by object_id (metric ID)
        period_type: Filter by period type (year, month, week, day)
        
    Returns:
        Dictionary with query results and metadata
    """
    logger = logging.getLogger(__name__)
    logger.info(f"""
=== query_anomalies_db called ===
Parameters:
- query_type: {query_type}
- limit: {limit}
- group_filter: {group_filter}
- date_start: {date_start}
- date_end: {date_end}
- only_anomalies: {only_anomalies}
- metric_name: {metric_name}
- district_filter: {district_filter}
- metric_id: {metric_id}
- period_type: {period_type}
""")
    
    try:
        # Get database connection parameters from environment variables
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_name = os.getenv("POSTGRES_DB", "transparentsf")
        
        logger.info(f"Using database connection: {db_host}:{db_port}/{db_name} (user: {db_user})")
        
        # Use the get_anomalies function from store_anomalies.py
        # First, we need to convert query_type from our naming convention to store_anomalies' naming
        sa_query_type = query_type
        
        # Special handling for by_metric_id query type
        if query_type == 'by_metric_id' and metric_id:
            # Keep the query_type as is, since we're filtering by metric_id
            logger.info(f"Using query_type 'by_metric_id' with metric_id={metric_id}")
        elif query_type == 'top':
            sa_query_type = 'by_anomaly_severity'  # Most significant anomalies first
            logger.info(f"Converting query_type 'top' to 'by_anomaly_severity'")
        elif query_type == 'bottom':
            sa_query_type = 'by_anomaly_severity'  # We'll manually reverse the order after
            logger.info(f"Converting query_type 'bottom' to 'by_anomaly_severity'")
        elif query_type in ['by_metric', 'by_metric_id']:
            sa_query_type = 'recent'  # Default to recent for these types, we'll filter by metric
            logger.info(f"Converting query_type '{query_type}' to 'recent'")
        
        logger.info(f"Calling get_anomalies with query_type={sa_query_type}, limit={limit}")
        logger.info(f"Additional filters: group_filter={group_filter}, date_start={date_start}, date_end={date_end}, only_anomalies={only_anomalies}, metric_name={metric_name}, district_filter={district_filter}, metric_id={metric_id}, period_type={period_type}")
        
        start_time = time_module.time()
        result = get_anomalies(
            query_type=sa_query_type,
            limit=limit,
            group_filter=group_filter,
            date_start=date_start,
            date_end=date_end,
            only_anomalies=only_anomalies,
            metric_name=metric_name,
            district_filter=district_filter,
            metric_id=metric_id,
            period_type=period_type,
            db_host=db_host,
            db_port=int(db_port),
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )
        query_time = time_module.time() - start_time
        logger.info(f"get_anomalies query completed in {query_time:.2f} seconds")
        
        if result["status"] == "error":
            logger.error(f"Error from get_anomalies: {result['message']}")
            return {
                'error': result["message"],
                'results': [],
                'count': 0,
                'query_info': {
                    'type': query_type,
                    'limit': limit
                }
            }
        
        logger.info(f"get_anomalies returned {len(result.get('results', []))} results")
        
        # Process results
        processed_results = []
        start_time = time_module.time()
        
        for row in result["results"]:
            # Convert to regular dictionary (should already be a dict)
            row_dict = dict(row)
            
            # Extract metadata information
            metadata = row_dict.get("metadata", {})
            
            # Add missing fields from metadata
            row_dict["field_name"] = metadata.get("numeric_field", row_dict.get("field_name", ""))
            row_dict["object_id"] = metadata.get("object_id", row_dict.get("object_id", ""))
            row_dict["object_name"] = metadata.get("object_name", "")
            row_dict["period_type"] = metadata.get("period_type", row_dict.get("period_type", ""))
            
            # Calculate percent change
            if row_dict.get('comparison_mean') and row_dict.get('comparison_mean') != 0:
                percent_change = ((row_dict.get('recent_mean', 0) - row_dict.get('comparison_mean', 0)) / 
                                 abs(row_dict.get('comparison_mean', 0)) * 100)
                row_dict['percent_change'] = round(percent_change, 2)
            else:
                row_dict['percent_change'] = None
            
            processed_results.append(row_dict)
        
        processing_time = time_module.time() - start_time
        logger.info(f"Processed {len(processed_results)} results in {processing_time:.2f} seconds")
        
        # Handle reverse sorting for 'bottom' query type
        if query_type == 'bottom':
            logger.info("Sorting results by difference (ascending) for 'bottom' query type")
            processed_results = sorted(processed_results, key=lambda x: x.get('difference', 0))
        
        # Build response
        response = {
            'results': processed_results,
            'count': len(processed_results),
            'query_info': {
                'type': query_type,
                'limit': limit,
                'filters_applied': {
                    'group_filter': group_filter,
                    'date_start': date_start,  # Don't call isoformat() on potentially string values
                    'date_end': date_end,      # Don't call isoformat() on potentially string values
                    'only_anomalies': only_anomalies,
                    'metric_name': metric_name,
                    'district_filter': district_filter,
                    'metric_id': metric_id,
                    'period_type': period_type
                }
            }
        }
        
        logger.info(f"Returning {len(processed_results)} results for query_type={query_type}")
        return response
    except Exception as e:
        logger.error(f"Error in query_anomalies_db: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'error': str(e),
            'results': [],
            'count': 0,
            'query_info': {
                'type': query_type,
                'limit': limit
            }
        }

def get_anomaly_details(context_variables, anomaly_id):
    """
    Tool to retrieve detailed information about a specific anomaly by ID.
    
    Args:
        context_variables: The context variables dictionary
        anomaly_id: The ID of the anomaly to retrieve
        
    Returns:
        Dictionary with detailed anomaly information
    """
    try:
        # Use the imported function from store_anomalies.py
        result = get_anomaly_details_from_db(anomaly_id=anomaly_id)
        
        # If successful, format the result
        if result["status"] == "success":
            item = result["anomaly"]
            
            # Extract metadata for easier access
            if 'metadata' in item and item['metadata']:
                metadata = item['metadata']
                item['metric_name'] = metadata.get('object_name', metadata.get('title', 'unknown'))
                item['metric_field'] = metadata.get('numeric_field', 'unknown')
                item['group_field'] = metadata.get('group_field', 'unknown')
                item['period_type'] = metadata.get('period_type', 'month')
                
                # Extract period information
                if 'recent_period' in metadata:
                    item['recent_period'] = metadata['recent_period']
                if 'comparison_period' in metadata:
                    item['comparison_period'] = metadata['comparison_period']
            
            # Generate an explanation summary based on the anomaly data
            explanation = f"Analysis of anomaly for '{item['group_value']}' detected on {item['created_at'].split('T')[0]}:\n"
            
            # Add district information if available
            if item.get('district') and item['district'] != 'None' and item['district'] != 'unknown':
                explanation += f"District: {item['district']}\n"
            
            if item.get('out_of_bounds'):
                direction = "increased" if item.get('difference', 0) > 0 else "decreased"
                percent_change = abs(item.get('difference', 0) / item.get('comparison_mean', 1) * 100) if item.get('comparison_mean') else 0
                explanation += f"The value {direction} by {percent_change:.1f}% from {item.get('comparison_mean', 0):.2f} to {item.get('recent_mean', 0):.2f}.\n"
                explanation += f"This change is {abs(item.get('difference', 0) / item.get('std_dev', 1)):.1f} standard deviations from the mean."
            else:
                explanation += "This data point is within normal range of expected values."
                
            # Add the explanation to the item
            item['explanation'] = explanation
            
            return {
                "status": "success",
                "anomaly": item
            }
        else:
            return result  # Return error as is
            
    except Exception as e:
        logger.error(f"Error retrieving anomaly details: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to retrieve anomaly details: {str(e)}"
        }


# Define the anomaly finder agent

analyst_agent = Agent(
    model=AGENT_MODEL,
    name="Analyst",
     instructions="""
    **Function Usage:**

    - Use `query_docs(context_variables, "SFPublicData", query)` to search for datasets. The `query` parameter is a string describing the data the user is interested in. always pass the context_variables and the collection name is allways "SFPublicData"
    - Use the `transfer_to_researcher_agent` function (without any parameters) to transfer to the researcher agent. 
    - Use `set_dataset(context_variables, endpoint="dataset-id", query="your-soql-query")` to set the dataset. Both parameters are required:
        - endpoint: The dataset identifier WITHOUT the .json extension (e.g., 'ubvf-ztfx')
        - query: The complete SoQL query string using standard SQL syntax
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
        
    - Use `get_dataset(context_variables)` to retrieve the current dataset stored in context_variables:
        - This function takes only the context_variables parameter
        - Returns the pandas DataFrame if available, or an error dictionary if the dataset is unavailable or empty
        - Use this to check if a dataset is loaded or to perform custom analysis on the raw data
        - Example usage: `dataset = get_dataset(context_variables)`
        
    - Use `generate_time_series_chart(context_variables, column_name, start_date, end_date, aggregation_period, return_html=False)` to generate a time series chart. 
    - Use `get_dashboard_metric(context_variables, district_number, metric_id)` to retrieve dashboard metric data:
        - district_number: Integer from 0 (citywide) to 11 (specific district)
        - metric_id: Optional. The specific metric ID to retrieve (e.g., '🚨_violent_crime_incidents_ytd'). If not provided, returns the top-level district summary. Sometimes this will be passed in as a metric_id number, for that pass it as an integer..
    - Use `generate_chart_message(context_variables, chart_data=data_object, chart_type="anomaly")` to create and display charts in your responses:
        - chart_data: A dictionary containing the data for the chart
        - chart_type: The type of chart (default: "anomaly")
        - Example: 
          anomaly_data = get_anomaly_details(context_variables, anomaly_id=123)
          chart_message = generate_chart_message(context_variables, chart_data=anomaly_data, chart_type="anomaly")
          return chart_message
    
    """,
    functions=[query_docs, set_dataset, get_dataset, set_columns, get_data_summary, anomaly_detection, generate_time_series_chart, get_dashboard_metric, transfer_to_researcher_agent, generate_chart_message],
    context_variables=context_variables,
    debug=True,
)

# Define the explainer agent directly like in webChat.py
ANOMALY_EXPLAINER_INSTRUCTIONS = """You are an anomaly explanation agent that specializes in providing deep insights into detected anomalies.

IMPORTANT: You MUST use tools to gather data BEFORE responding. Direct explanations without tool usage are NOT acceptable.

Your task is to:
1. Take an change that has already been identified in dashboard metrics
2. Research that change to explain what changed and where or what variables explain the change
3. Analyze anomalies in the dataset to see if they are related to the change
4. Provide clear, comprehensive explanations with supporting evidence.  You don't need to be breif, more is more, so be as complete and thorough as possible.
5. Return your findings in the form of a JSON object with the following keys:
    - "explanation": A string with your explanation
    - "charts": a list of charts placeholders, formatted ONLY as either [CHART:anomaly:anomaly_id] or [CHART:time_series:metric_id:district_id:period_type]
    - "trend_analysis" - Your discussion of the trend in the metric short, medium, and long term. 

MANDATORY WORKFLOW (follow this exact sequence):
1. FIRST, check your notes!
2. SECOND, Query the anomalies_db for this metric and period_type and group_filter and district_filter and limit 30 and only_anomalies=True to see whats happening in this metric in this period for this group in this district. 
3. THIRD, Get information about the metric from the get_dashboard_metric tool, there may be enough information there to thoroughly explain the change.
4. FOURTH, contextualize this change vs the historical data, you can use the data from get_dashboard_metric to do this. 
5. FIFTH, if an anomaly is explanatory, then be sure to include a link to the anomaly chart, like this: [CHART:anomaly:anomaly_id]
6. SIXTH, if you still don't have enough information to understand the data, then use set_dataset and get_dataset to get exactly what you need from DataSF.  You can use the queries that you see in the get_dashboard_metric tool data as a starting point, make sure to use the righ fieldNames with the right case.  Read more about htat in the set_dataset() tool. 
7. SEVENTH, if the data has a strong geographic component, create a map visualization to show spatial patterns using the generate_map function.

Best Practices for explaining certain categories: 
1. Housing - If the you are being asked to explain is in housing, then you should query for the actual properties that have new units, and include the address, and the units certified in your explanation.
set_dataset
Arguments: { "endpoint": "j67f-aayr", "query": "SELECT building_permit_application, building_address, date_issued, document_type, number_of_units_certified ORDER BY date_issued DESC LIMIT 10" }

2. If you are being asked to explain a change in business registrations or closures, then you should query for the actual businesses that have closed, and include the DBA name, and the date of closure in your explanation.
set_dataset
Arguments: { "endpoint": "g8m3-pdis", "query": "SELECT dba_name, location, full_business_address, dba_end_date, naic_code_description, supervisor_district WHERE administratively_closed is not null ORDER BY dba_end_date DESC LIMIT 10" }


IMPORTANT CHART GENERATION RULES:

You should include charts and graphs to help explain the change. For a time series chart, use this simplified format:

[CHART:time_series:metric_id:district_id:period_type]
For example: [CHART:time_series:1:0:year]

For an anomaly chart, use this simplified format:

[CHART:anomaly:anomaly_id]

For example: [CHART:anomaly:27338]

These simplified references will be automatically expanded with the full HTML when the report is generated.


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
  - location_data: List of data objects containing location and value information
     * For district maps: [{"district": "1", "value": 120}, {"district": "2", "value": 85}]
     * For point maps: [{"lat": 37.7749, "lon": -122.4194, "title": "City Hall", "description": "Description"}]
     * For address maps: [{"address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA", "title": "City Hall", "description": "Description"}]
     * For intersection maps: [{"intersection": "Market St and Castro St", "title": "Market & Castro", "description": "Description"}]
  - map_metadata: Optional dictionary with additional information about the map
  
  Best practices:
  - Use district maps for showing aggregate statistics by district
  - Use point or address maps for showing specific locations
  - Include descriptive titles and explanatory text for each location
  - When mapping points, include both a title and description for each point
  - For address maps, include "San Francisco, CA" in the address for better geocoding
  
  Example usage:
  ```
  # Creating a supervisor district map
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

- create_datawrapper_map: Generate an interactive Datawrapper visualization for a map
  USAGE: create_datawrapper_map(map_id="uuid-string")
  Creates and publishes a Datawrapper visualization for a previously generated map.
  Returns the public URL where the map can be viewed.

"""
# - generate_chart_message: Create a chart message to display to the user
#   USAGE: generate_chart_message(context_variables, chart_data=data_object, chart_type="anomaly")
#   Use this to create and display charts in your responses. The chart_data should be a dictionary containing the data for the chart.
#   For anomaly charts, you can use the data returned by get_anomaly_details.
#   Example: 
#   anomaly_data = get_anomaly_details(context_variables, anomaly_id=123)
#   chart_message = generate_chart_message(context_variables, chart_data=anomaly_data, chart_type="anomaly")
#   return chart_message

# Create the explainer agent directly (similar to how Researcher_agent is created in webChat.py)
anomaly_explainer_agent = Agent(
    model=AGENT_MODEL,
    name="Explainer",
    instructions=ANOMALY_EXPLAINER_INSTRUCTIONS,
    functions=[
        get_notes,
        get_dataset,
        set_dataset,
        query_docs,
        query_anomalies_db,
        get_dashboard_metric,
        get_anomaly_details,
        get_dataset_columns,
        generate_map,
        get_map_by_id,
        get_recent_maps,
        create_datawrapper_map,
    ],
    context_variables=context_variables,
    debug=True,
    logger=logging.getLogger('explainer_agent')  # Use our configured logger
)



def update_agent_instructions_with_columns(columns):
    """
    Updates the analyst agent's instructions to include the available columns.
    """
    column_list_str = ', '.join(columns)
    logger.info(f"Updating agent instructions with columns: {column_list_str}")  # Log before updating

    # Split the base instructions at the Dataset Awareness section if it exists
    base_instructions = analyst_agent.instructions.split("Dataset Awareness:", 1)[0]

    # Create the new instructions by combining base instructions with column information
    analyst_agent.instructions = base_instructions + f"""Dataset Awareness:
Available Columns in the Dataset:
You have access to the following columns: {column_list_str}. ANNOUNCE THESE TO THE USER.
"""

def load_and_combine_climate_data():
    data_folder = 'data/climate'
    vera_file = os.path.join(data_folder, 'cleaned_vcusNov19.csv')
    
    # Load the CSV files with detected encoding
    vera_df = read_csv_with_encoding(vera_file)
    
    # Combine the DataFrames
    set_dataset_in_context(context_variables, vera_df)

    return vera_df

def format_table(context_variables, title=None):
    """
    Formats data from context_variables into a markdown table with an optional title.
    Args:
        context_variables: The context variables containing the dataset
        title: Optional title for the table
    Returns:
        The formatted table as markdown
    """
    logger.info(f"""
=== format_table called ===
Title: {title}
Context variables keys: {list(context_variables.keys())}
""")
    
    try:
        data = context_variables.get("dataset")
        logger.info(f"""
Data retrieved from context:
Type: {type(data)}
Is None: {data is None}
""")
            
        if data is None:
            logger.error("No dataset found in context")
            return {"error": "No dataset found in context"}
            
        # If it's a dict with 'status' or 'error', return that directly
        if isinstance(data, dict):
            logger.info(f"Data is dict with keys: {list(data.keys())}")
            if 'error' in data:
                logger.error(f"Error in data: {data['error']}")
                return {"error": data['error']}
            if 'status' in data:
                logger.info(f"Status in data: {data['status']}")
                return {"status": data['status']}
        
        # Format as DataFrame
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            df = data
            logger.info(f"""
DataFrame details:
Shape: {df.shape}
Columns: {df.columns.tolist()}
""")
        else:
            logger.info(f"Converting data to DataFrame from type: {type(data)}")
            df = pd.DataFrame(data)

        # Get total row count
        total_rows = len(df)
        rows_per_page = 50
        
        logger.info(f"""
Pagination details:
Total rows: {total_rows}
Rows per page: {rows_per_page}
""")
        
        # Start with padding and separator
        markdown = "\n---\n"
        
        if title:
            markdown += f"### {title}\n\n"
        
        # Add dataset size info
        if total_rows > rows_per_page:
            markdown += f"*Showing first {rows_per_page} of {total_rows} rows*\n\n"
            # Take only first 50 rows for display
            display_df = df.head(rows_per_page)
        else:
            markdown += f"*Total rows: {total_rows}*\n\n"
            display_df = df
            
        # Convert DataFrame to markdown and add padding
        table_md = display_df.to_markdown(index=False)
        markdown += table_md + "\n"
        
        # Add pagination info if needed
        if total_rows > rows_per_page:
            remaining = total_rows - rows_per_page
            markdown += f"\n*{remaining:,} more rows not shown*\n"
        
        # Add bottom padding and separator
        markdown += "---\n"
        
        logger.info(f"""
Generated markdown:
Length: {len(markdown)}
Preview: {markdown[:500]}...
""")
        
        # Store pagination info in context for potential follow-up
        context_variables["table_pagination"] = {
            "total_rows": total_rows,
            "current_page": 1,
            "rows_per_page": rows_per_page,
            "total_pages": (total_rows + rows_per_page - 1) // rows_per_page
        }
        
        # Return the markdown directly - no more direct response bypass
        return {"status": "Table formatted successfully", "content": markdown}
        
    except Exception as e:
        logger.error(f"Error formatting table: {str(e)}", exc_info=True)
        return {"error": f"Error formatting table: {str(e)}"}

# New function to get dashboard metric data

# Add a new function to handle pagination
def format_table_page(context_variables, page_number, title=None):
    """
    Formats a specific page of the dataset.
    Args:
        context_variables: The context variables containing the dataset
        page_number: The page number to display (1-based)
        title: Optional title for the table
    Returns:
        Status message for the LLM
    """
    try:
        data = context_variables.get("dataset")
        if data is None:
            return {"error": "No dataset found in context"}
            
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)

        pagination = context_variables.get("table_pagination", {})
        if not pagination:
            return {"error": "No pagination info found. Please display the table first."}

        rows_per_page = pagination["rows_per_page"]
        total_rows = len(df)
        total_pages = (total_rows + rows_per_page - 1) // rows_per_page

        if not (1 <= page_number <= total_pages):
            return {"error": f"Invalid page number. Please specify a page between 1 and {total_pages}"}

        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        markdown = "\n---\n"
        
        if title:
            markdown += f"### {title}\n\n"
        
        markdown += f"*Page {page_number} of {total_pages} (rows {start_idx + 1}-{end_idx} of {total_rows})*\n\n"
            
        # Get the rows for this page
        display_df = df.iloc[start_idx:end_idx]
        table_md = display_df.to_markdown(index=False)
        markdown += table_md + "\n"
        
        # Add navigation info
        nav_info = []
        if page_number > 1:
            nav_info.append(f"Previous: Page {page_number - 1}")
        if page_number < total_pages:
            nav_info.append(f"Next: Page {page_number + 1}")
        
        if nav_info:
            markdown += f"\n*{' | '.join(nav_info)}*\n"
        
        markdown += "---\n"
        
        # Store the formatted table in context for direct streaming
        context_variables["last_formatted_table"] = markdown
        
        # Update pagination info
        context_variables["table_pagination"]["current_page"] = page_number
        
        return {"status": "Table page formatted successfully", "pagination": context_variables["table_pagination"]}
        
    except Exception as e:
        logger.error(f"Error formatting table page: {str(e)}")
        return {"error": f"Error formatting table page: {str(e)}"}

# Function mapping
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
    'create_datawrapper_map': create_datawrapper_map,
    'query_anomalies_db': query_anomalies_db,
    'get_anomaly_details': get_anomaly_details,
    'get_dataset_columns': get_dataset_columns,
}


Researcher_agent = Agent(
    model=AGENT_MODEL,
    name="Researcher",
    instructions="""
    Role: You are a researcher for Transparent SF, focusing on trends in city data.
    Purpose: help the user find objective data and specific details on their question. 
    
    - get_notes() ALWAYS Start here. This is a summary of all the analysis you have available to you in your docs. Use it to determine what data is available, and what to search for in your query_docs() calls.  It contains no links or charts, so don't share any links or charts with the user without checking your docs first. 
    
    - Use query_docs(context_variables, collection_name, query) to search for datasets. The collection_name should be "SFPublicData" when searching for city datasets. The query should be descriptive of what you're looking for. When you get results, pay attention to the field names in the "Columns" section - these are the exact field names you'll need to use in any subsequent queries. Field names are always lowercase with underscores (e.g., "city_elective_office" not "City Elective Office").
    
    
    - Use `get_dashboard_metric(district_number, metric_id)` to retrieve dashboard metric data:
        - district_number: Integer from 0 (citywide) to 11 (specific district)
        - metric_id: Optional. The specific metric ID to retrieve (e.g., '4' or '🚨_violent_crime_incidents_ytd'). If not provided, returns the top-level district summary.
        Example usage:
        get_dashboard_metric(0, 1) # Get citywide police incident data
    - Use `transfer_to_analyst_agent()` to transfer the conversation to the analyst agent only if asked. 
    
    If you are ever asked to "Evaluate the recent monthly and annual trends" for the a dashboard metric, always use get_dashboard_metric() to get the data you need.
    
    Review the data with an eye for explaining recent changes. 
    Are there any recent anomalies that might help illuminate current trends?
    Is the change happening in a paricular area of the city?

    Summarize this down to a punchy markdown story with with supporting charts or tables.  
    Do use good markdown text to format and tell the story. 
    If you are referring to a field in the data, or a value show it in italics
    Do not re-state the data, but summarize it in a easy to understad way.
    Ground the story in historical (annual) context. 

    
    Always ground recent changes in long term trends. 
    Don't draw conclusions, just report on the data.
    Don't speculate as to causes or "WHY" something is happening.  Just report on WHAT is happening. 
    Never add or change any markdown links or HTML URLS that you get from your tool calls in your response.  If there are relative Links or URLS beginning with "/", just leave them as is. 
    
    When displaying data:
    1. Whenever possible, use charts and graphs from your docs to illustrate your findings.  To better find charts add the term charts to your query.
    2. Return them in the same markdown format you find in the docs, with no changes. DO NOT ADD or CHANGE THE URLS
    3. Include relevant titles and context with your tables and charts. 
    4. Follow up tables with explanations of key insights or trends. 
    5. If you can't find the data you need, just say so.  Don't make up data or information. 
    6. Don't speculate as to causes or "WHY" something is happening.  Just report on WHAT is happening. 
    
    """,
    functions=[get_notes, get_dashboard_metric, transfer_to_analyst_agent],
    context_variables=context_variables,
    debug=True
)
def set_dataset_in_context(context_variables, dataset):
    """
    Sets the dataset in the context variables and updates the agent's instructions with the column names.
    """
    context_variables["dataset"] = dataset
    
    # Retrieve and update column names in the agent's instructions
    columns = get_columns(context_variables).get("columns", [])
    if columns:
        update_agent_instructions_with_columns(columns)

def load_data(context_variables):
    """
    Loads and combines data, setting the dataset and updating the agent instructions with column names.
    """
    combined_data=combined_df["dataset"]
    # combined_data = load_and_combine_climate_data()
    set_dataset_in_context(context_variables, combined_data)
    return combined_data

# Ensure the dataset is set in the context variables when initializing
combined_df = {"dataset": load_data(context_variables)}

def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""
    MAX_BUFFER_SIZE = 1024 * 1024  # 1MB limit

    for chunk in response:
        # Reset buffer if it gets too large
        if len(content) > MAX_BUFFER_SIZE:
            content = ""
            logger.warning("Content buffer exceeded maximum size, clearing buffer")

        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

                function_to_call = function_mapping.get(name)
                if function_to_call:
                    try:
                        result = function_to_call(**json.loads(f["arguments"]))
                        
                        # Add logging for chart generation results
                        if name == "generate_time_series_chart":
                            logger.debug(f"Chart generation result type: {type(result)}")
                            if isinstance(result, tuple):
                                markdown_content, _ = result  # Ignore the HTML content
                                logger.info(f"Chart generated successfully. Markdown length: {len(markdown_content)}")
                                
                                # Only store and send the markdown content
                                content = ""  # Clear the buffer after processing
                                return markdown_content  # Return just the markdown portion
                            else:
                                logger.debug(f"Unexpected result format: {result}")
                        
                        content = ""  # Clear the buffer after processing
                        return result
                    except Exception as e:
                        logger.error(f"Error processing tool call {name}: {e}")
                        content = ""  # Clear the buffer on error
                        return None

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""  # Clear the buffer

        if "response" in chunk:
            content = ""  # Clear the buffer
            return chunk["response"]

    # Clear any remaining content at the end
    content = ""
    return None

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


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
    'create_datawrapper_map': create_datawrapper_map,
    'query_anomalies_db': query_anomalies_db,
    'get_anomaly_details': get_anomaly_details,
    'get_dataset_columns': get_dataset_columns,
}


# ------------------------------------
# Web Routes and Session Management
# ------------------------------------

def get_session(session_id: str = Cookie(None)):
    if session_id and session_id in sessions:
        return sessions[session_id]
    else:
        # Initialize new session
        new_session_id = str(uuid.uuid4())
        sessions[new_session_id] = {
            "messages": [],
            "agent": Researcher_agent,  # Start with researcher
            "context_variables": {"dataset": combined_df["dataset"],"notes": combined_notes}  # Initialize context_variables
        }
        return sessions[new_session_id]

def summarize_conversation(messages):
    """
    Generates a summary of the conversation.
    You can implement this using a separate OpenAI call or a simple heuristic.
    """
    # Example: Simple concatenation (for demonstration; consider using a model for better summaries)
    summary = "Conversation Summary:\n"
    for msg in messages:
        summary += f"{msg['role']}: {msg['content']}\n"
    return summary

# Define maximum message length (OpenAI's limit)
MAX_MESSAGE_LENGTH = 1048576  # 1MB in characters
MAX_SINGLE_MESSAGE = 100000   # Maximum length for any single message

def truncate_messages(messages):
    """
    Truncate messages to stay within OpenAI's context limits.
    More aggressive truncation that preserves the most recent context.
    """
    # Set stricter limits to leave room for functions and responses
    MAX_SINGLE_MESSAGE = 24000  # ~6k tokens
    MAX_TOTAL_LENGTH = 100000   # ~25k tokens, leaving plenty of room for functions and responses
    
    logger.info(f"""
=== Message Truncation Started ===
Initial message count: {len(messages)}
""")
    
    # Always keep the last message (user input)
    if not messages:
        return messages
        
    last_message = messages[-1]
    truncated_messages = [last_message]
    
    # First pass: Truncate any individual messages that are too long
    for msg in messages[:-1]:  # Skip the last message as we're keeping it complete
        content = msg.get("content", "")
        if len(content) > MAX_SINGLE_MESSAGE:
            msg["content"] = f"[TRUNCATED]...{content[-MAX_SINGLE_MESSAGE//2:]}"
            logger.info(f"""
Truncated large message:
Original length: {len(content)}
New length: {len(msg['content'])}
""")
    
    # Calculate how much space we have left
    current_length = len(str(last_message.get("content", "")))
    
    # Add messages from most recent to oldest until we approach the limit
    for msg in reversed(messages[:-1]):
        msg_length = len(str(msg.get("content", "")))
        if current_length + msg_length > MAX_TOTAL_LENGTH:
            break
        current_length += msg_length
        truncated_messages.insert(0, msg)
    
    # If we truncated any messages, add a summary message
    if len(truncated_messages) < len(messages):
        removed_count = len(messages) - len(truncated_messages)
        summary = {
            "role": "system",
            "content": f"[Note: {removed_count} earlier messages were removed to stay within context limits]"
        }
        truncated_messages.insert(0, summary)
    
    logger.info(f"""
=== Message Truncation Complete ===
Original messages: {len(messages)}
Truncated messages: {len(truncated_messages)}
Estimated total length: {current_length:,} characters
""")
    
    return truncated_messages

@router.post("/api/chat")
async def chat(request: Request, session_id: str = Cookie(None)):
    logger.info("Chat endpoint called")
    try:
        data = await request.json()
        user_input = data.get("query")
        agent_name = data.get("agent", "researcher")  # Default to researcher if not specified
        
        # Get session ID from request body if not in cookies
        body_session_id = data.get("session_id")
        if body_session_id:
            session_id = body_session_id
        
        logger.info(f"""
=== New User Message ===
Session ID: {session_id if session_id else 'New Session'}
Timestamp: {datetime.now().isoformat()}
Message:
{user_input}
Agent: {agent_name}
===============================
""")

        # Get or create session data
        if session_id is None or session_id not in sessions:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Map agent name to agent object
            agent_map = {
                "researcher": Researcher_agent,
                "analyst": analyst_agent,
                "explainer": anomaly_explainer_agent
            }
            
            # Get the agent object based on the name
            agent = agent_map.get(agent_name, Researcher_agent)  # Default to researcher if unknown
            
            sessions[session_id] = {
                "messages": [],
                "agent": agent,
                "context_variables": {"dataset": combined_df["dataset"], "notes": combined_notes}
            }
            logger.info(f"Created new session: {session_id} with agent: {agent_name}")
        else:
            # If agent is specified and different from current, update it
            if agent_name and agent_name != sessions[session_id]["agent"].name.lower():
                agent_map = {
                    "researcher": Researcher_agent,
                    "analyst": analyst_agent,
                    "explainer": anomaly_explainer_agent
                }
                
                if agent_name in agent_map:
                    sessions[session_id]["agent"] = agent_map[agent_name]
                    logger.info(f"Updated agent to {agent_name} for session {session_id}")

        session_data = sessions[session_id]

        # Create StreamingResponse
        response = StreamingResponse(
            generate_response(user_input, session_data),
            media_type="text/plain"
        )

        # Set cookie attributes based on environment
        cookie_settings = {
            "key": "session_id",
            "value": session_id,
            "httponly": True,
            "max_age": 3600,
            "path": "/chat"
        }

        # Add secure and samesite settings only in production
        if IS_PRODUCTION:
            cookie_settings.update({
                "secure": True,
                "samesite": "None"
            })
        else:
            cookie_settings.update({
                "secure": False,
                "samesite": "Lax"
            })

        response.set_cookie(**cookie_settings)
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise

@router.post("/api/reset")
async def reset_conversation(request: Request, session_id: str = Cookie(None)):
    """Reset the conversation state for the current session."""
    try:
        data = await request.json()
        
        # Get session ID from request body if not in cookies
        body_session_id = data.get("session_id")
        if body_session_id:
            session_id = body_session_id
        
        if session_id and session_id in sessions:
            # Reinitialize the session with default values
            sessions[session_id] = {
                "messages": [],
                "agent": Researcher_agent,  # Reset to default agent
                "context_variables": {"dataset": combined_df["dataset"], "notes": combined_notes}
            }
            response = JSONResponse({"status": "success"})
            
            # Set cookie attributes based on environment
            cookie_settings = {
                "key": "session_id",
                "value": session_id,
                "httponly": True,
                "max_age": 3600,
                "path": "/chat"
            }
            
            # Add secure and samesite settings only in production
            if IS_PRODUCTION:
                cookie_settings.update({
                    "secure": True,
                    "samesite": "None"
                })
            else:
                cookie_settings.update({
                    "secure": False,
                    "samesite": "Lax"
                })
            
            response.set_cookie(**cookie_settings)
            return response
        else:
            # Create a new session if none exists
            if not session_id:
                session_id = str(uuid.uuid4())
            
            sessions[session_id] = {
                "messages": [],
                "agent": Researcher_agent,  # Reset to default agent
                "context_variables": {"dataset": combined_df["dataset"], "notes": combined_notes}
            }
            
            # Return response with session ID
            response = JSONResponse({"status": "success", "session_id": session_id})
            
            # Set cookie attributes based on environment
            cookie_settings = {
                "key": "session_id",
                "value": session_id,
                "httponly": True,
                "max_age": 3600,
                "path": "/chat"
            }
            
            # Add secure and samesite settings only in production
            if IS_PRODUCTION:
                cookie_settings.update({
                    "secure": True,
                    "samesite": "None"
                })
            else:
                cookie_settings.update({
                    "secure": False,
                    "samesite": "Lax"
                })
            
            response.set_cookie(**cookie_settings)
            return response
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"Error resetting conversation: {str(e)}"}
        )

@router.post("/api/switch-agent")
async def switch_agent(request: Request, session_id: str = Cookie(None)):
    """Switch the agent for the current session."""
    try:
        data = await request.json()
        agent_name = data.get("agent")
        
        # Get session ID from request body if not in cookies
        body_session_id = data.get("session_id")
        if body_session_id:
            session_id = body_session_id
        
        if session_id and session_id in sessions:
            # Map agent name to agent object
            agent_map = {
                "researcher": Researcher_agent,
                "analyst": analyst_agent,
                "explainer": anomaly_explainer_agent
            }
            
            if agent_name in agent_map:
                # Update the agent in the session
                sessions[session_id]["agent"] = agent_map[agent_name]
                logger.info(f"Switched to {agent_name} agent for session {session_id}")
                return JSONResponse({"status": "success", "agent": agent_name})
            else:
                return JSONResponse(
                    status_code=400, 
                    content={"status": "error", "message": f"Unknown agent: {agent_name}"}
                )
        else:
            # Create a new session if none exists
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Map agent name to agent object
            agent_map = {
                "researcher": Researcher_agent,
                "analyst": analyst_agent,
                "explainer": anomaly_explainer_agent
            }
            
            # Get the agent object based on the name
            agent = agent_map.get(agent_name, Researcher_agent)  # Default to researcher if unknown
            
            sessions[session_id] = {
                "messages": [],
                "agent": agent,
                "context_variables": {"dataset": combined_df["dataset"], "notes": combined_notes}
            }
            logger.info(f"Created new session: {session_id} with agent: {agent_name}")
            
            # Return response with session ID
            response = JSONResponse({"status": "success", "agent": agent_name, "session_id": session_id})
            
            # Set cookie attributes based on environment
            cookie_settings = {
                "key": "session_id",
                "value": session_id,
                "httponly": True,
                "max_age": 3600,
                "path": "/chat"
            }
            
            # Add secure and samesite settings only in production
            if IS_PRODUCTION:
                cookie_settings.update({
                    "secure": True,
                    "samesite": "None"
                })
            else:
                cookie_settings.update({
                    "secure": False,
                    "samesite": "Lax"
                })
            
            response.set_cookie(**cookie_settings)
            return response
    except Exception as e:
        logger.error(f"Error switching agent: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"Error switching agent: {str(e)}"}
        )

async def generate_response(user_input, session_data):
    logger.info(f"""
=== Starting Agent Response ===
Session ID: {id(session_data)}
Current Agent: {session_data['agent'].name}
Timestamp: {datetime.now().isoformat()}
""")
    
    messages = session_data["messages"]
    agent = session_data["agent"]
    context_variables = session_data.get("context_variables") or {}
    current_function_name = None  # Initialize at the top level

    # Append user message
    messages.append({"role": "user", "content": user_input})
    
    # Truncate messages before sending to agent
    truncated_messages = truncate_messages(messages)

    try:
        # Run the agent
        response_generator = swarm_client.run(
            agent=agent,
            messages=truncated_messages,
            context_variables=context_variables,
            stream=True,
            debug=False,
        )
        
        # Initialize assistant message
        assistant_message = {"role": "assistant", "content": "", "sender": agent.name}
        incomplete_tool_call = None

        for chunk in response_generator:
            # Handle tool calls first
            if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                for tool_call in chunk["tool_calls"]:
                    function_info = tool_call.get("function")
                    if not function_info:
                        continue

                    if function_info.get("name"):
                        current_function_name = function_info["name"]
                        logger.debug(f"Receiving tool call: {current_function_name}")

                    if not current_function_name:
                        continue

                    arguments_fragment = function_info.get("arguments", "")

                    if incomplete_tool_call is None or incomplete_tool_call["function_name"] != current_function_name:
                        incomplete_tool_call = {
                            "type": "tool_call",
                            "sender": assistant_message["sender"],
                            "function_name": current_function_name,
                            "arguments": ""
                        }

                    incomplete_tool_call["arguments"] += arguments_fragment

                    try:
                        arguments_json = json.loads(incomplete_tool_call["arguments"])
                        logger.info(f"""
=== Tool Call ===
Function: {current_function_name}
Arguments: {json.dumps(arguments_json, indent=2)}
""")

                        incomplete_tool_call["arguments"] = arguments_json
                        message = json.dumps(incomplete_tool_call) + "\n"
                        yield message

                        # Process the function call
                        function_to_call = function_mapping.get(current_function_name)
                        if function_to_call:
                            try:
                                # Special handling for generate_chart_message function
                                if current_function_name == 'generate_chart_message':
                                    # Extract chart_data and chart_type from arguments_json
                                    chart_data = arguments_json.get('chart_data')
                                    chart_type = arguments_json.get('chart_type', 'anomaly')
                                    
                                    # Call the function with the correct arguments
                                    result = function_to_call(chart_data=chart_data, chart_type=chart_type)
                                else:
                                    # IMPROVED ARGUMENT HANDLING for other functions
                                    # Check if we have the args/kwargs pattern that needs special handling
                                    if 'args' in arguments_json and 'kwargs' in arguments_json:
                                        # Handle case where kwargs is a JSON string
                                        if isinstance(arguments_json['kwargs'], str) and arguments_json['kwargs'].startswith('{'):
                                            try:
                                                # Parse the kwargs JSON string into a dict
                                                kwargs_dict = json.loads(arguments_json['kwargs'])
                                                # Call function with parsed kwargs
                                                logger.info(f"Calling {current_function_name} with extracted kwargs: {kwargs_dict}")
                                                result = function_to_call(context_variables, **kwargs_dict)
                                            except json.JSONDecodeError:
                                                # If kwargs can't be parsed as JSON, use it as is
                                                logger.warning(f"Couldn't parse kwargs as JSON: {arguments_json['kwargs']}")
                                                result = function_to_call(context_variables, **arguments_json)
                                        else:
                                            # Standard call if kwargs is not a JSON string
                                            result = function_to_call(context_variables, **arguments_json)
                                    else:
                                        # Standard function call with normal arguments
                                        # Remove context_variables from arguments_json if it exists
                                        if 'context_variables' in arguments_json:
                                            del arguments_json['context_variables']
                                        result = function_to_call(context_variables, **arguments_json)
                                
                                logger.info(f"""
=== Tool Result ===
Function: {current_function_name}
Result: {str(result)[:500]}{'...' if len(str(result)) > 500 else ''}
""")
                                # Check if this is an agent transfer function
                                if current_function_name in ['transfer_to_analyst_agent', 'transfer_to_researcher_agent']:
                                    # Update the current agent
                                    session_data['agent'] = result
                                    logger.info(f"""
=== Agent Transfer ===
New Agent: {result.name}
""")
                                    # Add a system message about the transfer
                                    transfer_message = {
                                        "type": "content",
                                        "sender": "System",
                                        "content": f"Transferring to {result.name} Agent..."
                                    }
                                    yield json.dumps(transfer_message) + "\n"
                                    
                                # If the result has content (like from format_table), send it as a message
                                elif isinstance(result, dict) and "content" in result:
                                    message = {
                                        "type": "content",
                                        "sender": assistant_message["sender"],
                                        "content": result["content"]
                                    }
                                    yield json.dumps(message) + "\n"
                                # Handle chart messages
                                elif isinstance(result, dict) and result.get("type") == "chart":
                                    logger.info(f"""
=== Chart Message Detected ===
Chart ID: {result.get("chart_id")}
Chart Type: {result.get("chart_type")}
Chart Data: {str(result.get("chart_data"))[:200]}...
Chart HTML Length: {len(result.get("chart_html", ""))}
""")
                                    try:
                                        # Ensure all chart data is JSON serializable
                                        chart_data = result.get("chart_data")
                                        
                                        # If chart_data is a string, try to parse it as JSON
                                        if isinstance(chart_data, str):
                                            try:
                                                # Try to parse it as JSON
                                                chart_data = json.loads(chart_data)
                                            except json.JSONDecodeError:
                                                # If it's not valid JSON, it might be a Python string representation
                                                # Try to evaluate it safely
                                                try:
                                                    # Replace Python literals with JSON equivalents
                                                    safe_str = chart_data.replace("True", "true").replace("False", "false").replace("None", "null")
                                                    # Remove any datetime objects that might cause issues
                                                    safe_str = re.sub(r'datetime\.date\([^)]+\)', '"date"', safe_str)
                                                    safe_str = re.sub(r'datetime\.datetime\([^)]+\)', '"datetime"', safe_str)
                                                    chart_data = json.loads(safe_str)
                                                except Exception as e:
                                                    logger.error(f"Failed to parse chart data string: {e}")
                                                    # Fall back to a simplified version
                                                    chart_data = {"error": "Could not parse chart data"}
                                        
                                        message = {
                                            "type": "chart",
                                            "sender": assistant_message["sender"],
                                            "chart_id": result.get("chart_id"),
                                            "chart_type": result.get("chart_type"),
                                            "chart_data": chart_data,
                                            "chart_html": result.get("chart_html")
                                        }
                                        
                                        # Ensure the message is JSON serializable
                                        json_message = json.dumps(message)
                                        yield json_message + "\n"
                                        logger.info("Chart message sent to client")
                                    except Exception as e:
                                        logger.error(f"Error sending chart message: {str(e)}")
                                        # Send a fallback message instead
                                        fallback_message = {
                                            "type": "content",
                                            "sender": assistant_message["sender"],
                                            "content": f"Error generating chart: {str(e)}"
                                        }
                                        yield json.dumps(fallback_message) + "\n"
                            except Exception as e:
                                logger.error(f"""
=== Tool Error ===
Function: {current_function_name}
Error: {str(e)}
""")
                                raise

                        incomplete_tool_call = None
                        current_function_name = None
                    except json.JSONDecodeError:
                        # Still accumulating arguments
                        pass

            # Handle content
            elif "content" in chunk and chunk["content"] is not None:
                content_piece = chunk["content"]
                assistant_message["content"] += content_piece
                message = {
                    "type": "content",
                    "sender": assistant_message["sender"],
                    "content": content_piece
                }
                yield json.dumps(message) + "\n"

            # Handle end of message
            if "delim" in chunk and chunk["delim"] == "end":
                # Always append assistant message if it has content
                if assistant_message["content"]:
                    messages.append(assistant_message)
                    logger.info(f"""
=== Agent Response Complete ===
Timestamp: {datetime.now().isoformat()}
Agent: {assistant_message['sender']}
Response:
{assistant_message['content']}
===============================
""")
                # Reset for next message
                assistant_message = {"role": "assistant", "content": "", "sender": agent.name}

    except Exception as e:
        logger.error(f"""
=== Error in Response Generation ===
Error type: {type(e).__name__}
Error message: {str(e)}
Current function: {current_function_name}
Timestamp: {datetime.now().isoformat()}
===============================
""")
        raise

    logger.info(f"""
=== Chat Interaction Summary ===
Total messages: {len(messages)}
Timestamp: {datetime.now().isoformat()}
===============================
""")

@router.get("/backend/get-notes")
async def get_notes_file():
    """
    Serves the combined notes file from the output/notes directory.
    """
    logger = logging.getLogger(__name__)
    
    script_dir = Path(__file__).parent
    notes_file = script_dir / 'output' / 'notes' / 'combined_notes.txt'
    
    if not notes_file.exists():
        logger.error("Notes file not found")
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Notes file not found"}
        )
    
    try:
        with open(notes_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return JSONResponse({
            "success": True,
            "content": content,
            "token_count": len(content.split())  # Approximate token count
        })
    except Exception as e:
        logger.error(f"Error reading notes file: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error reading notes file: {str(e)}"}
        )

@router.get("/")
async def index(request: Request, session_id: str = Cookie(None)):
    """
    Serve the webchat interface with optional parameters for initializing a session.
    """
    # Get query parameters
    params = dict(request.query_params)
    
    # If we have metric data parameters, initialize a session with the explainer agent
    if "metric_id" in params and "metric_name" in params:
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Initialize session with the explainer agent
        if session_id not in sessions:
            sessions[session_id] = {
                "messages": [],
                "agent": anomaly_explainer_agent,
                "context_variables": {"dataset": combined_df["dataset"], "notes": combined_notes}
            }
        
        # Add a system message with the metric data
        metric_data = {
            "metric_id": params.get("metric_id"),
            "metric_name": params.get("metric_name"),
            "district": params.get("district", "0"),
            "period_type": params.get("period_type", "month"),
            "previous_value": params.get("previous_value"),
            "recent_value": params.get("recent_value"),
            "previous_period": params.get("previous_period"),
            "recent_period": params.get("recent_period"),
            "delta": params.get("delta"),
            "percent_change": params.get("percent_change")
        }
        
        # Create a prompt for the agent
        prompt = f"""I need you to analyze a significant change in the metric '{metric_data['metric_name']}' (ID: {metric_data['metric_id']}) for district {metric_data['district']}.

METRIC DETAILS:
- Name: {metric_data['metric_name']} 
- ID: {metric_data['metric_id']}
- District: {metric_data['district']}
- Period Type: {metric_data['period_type']}
- Previous Period: {metric_data['previous_period']} - Value: {metric_data['previous_value']}
- Recent Period: {metric_data['recent_period']} - Value: {metric_data['recent_value']}
- Change: {metric_data['delta']} ({metric_data['percent_change']}%)

Please analyze this metric change and explain why it occurred. Consider:
1. Historical context and trends
2. Possible contributing factors
3. Similar patterns in related metrics
4. Whether this is part of a longer trend

Use the available tools to gather data and provide a comprehensive explanation."""
        
        # Add the user message to the session
        sessions[session_id]["messages"].append({"role": "user", "content": prompt})
        
        # Trigger the agent to respond
        asyncio.create_task(generate_response(prompt, sessions[session_id]))
    
    # Return the index.html template
    return templates.TemplateResponse("index.html", {"request": request})

def get_anomaly_details(anomaly_id):
    """Get detailed information about a specific anomaly by ID."""
    try:
        # Get the anomaly data from the database
        anomaly = get_anomaly_by_id(anomaly_id)
        if not anomaly:
            return {"error": "Anomaly not found"}
            
        # Convert date objects to ISO format strings
        if isinstance(anomaly.get('created_at'), date):
            anomaly['created_at'] = anomaly['created_at'].isoformat()
        if isinstance(anomaly.get('recent_date'), date):
            anomaly['recent_date'] = anomaly['recent_date'].isoformat()
            
        # Convert dates in chart data
        if 'chart_data' in anomaly:
            chart_data = anomaly['chart_data']
            if 'dates' in chart_data:
                chart_data['dates'] = [d.isoformat() if hasattr(d, 'isoformat') else d for d in chart_data['dates']]
                
        # Generate the chart HTML
        chart_html = generate_anomaly_chart_html(anomaly)
        
        # Add the chart HTML to the response
        anomaly['chart_html'] = chart_html
        
        return anomaly
        
    except Exception as e:
        logger.error(f"Error getting anomaly details: {str(e)}")
        return {"error": str(e)}

def generate_chart_message(chart_data, chart_type="anomaly"):
    """
    Generate a special message type for chart data that can be sent to the client.
    
    Args:
        chart_data: The data for the chart
        chart_type: The type of chart (default: "anomaly")
        
    Returns:
        A dictionary with the chart message
    """
    logger.info(f"Generating chart message of type: {chart_type}")
    
    # Create a unique ID for the chart container
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"
    
    # Generate the chart HTML based on the chart type
    if chart_type == "anomaly":
        chart_html = generate_anomaly_chart_html(chart_data)
    else:
        # Default to a simple chart if type is not recognized
        chart_html = f"""
        <div id="{chart_id}" style="width: 100%; max-width: 800px; margin: 20px auto;">
            <div class="chart-wrapper" style="width: 100%; overflow: hidden; border-radius: 12px; border: 3px solid #4A7463; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <div id="chart-container-{chart_id}" style="width: 100%; padding: 20px; box-sizing: border-box;"></div>
            </div>
        </div>
        """
    
    # Create the chart message
    chart_message = {
        "type": "chart",
        "chart_id": chart_id,
        "chart_type": chart_type,
        "chart_data": chart_data,
        "chart_html": chart_html
    }
    
    return chart_message


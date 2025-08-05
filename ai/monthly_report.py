import os
import json
import logging
import sys
import psycopg2
import psycopg2.extras
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import traceback
import base64
from io import BytesIO
from PIL import Image
import time
import re
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tools.generate_report_text import generate_report_text
from tools.dashboard_metric_tool import get_dashboard_metric

# Set up paths to look for .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add the project root and script directory to the Python path
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

possible_env_paths = [
    os.path.join(script_dir, '.env'),  # ai/.env
    os.path.join(project_root, '.env'),  # /.env (project root)
    os.path.join(os.path.expanduser('~'), '.env')  # ~/.env (home directory)
]

# Try loading .env from each location
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        print(f"Loading environment variables from: {env_path}")
        load_dotenv(env_path)
        break

# Configure logging - IMPORTANT: Do this before importing any other modules
# that might configure their own loggers

# Get logging level from environment or default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure file handler with absolute path
log_file = os.path.join(logs_dir, 'monthly_report.log')
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(log_level)  # Set handler level to match log_level

# Configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)  # Set handler level to match log_level

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level)  # Set root logger level to match log_level

# Remove any existing handlers from the root logger to avoid duplicate logs
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Create a module-specific logger that will use the root logger's configuration
logger = logging.getLogger(__name__)

# Now log initialization messages
logger.warning(f"Monthly report logging initialized with level: {log_level_name}")
logger.info("This is an INFO level message - should not appear if LOG_LEVEL is WARNING")
logger.warning(f"Log file location: {log_file}")
logger.debug("This is a DEBUG level message - should never appear if LOG_LEVEL is WARNING")
logger.warning(f"Current working directory: {os.getcwd()}")
logger.warning(f"Script directory: {script_dir}")
logger.warning(f"Project root: {project_root}")
logger.info(f"Python path: {sys.path}")  # This should not appear if LOG_LEVEL is WARNING

# Test write access to log file
try:
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\nTest write at {datetime.now()}\n")
    logger.warning(f"Successfully wrote to log file: {log_file}")
except Exception as e:
    logger.error(f"Error writing to log file: {e}")

# Prevent imported modules from modifying our logging configuration
# This is done by defining a no-op basicConfig function
original_basic_config = logging.basicConfig
def no_op_basic_config(*args, **kwargs):
    # If an imported module tries to configure logging, we'll maintain our settings
    # But we'll log it at debug level to help troubleshoot if needed
    logger.debug(f"Ignored attempt to call logging.basicConfig with args: {args}, kwargs: {kwargs}")
    
    # If the caller is trying to set a level, we'll check against our level
    if 'level' in kwargs:
        requested_level = kwargs['level']
        logger.debug(f"Module requested logging level: {requested_level}, our level: {log_level}")
    
logging.basicConfig = no_op_basic_config

# Import database utilities
try:
    # First try to import from the ai package
    from ai.tools.db_utils import get_postgres_connection, execute_with_connection, CustomJSONEncoder
    logger.warning("Successfully imported from ai.tools.db_utils")
    from ai.tools.genChartdw import create_datawrapper_chart # New import
    logger.warning("Successfully imported from ai.tools.genChartdw")
    from ai.tools.gen_anomaly_chart_dw import generate_anomaly_chart_dw # Import for anomaly charts
except ImportError:
    try:
        # If that fails, try to import from the local directory
        from tools.db_utils import get_postgres_connection, execute_with_connection, CustomJSONEncoder
        logger.warning("Successfully imported from tools.db_utils")
        from tools.genChartdw import create_datawrapper_chart # New import - local fallback
        logger.warning("Successfully imported from tools.genChartdw")
        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_dw # Import for anomaly charts - local fallback
    except ImportError:
        logger.error("Failed to import from db_utils or genChartdw", exc_info=True)
        raise

# Import necessary functions from other modules
try:
    # First try to import from the ai package
    from ai.webChat import get_dashboard_metric, swarm_client, context_variables, client, AGENT_MODEL, load_and_combine_notes
    from ai.agents.explainer_agent import create_explainer_agent
    logger.warning("Successfully imported from ai.webChat and ai.agents.explainer_agent")
except ImportError:
    try:
        # If that fails, try to import from the local directory
        from webChat import swarm_client, context_variables, client, AGENT_MODEL, load_and_combine_notes
        from agents.explainer_agent import create_explainer_agent
        logger.warning("Successfully imported from webChat and agents.explainer_agent")
    except ImportError:
        logger.error("Failed to import from webChat or explainer_agent", exc_info=True)
        raise

# Restore original basicConfig if needed
logging.basicConfig = original_basic_config

# Load environment variables
load_dotenv()

# Only log non-sensitive environment variables
non_sensitive_vars = {k: v for k, v in os.environ.items() 
                     if k.startswith('POSTGRES_') and not k.endswith('PASSWORD')}
logger.warning(f"Environment variables (excluding sensitive data): {non_sensitive_vars}")

# Database connection parameters - use POSTGRES_* variables directly
DB_HOST = os.getenv("POSTGRES_HOST", 'localhost')
DB_PORT = os.getenv("POSTGRES_PORT", '5432')
DB_USER = os.getenv("POSTGRES_USER", 'postgres')
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", 'postgres')
DB_NAME = os.getenv("POSTGRES_DB", 'transparentsf')

# Log only non-sensitive connection parameters
logger.warning(f"Using database connection parameters: HOST={DB_HOST}, PORT={DB_PORT}, USER={DB_USER}, DB={DB_NAME}")

# Validate and convert DB_PORT to int if it exists
try:
    DB_PORT = int(DB_PORT)
except ValueError:
    logger.error(f"Invalid POSTGRES_PORT value: {DB_PORT}. Must be an integer.")
    DB_PORT = 5432  # Default port

# API Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
logger.warning(f"Using API_BASE_URL: {API_BASE_URL}")

# Perplexity API Key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if PERPLEXITY_API_KEY:
    logger.warning("Perplexity API key found")
else:
    logger.warning("No Perplexity API key found. Perplexity API features will not be available.")

# Global variable to store prompts
_PROMPTS = None

def load_prompts():
    """
    Load prompts from the JSON file and store them in the global _PROMPTS variable.
    Only loads the prompts once.
    
    Returns:
        Dictionary containing all prompts
    """
    global _PROMPTS
    
    if _PROMPTS is None:
        try:
            prompts_path = Path(__file__).parent / 'data' / 'prompts.json'
            logger.info(f"Loading prompts from {prompts_path}")
            
            with open(prompts_path, 'r') as f:
                _PROMPTS = json.load(f)
                
            logger.info(f"Successfully loaded prompts with keys: {list(_PROMPTS.keys())}")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
    
    return _PROMPTS

def initialize_monthly_reporting_table():
    """
    Initialize the monthly_reporting table in PostgreSQL if it doesn't exist.
    Also creates the reports table if it doesn't exist.
    """
    logger.info("Initializing tables for monthly reporting")
    
    def init_table_operation(connection):
        """Inner function to initialize the tables within a connection."""
        cursor = connection.cursor()
        
        # First, check and create the reports table if it doesn't exist
        logger.info("Checking if reports table exists...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'reports'
            );
        """)
        reports_table_exists = cursor.fetchone()[0]
        
        if not reports_table_exists:
            logger.info("Creating reports table...")
            try:
                cursor.execute("""
                    CREATE TABLE reports (
                        id SERIAL PRIMARY KEY,
                        max_items INTEGER DEFAULT 10,
                        district TEXT DEFAULT '0',
                        period_type TEXT DEFAULT 'month',
                        original_filename TEXT,
                        revised_filename TEXT,
                        published_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                connection.commit()
                logger.info("Reports table created successfully")
            except Exception as e:
                logger.error(f"Error creating reports table: {e}")
                connection.rollback()
                return False
        else:
            logger.info("Reports table already exists")
            
            # Check if the published_url column exists in the reports table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'reports' AND column_name = 'published_url'
            """)
            
            published_url_exists = cursor.fetchone()
            
            if not published_url_exists:
                logger.info("Adding published_url column to reports table")
                try:
                    cursor.execute("""
                        ALTER TABLE reports ADD COLUMN published_url TEXT
                    """)
                    connection.commit()
                    logger.info("Added published_url column to reports table")
                except Exception as e:
                    logger.error(f"Error adding published_url column: {e}")
                    connection.rollback()

            # Check if the proofread_feedback column exists in the reports table
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'reports' AND column_name = 'proofread_feedback'
            """)
            proofread_feedback_exists = cursor.fetchone()

            if not proofread_feedback_exists:
                logger.info("Adding proofread_feedback column to reports table")
                try:
                    cursor.execute("""
                        ALTER TABLE reports ADD COLUMN proofread_feedback TEXT
                    """)
                    connection.commit()
                    logger.info("Added proofread_feedback column to reports table")
                except Exception as e:
                    logger.error(f"Error adding proofread_feedback column: {e}")
                    connection.rollback()

            # Check if the headlines column exists in the reports table
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'reports' AND column_name = 'headlines'
            """)
            headlines_exists = cursor.fetchone()

            if not headlines_exists:
                logger.info("Adding headlines column to reports table")
                try:
                    cursor.execute("""
                        ALTER TABLE reports ADD COLUMN headlines JSONB
                    """)
                    connection.commit()
                    logger.info("Added headlines column to reports table")
                except Exception as e:
                    logger.error(f"Error adding headlines column: {e}")
                    connection.rollback()
        
        # Now check if the monthly_reporting table exists and get its columns
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'monthly_reporting'
            ORDER BY ordinal_position
        """)
        
        existing_columns = [col[0] for col in cursor.fetchall()]
        
        # If the table exists but with a different schema, we need to handle it
        if existing_columns:
            logger.info(f"Table monthly_reporting exists with columns: {existing_columns}")
            
            # Check if we have the required columns for our new schema
            required_columns = [
                "id", "report_date", "metric_name", "group_value", 
                "group_field_name", "period_type", "comparison_mean", 
                "recent_mean", "difference", "std_dev", "percent_change", 
                "explanation", "priority", "report_text", "district", 
                "chart_data", "metadata", "created_at", "metric_id", "report_id", "item_title"
            ]
            
            missing_columns = [col for col in required_columns if col not in existing_columns]
            
            if missing_columns:
                logger.info(f"Adding missing columns to monthly_reporting table: {missing_columns}")
                
                # Add missing columns one by one
                for column in missing_columns:
                    try:
                        if column == "id":
                            # Skip id as it's likely already the primary key
                            continue
                        elif column == "report_date":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN report_date DATE DEFAULT CURRENT_DATE")
                        elif column == "metric_name":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN metric_name TEXT")
                        elif column == "group_value":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN group_value TEXT")
                        elif column == "group_field_name":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN group_field_name TEXT")
                        elif column == "period_type":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN period_type TEXT DEFAULT 'month'")
                        elif column == "comparison_mean":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN comparison_mean FLOAT")
                        elif column == "recent_mean":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN recent_mean FLOAT")
                        elif column == "difference":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN difference FLOAT")
                        elif column == "std_dev":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN std_dev FLOAT")
                        elif column == "percent_change":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN percent_change FLOAT")
                        elif column == "explanation":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN explanation TEXT")
                        elif column == "priority":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN priority INTEGER")
                        elif column == "report_text":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN report_text TEXT")
                        elif column == "district":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN district TEXT")
                        elif column == "chart_data":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN chart_data JSONB")
                        elif column == "metadata":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN metadata JSONB")
                        elif column == "created_at":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                        elif column == "metric_id":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN metric_id TEXT")
                        elif column == "report_id":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN report_id INTEGER REFERENCES reports(id) ON DELETE CASCADE")
                        elif column == "item_title":
                            cursor.execute("ALTER TABLE monthly_reporting ADD COLUMN item_title TEXT")

                    except Exception as e:
                        logger.warning(f"Error adding column {column}: {e}")
                        # Continue with other columns even if one fails
                
                connection.commit()
                logger.info("Added missing columns to monthly_reporting table")
            else:
                logger.info("Table monthly_reporting already has all required columns")
        else:
            # Create monthly_reporting table if it doesn't exist
            logger.info("Creating monthly_reporting table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monthly_reporting (
                    id SERIAL PRIMARY KEY,
                    report_id INTEGER REFERENCES reports(id) ON DELETE CASCADE,
                    report_date DATE DEFAULT CURRENT_DATE,
                    item_title TEXT,
                    metric_name TEXT,
                    metric_id TEXT,
                    group_value TEXT,
                    group_field_name TEXT,
                    period_type TEXT DEFAULT 'month',
                    comparison_mean FLOAT,
                    recent_mean FLOAT,
                    difference FLOAT,
                    std_dev FLOAT,
                    percent_change FLOAT,
                    explanation TEXT,
                    priority INTEGER,
                    report_text TEXT,
                    district TEXT,
                    chart_data JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS monthly_reporting_report_date_idx ON monthly_reporting (report_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS monthly_reporting_district_idx ON monthly_reporting (district)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS monthly_reporting_priority_idx ON monthly_reporting (priority)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS monthly_reporting_report_id_idx ON monthly_reporting (report_id)
            """)
            
            connection.commit()
            logger.info("Monthly reporting table created successfully")
        
        cursor.close()
        return True
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=init_table_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success":
        return True
    else:
        logger.error(f"Failed to initialize tables: {result['message']}")
        return False

def select_deltas_to_discuss(period_type='month', top_n=20, bottom_n=20, district=""):
    """
    Step 1: Select the top and bottom N differences to discuss.
    Uses the anomaly_analyzer API endpoint to find the most significant changes.
    
    Args:
        period_type: The time period type (month, quarter, year)
        top_n: Number of top increasing differences to select
        bottom_n: Number of top decreasing differences to select
        district: District number to analyze
        
    Returns:
        Dictionary with top and bottom differences
    """
    logger.info(f"Selecting top {top_n} and bottom {bottom_n} deltas for {period_type}ly report")
    
    try:
        # Use the top-metric-changes API endpoint to get both positive and negative changes
        # The updated API returns both in one call, so we use the max of top_n and bottom_n for the limit
        limit = max(top_n, bottom_n)
        url = f"{API_BASE_URL}/anomaly-analyzer/api/top-metric-changes?period_type={period_type}&limit={limit}&district={district}&show_both=true&show_on_dash=true"
        logger.info(f"Requesting metric changes from: {url}")
        
        response = requests.get(url)
        logger.info(f"Metric changes API response status code: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"Failed to get metric changes: Status code {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        try:
            data = response.json()
            logger.info(f"Received metric changes data: {json.dumps(data)[:200]}...")
            
            # Log more details about the returned data structure
            if isinstance(data, dict):
                logger.info(f"Data keys: {list(data.keys())}")
                if "positive_changes" in data:
                    logger.info(f"Positive changes count: {len(data['positive_changes'])}")
                    if data['positive_changes'] and len(data['positive_changes']) > 0:
                        sample_item = data['positive_changes'][0]
                        logger.info(f"Sample positive change item keys: {list(sample_item.keys())}")
                        logger.info(f"Sample positive change item: {json.dumps(sample_item)}")
                if "negative_changes" in data:
                    logger.info(f"Negative changes count: {len(data['negative_changes'])}")
                    if data['negative_changes'] and len(data['negative_changes']) > 0:
                        sample_item = data['negative_changes'][0]
                        logger.info(f"Sample negative change item keys: {list(sample_item.keys())}")
                        logger.info(f"Sample negative change item: {json.dumps(sample_item)}")
        except Exception as e:
            logger.error(f"Error parsing metric changes response as JSON: {str(e)}")
            logger.error(f"Raw response content: {response.text[:500]}")
            raise
        
        # Format the results from the API to match what we need
        top_changes = []
        for item in data.get("positive_changes", [])[:top_n]:  # Limit to top_n
            try:
                # Safely handle None values
                recent_value = float(item.get("recent_value", 0)) if item.get("recent_value") is not None else 0
                previous_value = float(item.get("previous_value", 0)) if item.get("previous_value") is not None else 0
                difference = recent_value - previous_value
                
                top_changes.append({
                    "metric": item.get("object_name", "Unknown"),
                    "metric_id": item.get("object_id", "Unknown"),
                    "group": item.get("group_value") or "All",  # Handle None values properly
                    "recent_mean": recent_value,
                    "comparison_mean": previous_value,
                    "difference_value": difference,
                    "difference": difference,
                    "district": district,
                    "change_type": "positive",  # Mark as positive change (good change based on greendirection)
                    "percent_change": item.get("percent_change"),  # Preserve percent_change from API
                    "greendirection": item.get("greendirection"),  # Preserve greendirection for reference
                    "citywide_changes": item.get("citywide_changes", "")  # Include citywide changes data
                })
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing positive change item: {e}")
                logger.error(f"Problematic item data: {json.dumps(item)}")
                continue
        
        bottom_changes = []
        for item in data.get("negative_changes", [])[:bottom_n]:  # Limit to bottom_n
            try:
                # Safely handle None values
                recent_value = float(item.get("recent_value", 0)) if item.get("recent_value") is not None else 0
                previous_value = float(item.get("previous_value", 0)) if item.get("previous_value") is not None else 0
                difference = recent_value - previous_value
                
                bottom_changes.append({
                    "metric": item.get("object_name", "Unknown"),
                    "metric_id": item.get("object_id", "Unknown"),
                    "group": item.get("group_value") or "All",  # Handle None values properly
                    "recent_mean": recent_value,
                    "comparison_mean": previous_value,
                    "difference_value": difference,
                    "difference": difference,
                    "district": district,
                    "change_type": "negative",  # Mark as negative change (bad change based on greendirection)
                    "percent_change": item.get("percent_change"),  # Preserve percent_change from API
                    "greendirection": item.get("greendirection"),  # Preserve greendirection for reference
                    "citywide_changes": item.get("citywide_changes", "")  # Include citywide changes data
                })
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing negative change item: {e}")
                logger.error(f"Problematic item data: {json.dumps(item)}")
                continue
        
        # Combine results
        all_deltas = {
            "top_changes": top_changes,
            "bottom_changes": bottom_changes,
            "status": "success",
            "period_type": period_type,
            "district": district
        }
        
        logger.info(f"Found {len(all_deltas['top_changes'])} top changes and {len(all_deltas['bottom_changes'])} bottom changes")
        return all_deltas
        
    except Exception as e:
        error_msg = f"Error in select_deltas_to_discuss: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def prioritize_deltas(deltas, max_items=10):
    """
    Step 2: Prioritize the deltas for discussion based on their importance.
    This uses the explainer agent to determine which changes are most significant.
    
    Args:
        deltas: Dictionary with top and bottom changes from select_deltas_to_discuss
        max_items: Maximum number of items to prioritize
        
    Returns:
        List of prioritized items with explanations
    """
    from webChat import client, AGENT_MODEL, load_and_combine_notes
    
    logger.info(f"Prioritizing deltas for discussion (max {max_items} items)")
    
    try:
        if deltas.get("status") != "success":
            return {"status": "error", "message": "Invalid deltas input"}
        
        # Combine top and bottom changes
        combined_changes = deltas.get("top_changes", []) + deltas.get("bottom_changes", [])
        
        # Get combined notes with YTD metrics for reference
        notes_text = load_and_combine_notes()
        logger.info(f"Loaded {len(notes_text)} characters of combined notes for context")
        
        # Format changes for the agent to analyze
        changes_text = "Here are the metrics with significant changes:\n\n"
        for idx, change in enumerate(combined_changes, 1):
            metric = change.get("metric", "Unknown")
            metric_id = change.get("metric_id", "Unknown")
            group = change.get("group", "Unknown")
            # Safely handle None values for group
            if group is None:
                group = "All"
            diff = change.get("difference_value", 0)
            recent = change.get("recent_mean", 0)
            comparison = change.get("comparison_mean", 0)
            district = change.get("district", "0")
            citywide_changes = change.get("citywide_changes", "")
            greendirection = change.get("greendirection", "neutral")
            
            # Add an index field to each change for reference
            change["index"] = idx
            
            # Calculate percent change
            if comparison != 0:
                pct_change = (diff / comparison) * 100
                pct_text = f"{pct_change:+.1f}%"
            else:
                pct_text = "N/A"
            
            # Format the change text with additional context
            changes_text += f"{idx}. {metric} for {group}: {recent:.1f} vs {comparison:.1f} ({pct_text}), District: {district}\n"
            changes_text += f"   Direction: {greendirection}, Citywide Changes: {citywide_changes}\n\n"
        
        # Load prompt from JSON file
        prompts = load_prompts()
        prompt_template = prompts['monthly_report']['prioritize_deltas']['prompt']
        system_message = prompts['monthly_report']['prioritize_deltas']['system']
        
        # Truncate notes_text for the prompt
        notes_text_short = notes_text
        
        # Format the prompt with the required variables
        prompt = prompt_template.format(
            max_items=max_items,
            changes_text=changes_text,
            notes_text_short=notes_text_short
        )

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(project_root,'ai', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save the complete prompts to a log file
        prompt_log_path = os.path.join(logs_dir, 'prioritize_deltas_prompt.txt')
        try:
            with open(prompt_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"System Message:\n{system_message}\n\n")
                f.write(f"User Prompt:\n{prompt}\n\n")
                f.write(f"Input Deltas Data:\n{json.dumps(deltas, indent=2)}\n")
            logger.info(f"Saved complete prioritize deltas prompts to {prompt_log_path}")
        except Exception as log_error:
            logger.error(f"Error saving prioritize deltas prompts to log file: {log_error}")

        # Make API call to get prioritized list
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "system", "content": system_message},
                     {"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content
        logger.info(f"Received JSON response of length: {len(response_content)}")
        
        try:
            # Parse the JSON response
            prioritized_json = json.loads(response_content)
            logger.info(f"Parsed JSON: {json.dumps(prioritized_json)[:200]}...")
            
            # Extract the array if the response is wrapped in an object
            prioritized_items_raw = []
            if isinstance(prioritized_json, dict):
                # Try different possible keys that might contain the items array
                for key in ["items", "prioritized_items", "metrics", "results", "data"]:
                    if key in prioritized_json:
                        prioritized_items_raw = prioritized_json.get(key, [])
                        logger.info(f"Found items under key: {key}")
                        break
                
                # If no known key was found but there's a single array in the dict, use it
                if not prioritized_items_raw:
                    for value in prioritized_json.values():
                        if isinstance(value, list):
                            prioritized_items_raw = value
                            logger.info(f"Found items array as a value")
                            break
                            
                # If still no array, treat the entire object as a single-item array if it has 'metric'
                if not prioritized_items_raw and "metric" in prioritized_json:
                    prioritized_items_raw = [prioritized_json]
                    logger.info("Treating entire object as a single item")
            elif isinstance(prioritized_json, list):
                prioritized_items_raw = prioritized_json
                logger.info("Found direct array of items")
            else:
                logger.warning(f"Unexpected JSON structure: {type(prioritized_json)}")
                
            # Print what we found for debugging
            logger.info(f"Extracted {len(prioritized_items_raw)} items from JSON")
            for i, item in enumerate(prioritized_items_raw[:3]):  # Print first 3 for debugging
                logger.info(f"  Item {i+1}: {json.dumps(item)[:100]}...")
            
            # Process each item to match with original data
            prioritized_items = []
            for item in prioritized_items_raw:
                # Get the item index to match with original data
                item_index = item.get("index")
                if item_index is not None:
                    # Convert index to integer if it's a string
                    if isinstance(item_index, str):
                        try:
                            item_index = int(item_index)
                        except ValueError:
                            logger.warning(f"Invalid index format: {item_index}")
                            continue
                    
                    # Find the original change data by index
                    original_change = None
                    for change in combined_changes:
                        if change.get("index") == item_index:
                            original_change = change
                            break
                    
                    if original_change:
                        logger.info(f"Found original data for item index {item_index}: {original_change.get('metric')}")
                        # Create prioritized item with data from both the AI response and original change
                        prioritized_items.append({
                            "metric": original_change.get("metric"),
                            "metric_id": original_change.get("metric_id", "Unknown"),
                            "group": original_change.get("group", "All"),
                            "priority": item.get("priority", 999),
                            "recent_mean": original_change.get("recent_mean", 0),
                            "comparison_mean": original_change.get("comparison_mean", 0),
                            "difference": original_change.get("difference_value", 0),
                            "district": original_change.get("district", "0"),
                            "rationale": item.get("explanation", ""),
                            "trend_analysis": item.get("trend_analysis", ""),
                            "follow_up": item.get("follow_up", ""),
                            "change_type": original_change.get("change_type", "neutral"),  # Preserve change_type
                            "percent_change": original_change.get("percent_change"),  # Preserve percent_change
                            "greendirection": original_change.get("greendirection"),  # Preserve greendirection
                            "citywide_changes": original_change.get("citywide_changes", "")  # Include citywide changes
                        })
                    else:
                        logger.warning(f"Could not find original data for item index {item_index}")
                        # Try to find by metric name and group as fallback
                        metric_name = item.get("metric", "").strip()
                        group_value = item.get("group", "All").strip()
                        
                        # Find the original change data
                        original_change = None
                        for change in combined_changes:
                            if change.get("metric") == metric_name and (change.get("group") == group_value or (change.get("group") is None and group_value == "All")):
                                original_change = change
                                break
                        
                        if original_change:
                            logger.info(f"Found original data for metric: {metric_name} by name matching")
                            prioritized_items.append({
                                "metric": metric_name,
                                "metric_id": original_change.get("metric_id", "Unknown"),
                                "group": group_value,
                                "priority": item.get("priority", 999),
                                "recent_mean": original_change.get("recent_mean", 0),
                                "comparison_mean": original_change.get("comparison_mean", 0),
                                "difference": original_change.get("difference_value", 0),
                                "district": original_change.get("district", "0"),
                                "rationale": item.get("explanation", ""),
                                "trend_analysis": item.get("trend_analysis", ""),
                                "follow_up": item.get("follow_up", ""),
                                "change_type": original_change.get("change_type", "neutral"),  # Preserve change_type
                                "percent_change": original_change.get("percent_change"),  # Preserve percent_change
                                "greendirection": original_change.get("greendirection"),  # Preserve greendirection
                                "citywide_changes": original_change.get("citywide_changes", "")  # Include citywide changes
                            })
                        else:
                            logger.warning(f"Could not find original data for metric: {metric_name}, group: {group_value}")
                else:
                    # Fallback to matching by name if no index
                    metric_name = item.get("metric", "").strip()
                    group_value = item.get("group", "All").strip()
                    
                    # Find the original change data
                    original_change = None
                    for change in combined_changes:
                        if change.get("metric") == metric_name and (change.get("group") == group_value or (change.get("group") is None and group_value == "All")):
                            original_change = change
                            break
                    
                    if original_change:
                        logger.info(f"Found original data for metric: {metric_name} by name matching")
                        prioritized_items.append({
                            "metric": metric_name,
                            "metric_id": original_change.get("metric_id", "Unknown"),
                            "group": group_value,
                            "priority": item.get("priority", 999),
                            "recent_mean": original_change.get("recent_mean", 0),
                            "comparison_mean": original_change.get("comparison_mean", 0),
                            "difference": original_change.get("difference_value", 0),
                            "district": original_change.get("district", "0"),
                            "rationale": item.get("explanation", ""),
                            "trend_analysis": item.get("trend_analysis", ""),
                            "follow_up": item.get("follow_up", ""),
                            "change_type": original_change.get("change_type", "neutral"),  # Preserve change_type
                            "percent_change": original_change.get("percent_change"),  # Preserve percent_change
                            "greendirection": original_change.get("greendirection"),  # Preserve greendirection
                            "citywide_changes": original_change.get("citywide_changes", "")  # Include citywide changes
                        })
                    else:
                        logger.warning(f"Could not find original data for metric: {metric_name}, group: {group_value}")
            
            # Sort by priority
            prioritized_items.sort(key=lambda x: x.get("priority", 999))
            
            # Log prioritized items
            for item in prioritized_items:
                logger.info(f"Prioritized item: {item.get('metric')} - {item.get('group')} (Priority: {item.get('priority')})")
                logger.info(f"  Explanation length: {len(item.get('rationale', ''))}")
                logger.info(f"  Trend analysis: {item.get('trend_analysis', '')[:50]}...")
            
            logger.info(f"Prioritized {len(prioritized_items)} items for the report")
            return {"status": "success", "prioritized_items": prioritized_items}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_content[:200]}...")
            return {"status": "error", "message": f"Failed to parse JSON response: {e}"}
        
    except Exception as e:
        error_msg = f"Error in prioritize_deltas: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def store_prioritized_items(prioritized_items, period_type='month', district=None):
    """
    Store prioritized items in the monthly_reporting table.
    Args:
        prioritized_items: List of prioritized items to store
        period_type: Type of period (month, quarter, year)
        district: District number
    Returns:
        Status dictionary
    """
    logger.info(f"Storing {len(prioritized_items)} prioritized items in the database")
    
    if not prioritized_items or not isinstance(prioritized_items, list):
        return {"status": "error", "message": "Invalid prioritized items"}
    
    def store_items_operation(connection):
        cursor = connection.cursor()
        stored_count = 0
        
        # Get the current date for the report
        report_date = datetime.now().date()
        
        # Generate filenames for the report (use 'multi' for multi-district reports)
        if district is None:
            # If no district specified, check if all items have the same district
            districts = [item.get('district', '0') for item in prioritized_items]
            unique_districts = set(districts)
            if len(unique_districts) == 1:
                report_district = districts[0]
            else:
                report_district = 'multi'
        else:
            report_district = 'multi' if any(item.get('district', district) != district for item in prioritized_items) else district
        original_filename = f"monthly_report_{report_district}_{report_date.strftime('%Y_%m')}.html"
        revised_filename = f"monthly_report_{report_district}_{report_date.strftime('%Y_%m')}_revised.html"
        
        # Create a record in the reports table first
        cursor.execute("""
            INSERT INTO reports (
                max_items, district, period_type, original_filename, revised_filename,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            ) RETURNING id
        """, (
            len(prioritized_items),
            report_district,
            period_type,
            original_filename,
            revised_filename
        ))
        
        report_id = cursor.fetchone()[0]
        logger.info(f"Created report record with ID: {report_id}")
        
        # Insert each prioritized item
        inserted_ids = []
        for item in prioritized_items:
            # Create metadata dictionary with citywide changes
            metadata = {
                "trend_analysis": item.get("trend_analysis", ""),
                "follow_up": item.get("follow_up", ""),
                "change_type": item.get("change_type", "neutral"),  # Store change_type
                "greendirection": item.get("greendirection"),  # Store greendirection
                "citywide_changes": item.get("citywide_changes", "")  # Store citywide changes
            }
            
            # Create item title from metric name and group
            metric_name = item.get("metric", "")
            metric_id = item.get("metric_id", "Unknown")
            group_value = item.get("group") or "All"  # Handle None values properly
            item_title = f"{metric_name} - {group_value}" if group_value != "All" else metric_name
            
            cursor.execute("""
                INSERT INTO monthly_reporting (
                    report_id, report_date, item_title, metric_name, metric_id, group_value, group_field_name, 
                    period_type, comparison_mean, recent_mean, difference, 
                    percent_change, rationale, explanation, priority, district, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                report_id,
                report_date,
                item_title,
                metric_name,
                metric_id,
                group_value,
                "group", # Default group field name
                period_type,
                item.get("comparison_mean", 0),
                item.get("recent_mean", 0),
                item.get("difference", 0),
                item.get("percent_change", 0),
                item.get("rationale", ""),
                "",  # explanation: will be filled in by generate_explanations
                item.get("priority", 999),
                item.get("district", district),
                json.dumps(metadata)
            ))
            
            inserted_id = cursor.fetchone()[0]
            inserted_ids.append(inserted_id)
            stored_count += 1
            
        connection.commit()
        cursor.close()
        return {"status": "success", "inserted_ids": inserted_ids, "report_id": report_id}
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=store_items_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success":
        return {"status": "success", "inserted_ids": result["result"]["inserted_ids"], "report_id": result["result"]["report_id"]}
    else:
        return {"status": "error", "message": result["message"]}

def generate_explanations(report_ids):
    """
    Step 3: Generate detailed explanations for each prioritized item using the explainer agent.
    Updates the monthly_reporting table with the explanations.
    
    Args:
        report_ids: List of report IDs to explain
        
    Returns:
        Status dictionary
    """
    logger.info(f"Generating explanations for {len(report_ids)} items")
    
    def generate_explanations_operation(connection):
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        successful_explanations = 0
        
        for report_id in report_ids:
            # Get the report item
            cursor.execute("""
                SELECT * FROM monthly_reporting WHERE id = %s
            """, (report_id,))
            
            item = cursor.fetchone()
            if not item:
                logger.warning(f"Report item with ID {report_id} not found")
                continue
            
            # Format the data for the explainer agent
            metric_name = item["metric_name"]
            metric_id = item["metric_id"]  # Get the metric_id from the item
            group_value = item["group_value"]
            recent_mean = item["recent_mean"]
            comparison_mean = item["comparison_mean"]
            difference = item["difference"]
            percent_change = item["percent_change"]
            district = item["district"]
            period_type = item["period_type"]
            
            # Calculate time periods for context - check the report date
            report_date = item.get("report_date") or datetime.now().date()
            if isinstance(report_date, str):
                report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
            
            # Get the previous month's date (the more recent month in the comparison)
            # Using a safer approach to handle edge cases with month days
            previous_month_date = report_date - relativedelta(months=1)
            
            # Get the month before the previous month (the earlier month in the comparison)
            # Using a safer approach to handle edge cases with month days
            comparison_month_date = previous_month_date - relativedelta(months=1)
            
            # Format the month names
            recent_month = previous_month_date.strftime("%B %Y")
            previous_month = comparison_month_date.strftime("%B %Y")
            
            logger.info(f"Comparing data between {previous_month} and {recent_month}")
            
            # Calculate delta if not already in the item
            delta = item.get("difference", recent_mean - comparison_mean)
            
            # Add detailed context about the item for logging
            logger.info(f"Generating explanation for report_id={report_id}, metric={metric_name} (ID: {metric_id}), group={group_value}")
            logger.info(f"  Values: recent={recent_mean}, comparison={comparison_mean}, diff={difference}, percent_change={percent_change}")
            logger.info(f"  Period: {period_type}, District: {district}")
            
            # Create a simple session for the explainer agent
            session_id = f"report_{report_id}"
            session_context = context_variables.copy()
            
            # Create the explainer agent instance
            explainer_agent = create_explainer_agent(session_context)
            
            # Create a simple, direct prompt for the agent
            direction = "increased" if delta > 0 else "decreased"
            percent_change_str = f"{abs(percent_change):.2f}%" if percent_change is not None else "unknown percentage"
            
            prompt = f"""Please explain why the metric '{metric_name}' (ID: {metric_id})  {direction} from {comparison_mean} to {recent_mean} ({percent_change_str}) between {previous_month} and {recent_month} for district {district}.

Use the available tools to research this change and provide a comprehensive explanation that can be included in a monthly newsletter for city residents.
Please share anomalies, groups or data-points that explain a large portion of the difference in the metric.  Your goal is a clear explanation of the change.  

Your response MUST be returned as a properly formatted JSON object with the following fields:
- "explanation": A clear and thorough explanation of what happened (at least 3 paragraphs)
- "trend_analysis": A discussion of how this change fits into longer-term trends
- "charts": A list of chart references to include (if any)

DO NOT include any additional content, headers, or formatting outside of this JSON structure. The data will be extracted directly for use in a report."""
            
            logger.info(f"Prompt for explainer agent: {prompt[:200]}...")
            
            # Process with the agent
            try:
                logger.info(f"Running explainer agent for metric: {metric_id}")
                # Use the new explainer agent's synchronous method with JSON return
                response = explainer_agent.explain_change_sync(prompt, return_json=True)
                
                explanation = ""
                chart_data = None
                explainer_metadata = {}
                
                # Check if response contains content
                if response and response.get('success'):
                    content = response.get('content', '')
                    logger.info(f"Explainer agent response content: {content[:200]}...")
                    
                    # Try to extract JSON data from the response
                    if response.get('parsed_json'):
                        # Use the pre-parsed JSON if available
                        explainer_data = response['parsed_json']
                        logger.info(f"Using pre-parsed JSON data: {json.dumps(explainer_data)[:200]}...")
                        
                        # Extract the specific fields we're looking for
                        if "explanation" in explainer_data:
                            explanation = explainer_data["explanation"]
                            logger.info(f"Extracted explanation from JSON: {explanation[:100]}...")
                            
                        # Store all fields in metadata
                        explainer_metadata = explainer_data
                        
                        # Keep chart data separate
                        if "chart_data" in explainer_data:
                            chart_data = explainer_data["chart_data"]
                        elif "charts" in explainer_data:
                            chart_data = explainer_data["charts"]
                    else:
                        # Fallback to parsing from content
                        try:
                            # Extract JSON content from the response_content
                            # First look for JSON content between ```json and ``` markers
                            json_pattern = r'```json\s*([\s\S]*?)\s*```'
                            json_match = re.search(json_pattern, content)
                            
                            if json_match:
                                json_str = json_match.group(1)
                                logger.info(f"Found JSON content in code block: {json_str[:200]}...")
                                explainer_data = json.loads(json_str)
                            else:
                                # Try to parse the entire response as JSON
                                logger.info("Attempting to parse entire response as JSON")
                                explainer_data = json.loads(content)
                            
                            logger.info(f"Successfully parsed JSON data: {json.dumps(explainer_data)[:200]}...")
                            
                            # Extract the specific fields we're looking for
                            if "explanation" in explainer_data:
                                explanation = explainer_data["explanation"]
                                logger.info(f"Extracted explanation from JSON: {explanation[:100]}...")
                                
                            # Store all fields in metadata
                            explainer_metadata = explainer_data
                            
                            # Keep chart data separate
                            if "chart_data" in explainer_data:
                                chart_data = explainer_data["chart_data"]
                            elif "charts" in explainer_data:
                                chart_data = explainer_data["charts"]
                                
                        except json.JSONDecodeError as json_err:
                            logger.warning(f"Failed to parse JSON from response: {json_err}")
                            logger.warning(f"Response content: {content[:500]}...")
                            # If JSON parsing fails, attempt to extract explanation using existing methods
                            explanation_pattern = r'EXPLANATION:\s*([\s\S]*?)(?:\Z|(?:TREND_ANALYSIS:|CHARTS:))'
                            explanation_match = re.search(explanation_pattern, content)
                            if explanation_match:
                                explanation = explanation_match.group(1).strip()
                                logger.info(f"Extracted explanation using regex: {explanation[:100]}...")
                                
                            # Also try to extract trend analysis
                            trend_pattern = r'TREND_ANALYSIS:\s*([\s\S]*?)(?:\Z|CHARTS:)'
                            trend_match = re.search(trend_pattern, content)
                            if trend_match:
                                explainer_metadata["trend_analysis"] = trend_match.group(1).strip()
                                logger.info(f"Extracted trend analysis using regex: {explainer_metadata['trend_analysis'][:100]}...")
                else:
                    logger.error(f"Explainer agent failed: {response.get('error', 'Unknown error')}")
                    explanation = f"Error generating explanation: {response.get('error', 'Unknown error')}"
            
            except Exception as e:
                logger.error(f"Error running explainer agent: {str(e)}", exc_info=True)
                logger.error(f"Traceback: {traceback.format_exc()}")
                explanation = f"Error generating explanation for {metric_name}: {str(e)}"
            
            # Validate the explanation
            if explanation:
                if len(explanation) < 10:  # If explanation is too short
                    logger.warning(f"Extracted explanation is too short: {explanation}")
                    explanation = ""
                elif len(explanation) > 5000:  # If explanation is too long
                    logger.warning(f"Extracted explanation is too long ({len(explanation)} chars), truncating")
                    explanation = explanation[:5000] + "..."
            
            # If explanation is still empty after all attempts, use a fallback
            if not explanation:
                logger.warning(f"No explanation returned from agent for {metric_name}")
                
                # First try to use the explanation already in the database
                if item.get("explanation"):
                    explanation = item.get("explanation")
                    logger.info(f"Using existing explanation from prioritization step: {explanation[:100]}...")
                else:
                    # If there's no explanation at all, create a generic one
                    explanation = f"Unable to generate an automated explanation for the change in {metric_name} ({percent_change_str}) between {previous_month} and {recent_month}."
                
            # Log the final explanation that will be stored
            logger.info(f"Final explanation to be stored for {metric_name} (length: {len(explanation)}): {explanation[:200]}...")
            
            # Prepare chart_html for storage in the database
            chart_html = None
            if chart_data:
                try:
                    # Extract chart HTML if it exists
                    if hasattr(chart_data, 'content'):
                        chart_html = chart_data.content
                    elif isinstance(chart_data, dict) and 'content' in chart_data:
                        chart_html = chart_data['content']
                    elif isinstance(chart_data, str) and ('<div' in chart_data or '<svg' in chart_data):
                        chart_html = chart_data
                    
                    if chart_html:
                        logger.info(f"Extracted chart HTML (first 100 chars): {str(chart_html)[:100]}...")
                except Exception as e:
                    logger.error(f"Error extracting chart HTML: {e}")
            
            # Get existing metadata from the database and update it
            existing_metadata = item.get("metadata", {}) or {}
            if isinstance(existing_metadata, str):
                try:
                    existing_metadata = json.loads(existing_metadata)
                except:
                    existing_metadata = {}
                    
            # Merge the explainer_metadata into existing_metadata
            if explainer_metadata:
                for key, value in explainer_metadata.items():
                    existing_metadata[key] = value
                logger.info(f"Updated metadata with explainer data: {json.dumps(list(explainer_metadata.keys()))}")
            
            # Update the database with the detailed explanation, chart data, and updated metadata
            if explanation:  # Only update if we have a valid explanation
                try:
                    if chart_html:
                        cursor.execute("""
                            UPDATE monthly_reporting
                            SET explanation = %s,
                                chart_data = %s,
                                metadata = %s
                            WHERE id = %s
                        """, (explanation, json.dumps({"html": str(chart_html)}), json.dumps(existing_metadata), report_id))
                        logger.info(f"Updated explanation, chart data, and metadata for report ID {report_id}")
                    else:
                        cursor.execute("""
                            UPDATE monthly_reporting
                            SET explanation = %s,
                                metadata = %s
                            WHERE id = %s
                        """, (explanation, json.dumps(existing_metadata), report_id))
                        logger.info(f"Updated explanation, and metadata for report ID {report_id}")
                    successful_explanations += 1
                except Exception as db_error:
                    logger.error(f"Database error while updating explanation: {str(db_error)}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"No valid explanation to store for report ID {report_id}")
        
        connection.commit()
        cursor.close()
        
        return successful_explanations
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=generate_explanations_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success":
        logger.info(f"Successfully generated explanations for {result['result']} items")
        return {"status": "success", "message": f"Explanations generated for {result['result']} items"}
    else:
        return {"status": "error", "message": result["message"]}

def generate_monthly_report(report_date=None, district="0", original_filename=None):
    """
    Step 4: Generate the final monthly report with charts and annotations
    
    Args:
        report_date: The date for the report (defaults to current month)
        district: District number
        original_filename: Optional original filename to use instead of generating a new one
        
    Returns:
        Path to the generated report file
    """
    from webChat import client, AGENT_MODEL
    
    logger.info(f"Generating monthly report for district {district}")
    
    # Set report date to current date if not provided
    if not report_date:
        report_date = datetime.now().date()
    
    def generate_report_operation(connection):
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get all report items for this date and district, ordered by priority
        cursor.execute("""
            SELECT *, 
                   metadata->>'anomaly_id' as anomaly_id, -- Extract potential anomaly_id from metadata
                   metadata->>'perplexity_context' as perplexity_context,
                   metadata->'perplexity_response' as perplexity_response_json,
                   metadata->'perplexity_response'->'citations' as perplexity_citations,
                   metadata->>'trend_analysis' as trend_analysis, -- Extract trend_analysis from metadata
                   metadata->>'charts' as charts, -- Extract charts from metadata
                   metadata->>'follow_up' as follow_up, -- Extract follow_up from metadata
                   metric_id -- Placeholder for metric_id, needs proper retrieval logic
            FROM monthly_reporting 
            WHERE report_date = %s AND district = %s
            ORDER BY priority
        """, (report_date, district))
        
        items = cursor.fetchall()
        logger.info(f"Found {len(items)} report items for date {report_date} and district {district}")
        if not items:
            logger.warning(f"No report items found for date {report_date} and district {district}")
            cursor.close()
            return None
        
        # Format the report data
        report_data = []
        for item in items:
            # Check if there's chart data
            chart_html = None
            if item["chart_data"]:
                try:
                    chart_data = item["chart_data"]
                    if isinstance(chart_data, dict) and "html" in chart_data:
                        chart_html = chart_data["html"]
                    elif isinstance(chart_data, str):
                        chart_data_dict = json.loads(chart_data)
                        if "html" in chart_data_dict:
                            chart_html = chart_data_dict["html"]
                except Exception as e:
                    logger.error(f"Error parsing chart data: {e}")
            
            # Attempt to get IDs for chart placeholders
            metric_id = item.get("metric_id") # Placeholder value for now
            anomaly_id = item.get("anomaly_id") # Might be None if not an anomaly item
            
            # Initialize chart placeholders as a list instead of a single string
            chart_placeholders = []
            
            # # Add anomaly chart if available
            # if anomaly_id:
            #     try:
            #         # Validate anomaly_id is an integer
            #         int(anomaly_id)
            #         anomaly_placeholder = f"[CHART:anomaly:{anomaly_id}]"
                    
            #         # Try to get the caption from anomaly details
            #         try:
            #             # Import the get_anomaly_details function from anomalyAnalyzer
            #             from ai.anomalyAnalyzer import get_anomaly_details
                        
            #             # Get anomaly details
            #             context_variables = {}
            #             anomaly_details = get_anomaly_details(context_variables, int(anomaly_id))
                        
            #             # Check if successful and extract caption from metadata
            #             if anomaly_details.get("status") == "success" and anomaly_details.get("anomaly"):
            #                 metadata = anomaly_details.get("anomaly", {}).get("metadata", {})
            #                 caption = metadata.get("caption", "")
                            
            #                 # Append caption to the chart placeholder if available
            #                 if caption:
            #                     anomaly_placeholder = f"{anomaly_placeholder}\n{caption}"
            #                     logger.info(f"Added caption to anomaly chart {anomaly_id}: {caption}")
            #         except Exception as e:
            #             logger.error(f"Error getting anomaly details for caption: {e}")
                    
            #         chart_placeholders.append(anomaly_placeholder)
            #     except (ValueError, TypeError):
            #         anomaly_id = None # Invalid anomaly_id
            
            # # Add time series chart if available
            # if not anomaly_id and metric_id is not None:
            #     try:
            #         # Validate metric_id, district, period_type
            #         int(metric_id)
            #         str(item["district"])
            #         str(item["period_type"])
            #         chart_placeholders.append(f"[CHART:time_series:{metric_id}:{item['district']}:{item['period_type']}]")
            #     except (ValueError, TypeError, KeyError):
            #         metric_id = None # Invalid data
                    
            # Add charts from explainer response
            charts_from_metadata = item.get("charts")
            if charts_from_metadata:
                try:
                    # Check if it's a JSON string
                    if isinstance(charts_from_metadata, str):
                        try:
                            charts_parsed = json.loads(charts_from_metadata)
                            if isinstance(charts_parsed, list):
                                # Add all chart references
                                for chart_ref in charts_parsed:
                                    if isinstance(chart_ref, str):
                                        chart_placeholders.append(chart_ref)
                        except json.JSONDecodeError:
                            # If it's not valid JSON but a single chart reference
                            if charts_from_metadata.startswith("[CHART:"):
                                chart_placeholders.append(charts_from_metadata)
                    # If it's already parsed as JSON list
                    elif isinstance(charts_from_metadata, list):
                        for chart_ref in charts_from_metadata:
                            if isinstance(chart_ref, str):
                                chart_placeholders.append(chart_ref)
                except Exception as e:
                    logger.warning(f"Error parsing charts from metadata: {e}")
                    
            # Remove any duplicates from chart_placeholders
            chart_placeholders = list(dict.fromkeys(chart_placeholders))
            logger.info(f"Collected {len(chart_placeholders)} chart placeholders for {item['metric_name']}")
            
            # Format Perplexity citations if available
            perplexity_citations_text = ""
            citation_map = {}  # Map to store citation numbers for reference matching
            
            # Direct extraction of citations from the database column
            # This is the most reliable way to get the citations
            perplexity_citations = item.get("perplexity_citations")
            logger.info(f"Raw perplexity_citations for {item['metric_name']}: {json.dumps(perplexity_citations)[:300]}...")
            
            # Try to format the citations
            if perplexity_citations:
                try:
                    # Handle string format (needs parsing)
                    if isinstance(perplexity_citations, str):
                        try:
                            perplexity_citations = json.loads(perplexity_citations)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse perplexity_citations as JSON: {perplexity_citations}")
                    
                    # Handle list format
                    if isinstance(perplexity_citations, list) and len(perplexity_citations) > 0:
                        perplexity_citations_text = "\n\nRelevant Sources:\n"
                        for idx, citation in enumerate(perplexity_citations, 1):
                            if isinstance(citation, str):
                                # Simple URL citation
                                citation_map[str(idx)] = citation
                                perplexity_citations_text += f"{idx}. {citation}\n"
                            elif isinstance(citation, dict):
                                # Citation with title and URL
                                title = citation.get("title", "Untitled")
                                url = citation.get("url", citation.get("link", "No URL"))
                                citation_map[str(idx)] = {"title": title, "url": url}
                                perplexity_citations_text += f"{idx}. {title}: {url}\n"
                        
                        logger.info(f"Formatted {len(perplexity_citations)} citations for {item['metric_name']}")
                        logger.info(f"Citation map: {json.dumps(citation_map)}")
                except Exception as e:
                    logger.error(f"Error formatting citations: {e}")
                    logger.error(f"Problematic citations data: {perplexity_citations}")
            
            # If no citations found yet, try a fallback approach
            if not perplexity_citations_text:
                logger.warning(f"No citations found for {item['metric_name']} in primary method, trying fallback approaches")
                
                # Fallback 1: Try to get the full perplexity_response and extract citations
                try:
                    perplexity_response_json = item.get("perplexity_response_json")
                    if perplexity_response_json:
                        # Handle string format (needs parsing)
                        if isinstance(perplexity_response_json, str):
                            try:
                                perplexity_response_json = json.loads(perplexity_response_json)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse perplexity_response_json as JSON")
                        
                        # Extract citations from the response JSON
                        if isinstance(perplexity_response_json, dict) and "citations" in perplexity_response_json:
                            citations = perplexity_response_json["citations"]
                            if isinstance(citations, list) and len(citations) > 0:
                                perplexity_citations_text = "\n\nRelevant Sources:\n"
                                for idx, citation in enumerate(citations, 1):
                                    if isinstance(citation, str):
                                        citation_map[str(idx)] = citation
                                        perplexity_citations_text += f"{idx}. {citation}\n"
                                    elif isinstance(citation, dict):
                                        title = citation.get("title", "Untitled")
                                        url = citation.get("url", citation.get("link", "No URL"))
                                        citation_map[str(idx)] = {"title": title, "url": url}
                                        perplexity_citations_text += f"{idx}. {title}: {url}\n"
                                
                                logger.info(f"Fallback: Extracted {len(citations)} citations from perplexity_response_json")
                except Exception as e:
                    logger.error(f"Error in citation fallback approach: {e}")
            
            # Process the perplexity context to match citation references
            perplexity_context = item.get("perplexity_context", "")
            processed_context = perplexity_context
            
            # Only process if we have both citations and context
            if citation_map and perplexity_context:
                # Look for citation references like [1], [2], etc.
                try:
                    # Find all citation references in the format [n]
                    citation_pattern = r'\[(\d+)\]'
                    citation_refs = re.findall(citation_pattern, perplexity_context)
                    
                    # Sort and deduplicate the citation references
                    unique_refs = sorted(set(citation_refs), key=int)
                    
                    logger.info(f"Found citation references in context: {unique_refs}")
                    
                    # Replace each citation reference with a hyperlink if possible
                    for ref in unique_refs:
                        if ref in citation_map:
                            citation = citation_map[ref]
                            if isinstance(citation, dict):
                                url = citation.get("url", "")
                                if url:
                                    # Replace [n] with [n](url)
                                    processed_context = processed_context.replace(
                                        f"[{ref}]", 
                                        f"[{ref}]({url})"
                                    )
                            elif isinstance(citation, str) and citation.startswith("http"):
                                # If citation is just a URL string
                                processed_context = processed_context.replace(
                                    f"[{ref}]", 
                                    f"[{ref}]({citation})"
                                )
                except Exception as e:
                    logger.error(f"Error processing citation references: {e}")
                    # Keep the original context if there's an error

            # Get the trend analysis from metadata
            trend_analysis = item.get("trend_analysis") or ""
            if not trend_analysis and item["metadata"] and isinstance(item["metadata"], dict):
                trend_analysis = item["metadata"].get("trend_analysis", "")
                
            # Get the follow-up questions from metadata
            follow_up = item.get("follow_up") or ""
            if not follow_up and item["metadata"] and isinstance(item["metadata"], dict):
                follow_up = item["metadata"].get("follow_up", "")

            report_data.append({
                "metric": item["metric_name"],
                "metric_id": item["metric_id"],
                "group": item["group_value"],
                "recent_mean": item["recent_mean"],
                "comparison_mean": item["comparison_mean"],
                "difference": item["difference"],
                "percent_change": item["percent_change"],
                "rationale": item["rationale"],
                "explanation": item["explanation"],
                "report_text": item["report_text"],
                "priority": item["priority"],
                "chart_html": chart_html, # Keep original chart_html if already generated
                "chart_placeholders": chart_placeholders, # Add all chart placeholders as a list
                "trend_analysis": trend_analysis,
                "follow_up": follow_up,
                "perplexity_context": processed_context,  # Use the processed context with linked citations
                "perplexity_citations_text": perplexity_citations_text,
                "citation_map": citation_map,  # Include the citation map for reference
            })
            
            # Only add perplexity_response if it exists
            if 'perplexity_response_json' in locals():
                report_data[-1]["perplexity_response"] = perplexity_response_json
        
        cursor.close()
        
        # Get district name for the report title
        district_name = f"District {district}"
        if district == "0":
            district_name = "Citywide"
            
        # Load officials data
        officials_data = {}
        try:
            officials_path = Path(__file__).parent / 'data' / 'officials.json'
            with open(officials_path, 'r') as f:
                officials_json = json.load(f)
                for official in officials_json.get("officials", []):
                    officials_data[official.get("district")] = official
        except Exception as e:
            logger.error(f"Error loading officials data: {e}")
            
        # Get the district official
        district_official = officials_data.get(district, {})
        official_name = district_official.get("name", "")
        official_role = district_official.get("role", "")
        
        # Create a prompt for generating the report
        report_items_text = ""
        for i, item in enumerate(report_data, 1):
            report_items_text += f"\nITEM {i}: {item['metric']} - {item['group']}\n{item['report_text']}\n\n"

        # Calculate the month for the report title (the more recent month in the comparison)
        if report_date.month == 1:  # January case
            report_month_date = report_date.replace(month=12, year=report_date.year-1)
        else:
            report_month_date = report_date.replace(month=report_date.month-1)
            
        # Format the month name for the report title
        current_month = report_month_date.strftime("%B %Y")
        
        # Load prompt from JSON file
        try:
            prompts = load_prompts()
            prompt_template = prompts['monthly_report']['generate_report']['prompt']
            system_message = prompts['monthly_report']['generate_report']['system']
            
            # Format the prompt with the required variables
            prompt = prompt_template.format(
                district_name=district_name,
                current_month=current_month,
                official_name=official_name,
                official_role=official_role,
                report_items_text=report_items_text
            )
            
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(script_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save the complete prompt to a log file (overwrite for each run)
            prompt_log_path = os.path.join(logs_dir, 'monthly_report_prompt.txt')
            try:
                with open(prompt_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"System Message:\n{system_message}\n\n")
                    f.write(f"User Prompt:\n{prompt}")
                logger.info(f"Saved complete monthly report prompt to {prompt_log_path}")
            except Exception as log_error:
                logger.error(f"Error saving monthly report prompt to log file: {log_error}")
            
            # Log the generated report response (append)
            try:
                with open(prompt_log_path, 'a', encoding='utf-8') as f:
                    f.write('\n\n=== GENERATED REPORT RESPONSE ===\n')
                    # Only try to write report_text if it exists (it doesn't exist yet at this point)
                    # The report_text variable is defined after the API call further down
                    f.write("Will be added after generation")
                    f.write('\n=== END GENERATED REPORT RESPONSE ===\n')
            except Exception as log_error:
                logger.error(f"Error logging generated report response: {log_error}")
        except Exception as e:
            logger.error(f"Error loading prompts from JSON: {e}")
            raise

        # Make API call to generate the report
        logger.info("Making API call to generate report content")
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "system", "content": system_message},
                     {"role": "user", "content": prompt}],
            temperature=0.2
        )

        report_text = response.choices[0].message.content
        logger.info(f"Received report content (length: {len(report_text)})")
        
        # Log the generated report content to a file
        try:
            with open(prompt_log_path, 'a', encoding='utf-8') as f:
                f.write('\n\n=== GENERATED REPORT RESPONSE ===\n')
                f.write(report_text)
                f.write('\n=== END GENERATED REPORT RESPONSE ===\n')
            logger.info(f"Successfully logged generated report content to {prompt_log_path}")
        except Exception as log_error:
            logger.error(f"Error logging generated report content: {log_error}")
        
        # # Once we have the report text, insert any charts if they exist
        # for i, item in enumerate(report_data, 1):
        #     # Try to insert all charts for this item
        #     if item.get('chart_placeholders') and len(item['chart_placeholders']) > 0:
        #         metric_mentions = [
        #             item['metric'],
        #             # Add variations if needed (e.g., removing emojis)
        #             item['metric'].replace("🏠", "Housing").replace("💊", "Drug").replace("🚫", "Business") 
        #         ]
                
        #         # Find all instances where the metric is mentioned
        #         mention_positions = []
        #         for mention in metric_mentions:
        #             pos = 0
        #             while True:
        #                 pos = report_text.find(mention, pos)
        #                 if pos == -1:
        #                     break
        #                 mention_positions.append({'pos': pos, 'len': len(mention)})
        #                 pos += len(mention)
        #         # Sort positions in ascending order
        #         mention_positions.sort(key=lambda x: x['pos'])
        #         # For each chart placeholder, find a suitable insertion point
        #         for idx, chart_placeholder in enumerate(item['chart_placeholders']):
        #             # If we have multiple charts, space them out across mentions
        #             if mention_positions:
        #                 # If we have more charts than mentions, cycle through mentions
        #                 mention_index = idx % len(mention_positions)
        #                 mention_pos = mention_positions[mention_index]['pos']
        #                 mention_len = mention_positions[mention_index]['len']
        #                 # Find the end of the paragraph containing the mention
        #                 paragraph_end = report_text.find("\n\n", mention_pos)
        #                 if paragraph_end != -1:
        #                     insertion_point = paragraph_end + 2 # Insert after the double newline
        #                 else:
        #                     # If no double newline found, try finding the end of the line
        #                     line_end = report_text.find("\n", mention_pos)
        #                     if line_end != -1:
        #                         insertion_point = line_end + 1 # Insert after the newline
        #                     else:
        #                         # If no newline found, just insert after the mention (less ideal)
        #                         insertion_point = mention_pos + mention_len # Use length of the specific mention
        #                 # Insert the chart placeholder
        #                 report_text = report_text[:insertion_point] + f"\n{chart_placeholder}\n\n" + report_text[insertion_point:]
        #                 logger.info(f"Inserted chart placeholder '{chart_placeholder}' after mention of '{item['metric']}' at position {insertion_point}")
        #                 # Update all subsequent mention positions to account for the insertion
        #                 # This needs to be robust: find new positions for remaining original mentions
        #                 temp_report_text = report_text[insertion_point + len(f"\n{chart_placeholder}\n\n"):]
        #                 original_mention_positions = sorted(list(set([mp['pos'] for mp in mention_positions]))) # unique sorted positions
        #                 new_mention_positions = []
        #                 current_scan_offset = 0
        #                 for orig_pos in original_mention_positions:
        #                     if orig_pos <= mention_pos: # Mentions at or before the current one are unaffected relative to start
        #                         new_mention_positions.append({'pos': orig_pos, 'len': mention_len})
        #                     else: # Mentions after the insertion point need recalculation
        #                         new_mention_positions.append({'pos': orig_pos + len(f"\n{chart_placeholder}\n\n"), 'len': mention_len})
        #                 # Deduplicate by 'pos' (keep the first occurrence for each position)
        #                 seen = set()
        #                 deduped = []
        #                 for mp in new_mention_positions:
        #                     if mp['pos'] not in seen:
        #                         deduped.append(mp)
        #                         seen.add(mp['pos'])
        #                 mention_positions = sorted(deduped, key=lambda x: x['pos'])
        #             else:
        #                 # If no mentions found, DO NOT append the chart at the end of the report
        #                 logger.warning(f"No suitable mention found for '{item['metric']}' to insert chart placeholder '{chart_placeholder}'. Placeholder NOT inserted.")
            
        #     # Also insert chart_html if it exists and wasn't already part of the placeholders
        #     # This maintains backward compatibility with existing reports
        #     elif item.get('chart_html'):
        #         # Create the chart HTML content we want to insert
        #         chart_content_to_insert = item['chart_html']
                
        #         # Look for a section that mentions this metric to insert the chart nearby
        #         metric_mentions = [
        #             item['metric'],
        #             # Add variations if needed (e.g., removing emojis)
        #             item['metric'].replace("🏠", "Housing").replace("💊", "Drug").replace("🚫", "Business") 
        #         ]
                
        #         insertion_point = -1
        #         best_mention_len = 0

        #         current_search_offset = 0
        #         temp_mention_positions = []
        #         for mention in metric_mentions:
        #             pos = 0
        #             while True:
        #                 pos = report_text.find(mention, current_search_offset)
        #                 if pos == -1:
        #                     break
        #                 temp_mention_positions.append({'pos': pos, 'len': len(mention)})
        #                 pos += len(mention) # Continue searching after this mention
                
        #         if temp_mention_positions:
        #             # Prefer inserting after the first found mention
        #             first_mention = min(temp_mention_positions, key=lambda x: x['pos'])
        #             mention_pos = first_mention['pos']
        #             mention_len = first_mention['len']

        #             paragraph_end = report_text.find("\n\n", mention_pos)
        #             if paragraph_end != -1:
        #                 insertion_point = paragraph_end + 2 
        #             else:
        #                 line_end = report_text.find("\n", mention_pos)
        #                 if line_end != -1:
        #                     insertion_point = line_end + 1
        #                 else:
        #                     insertion_point = mention_pos + mention_len
                

        #         if insertion_point != -1:
        #             # Insert the chart HTML at the determined point
        #             report_text = report_text[:insertion_point] + f"\n{chart_content_to_insert}\n\n" + report_text[insertion_point:]
        #             logger.info(f"Inserted chart HTML for '{item['metric']}' after its mention.")
        #         else:
        #             # If we couldn't find a suitable mention, DO NOT append the chart at the end
        #             logger.warning(f"No suitable mention found for '{item['metric']}' to insert chart HTML. Chart HTML NOT inserted.")
        
        # Create directory for reports if it doesn't exist
        reports_dir = Path(__file__).parent / 'output' / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the report to a file as HTML
        if original_filename:
            # Use the provided original filename
            report_filename = original_filename
        else:
            # Generate a new filename based on current date
            report_filename = f"monthly_report_{district}_{report_date.strftime('%Y_%m')}.html"
        report_path = reports_dir / report_filename
        
        # Write the report directly without adding HTML structure
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Monthly newsletter saved to {report_path}")
        return str(report_path)
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=generate_report_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success" and result["result"]:
        return {"status": "success", "report_path": result["result"]}
    elif result["status"] == "success" and not result["result"]:
        return {"status": "error", "message": "No report items found"}
    else:
        return {"status": "error", "message": result["message"]}

def proofread_and_revise_report(report_path):
    """
    Step 5: Proofread and revise the monthly newsletter
    
    Args:
        report_path: Path to the newsletter file to proofread
        
    Returns:
        Path to the revised newsletter file
    """
    from webChat import client, AGENT_MODEL
    
    logger.info(f"Proofreading and revising newsletter at {report_path}")
    
    try:
        # Read the newsletter
        with open(report_path, 'r', encoding='utf-8') as f:
            newsletter_text = f.read()
        
        # Load prompt from JSON file
        prompts = load_prompts()
        prompt_template = prompts['monthly_report']['proofread']['prompt']
        system_message = prompts['monthly_report']['proofread']['system']
        
        # Format the prompt with the newsletter text
        prompt = prompt_template.format(
            newsletter_text=newsletter_text
        )

        # Add explicit instruction to return detailed JSON
        json_instruction = "\n\nPlease format your response as a detailed JSON object with the following structure:\n```json\n{\n  \"newsletter\": \"[full HTML content of the revised newsletter]\",\n  \"proofread_feedback\": \"[comprehensive, detailed summary of changes made and suggestions - please be thorough here]\",\n  \"headlines\": [\"headline 1\", \"headline 2\", \"headline 3\", \"headline 4\", \"headline 5\"]\n}\n```\nPlease provide extensive, detailed feedback in the proofread_feedback section, including specific examples of what you changed and why. I'd like at least 500 words of feedback."

        prompt += json_instruction

        # Use the AGENT_MODEL directly with higher max_tokens
        logger.info("Using AGENT_MODEL for proofreading with increased token limit")
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000,  # Request more tokens in the response
            response_format={"type": "json_object"}  # Expect JSON response
        )
        response_content = response.choices[0].message.content
        
        # Save the raw response for debugging
        debug_path = Path(report_path).parent / f"{Path(report_path).stem}_raw_response.txt"
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(response_content)
        logger.info(f"Saved raw proofread response to {debug_path}")
        
        # Parse the JSON response
        try:
            proofread_data = json.loads(response_content)
            revised_newsletter_content = proofread_data.get("newsletter")
            proofread_feedback = proofread_data.get("proofread_feedback")
            headlines = proofread_data.get("headlines") # Extract headlines
            
            if not revised_newsletter_content:
                logger.error("No 'newsletter' content found in proofread response.")
                return {"status": "error", "message": "Proofread response missing 'newsletter' content."}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse proofread response as JSON: {e}")
            logger.error(f"Raw response: {response_content[:500]}...")
            # Attempt to recover newsletter, headlines, and feedback using regex
            # Improved regex patterns to better handle incomplete JSON
            newsletter_match = re.search(r'"newsletter"\s*:\s*"(.*?)("|}|\Z)', response_content, re.DOTALL)
            if newsletter_match:
                # Get the content but remove any trailing characters if we matched end of string
                newsletter_content = newsletter_match.group(1)
                if newsletter_match.group(2) != '"':
                    # If we didn't match the closing quote, we need to handle a truncated string
                    logger.warning("Newsletter content appears to be truncated - trying to recover")
                    # Add closing quote if missing
                    newsletter_content += '"'
            else:
                newsletter_content = None
                
            feedback_match = re.search(r'"proofread_feedback"\s*:\s*"(.*?)("|}|\Z)', response_content, re.DOTALL)
            if feedback_match:
                # Get the content but remove any trailing characters if we matched end of string
                feedback_content = feedback_match.group(1)
                if feedback_match.group(2) != '"':
                    # If we didn't match the closing quote, we need to handle a truncated string
                    logger.warning("Feedback content appears to be truncated - trying to recover")
            else:
                feedback_content = None
                
            headlines_match = re.search(r'"headlines"\s*:\s*(\[.*?\])', response_content, re.DOTALL)
            
            revised_newsletter_content = newsletter_content
            proofread_feedback = feedback_content
            
            try:
                headlines = json.loads(headlines_match.group(1)) if headlines_match else None
            except Exception as headlines_error:
                logger.error(f"Error parsing headlines: {headlines_error}")
                headlines = None
                
            logger.warning("Recovered fields from malformed JSON: newsletter=%s, feedback=%s, headlines=%s", 
                          bool(revised_newsletter_content), bool(proofread_feedback), bool(headlines))
            
            if revised_newsletter_content:
                # Save the revised newsletter content
                report_path_obj = Path(report_path)
                revised_filename = f"{report_path_obj.stem}_revised{report_path_obj.suffix}"
                revised_path = report_path_obj.parent / revised_filename
                
                # The newsletter content might have escaped quotes we need to unescape
                try:
                    # Try to handle common JSON escape sequences
                    revised_newsletter_content = revised_newsletter_content.replace('\\"', '"')
                    revised_newsletter_content = revised_newsletter_content.replace('\\n', '\n')
                    revised_newsletter_content = revised_newsletter_content.replace('\\r', '\r')
                    revised_newsletter_content = revised_newsletter_content.replace('\\t', '\t')
                    revised_newsletter_content = revised_newsletter_content.replace('\\\\', '\\')
                except Exception as unescape_error:
                    logger.error(f"Error unescaping content: {unescape_error}")
                
                with open(revised_path, 'w', encoding='utf-8') as f:
                    f.write(revised_newsletter_content)
                logger.info(f"Revised newsletter saved to {revised_path} (recovered from malformed JSON)")
                return {"status": "partial", "revised_report_path": str(revised_path), "proofread_feedback": proofread_feedback, "headlines": headlines, "message": "Recovered from malformed JSON."}
            else:
                return {"status": "error", "message": f"Failed to parse proofread JSON response: {e}"}
        
        # Save the revised newsletter content
        report_path_obj = Path(report_path)
        revised_filename = f"{report_path_obj.stem}_revised{report_path_obj.suffix}"
        revised_path = report_path_obj.parent / revised_filename
        
        with open(revised_path, 'w', encoding='utf-8') as f:
            f.write(revised_newsletter_content)
        
        # Expand chart references in the revised report
        logger.info(f"Expanding chart references in revised report: {revised_path}")
        expand_result = expand_chart_references(revised_path)
        if not expand_result:
            logger.warning("Failed to expand chart references in the revised report")

        # Update the reports table with the revised filename and proofread_feedback
        def update_report_record_operation(connection):
            cursor = connection.cursor()
            original_filename = report_path_obj.name # Assuming original filename is the name part of report_path
            
            # Find the report_id based on the original filename
            # This assumes original_filename in the reports table matches report_path_obj.name
            cursor.execute("""
                SELECT id FROM reports WHERE original_filename = %s ORDER BY created_at DESC LIMIT 1
            """, (original_filename,))
            report_record = cursor.fetchone()
            
            if not report_record:
                logger.error(f"Could not find report record for original_filename: {original_filename}")
                return False
            
            report_id = report_record[0]
            
            update_query = """
                UPDATE reports
                SET revised_filename = %s, proofread_feedback = %s, headlines = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            cursor.execute(update_query, (revised_filename, proofread_feedback, json.dumps(headlines) if headlines else None, report_id))
            connection.commit()
            logger.info(f"Updated report ID {report_id} with revised filename, proofread feedback, and headlines.")
            return True

        update_db_result = execute_with_connection(
            operation=update_report_record_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )

        if update_db_result["status"] != "success" or not update_db_result["result"]:
            logger.error(f"Failed to update report table: {update_db_result.get('message')}")
            # Continue, but log the error. The file is saved.

        logger.info(f"Revised newsletter saved to {revised_path}")
        return {"status": "success", "revised_report_path": str(revised_path)}
        
    except Exception as e:
        error_msg = f"Error in proofread_and_revise_report: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def get_perplexity_context(report_items):
    """
    Use the Perplexity API to get additional context and information about 
    each individual item in the report. Enriches each item with relevant real-time information.
    
    Args:
        report_items: List of report items with metric information
        
    Returns:
        Dictionary with status and the enriched report items with context
    """
    logger.info(f"Getting additional context from Perplexity API for {len(report_items)} report items")
    
    # Check if Perplexity API key is available
    if not PERPLEXITY_API_KEY:
        logger.warning("No Perplexity API key available. Skipping context enrichment.")
        return {"status": "error", "message": "No Perplexity API key available"}
    
    try:
        # Load prompt from prompts.json
        prompts = load_prompts()
        prompt_template = prompts['monthly_report']['context_enrichment']['prompt']
        system_message = prompts['monthly_report']['context_enrichment']['system']
        
        # Process each item individually
        for item in report_items:
            metric_name = item.get("metric")
            group_value = item.get("group")
            recent_mean = item.get("recent_mean")
            comparison_mean = item.get("comparison_mean")
            percent_change = item.get("percent_change")
            explanation = item.get("explanation", "")
            report_text = item.get("report_text", "")
            
            # Prepare item-specific context for the prompt
            item_context = f"""
Metric: {metric_name}
Group: {group_value}
Recent Value: {recent_mean}
Previous Value: {comparison_mean}
Percent Change: {percent_change:+.2f}%
Explanation: {explanation}
Detailed Analysis: {report_text}
"""

            # Format the prompt with the item context
            prompt = prompt_template.format(
                text_content=item_context
            )
            
            # log the prompt
            logger.info(f"Prompt for Perplexity API:\nSystem: {system_message}\nUser: {prompt}")
            
            # Make the API call to Perplexity
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}"
            }
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Make the API request
            logger.info(f"Sending request to Perplexity API for {metric_name}")
            response = requests.post(url, json=payload, headers=headers)
            
            # Check if the request was successful
            if response.status_code == 200:
                logger.info(f"Successfully received response from Perplexity API for {metric_name}")
                result = response.json()
                
                # Log the raw response for debugging
                logger.info(f"Raw Perplexity response: {json.dumps(result)[:1000]}...")
                
                # Extract the content from the response
                context_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not context_content:
                    logger.warning(f"Received empty content from Perplexity API for {metric_name}")
                    continue
                
                # Initialize metadata if it doesn't exist
                if not item.get("metadata"):
                    item["metadata"] = {}
                
                # Store the context content
                item["metadata"]["perplexity_context"] = context_content
                
                # Store the complete response for citations and other data
                # Extract the relevant parts from the response to avoid storing unnecessary data
                perplexity_response = {}
                
                # FIXED: Extract citations from the top level of the response (not from message)
                if "citations" in result:
                    perplexity_response["citations"] = result["citations"]
                    logger.info(f"Found top-level citations: {json.dumps(result['citations'])}")
                else:
                    logger.info("No top-level citations found in Perplexity response")
                    
                    # Fallback: Check if citations are in the message
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        logger.info(f"Message from Perplexity: {json.dumps(message)[:1000]}...")
                        
                        if "citations" in message:
                            perplexity_response["citations"] = message["citations"]
                            logger.info(f"Found message-level citations: {json.dumps(message['citations'])}")
                        else:
                            logger.info("No message-level citations found in Perplexity response")
                
                # Store other relevant fields from the top level
                for field in ["links", "search_queries", "attachments", "tool_calls"]:
                    if field in result:
                        perplexity_response[field] = result[field]
                        logger.info(f"Found top-level {field} in Perplexity response")
                
                # Save the full response data
                if perplexity_response:
                    item["metadata"]["perplexity_response"] = perplexity_response
                    logger.info(f"Added Perplexity response data to {metric_name}: {json.dumps(perplexity_response)}")
                else:
                    logger.warning(f"No perplexity_response data found for {metric_name}")
                
                logger.info(f"Added Perplexity context to {metric_name} (length: {len(context_content)})")
            else:
                logger.error(f"Perplexity API error for {metric_name}: {response.status_code} - {response.text}")
        
        logger.info("Successfully retrieved additional context from Perplexity API for all items")
        return {
            "status": "success", 
            "report_items": report_items
        }
        
    except Exception as e:
        error_msg = f"Error in get_perplexity_context: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def run_monthly_report_process(district="0", period_type="month", max_report_items=10):
    """
    Run the complete monthly newsletter generation process
    
    Args:
        district: District number (0 for citywide)
        period_type: Time period type (month, quarter, year)
        max_report_items: Maximum number of items to include in the newsletter
        
    Returns:
        Status dictionary with newsletter path
    """
    logger.info(f"Starting monthly newsletter process for district {district}")
    
    try:
        # Step 0: Initialize the monthly_reporting table
        logger.info("Step 0: Initializing monthly_reporting table")
        init_result = initialize_monthly_reporting_table()
        if not init_result:
            return {"status": "error", "message": "Failed to initialize monthly_reporting table"}
        
        # Step 1: Select deltas to discuss
        logger.info("Step 1: Selecting deltas to discuss")
        deltas = select_deltas_to_discuss(period_type=period_type, district=district)
        if deltas.get("status") != "success":
            return deltas
            
        # Step 2: Prioritize deltas and get explanations
        logger.info("Step 2: Prioritizing deltas")
        prioritized = prioritize_deltas(deltas, max_items=max_report_items)
        if prioritized.get("status") != "success":
            return prioritized
        
        # Store prioritized items in the database
        logger.info("Step 2.5: Storing prioritized items")
        store_result = store_prioritized_items(
            prioritized.get("prioritized_items", []),
            period_type=period_type,
            district=district
        )
        if store_result.get("status") != "success":
            return store_result
            
        # Step 3: Generate detailed explanations
        logger.info("Step 3: Generating explanations")
        explanation_result = generate_explanations(store_result.get("inserted_ids", []))
        if explanation_result.get("status") != "success":
            return explanation_result
            
        # Get the report data for context enrichment
        def get_report_items_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Get all report items for this district, ordered by priority
            cursor.execute("""
                SELECT mr.*, r.id as report_id 
                FROM monthly_reporting mr
                JOIN reports r ON mr.report_id = r.id
                WHERE r.district = %s
                ORDER BY mr.priority
            """, (district,))
            
            items = cursor.fetchall()
            
            # Format the report data
            report_data = []
            for item in items:
                report_data.append({
                    "id": item["id"],
                    "metric": item["metric_name"],
                    "metric_id": item["metric_id"],
                    "group": item["group_value"],
                    "recent_mean": item["recent_mean"],
                    "comparison_mean": item["comparison_mean"],
                    "difference": item["difference"],
                    "percent_change": item["percent_change"],
                    "rationale": item["rationale"],
                    "explanation": item["explanation"],
                    "report_text": item["report_text"],
                    "priority": item["priority"],
                    "metadata": item["metadata"] if item["metadata"] else {}
                })
            
            cursor.close()
            return report_data
        
        # Step 4: Get report items for context enrichment
        logger.info("Step 4: Getting report items for Perplexity context enrichment")
        result = execute_with_connection(
            operation=get_report_items_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] != "success":
            logger.warning(f"Failed to get report items for context enrichment: {result['message']}")
            report_items = []
        else:
            report_items = result["result"]
            
        # Step 5: Get additional context for each report item
        if report_items:
            logger.info("Step 5: Getting additional context from Perplexity API for each report item")
            context_result = get_perplexity_context(report_items)
            
            if context_result.get("status") == "success":
                # Update the items in the database with the new context
                def update_items_with_context_operation(connection):
                    cursor = connection.cursor()
                    updated_count = 0
                    
                    for item in context_result.get("report_items", []):
                        metric_name = item["metric"]
                        group_value = item["group"]
                        
                        if "perplexity_context" in item.get("metadata", {}):
                            perplexity_context = item["metadata"]["perplexity_context"]
                            logger.info(f"Updating perplexity_context for {metric_name} - {group_value} (length: {len(perplexity_context)})")
                            
                            # Find the item in the database by metric_name and group_value
                            cursor.execute("""
                                UPDATE monthly_reporting
                                SET metadata = jsonb_set(
                                    CASE WHEN metadata IS NULL THEN '{}'::jsonb ELSE metadata END, 
                                    '{perplexity_context}', 
                                    %s::jsonb
                                )
                                WHERE metric_name = %s AND group_value = %s AND district = %s
                                RETURNING id
                            """, (
                                json.dumps(perplexity_context),
                                metric_name,
                                group_value,
                                district
                            ))
                            updated_ids = cursor.fetchall()
                            updated_count += len(updated_ids)
                            logger.info(f"Updated {len(updated_ids)} records for perplexity_context ({metric_name} - {group_value})")
                        
                        # Also update perplexity_response if available
                        if "perplexity_response" in item.get("metadata", {}):
                            perplexity_response = item["metadata"]["perplexity_response"]
                            logger.info(f"Updating perplexity_response for {metric_name} - {group_value}: {json.dumps(perplexity_response)[:100]}...")
                            
                            try:
                                # FIXED: Ensure perplexity_response is properly serialized to JSON as a string
                                # PostgreSQL jsonb_set expects a string that can be parsed as JSON
                                perplexity_response_json = json.dumps(perplexity_response)
                                
                                # Log the JSON string we're using
                                logger.info(f"Serialized perplexity_response JSON: {perplexity_response_json[:100]}...")
                                
                                # Here's the corrected SQL query:
                                # - We need to cast the JSON string to jsonb using ::jsonb
                                # - We're using the ? operator to check if a key exists in the JSONB
                                cursor.execute("""
                                    UPDATE monthly_reporting
                                    SET metadata = jsonb_set(
                                        CASE WHEN metadata IS NULL THEN '{}'::jsonb ELSE metadata END, 
                                        '{perplexity_response}', 
                                        %s::jsonb
                                    )
                                    WHERE metric_name = %s AND group_value = %s AND district = %s
                                    RETURNING id
                                """, (
                                    perplexity_response_json,
                                    metric_name,
                                    group_value,
                                    district
                                ))
                                response_updated_ids = cursor.fetchall()
                                logger.info(f"Updated {len(response_updated_ids)} records for perplexity_response ({metric_name} - {group_value})")
                                
                                # Verify this specific update worked
                                cursor.execute("""
                                    SELECT metadata->'perplexity_response' 
                                    FROM monthly_reporting 
                                    WHERE metric_name = %s AND group_value = %s AND district = %s
                                    LIMIT 1
                                """, (metric_name, group_value, district))
                                result = cursor.fetchone()
                                if result and result[0]:
                                    logger.info(f"Verification successful! perplexity_response was stored for {metric_name}: {json.dumps(result[0])[:100]}...")
                                else:
                                    logger.warning(f"Verification failed! perplexity_response was not stored for {metric_name}")
                            except Exception as e:
                                logger.error(f"Error updating perplexity_response: {str(e)}")
                                logger.error(f"Problematic perplexity_response: {perplexity_response}")
                                
                                # Try a direct update as a fallback (update the entire metadata)
                                try:
                                    logger.info("Attempting fallback update method for perplexity_response...")
                                    
                                    # First get the current metadata
                                    cursor.execute("""
                                        SELECT metadata FROM monthly_reporting
                                        WHERE metric_name = %s AND group_value = %s AND district = %s
                                        LIMIT 1
                                    """, (metric_name, group_value, district))
                                    
                                    current_metadata = cursor.fetchone()[0] or {}
                                    if not isinstance(current_metadata, dict):
                                        current_metadata = {}
                                    
                                    # Update the metadata with the new perplexity_response
                                    current_metadata['perplexity_response'] = perplexity_response
                                    
                                    # Update the entire metadata object
                                    cursor.execute("""
                                        UPDATE monthly_reporting
                                        SET metadata = %s::jsonb
                                        WHERE metric_name = %s AND group_value = %s AND district = %s
                                        RETURNING id
                                    """, (
                                        json.dumps(current_metadata),
                                        metric_name,
                                        group_value,
                                        district
                                    ))
                                    
                                    fallback_updated_ids = cursor.fetchall()
                                    logger.info(f"Fallback update successful for {len(fallback_updated_ids)} records")
                                except Exception as fallback_error:
                                    logger.error(f"Fallback update also failed: {str(fallback_error)}")
                    
                    # Verify the updates worked
                    cursor.execute("""
                        SELECT COUNT(*) FROM monthly_reporting 
                        WHERE district = %s AND metadata ? 'perplexity_context'
                    """, (district,))
                    context_count = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT COUNT(*) FROM monthly_reporting 
                        WHERE district = %s AND metadata ? 'perplexity_response'
                    """, (district,))
                    response_count = cursor.fetchone()[0]
                    
                    logger.info(f"Verification counts: {context_count} records with perplexity_context, {response_count} with perplexity_response")
                    
                    connection.commit()
                    cursor.close()
                    return updated_count
                
                # Update the items in the database
                logger.info("Step 5.5: Updating database with Perplexity context")
                update_result = execute_with_connection(
                    operation=update_items_with_context_operation,
                    db_host=DB_HOST,
                    db_port=DB_PORT,
                    db_name=DB_NAME,
                    db_user=DB_USER,
                    db_password=DB_PASSWORD
                )
                
                if update_result["status"] == "success":
                    logger.info(f"Updated {update_result['result']} items with Perplexity context")
                else:
                    logger.warning(f"Failed to update items with Perplexity context: {update_result['message']}")
            else:
                logger.warning(f"Failed to get additional context: {context_result.get('message')}")
                # Don't fail the process if context retrieval fails
            
        # Step 6: Generate final report_text for each item
        logger.info("Step 6: Generating final report_text for each item")
        report_item_ids = [item['id'] for item in report_items if 'id' in item]
        if report_item_ids:
            report_text_result = generate_report_text(
                report_item_ids,
                execute_with_connection,
                load_prompts,
                AGENT_MODEL,
                client,
                logger
            )
            if report_text_result.get("status") != "success":
                logger.warning(f"Failed to generate report_text: {report_text_result.get('message')}")

        # Step 7: Generate the monthly newsletter (with enriched context already in the database)
        logger.info("Step 7: Generating monthly newsletter")
        
        # Get the original filename from the database for this district
        def get_original_filename_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Get the most recent report for this district
            cursor.execute("""
                SELECT original_filename 
                FROM reports 
                WHERE district = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (district,))
            
            result = cursor.fetchone()
            cursor.close()
            return result['original_filename'] if result else None
        
        # Get the original filename
        filename_result = execute_with_connection(
            operation=get_original_filename_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        original_filename = None
        if filename_result["status"] == "success" and filename_result["result"]:
            original_filename = filename_result["result"]
            logger.info(f"Using original filename: {original_filename}")
        else:
            logger.info("No original filename found, will generate new filename")
        
        newsletter_result = generate_monthly_report(district=district, original_filename=original_filename)
        if newsletter_result.get("status") != "success":
            return newsletter_result
            
        # Step 8: Proofreading and revising the newsletter
        logger.info("Step 8: Proofreading and revising newsletter")
        revised_result = proofread_and_revise_report(newsletter_result.get("report_path"))
        if revised_result.get("status") != "success":
            return revised_result
        
        # Use the revised newsletter path
        revised_newsletter_path = revised_result.get("revised_report_path")
        # Step 9: Expand chart references in the revised newsletter
        logger.info("Step 9: Expanding chart references in the revised newsletter")
        expand_result = expand_chart_references(revised_newsletter_path)
        if not expand_result:
            logger.warning("Failed to expand chart references in the revised newsletter")
        
        # Step 10: Generate an email-compatible version of the newsletter
        logger.info("Step 10: Generating email-compatible version of newsletter")
        email_result = None
        if revised_newsletter_path:
            email_result = generate_email_compatible_report(revised_newsletter_path)
            if email_result:
                logger.info(f"Generated email-compatible newsletter at {email_result}")
            else:
                logger.warning("Failed to generate email-compatible newsletter")
            
        # Update the district's top_level.json with the revised report filename
        try:
            # Get the district directory path
            script_dir = Path(__file__).parent
            dashboard_dir = script_dir / 'output' / 'dashboard'
            district_dir = dashboard_dir / district
            
            # Read the current top_level.json
            top_level_file = district_dir / 'top_level.json'
            if top_level_file.exists():
                with open(top_level_file, 'r', encoding='utf-8') as f:
                    top_level_data = json.load(f)
                
                # Add the revised report filename
                top_level_data['monthly_report'] = Path(revised_newsletter_path).name
                
                # Save the updated top_level.json
                with open(top_level_file, 'w', encoding='utf-8') as f:
                    json.dump(top_level_data, f, indent=2)
                logger.info(f"Updated top_level.json for district {district} with revised report filename")
            else:
                logger.warning(f"top_level.json not found for district {district}")
        except Exception as e:
            logger.error(f"Error updating top_level.json: {str(e)}")
            # Don't fail the process if this update fails
            
        logger.info("Monthly newsletter process completed successfully")
        return {
            "status": "success",
            "newsletter_path": newsletter_result.get("report_path"),
            "revised_newsletter_path": revised_newsletter_path,
            "email_newsletter_path": email_result
         }
        
    except Exception as e:
        error_msg = f"Error in run_monthly_report_process: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def get_monthly_reports_list():
    """
    Get a list of all monthly newsletters from the database.
    
    Returns:
        List of newsletter objects with id, report_date, district, filename, etc.
    """
    logger.info("Getting list of monthly newsletters from database")
    
    def get_reports_operation(connection):
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the reports table
        cursor.execute("""
            SELECT id, district, period_type, max_items, original_filename, revised_filename, 
                   created_at, updated_at, published_url
            FROM reports
            ORDER BY created_at DESC
        """)
        
        reports = cursor.fetchall()
        
        # Format the reports
        formatted_reports = []
        for report in reports:
            # Format district name
            district = report['district']
            district_name = f"District {district}"
            if district == "0":
                district_name = "Citywide"
            
            # Extract date from filename (format: monthly_report_0_2025_04.html or manual_report_20250803_094231.html)
            filename = report['original_filename']
            parts = filename.split('_')
            
            if len(parts) >= 4:
                # Try the original format first (monthly_report_0_2025_04.html)
                if parts[0] == 'monthly' and parts[1] == 'report':
                    year = parts[3]
                    month = parts[4].split('.')[0]  # Remove .html extension
                    report_date = f"{year}-{month}-01T00:00:00"  # ISO format date
                # Try the manual format (manual_report_20250803_094231.html)
                elif parts[0] == 'manual' and parts[1] == 'report':
                    # Extract date from manual_report_20250803_094231.html
                    date_part = parts[2]  # 20250803
                    if len(date_part) == 8:  # YYYYMMDD format
                        year = date_part[:4]
                        month = date_part[4:6]
                        day = date_part[6:8]
                        report_date = f"{year}-{month}-{day}T00:00:00"  # ISO format date
                    else:
                        # Fallback to created_at if date parsing fails
                        report_date = report['created_at'].isoformat()
                else:
                    # Fallback to created_at if filename doesn't match expected format
                    report_date = report['created_at'].isoformat()
            else:
                # Fallback to created_at if filename doesn't match expected format
                report_date = report['created_at'].isoformat()
            
            # Count the number of items for this report
            cursor.execute("""
                SELECT COUNT(*) FROM monthly_reporting WHERE report_id = %s
            """, (report['id'],))
            
            item_count = cursor.fetchone()[0]
            
            # Fetch the rationale of the highest-priority item (lowest priority value)
            cursor.execute("""
                SELECT rationale FROM monthly_reporting WHERE report_id = %s ORDER BY priority ASC LIMIT 1
            """, (report['id'],))
            rationale_row = cursor.fetchone()
            rationale = rationale_row[0] if rationale_row else None

            # Create a report object
            formatted_reports.append({
                "id": report['id'],
                "report_date": report_date,
                "district": district,
                "district_name": district_name,
                "period_type": report['period_type'],
                "max_items": report['max_items'],
                "item_count": item_count,
                "rationale": rationale,
                "original_filename": report['original_filename'],
                "revised_filename": report['revised_filename'],
                "published_url": report['published_url'],
                "created_at": report['created_at'].isoformat(),
                "updated_at": report['updated_at'].isoformat()
            })
        
        cursor.close()
        return formatted_reports
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=get_reports_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success":
        return result["result"]
    else:
        logger.error(f"Error getting monthly reports: {result['message']}")
        return []

def delete_monthly_report(report_id):
    """
    Delete a specific monthly newsletter file and its database records.
    
    Args:
        report_id: The ID of the newsletter to delete (index in the list)
        
    Returns:
        Status dictionary
    """
    logger.info(f"Deleting monthly newsletter with ID {report_id}")
    
    def delete_report_operation(connection):
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get the report details
        cursor.execute("""
            SELECT id, original_filename, revised_filename
            FROM reports
            WHERE id = %s
        """, (report_id,))
        
        report = cursor.fetchone()
        if not report:
            cursor.close()
            return None
        
        # Delete records from monthly_reporting table
        cursor.execute("""
            DELETE FROM monthly_reporting
            WHERE report_id = %s
        """, (report_id,))
        
        # Delete record from reports table
        cursor.execute("""
            DELETE FROM reports
            WHERE id = %s
        """, (report_id,))
        
        connection.commit()
        cursor.close()
        
        return dict(report)
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=delete_report_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] != "success":
        return {"status": "error", "message": result["message"]}
    
    if not result["result"]:
        return {"status": "error", "message": f"Report with ID {report_id} not found in database"}
    
    # Now delete the files
    try:
        report = result["result"]
        reports_dir = Path(__file__).parent / 'output' / 'reports'
        
        # Check if the directory exists
        if not reports_dir.exists():
            logger.warning(f"Reports directory does not exist: {reports_dir}")
            return {"status": "error", "message": "Reports directory not found"}
        
        # Delete original file
        original_path = reports_dir / report['original_filename']
        if original_path.exists():
            original_path.unlink()
            logger.info(f"Deleted original newsletter file: {original_path}")
        
        # Delete revised file if it exists
        if report['revised_filename']:
            revised_path = reports_dir / report['revised_filename']
            if revised_path.exists():
                revised_path.unlink()
                logger.info(f"Deleted revised newsletter file: {revised_path}")
        
        return {"status": "success", "message": f"Newsletter {report['original_filename']} and its database records deleted successfully"}
    except Exception as e:
        error_msg = f"Error deleting newsletter files: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def generate_chart_image(chart_type, params, output_dir=None):
    """
    Generate a static image of a chart for email compatibility.
    
    Args:
        chart_type: Type of chart ('time_series' or 'anomaly')
        params: Parameters for the chart (metric_id, district, etc.)
        output_dir: Directory to save the image (defaults to a temporary directory)
        
    Returns:
        Path to the generated image file
    """
    logger.info(f"Generating static image for {chart_type} chart with params: {params}")
    
    try:
        # Create output directory if not provided
        if not output_dir:
            output_dir = Path(__file__).parent / 'output' / 'charts'
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        filename = f"{chart_type}_{timestamp}.png"
        output_path = output_dir / filename
        
        # Construct the URL for the chart
        if chart_type == 'time_series':
            url = f"/backend/time-series-chart?metric_id={params.get('metric_id', 1)}&district={params.get('district', 0)}&period_type={params.get('period_type', 'year')}#chart-section"
        elif chart_type == 'anomaly':
            url = f"/anomaly-analyzer/anomaly-chart?id={params.get('id', 27338)}#chart-section"
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Use a headless browser service to capture the chart as an image
        # This could be implemented using Selenium, Playwright, or a service like Browserless
        # For this example, we'll use a simple approach with requests and PIL
        
        # Make a request to the chart URL
        full_url = f"{API_BASE_URL}{url}"  # Only use API_BASE_URL for server-side requests
        response = requests.get(full_url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch chart from {full_url}: {response.status_code}")
            return None
        
        # For a real implementation, you would use a headless browser to render the page
        # and capture the chart element as an image
        # For this example, we'll create a placeholder image
        
        # Create a placeholder image (in a real implementation, this would be the actual chart)
        img = Image.new('RGB', (800, 450), color=(255, 255, 255))
        
        # Save the image
        img.save(output_path)
        
        logger.info(f"Generated chart image at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating chart image: {str(e)}", exc_info=True)
        return None

def generate_email_compatible_report(report_path, output_path=None):
    """
    Generate an email-compatible version of the report by replacing iframes with URLs.
    Args:
        report_path: Path to the original report HTML file
        output_path: Path to save the email-compatible report (defaults to report_path with _email suffix)
    Returns:
        Path to the email-compatible report
    """
    logger.info(f"Generating email-compatible version of report: {report_path}")
    try:
        # Set default output path if not provided
        if not output_path:
            report_path_obj = Path(report_path)
            output_path = report_path_obj.parent / f"{report_path_obj.stem}_email{report_path_obj.suffix}"
        # Read the original report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        replacements_made = 0
        # 1. Replace all <iframe ...src="...">...</iframe> with the URL
        iframe_pattern = r'<iframe[^>]+src=["\']([^"\']+)["\'][^>]*>.*?</iframe>'
        def iframe_replacer(match):
            nonlocal replacements_made
            url = match.group(1)
            replacements_made += 1
            return f'\n{url}\n'
        processed_html = re.sub(iframe_pattern, iframe_replacer, report_html, flags=re.IGNORECASE | re.DOTALL)
        # 2. Remove any <div class="chart-container">...</div> blocks that no longer contain an iframe
        empty_container_pattern = r'<div[^>]*class=["\']chart-container["\'][^>]*>\s*</div>'
        processed_html = re.sub(empty_container_pattern, '', processed_html, flags=re.IGNORECASE | re.DOTALL)
        logger.info(f"Total iframe replacements made: {replacements_made}")
        # Write the processed HTML to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_html)
        logger.info(f"Generated email-compatible report at {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error generating email-compatible report: {str(e)}", exc_info=True)
        return None

def generate_chart_data_url(chart_type, params):
    """
    Generate a chart as a data URL for direct embedding in HTML.
    
    Args:
        chart_type: Type of chart ('time_series', 'anomaly', or 'map')
        params: Parameters for the chart (metric_id, district, etc.)
        
    Returns:
        Data URL of the chart image
    """
    logger.info(f"Generating data URL for {chart_type} chart with params: {params}")
    
    try:
        # For anomaly charts, try to use Datawrapper
        if chart_type == 'anomaly':
            anomaly_id = params.get('id', '27338')
            try:
                # Make a request to get the anomaly data
                anomaly_data_url = f"{API_BASE_URL}/anomaly-analyzer/api/anomaly/{anomaly_id}"
                logger.info(f"Requesting anomaly data for data URL: {anomaly_data_url}")
                
                response = requests.get(anomaly_data_url)
                if response.status_code == 200:
                    anomaly_response = response.json()
                    
                    # Extract the needed data for the chart
                    anomaly_data = anomaly_response.get("data", {})
                    if anomaly_data and "dates" in anomaly_data and "counts" in anomaly_data:
                        logger.info(f"Successfully retrieved data for anomaly data URL ID: {anomaly_id}")
                        
                        # Create a Datawrapper chart
                        chart_title = f"Anomaly {anomaly_id}: {anomaly_data.get('group_value', 'Trend Analysis')}"
                        metadata = anomaly_response.get("metadata", {})
                        
                        # Generate a Datawrapper chart
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly data URL: {chart_url}")
                            # Return the chart URL directly instead of using Playwright to take a screenshot
                            return chart_url
            except Exception as e:
                logger.error(f"Error using Datawrapper for anomaly data URL: {str(e)}", exc_info=True)
                logger.info("Falling back to traditional chart approach for data URL")
        
        # For maps, try to use Datawrapper
        elif chart_type == 'map':
            map_id = params.get('id', params.get('map_id'))
            if map_id:
                try:
                    # First try to get the existing map URL from the database
                    logger.info(f"Getting existing map URL for data URL: {map_id}")
                    
                    map_url = get_existing_map_url(map_id)
                    
                    if map_url:
                        logger.info(f"Successfully found existing map URL for data URL: {map_url}")
                        # Return the map URL directly
                        return map_url
                    else:
                        # If no existing map URL found, try to create/publish the map
                        logger.info(f"No existing map URL found for data URL {map_id}, attempting to create it")
                        from tools.gen_map_dw import create_datawrapper_map
                        map_url = create_datawrapper_map(map_id)
                        
                        if map_url:
                            logger.info(f"Successfully created Datawrapper map for data URL: {map_url}")
                            # Return the map URL directly
                            return map_url
                except Exception as e:
                    logger.error(f"Error using Datawrapper for map data URL: {str(e)}", exc_info=True)
                    logger.info("Falling back to traditional map approach for data URL")
                
        # Construct the URL for the chart/map
        if chart_type == 'time_series':
            url = f"/backend/time-series-chart?metric_id={params.get('metric_id', 1)}&district={params.get('district', 0)}&period_type={params.get('period_type', 'year')}#chart-section"
        elif chart_type == 'anomaly':
            url = f"/anomaly-analyzer/anomaly-chart?id={params.get('id', 27338)}#chart-section"
        elif chart_type == 'map':
            url = f"/backend/map-chart?id={params.get('id', params.get('map_id', ''))}"
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Generate a full URL to the chart
        full_url = f"{API_BASE_URL}{url}"
        logger.info(f"Using direct URL for chart: {full_url}")
        return full_url
        
    except Exception as e:
        logger.error(f"Error generating chart data URL: {str(e)}", exc_info=True)
        return None

def generate_inline_report(report_path, output_path=None):
    """
    Generate a report with inline data URLs for charts, making it suitable for email.
    
    Args:
        report_path: Path to the original report HTML file
        output_path: Path to save the inline report (defaults to report_path with _inline suffix)
        
    Returns:
        Path to the inline report
    """
    logger.info(f"Generating inline version of report: {report_path}")
    
    try:
        # Set default output path if not provided
        if not output_path:
            report_path_obj = Path(report_path)
            output_path = report_path_obj.parent / f"{report_path_obj.stem}_inline{report_path_obj.suffix}"
        
        # Read the original report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
            
        # Expand chart references in the report
        logger.info(f"Expanding chart references in report for inline version: {report_path}")
        expand_result = expand_chart_references(report_path)
        if not expand_result:
            logger.warning("Failed to expand chart references in the report for inline version")
            
        # Read the report again after expanding chart references
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Find all iframe elements
        import re
        iframe_pattern = r'<iframe[^>]*src="([^"]*)"[^>]*></iframe>'
        iframes = re.findall(iframe_pattern, report_html)
        
        # Replace each iframe with an inline image
        for iframe_url in iframes:
            # Extract chart type and parameters from the URL
            if 'time-series-chart' in iframe_url:
                # Extract parameters from URL
                params = {}
                for param in iframe_url.split('?')[1].split('&'):
                    key, value = param.split('=')
                    params[key] = value
                
                # Generate a data URL for the time series chart
                data_url = generate_time_series_chart_data_url(
                    metric_id=params.get('metric_id', 1),
                    district=params.get('district', 0),
                    period_type=params.get('period_type', 'year')
                )
                if not data_url:
                    logger.warning(f"Failed to generate data URL for {iframe_url}")
                    continue
                
                # Create an img tag with the data URL
                img_tag = f'<img src="{data_url}" alt="Time Series Chart" style="max-width: 100%; height: auto;" />'
                
                # Replace the iframe with the img tag
                iframe_tag = f'<iframe[^>]*src="{iframe_url}"[^>]*></iframe>'
                report_html = re.sub(iframe_tag, img_tag, report_html)
                
            elif 'anomaly-chart' in iframe_url:
                chart_type = 'anomaly'
                # Extract parameters from URL
                params = {}
                for param in iframe_url.split('?')[1].split('&'):
                    key, value = param.split('=')
                    params[key] = value
                
                # Generate a data URL for the chart
                data_url = generate_chart_data_url(chart_type, params)
                if not data_url:
                    logger.warning(f"Failed to generate data URL for {iframe_url}")
                    continue
                
                # Create an img tag with the data URL
                img_tag = f'<img src="{data_url}" alt="Chart" style="max-width: 100%; height: auto;" />'
                
                # Replace the iframe with the img tag
                iframe_tag = f'<iframe[^>]*src="{iframe_url}"[^>]*></iframe>'
                report_html = re.sub(iframe_tag, img_tag, report_html)
                
            elif 'map-chart' in iframe_url:
                chart_type = 'map'
                # Extract parameters from URL
                params = {}
                if '?' in iframe_url:
                    for param in iframe_url.split('?')[1].split('&'):
                        if '=' in param:
                            key, value = param.split('=')
                            params[key] = value
                
                # Generate a data URL for the map
                data_url = generate_chart_data_url(chart_type, params)
                if not data_url:
                    logger.warning(f"Failed to generate data URL for {iframe_url}")
                    continue
                
                # Create an img tag with the data URL
                img_tag = f'<img src="{data_url}" alt="Map" style="max-width: 100%; height: auto;" />'
                
                # Replace the iframe with the img tag
                iframe_tag = f'<iframe[^>]*src="{iframe_url}"[^>]*></iframe>'
                report_html = re.sub(iframe_tag, img_tag, report_html)
                
            else:
                logger.warning(f"Unsupported iframe URL: {iframe_url}")
                continue
        
        # Write the inline report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Generated inline report at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating inline report: {str(e)}", exc_info=True)
        return None

def generate_time_series_chart_image(metric_id, district, period_type, output_dir=None):
    """
    Generate a static image of a time series chart for email compatibility.
    
    Args:
        metric_id: ID of the metric to chart
        district: District number
        period_type: Period type (year, month, etc.)
        output_dir: Directory to save the image (defaults to a temporary directory)
        
    Returns:
        Path to the generated image file
    """
    logger.info(f"Generating static image for time series chart: metric_id={metric_id}, district={district}, period_type={period_type}")
    
    try:
        # Create output directory if not provided
        if not output_dir:
            output_dir = Path(__file__).parent / 'output' / 'charts'
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        filename = f"time_series_{metric_id}_{district}_{period_type}_{timestamp}.png"
        output_path = output_dir / filename
        
        # Construct the URL for the chart
        url = f"/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}#chart-section"
        
        # Use a headless browser service to capture the chart as an image
        # This could be implemented using Selenium, Playwright, or a service like Browserless
        # For this example, we'll use a simple approach with requests and PIL
        
        # Make a request to the chart URL
        full_url = f"{API_BASE_URL}{url}"  # Only use API_BASE_URL for server-side requests
        response = requests.get(full_url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch chart from {full_url}: {response.status_code}")
            return None
        
        # For a real implementation, you would use a headless browser to render the page
        # and capture the chart element as an image
        # For this example, we'll create a placeholder image
        
        # Create a placeholder image (in a real implementation, this would be the actual chart)
        img = Image.new('RGB', (800, 450), color=(255, 255, 255))
        
        # Save the image
        img.save(output_path)
        
        logger.info(f"Generated time series chart image at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating time series chart image: {str(e)}", exc_info=True)
        return None

def generate_time_series_chart_data_url(metric_id, district, period_type):
    """
    Generate a time series chart as a data URL for direct embedding in HTML.
    
    Args:
        metric_id: ID of the metric to chart
        district: District number
        period_type: Period type (year, month, etc.)
        
    Returns:
        Data URL of the chart image
    """
    logger.info(f"Generating data URL for time series chart: metric_id={metric_id}, district={district}, period_type={period_type}")
    
    try:
        # Construct the URL for the chart
        url = f"/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}#chart-section"
        
        # Use a headless browser service to capture the chart as an image
        # This could be implemented using Selenium, Playwright, or a service like Browserless
        # For this example, we'll use a simple approach with requests and PIL
        
        # Make a request to the chart URL
        full_url = f"{API_BASE_URL}{url}"  # Only use API_BASE_URL for server-side requests
        response = requests.get(full_url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch chart from {full_url}: {response.status_code}")
            return None
        
        # For a real implementation, you would use a headless browser to render the page
        # and capture the chart element as an image
        # For this example, we'll create a placeholder image
        
        # Create a placeholder image (in a real implementation, this would be the actual chart)
        img = Image.new('RGB', (800, 450), color=(255, 255, 255))
        
        # Convert the image to a data URL
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        
        logger.info(f"Generated data URL for time series chart")
        return data_url
        
    except Exception as e:
        logger.error(f"Error generating time series chart data URL: {str(e)}", exc_info=True)
        return None

class PathEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Path objects"""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def expand_chart_references(report_path):
    """
    Process a report file and replace simplified chart references with full HTML.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references in report: {report_path}")
    
    try:
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]'  # New pattern for time_series_id
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]'  # Changed to support alphanumeric IDs
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]'  # Add pattern for map references
        
        # Define pattern for direct image references
        image_pattern_with_alt = r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]+)"[^>]*>'
        image_pattern_without_alt = r'<img[^>]*src="([^"]+)"[^>]*>'
        
        # Replace time series chart references
        def replace_time_series(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Attempting to generate Datawrapper chart for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            dw_chart_url = create_datawrapper_chart(
                metric_id=metric_id,
                district=district,
                period_type=period_type
            )
            
            if dw_chart_url:
                logger.info(f"Successfully generated Datawrapper chart: {dw_chart_url}")
                # Use Datawrapper's responsive iframe embedding approach
                iframe_html = (
                    f'<div class="chart-container">\n'
                    f'    <div class="datawrapper-chart-embed">\n'
                    f'        <iframe src="{dw_chart_url}"\n'
                    f'                style="width: 100%; border: none;" \n'
                    f'                height="600"\n'
                    f'                frameborder="0" \n'
                    f'                scrolling="no"\n'
                    f'                allowfullscreen="true">\n'
                    f'        </iframe>\n'
                    f'    </div>\n'
                    f'</div>'
                )
                return iframe_html
            else:
                logger.warning(f"Failed to generate Datawrapper chart for metric_id: {metric_id}. Using placeholder.")
                return f"<!-- Datawrapper chart generation failed for metric_id: {metric_id}, district: {district}, period: {period_type} -->"
        
        # Replace anomaly chart references
        def replace_anomaly(match):
            anomaly_id = match.group(1)
            
            try:
                # Use the new helper function to generate a chart directly from the anomaly ID
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                logger.info(f"Generating chart for anomaly ID: {anomaly_id} using direct ID function")
                
                chart_url = generate_anomaly_chart_from_id(anomaly_id)
                
                if chart_url:
                    logger.info(f"Successfully generated Datawrapper chart for anomaly {anomaly_id}: {chart_url}")
                    # Use Datawrapper's responsive iframe embedding approach
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{chart_url}"\n'
                        f'                title="Anomaly {anomaly_id}: Trend Analysis"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="600"\n'
                        f'                frameborder="0" \n'
                        f'                scrolling="no"\n'
                        f'                aria-label="Anomaly {anomaly_id}: Trend Analysis"\n'
                        f'                allowfullscreen="true">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
            except Exception as e:
                logger.error(f"Error generating Datawrapper chart for anomaly {anomaly_id}: {str(e)}")
            
            # Fallback to the original iframe method if Datawrapper generation fails
            logger.info(f"Using fallback iframe for anomaly ID: {anomaly_id}")
            return f"""
<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 100%;">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no">
        </iframe>
    </div>
</div>"""
        
        # Replace direct image references with iframes
        def replace_image_with_alt(match):
            img_tag = match.group(0)
            img_src = match.group(1)
            img_alt = match.group(2)
            
            # Check if it's an anomaly chart image
            if img_src.startswith("anomaly_"):
                # Extract anomaly ID from the filename
                parts = img_src.split("_")
                if len(parts) >= 2:
                    anomaly_id = parts[1].split(".")[0]  # Remove file extension
                    
                    # Try to use Datawrapper for this anomaly image
                    try:
                        # Use the new helper function to generate a chart directly from the anomaly ID
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        logger.info(f"Generating chart for anomaly image ID: {anomaly_id} using direct ID function")
                        
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly image {anomaly_id}: {chart_url}")
                            # Use Datawrapper's responsive iframe embedding approach
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{chart_url}"\n'
                                f'                title="Anomaly {anomaly_id}: {img_alt}"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="600"\n'
                                f'                frameborder="0" \n'
                                f'                scrolling="no"\n'
                                f'                aria-label="Anomaly {anomaly_id}: {img_alt}"\n'
                                f'                allowfullscreen="true">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as e:
                        logger.error(f"Error generating Datawrapper chart for anomaly image {anomaly_id}: {str(e)}")
                    
                    # Fallback to the original iframe method
                    logger.info(f"Using fallback iframe for anomaly image ID: {anomaly_id}")
                    return f"""
<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 100%;">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no">
        </iframe>
    </div>
</div>"""
            
            # If we can't determine the chart type, keep the original image
            return img_tag
        
        def replace_image_without_alt(match):
            img_tag = match.group(0)
            img_src = match.group(1)
            
            # Check if it's an anomaly chart image
            if img_src.startswith("anomaly_"):
                # Extract anomaly ID from the filename
                parts = img_src.split("_")
                if len(parts) >= 2:
                    anomaly_id = parts[1].split(".")[0]  # Remove file extension
                    
                    # Try to use Datawrapper for this anomaly image
                    try:
                        # Use the new helper function to generate a chart directly from the anomaly ID
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        logger.info(f"Generating chart for anomaly image ID (no alt): {anomaly_id} using direct ID function")
                        
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly image {anomaly_id}: {chart_url}")
                            # Use Datawrapper's responsive iframe embedding approach
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{chart_url}"\n'
                                f'                title="Anomaly {anomaly_id}: Trend Analysis"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="600"\n'
                                f'                frameborder="0" \n'
                                f'                scrolling="no"\n'
                                f'                aria-label="Anomaly {anomaly_id}: Trend Analysis"\n'
                                f'                allowfullscreen="true">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as e:
                        logger.error(f"Error generating Datawrapper chart for anomaly image (no alt) {anomaly_id}: {str(e)}")
                    
                    # Fallback to the original iframe method
                    logger.info(f"Using fallback iframe for anomaly image ID (no alt): {anomaly_id}")
                    return f"""
<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 100%;">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no">
        </iframe>
    </div>
</div>"""
            
            # If we can't determine the chart type, keep the original image
            return img_tag
        
        # Replace map references
        def replace_map(match):
            map_id = match.group(1)
            
            try:
                # First try to get the existing map URL from the database
                logger.info(f"Getting existing map URL for map ID: {map_id}")
                
                map_url = get_existing_map_url(map_id)
                
                if map_url:
                    logger.info(f"Successfully found existing map URL for map {map_id}: {map_url}")
                    # Use Datawrapper's responsive iframe embedding approach
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{map_url}"\n'
                        f'                title="Map {map_id}"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="600"\n'
                        f'                frameborder="0" \n'
                        f'                scrolling="no"\n'
                        f'                aria-label="Map {map_id}"\n'
                        f'                allowfullscreen="true">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
                else:
                    # If no existing map URL found, try to create/publish the map
                    logger.info(f"No existing map URL found for map {map_id}, attempting to create it")
                    try:
                        from tools.gen_map_dw import create_datawrapper_map
                        map_url = create_datawrapper_map(map_id)
                        
                        if map_url:
                            logger.info(f"Successfully created Datawrapper map for map {map_id}: {map_url}")
                            # Use Datawrapper's responsive iframe embedding approach
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{map_url}"\n'
                                f'                title="Map {map_id}"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="600"\n'
                                f'                frameborder="0" \n'
                                f'                scrolling="no"\n'
                                f'                aria-label="Map {map_id}"\n'
                                f'                allowfullscreen="true">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as create_error:
                        logger.error(f"Error creating Datawrapper map for map {map_id}: {str(create_error)}")
            except Exception as e:
                logger.error(f"Error getting existing map URL for map {map_id}: {str(e)}")
            
            # Fallback to the backend map chart endpoint if both existing URL and creation fail
            logger.info(f"Using fallback iframe for map ID: {map_id}")
            return f"""<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 75%;">
        <iframe src="/backend/map-chart?id={map_id}" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no"
                title="Map {map_id}">
        </iframe>
    </div>
</div>"""
        
        # Replace time series chart references by chart_id
        def replace_time_series_id(match):
            chart_id = match.group(1)
            
            logger.info(f"Attempting to generate Datawrapper chart for time_series_id: {chart_id}")
            
            try:
                # Import the tools we need
                from tools.store_time_series import get_time_series_metadata, get_time_series_data
                from tools.genChartdw import create_time_series_chart_from_data
                
                # Get the metadata for this chart
                metadata_df = get_time_series_metadata(chart_id=int(chart_id))
                if metadata_df.empty:
                    logger.error(f"No metadata found for time_series_id: {chart_id}")
                    return f"<!-- No metadata found for time_series_id: {chart_id} -->"
                
                metadata_row = metadata_df.iloc[0]
                
                # Get the time series data
                data_df = get_time_series_data(chart_id=int(chart_id))
                if data_df.empty:
                    logger.error(f"No data found for time_series_id: {chart_id}")
                    return f"<!-- No data found for time_series_id: {chart_id} -->"
                
                # Extract metadata information
                # First check if there's additional metadata in the metadata JSONB field
                metadata_json = metadata_row.get('metadata', {})
                if isinstance(metadata_json, str):
                    try:
                        metadata_json = json.loads(metadata_json)
                    except json.JSONDecodeError:
                        metadata_json = {}
                
                # Try multiple sources for the chart title with proper fallback
                chart_title = (
                    metadata_json.get('title') or         # First: title from metadata JSONB
                    metadata_json.get('chart_title') or   # Second: chart_title from metadata JSONB  
                    metadata_row.get('object_name') or    # Third: object_name from table
                    f"Time Series Chart {chart_id}"       # Fallback
                )
                
                object_name = metadata_row.get('object_name', 'Unknown')
                field_name = metadata_row.get('field_name', 'Value')
                period_type = metadata_row.get('period_type', 'month')
                district = metadata_row.get('district', 0)
                
                # Prepare chart data in the format expected by create_time_series_chart_from_data
                chart_data = []
                for _, row in data_df.iterrows():
                    chart_data.append({
                        'time_period': row['time_period'],
                        'value': row['numeric_value'],
                        'group_value': row.get('group_value')
                    })
                
                # Create chart metadata
                chart_metadata = {
                    'title': chart_title,
                    'object_name': object_name,
                    'field_name': field_name,
                    'period_type': period_type,
                    'district': district,
                    'chart_id': chart_id
                }
                
                # Generate the Datawrapper chart
                dw_chart_url = create_time_series_chart_from_data(
                    chart_data=chart_data,
                    metadata=chart_metadata
                )
                
                if dw_chart_url:
                    logger.info(f"Successfully generated Datawrapper chart for time_series_id {chart_id}: {dw_chart_url}")
                    # Use Datawrapper's responsive iframe embedding approach
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{dw_chart_url}"\n'
                        f'                title="{chart_title}"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="600"\n'
                        f'                frameborder="0" \n'
                        f'                scrolling="no"\n'
                        f'                aria-label="{chart_title}"\n'
                        f'                allowfullscreen="true">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
                else:
                    logger.warning(f"Failed to generate Datawrapper chart for time_series_id: {chart_id}")
                    return f"<!-- Datawrapper chart generation failed for time_series_id: {chart_id} -->"
                    
            except ImportError as e:
                logger.error(f"Missing required tools for time_series_id processing: {str(e)}")
                return f"<!-- Missing tools for time_series_id: {chart_id} - {str(e)} -->"
            except Exception as e:
                logger.error(f"Error generating Datawrapper chart for time_series_id {chart_id}: {str(e)}")
                return f"<!-- Error generating chart for time_series_id: {chart_id} - {str(e)} -->"
        
        # Apply replacements
        report_html = re.sub(time_series_pattern, replace_time_series, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id, report_html)  # Add time_series_id support
        report_html = re.sub(anomaly_pattern, replace_anomaly, report_html)
        report_html = re.sub(map_pattern, replace_map, report_html)
        report_html = re.sub(image_pattern_with_alt, replace_image_with_alt, report_html)
        report_html = re.sub(image_pattern_without_alt, replace_image_without_alt, report_html)
        
        # Write the processed report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def request_chart_image(chart_type, params, output_path=None):
    """
    Request a chart URL instead of rendering an image.
    
    Args:
        chart_type: Type of chart ('time_series', 'anomaly', or 'map')
        params: Parameters for the chart (metric_id, district, etc.)
        output_path: Ignored parameter (kept for backwards compatibility)
        
    Returns:
        URL string for the chart if successful, None otherwise
    """
    logger.info(f"Getting URL for {chart_type} chart with params: {params}")
    
    try:
        # For anomaly charts, use Datawrapper if possible
        if chart_type == 'anomaly':
            anomaly_id = params.get('id', '27338')
            try:
                # Make a request to get the anomaly data
                anomaly_data_url = f"{API_BASE_URL}/anomaly-analyzer/api/anomaly/{anomaly_id}"
                logger.info(f"Requesting anomaly data for URL: {anomaly_data_url}")
                
                response = requests.get(anomaly_data_url)
                if response.status_code == 200:
                    anomaly_response = response.json()
                    
                    # Extract the needed data for the chart
                    anomaly_data = anomaly_response.get("data", {})
                    if anomaly_data and "dates" in anomaly_data and "counts" in anomaly_data:
                        logger.info(f"Successfully retrieved data for anomaly URL ID: {anomaly_id}")
                        
                        # Create a Datawrapper chart
                        chart_title = f"Anomaly {anomaly_id}: {anomaly_data.get('group_value', 'Trend Analysis')}"
                        metadata = anomaly_response.get("metadata", {})
                        
                        # Generate a Datawrapper chart and get the public URL
                        chart_url = generate_anomaly_chart_dw(anomaly_data, chart_title, metadata)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly: {chart_url}")
                            return chart_url
            except Exception as e:
                logger.error(f"Error using Datawrapper for anomaly chart: {str(e)}", exc_info=True)
                
            # Fallback to direct anomaly page URL if Datawrapper fails
            landing_page_url = f"{API_BASE_URL}/anomaly-analyzer/anomaly/{anomaly_id}"
            logger.info(f"Using anomaly page URL: {landing_page_url}")
            return landing_page_url
            
        elif chart_type == 'time_series':
            metric_id = params.get('metric_id', '1')
            district = params.get('district', '0')
            period_type = params.get('period_type', 'year')
            
            # Try to generate a Datawrapper chart first
            try:
                logger.info(f"Attempting to create Datawrapper chart for metric_id: {metric_id}, district: {district}, period: {period_type}")
                dw_chart_url = create_datawrapper_chart(
                    metric_id=metric_id,
                    district=district,
                    period_type=period_type
                )
                
                if dw_chart_url:
                    logger.info(f"Successfully generated Datawrapper chart: {dw_chart_url}")
                    return dw_chart_url
            except Exception as e:
                logger.error(f"Error creating Datawrapper chart: {str(e)}", exc_info=True)
            
            # Fallback to metric page URL
            landing_page_url = f"{API_BASE_URL}/backend/metric/{metric_id}?district={district}"
            logger.info(f"Using metric page URL: {landing_page_url}")
            return landing_page_url
            
        elif chart_type == 'map':
            map_id = params.get('id', params.get('map_id'))
            
            if not map_id:
                logger.error("No map ID provided for map chart request")
                return None
            
            # Try to get the existing map URL from the database first
            try:
                logger.info(f"Getting existing map URL for map_id: {map_id}")
                
                map_url = get_existing_map_url(map_id)
                
                if map_url:
                    logger.info(f"Successfully found existing map URL: {map_url}")
                    return map_url
                else:
                    # If no existing map URL found, try to create/publish the map
                    logger.info(f"No existing map URL found for map_id {map_id}, attempting to create it")
                    from tools.gen_map_dw import create_datawrapper_map
                    map_url = create_datawrapper_map(map_id)
                    
                    if map_url:
                        logger.info(f"Successfully created Datawrapper map: {map_url}")
                        return map_url
            except Exception as e:
                logger.error(f"Error getting/creating Datawrapper map: {str(e)}", exc_info=True)
            
            # Fallback to backend map chart endpoint
            landing_page_url = f"{API_BASE_URL}/backend/map-chart?id={map_id}"
            logger.info(f"Using map chart page URL: {landing_page_url}")
            return landing_page_url
            
        else:
            logger.error(f"Unsupported chart type: {chart_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting chart URL: {e}", exc_info=True)
        return None

def safe_json_serialize(obj, default_msg="<not serializable>"):
    """
    Safely serialize an object to JSON, handling non-serializable objects.
    
    Args:
        obj: Object to serialize
        default_msg: Default message to return if serialization fails
        
    Returns:
        JSON string or default message
    """
    try:
        # Use our custom encoder that handles Path objects
        return json.dumps(obj, cls=PathEncoder)
    except (TypeError, OverflowError, ValueError) as e:
        # For non-serializable objects, return a string representation
        if isinstance(obj, dict):
            # Try to serialize each key/value pair individually
            safe_dict = {}
            for k, v in obj.items():
                try:
                    # Try to serialize the value
                    json.dumps(v)
                    safe_dict[k] = v
                except (TypeError, OverflowError, ValueError):
                    # If it fails, use string representation
                    safe_dict[k] = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
            try:
                return json.dumps(safe_dict)
            except:
                return default_msg
        elif isinstance(obj, list):
            # Try to serialize each item individually
            safe_list = []
            for item in obj:
                try:
                    # Try to serialize the item
                    json.dumps(item)
                    safe_list.append(item)
                except (TypeError, OverflowError, ValueError):
                    # If it fails, use string representation
                    safe_list.append(str(item)[:100] + "..." if len(str(item)) > 100 else str(item))
            try:
                return json.dumps(safe_list)
            except:
                return default_msg
        else:
            # For other types, just return string representation
            return f'"{str(obj)[:100] + "..." if len(str(obj)) > 100 else str(obj)}"'

def generate_narrated_report(report_path, output_path=None):
    """
    Generate a narrated audio version of the report using ElevenLabs API.
    
    Args:
        report_path: Path to the email-compatible report file
        output_path: Path to save the audio file (defaults to same name with .mp3 extension)
        
    Returns:
        Dictionary containing paths to the generated audio file and script HTML file, or None if failed
    """
    from webChat import client, AGENT_MODEL
    
    logger.info(f"Generating narrated version of report: {report_path}")
    
    # Check if ElevenLabs API key is available
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        logger.warning("No ElevenLabs API key found. Cannot generate narration.")
        return None
    
    try:
        # Set default output path if not provided
        if not output_path:
            report_path_obj = Path(report_path)
            # Create narration directory if it doesn't exist
            narration_dir = Path(__file__).parent / 'output' / 'narration'
            narration_dir.mkdir(parents=True, exist_ok=True)
            output_path = narration_dir / f"{report_path_obj.stem}.mp3"
        
        # Create reports directory if it doesn't exist
        reports_dir = Path(__file__).parent / 'output' / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up audio script HTML file path
        script_html_path = reports_dir / f"{Path(report_path).stem}_audio_script.html"
        
        # Read the email-compatible report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Load prompt from JSON file
        prompts = load_prompts()
        prompt_template = prompts['monthly_report']['audio_transformation']['prompt']
        system_message = prompts['monthly_report']['audio_transformation']['system']
        
        # Format the prompt with the newsletter content
        prompt = prompt_template.format(newsletter_content=report_html)
        
        logger.info("Using AI to transform newsletter content for audio narration")
        
        # Use AI to transform the content for audio
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        audio_script = response.choices[0].message.content
        logger.info(f"Generated audio script: {len(audio_script)} characters")
        logger.info(f"Sample audio script: {audio_script[:200]}...")
        
        # Save the audio script as HTML
        try:
            script_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Audio Script - {Path(report_path).stem}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        .script-container {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .script-content {{
            white-space: pre-wrap;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <h1>Audio Script</h1>
    <div class="metadata">
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Report: {Path(report_path).name}</p>
        <p>Characters: {len(audio_script)}</p>
    </div>
    <div class="script-container">
        <div class="script-content">{audio_script}</div>
    </div>
</body>
</html>"""
            
            with open(script_html_path, 'w', encoding='utf-8') as f:
                f.write(script_html)
            logger.info(f"Successfully saved audio script to {script_html_path}")
        except Exception as script_err:
            logger.error(f"Failed to save audio script HTML: {script_err}")
            script_html_path = None
        
        # Limit text length for ElevenLabs (API has character limits)
        max_chars = 2500  # Conservative limit for ElevenLabs
        if len(audio_script) > max_chars:
            # Find a good breaking point (end of sentence)
            truncate_point = audio_script.rfind('.', 0, max_chars)
            if truncate_point > max_chars // 2:  # Make sure we didn't truncate too much
                audio_script = audio_script[:truncate_point + 1]
            else:
                audio_script = audio_script[:max_chars]
            
            logger.info(f"Truncated audio script to {len(audio_script)} characters for ElevenLabs API")
        
        # ElevenLabs API configuration
        # Using a default voice ID - this could be made configurable
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Default to Reverend voice
        
        # ElevenLabs API endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        data = {
            "text": audio_script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        # Make the API request
        logger.info("Sending request to ElevenLabs API")
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully generated narrated report at {output_path}")
            return {
                "audio_path": str(output_path),
                "script_path": str(script_html_path) if script_html_path else None
            }
        else:
            logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating narrated report: {str(e)}", exc_info=True)
        return None

# Add this new function after the existing functions and before expand_chart_references

def get_existing_map_url(map_id):
    """
    Get the published URL of an existing map from the database instead of creating a new one.
    
    Args:
        map_id: The ID of the map to retrieve
        
    Returns:
        The published URL of the existing map, or None if not found
    """
    logger.info(f"Getting existing map URL for map_id: {map_id}")
    
    def get_map_operation(connection):
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the map by ID
        cursor.execute("""
            SELECT id, title, published_url, chart_id, edit_url, type, metadata
            FROM maps 
            WHERE id = %s AND active = TRUE
        """, (map_id,))
        
        map_record = cursor.fetchone()
        cursor.close()
        
        if not map_record:
            logger.warning(f"Map with ID {map_id} not found in database")
            return None
        
        # Check if we have a published URL
        if map_record['published_url']:
            logger.info(f"Found existing published URL for map {map_id}: {map_record['published_url']}")
            return map_record['published_url']
        else:
            logger.warning(f"Map {map_id} exists but has no published URL")
            return None
    
    # Execute the operation with proper connection handling
    result = execute_with_connection(
        operation=get_map_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success":
        return result["result"]
    else:
        logger.error(f"Error getting existing map URL: {result['message']}")
        return None

def expand_chart_references(report_path):
    """
    Process a report file and replace simplified chart references with full HTML.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references in report: {report_path}")
    
    try:
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]'  # New pattern for time_series_id
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]'  # Changed to support alphanumeric IDs
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]'  # Add pattern for map references
        
        # Define pattern for direct image references
        image_pattern_with_alt = r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]+)"[^>]*>'
        image_pattern_without_alt = r'<img[^>]*src="([^"]+)"[^>]*>'
        
        # Replace time series chart references
        def replace_time_series(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Attempting to generate Datawrapper chart for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            dw_chart_url = create_datawrapper_chart(
                metric_id=metric_id,
                district=district,
                period_type=period_type
            )
            
            if dw_chart_url:
                logger.info(f"Successfully generated Datawrapper chart: {dw_chart_url}")
                # Use Datawrapper's responsive iframe embedding approach
                iframe_html = (
                    f'<div class="chart-container">\n'
                    f'    <div class="datawrapper-chart-embed">\n'
                    f'        <iframe src="{dw_chart_url}"\n'
                    f'                style="width: 100%; border: none;" \n'
                    f'                height="600"\n'
                    f'                frameborder="0" \n'
                    f'                scrolling="no"\n'
                    f'                allowfullscreen="true">\n'
                    f'        </iframe>\n'
                    f'    </div>\n'
                    f'</div>'
                )
                return iframe_html
            else:
                logger.warning(f"Failed to generate Datawrapper chart for metric_id: {metric_id}. Using placeholder.")
                return f"<!-- Datawrapper chart generation failed for metric_id: {metric_id}, district: {district}, period: {period_type} -->"
        
        # Replace anomaly chart references
        def replace_anomaly(match):
            anomaly_id = match.group(1)
            
            try:
                # Use the new helper function to generate a chart directly from the anomaly ID
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                logger.info(f"Generating chart for anomaly ID: {anomaly_id} using direct ID function")
                
                chart_url = generate_anomaly_chart_from_id(anomaly_id)
                
                if chart_url:
                    logger.info(f"Successfully generated Datawrapper chart for anomaly {anomaly_id}: {chart_url}")
                    # Use Datawrapper's responsive iframe embedding approach
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{chart_url}"\n'
                        f'                title="Anomaly {anomaly_id}: Trend Analysis"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="600"\n'
                        f'                frameborder="0" \n'
                        f'                scrolling="no"\n'
                        f'                aria-label="Anomaly {anomaly_id}: Trend Analysis"\n'
                        f'                allowfullscreen="true">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
            except Exception as e:
                logger.error(f"Error generating Datawrapper chart for anomaly {anomaly_id}: {str(e)}")
            
            # Fallback to the original iframe method if Datawrapper generation fails
            logger.info(f"Using fallback iframe for anomaly ID: {anomaly_id}")
            return f"""
<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 100%;">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no">
        </iframe>
    </div>
</div>"""
        
        # Replace direct image references with iframes
        def replace_image_with_alt(match):
            img_tag = match.group(0)
            img_src = match.group(1)
            img_alt = match.group(2)
            
            # Check if it's an anomaly chart image
            if img_src.startswith("anomaly_"):
                # Extract anomaly ID from the filename
                parts = img_src.split("_")
                if len(parts) >= 2:
                    anomaly_id = parts[1].split(".")[0]  # Remove file extension
                    
                    # Try to use Datawrapper for this anomaly image
                    try:
                        # Use the new helper function to generate a chart directly from the anomaly ID
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        logger.info(f"Generating chart for anomaly image ID: {anomaly_id} using direct ID function")
                        
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly image {anomaly_id}: {chart_url}")
                            # Use Datawrapper's responsive iframe embedding approach
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{chart_url}"\n'
                                f'                title="Anomaly {anomaly_id}: {img_alt}"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="600"\n'
                                f'                frameborder="0" \n'
                                f'                scrolling="no"\n'
                                f'                aria-label="Anomaly {anomaly_id}: {img_alt}"\n'
                                f'                allowfullscreen="true">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as e:
                        logger.error(f"Error generating Datawrapper chart for anomaly image {anomaly_id}: {str(e)}")
                    
                    # Fallback to the original iframe method
                    logger.info(f"Using fallback iframe for anomaly image ID: {anomaly_id}")
                    return f"""
<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 100%;">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no">
        </iframe>
    </div>
</div>"""
            
            # If we can't determine the chart type, keep the original image
            return img_tag
        
        def replace_image_without_alt(match):
            img_tag = match.group(0)
            img_src = match.group(1)
            
            # Check if it's an anomaly chart image
            if img_src.startswith("anomaly_"):
                # Extract anomaly ID from the filename
                parts = img_src.split("_")
                if len(parts) >= 2:
                    anomaly_id = parts[1].split(".")[0]  # Remove file extension
                    
                    # Try to use Datawrapper for this anomaly image
                    try:
                        # Use the new helper function to generate a chart directly from the anomaly ID
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        logger.info(f"Generating chart for anomaly image ID (no alt): {anomaly_id} using direct ID function")
                        
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly image {anomaly_id}: {chart_url}")
                            # Use Datawrapper's responsive iframe embedding approach
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{chart_url}"\n'
                                f'                title="Anomaly {anomaly_id}: Trend Analysis"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="600"\n'
                                f'                frameborder="0" \n'
                                f'                scrolling="no"\n'
                                f'                aria-label="Anomaly {anomaly_id}: Trend Analysis"\n'
                                f'                allowfullscreen="true">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as e:
                        logger.error(f"Error generating Datawrapper chart for anomaly image (no alt) {anomaly_id}: {str(e)}")
                    
                    # Fallback to the original iframe method
                    logger.info(f"Using fallback iframe for anomaly image ID (no alt): {anomaly_id}")
                    return f"""
<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 100%;">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no">
        </iframe>
    </div>
</div>"""
            
            # If we can't determine the chart type, keep the original image
            return img_tag
        
        # Replace map references
        def replace_map(match):
            map_id = match.group(1)
            
            try:
                # Try to use Datawrapper for the map
                from tools.gen_map_dw import create_datawrapper_map
                logger.info(f"Generating Datawrapper map for map ID: {map_id}")
                
                map_url = create_datawrapper_map(map_id)
                
                if map_url:
                    logger.info(f"Successfully generated Datawrapper map for map {map_id}: {map_url}")
                    # Use Datawrapper's responsive iframe embedding approach
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{map_url}"\n'
                        f'                title="Map {map_id}"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="600"\n'
                        f'                frameborder="0" \n'
                        f'                scrolling="no"\n'
                        f'                aria-label="Map {map_id}"\n'
                        f'                allowfullscreen="true">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
            except Exception as e:
                logger.error(f"Error generating Datawrapper map for map {map_id}: {str(e)}")
            
            # Fallback to the backend map chart endpoint if Datawrapper fails
            logger.info(f"Using fallback iframe for map ID: {map_id}")
            return f"""<div class="chart-container">
    <div style="position: relative; width: 100%; padding-bottom: 75%;">
        <iframe src="/backend/map-chart?id={map_id}" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" 
                frameborder="0" 
                scrolling="no"
                title="Map {map_id}">
        </iframe>
    </div>
</div>"""
        
        # Replace time series chart references by chart_id
        def replace_time_series_id(match):
            chart_id = match.group(1)
            
            logger.info(f"Attempting to generate Datawrapper chart for time_series_id: {chart_id}")
            
            try:
                # Import the tools we need
                from tools.store_time_series import get_time_series_metadata, get_time_series_data
                from tools.genChartdw import create_time_series_chart_from_data
                
                # Get the metadata for this chart
                metadata_df = get_time_series_metadata(chart_id=int(chart_id))
                if metadata_df.empty:
                    logger.error(f"No metadata found for time_series_id: {chart_id}")
                    return f"<!-- No metadata found for time_series_id: {chart_id} -->"
                
                metadata_row = metadata_df.iloc[0]
                
                # Get the time series data
                data_df = get_time_series_data(chart_id=int(chart_id))
                if data_df.empty:
                    logger.error(f"No data found for time_series_id: {chart_id}")
                    return f"<!-- No data found for time_series_id: {chart_id} -->"
                
                # Extract metadata information
                # First check if there's additional metadata in the metadata JSONB field
                metadata_json = metadata_row.get('metadata', {})
                if isinstance(metadata_json, str):
                    try:
                        metadata_json = json.loads(metadata_json)
                    except json.JSONDecodeError:
                        metadata_json = {}
                
                # Try multiple sources for the chart title with proper fallback
                chart_title = (
                    metadata_json.get('title') or         # First: title from metadata JSONB
                    metadata_json.get('chart_title') or   # Second: chart_title from metadata JSONB  
                    metadata_row.get('object_name') or    # Third: object_name from table
                    f"Time Series Chart {chart_id}"       # Fallback
                )
                
                object_name = metadata_row.get('object_name', 'Unknown')
                field_name = metadata_row.get('field_name', 'Value')
                period_type = metadata_row.get('period_type', 'month')
                district = metadata_row.get('district', 0)
                
                # Prepare chart data in the format expected by create_time_series_chart_from_data
                chart_data = []
                for _, row in data_df.iterrows():
                    chart_data.append({
                        'time_period': row['time_period'],
                        'value': row['numeric_value'],
                        'group_value': row.get('group_value')
                    })
                
                # Create chart metadata
                chart_metadata = {
                    'title': chart_title,
                    'object_name': object_name,
                    'field_name': field_name,
                    'period_type': period_type,
                    'district': district,
                    'chart_id': chart_id
                }
                
                # Generate the Datawrapper chart
                dw_chart_url = create_time_series_chart_from_data(
                    chart_data=chart_data,
                    metadata=chart_metadata
                )
                
                if dw_chart_url:
                    logger.info(f"Successfully generated Datawrapper chart for time_series_id {chart_id}: {dw_chart_url}")
                    # Use Datawrapper's responsive iframe embedding approach
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{dw_chart_url}"\n'
                        f'                title="{chart_title}"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="600"\n'
                        f'                frameborder="0" \n'
                        f'                scrolling="no"\n'
                        f'                aria-label="{chart_title}"\n'
                        f'                allowfullscreen="true">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
                else:
                    logger.warning(f"Failed to generate Datawrapper chart for time_series_id: {chart_id}")
                    return f"<!-- Datawrapper chart generation failed for time_series_id: {chart_id} -->"
                    
            except ImportError as e:
                logger.error(f"Missing required tools for time_series_id processing: {str(e)}")
                return f"<!-- Missing tools for time_series_id: {chart_id} - {str(e)} -->"
            except Exception as e:
                logger.error(f"Error generating Datawrapper chart for time_series_id {chart_id}: {str(e)}")
                return f"<!-- Error generating chart for time_series_id: {chart_id} - {str(e)} -->"
        
        # Apply replacements
        report_html = re.sub(time_series_pattern, replace_time_series, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id, report_html)  # Add time_series_id support
        report_html = re.sub(anomaly_pattern, replace_anomaly, report_html)
        report_html = re.sub(map_pattern, replace_map, report_html)
        report_html = re.sub(image_pattern_with_alt, replace_image_with_alt, report_html)
        report_html = re.sub(image_pattern_without_alt, replace_image_without_alt, report_html)
        
        # Write the processed report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def reprioritize_deltas_for_report(filename, district="0", period_type="month", max_report_items=10):
    """
    Re-prioritize deltas for an existing newsletter by clearing the monthly_reporting data
    and regenerating it from scratch. This function stops after storing the prioritized items
    and does NOT generate explanations (use regenerate_explanations_for_report for that).
    
    Args:
        filename: The newsletter filename (e.g., "monthly_report_0_2024_01.html")
        district: District number (0 for citywide)
        period_type: Time period type (month, quarter, year)
        max_report_items: Maximum number of items to include in the newsletter
        
    Returns:
        Status dictionary with success/error information
    """
    logger.info(f"Re-prioritizing deltas for newsletter: {filename}")
    
    try:
        # Step 1: Find the existing report by filename
        def find_and_clear_report_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Find the report by original filename
            cursor.execute("""
                SELECT id, district, period_type, max_items 
                FROM reports 
                WHERE original_filename = %s
                ORDER BY created_at DESC 
                LIMIT 1
            """, (filename,))
            
            report = cursor.fetchone()
            if not report:
                cursor.close()
                return None
                
            report_id = report['id']
            logger.info(f"Found report with ID: {report_id}")
            
            # Clear all monthly_reporting records for this report
            cursor.execute("""
                DELETE FROM monthly_reporting 
                WHERE report_id = %s
            """, (report_id,))
            
            deleted_count = cursor.rowcount
            logger.info(f"Deleted {deleted_count} existing monthly_reporting records for report {report_id}")
            
            connection.commit()
            cursor.close()
            
            return {
                "report_id": report_id,
                "deleted_count": deleted_count,
                "original_district": report['district'],
                "original_period_type": report['period_type'],
                "original_max_items": report['max_items']
            }
        
        # Execute the find and clear operation
        result = execute_with_connection(
            operation=find_and_clear_report_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] != "success" or not result["result"]:
            return {"status": "error", "message": f"Newsletter with filename '{filename}' not found"}
        
        report_info = result["result"]
        report_id = report_info["report_id"]
        
        logger.info(f"Successfully cleared {report_info['deleted_count']} records for report {report_id}")
        
        # Use original report parameters if not overridden
        actual_district = district if district != "0" else report_info["original_district"]
        actual_period_type = period_type if period_type != "month" else report_info["original_period_type"]
        actual_max_items = max_report_items if max_report_items != 10 else report_info["original_max_items"]
        
        logger.info(f"Using parameters: district={actual_district}, period_type={actual_period_type}, max_items={actual_max_items}")
        
        # Step 2: Select deltas to discuss (same as original process)
        logger.info("Step 2: Selecting deltas to discuss")
        deltas = select_deltas_to_discuss(period_type=actual_period_type, district=actual_district)
        if deltas.get("status") != "success":
            return {"status": "error", "message": f"Failed to select deltas: {deltas.get('message')}"}
            
        # Step 3: Prioritize deltas and get explanations
        logger.info("Step 3: Prioritizing deltas")
        prioritized = prioritize_deltas(deltas, max_items=actual_max_items)
        if prioritized.get("status") != "success":
            return {"status": "error", "message": f"Failed to prioritize deltas: {prioritized.get('message')}"}
        
        # Step 4: Store new prioritized items with the existing report_id
        logger.info("Step 4: Storing new prioritized items")
        
        def store_reprioritized_items_operation(connection):
            cursor = connection.cursor()
            
            # Get the report date from the existing report
            cursor.execute("""
                SELECT created_at FROM reports WHERE id = %s
            """, (report_id,))
            
            report_record = cursor.fetchone()
            if not report_record:
                cursor.close()
                return []
                
            report_date = report_record[0].date()
            
            # Insert each prioritized item with the existing report_id
            inserted_ids = []
            for item in prioritized.get("prioritized_items", []):
                # Calculate percent change
                comparison_mean = item.get("comparison_mean", 0)
                if item.get("percent_change") is not None:
                    # Use the percent_change from the API if available
                    percent_change = item.get("percent_change")
                elif comparison_mean != 0:
                    # Fallback to calculating it if not provided
                    percent_change = (item.get("difference", 0) / comparison_mean) * 100
                else:
                    percent_change = 0
                    
                # Log the data before insertion
                logger.info(f"Storing item: {item.get('metric')} - Priority: {item.get('priority')}")
                
                # Create metadata JSON to store additional fields
                metadata = {
                    "trend_analysis": item.get("trend_analysis", ""),
                    "follow_up": item.get("follow_up", ""),
                    "change_type": item.get("change_type", "neutral"),  # Store change_type
                    "greendirection": item.get("greendirection"),  # Store greendirection
                    "citywide_changes": item.get("citywide_changes", "")  # Store citywide changes
                }
                
                # Create item title from metric name and group
                metric_name = item.get("metric", "")
                metric_id = item.get("metric_id", "Unknown")
                group_value = item.get("group") or "All"
                item_title = f"{metric_name} - {group_value}" if group_value != "All" else metric_name
                    
                # Insert the new record
                cursor.execute("""
                    INSERT INTO monthly_reporting (
                        report_id, report_date, item_title, metric_name, metric_id, group_value, group_field_name, 
                        period_type, comparison_mean, recent_mean, difference, 
                        percent_change, rationale, explanation, priority, district, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id
                """, (
                    report_id,  # Use existing report_id
                    report_date,
                    item_title,
                    metric_name,
                    metric_id,
                    group_value,
                    "group",
                    actual_period_type,
                    item.get("comparison_mean", 0),
                    item.get("recent_mean", 0),
                    item.get("difference", 0),
                    percent_change,
                    item.get("rationale", ""),
                    "",  # explanation: will be empty until regenerate_explanations is called
                    item.get("priority", 999),
                    actual_district,
                    json.dumps(metadata)
                ))
                
                inserted_id = cursor.fetchone()[0]
                inserted_ids.append(inserted_id)
            
            connection.commit()
            cursor.close()
            
            logger.info(f"Successfully stored {len(inserted_ids)} reprioritized items")
            return inserted_ids
        
        # Execute the store operation
        store_result = execute_with_connection(
            operation=store_reprioritized_items_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if store_result["status"] != "success":
            return {"status": "error", "message": f"Failed to store reprioritized items: {store_result['message']}"}
            
        inserted_ids = store_result["result"]
        
        logger.info(f"Successfully re-prioritized deltas for newsletter {filename} (explanations not generated)")
        return {
            "status": "success",
            "message": f"Successfully re-prioritized deltas for newsletter {filename}. Use 'Re-explain deltas' to generate explanations.",
            "report_id": report_id,
            "deleted_count": report_info["deleted_count"],
            "new_items_count": len(inserted_ids),
            "inserted_ids": inserted_ids
        }
        
    except Exception as e:
        error_msg = f"Error in reprioritize_deltas_for_report: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

def regenerate_explanations_for_report(filename):
    """
    Regenerate explanations for an existing newsletter's monthly_reporting items.
    This function finds the report by filename and regenerates explanations for all its items.
    Now also includes perplexity context enrichment and newsletter text generation.
    
    Args:
        filename: The newsletter filename (e.g., "monthly_report_0_2024_01.html")
        
    Returns:
        Status dictionary with success/error information
    """
    logger.info(f"Regenerating explanations for newsletter: {filename}")
    
    try:
        # Step 1: Find the existing report and get its monthly_reporting items
        def find_report_items_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Find the report by original filename
            cursor.execute("""
                SELECT r.id as report_id, mr.id as item_id
                FROM reports r
                JOIN monthly_reporting mr ON r.id = mr.report_id
                WHERE r.original_filename = %s
                ORDER BY r.created_at DESC, mr.priority ASC
            """, (filename,))
            
            results = cursor.fetchall()
            cursor.close()
            
            if not results:
                return None
                
            report_id = results[0]['report_id']
            item_ids = [row['item_id'] for row in results]
            
            logger.info(f"Found report {report_id} with {len(item_ids)} items to explain")
            return {
                "report_id": report_id,
                "item_ids": item_ids
            }
        
        # Execute the find operation
        result = execute_with_connection(
            operation=find_report_items_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] != "success" or not result["result"]:
            return {"status": "error", "message": f"Newsletter with filename '{filename}' not found or has no items"}
        
        report_info = result["result"]
        item_ids = report_info["item_ids"]
        
        # Step 2: Generate explanations for all items
        logger.info(f"Generating explanations for {len(item_ids)} items")
        explanation_result = generate_explanations(item_ids)
        
        if explanation_result.get("status") != "success":
            return {"status": "error", "message": f"Failed to generate explanations: {explanation_result.get('message')}"}
        
        # Step 3: Get perplexity context for all items
        logger.info(f"Getting perplexity context for {len(item_ids)} items")
        
        def get_items_for_context_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Get all items with their data for perplexity context
            cursor.execute("""
                SELECT * FROM monthly_reporting WHERE id = ANY(%s)
                ORDER BY priority
            """, (item_ids,))
            
            items = cursor.fetchall()
            cursor.close()
            
            # Convert to list of dictionaries for perplexity context function
            report_items = []
            for item in items:
                report_items.append({
                    "metric": item["metric_name"],
                    "group": item["group_value"],
                    "recent_mean": item["recent_mean"],
                    "comparison_mean": item["comparison_mean"],
                    "percent_change": item["percent_change"],
                    "explanation": item["explanation"],
                    "report_text": item.get("report_text", "")
                })
            
            return report_items
        
        # Execute the get items operation
        items_result = execute_with_connection(
            operation=get_items_for_context_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if items_result["status"] != "success":
            logger.warning(f"Failed to get items for perplexity context: {items_result.get('message')}")
        else:
            # Get perplexity context
            context_result = get_perplexity_context(items_result["result"])
            
            if context_result.get("status") == "success":
                # Update items with perplexity context
                logger.info("Updating items with perplexity context")
                
                def update_items_with_context_operation(connection):
                    cursor = connection.cursor()
                    updated_count = 0
                    
                    for item in context_result["report_items"]:
                        # Find the corresponding database item
                        cursor.execute("""
                            SELECT id, metadata FROM monthly_reporting 
                            WHERE metric_name = %s AND group_value = %s AND id = ANY(%s)
                        """, (item["metric"], item["group"], item_ids))
                        
                        db_item = cursor.fetchone()
                        if db_item:
                            item_id, current_metadata = db_item
                            
                            # Parse existing metadata
                            if current_metadata:
                                if isinstance(current_metadata, str):
                                    try:
                                        metadata = json.loads(current_metadata)
                                    except:
                                        metadata = {}
                                else:
                                    metadata = current_metadata
                            else:
                                metadata = {}
                            
                            # Add perplexity context
                            if "perplexity_context" in item.get("metadata", {}):
                                metadata["perplexity_context"] = item["metadata"]["perplexity_context"]
                            
                            # Add perplexity response data
                            if "perplexity_response" in item.get("metadata", {}):
                                metadata["perplexity_response"] = item["metadata"]["perplexity_response"]
                            
                            # Update the database
                            cursor.execute("""
                                UPDATE monthly_reporting 
                                SET metadata = %s 
                                WHERE id = %s
                            """, (json.dumps(metadata), item_id))
                            
                            updated_count += 1
                    
                    connection.commit()
                    cursor.close()
                    return updated_count
                
                # Execute the update operation
                update_result = execute_with_connection(
                    operation=update_items_with_context_operation,
                    db_host=DB_HOST,
                    db_port=DB_PORT,
                    db_name=DB_NAME,
                    db_user=DB_USER,
                    db_password=DB_PASSWORD
                )
                
                if update_result["status"] == "success":
                    logger.info(f"Successfully updated {update_result['result']} items with perplexity context")
                else:
                    logger.warning(f"Failed to update items with perplexity context: {update_result.get('message')}")
            else:
                logger.warning(f"Failed to get perplexity context: {context_result.get('message')}")
        
        # Step 4: Generate newsletter text for all items
        logger.info(f"Generating newsletter text for {len(item_ids)} items")
        
        # Import the generate_report_text function
        from tools.generate_report_text import generate_report_text
        from webChat import client, AGENT_MODEL
        
        text_result = generate_report_text(
            item_ids, 
            execute_with_connection, 
            load_prompts, 
            AGENT_MODEL, 
            client, 
            logger
        )
        
        if text_result.get("status") != "success":
            logger.warning(f"Failed to generate newsletter text: {text_result.get('message')}")
        else:
            logger.info(f"Successfully generated newsletter text: {text_result.get('message')}")
        
        logger.info(f"Successfully regenerated explanations, context, and newsletter text for newsletter {filename}")
        return {
            "status": "success",
            "message": f"Successfully regenerated explanations, perplexity context, and newsletter text for {len(item_ids)} items in newsletter {filename}",
            "report_id": report_info["report_id"],
            "explained_items_count": len(item_ids),
            "explanations_generated": explanation_result.get("status") == "success",
            "context_enriched": context_result.get("status") == "success" if 'context_result' in locals() else False,
            "newsletter_text_generated": text_result.get("status") == "success" if 'text_result' in locals() else False
        }
        
    except Exception as e:
        error_msg = f"Error in regenerate_explanations_for_report: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

if __name__ == "__main__":
    # Run the monthly report process
    # You can adjust the max_report_items to control how many items appear in the report
    result = run_monthly_report_process(max_report_items=3, district="0")
    print(json.dumps(result, indent=2, cls=PathEncoder)) 
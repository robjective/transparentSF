from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from generate_dashboard_metrics import main as generate_metrics
from dotenv import load_dotenv
import re
import glob
from typing import List, Optional
import uvicorn
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from tools.db_utils import get_postgres_connection

# --- Logging Configuration Moved Up ---
# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file in the current directory
dotenv_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Configure logging based on environment variable
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
# Map to Python logging levels for potential internal use if needed
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_level_map.get(log_level_str, logging.INFO)

# Use Python logging getLogger for initial setup messages BEFORE Uvicorn takes over
# These might still appear depending on exact timing, but Uvicorn config below is key
temp_logger = logging.getLogger("__main__") 
# Optionally set a level here if you *want* these startup messages regardless of .env
# temp_logger.setLevel(logging.INFO) 

temp_logger.debug(f"Current directory: {current_dir}") # Use temp_logger
temp_logger.info(f"Loaded environment variables from {dotenv_path}") # Use temp_logger
temp_logger.info(f"Root log level determined from .env: {log_level_str}") # Use temp_logger

# Define logs directory
logs_dir = os.path.join(current_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# --- Uvicorn Logging Configuration Dictionary ---

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Preserve existing loggers
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Keep your format
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
             "use_colors": None,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr", # Log general messages to stderr
        },
        "access": {
             "formatter": "access",
             "class": "logging.StreamHandler",
             "stream": "ext://sys.stdout", # Log access messages to stdout
        },
        "weekly_analysis_file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": os.path.join(logs_dir, 'weekly_metric_analysis.log'),
            "mode": "a",
            "encoding": "utf-8",
        },
         # Optional: Define a file handler if you want Uvicorn to manage file logging
         # "file": {
         #     "formatter": "default",
         #     "class": "logging.FileHandler",
         #     "filename": os.path.join(logs_dir, 'app.log'), # Make sure logs_dir is defined if used
         # },
    },
    "loggers": {
        # Root logger configuration
        "": {
            "handlers": ["default"], # Use handlers defined above (e.g., "default", "file")
            "level": log_level_str, # Use uppercase level
            "propagate": True, # Allow propagation if needed
        },
        # Uvicorn's own loggers
        "uvicorn": {"handlers": ["default"], "level": log_level_str, "propagate": False}, # Use uppercase level
        "uvicorn.error": {"level": log_level_str, "propagate": True}, # Use uppercase level
        "uvicorn.access": {"handlers": ["access"], "level": log_level_str, "propagate": False}, # Use uppercase level

        # Explicitly set levels for known application/library loggers if needed
        # This ensures they obey the overall level setting
        "webChat": {"level": log_level_str, "propagate": True},
        "backend": {"level": log_level_str, "propagate": True},
        "anomalyAnalyzer": {"level": log_level_str, "propagate": True},
        "generate_weekly_analysis": {"level": log_level_str, "propagate": True},
        "metrics_manager": {"level": log_level_str, "propagate": True},
        "monthly_report": {"level": log_level_str, "propagate": True},
        
        # Third-party library loggers - set to WARNING or higher to suppress INFO logs
        "httpx": {"level": log_level_str, "propagate": True},
        "urllib3": {"level": log_level_str, "propagate": True},
        "requests": {"level": log_level_str, "propagate": True},
        "openai": {"level": log_level_str, "propagate": True},
        "anthropic": {"level": log_level_str, "propagate": True},
        "psycopg2": {"level": log_level_str, "propagate": True},
        "sqlalchemy": {"level": log_level_str, "propagate": True},
        "pandas": {"level": log_level_str, "propagate": True},
        "numpy": {"level": log_level_str, "propagate": True},
        "matplotlib": {"level": log_level_str, "propagate": True},
        "plotly": {"level": log_level_str, "propagate": True},
        "PIL": {"level": log_level_str, "propagate": True},
        "Pillow": {"level": log_level_str, "propagate": True},
        "jinja2": {"level": log_level_str, "propagate": True},
        "fastapi": {"level": log_level_str, "propagate": True},
        "starlette": {"level": log_level_str, "propagate": True},
        "pydantic": {"level": log_level_str, "propagate": True},
        "asyncio": {"level": log_level_str, "propagate": True},
        "aiohttp": {"level": log_level_str, "propagate": True},
        "websockets": {"level": log_level_str, "propagate": True},
        "schedule": {"level": log_level_str, "propagate": True},
        "dotenv": {"level": log_level_str, "propagate": True},
        
        # Weekly analysis modules - add file handler for these
        "tools.analysis.weekly": {"handlers": ["default", "weekly_analysis_file"], "level": log_level_str, "propagate": False},
        "tools.analysis.weekly.analysis_engine": {"handlers": ["default", "weekly_analysis_file"], "level": log_level_str, "propagate": False},
        "tools.analysis.weekly.data_processor": {"handlers": ["default", "weekly_analysis_file"], "level": log_level_str, "propagate": False},
        "tools.analysis.weekly.time_utils": {"handlers": ["default", "weekly_analysis_file"], "level": log_level_str, "propagate": False},
        "tools.analysis.weekly.report_generator": {"handlers": ["default", "weekly_analysis_file"], "level": log_level_str, "propagate": False},
        "tools.analysis.weekly.scheduler": {"handlers": ["default", "weekly_analysis_file"], "level": log_level_str, "propagate": False},
        
        # Other tools modules
        "tools": {"level": log_level_str, "propagate": True},
        "tools.db_utils": {"level": log_level_str, "propagate": True},
        "tools.data_fetcher": {"level": log_level_str, "propagate": True},
        "tools.anomaly_detection": {"level": log_level_str, "propagate": True},
        "tools.genChart": {"level": log_level_str, "propagate": True},
        "tools.genChartdw": {"level": log_level_str, "propagate": True},
        "tools.store_anomalies": {"level": log_level_str, "propagate": True},
        
        # Routes modules
        "routes": {"level": log_level_str, "propagate": True},
        "routes.dw_charts": {"level": log_level_str, "propagate": True},
        "routes.database_admin": {"level": log_level_str, "propagate": True},
        "routes.weekly_analysis": {"level": log_level_str, "propagate": True},
        
        # Agents modules
        "agents": {"level": log_level_str, "propagate": True},
        "agents.explainer_agent": {"level": log_level_str, "propagate": True},
    },
}
# --- End Logging Configuration ---

# Get the main logger for use within main.py itself
# This logger will inherit the configuration from LOGGING_CONFIG via Uvicorn
logger = logging.getLogger(__name__)

# --- End Uvicorn Logging Configuration Dictionary ---

# Import routers
from webChat import router as webchat_router
from backend import router as backend_router, set_templates, get_chart_by_metric, get_chart_data
from metrics_manager import router as metrics_router
from anomalyAnalyzer import router as anomaly_analyzer_router, set_templates as set_anomaly_templates
from routes.dw_charts import router as dw_charts_router
from routes.database_admin import router as database_admin_router, set_templates as set_database_admin_templates
from evals_routes import router as evals_router, set_templates as set_evals_templates
from routes.weekly_analysis import router as weekly_analysis_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dashboard.transparentsf.com",
        "https://transparentsf.com",
        "https://platform.transparentsf.com",
        "https://*.replit.app",
        "http://localhost:8081",
        "https://*.replit.dev",
        "http://c8f21de5-50d5-4932-9f30-db3b51e8af74-00-3cldcjjo20ql9.riker.replit.dev",
        "https://0ea7c615-87bb-43af-8dc3-9d1816811571-00-3jirn7oguyp0q.picard.replit.dev"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
    max_age=3600,
)

# Initialize templates with absolute path
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
logger.debug(f"Templates directory: {os.path.join(current_dir, 'templates')}")

# Mount static files
static_dir = os.path.join(current_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
logger.debug(f"Static directory: {static_dir}")

# Mount output directory
output_dir = os.path.join(current_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a custom route for monthly files BEFORE mounting the static directory
@app.get("/output/monthly/{district}/{metric_id}.md", response_class=HTMLResponse)
async def get_monthly_file(district: str, metric_id: str):
    """
    Serve monthly analysis files only. No fallback to annual files.
    """
    # Try to get the monthly file only
    monthly_path = os.path.join(output_dir, "monthly", district, f"{metric_id}.md")
    if os.path.exists(monthly_path):
        logger.debug(f"Serving monthly file from {monthly_path}")
        with open(monthly_path, 'r') as f:
            return f.read()

    # If monthly doesn't exist, return a 404 error
    logger.error(f"Monthly file not found for district {district}, metric {metric_id}")
    raise HTTPException(status_code=404, detail=f"Monthly analysis file not found for metric {metric_id}")

# Now mount the static directory after defining our custom route
app.mount("/output", StaticFiles(directory=output_dir), name="output")
logger.debug(f"Output directory: {output_dir}")

# Define OUTPUT_DIR constant for compatibility with other modules
OUTPUT_DIR = output_dir

# Mount weekly output directory
weekly_dir = os.path.join(output_dir, "weekly")
if not os.path.exists(weekly_dir):
    os.makedirs(weekly_dir)
app.mount("/weekly", StaticFiles(directory=weekly_dir), name="weekly")
logger.debug(f"Weekly output directory: {weekly_dir}")

# Mount logs directory
logs_dir = os.path.join(current_dir, "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
app.mount("/logs", StaticFiles(directory=logs_dir), name="logs")
logger.debug(f"Logs directory: {logs_dir}")

# Define metrics data directory
metrics_dir = os.path.join(current_dir, "data/dashboard")
logger.debug(f"Metrics directory: {metrics_dir}")

@app.get("/api/metrics/{filename}")
async def get_metrics(filename: str):
    """Serve metrics data from JSON files."""
    # Ensure filename ends with .json
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    # Check if this is a district file request (district_X.json)
    if filename.startswith('district_') and '_' in filename:
        # Extract district number
        district_str = filename.split('_')[1].split('.')[0]
        
        # Try to serve top_level.json from the district subfolder
        new_file_path = os.path.join(output_dir, 'dashboard', district_str, 'top_level.json')
        logger.debug(f"Attempting to read metrics from: {new_file_path}")
        try:
            with open(new_file_path, 'r') as f:
                data = json.load(f)
                return JSONResponse(content=data)
        except FileNotFoundError:
            logger.error(f"Metrics file not found: {new_file_path}")
            raise HTTPException(status_code=404, detail=f"Metrics file '{filename}' not found")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in metrics file: {new_file_path}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON in metrics file '{filename}'")
        except Exception as e:
            logger.error(f"Error reading metrics file {new_file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Check if this is a metric ID request (metric_id.json)
    # Format could be something like "crime_incidents_ytd.json"
    elif '_' in filename and not filename.startswith('district_'):
        # This is likely a metric file request
        # We need to determine which district it belongs to
        # First, try to find it in each district folder
        for district_num in range(12):  # 0-11 for citywide and districts 1-11
            district_str = str(district_num)
            metric_file_path = os.path.join(output_dir, 'dashboard', district_str, filename)
            if os.path.exists(metric_file_path):
                logger.debug(f"Found metric file in district {district_str}: {metric_file_path}")
                try:
                    with open(metric_file_path, 'r') as f:
                        data = json.load(f)
                        return JSONResponse(content=data)
                except Exception as e:
                    logger.error(f"Error reading metric file {metric_file_path}: {str(e)}")
                    # Continue to try other districts
        
        # If we get here, the metric file wasn't found in any district folder
        logger.error(f"Metric file not found in any district folder: {filename}")
        raise HTTPException(status_code=404, detail=f"Metric file '{filename}' not found")

def build_enhanced_queries_from_db():
    """Build enhanced dashboard queries from the database and existing JSON files."""
    try:
        # First load the base dashboard queries structure
        queries_file = os.path.join(current_dir, "data", "dashboard", "dashboard_queries.json")
        if not os.path.exists(queries_file):
            logger.error(f"Base dashboard queries file not found: {queries_file}")
            return None
        
        with open(queries_file, 'r') as f:
            dashboard_queries = json.load(f)
        
        # Load existing enhanced queries to preserve custom settings
        enhanced_queries_file = os.path.join(current_dir, "data", "dashboard", "dashboard_queries_enhanced.json")
        existing_enhanced_queries = {}
        if os.path.exists(enhanced_queries_file):
            try:
                with open(enhanced_queries_file, 'r') as f:
                    existing_enhanced_queries = json.load(f)
                logger.info("Loaded existing enhanced queries to preserve custom settings")
            except Exception as e:
                logger.warning(f"Could not load existing enhanced queries: {str(e)}")
        
        # Get database connection
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to connect to database for enhanced queries")
            # Fallback to file-based approach
            if existing_enhanced_queries:
                return existing_enhanced_queries
            return None
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Get all datasets from database
        cursor.execute("""
            SELECT endpoint, title, category, description, columns
            FROM datasets 
            WHERE is_active = true
        """)
        
        datasets_by_endpoint = {}
        for row in cursor.fetchall():
            datasets_by_endpoint[row['endpoint']] = row
        
        cursor.close()
        connection.close()
        
        # Build enhanced queries structure
        enhanced_queries = {}
        id_counter = 1
        
        # Process each category
        for category, subcategories in dashboard_queries.items():
            enhanced_queries[category] = {}
            
            # Process each subcategory
            for subcategory, subcategory_data in subcategories.items():
                enhanced_queries[category][subcategory] = {}
                
                # Copy existing subcategory data
                for key, value in subcategory_data.items():
                    if key != "queries":
                        enhanced_queries[category][subcategory][key] = value
                
                # Add queries with enhancements
                enhanced_queries[category][subcategory]["queries"] = {}
                
                # Process each query
                if "queries" in subcategory_data:
                    for metric_name, metric_data in subcategory_data["queries"].items():
                        # Assign a numeric ID
                        metric_id = id_counter
                        id_counter += 1
                        
                        # Create enhanced metric data
                        enhanced_metric = {
                            "id": metric_id,
                            **metric_data  # Copy all existing metric data
                        }
                        
                        # Get endpoint
                        endpoint_id = metric_data.get("endpoint")
                        
                        # Check if we already have custom category_fields in existing enhanced queries
                        existing_category_fields = None
                        if existing_enhanced_queries:
                            # Try to find the metric with this name or endpoint in existing enhanced queries
                            for ex_category in existing_enhanced_queries.values():
                                for ex_subcategory in ex_category.values():
                                    for ex_metric_name, ex_metric_data in ex_subcategory.get("queries", {}).items():
                                        if (ex_metric_name == metric_name or 
                                            (endpoint_id and ex_metric_data.get("endpoint") == endpoint_id)):
                                            existing_category_fields = ex_metric_data.get("category_fields")
                                            if existing_category_fields:
                                                logger.debug(f"Found existing category_fields for metric {metric_name}")
                                            break
                        
                        # If we have existing custom category_fields, use them
                        if existing_category_fields:
                            enhanced_metric["category_fields"] = existing_category_fields
                            logger.debug(f"Using existing category_fields for metric {metric_name}")
                        else:
                            # Otherwise, get dataset info from database if endpoint is available
                            if endpoint_id and endpoint_id in datasets_by_endpoint:
                                dataset_row = datasets_by_endpoint[endpoint_id]
                                
                                # Extract location and category fields from database
                                location_fields = []
                                category_fields = []
                                
                                columns = dataset_row['columns'] or []
                                
                                for column in columns:
                                    field_name = column.get("fieldName", "").lower() if column.get("fieldName") else ""
                                    description = column.get("description", "").lower() if column.get("description") else ""
                                    name = column.get("name", "")
                                    
                                    # Only include supervisor_district as a location field
                                    if field_name == "supervisor_district":
                                        location_fields.append({
                                            "name": name,
                                            "fieldName": column.get("fieldName", ""),
                                            "description": column.get("description", "")
                                        })
                                    
                                    # Identify category fields, excluding neighborhood/district fields
                                    if (any(term in field_name for term in ["category", "type", "class", "group", "status", "subcategory"]) or \
                                        "categor" in description or "classif" in description or "type of" in description) and \
                                        not any(term in field_name for term in ["neighborhood", "district"]):
                                        category_fields.append({
                                            "name": name,
                                            "fieldName": column.get("fieldName", ""),
                                            "description": column.get("description", "")
                                        })
                                
                                # Add dataset info from database
                                enhanced_metric["dataset_title"] = dataset_row['title'] or ""
                                enhanced_metric["dataset_category"] = dataset_row['category'] or ""
                                enhanced_metric["location_fields"] = location_fields
                                enhanced_metric["category_fields"] = category_fields
                                
                                logger.debug(f"Added database dataset info to metric {metric_name}")
                            else:
                                logger.warning(f"No dataset info available in database for metric {metric_name} with endpoint {endpoint_id}")
                        
                        # Add to enhanced queries
                        enhanced_queries[category][subcategory]["queries"][metric_name] = enhanced_metric
        
        logger.info("Successfully built enhanced queries from database")
        return enhanced_queries
        
    except Exception as e:
        logger.error(f"Error building enhanced queries from database: {str(e)}")
        # Fallback to existing file if available
        if existing_enhanced_queries:
            logger.info("Falling back to existing enhanced queries file")
            return existing_enhanced_queries
        return None

# Removed - now handled by metrics_manager.py

@app.get("/api/metric-id-mapping")
async def get_metric_id_mapping():
    """Serve the metric ID mapping file."""
    file_path = os.path.join(current_dir, "data", "dashboard", "metric_id_mapping.json")
    logger.debug(f"Attempting to read metric ID mapping from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return JSONResponse(content=data)
    except FileNotFoundError:
        logger.error(f"Metric ID mapping file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Metric ID mapping file not found")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in metric ID mapping file: {file_path}")
        raise HTTPException(status_code=500, detail="Invalid JSON in metric ID mapping file")
    except Exception as e:
        logger.error(f"Error reading metric ID mapping file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/metric/{metric_id}")
async def get_metric_by_id(metric_id: int):
    """Get metric details by its numeric ID."""
    try:
        from tools.db_utils import get_postgres_connection
        import psycopg2.extras
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get the metric details from the database
        cursor.execute("""
            SELECT 
                m.id,
                m.metric_name as name,
                m.metric_key,
                m.category,
                m.subcategory,
                m.endpoint,
                m.summary,
                m.definition,
                m.data_sf_url,
                m.ytd_query,
                m.metric_query,
                m.dataset_title,
                m.dataset_category,
                m.show_on_dash,
                m.item_noun,
                m.greendirection,
                m.location_fields,
                m.category_fields,
                m.metadata,
                m.is_active
            FROM metrics m
            WHERE m.id = %s
        """, (metric_id,))
        
        metric = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not metric:
            logger.error(f"Metric ID {metric_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Metric ID {metric_id} not found")
        
        # Convert to dictionary and handle JSON fields
        metric_dict = dict(metric)
        
        # Ensure location_fields and category_fields are lists
        if metric_dict.get('location_fields') is None:
            metric_dict['location_fields'] = []
        if metric_dict.get('category_fields') is None:
            metric_dict['category_fields'] = []
            
        return JSONResponse(content=metric_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metric {metric_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metric: {str(e)}")

@app.get("/api/district/{district_id}/metric/{metric_id}")
async def get_district_metric(district_id: str, metric_id: str):
    """Serve a specific metric for a specific district."""
    # Ensure metric_id ends with .json
    if not metric_id.endswith('.json'):
        metric_id = f"{metric_id}.json"
    
    # Construct the file path
    file_path = os.path.join(output_dir, 'dashboard', district_id, metric_id)
    logger.debug(f"Attempting to read district metric from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return JSONResponse(content=data)
    except FileNotFoundError:
        logger.error(f"District metric file not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"Metric '{metric_id}' for district '{district_id}' not found")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in district metric file: {file_path}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in metric file '{metric_id}' for district '{district_id}'")
    except Exception as e:
        logger.error(f"Error reading district metric file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/district/{district_id}")
async def get_district_top_level(district_id: str):
    """Serve the top-level metrics for a specific district."""
    # Construct the file path
    file_path = os.path.join(output_dir, 'dashboard', district_id, 'top_level.json')
    logger.debug(f"Attempting to read district top-level metrics from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return JSONResponse(content=data)
    except FileNotFoundError:
        logger.error(f"District top-level file not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"Top-level metrics for district '{district_id}' not found")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in district top-level file: {file_path}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in top-level file for district '{district_id}'")
    except Exception as e:
        logger.error(f"Error reading district top-level file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/district/{district_id}/metrics")
async def list_district_metrics(district_id: str):
    """List all available metrics for a specific district."""
    # Construct the directory path
    dir_path = os.path.join(output_dir, 'dashboard', district_id)
    logger.debug(f"Attempting to list metrics in directory: {dir_path}")
    
    try:
        if not os.path.exists(dir_path):
            logger.error(f"District directory not found: {dir_path}")
            raise HTTPException(status_code=404, detail=f"District '{district_id}' not found")
        
        # Get all JSON files in the directory except top_level.json
        metric_files = [f for f in os.listdir(dir_path) if f.endswith('.json') and f != 'top_level.json']
        
        # Extract metric IDs (remove .json extension)
        metric_ids = [f[:-5] for f in metric_files]
        
        # Return the list of metric IDs
        return JSONResponse(content={"metrics": metric_ids})
    except Exception as e:
        logger.error(f"Error listing district metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/districts")
async def list_districts():
    """List all available districts."""
    try:
        districts = []
        dashboard_dir = os.path.join(output_dir, 'dashboard')
        
        # Look for numbered directories (0-11) representing districts
        for item in os.listdir(dashboard_dir):
            item_path = os.path.join(dashboard_dir, item)
            # Check if it's a directory and represents a district number
            if os.path.isdir(item_path) and item.isdigit() and 0 <= int(item) <= 11:
                district_id = item
                # Load the district's top-level data for its name
                top_level_path = os.path.join(item_path, 'top_level.json')
                try:
                    with open(top_level_path, 'r') as f:
                        top_level_data = json.load(f)
                        district_name = top_level_data.get('name', f"District {district_id}")
                        districts.append({
                            'id': district_id,
                            'name': district_name
                        })
                except:
                    # If we can't read top_level.json, just use the district number
                    districts.append({
                        'id': district_id,
                        'name': f"District {district_id}"
                    })
        
        # Sort districts by ID, ensuring numeric sorting (district 0 is first, then 1-11)
        districts.sort(key=lambda d: int(d['id']))
        return JSONResponse(content=districts)
    except Exception as e:
        logger.error(f"Error listing districts: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Root route to serve index.html
@app.get("/")
async def root(request: Request):
    """Serve the main chat interface."""
    logger.debug("Serving index.html template")
    return templates.TemplateResponse("index.html", {"request": request})

# Set templates for backend router
set_templates(templates)
logger.debug("Set templates for backend router")

# Set templates for anomaly analyzer router
set_anomaly_templates(templates)
logger.debug("Set templates for anomaly analyzer router")

# Mount anomaly analyzer router
app.include_router(anomaly_analyzer_router, prefix="/anomaly-analyzer", tags=["anomaly-analyzer"])
logger.debug("Included anomaly analyzer router at /anomaly-analyzer")

# Mount DW charts router
app.include_router(dw_charts_router, prefix="/backend", tags=["dw-charts"])
logger.debug("Included DW charts router at /backend")

# Include routers
app.include_router(webchat_router, prefix="/chat")  # Chat router at /chat
logger.debug("Included chat router at /chat")

# Mount backend router without trailing slash
app.include_router(backend_router, prefix="/backend", tags=["backend"])
logger.debug("Included backend router at /backend")

# Mount metrics router (routes define their own paths)
app.include_router(metrics_router, tags=["metrics"])
logger.debug("Included metrics router")

# Mount evals router
set_evals_templates(templates)
app.include_router(evals_router, prefix="/backend", tags=["evals"])
logger.debug("Included evals router at /backend")

# Mount database admin router
set_database_admin_templates(templates)
app.include_router(database_admin_router, prefix="/backend", tags=["database-admin"])
logger.debug("Included database admin router at /backend")

# Mount weekly analysis router
app.include_router(weekly_analysis_router, tags=["weekly-analysis"])
logger.debug("Included weekly analysis router")

# Add redirect for anomaly analyzer without trailing slash
@app.get("/anomaly-analyzer")
async def redirect_to_anomaly_analyzer():
    return RedirectResponse(url="/anomaly-analyzer/")

# Add direct routes for API endpoints that need to be accessible without the /backend prefix
@app.post("/api/add_subscriber")
async def api_add_subscriber(request: Request):
    """Forward API subscriber requests to the backend endpoint."""
    logger.debug("Forwarding /api/add_subscriber request to backend handler")
    
    # Import the backend handler directly
    from backend import add_subscriber as backend_add_subscriber
    
    # Forward the request to the backend handler
    return await backend_add_subscriber(request)

@app.get("/api/chart-by-metric")
async def forward_chart_by_metric(
    metric_id: str,
    district: int = 0,
    period_type: str = 'year',
    group_field: str = None,
    groups: str = None
):
    """
    Forward requests from /api/chart-by-metric to /backend/api/chart-by-metric
    """
    return await get_chart_by_metric(metric_id, district, period_type, group_field, groups)

@app.get("/api/chart/{chart_id}")
async def forward_chart_data(chart_id: int):
    """
    Forward requests from /api/chart/{chart_id} to /backend/api/chart/{chart_id}
    """
    return await get_chart_data(chart_id)

@app.get("/api/district-shapes/{district_type}")
async def get_district_shapes(district_type: str, district_ids: Optional[List[str]] = None):
    """
    Endpoint to fetch district shape data from the city's API.
    
    Args:
        district_type: Type of district ("supervisor" or "police")
        district_ids: Optional list of specific district IDs to fetch
        
    Returns:
        JSON response containing district polygon data
    """
    logger.info(f"API request for {district_type} district shapes, districts: {district_ids}")
    
    try:
        # Validate district type
        if district_type.lower() not in ["supervisor", "police"]:
            logger.error(f"Invalid district type: {district_type}")
            raise HTTPException(status_code=400, detail=f"Invalid district type. Must be 'supervisor' or 'police'")
        
        # Set up endpoint ID and field name based on district type
        if district_type.lower() == "supervisor":
            endpoint_id = "f2zs-jevy"  # Supervisor districts endpoint (2022)
            id_field = "sup_dist"
        else:  # Police districts
            endpoint_id = "wkhw-cjsf"  # Police districts endpoint
            id_field = "district"
        
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
            raise HTTPException(status_code=500, detail=f"Error fetching district shapes: {result['error']}")
        
        # Check if we got data
        dataset = context_variables.get("dataset")
        if dataset is None or dataset.empty:
            logger.error("No district data retrieved")
            raise HTTPException(status_code=404, detail="No district data found")
        
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
                district_shapes[district_id] = {
                    "type": "Feature",
                    "properties": {
                        "district": district_id,
                        "name": f"District {district_id}" if district_type.lower() == "supervisor" else district_id
                    },
                    "geometry": geometry
                }
            else:
                logger.warning(f"No polygon data found for district {district_id}")
        
        # Return GeoJSON FeatureCollection
        geojson_response = {
            "type": "FeatureCollection",
            "features": list(district_shapes.values())
        }
        
        # Add metadata to the response
        response_data = {
            "status": "success",
            "district_type": district_type,
            "count": len(district_shapes),
            "geojson": geojson_response
        }
        
        return response_data
    
    except Exception as e:
        logger.error(f"Error processing district shapes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing district shapes: {str(e)}")

@app.get("/api/district-ids/{district_type}")
async def get_district_ids(district_type: str):
    """
    Endpoint to fetch just the list of district IDs of a specific type.
    
    Args:
        district_type: Type of district ("supervisor" or "police")
        
    Returns:
        JSON response containing list of district IDs
    """
    logger.info(f"API request for {district_type} district IDs")
    
    try:
        # Validate district type
        if district_type.lower() not in ["supervisor", "police"]:
            logger.error(f"Invalid district type: {district_type}")
            raise HTTPException(status_code=400, detail=f"Invalid district type. Must be 'supervisor' or 'police'")
        
        # Set up endpoint ID and field name based on district type
        if district_type.lower() == "supervisor":
            endpoint_id = "f2zs-jevy"  # Supervisor districts endpoint (2022)
            id_field = "sup_dist"
        else:  # Police districts
            endpoint_id = "wkhw-cjsf"  # Police districts endpoint
            id_field = "district"
        
        # Import the set_dataset function
        from tools.data_fetcher import set_dataset
        
        # Build the SoQL query - only select the district ID field
        query = f"SELECT {id_field}"
        
        # Create context variables dictionary
        context_variables = {}
        
        # Use set_dataset to fetch the data
        result = set_dataset(context_variables, endpoint=endpoint_id, query=query)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Error fetching district IDs: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error fetching district IDs: {result['error']}")
        
        # Check if we got data
        dataset = context_variables.get("dataset")
        if dataset is None or dataset.empty:
            logger.error("No district data retrieved")
            raise HTTPException(status_code=404, detail="No district data found")
        
        # Extract the district IDs
        district_ids = dataset[id_field].astype(str).unique().tolist()
        
        # Return the list of district IDs
        return {
            "status": "success",
            "district_type": district_type,
            "count": len(district_ids),
            "district_ids": district_ids
        }
    
    except Exception as e:
        logger.error(f"Error processing district IDs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing district IDs: {str(e)}")

async def schedule_metrics_generation():
    """Schedule metrics generation to run daily at 5 AM and 11 AM."""
    while True:
        try:
            # Calculate time until next run (either 5 AM or 11 AM)
            now = datetime.now()
            target_5am = now.replace(hour=5, minute=0, second=0, microsecond=0)
            target_11am = now.replace(hour=11, minute=0, second=0, microsecond=0)
            
            # If we're past 11 AM, set targets to next day
            if now.hour >= 11:
                target_5am += timedelta(days=1)
                target_11am += timedelta(days=1)
            # If we're past 5 AM but before 11 AM, only adjust the 5 AM target
            elif now.hour >= 5:
                target_5am += timedelta(days=1)
            
            # Find the next closest target time
            next_run = min(target_5am, target_11am)
            
            # Wait until the next scheduled time
            wait_seconds = (next_run - now).total_seconds()
            logger.info(f"Next metrics generation scheduled for {next_run} (in {wait_seconds/3600:.2f} hours)")
            await asyncio.sleep(wait_seconds)
            
            # Generate metrics
            logger.info("Starting scheduled metrics generation...")
            try:
                generate_metrics()
                logger.info("Metrics generation completed successfully")
            except Exception as e:
                logger.error(f"Error during metrics generation: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in metrics generation scheduler: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying if there's an error

async def cleanup_logs():
    """Trim log files to keep only the last 7 days of logs. Runs daily."""
    while True:
        try:
            # Schedule to run daily at 11:30 AM
            now = datetime.now()
            
            # Calculate time until tomorrow at 11:30 AM
            target_run = now.replace(hour=11, minute=30, second=0, microsecond=0)
            if now.hour > 11 or (now.hour == 11 and now.minute >= 30):
                target_run += timedelta(days=1)  # If it's already past 11:30 AM, schedule for tomorrow
            
            # Wait until the scheduled time
            wait_seconds = (target_run - now).total_seconds()
            logger.info(f"Next log cleanup scheduled for {target_run} (in {wait_seconds/3600:.2f} hours)")
            await asyncio.sleep(wait_seconds)
            
            # Trim log files
            logger.info("Starting scheduled log cleanup...")
            try:
                # Get all log files
                log_files = glob.glob(os.path.join(logs_dir, "*.log"))
                trimmed_count = 0
                
                # Calculate the cutoff date (7 days ago)
                cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                
                for log_file in log_files:
                    try:
                        # Create a temporary file to store recent logs
                        temp_file = log_file + ".temp"
                        with open(log_file, 'r') as original, open(temp_file, 'w') as temp:
                            # For each line, check if it's newer than the cutoff date
                            kept_lines = 0
                            for line in original:
                                # Try to extract the date from the line (assuming standard log format)
                                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                                if date_match and date_match.group(1) >= cutoff_date:
                                    # Keep lines newer than or equal to the cutoff date
                                    temp.write(line)
                                    kept_lines += 1
                            
                        # If we kept any lines or the original file is empty, replace the original with the temp file
                        if kept_lines > 0 or os.path.getsize(log_file) == 0:
                            os.replace(temp_file, log_file)
                            trimmed_count += 1
                            logger.debug(f"Trimmed log file: {log_file}, kept {kept_lines} lines")
                        else:
                            # If we didn't keep any lines, don't replace the file (it might be a special case)
                            os.remove(temp_file)
                            logger.debug(f"No recent logs found in {log_file}, file not modified")
                            
                    except Exception as e:
                        logger.error(f"Error trimming log file {log_file}: {str(e)}")
                        # Clean up temp file if it exists
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                
                logger.info(f"Log cleanup completed. Trimmed {trimmed_count} log files to keep entries from the last 7 days.")
            except Exception as e:
                logger.error(f"Error during log cleanup: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in log cleanup scheduler: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying if there's an error

@app.on_event("startup")
async def startup_event():
    # Start background job cleanup task
    from background_jobs import start_cleanup_task
    start_cleanup_task()
    
    # Existing startup code
    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            routes.append(f"{route.path} [{route.methods}]")
        elif hasattr(route, "path"):
            routes.append(f"{route.path} [Mount]")
    
    logger.debug("Registered routes:")
    for route in routes:
        logger.debug(f"  {route}")
        
    # Start the metrics generation scheduler
    asyncio.create_task(schedule_metrics_generation())
    logger.info("Started metrics generation scheduler")
    
    # Start the log cleanup scheduler
    asyncio.create_task(cleanup_logs())
    logger.info("Started log cleanup scheduler")

if __name__ == "__main__":
    # Use the temporary logger for this final startup message
    temp_logger.info("Starting server with Uvicorn...") 
    # Pass the config dictionary to uvicorn.run, remove log_level argument
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=False,  # Disable reload for deployment
                workers=1,     # Use single worker for deployment stability
                log_config=LOGGING_CONFIG) 
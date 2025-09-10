from fastapi import APIRouter, HTTPException, Request, Depends, Response
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote
import secrets # Added for password comparison
from fastapi.security import HTTPBasic, HTTPBasicCredentials # Added for basic auth

import os
import sys
import json
import math
from datetime import datetime, date
import pytz
import subprocess  # ADDED
import glob
import shutil  # Add this import at the top with other imports


# periodic_analysis import removed - functionality deprecated
import logging
from generate_dashboard_metrics import main as generate_metrics
from tools.data_fetcher import fetch_data_from_api
import pandas as pd
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
import time
from tools.enhance_dashboard_queries import enhance_dashboard_queries  # Add this import
import re
import psycopg2.extras

# Import centralized database utilities
from tools.db_utils import get_postgres_connection, execute_with_connection
import asyncio

from monthly_report import expand_chart_references, generate_email_compatible_report
from typing import Optional
from tools.analysis.weekly import run_weekly_analysis
from background_jobs import job_manager

# The evals module now handles agent creation internally
from evals import run_all_evals, run_model_comparison
from ai.evals import run_single_eval_langchain

# Use the root logger configured elsewhere (e.g., in monthly_report.py)
logger = logging.getLogger(__name__)

def clean_json_value(value):
    """Clean a value to ensure it's JSON serializable."""
    if isinstance(value, (int, str, bool, type(None))):
        return value
    elif isinstance(value, float):
        # Handle infinite and NaN values
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    elif isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: clean_json_value(v) for k, v in value.items()}
    else:
        # Convert other types to string
        return str(value)

router = APIRouter()
templates = None  # Will be set by main.py
security = HTTPBasic() # Added for basic auth

# Define the absolute path to the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the /ai directory
output_dir = os.path.join(script_dir, 'output')  # /ai/output

logger.info(f"Script directory: {script_dir}")
logger.info(f"Output directory: {output_dir}")

def get_db_connection():
    """Helper function to create a database connection using environment variables."""
    try:
        return get_postgres_connection()
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logger.info("Templates set in backend router")

# --- ADDED: Basic Authentication Dependency ---
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    correct_username = secrets.compare_digest(credentials.username, os.environ.get("BACKEND_USERNAME", "admin"))
    correct_password = secrets.compare_digest(credentials.password, os.environ.get("BACKEND_PASSWORD", "password"))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
# --- END: Basic Authentication Dependency ---

def find_output_files_for_endpoint(endpoint: str, output_dir: str):
    """Search the output directory for files matching the endpoint."""
    logger.debug(f"Searching for files matching endpoint '{endpoint}' in directory: {output_dir}")
    output_files = {}
    
    # Remove .json extension if present
    endpoint = endpoint.replace('.json', '')
    logger.debug(f"Cleaned endpoint: {endpoint}")
    
    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return output_files
        
    # Log all files in output directory
    logger.debug("Full output directory contents:")
    for root, dirs, files in os.walk(output_dir):
        #logger.debug(f"Directory: {root}")
        #logger.debug(f"Files: {files}")
        
        matching_files = {}
        latest_timestamp = 0
        
        # Find the most recent file of each type
        for file in files:
            if endpoint in file:
                file_path = os.path.join(root, file)
                file_timestamp = os.path.getmtime(file_path)
                
                if file.endswith('.html'):
                    if 'html' not in matching_files or file_timestamp > latest_timestamp:
                        matching_files['html'] = file_path
                        latest_timestamp = file_timestamp
                elif file.endswith('.md'):
                    if 'md' not in matching_files or file_timestamp > latest_timestamp:
                        matching_files['md'] = file_path
                        latest_timestamp = file_timestamp
                elif file.endswith('_summary.txt'):
                    if 'txt' not in matching_files or file_timestamp > latest_timestamp:
                        matching_files['txt'] = file_path
                        latest_timestamp = file_timestamp
        
        if matching_files:
            rel_path = os.path.relpath(root, output_dir)
            output_files[rel_path] = matching_files
            logger.debug(f"Added matching files for path {rel_path}: {matching_files}")
     #   else:
            # logger.debug(f"No matching files found in directory {root}")

    # logger.debug(f"Final output files found: {json.dumps(output_files, indent=2)}")
    return output_files

def get_output_file_url(file_path: str, output_dir: str):
    """
    Convert a file path to a URL accessible via the FastAPI app.
    """
    if not isinstance(file_path, (str, bytes, os.PathLike)):
        logger.warning(f"Invalid file_path type: {type(file_path)}. Expected string path.")
        return None
    try:
        relative_path = os.path.relpath(file_path, output_dir)
        url_path = "/output/" + "/".join(quote(part) for part in relative_path.split(os.sep))
        logger.debug(f"Converted file path '{file_path}' to URL '{url_path}'")
        return url_path
    except Exception as e:
        logger.error(f"Error converting file path to URL: {str(e)}")
        return None



def get_datasets_from_db():
    """
    Get datasets from the PostgreSQL database instead of files.
    Returns an empty list since the template doesn't actually use datasets.
    """
    logger.debug("Getting datasets from database (currently returning empty list)")
    return []

def get_md_file_date(output_files):
    """Get the most recent date from MD or HTML files in output_files."""
    latest_date = None
    for folder, files in output_files.items():
        for file_type, file_path in files.items():
            if file_type in ['md', 'html']:
                try:
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                except Exception:
                    continue
    return latest_date

@router.get("/")
async def backend_root(request: Request, username: str = Depends(get_current_username)): # Added username dependency
    """Serve the backend interface."""
    logger.debug("Backend root route called")
    if templates is None:
        logger.error("Templates not initialized in backend router")
        raise RuntimeError("Templates not initialized")
    
    # Load datasets here
    try:
        datasets = get_datasets_from_db()
        logger.debug(f"Loaded {len(datasets)} datasets")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        datasets = []

    logger.debug("Serving backend.html template")
    return templates.TemplateResponse("backend.html", {
        "request": request,
        "datasets": datasets
    })



@router.get("/run_analysis/{endpoint}")
async def run_analysis(endpoint: str, period_type: str = 'year'):
    """Run analysis for a given endpoint."""
    logger.info(f"Run analysis called for endpoint: {endpoint} with period_type: {period_type}")
    
    # Remove .json extension if present
    endpoint = endpoint.replace('.json', '')
    
    error_log = []
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create period-specific output folder
        period_folder_map = {
            'year': 'annual',
            'month': 'monthly',
            'day': 'daily',
            'ytd': 'ytd'
        }
        
        if period_type not in period_folder_map:
            raise ValueError(f"Invalid period_type: {period_type}. Must be one of: {', '.join(period_folder_map.keys())}")
            
        period_folder = period_folder_map[period_type]
        period_output_dir = os.path.join(output_dir, period_folder)
        os.makedirs(period_output_dir, exist_ok=True)
        
        logger.info(f"Periodic analysis functionality has been deprecated. Endpoint: {endpoint}, period_type: {period_type}")
        # export_for_endpoint functionality removed - using LangChain-based analysis instead
        
        if error_log:
            logger.warning(f"Analysis completed with warnings for endpoint '{endpoint}': {error_log}")
            return JSONResponse({
                'status': 'warning', 
                'message': 'Analysis completed with warnings.', 
                'errors': error_log
            })

        logger.info(f"Analysis for endpoint '{endpoint}' completed successfully.")
        return JSONResponse({
            'status': 'success', 
            'message': f'Analysis for endpoint {endpoint} completed successfully.'
        })
    except Exception as e:
        logger.exception(f"Error running analysis for endpoint '{endpoint}': {str(e)}")
        return JSONResponse({
            'status': 'error', 
            'message': str(e)
        }, status_code=500)

@router.get("/get-updated-links/{endpoint}")
async def get_updated_links(endpoint: str):
    """Returns the updated output links for a single endpoint."""
    logger.debug(f"Get updated links called for endpoint: {endpoint}")
    
    # Get the absolute path to the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    logger.debug(f"Looking for files in output directory: {output_dir}")
    
    # Verify directory exists and is accessible
    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return JSONResponse({'links': {}, 'message': 'Output directory not found'})
    
    if not os.path.isdir(output_dir):
        logger.error(f"Output path exists but is not a directory: {output_dir}")
        return JSONResponse({'links': {}, 'message': 'Invalid output directory'})
    
    try:
        # List directory contents
        dir_contents = os.listdir(output_dir)
        logger.debug(f"Output directory contents: {dir_contents}")
    except Exception as e:
        logger.error(f"Error reading output directory: {str(e)}")
        return JSONResponse({'links': {}, 'message': f'Error reading output directory: {str(e)}'})
    
    output_files = find_output_files_for_endpoint(endpoint, output_dir)
    logger.debug(f"Found output files: {json.dumps(output_files, indent=2)}")
    
    links = {}
    for folder, files in output_files.items():
        folder_links = {}
        for file_type, file_path in files.items():
            url = get_output_file_url(file_path, output_dir)
            if url:
                folder_links[file_type] = url
                logger.debug(f"Added URL for {file_type}: {url}")
        if folder_links:
            links[folder] = folder_links
    
    logger.debug(f"Returning links: {json.dumps(links, indent=2)}")
    return JSONResponse({'links': links})

# --- ADDED: Route to reload vector DB
@router.get("/reload_vector_db")
async def reload_vector_db():
    """
    Reload the vector DB by running the vector_loader_periodic.py script.
    """
    logger.debug("Reload vector DB called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "vector_loader_periodic.py")
        log_file = os.path.join(script_dir, "logs", "vector_loader.log")
        
        # Clear the log file before running
        with open(log_file, 'w') as f:
            f.write("")  # Clear the file
            
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        
        # Read the log file content
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
        except Exception as e:
            log_content = f"Error reading log file: {str(e)}"
            
        if result.returncode == 0:
            logger.info("Vector DB reloaded successfully.")
            return JSONResponse({
                "status": "success",
                "message": "Vector DB reloaded successfully.",
                "output": result.stdout,
                "log_content": log_content
            })
        else:
            logger.error(f"Failed to reload Vector DB: {result.stderr}")
            return JSONResponse({
                "status": "error",
                "message": "Failed to reload Vector DB.",
                "output": result.stderr,
                "log_content": log_content
            })
    except Exception as e:
        logger.exception(f"Error reloading Vector DB: {str(e)}")
        return JSONResponse({
            "status": "error", 
            "message": str(e),
            "log_content": "Error occurred before log file could be read"
        })

@router.get("/reload_sfpublic")
async def reload_sfpublic():
    """Reload the SF Public Data collection."""
    logger.debug("Reload SF Public Data collection called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "vector_loader_sfpublic.py")
        log_file = os.path.join(script_dir, "logs", "vector_loader.log")
        
        # Clear the log file before running
        with open(log_file, 'w') as f:
            f.write("")  # Clear the file
            
        # Run the script with proper working directory and environment
        logger.info(f"Running script: {script_path}")
        logger.info(f"Working directory: {script_dir}")
        logger.info(f"Python executable: {sys.executable}")
        
        result = subprocess.run(
            [sys.executable, script_path],  # Use sys.executable instead of "python"
            capture_output=True, 
            text=True,
            cwd=script_dir,  # Set working directory to script_dir
            env=os.environ.copy()  # Pass current environment variables
        )
        
        # Read the log file content
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
        except Exception as e:
            log_content = f"Error reading log file: {str(e)}"
            
        if result.returncode == 0:
            logger.info("SF Public Data collection reloaded successfully.")
            return JSONResponse({
                "status": "success",
                "message": "SF Public Data collection reloaded successfully.",
                "output": result.stdout,
                "log_content": log_content
            })
        else:
            logger.error(f"Failed to reload SF Public Data collection: {result.stderr}")
            return JSONResponse({
                "status": "error",
                "message": "Failed to reload SF Public Data collection.",
                "output": result.stderr,
                "log_content": log_content
            })
    except Exception as e:
        logger.exception(f"Error reloading SF Public Data collection: {str(e)}")
        return JSONResponse({
            "status": "error", 
            "message": str(e),
            "log_content": "Error occurred before log file could be read"
        })

@router.get("/dataset-json/{filename:path}")
async def get_dataset_json(filename: str):
    """Serve the JSON file for a dataset."""
    logger.debug(f"Get dataset JSON called for filename: {filename}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Remove any leading '../' from the filename for security
    clean_filename = filename.lstrip('.').lstrip('/')
    
    # Try the exact path first
    file_path = os.path.join(current_dir, clean_filename)
    
    if not os.path.exists(file_path):
        logger.error(f"JSON file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        file_path,
        media_type="application/json",
        filename=os.path.basename(clean_filename)
    )

@router.get("/endpoint-json/{endpoint}")
async def get_endpoint_json(endpoint: str):
    """
    Get the JSON data for a specific endpoint.
    """
    logger.debug(f"Get endpoint JSON called for '{endpoint}'")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # Find the most recent JSON file for this endpoint
        json_files = []
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.startswith(endpoint) and file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        if not json_files:
            raise HTTPException(status_code=404, detail=f"No JSON files found for endpoint '{endpoint}'")
        
        # Sort by modification time, most recent first
        json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        most_recent_file = json_files[0]
        
        try:
            with open(most_recent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return JSONResponse(data)
        except IOError:
            raise HTTPException(status_code=500, detail="Error reading file")
            
    except Exception as e:
        logger.exception(f"Error serving endpoint JSON for '{endpoint}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def count_tokens(text: str) -> int:
    """Rough estimate of token count based on words and punctuation."""
    # Split on whitespace and count punctuation as separate tokens
    words = text.split()
    # Add count of common punctuation marks that might be separate tokens
    punctuation_count = text.count('.') + text.count(',') + text.count('!') + \
                       text.count('?') + text.count(';') + text.count(':')
    return len(words) + punctuation_count

@router.get("/get-aggregated-summary")
async def get_aggregated_summary():
    """
    Aggregate all summary .txt files from the output directory and return their combined content.
    Also includes current YTD metrics data.
    """
    logger.debug("Get aggregated summary called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # Find all summary .txt files
        summary_files = []
        total_content = ""
        
        # Walk through output directory and find all summary text files
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('_summary.txt'):
                    summary_files.append(os.path.join(root, file))
        
        if not summary_files:
            logger.debug("No summary files found")
            return JSONResponse({"content": "", "token_count": 0})
        
        # Aggregate content from all summary files
        aggregated_content = []
        for i, summary_file in enumerate(summary_files, 1):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_content += content
                
                # Add file header and content to aggregated content
                filename = os.path.basename(summary_file)
                section = f"\n{'='*80}\nSummary {i}: {filename}\n{'='*80}\n\n{content}\n\n"
                aggregated_content.append(section)
                    
            except Exception as e:
                logger.error(f"Error reading summary file {summary_file}: {str(e)}")
                continue
        
        # Add YTD metrics data
        ytd_file = os.path.join(script_dir, 'data', 'dashboard', 'ytd_metrics.json')
        if os.path.exists(ytd_file):
            try:
                with open(ytd_file, 'r', encoding='utf-8') as f:
                    ytd_data = json.load(f)
                
                # Format YTD metrics as text
                ytd_text = "\n" + "="*80 + "\nYear-to-Date (YTD) Metrics Summary:\n" + "="*80 + "\n\n"
                
                # Add citywide metrics
                if "districts" in ytd_data and "0" in ytd_data["districts"]:
                    citywide = ytd_data["districts"]["0"]
                    ytd_text += f"Citywide Statistics:\n"
                    
                    for category in citywide.get("categories", []):
                        ytd_text += f"\n{category['category']}:\n"
                        for metric in category.get("metrics", []):
                            name = metric.get("name", "")
                            this_year = metric.get("thisYear", 0)
                            last_year = metric.get("lastYear", 0)
                            last_date = metric.get("lastDataDate", "")
                            
                            # Calculate percent change
                            if last_year != 0:
                                pct_change = ((this_year - last_year) / last_year) * 100
                                change_text = f"({pct_change:+.1f}% vs last year)"
                            else:
                                change_text = "(no prior year data)"
                            
                            ytd_text += f"- {name}: {this_year:,} {change_text} as of {last_date}\n"
                
                aggregated_content.append(ytd_text)
                logger.info("Successfully added YTD metrics to aggregated summary")
                
            except Exception as e:
                logger.error(f"Error processing YTD metrics: {str(e)}")
        
        # Join all content
        final_content = "".join(aggregated_content)
        
        # Calculate total token count
        token_count = count_tokens(total_content)
        
        return JSONResponse({
            "content": final_content,
            "token_count": token_count
        })
        
    except Exception as e:
        logger.exception(f"Error creating aggregated summary: {str(e)}")
        return JSONResponse({"status": "error", "message": str(e)})

@router.get("/get-log-files")
async def get_log_files():
    """Get a list of all log files in the logs directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        
        if not os.path.exists(logs_dir):
            return JSONResponse({"files": []})
            
        log_files = []
        for file in os.listdir(logs_dir):
            if file.endswith('.log') or file.endswith('.txt'):
                file_path = os.path.join(logs_dir, file)
                # Get file stats
                stats = os.stat(file_path)
                log_files.append({
                    "name": file,
                    "size": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
                
        # Sort by modification time, most recent first
        log_files.sort(key=lambda x: x["modified"], reverse=True)
        
        return JSONResponse({"files": log_files})
    except Exception as e:
        logger.exception(f"Error getting log files: {str(e)}")
        return JSONResponse({"error": str(e)})

@router.get("/disk-space")
async def get_disk_space():
    """Get disk space information for the current drive."""
    try:
        # Get disk usage for the current directory's drive
        total, used, free = shutil.disk_usage(os.path.abspath(os.sep))
        
        return JSONResponse({
            "total": total,
            "available": free,
            "used": used
        })
    except Exception as e:
        logger.exception(f"Error getting disk space: {str(e)}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@router.post("/clear-html-files")
async def clear_html_files():
    """Delete all HTML files from the output directory and its subdirectories."""
    logger.debug("Clear HTML files called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # Count of deleted files
        deleted_count = 0
        
        # Walk through all subdirectories
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        logger.info(f"Successfully deleted {deleted_count} HTML files")
        return JSONResponse({
            "status": "success",
            "message": f"Successfully deleted {deleted_count} HTML files"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing HTML files: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.post("/clear-period-files/{period_type}")
async def clear_period_files(period_type: str):
    """Delete all files from a specific period folder (monthly or annual)."""
    logger.debug(f"Clear {period_type} files called")
    
    # Validate period_type
    period_folder_map = {
        'year': 'annual',
        'month': 'monthly',
        'day': 'daily',
        'ytd': 'ytd'
    }
    
    if period_type not in period_folder_map:
        logger.error(f"Invalid period_type: {period_type}")
        return JSONResponse({
            "status": "error",
            "message": f"Invalid period_type: {period_type}. Must be one of: {', '.join(period_folder_map.keys())}"
        }, status_code=400)
        
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        period_folder = period_folder_map[period_type]
        output_dir = os.path.join(script_dir, 'output', period_folder)
        
        # Check if directory exists
        if not os.path.exists(output_dir):
            logger.debug(f"Directory does not exist: {output_dir}")
            return JSONResponse({
                "status": "success",
                "message": f"No {period_folder} files to delete"
            })
        
        # Count of deleted files
        deleted_count = 0
        
        # Delete all files and subdirectories
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            # Delete empty directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    # Only remove if empty
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.debug(f"Deleted directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting directory {dir_path}: {str(e)}")
        
        logger.info(f"Successfully deleted {deleted_count} files from {period_folder} folder")
        return JSONResponse({
            "status": "success", 
            "message": f"Successfully deleted {deleted_count} files from {period_folder} folder"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing {period_type} files: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.get("/generate_ytd_metrics")
async def generate_ytd_metrics():
    """Generate YTD metrics on demand."""
    logger.debug("Generate YTD metrics called")
    try:
        # Import and call the main function in a background thread to prevent blocking
        from generate_dashboard_metrics import main as generate_metrics_main
        import asyncio
        
        logger.info("Starting YTD metrics generation in background...")
        
        # Run the main function in a thread pool to prevent blocking the web server
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_metrics_main)
        
        logger.info("YTD metrics generated successfully")
        return JSONResponse({
            "status": "success",
            "message": "YTD metrics generated successfully"
        })
    except Exception as e:
        logger.exception(f"Error generating YTD metrics: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/generate_ytd_metrics_async")
async def generate_ytd_metrics_async():
    """Generate YTD metrics asynchronously with job tracking."""
    logger.debug("Generate YTD metrics async called")
    try:
        from generate_dashboard_metrics import main as generate_metrics_main
        
        # Create a background job
        job_id = job_manager.create_job("ytd_metrics", "Generate YTD dashboard metrics")
        
        # Start the job in the background
        asyncio.create_task(job_manager.run_job(job_id, generate_metrics_main))
        
        return JSONResponse({
            "status": "success",
            "message": "YTD metrics generation started",
            "job_id": job_id
        })
    except Exception as e:
        logger.exception(f"Error starting YTD metrics generation: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a background job."""
    job = job_manager.get_job(job_id)
    if not job:
        return JSONResponse({
            "status": "error",
            "message": "Job not found"
        }, status_code=404)
    
    return JSONResponse({
        "status": "success",
        "job": job.to_dict()
    })

@router.get("/jobs")
async def list_jobs():
    """List all background jobs."""
    jobs = job_manager.get_all_jobs()
    return JSONResponse({
        "status": "success",
        "jobs": [job.to_dict() for job in jobs.values()]
    })

@router.get("/generate_weekly_report")
async def generate_weekly_report():
    """Generate weekly report on demand."""
    logger.debug("Generate weekly report called")
    try:
        # Import the necessary functions
        from tools.analysis.weekly import run_weekly_analysis
        # from tools.analysis.weekly import generate_weekly_newsletter
        import asyncio
        
        # Run the weekly analysis with specific metrics instead of empty list
        # Using metric IDs 1-3 as defaults, which are usually the most reliable metrics
        logger.info("Running weekly analysis for default metrics in background")
        metrics_to_process = ["1", "2", "3"]  # Convert to strings as the function expects string IDs
        
        # Run in background thread to prevent blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            run_weekly_analysis,
            metrics_to_process,
            True  # process_districts
        )
        
        # Generate a newsletter
        # Newsletter generation temporarily disabled
        # newsletter_path = generate_weekly_newsletter(results)
        
        if results and len(results) > 0:
            successful = len(results)
            failed = 0
            
            # Get the metric IDs for the successful analyses
            metric_ids = [result.get('metric_id', 'unknown') for result in results]
            
            # After generating the report, redirect to the weekly-report page
            logger.info(f"Weekly report generated successfully with {successful} metrics")
            return RedirectResponse(url="/weekly-report", status_code=302)
        else:
            error_message = "Weekly analysis returned no results. Check logs for details."
            logger.error(error_message)
            return JSONResponse({
                "status": "error",
                "message": error_message
            }, status_code=500)
    except ImportError as e:
        error_message = f"Could not import weekly analysis functions: {str(e)}"
        logger.error(error_message)
        return JSONResponse({
            "status": "error",
            "message": error_message
        }, status_code=500)
    except Exception as e:
        error_message = f"Error generating weekly report: {str(e)}"
        logger.exception(error_message)
        return JSONResponse({
            "status": "error",
            "message": error_message
        }, status_code=500)

@router.get("/run_all_metrics")
async def run_all_metrics(period_type: str = 'year'):
    """Run analysis for all available endpoints."""
    logger.info(f"Run all metrics called with period_type: {period_type}")
    
    # Handle weekly analysis separately
    if period_type == 'week':
        try:
            logger.info("Running weekly analysis for all default metrics in background")
            
            # Import necessary functions
            from tools.analysis.weekly import run_weekly_analysis
            import asyncio
            
            # Run the weekly analysis in background thread to prevent blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, run_weekly_analysis, None, True)
            
            if results and len(results) > 0:
                # Generate a newsletter
                # Newsletter generation temporarily disabled
                # newsletter_path = generate_weekly_newsletter(results)
                
                successful = len(results)
                failed = 0
                
                # Get the metric IDs for the successful analyses
                metric_ids = [result.get('metric_id', 'unknown') for result in results]
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Weekly analysis completed successfully for {successful} metrics.",
                    "results": {
                        "total": successful,
                        "successful": successful,
                        "failed": 0,
                        "metrics": metric_ids,
                        # "newsletter_path": newsletter_path
                    }
                })
            else:
                return JSONResponse({
                    "status": "error",
                    "message": "Weekly analysis returned no results. Check logs for details.",
                    "results": {
                        "total": 0,
                        "successful": 0,
                        "failed": 0
                    }
                })
        except ImportError as e:
            logger.error(f"Could not import weekly analysis functions: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Missing required module for weekly analysis: {str(e)}"
            }, status_code=500)
        except Exception as e:
            logger.exception(f"Error running weekly analysis: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Error running weekly analysis: {str(e)}"
            }, status_code=500)
    
    # For other period types, use the run_all_metrics.py script
    try:
        # Run the run_all_metrics.py script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "run_all_metrics.py")
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(script_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Run the script
        logger.info(f"Running script: {script_path}")
        
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("All metrics processed successfully")
            
            # Parse the output to find success/failure counts
            output = result.stdout
            
            # Try to extract success and failure counts from the last line
            lines = output.strip().split('\n')
            last_line = lines[-1] if lines else ""
            
            # Default values if parsing fails
            successful = 0
            failed = 0
            total = 0
            
            # Try to parse the completion message: "Completed all analyses. Successful: X, Failed: Y"
            if "Completed all analyses" in last_line:
                parts = last_line.split()
                for i, part in enumerate(parts):
                    if part == "Successful:":
                        try:
                            successful = int(parts[i+1].rstrip(','))
                        except (IndexError, ValueError):
                            pass
                    elif part == "Failed:":
                        try:
                            failed = int(parts[i+1])
                        except (IndexError, ValueError):
                            pass
            
            total = successful + failed
            
            # Return success with counts
            return JSONResponse({
                "status": "success" if failed == 0 else "partial",
                "message": f"Processed {successful} of {total} metrics successfully. {failed} metrics failed.",
                "results": {
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "output": output
                }
            })
        else:
            logger.error(f"Error running all metrics: {result.stderr}")
            return JSONResponse({
                "status": "error",
                "message": "Error running all metrics",
                "error": result.stderr
            }, status_code=500)
    except Exception as e:
        logger.exception(f"Error running all metrics: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/query")
async def query_page(request: Request):
    """Serve the query interface page."""
    logger.debug("Query page route called")
    if templates is None:
        logger.error("Templates not initialized in backend router")
        raise RuntimeError("Templates not initialized")
    
    return templates.TemplateResponse("query.html", {
        "request": request
    })

@router.post("/execute-query")
async def execute_query(request: Request):
    """Execute a query and return results as HTML or Markdown table."""
    try:
        form_data = await request.form()
        endpoint = form_data.get('endpoint', '').strip()
        query = form_data.get('query', '').strip()
        format_type = form_data.get('format', 'html')
        
        if not endpoint or not query:
            return JSONResponse({
                'status': 'error',
                'message': 'Both endpoint and query are required'
            })
            
        # Execute query using existing data_fetcher
        result = fetch_data_from_api({'endpoint': endpoint, 'query': query})
        
        if 'error' in result:
            return JSONResponse({
                'status': 'error',
                'message': result['error'],
                'queryURL': result.get('queryURL')
            })
            
        # Convert data to DataFrame
        df = pd.DataFrame(result['data'])
        
        # Generate table based on format
        if format_type == 'markdown':
            table = df.to_markdown(index=False)
        else:  # html
            table = df.to_html(index=False, classes=['table', 'table-striped', 'table-hover'])
            
        return JSONResponse({
            'status': 'success',
            'table': table,
            'queryURL': result.get('queryURL'),
            'rowCount': len(df)
        })
        
    except Exception as e:
        logger.exception(f"Error executing query: {str(e)}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        })

@router.get("/logs/{filename}")
async def get_log_file(filename: str):
    """
    Serve a log file directly.
    """
    logger.info(f"Request for log file: {filename}")
    
    try:
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Script directory: {script_dir}")
        
        # Check if this is a monthly report file
        if filename.startswith("monthly_report_") and filename.endswith(".md"):
            # Construct the path to the reports directory
            reports_dir = os.path.join(script_dir, 'output', 'reports')
            file_path = os.path.join(reports_dir, filename)
            
            logger.info(f"Monthly report file path: {file_path}")
            logger.info(f"Reports directory: {reports_dir}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            
            # Security check to ensure the file is within the reports directory
            if not os.path.abspath(file_path).startswith(os.path.abspath(reports_dir)):
                logger.error(f"Security check failed: {file_path} is not within {reports_dir}")
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Check if the file exists
            if os.path.exists(file_path):
                logger.info(f"Serving monthly report file: {file_path}")
                return FileResponse(file_path, media_type="text/markdown")
            else:
                # Try alternative path resolution
                alt_path = os.path.join(os.getcwd(), 'ai', 'output', 'reports', filename)
                logger.info(f"Trying alternative path: {alt_path}")
                logger.info(f"Alternative path exists: {os.path.exists(alt_path)}")
                
                if os.path.exists(alt_path):
                    logger.info(f"Serving monthly report file from alternative path: {alt_path}")
                    return FileResponse(alt_path, media_type="text/markdown")
                else:
                    logger.error(f"Monthly report file not found: {file_path} or {alt_path}")
                    raise HTTPException(status_code=404, detail=f"Monthly report file not found: {filename}")
        
        # For regular log files, check the logs directory
        file_path = os.path.join(script_dir, 'logs', filename)
        
        # Security check to ensure the file is within the logs directory
        logs_dir = os.path.join(script_dir, 'logs')
        if not os.path.abspath(file_path).startswith(os.path.abspath(logs_dir)):
            logger.error(f"Security check failed: {file_path} is not within {logs_dir}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if the file exists
        if os.path.exists(file_path):
            logger.info(f"Serving log file: {file_path}")
            return FileResponse(file_path)
        else:
            logger.error(f"Log file not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@router.post("/execute-qdrant-query")
async def execute_qdrant_query(request: Request):
    """Execute a semantic search query using Qdrant."""
    try:
        form_data = await request.form()
        collection_name = form_data.get('collection', '').strip()
        query = form_data.get('query', '').strip()
        
        # Connect to Qdrant
        qdrant = QdrantClient(host='localhost', port=6333)
        
        # If no parameters provided, just return collections list
        if not collection_name and not query:
            collections = [c.name for c in qdrant.get_collections().collections]
            collections.sort()  # Sort alphabetically
            logger.info(f"Available collections: {collections}")
            
            # Get collection info
            collection_info = {}
            for coll in collections:
                try:
                    info = qdrant.get_collection(coll)
                    points_count = info.points_count
                    collection_info[coll] = {
                        "points_count": points_count,
                        "vector_size": info.config.params.vectors.size
                    }
                    logger.info(f"Collection {coll}: {points_count} points, vector size: {info.config.params.vectors.size}")
                except Exception as e:
                    logger.error(f"Error getting info for collection {coll}: {e}")
                    
            return JSONResponse({
                'status': 'success',
                'collections': collections,
                'collection_info': collection_info,
                'results': []
            })
        
        # For actual search, require both parameters
        if not collection_name or not query:
            return JSONResponse({
                'status': 'error',
                'message': 'Both collection and query are required for search'
            })
            
        limit = int(form_data.get('limit', '5'))
        logger.info(f"Searching collection '{collection_name}' for query: '{query}' (limit: {limit})")
            
        # Get embedding for query
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_vector = response.data[0].embedding
        logger.debug(f"Generated query vector of size {len(query_vector)}")
        
        # Get collection info before search
        collection_info = qdrant.get_collection(collection_name)
        logger.info(f"Collection {collection_name} has {collection_info.points_count} points")
        
        # Query Qdrant with lower score threshold
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.05  # Lower threshold to catch more potential matches
        )
        
        logger.info(f"Found {len(search_result)} results")
        
        # Format results with score interpretation
        results = []
        for hit in search_result:
            score = round(hit.score, 3)
            # Add score interpretation
            if score >= 0.5:
                relevance = "Very High"
            elif score >= 0.3:
                relevance = "High"
            elif score >= 0.2:
                relevance = "Medium"
            elif score >= 0.1:
                relevance = "Low"
            elif score >= 0.05:
                relevance = "Very Low"
            else:
                relevance = "Minimal"
                
            # Handle SFPublicData collection differently
            if collection_name == 'SFPublicData':
                # Format content from dataset fields
                content_parts = []
                
                title = hit.payload.get('title', '')
                if title:
                    content_parts.append(f"Title: {title}")
                
                description = hit.payload.get('description', '')
                if description:
                    content_parts.append(f"Description: {description}")
                
                url = hit.payload.get('url', '')
                if url:
                    content_parts.append(f"URL: {url}")
                
                endpoint = hit.payload.get('endpoint', '')
                if endpoint:
                    content_parts.append(f"Endpoint: {endpoint}")
                
                category = hit.payload.get('category', '')
                if category:
                    content_parts.append(f"Category: {category}")
                
                publishing_department = hit.payload.get('publishing_department', '')
                if publishing_department:
                    content_parts.append(f"Publishing Department: {publishing_department}")
                
                # Format columns information
                columns = hit.payload.get('columns', {})
                if columns:
                    column_list = []
                    for col_name, col_details in columns.items():
                        col_type = col_details.get('dataTypeName', '')
                        col_info = f"{col_name} ({col_type})"
                        column_list.append(col_info)
                    
                    if column_list:
                        content_parts.append(f"Columns: {', '.join(column_list)}")
                
                # Use endpoint as filename for SFPublicData
                filename = endpoint if endpoint else 'N/A'
                content = '\n'.join(content_parts)
            else:
                # For other collections, use the standard format
                filename = hit.payload.get('filename', 'N/A')
                content = hit.payload.get('content', 'No content')  # Show full content
                
            result = {
                'score': score,
                'relevance': relevance,
                'filename': filename,
                'content': content
            }
            logger.debug(f"Result: score={score} ({relevance}), file={result['filename']}")
            results.append(result)
            
        # Get list of available collections
        collections = [c.name for c in qdrant.get_collections().collections]
        collections.sort()  # Sort alphabetically
            
        return JSONResponse({
            'status': 'success',
            'results': results,
            'collections': collections,
            'query_info': {
                'collection': collection_name,
                'query': query,
                "vector_size": len(query_vector),
                'total_points': collection_info.points_count,
                'score_guide': {
                    '0.5+': 'Very High Relevance',
                    '0.3-0.5': 'High Relevance',
                    '0.2-0.3': 'Medium Relevance',
                    '0.1-0.2': 'Low Relevance',
                    '0.05-0.1': 'Very Low Relevance',
                    '<0.05': 'Minimal Relevance'
                }
            }
        })
        
    except Exception as e:
        logger.exception(f"Error executing Qdrant query: {str(e)}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        })

@router.delete("/delete-collection/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a Qdrant collection."""
    logger.debug(f"Delete collection called for: {collection_name}")
    try:
        # Connect to Qdrant
        qdrant = QdrantClient(host='localhost', port=6333)
        
        # Check if collection exists
        if not qdrant.collection_exists(collection_name):
            return JSONResponse({
                'status': 'error',
                'message': f'Collection {collection_name} does not exist'
            })
            
        # Delete collection
        qdrant.delete_collection(collection_name)
        time.sleep(2)  # Wait for deletion to complete
        
        # Verify deletion
        if not qdrant.collection_exists(collection_name):
            logger.info(f"Successfully deleted collection {collection_name}")
            return JSONResponse({
                'status': 'success',
                'message': f'Collection {collection_name} deleted successfully'
            })
        else:
            logger.error(f"Collection {collection_name} still exists after deletion attempt")
            return JSONResponse({
                'status': 'error',
                'message': f'Failed to delete collection {collection_name}'
            })
            
    except Exception as e:
        logger.exception(f"Error deleting collection {collection_name}: {str(e)}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        })

@router.get("/metric-control")
async def metric_control(request: Request):
    """Serve the metric control interface."""
    logger.debug("Metric control route called")
    if templates is None:
        logger.error("Templates not initialized in backend router")
        raise RuntimeError("Templates not initialized")
    
    return templates.TemplateResponse("metric_control.html", {
        "request": request
    })

# Removed - migrated to metrics_manager.py

# Removed - migrated to metrics_manager.py

# Removed - migrated to metrics_manager.py

# Removed - migrated to metrics_manager.py

@router.get("/get_output_files")
async def get_output_files(metric_id: str, district_id: str = '0', period_type: str = 'year'):
    """
    Get output files for a specific metric based on metric ID, district ID, and period type.
    Returns files organized by folder type (dashboard, monthly, annual, ytd, weekly).
    """
    try:
        logger.info(f"Fetching output files for metric ID: {metric_id}, district: {district_id}, period: {period_type}")
        
        # Define the directory structure
        output_folder = os.path.join(script_dir, 'output')
        district_dir = f"district_{district_id}"
        
        # Dictionary to store files by category
        result = {
            "dashboard": [],
            "monthly": [],
            "annual": [],
            "ytd": [],
            "weekly": []
        }
        
        # Check both directory patterns for each output type
        
        # Check dashboard folder - both directory structures
        dashboard_dir_new = os.path.join(output_folder, "dashboard", district_dir, metric_id)  # district_0/6/
        dashboard_dir_old = os.path.join(output_folder, "dashboard", district_id)  # 0/
        dashboard_file_old = os.path.join(dashboard_dir_old, f"{metric_id}.json")  # 0/6.json
        
        if os.path.exists(dashboard_dir_new):
            result["dashboard"] = [f for f in os.listdir(dashboard_dir_new) if os.path.isfile(os.path.join(dashboard_dir_new, f))]
            logger.debug(f"Found {len(result['dashboard'])} files in dashboard/district_{district_id}/{metric_id} folder")
        elif os.path.exists(dashboard_dir_old):
            # Check if the specific metric file exists in the old structure
            if os.path.exists(dashboard_file_old):
                result["dashboard"] = [f"{metric_id}.json"]
                logger.debug(f"Found metric file at dashboard/{district_id}/{metric_id}.json")
        
        # Check monthly folder - both directory structures
        monthly_dir_new = os.path.join(output_folder, "monthly", district_dir, metric_id)  # district_0/6/
        monthly_dir_old = os.path.join(output_folder, "monthly", district_id)  # 0/
        monthly_file_old = os.path.join(monthly_dir_old, f"{metric_id}.json")  # 0/6.json
        
        if os.path.exists(monthly_dir_new):
            result["monthly"] = [f for f in os.listdir(monthly_dir_new) if os.path.isfile(os.path.join(monthly_dir_new, f))]
            logger.debug(f"Found {len(result['monthly'])} files in monthly/district_{district_id}/{metric_id} folder")
        elif os.path.exists(monthly_dir_old):
            # Check if the specific metric file exists in the old structure
            if os.path.exists(monthly_file_old):
                result["monthly"] = [f"{metric_id}.json"]
                logger.debug(f"Found metric file at monthly/{district_id}/{metric_id}.json")
        
        # Check annual folder - both directory structures
        annual_dir_new = os.path.join(output_folder, "annual", district_dir, metric_id)  # district_0/6/
        annual_dir_old = os.path.join(output_folder, "annual", district_id)  # 0/
        annual_file_old = os.path.join(annual_dir_old, f"{metric_id}.json")  # 0/6.json
        
        if os.path.exists(annual_dir_new):
            result["annual"] = [f for f in os.listdir(annual_dir_new) if os.path.isfile(os.path.join(annual_dir_new, f))]
            logger.debug(f"Found {len(result['annual'])} files in annual/district_{district_id}/{metric_id} folder")
        elif os.path.exists(annual_dir_old):
            # Check if the specific metric file exists in the old structure
            if os.path.exists(annual_file_old):
                result["annual"] = [f"{metric_id}.json"]
                logger.debug(f"Found metric file at annual/{district_id}/{metric_id}.json")
        
        # Check ytd folder - both directory structures
        ytd_dir_new = os.path.join(output_folder, "ytd", district_dir, metric_id)  # district_0/6/
        ytd_dir_old = os.path.join(output_folder, "ytd", district_id)  # 0/
        ytd_file_old = os.path.join(ytd_dir_old, f"{metric_id}.json")  # 0/6.json
        
        if os.path.exists(ytd_dir_new):
            result["ytd"] = [f for f in os.listdir(ytd_dir_new) if os.path.isfile(os.path.join(ytd_dir_new, f))]
            logger.debug(f"Found {len(result['ytd'])} files in ytd/district_{district_id}/{metric_id} folder")
        elif os.path.exists(ytd_dir_old):
            # Check if the specific metric file exists in the old structure
            if os.path.exists(ytd_file_old):
                result["ytd"] = [f"{metric_id}.json"]
                logger.debug(f"Found metric file at ytd/{district_id}/{metric_id}.json")
        
        # Check weekly folder
        weekly_dir = os.path.join(output_folder, "weekly")
        
        # First try to find files with the metric ID in their filename
        if os.path.exists(weekly_dir):
            weekly_files = []
            for file in os.listdir(weekly_dir):
                # Look for files that start with the metric ID and have a date component
                # Format: metric_id_YYYY-MM-DD.md
                if file.startswith(f"{metric_id}_") and file.endswith(".md"):
                    weekly_files.append(file)
            result["weekly"] = weekly_files
            logger.debug(f"Found {len(result['weekly'])} files in weekly folder matching metric ID {metric_id}")
        
        # Log the total number of files found
        total_files = sum(len(files) for files in result.values())
        logger.info(f"Found a total of {total_files} output files for metric {metric_id}")
        
        return result
    except Exception as e:
        logger.error(f"Error fetching output files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch output files: {str(e)}"}
        )

@router.post("/enhance_queries")
async def enhance_queries():
    """Enhance dashboard queries with IDs and category fields."""
    try:
        # Define file paths - ensure we use absolute paths to avoid confusion
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Script directory: {script_dir}")
        
        # Set up paths for dashboard queries
        queries_file = os.path.join(script_dir, "data", "dashboard", "dashboard_queries.json")
        logger.info(f"Queries file path: {queries_file}")
        
        # Check if the queries file exists, if not try alternative paths
        if not os.path.exists(queries_file):
            logger.warning(f"Queries file not found at: {queries_file}")
            alt_queries_path = os.path.join(script_dir, "..", "ai", "data", "dashboard", "dashboard_queries.json")
            if os.path.exists(alt_queries_path):
                logger.info(f"Found queries file at alternate path: {alt_queries_path}")
                queries_file = alt_queries_path
            else:
                raise FileNotFoundError(f"Dashboard queries file not found at {queries_file} or {alt_queries_path}")
        
        # Use database for dataset info instead of files
        datasets_dir = None
        logger.info("Using database for dataset information")
        
        # Set up path for output file
        output_file = os.path.join(script_dir, "data", "dashboard", "dashboard_queries_enhanced.json")
        logger.info(f"Output file path: {output_file}")
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run the enhancement process
        enhance_dashboard_queries(queries_file, datasets_dir, output_file)
        
        # Verify the output file was created
        if os.path.exists(output_file):
            logger.info(f"Successfully created enhanced queries file at: {output_file}")
            
            # Check file size to ensure it's not empty
            file_size = os.path.getsize(output_file)
            logger.info(f"Enhanced queries file size: {file_size} bytes")
            
            if file_size == 0:
                logger.error("Enhanced queries file is empty!")
                return JSONResponse({
                    "status": "error",
                    "message": "Dashboard queries enhancement completed but produced an empty file"
                }, status_code=500)
        else:
            logger.error(f"Enhanced queries file was not created at: {output_file}")
            return JSONResponse({
                "status": "error",
                "message": f"Failed to create enhanced queries file at: {output_file}"
            }, status_code=500)
        
        return JSONResponse({
            "status": "success",
            "message": "Dashboard queries have been enhanced successfully"
        })
    except Exception as e:
        logger.exception(f"Error enhancing dashboard queries: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error enhancing dashboard queries: {str(e)}"
        }, status_code=500)

@router.post("/execute-postgres-query")
async def execute_postgres_query(request: Request):
    """Execute a PostgreSQL query and return results."""
    try:
        data = await request.json()
        query = data.get('query', '').strip()
        parameters = data.get('parameters', {})
        
        if not query:
            return JSONResponse({
                'status': 'error',
                'message': 'Query is required'
            })
        
        # Connect to PostgreSQL
        conn = get_db_connection()
        if not conn:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to connect to database"}
            )
        
        try:
            # Create a cursor with dictionary-like results
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Execute the query with parameters
            cursor.execute(query, parameters)
            
            # For non-SELECT queries, commit the transaction
            if not query.strip().upper().startswith('SELECT'):
                conn.commit()
                return JSONResponse({
                    'status': 'success',
                    'message': 'Query executed successfully',
                    'rowCount': cursor.rowcount
                })
            
            # For SELECT queries, fetch and return results
            results = cursor.fetchall()
            
            # Convert results to list of dictionaries and handle datetime serialization
            results_list = []
            for row in results:
                row_dict = dict(row)
                # Convert datetime objects to ISO format strings
                for key, value in row_dict.items():
                    if isinstance(value, (datetime, date)):
                        row_dict[key] = value.isoformat()
                results_list.append(row_dict)
            
            return JSONResponse({
                'status': 'success',
                'rowCount': len(results_list),
                'query': query,
                'results': results_list
            })
            
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.exception(f"Error executing PostgreSQL query: {str(e)}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

@router.post("/get-biggest-deltas")
async def get_biggest_deltas_api(request: Request):
    """Get metrics with the biggest deltas between time periods."""
    try:
        data = await request.json()
        current_period = data.get('current_period')
        comparison_period = data.get('comparison_period')
        limit = data.get('limit', 10)
        district = data.get('district')
        object_type = data.get('object_type')
        
        # Import function here to avoid circular imports
        from tools.store_time_series import get_biggest_deltas
        
        result = get_biggest_deltas(
            current_period=current_period,
            comparison_period=comparison_period,
            limit=limit,
            district=district,
            object_type=object_type
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.exception(f"Error getting biggest deltas: {str(e)}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

@router.post("/clear-postgres-data")
async def clear_postgres_data():
    """Clear all data from PostgreSQL database tables."""
    logger.debug("Clear PostgreSQL data called")
    try:
        # Connect to PostgreSQL
        conn = get_db_connection()
        if not conn:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to connect to database"}
            )
        
        table_count = 0
        
        try:
            # Create a cursor
            cursor = conn.cursor()
            
            # Get list of all tables
            cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Truncate each table (clear all data)
            for table in tables:
                try:
                    cursor.execute(f'TRUNCATE TABLE "{table}" CASCADE;')
                    table_count += 1
                    logger.debug(f"Truncated table: {table}")
                except Exception as e:
                    logger.error(f"Error truncating table {table}: {str(e)}")
            
            # Commit the transaction
            conn.commit()
            
        finally:
            cursor.close()
            conn.close()
        
        logger.info(f"Successfully cleared data from {table_count} tables in PostgreSQL database")
        return JSONResponse({
            "status": "success", 
            "message": f"Successfully cleared data from {table_count} tables in PostgreSQL database"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing PostgreSQL data: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.post("/clear-inactive-timeseries")
async def clear_inactive_timeseries():
    """Delete inactive time series data and reclaim database space."""
    logger.debug("Clear inactive time series data called")
    
    # First connection for deletion
    conn = None
    cursor = None
    
    try:
        # Connect to PostgreSQL
        conn = get_db_connection()
        if not conn:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to connect to database"}
            )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # First, count how many records will be affected
        cursor.execute("""
            SELECT COUNT(*) FROM time_series_metadata WHERE is_active = FALSE
        """)
        inactive_metadata_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM time_series_data 
            WHERE chart_id IN (
                SELECT chart_id FROM time_series_metadata WHERE is_active = FALSE
            )
        """)
        inactive_data_count = cursor.fetchone()[0]
        
        if inactive_metadata_count == 0 and inactive_data_count == 0:
            cursor.close()
            conn.close()
            return JSONResponse({
                "status": "success",
                "message": "No inactive time series data found to delete"
            })
        
        # Delete inactive time series data first (due to foreign key constraint)
        cursor.execute("""
            DELETE FROM time_series_data 
            WHERE chart_id IN (
                SELECT chart_id FROM time_series_metadata WHERE is_active = FALSE
            )
        """)
        deleted_data_count = cursor.rowcount
        
        # Delete inactive metadata
        cursor.execute("""
            DELETE FROM time_series_metadata WHERE is_active = FALSE
        """)
        deleted_metadata_count = cursor.rowcount
        
        # Commit the transaction
        conn.commit()
        
        # Close the first connection
        cursor.close()
        conn.close()
        
        # Now reconnect to run VACUUM commands outside of any transaction
        logger.info("Reconnecting to run VACUUM commands...")
        conn = get_db_connection()
        if not conn:
            return JSONResponse({
                "status": "error",
                "message": "Failed to reconnect to database for VACUUM operations"
            })
        
        cursor = conn.cursor()
        
        # Set autocommit to True for VACUUM commands
        conn.autocommit = True
        
        # Run VACUUM FULL on the affected tables
        logger.info("Running VACUUM FULL on time_series_metadata...")
        cursor.execute("VACUUM FULL time_series_metadata")
        
        logger.info("Running VACUUM FULL on time_series_data...")
        cursor.execute("VACUUM FULL time_series_data")
        
        # Update table statistics
        logger.info("Running ANALYZE on time_series_metadata...")
        cursor.execute("ANALYZE time_series_metadata")
        
        logger.info("Running ANALYZE on time_series_data...")
        cursor.execute("ANALYZE time_series_data")
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully deleted {deleted_metadata_count} inactive metadata records and {deleted_data_count} data records, then reclaimed space")
        return JSONResponse({
            "status": "success",
            "message": f"Successfully deleted {deleted_metadata_count} inactive metadata records and {deleted_data_count} data records, then reclaimed database space with VACUUM FULL and ANALYZE"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing inactive time series data: {str(e)}")
        
        # Rollback if we still have an active transaction
        if conn and not conn.autocommit:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
        
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
    
    finally:
        # Ensure connections are closed
        if cursor:
            try:
                cursor.close()
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

@router.post("/clear-all-output")
async def clear_all_output():
    """Delete all files from the output directory."""
    logger.debug("Clear all output files called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # Count of deleted files
        deleted_count = 0
        
        # Delete all files and subdirectories
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            # Delete empty directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    # Only remove if empty
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.debug(f"Deleted directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting directory {dir_path}: {str(e)}")
        
        # Recreate output directory structure
        os.makedirs(os.path.join(output_dir, 'annual'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'monthly'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ytd'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'weekly'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'dashboard'), exist_ok=True)
        
        logger.info(f"Successfully deleted {deleted_count} files from output directory")
        return JSONResponse({
            "status": "success", 
            "message": f"Successfully deleted {deleted_count} files from output directory"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing all output files: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.post("/clear-vector-db")
async def clear_vector_db():
    """Clear all collections from the vector database except SFPublicData."""
    logger.debug("Clear vector database called")
    try:
        # Connect to Qdrant
        qdrant = QdrantClient(host='localhost', port=6333)
        
        # Get all collections
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Count of deleted collections
        deleted_count = 0
        
        # Delete all collections except SFPublicData
        for name in collection_names:
            if name != "SFPublicData":
                try:
                    qdrant.delete_collection(name)
                    deleted_count += 1
                    logger.debug(f"Deleted collection: {name}")
                except Exception as e:
                    logger.error(f"Error deleting collection {name}: {str(e)}")
        
        logger.info(f"Successfully deleted {deleted_count} collections from vector database")
        return JSONResponse({
            "status": "success", 
            "message": f"Successfully deleted {deleted_count} collections from vector database"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing vector database: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.get("/get-notes")
async def get_notes_file():
    """
    Serves the combined notes using the same notes_manager function as the explainer agent.
    """
    logger.info("Backend get-notes route called")
    
    try:
        # Import the notes manager function
        from tools.notes_manager import get_notes
        
        # Create empty context_variables for the function
        context_variables = {}
        
        # Call the same function the explainer agent uses
        result = get_notes(context_variables)
        
        if "error" in result:
            logger.error(f"Notes manager error: {result['error']}")
            return JSONResponse({
                "success": False, 
                "error": result["error"]
            }, status_code=500)
        
        content = result.get("notes", "")
        
        # Simple token count approximation
        token_count = len(content.split())
        
        return JSONResponse({
            "success": True,
            "content": content,
            "token_count": token_count
        })
        
    except Exception as e:
        logger.exception(f"Error getting notes: {str(e)}")
        return JSONResponse({
            "success": False, 
            "error": f"Error getting notes: {str(e)}"
        }, status_code=500)

@router.get("/run-evals")
async def run_evals_endpoint(query: str, model_key: str = None):
    """Run a single eval with the specified query and return results."""
    logger.info(f"Running single eval with query: '{query}' and model: {model_key or 'default'}")
    
    try:
        # Run the query through the new single-eval function
        result = run_single_eval_langchain(query, model_key=model_key)
        
        # Check if the log file exists
        log_filename = result.get("log_filename")
        if not log_filename or not os.path.exists(os.path.join('ai/logs/evals', log_filename)):
            # Also check the full path
            if not log_filename or not os.path.exists(log_filename):
                return JSONResponse({
                    "status": "error",
                    "message": "Log file not found after running eval"
                }, status_code=500)

        # Ensure we have the full path to the log file for reading
        full_log_path = log_filename if os.path.exists(log_filename) else os.path.join('ai/logs/evals', log_filename)
        
        # Read the log file content
        with open(full_log_path, 'r') as log_file:
            log_content = log_file.read()
        
        return JSONResponse({
            "status": "success",
            "message": f"Eval completed successfully for query: {query}",
            "tool_calls_count": result.get("tool_calls_count", 0),
            "log_filename": os.path.basename(log_filename),
            "log_content": log_content
        })
    except Exception as e:
        logger.exception(f"Error running single eval: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/run-all-evals")
async def run_all_evals_endpoint(model_key: str = None):
    """Run all evals with the specified query and return results."""
    logger.info(f"Running all evals with model: {model_key or 'default'}")
    
    try:
        # We need a new function in evals.py to handle all evals at once
        from evals import run_all_evals, run_model_comparison
        
        # Run all evals and get results
        results = run_all_evals(model_key=model_key)
        
        return JSONResponse(content=results)
    except Exception as e:
        logger.exception(f"Error running all evals: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "data": {
                    "time_series_charts": [],
                    "anomaly_charts": [],
                    "map_charts": [],
                    "summary": {"total_charts": 0, "error": str(e)}
                }
            }
        )

@router.get("/evals-interface")
async def evals_interface(request: Request):
    """Serve the evals interface."""
    logger.debug("Evals interface route called")
    if templates is None:
        logger.error("Templates not initialized in backend router")
        raise RuntimeError("Templates not initialized")
    
    return templates.TemplateResponse("evals.html", {
        "request": request
    })

@router.get("/dashboard")
async def dashboard_page(request: Request):
    """Serve the dashboard page."""
    logger.debug("Dashboard page route called")
    if templates is None:
        logger.error("Templates not initialized in backend router")
        raise RuntimeError("Templates not initialized")
    
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/api/datasets-count")
async def get_datasets_count():
    """Get the count of datasets from the database."""
    try:
        # Try to get count from database first
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Query the datasets table
                cursor.execute("SELECT COUNT(*) FROM datasets WHERE is_active = true")
                
                result = cursor.fetchone()
                count = int(result[0]) if result else 0
                
                cursor.close()
                connection.close()
                
                # Mock change value for now
                # In a real implementation, this would be calculated from historical data
                change = 0.0
                
                logger.info(f"Dataset count from database: {count}")
                
                return JSONResponse(content={
                    "count": count,
                    "change": float(change)
                })
                
            except Exception as db_error:
                logger.error(f"Database error in get_datasets_count: {db_error}")
                if connection:
                    connection.close()
                # Fall through to file-based approach
        
        # Database approach failed, return default values
        logger.error("Database approach failed, no fallback available")
        return JSONResponse(content={
            "count": 0,
            "change": 0.0
        })
    except Exception as e:
        logger.error(f"Error getting datasets count: {str(e)}")
        # Return default value in case of error
        return JSONResponse(content={
            "count": 0,
            "change": 0.0
        })

@router.get("/api/list-datasets-for-typeahead")
async def list_datasets_for_typeahead():
    """Get a list of datasets for the typeahead functionality from database."""
    try:
        # Try to get datasets from database first
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if connection:
            try:
                cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Query the datasets table
                cursor.execute("""
                    SELECT endpoint, title, description, category
                    FROM datasets 
                    WHERE is_active = true
                    ORDER BY title
                """)
                
                datasets = []
                for row in cursor.fetchall():
                    dataset_info = {
                        'endpoint': row['endpoint'],
                        'title': row['title'] or 'Untitled',
                        'description': row['description'] or '',
                        'category': row['category'] or 'Uncategorized'
                    }
                    datasets.append(dataset_info)
                
                cursor.close()
                connection.close()
                
                logger.info(f"Retrieved {len(datasets)} datasets from database")
                return JSONResponse(content=datasets)
                
            except Exception as db_error:
                logger.error(f"Database error in list_datasets_for_typeahead: {db_error}")
                if connection:
                    connection.close()
                # Fall through to file-based approach
        
        # Database approach failed, return empty list
        logger.error("Database approach failed, no fallback available")
        return JSONResponse(content=[])
    except Exception as e:
        logger.error(f"Error listing datasets for typeahead: {str(e)}")
        return JSONResponse(content=[], status_code=500)

@router.get("/api/time-series-count")
async def get_time_series_count():
    """Get the count of rows in the time_series_metadata table."""
    try:
        # Connect to PostgreSQL
        conn = get_postgres_connection()
            
        cursor = conn.cursor()
            
        # Query to count rows in the time_series_metadata table
        cursor.execute("SELECT COUNT(*) FROM time_series_metadata")
            
        result = cursor.fetchone()
        count = int(result[0]) if result else 0
            
        cursor.close()
        conn.close()
            
        # Mock change value for now
        # In a real implementation, this would be calculated from historical data
        change = 0
        
        logger.info(f"Time series metadata count: {count}")
        
        return JSONResponse(content={
            "count": count,
            "change": change
        })
    except Exception as e:
        logger.error(f"Error getting time series metadata count: {str(e)}")
        # Return default value in case of error
        return JSONResponse(content={
            "count": 0,
            "change": 0
        })

@router.get("/api/anomalies-count")
async def get_anomalies_count():
    """Get the count of rows in the anomalies table."""
    try:
        # Connect to PostgreSQL
        conn = get_postgres_connection()
            
        cursor = conn.cursor()
            
        # Query to count rows in the anomalies table
        cursor.execute("SELECT COUNT(*) FROM anomalies")
            
        result = cursor.fetchone()
        count = int(result[0]) if result else 0
            
        cursor.close()
        conn.close()
            
        # Mock change value for now
        # In a real implementation, this would be calculated from historical data
        change = 0
        
        logger.info(f"Anomalies count: {count}")
        
        return JSONResponse(content={
            "count": count,
            "change": change
        })
    except Exception as e:
        logger.error(f"Error getting anomalies count: {str(e)}")
        # Return default value in case of error
        return JSONResponse(content={
            "count": 0,
            "change": 0
        })

@router.get("/api/postgres-size")
async def get_postgres_size():
    """Get the size of the PostgreSQL database in MB."""
    try:
        # Connect to PostgreSQL and get the database size
        # This requires the database connection to be set up
        try:
            conn = get_postgres_connection()
            
            cursor = conn.cursor()
            
            # Query to get the database size
            cursor.execute("""
                SELECT pg_database_size(current_database()) / 1024.0 / 1024.0 as size_mb
            """)
            
            result = cursor.fetchone()
            # Convert Decimal to float for JSON serialization
            size_mb = float(result[0]) if result else 0.0
            
            cursor.close()
            conn.close()
            
        except Exception as db_err:
            logger.error(f"Database error: {str(db_err)}")
            # If can't connect to DB, return estimated size
            size_mb = 0.0
        
        # Mock change value for now
        # In a real implementation, this would be calculated from historical data
        change = 0
        
        return JSONResponse(content={
            "size_mb": size_mb,
            "change": change
        })
    except Exception as e:
        logger.error(f"Error getting PostgreSQL size: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting PostgreSQL size")

@router.get("/api/vectordb-size")
async def get_vectordb_size():
    """Get the size of the Vector DB in MB using metrics API and filesystem fallback."""
    try:
        # Method 1: Try to get size from Qdrant metrics API
        try:
            import requests
            logger.info("Attempting to get Qdrant size from metrics API")
            response = requests.get("http://localhost:6333/metrics", timeout=5)
            
            if response.ok:
                metrics = response.text
                # Look for disk usage metrics in the response
                disk_metrics = {}
                
                # Parse metrics for disk usage
                for line in metrics.splitlines():
                    if line.startswith("qdrant_storage_total_") and not line.startswith("#"):
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                key = parts[0].split("{")[0]  # Get metric name without labels
                                value = float(parts[1])
                                disk_metrics[key] = value
                        except Exception as parse_err:
                            logger.warning(f"Error parsing metric line '{line}': {str(parse_err)}")
                
                # Look for total bytes metric
                if "qdrant_storage_total_bytes" in disk_metrics:
                    storage_bytes = disk_metrics["qdrant_storage_total_bytes"]
                    storage_mb = storage_bytes / (1024 * 1024)  # Convert bytes to MB
                    logger.info(f"Got Qdrant size from metrics API: {storage_mb:.2f} MB")
                    return JSONResponse(content={
                        "size_mb": float(storage_mb),
                        "change": 0.0,
                        "source": "metrics_api"
                    })
                else:
                    logger.warning("Storage metrics not found in Qdrant metrics API response")
            else:
                logger.warning(f"Metrics API request failed with status {response.status_code}")
        
        except Exception as metrics_err:
            logger.warning(f"Error getting size from metrics API: {str(metrics_err)}")
        
        # Method 2: Try filesystem measurement
        try:
            # Get Qdrant storage path - default is './storage' in Qdrant's working directory
            # This needs to be adjusted based on your Qdrant configuration
            logger.info("Attempting to measure Qdrant size from filesystem")
            
            # Try to get storage path from environment variable or use default
            qdrant_storage_path = os.environ.get("QDRANT_STORAGE_PATH", "/var/lib/qdrant")
            
            if not os.path.exists(qdrant_storage_path):
                # Try alternate paths
                alternate_paths = [
                    "./storage",  # Default in local development
                    "/var/lib/qdrant/storage",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage")
                ]
                
                for path in alternate_paths:
                    if os.path.exists(path):
                        qdrant_storage_path = path
                        break
            
            if os.path.exists(qdrant_storage_path):
                total_size = 0
                for dirpath, _, filenames in os.walk(qdrant_storage_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            total_size += os.path.getsize(fp)
                        except (FileNotFoundError, PermissionError) as e:
                            logger.warning(f"Error getting size of {fp}: {str(e)}")
                
                storage_mb = total_size / (1024 * 1024)  # Convert bytes to MB
                logger.info(f"Got Qdrant size from filesystem: {storage_mb:.2f} MB")
                return JSONResponse(content={
                    "size_mb": float(storage_mb),
                    "change": 0.0,
                    "source": "filesystem"
                })
            else:
                logger.warning(f"Qdrant storage path not found at {qdrant_storage_path}")
        except Exception as fs_err:
            logger.warning(f"Error measuring size from filesystem: {str(fs_err)}")
        
        # Method 3: Fallback to collection-based estimation if both methods above fail
        total_points = 0
        collections_count = 0
        
        try:
            # Initialize Qdrant client
            logger.info("Falling back to collection-based size estimation")
            qdrant_client = QdrantClient(os.environ.get("QDRANT_URL", "localhost"), port=6333)
            
            # Get list of collections
            collections_response = qdrant_client.get_collections()
            collections_count = len(collections_response.collections)
            
            # Iterate through each collection to get statistics
            for collection_info in collections_response.collections:
                collection_name = collection_info.name
                
                try:
                    # Get collection info with points count
                    collection_detail = qdrant_client.get_collection(collection_name=collection_name)
                    points_count = collection_detail.points_count
                    
                    if points_count is not None:
                        total_points += points_count
                except Exception as coll_err:
                    logger.error(f"Error getting details for collection {collection_name}: {str(coll_err)}")
                    continue
                    
        except Exception as qdrant_err:
            logger.error(f"Error querying Qdrant: {str(qdrant_err)}")
        
        # Estimate size based on points (more realistic estimate: ~2KB per vector on average)
        estimated_size_mb = (total_points * 2.0) / 1024
        
        # If it's a very small number, set a minimum
        if estimated_size_mb < 0.1 and total_points > 0:
            estimated_size_mb = 0.1
            
        logger.info(f"Estimated size from collections: {estimated_size_mb:.2f} MB")
        return JSONResponse(content={
            "size_mb": float(estimated_size_mb),
            "change": 0.0,
            "source": "estimate"
        })
    except Exception as e:
        logger.error(f"Error getting Vector DB size: {str(e)}")
        # Return a default value in case of error
        return JSONResponse(content={
            "size_mb": 0.0,
            "change": 0.0,
            "source": "error"
        })

@router.get("/api/system-status")
async def get_system_status():
    """Get system status for various components."""
    try:
        status_items = []
        
        # Check PostgreSQL status
        postgres_status = "error"
        postgres_value = "Offline"
        try:
            conn = get_postgres_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            postgres_status = "healthy"
            postgres_value = "Online"
        except Exception as db_err:
            logger.error(f"PostgreSQL status check failed: {str(db_err)}")
        
        status_items.append({
            "name": "PostgreSQL Database",
            "status": postgres_status,
            "value": postgres_value
        })
        
        # Check Vector DB (Qdrant) status
        vectordb_status = "error"
        vectordb_value = "Offline"
        try:
            qdrant_client = QdrantClient(os.environ.get("QDRANT_URL", "localhost"), port=6333)
            collections = qdrant_client.get_collections()
            vectordb_status = "healthy"
            vectordb_value = f"{len(collections.collections)} collections"
        except Exception as vdb_err:
            logger.error(f"Vector DB status check failed: {str(vdb_err)}")
        
        status_items.append({
            "name": "Vector Database",
            "status": vectordb_status,
            "value": vectordb_value
        })
        
        # Check disk space
        disk_status = "healthy"
        disk_value = ""
        try:
            total, used, free = shutil.disk_usage("/")
            percent_used = (used / total) * 100
            
            if percent_used > 90:
                disk_status = "error"
            elif percent_used > 70:
                disk_status = "warning"
            
            disk_value = f"{round(percent_used, 1)}% used"
        except Exception as disk_err:
            logger.error(f"Disk status check failed: {str(disk_err)}")
            disk_status = "error"
            disk_value = "Unknown"
        
        status_items.append({
            "name": "Disk Space",
            "status": disk_status,
            "value": disk_value
        })
        
        # Add server uptime
        uptime_status = "healthy"
        uptime_value = ""
        try:
            if os.name == 'posix':  # Linux or MacOS
                try:
                    # Try using uptime command (works on both Linux and MacOS)
                    uptime_output = subprocess.check_output(['uptime']).decode('utf-8')
                    # Extract the uptime information
                    if 'day' in uptime_output:
                        # Format with days
                        days_part = uptime_output.split('up ')[1].split(' day')[0].strip()
                        days = int(days_part)
                        uptime_value = f"{days} days"
                    else:
                        # Less than a day
                        uptime_value = "Less than a day"
                except:
                    # Fallback for Linux only
                    if os.path.exists('/proc/uptime'):
                        with open('/proc/uptime', 'r') as f:
                            uptime_seconds = float(f.readline().split()[0])
                            uptime_days = uptime_seconds / 86400  # Convert to days
                            uptime_value = f"{round(uptime_days, 1)} days"
                    else:
                        uptime_value = "Unknown"
            else:
                # Default value for non-posix systems
                uptime_value = "Unknown"
        except Exception as uptime_err:
            logger.error(f"Uptime check failed: {str(uptime_err)}")
            uptime_status = "warning"
            uptime_value = "Unknown"
        
        status_items.append({
            "name": "Server Uptime",
            "status": uptime_status,
            "value": uptime_value
        })
        
        return JSONResponse(content=status_items)
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting system status")

@router.get("/api/time-series-data-count")
async def get_time_series_data_count():
    """
    Returns the count of time series data points in the system.
    Used by the dashboard to display time series data metrics.
    """
    try:
        # Connect to the database
        conn = None
        try:
            conn = get_postgres_connection()
            
            cursor = conn.cursor()
            
            # Query to count time series data points
            query = """
                SELECT COUNT(*) 
                FROM time_series_data;
            """
            
            cursor.execute(query)
            result = cursor.fetchone()
            count = result[0] if result else 0
            
            # Get change from yesterday (last 24 hours)
            change_query = """
                SELECT COUNT(*) 
                FROM time_series_data 
                WHERE created_at >= NOW() - INTERVAL '24 hours';
            """
            
            cursor.execute(change_query)
            change_result = cursor.fetchone()
            change = change_result[0] if change_result else 0
            
            return {
                "count": count,
                "change": change,
                "change_period": "24 hours"
            }
            
        except Exception as e:
            logger.error(f"Database error when counting time series data: {str(e)}")
            return {"count": "DB Error", "change": 0}
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.exception(f"Error getting time series data count: {str(e)}")
        return {"count": "Error", "change": 0}







@router.get("/monthly-reports")
async def monthly_reports_page():
    """Serve the monthly reports interface page."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "templates", "monthly_reports.html")
    return FileResponse(template_path)



@router.post("/update_published_url/{report_id}")
async def update_published_url(report_id: int, request: Request):
    """Update the published URL for a specific monthly report."""
    try:
        # Get request data
        data = await request.json()
        published_url = data.get("published_url")
        
        # Allow None/null to clear the published URL
        if published_url == "":
            published_url = None
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor()
        
        # Check if report exists
        cursor.execute("SELECT id FROM reports WHERE id = %s", (report_id,))
        report = cursor.fetchone()
        
        if not report:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Update the published URL
        cursor.execute("""
            UPDATE reports 
            SET published_url = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (published_url, report_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Updated published URL for report {report_id}: {published_url}")
        
        return JSONResponse({
            "status": "success",
            "message": "Published URL updated successfully",
            "published_url": published_url
        })
        
    except Exception as e:
        logger.error(f"Error updating published URL for report {report_id}: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error updating published URL: {str(e)}"
        }, status_code=500)



@router.get("/api/total-metrics-count")
async def get_total_metrics_count():
    """Get the total count of metrics from the database."""
    try:
        # Try to get count from database first
        from tools.db_utils import get_postgres_connection
        
        connection = get_postgres_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Query the metrics table
                cursor.execute("SELECT COUNT(*) FROM metrics")
                
                result = cursor.fetchone()
                count = int(result[0]) if result else 0
                
                cursor.close()
                connection.close()
                
                logger.info(f"Metrics count from database: {count}")
                
                return JSONResponse({
                    "count": count,
                    "change": 0
                })
                
            except Exception as db_error:
                logger.error(f"Database error in get_total_metrics_count: {db_error}")
                if connection:
                    connection.close()
                # Fall through to file-based approach
        
        # Fallback to file-based approach
        logger.warning("Database approach failed, falling back to file-based approach")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        queries_file = os.path.join(script_dir, "data", "dashboard", "dashboard_queries_enhanced.json")
        
        if not os.path.exists(queries_file):
            logger.error(f"dashboard_queries_enhanced.json not found at {queries_file}")
            return JSONResponse({
                "count": "Error",
                "change": 0
            })
        
        with open(queries_file, 'r') as f:
            queries_data = json.load(f)
        
        # Count total metrics across all categories and subcategories
        total_metrics = 0
        for category in queries_data.values():
            for subcategory in category.values():
                if "queries" in subcategory:
                    total_metrics += len(subcategory["queries"])
        
        return JSONResponse({
            "count": total_metrics,
            "change": 0
        })
        
    except Exception as e:
        logger.exception(f"Error getting total metrics count: {str(e)}")
        return JSONResponse({
            "count": "Error",
            "change": 0
        }, status_code=500)

@router.get("/api/anomalies-count-by-status")
async def get_anomalies_count_by_status():
    """Get the count of anomalies grouped by out_of_bounds status."""
    try:
        # Connect to PostgreSQL
        conn = get_postgres_connection()
            
        cursor = conn.cursor()
            
        # Query to count rows in the anomalies table grouped by out_of_bounds
        cursor.execute("""
            SELECT out_of_bounds, COUNT(*) as count 
            FROM anomalies 
            GROUP BY out_of_bounds
        """)
            
        results = cursor.fetchall()
        
        # Initialize counts
        out_of_bounds_count = 0
        in_bounds_count = 0
        
        # Process results
        for row in results:
            if row[0]:  # out_of_bounds is True
                out_of_bounds_count = int(row[1])
            else:  # out_of_bounds is False
                in_bounds_count = int(row[1])
            
        cursor.close()
        conn.close()
            
        logger.info(f"Anomalies count by status - Out of bounds: {out_of_bounds_count}, In bounds: {in_bounds_count}")
        
        return JSONResponse(content={
            "out_of_bounds": out_of_bounds_count,
            "in_bounds": in_bounds_count,
            "total": out_of_bounds_count + in_bounds_count
        })
    except Exception as e:
        logger.error(f"Error getting anomalies count by status: {str(e)}")
        # Return default values in case of error
        return JSONResponse(content={
            "out_of_bounds": 0,
            "in_bounds": 0,
            "total": 0
        })

@router.get("/monthly-report/{report_id}")
async def get_monthly_report_by_id(report_id: int):
    """
    Get a monthly report by its ID from the database.
    """
    try:
        logger.info(f"Requesting monthly report by ID: {report_id}")
        
        # Connect to database
        conn = get_postgres_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get the report details
        cur.execute("""
            SELECT id, district, period_type, max_items, created_at, updated_at, 
                   original_filename, revised_filename, published_url, audio_file, headlines
            FROM reports
            WHERE id = %s
        """, (report_id,))
        
        report = cur.fetchone()
        
        if not report:
            logger.error(f"Monthly report not found with ID: {report_id}")
            return JSONResponse(
                status_code=404,
                content={"detail": f"Monthly report not found with ID: {report_id}"}
            )
        
        # Get the metrics for this report
        cur.execute("""
            SELECT *
            FROM monthly_reporting
            WHERE report_id = %s
            ORDER BY priority
        """, (report_id,))
        
        metrics = cur.fetchall()
        
        # Format the report data
        report_data = {
            "id": report["id"],
            "district": report["district"],
            "district_name": f"District {report['district']}" if report["district"] != "0" else "Citywide",
            "period_type": report["period_type"],
            "max_items": report["max_items"],
            "created_at": report["created_at"].isoformat() if report["created_at"] else None,
            "updated_at": report["updated_at"].isoformat() if report["updated_at"] else None,
            "original_filename": report["original_filename"],
            "revised_filename": report["revised_filename"],
            "published_url": report["published_url"],
            "audio_file": report["audio_file"],
            "headlines": report["headlines"],
            "metrics": []
        }
        
        # Format the metrics data
        for metric in metrics:
            report_data["metrics"].append({
                "id": metric["id"],
                "report_id": metric["report_id"],
                "item_title": metric["item_title"],
                "metric_name": metric["metric_name"],
                "group_value": metric["group_value"],
                "group_field_name": metric["group_field_name"],
                "period_type": metric["period_type"],
                "comparison_mean": metric["comparison_mean"],
                "recent_mean": metric["recent_mean"],
                "difference": metric["difference"],
                "std_dev": metric["std_dev"],
                "percent_change": metric["percent_change"],
                "rationale": metric["rationale"],
                "explanation": metric["explanation"],
                "priority": metric["priority"],
                "report_text": metric["report_text"],
                "district": metric["district"],
                "chart_data": metric["chart_data"],
                "metadata": metric["metadata"],
                "created_at": metric["created_at"].isoformat() if metric["created_at"] else None
            })
        
        cur.close()
        conn.close()
        
        return JSONResponse(content={
            "status": "success",
            "report": report_data
        })
    except Exception as e:
        logger.error(f"Error getting monthly report by ID: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting monthly report by ID: {str(e)}"}
        )

@router.get("/time-series-chart")
async def time_series_chart_page(request: Request):
    """
    Serve the time series chart page.
    
    Query Parameters:
        metric_id: ID of the metric to display
        district: District ID (default: 0 for citywide)
        period_type: Period type (year, month, etc.)
        groups: Comma-separated list of group values to include (optional)
        format: Set to 'image' to return a static image instead of HTML
    """
    # Check if format=image is in the query parameters
    format_param = request.query_params.get('format')
    
    if format_param == 'image':
        try:
            # Extract parameters for the chart
            metric_id = request.query_params.get('metric_id')
            district = request.query_params.get('district', '0')
            period_type = request.query_params.get('period_type', 'year')
            
            if not metric_id:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "metric_id is required for image export"}
                )
            
            # Get chart data
            from routes.charts import get_chart_by_metric_legacy
            chart_data_response = await get_chart_by_metric_legacy(
                metric_id=metric_id,
                district=int(district),
                period_type=period_type
            )
            
            # Check if the response was successful and parse the JSON body
            if chart_data_response.status_code == 200:
                try:
                    # The body is bytes, decode it and parse JSON
                    chart_data = json.loads(chart_data_response.body.decode('utf-8'))
                    logger.info(f"Successfully parsed chart data from JSONResponse for metric {metric_id}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse JSON from get_chart_by_metric response: {json_err}")
                    return JSONResponse(
                        status_code=500,
                        content={"detail": "Failed to parse chart data response"}
                    )
            else:
                # Handle non-200 response from get_chart_by_metric
                logger.error(f"get_chart_by_metric returned status {chart_data_response.status_code}")
                # Try to return the original error detail if possible
                error_detail = "Failed to fetch chart data"
                try:
                    error_body = json.loads(chart_data_response.body.decode('utf-8'))
                    error_detail = error_body.get("detail", error_detail)
                except: # Ignore errors parsing the error body
                    pass
                    
                return JSONResponse(
                    status_code=chart_data_response.status_code, # Forward the status code
                    content={"detail": error_detail}
                )
                
            # Now, use the parsed chart_data dictionary
            if not chart_data or not chart_data.get('data'):
                logger.warning(f"No chart data found in parsed response for metric {metric_id}")
                return JSONResponse(
                    status_code=404,
                    content={"detail": "No chart data found for the specified parameters"}
                )
            
            # Use Plotly to create the chart image
            import plotly.graph_objects as go
            import plotly.io as pio
            from io import BytesIO
            import base64
            
            # Create the figure
            fig = go.Figure()
            
            # Get metadata
            metadata = chart_data.get('metadata', {})
            object_name = metadata.get('object_name', 'Time Series')
            
            # Group data by group_value if it exists
            data_by_group = {}
            for item in chart_data.get('data', []):
                group = item.get('group_value', 'Default')
                if group not in data_by_group:
                    data_by_group[group] = []
                data_by_group[group].append(item)
            
            # Define colors for groups
            colors = [
                '#ad35fa', '#FF6B5A', '#4A7463', '#71B2CA', '#FFC107', 
                '#9C27B0', '#2196F3', '#E91E63', '#4CAF50', '#FF5722'
            ]
            
            # Add traces for each group
            for i, (group, data) in enumerate(data_by_group.items()):
                color = colors[i % len(colors)]
                
                # Sort data by time_period
                data.sort(key=lambda x: x.get('time_period', ''))
                
                # Extract x and y values
                x_values = [item.get('time_period', '') for item in data]
                y_values = [item.get('numeric_value', 0) for item in data]
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=group,
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=8)
                ))
            
            # Calculate overall mean for reference line
            all_values = [item.get('numeric_value', 0) for item in chart_data.get('data', [])]
            if all_values:
                mean_value = sum(all_values) / len(all_values)
                
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=[x_values[0], x_values[-1]] if len(x_values) > 0 else [],
                    y=[mean_value, mean_value],
                    mode='lines',
                    name=f'Average: {mean_value:.2f}',
                    line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash')
                ))
            
            # Update layout
            district_name = "Citywide" if district == "0" else f"District {district}"
            fig.update_layout(
                title=f"{object_name} - {district_name}",
                xaxis_title="Date",
                yaxis_title=metadata.get('y_axis_label', 'Value'),
                legend=dict(orientation="h", y=-0.2),
                autosize=False,
                width=1000,
                height=600,
                margin=dict(l=80, r=80, t=100, b=100),
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(255,255,255,0.9)'
            )
            
            # Convert to PNG image
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format='png')
            img_bytes.seek(0)
            
            # Return the image
            return StreamingResponse(
                img_bytes, 
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename=time_series_{metric_id}_{district}_{period_type}.png"
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating time series chart image: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error generating chart image: {str(e)}"}
            )
    
    # Default behavior - return HTML page
    return templates.TemplateResponse("time_series_chart.html", {"request": request})

@router.get("/monthly-report/file/{filename}")
@router.head("/monthly-report/file/{filename}")
async def get_monthly_report_file(filename: str, request: Request):
    """
    Serve a monthly report file directly by filename from the output/reports directory.
    """
    try:
        logger.info(f"Requesting monthly report file: {filename} (method: {request.method})")
        
        # Construct the path to the reports directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"get_monthly_report_file - script_dir: {script_dir}") # ADDED LOGGING
        reports_dir = os.path.join(script_dir, "output", "reports")
        logger.info(f"get_monthly_report_file - reports_dir: {reports_dir}") # ADDED LOGGING
        file_path = os.path.join(reports_dir, filename)
        logger.info(f"get_monthly_report_file - constructed file_path: {file_path}") # ADDED LOGGING
        
        # Security check to prevent accessing files outside the reports directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(reports_dir)):
            logger.error(f"Security check failed: Attempt to access file outside reports directory: {filename}")
            raise HTTPException(status_code=403, detail="Access denied")
            
        # Check if the file exists
        if os.path.exists(file_path) and os.path.isfile(file_path):
            logger.info(f"Serving file: {file_path}")
            # Determine media type based on extension
            media_type = "text/html" if filename.lower().endswith(".html") else "application/octet-stream"
            
            # For HEAD requests, return just headers without content
            if request.method == "HEAD":
                from fastapi import Response
                response = Response()
                response.headers["Content-Type"] = media_type
                response.headers["Content-Length"] = str(os.path.getsize(file_path))
                return response
            else:
                return FileResponse(file_path, media_type=media_type)
        else:
            logger.error(f"Monthly report file not found at: {file_path}")
            logger.error(f"File exists check: {os.path.exists(file_path)}") # ADDED LOGGING
            logger.error(f"Is file check: {os.path.isfile(file_path)}") # ADDED LOGGING
            raise HTTPException(status_code=404, detail=f"Monthly report file not found: {filename}")
            
    except HTTPException as http_exc:
        # Re-raise HTTPException to ensure correct status code is sent
        raise http_exc
    except Exception as e:
        error_message = f"Error getting monthly report file: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/narration/{filename}")
@router.head("/narration/{filename}")
async def get_narration_file(filename: str, request: Request):
    """
    Serve a narration audio file directly by filename from the output/narration directory.
    Supports both GET (to serve the file) and HEAD (to check if file exists) methods.
    """
    try:
        logger.info(f"Requesting narration file: {filename} (method: {request.method})")
        
        # Construct the path to the narration directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        narration_dir = os.path.join(script_dir, "output", "narration")
        file_path = os.path.join(narration_dir, filename)
        
        # Security check to prevent accessing files outside the narration directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(narration_dir)):
            logger.error(f"Security check failed: Attempt to access file outside narration directory: {filename}")
            raise HTTPException(status_code=403, detail="Access denied")
            
        # Check if the file exists
        if os.path.exists(file_path) and os.path.isfile(file_path):
            logger.info(f"Narration file found: {file_path}")
            
            # For HEAD requests, just return the headers without content
            if request.method == "HEAD":
                media_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "application/octet-stream"
                return Response(
                    content=None,
                    media_type=media_type,
                    headers={
                        "content-length": str(os.path.getsize(file_path)),
                        "accept-ranges": "bytes"
                    }
                )
            
            # For GET requests, serve the file
            logger.info(f"Serving narration file: {file_path}")
            media_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "application/octet-stream"
            return FileResponse(file_path, media_type=media_type)
        else:
            logger.error(f"Narration file not found at: {file_path}")
            raise HTTPException(status_code=404, detail=f"Narration file not found: {filename}")
            
    except HTTPException as http_exc:
        # Re-raise HTTPException to ensure correct status code is sent
        raise http_exc
    except Exception as e:
        error_message = f"Error getting narration file: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/rerun_monthly_report_generation")
async def rerun_monthly_report_generation(request: Request):
    """
    Re-run the monthly report generation process
    
    This endpoint allows users to manually trigger the monthly report generation process
    """
    try:
        from monthly_report import run_monthly_report_process, generate_monthly_report, run_monthly_report_process_with_filename
        import asyncio
        
        # Get request data
        data = await request.json()
        district = data.get("district", "0")
        period_type = data.get("period_type", "month")
        max_report_items = data.get("max_report_items", 10)
        only_generate = data.get("only_generate", False)
        filename = data.get("filename")  # Get the original filename for regeneration
        model_key = data.get("model_key")  # Get the selected model key
        
        logger.info(f"Re-running monthly report generation for district {district}, period_type {period_type}, only_generate={only_generate}, filename={filename}, model_key={model_key}")
        
        # Run the monthly report process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        
        if only_generate:
            # Only run the generate_monthly_report function with original filename if provided
            result = await loop.run_in_executor(
                None,
                lambda: generate_monthly_report(district=district, original_filename=filename, model_key=model_key)
            )
        else:
            # Run the full monthly report process - but use the existing filename if provided
            # This ensures we don't create duplicate files with different timestamps
            if filename:
                # If we have an existing filename, use it to maintain consistency
                result = await loop.run_in_executor(
                    None,
                    lambda: run_monthly_report_process_with_filename(
                        district=district,
                        period_type=period_type,
                        max_report_items=max_report_items,
                        model_key=model_key,
                        original_filename=filename
                    )
                )
            else:
                # Run the full monthly report process with new filename generation
                result = await loop.run_in_executor(
                    None,
                    lambda: run_monthly_report_process(
                        district=district,
                        period_type=period_type,
                        max_report_items=max_report_items,
                        model_key=model_key
                    )
                )
        
        if result.get("status") == "success":
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Monthly report generation completed successfully",
                    "report_path": result.get("revised_report_path") or result.get("newsletter_path")
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": result.get("message", "Unknown error occurred during monthly report generation")
                }
            )
            
    except Exception as e:
        logger.error(f"Error re-running monthly report generation: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error re-running monthly report generation: {str(e)}"
            }
        )

@router.post("/rerun_monthly_report_proofreading")
async def rerun_monthly_report_proofreading(request: Request):
    """
    Re-run the proofreading process for a monthly report
    
    This endpoint allows users to manually trigger the proofreading process for a specific report
    """
    try:
        from monthly_report import proofread_and_revise_report
        import asyncio
        
        # Get request data
        data = await request.json()
        report_path = data.get("report_path")
        model_key = data.get("model_key")
        
        if not report_path:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing report_path parameter"
                }
            )
        
        logger.info(f"Re-running proofreading for report at {report_path} with model_key={model_key}")
        
        # Get the report_id from the filename for more reliable database updates
        report_id = None
        try:
            from pathlib import Path
            import os
            
            # Extract the original filename from the report path
            report_filename = Path(report_path).name
            
            # Handle different filename patterns to get the original filename
            original_filename = report_filename
            if '_for_proofreading' in report_filename:
                original_filename = report_filename.replace('_for_proofreading', '')
            elif '_with_placeholders' in report_filename:
                original_filename = report_filename.replace('_with_placeholders', '')
            elif '_revised' in report_filename:
                original_filename = report_filename.replace('_revised', '')
            elif '_email' in report_filename:
                original_filename = report_filename.replace('_email', '')
            elif '_final' in report_filename:
                original_filename = report_filename.replace('_final', '')
            
            # Look up the report_id
            def get_report_id_operation(connection):
                cursor = connection.cursor()
                cursor.execute("""
                    SELECT id FROM reports WHERE original_filename = %s ORDER BY created_at DESC LIMIT 1
                """, (original_filename,))
                result = cursor.fetchone()
                return result[0] if result else None
            
            # Get database connection parameters
            db_host = os.getenv("POSTGRES_HOST", 'localhost')
            db_port = os.getenv("POSTGRES_PORT", '5432')
            db_user = os.getenv("POSTGRES_USER", 'postgres')
            db_password = os.getenv("POSTGRES_PASSWORD", 'postgres')
            db_name = os.getenv("POSTGRES_DB", 'transparentsf')
            
            report_id_result = execute_with_connection(
                operation=get_report_id_operation,
                db_host=db_host,
                db_port=db_port,
                db_name=db_name,
                db_user=db_user,
                db_password=db_password
            )
            
            if report_id_result.get("status") == "success" and report_id_result.get("result"):
                report_id = report_id_result["result"]
                logger.info(f"Found report_id: {report_id} for filename: {original_filename}")
            else:
                logger.warning(f"Could not find report_id for filename: {original_filename}")
                
        except Exception as e:
            logger.warning(f"Error looking up report_id: {e}. Will use filename-based lookup as fallback.")
        
        # Run the proofreading process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: proofread_and_revise_report(report_path, model_key=model_key, report_id=report_id)
        )
        
        if result.get("status") == "success":
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Proofreading completed successfully",
                    "revised_report_path": result.get("revised_report_path")
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": result.get("message", "Unknown error occurred during proofreading")
                }
            )
            
    except Exception as e:
        logger.error(f"Error re-running proofreading: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error re-running proofreading: {str(e)}"
            }
        )

@router.post("/reprioritize_deltas")
async def reprioritize_deltas_endpoint(request: Request):
    """
    Re-prioritize deltas for an existing newsletter by clearing the monthly_reporting data
    and regenerating it from scratch.
    """
    try:
        # Get request data
        data = await request.json()
        filename = data.get("filename")
        district = data.get("district", "0")
        period_type = data.get("period_type", "month")
        max_report_items = data.get("max_report_items", 10)
        
        if not filename:
            return JSONResponse(
                content={"status": "error", "message": "Filename is required"},
                status_code=400
            )
        
        # Import here to avoid circular imports
        from ai.monthly_report import reprioritize_deltas_for_report
        
        # Run the re-prioritization in a background task with timeout
        import asyncio
        
        try:
            # Add a timeout to prevent hanging (5 minutes max)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    reprioritize_deltas_for_report,
                    filename=filename,
                    district=district,
                    period_type=period_type,
                    max_report_items=max_report_items
                ),
                timeout=300.0  # 5 minutes
            )
            
            if result.get("status") == "success":
                return JSONResponse(content=result)
            else:
                return JSONResponse(
                    content=result,
                    status_code=500
                )
                
        except asyncio.TimeoutError:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Re-prioritization timed out after 5 minutes. The process may still be running in the background."
                },
                status_code=408
            )
        except Exception as e:
            logger.error(f"Error in reprioritize_deltas_endpoint: {str(e)}", exc_info=True)
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Error re-prioritizing deltas: {str(e)}"
                },
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Error parsing request in reprioritize_deltas_endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            },
            status_code=400
        )

@router.post("/regenerate_explanations")
async def regenerate_explanations_endpoint(request: Request):
    """
    Regenerate explanations for an existing newsletter's monthly_reporting items.
    """
    try:
        # Get request data
        data = await request.json()
        filename = data.get("filename")
        model_key = data.get("model_key")  # Get the selected model key
        
        # Debug logging to see if endpoint is being called
        logger.info(f"🔧 DEBUG: regenerate_explanations_endpoint called with filename='{filename}', model_key='{model_key}'")
        
        if not filename:
            logger.error("🔧 DEBUG: No filename provided to regenerate_explanations_endpoint")
            return JSONResponse(
                content={"status": "error", "message": "Filename is required"},
                status_code=400
            )
        
        # Import here to avoid circular imports
        from ai.monthly_report import regenerate_explanations_for_report
        
        # Run the explanation generation in a background task with timeout
        import asyncio
        
        try:
            logger.info(f"🔧 DEBUG: About to call regenerate_explanations_for_report(filename='{filename}', model_key='{model_key}')")
            
            # Add a timeout to prevent hanging (10 minutes max for explanations)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    regenerate_explanations_for_report,
                    filename=filename,
                    model_key=model_key
                ),
                timeout=600.0  # 10 minutes
            )
            
            logger.info(f"🔧 DEBUG: regenerate_explanations_for_report returned: {result}")
            
            if result.get("status") == "success":
                return JSONResponse(content=result)
            else:
                return JSONResponse(
                    content=result,
                    status_code=500
                )
                
        except asyncio.TimeoutError:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Explanation generation timed out after 10 minutes. The process may still be running in the background."
                },
                status_code=408
            )
        except Exception as e:
            logger.error(f"Error in regenerate_explanations_endpoint: {str(e)}", exc_info=True)
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Error regenerating explanations: {str(e)}"
                },
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Error parsing request in regenerate_explanations_endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            },
            status_code=400
        )

@router.post("/update-chart-groups/{chart_id}")
async def update_chart_groups(chart_id: int, request: Request):
    """
    Update the group values for a specific chart.
    This is useful when the chart data exists but doesn't have group values set.
    It fetches the actual data from the source endpoint and adds group values to the time_series_data rows.
    """
    try:
        data = await request.json()
        group_field = data.get("group_field")
        source_query_modification = data.get("source_query_modification", "")
        
        if not group_field:
            return JSONResponse(
                status_code=400,
                content={"detail": "group_field parameter is required"}
            )
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # First, update the metadata to include the group_field
        cursor.execute("""
            UPDATE time_series_metadata
            SET group_field = %s
            WHERE chart_id = %s
            RETURNING *
        """, (group_field, chart_id))
        
        metadata_result = cursor.fetchone()
        
        if not metadata_result:
            conn.rollback()
            cursor.close()
            conn.close()
            return JSONResponse(
                status_code=404,
                content={"detail": f"Chart with ID {chart_id} not found"}
            )
        
        # Get the source query URL from the metadata
        executed_query_url = metadata_result["executed_query_url"]
        
        if not executed_query_url:
            conn.rollback()
            cursor.close()
            conn.close()
            return JSONResponse(
                status_code=400,
                content={"detail": "No source query URL found in metadata"}
            )
        
        # Modify the query if needed to include the group_field
        if source_query_modification:
            executed_query_url = executed_query_url.replace(
                "GROUP+BY+month_period", 
                f"GROUP+BY+month_period%2C+{group_field}"
            )
        
        # Fetch data from the source URL
        import requests
        import urllib.parse
        
        # Decode the URL for better readability of errors
        decoded_url = urllib.parse.unquote(executed_query_url)
        logger.info(f"Fetching data from: {decoded_url}")
        
        try:
            response = requests.get(executed_query_url)
            response.raise_for_status()
            source_data = response.json()
            
            logger.info(f"Fetched {len(source_data)} rows from source data")
            
            # Create a mapping of time periods to group values and numeric values
            # For example: {'2023-01-31': {'Group A': 10, 'Group B': 20}, '2023-02-28': {...}}
            grouped_data = {}
            
            for row in source_data:
                # Parse the time period from the data
                # Note: This might need adjustment based on the actual data format
                time_period_str = row.get("month_period", "")
                group_value = row.get(group_field, "")
                
                if not time_period_str or not group_value:
                    continue
                
                # Convert the time period string to a date object
                from datetime import datetime
                try:
                    time_period = datetime.strptime(time_period_str, "%Y-%m-%dT%H:%M:%S.%f")
                    # Format to match the format in the database (YYYY-MM-DD)
                    time_period_formatted = time_period.strftime("%Y-%m-%d")
                except ValueError:
                    try:
                        # Try another format if the first one fails
                        time_period = datetime.strptime(time_period_str, "%Y-%m-%d")
                        time_period_formatted = time_period_str
                    except ValueError:
                        logger.warning(f"Could not parse time period: {time_period_str}")
                        continue
                
                # Get the numeric value (might be called "value" or something else)
                numeric_value = float(row.get("value", 0))
                
                # Initialize the time period entry if it doesn't exist
                if time_period_formatted not in grouped_data:
                    grouped_data[time_period_formatted] = {}
                
                # Add the group value and numeric value
                if group_value in grouped_data[time_period_formatted]:
                    # If the group already exists for this time period, add to the value
                    grouped_data[time_period_formatted][group_value] += numeric_value
                else:
                    # Otherwise, initialize it
                    grouped_data[time_period_formatted][group_value] = numeric_value
            
            # Now, update or insert the data in the time_series_data table
            updated_count = 0
            inserted_count = 0
            
            for time_period, groups in grouped_data.items():
                for group_value, numeric_value in groups.items():
                    # Check if a row already exists for this combination
                    cursor.execute("""
                        SELECT id
                        FROM time_series_data
                        WHERE chart_id = %s AND time_period = %s AND group_value = %s
                    """, (chart_id, time_period, group_value))
                    
                    existing_row = cursor.fetchone()
                    
                    if existing_row:
                        # Update existing row
                        cursor.execute("""
                            UPDATE time_series_data
                            SET numeric_value = %s
                            WHERE id = %s
                        """, (numeric_value, existing_row["id"]))
                        updated_count += 1
                    else:
                        # Insert new row
                        cursor.execute("""
                            INSERT INTO time_series_data
                            (chart_id, time_period, group_value, numeric_value)
                            VALUES (%s, %s, %s, %s)
                        """, (chart_id, time_period, group_value, numeric_value))
                        inserted_count += 1
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Updated {updated_count} rows and inserted {inserted_count} rows")
            
            return JSONResponse({
                "status": "success",
                "message": f"Updated chart groups successfully. Updated {updated_count} rows and inserted {inserted_count} rows.",
                "details": {
                    "chart_id": chart_id,
                    "group_field": group_field,
                    "updated_count": updated_count,
                    "inserted_count": inserted_count,
                    "total_time_periods": len(grouped_data),
                    "total_groups": sum(len(groups) for groups in grouped_data.values())
                }
            })
            
        except requests.RequestException as e:
            conn.rollback()
            logger.error(f"Error fetching data from source URL: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error fetching data from source URL: {str(e)}", "url": decoded_url}
            )
            
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Database error: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"Error updating chart groups: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error updating chart groups: {str(e)}"}
        )
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

@router.get("/api/postgres-tables")
async def get_postgres_tables():
    """Get a list of all tables in the PostgreSQL database."""
    try:
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query to get all tables in the public schema
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        
        tables = [row['tablename'] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return JSONResponse(content={
            "tables": tables
        })
    except Exception as e:
        logger.error(f"Error getting PostgreSQL tables: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting PostgreSQL tables")

@router.post("/clear-chart-files")
async def clear_chart_files():
    """Delete all chart image files (chart*.png) from the output directory and its subdirectories."""
    logger.debug("Clear chart files called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # Count of deleted files
        deleted_count = 0
        
        # Walk through all subdirectories
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.startswith('chart') and file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted chart file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting chart file {file_path}: {str(e)}")
        
        logger.info(f"Successfully deleted {deleted_count} chart image files")
        return JSONResponse({
            "status": "success",
            "message": f"Successfully deleted {deleted_count} chart image files"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing chart files: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.post("/clear-logs")
async def clear_logs():
    """Truncate all log files in the logs directory except for webchat.log."""
    logger.debug("Clear logs called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        
        # Count of truncated files
        truncated_count = 0
        
        # Check if directory exists
        if not os.path.exists(logs_dir):
            logger.debug(f"Logs directory does not exist: {logs_dir}")
            return JSONResponse({
                "status": "success",
                "message": "No log files to truncate"
            })
        
        # Get all log files
        log_files = [f for f in os.listdir(logs_dir) 
                     if os.path.isfile(os.path.join(logs_dir, f)) and 
                     (f.endswith('.log') or f.endswith('.txt'))]
        
        # Truncate each log file except webchat.log
        for file in log_files:
            if file.lower() == 'webchat.log':
                logger.debug(f"Skipping webchat.log")
                continue
                
            file_path = os.path.join(logs_dir, file)
            try:
                # Truncate file (open in write mode and close immediately)
                with open(file_path, 'w') as f:
                    pass
                truncated_count += 1
                logger.debug(f"Truncated log file: {file_path}")
            except Exception as e:
                logger.error(f"Error truncating log file {file_path}: {str(e)}")
        
        logger.info(f"Successfully truncated {truncated_count} log files")
        return JSONResponse({
            "status": "success",
            "message": f"Successfully truncated {truncated_count} log files (webchat.log preserved)"
        })
        
    except Exception as e:
        logger.exception(f"Error clearing log files: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@router.get("/get_prompts")
async def get_prompts():
    """Get all prompts from the prompts.json file."""
    try:
        from ai.tools.prompts_loader import load_prompts_with_retry
        prompts = load_prompts_with_retry()
        
        return JSONResponse({
            "status": "success",
            "prompts": prompts
        })
    except Exception as e:
        error_message = f"Error getting prompts: {str(e)}"
        logger.error(error_message)
        return JSONResponse({
            "status": "error",
            "message": error_message
        }, status_code=500)

@router.post("/update_prompt")
async def update_prompt(request: Request):
    """Update a specific prompt in the prompts.json file."""
    try:
        # Get request data
        data = await request.json()
        category = data.get("category")
        key = data.get("key")
        prompt = data.get("prompt")
        
        if not category or not key or not prompt:
            return JSONResponse({
                "status": "error",
                "message": "Missing required parameters: category, key, or prompt"
            }, status_code=400)
        
        # Load prompts using the robust loader
        from ai.tools.prompts_loader import load_prompts_with_retry, save_prompts
        prompts = load_prompts_with_retry()
        
        # Check if the category and key exist
        if category not in prompts:
            prompts[category] = {}
        
        # Preserve the existing system message if available and not included in the update
        if category in prompts and key in prompts[category] and isinstance(prompts[category][key], dict):
            existing_prompt = prompts[category][key]
            
            # If the prompt is an object with a 'prompt' field but no 'system' field,
            # add the system field from the existing prompt if available
            if isinstance(prompt, dict) and 'prompt' in prompt and 'system' not in prompt and 'system' in existing_prompt:
                prompt['system'] = existing_prompt['system']
        
        # Update the prompt
        prompts[category][key] = prompt
        
        # Save the updated prompts using the robust saver
        if save_prompts(prompts):
            return JSONResponse({
                "status": "success",
                "message": f"Prompt {category}.{key} updated successfully"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"Failed to save updated prompt {category}.{key}"
            }, status_code=500)
    except Exception as e:
        error_message = f"Error updating prompt: {str(e)}"
        logger.error(error_message)
        return JSONResponse({
            "status": "error",
            "message": error_message
        }, status_code=500)

@router.get("/get-monthly-report-by-district/{district}")
async def get_monthly_report_by_district(district: str):
    """
    Get the latest revised monthly report by district.
    
    Args:
        district: The district ID (0 for citywide)
        
    Returns:
        The HTML content of the latest revised monthly report
    """
    logger.info(f"Getting latest revised monthly report for district {district}")
    
    conn = None
    cursor = None
    try:
        # Get database connection
        conn = get_db_connection()
        if not conn:
            error_msg = "Failed to connect to database"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Create cursor
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the reports table to find the latest report for the specified district
        cursor.execute("""
            SELECT id, district, revised_filename, published_url, audio_file
            FROM reports
            WHERE district = %s AND revised_filename IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, (district,))
        
        report = cursor.fetchone()
        
        if not report:
            error_msg = f"No revised monthly report found for district {district}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        report_dict = dict(report)
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        # Get the report file
        reports_dir = Path(__file__).parent / 'output' / 'reports'
        report_path = reports_dir / report_dict['revised_filename']
        
        if not report_path.exists():
            error_msg = f"Report file not found: {report_path}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Return the file content
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        logger.info(f"Successfully retrieved latest revised monthly report for district {district}")
        return JSONResponse({
            "status": "success",
            "district": district,
            "report_id": report_dict['id'],
            "filename": report_dict['revised_filename'],
            "published_url": report_dict['published_url'],
            "audio_file": report_dict['audio_file'],
            "content": html_content
        })
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error retrieving monthly report: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Ensure resources are closed
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@router.post("/rerun_email_version")
async def rerun_email_version(request: Request):
    """
    Re-run the email version generation for a monthly report
    
    This endpoint allows users to manually regenerate the email-compatible version of a report
    """
    try:
        from monthly_report import generate_email_compatible_report
        import asyncio
        
        # Get request data
        data = await request.json()
        report_path = data.get("report_path")
        
        if not report_path:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing report_path parameter"
                }
            )
        
        logger.info(f"Re-running email version generation for report at {report_path}")
        
        # Run the email generation process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_email_compatible_report(report_path)
        )
        
        if result:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Email version generated successfully",
                    "email_report_path": result
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to generate email version"
                }
            )
            
    except Exception as e:
        logger.error(f"Error re-running email version generation: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error re-running email version generation: {str(e)}"
            }
        )

@router.post("/rerun_final_version")
async def rerun_final_version(request: Request):
    """
    Re-run the final version generation for a monthly report with embedded charts
    
    This endpoint allows users to manually regenerate the final version with TransparentSF embedded charts
    """
    try:
        from monthly_report import generate_final_version_with_charts
        import asyncio
        
        # Get request data
        data = await request.json()
        report_path = data.get("report_path")
        
        if not report_path:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing report_path parameter"
                }
            )
        
        logger.info(f"Re-running final version generation for report at {report_path}")
        
        # Run the final version generation process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_final_version_with_charts(report_path)
        )
        
        if result:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Final version generated successfully",
                    "final_report_path": result
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to generate final version"
                }
            )
            
    except Exception as e:
        logger.error(f"Error re-running final version generation: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error re-running final version generation: {str(e)}"
            }
        )

@router.post("/expand_charts_local")
async def expand_charts_local(request: Request):
    """
    Expand chart references to local/internal charts for web sharing
    """
    try:
        from tools.chart_expansion import expand_chart_references_local
        
        # Get request data
        data = await request.json()
        report_path = data.get("report_path")
        
        if not report_path:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing report_path parameter"
                }
            )
        
        logger.info(f"Expanding charts to local/internal for report at {report_path}")
        
        # Run the chart expansion process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: expand_chart_references_local(report_path)
        )
        
        if result:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Charts expanded to local/internal successfully",
                    "report_path": result
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to expand charts to local/internal"
                }
            )
            
    except Exception as e:
        logger.error(f"Error expanding charts to local/internal: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error expanding charts to local/internal: {str(e)}"
            }
        )

@router.post("/expand_charts_dw")
async def expand_charts_dw(request: Request):
    """
    Expand chart references to DataWrapper charts for email/newsletter compatibility
    """
    try:
        from tools.chart_expansion import expand_chart_references_dw
        
        # Get request data
        data = await request.json()
        report_path = data.get("report_path")
        
        if not report_path:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing report_path parameter"
                }
            )
        
        logger.info(f"Expanding charts to DataWrapper for report at {report_path}")
        
        # Run the chart expansion process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: expand_chart_references_dw(report_path)
        )
        
        if result:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Charts expanded to DataWrapper successfully",
                    "report_path": result
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to expand charts to DataWrapper"
                }
            )
            
    except Exception as e:
        logger.error(f"Error expanding charts to DataWrapper: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error expanding charts to DataWrapper: {str(e)}"
            }
        )

@router.post("/generate_narrated_report")
async def generate_narrated_report_endpoint(request: Request):
    """
    Generate a narrated audio version of the monthly report using ElevenLabs API.
    
    This endpoint takes the final email version of a report and converts it to speech.
    """
    try:
        from monthly_report import generate_narrated_report
        import asyncio
        
        # Get request data
        data = await request.json()
        filename = data.get("filename")
        model_key = data.get("model_key")
        
        if not filename:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing filename parameter"
                }
            )
        
        logger.info(f"Generating narrated version for report: {filename} with model_key={model_key}")
        
        # Construct the path to the email version of the report
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(script_dir, "output", "reports")
        
        # Generate email version filename
        email_filename = filename.replace('.html', '_revised_email.html')
        email_report_path = os.path.join(reports_dir, email_filename)
        
        # Check if email version exists
        if not os.path.exists(email_report_path):
            # Try to find the revised version instead
            revised_filename = filename.replace('.html', '_revised.html')
            revised_report_path = os.path.join(reports_dir, revised_filename)
            
            if os.path.exists(revised_report_path):
                email_report_path = revised_report_path
                logger.info(f"Using revised version instead: {revised_report_path}")
            else:
                # Fall back to original report
                original_report_path = os.path.join(reports_dir, filename)
                if os.path.exists(original_report_path):
                    email_report_path = original_report_path
                    logger.info(f"Using original report: {original_report_path}")
                else:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "status": "error",
                            "message": f"Report file not found: {filename}"
                        }
                    )
        
        # Run the narration generation process in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_narrated_report(email_report_path, model_key=model_key)
        )
        
        if result:
            # Update the database with the audio file path
            try:
                # Extract the audio filename from the full path
                audio_filename = os.path.basename(result)
                
                # Find the report ID from the filename
                # Extract district and date from the filename to match with the database
                # Expected format: monthly_report_{district}_{date}.html
                # or monthly_report_{district}_{date}_revised.html
                base_filename = filename.replace('_revised.html', '.html').replace('.html', '')
                
                # Get database connection
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # Find the report by matching the original or revised filename
                    cursor.execute("""
                        SELECT id FROM reports 
                        WHERE original_filename = %s OR revised_filename = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (filename, filename))
                    
                    report_record = cursor.fetchone()
                    
                    if report_record:
                        report_id = report_record[0]
                        
                        # Update the report with the audio file path
                        cursor.execute("""
                            UPDATE reports 
                            SET audio_file = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (audio_filename, report_id))
                        
                        conn.commit()
                        logger.info(f"Updated report {report_id} with audio file: {audio_filename}")
                    else:
                        logger.warning(f"Could not find report record for filename: {filename}")
                    
                    cursor.close()
                    conn.close()
                else:
                    logger.error("Failed to get database connection for audio file update")
                    
            except Exception as db_error:
                logger.error(f"Error updating database with audio file: {str(db_error)}", exc_info=True)
                # Don't fail the request if database update fails
            
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Narrated report generated successfully",
                    "audio_path": result,
                    "audio_filename": os.path.basename(result)
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to generate narrated report. Check if ElevenLabs API key is configured."
                }
            )
            
    except Exception as e:
        logger.error(f"Error generating narrated report: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error generating narrated report: {str(e)}"
            }
        )

# Removed - migrated to metrics_manager.py

@router.get("/backend/api/active-charts")
async def get_active_charts(metric_id: str, district: str = "0", period_type: str = "month"):
    """
    Get all active chart IDs for a given metric ID, district, and period type.
    
    Args:
        metric_id: The ID of the metric to get charts for
        district: The district ID (default: "0" for citywide)
        period_type: The period type (month, year, etc.)
        
    Returns:
        JSON with a list of active charts including their IDs and metadata
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info(f"Getting active charts for metric_id={metric_id}, district={district}, period_type={period_type}")
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get all active charts for the specified metric, district, and period type
        query = """
        SELECT 
            chart_id, 
            object_name, 
            group_field, 
            object_id,
            district,
            period_type,
            is_active
        FROM time_series_metadata
        WHERE object_id = %s 
        AND district = %s 
        AND period_type = %s
        AND is_active = TRUE
        ORDER BY 
            CASE WHEN group_field IS NULL THEN 0 ELSE 1 END,
            group_field ASC
        """
        
        cursor.execute(query, [metric_id, district, period_type])
        charts = cursor.fetchall()
        
        # Convert any date objects to ISO format strings for JSON serialization
        for chart in charts:
            for key, value in chart.items():
                if hasattr(value, 'isoformat'):
                    chart[key] = value.isoformat()
        
        logger.info(f"Found {len(charts)} active charts for metric_id={metric_id}")
        
        cursor.close()
        conn.close()
        
        return JSONResponse(
            content={
                "status": "success",
                "metric_id": metric_id,
                "district": district,
                "period_type": period_type,
                "chart_count": len(charts),
                "charts": charts
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting active charts: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"Failed to get active charts: {str(e)}"
            }
        )

@router.post("/expand_chart_placeholders")
async def expand_chart_placeholders(request: Request):
    """
    Process a report file and replace simplified chart references with Datawrapper charts.
    
    Args:
        filename: The filename of the report to process
        
    Returns:
        Success status and message
    """
    # For JSON response
    from fastapi.responses import JSONResponse
    import psycopg2.extras
    
    try:
        data = await request.json()
        filename = data.get("filename")
        
        if not filename:
            return {"status": "error", "message": "No filename provided"}
        
        # Construct the path to the reports directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"expand_chart_placeholders - script_dir: {script_dir}")
        reports_dir = os.path.join(script_dir, "output", "reports")
        logger.info(f"expand_chart_placeholders - reports_dir: {reports_dir}")
        report_path = os.path.join(reports_dir, filename)
        logger.info(f"expand_chart_placeholders - constructed file_path: {report_path}")
        
        # Security check to prevent accessing files outside the reports directory
        if not os.path.abspath(report_path).startswith(os.path.abspath(reports_dir)):
            logger.error(f"Security check failed: Attempt to access file outside reports directory: {filename}")
            return {"status": "error", "message": "Access denied"}
            
        # Check if the file exists
        if not os.path.exists(report_path):
            logger.error(f"Monthly report file not found at: {report_path}")
            return {"status": "error", "message": f"Report file not found: {filename}"}
        
        # Define function to get chart data directly from the database
        def get_chart_data_direct(metric_id, district, period_type):
            """
            Direct database access function to get chart data without making HTTP requests
            """
            try:
                conn = get_db_connection()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Get the metric details
                cursor.execute("""
                    SELECT * FROM metrics WHERE id = %s
                """, (metric_id,))
                
                metric = cursor.fetchone()
                if not metric:
                    cursor.close()
                    conn.close()
                    logger.error(f"Metric {metric_id} not found in database")
                    return None
                
                # Get chart data - similar logic to get_chart_by_metric
                # This gets time series data for the metric
                cursor.execute("""
                    SELECT * FROM time_series_data 
                    WHERE metric_id = %s AND district = %s AND period_type = %s
                    ORDER BY date
                """, (metric_id, district, period_type))
                
                data_points = []
                rows = cursor.fetchall()
                for row in rows:
                    data_points.append({
                        "time_period": row["date"].strftime("%Y-%m-%d"),
                        "numeric_value": float(row["value"])
                    })
                
                # Get metadata
                metadata = {
                    "chart_title": f"{metric['name']} - {district if district != '0' else 'Citywide'}",
                    "y_axis_label": metric["name"],
                    "period_type": period_type,
                    "object_type": "dashboard_metric",
                    "object_id": str(metric_id),
                    "object_name": metric["name"],
                    "district": district
                }
                
                cursor.close()
                conn.close()
                
                return {
                    "data": data_points,
                    "metadata": metadata
                }
                
            except Exception as e:
                logger.error(f"Error getting chart data directly from database: {str(e)}", exc_info=True)
                return None
        
        # Import the original functions
        from monthly_report import expand_chart_references
        from tools.genChartdw import create_datawrapper_chart
        
        # Create a wrapper function that uses direct data access
        def create_datawrapper_chart_direct(metric_id, district="0", period_type="month", **kwargs):
            # Get chart data directly from the database
            direct_data = get_chart_data_direct(metric_id, district, period_type)
            if direct_data:
                # Call the original function with direct data
                return create_datawrapper_chart(
                    metric_id=metric_id,
                    district=district,
                    period_type=period_type,
                    direct_data=direct_data,
                    **kwargs
                )
            else:
                logger.error(f"Failed to get direct data for chart metric_id={metric_id}, district={district}, period_type={period_type}")
                return None
        
        # Monkey patch the create_datawrapper_chart function in the monthly_report module
        import sys
        import monthly_report
        
        # Store the original function for restoration later
        original_create_datawrapper_chart = monthly_report.create_datawrapper_chart
        
        # Patch the function with our direct data version
        monthly_report.create_datawrapper_chart = create_datawrapper_chart_direct
        
        try:
            # Call the expand_chart_references_with_tabs function with the patched version
            from tools.chart_expansion import expand_chart_references_with_tabs
            result = expand_chart_references_with_tabs(report_path)
            
            # Restore the original function
            monthly_report.create_datawrapper_chart = original_create_datawrapper_chart
            
            if result:
                return {"status": "success", "message": f"Chart references expanded with tabs in {filename}"}
            else:
                return {"status": "error", "message": f"Failed to expand chart references with tabs in {filename}"}
        finally:
            # Ensure we restore the original function even if an error occurs
            monthly_report.create_datawrapper_chart = original_create_datawrapper_chart
            
    except Exception as e:
        logger.error(f"Error in expand_chart_placeholders: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Error processing request: {str(e)}"}

@router.get("/api/district-maps")
@router.get("/backend/api/district-maps")  # Add an extra route
async def get_district_maps(metric_id: str = None):
    """
    Retrieve district maps for a specific metric ID.
    
    Args:
        metric_id: The ID of the metric to get maps for
        
    Returns:
        JSON with a list of maps including their IDs and URLs
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info(f"Getting district maps for metric_id={metric_id}")
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get maps for the specified metric
        query = """
        SELECT *
        FROM maps
        WHERE type = 'supervisor_district' AND active = TRUE
        """
        
        # Add metric_id filter if provided
        params = []
        if metric_id:
            # First try to use the direct metric_id column
            query = """
            SELECT *
            FROM maps
            WHERE type = 'supervisor_district' 
              AND active = TRUE
              AND (
                  metric_id = %s 
                  OR metadata::jsonb->>'metric_id' = %s
              )
            ORDER BY created_at DESC
            """
            params = [metric_id, metric_id]
        else:
            query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        maps = cursor.fetchall()
        
        # Convert any date objects to ISO format strings for JSON serialization
        for map_item in maps:
            for key, value in map_item.items():
                if hasattr(value, 'isoformat'):
                    map_item[key] = value.isoformat()
                # Parse JSON strings to objects
                elif key in ('metadata', 'location_data') and value:
                    if isinstance(value, str):
                        try:
                            map_item[key] = json.loads(value)
                        except:
                            # If parsing fails, keep as string
                            pass
                    # Ensure the metadata is properly formatted as a string when it's already an object
                    elif isinstance(value, dict) or isinstance(value, list):
                        try:
                            # Convert to JSON string for display
                            map_item[key] = json.dumps(value, indent=2)
                        except:
                            # If conversion fails, use string representation
                            map_item[key] = str(value)
        
        logger.info(f"Found {len(maps)} maps for metric_id={metric_id}")
        
        cursor.close()
        conn.close()
        
        return JSONResponse(
            content={
                "status": "success",
                "metric_id": metric_id,
                "map_count": len(maps),
                "maps": maps
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting district maps: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"Failed to get district maps: {str(e)}"
            }
        )

@router.get("/api/all-maps")
@router.get("/backend/api/all-maps")  # Add an extra route
async def get_all_maps(metric_id: str = None):
    """
    Retrieve all active maps for a specific metric ID (not just district maps).
    
    Args:
        metric_id: The ID of the metric to get maps for
        
    Returns:
        JSON with a list of all active maps
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info(f"Getting all maps for metric_id={metric_id}")
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get all active maps for the specified metric
        query = """
        SELECT *
        FROM maps
        WHERE active = TRUE
        """
        
        # Add metric_id filter if provided
        params = []
        if metric_id:
            query = """
            SELECT *
            FROM maps
            WHERE active = TRUE
              AND (
                  metric_id = %s 
                  OR metadata::jsonb->>'metric_id' = %s
              )
            ORDER BY created_at DESC
            """
            params = [metric_id, metric_id]
        else:
            query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        maps = cursor.fetchall()
        
        # Convert any date objects to ISO format strings for JSON serialization
        for map_item in maps:
            for key, value in map_item.items():
                if hasattr(value, 'isoformat'):
                    map_item[key] = value.isoformat()
                # Parse JSON strings to objects
                elif key in ('metadata', 'location_data') and value:
                    if isinstance(value, str):
                        try:
                            map_item[key] = json.loads(value)
                        except:
                            # If parsing fails, keep as string
                            pass
                    # Ensure the metadata is properly formatted as a string when it's already an object
                    elif isinstance(value, dict) or isinstance(value, list):
                        try:
                            # Convert to JSON string for display
                            map_item[key] = json.dumps(value, indent=2)
                        except:
                            # If conversion fails, use string representation
                            map_item[key] = str(value)
        
        logger.info(f"Found {len(maps)} total maps for metric_id={metric_id}")
        
        cursor.close()
        conn.close()
        
        return JSONResponse(
            content={
                "status": "success",
                "metric_id": metric_id,
                "map_count": len(maps),
                "maps": maps
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting all maps: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"Failed to get all maps: {str(e)}"
            }
        )

@router.post("/convert_charts_and_finalize")
async def convert_charts_and_finalize(request: Request):
    """
    Convert charts and create both email and web versions of the report.
    This endpoint combines chart expansion for both email and web formats.
    """
    try:
        from tools.chart_expansion import expand_charts_with_tabs_final, expand_chart_references_for_email
        from pathlib import Path
        import shutil
        
        # Get request data
        data = await request.json()
        report_path = data.get("report_path")
        
        if not report_path:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing report_path parameter"
                }
            )
        
        logger.info(f"Converting charts and finalizing report at {report_path}")
        
        # Ensure the report path exists
        if not Path(report_path).exists():
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Report file not found: {report_path}"
                }
            )
        
        # Create web version (with tabs)
        web_path = None
        try:
            # Create a copy for the web version
            report_path_obj = Path(report_path)
            if '_revised' in report_path_obj.name:
                web_name = report_path_obj.name.replace('_revised', '_final')
            else:
                web_name = f"{report_path_obj.stem}_final{report_path_obj.suffix}"
            web_path = report_path_obj.parent / web_name
            
            # Copy the report to create the web version
            shutil.copy2(report_path, web_path)
            
            # Expand charts with tabs for web version
            loop = asyncio.get_event_loop()
            web_result = await loop.run_in_executor(
                None,
                lambda: expand_charts_with_tabs_final(web_path)
            )
            
            if not web_result:
                logger.warning("Failed to expand charts with tabs for web version")
                
        except Exception as e:
            logger.error(f"Error creating web version: {str(e)}")
            web_path = None
        
        # Create email version
        email_path = None
        try:
            # Create a copy for the email version
            report_path_obj = Path(report_path)
            if '_revised' in report_path_obj.name:
                email_name = report_path_obj.name.replace('_revised', '_email')
            else:
                email_name = f"{report_path_obj.stem}_email{report_path_obj.suffix}"
            email_path = report_path_obj.parent / email_name
            
            # Copy the report to create the email version
            shutil.copy2(report_path, email_path)
            
            # Expand charts for email version
            loop = asyncio.get_event_loop()
            email_result = await loop.run_in_executor(
                None,
                lambda: expand_chart_references_for_email(email_path)
            )
            
            if not email_result:
                logger.warning("Failed to expand charts for email version")
                
        except Exception as e:
            logger.error(f"Error creating email version: {str(e)}")
            email_path = None
        
        # Count charts processed (simple estimation based on file content)
        charts_processed = 0
        charts_converted = 0
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count chart placeholders
                charts_processed = content.count('[CHART:')
                # Estimate converted charts (this is a rough estimate)
                charts_converted = charts_processed
        except Exception as e:
            logger.warning(f"Error counting charts: {str(e)}")
        
        # Return success response
        return JSONResponse(
            content={
                "status": "success",
                "message": "Charts converted and outputs finalized successfully",
                "charts_processed": charts_processed,
                "charts_converted": charts_converted,
                "web_path": str(web_path) if web_path else None,
                "email_path": str(email_path) if email_path else None
            }
        )
        
    except Exception as e:
        logger.error(f"Error converting charts and finalizing: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error converting charts and finalizing: {str(e)}"
            }
        )

@router.get("/backend/map-chart")
@router.get("/map-chart")
async def get_map_chart(request: Request, id: str):
    """
    Retrieve a map by ID and either redirect to its published DataWrapper URL or serve embedded template.
    
    Args:
        id: The map ID to retrieve
        
    Returns:
        Redirect to the published DataWrapper URL, embedded template, or error if not found
    """
    try:
        import psycopg2
        import psycopg2.extras
        from fastapi.responses import RedirectResponse
        
        logger.info(f"Getting map chart for id={id}")
        
        # Check if this is an embedded request by looking for embedded parameter or referrer
        is_embedded = request.query_params.get('embedded') == 'true' or 'embedded' in str(request.url)
        logger.info(f"Map chart request - id={id}, is_embedded={is_embedded}, query_params={dict(request.query_params)}")
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get the map by ID
        cursor.execute("""
            SELECT id, title, published_url, chart_id, edit_url, type, metadata, location_data
            FROM maps 
            WHERE id = %s AND active = TRUE
        """, (id,))
        
        map_record = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not map_record:
            logger.warning(f"Map with ID {id} not found")
            return JSONResponse({
                "status": "error",
                "message": f"Map with ID {id} not found"
            }, status_code=404)
        
        logger.info(f"Map record found - id={map_record['id']}, title={map_record['title']}, published_url={bool(map_record['published_url'])}")
        
        # For embedded mode, always serve the template regardless of published URL
        if is_embedded:
            logger.info(f"Serving embedded map template for id={id}")
            logger.info(f"Map has published_url: {bool(map_record['published_url'])}")
            
            # Get location data
            location_data = map_record.get('location_data', [])
            if isinstance(location_data, str):
                try:
                    location_data = json.loads(location_data)
                except:
                    location_data = []
            
            # Get Mapbox token
            mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "")
            
            # Create metadata for the template
            metadata = {
                "metric_id": map_record.get('metadata', {}).get('metric_id', 'Unknown'),
                "district": map_record.get('metadata', {}).get('district', 'All'),
                "period_type": map_record.get('metadata', {}).get('period_type', 'Unknown'),
                "time_periods": map_record.get('metadata', {}).get('time_periods'),
                "anomaly_type": map_record.get('metadata', {}).get('anomaly_type'),
                "anomaly_field_name": map_record.get('metadata', {}).get('anomaly_field_name'),
            }
            
            return templates.TemplateResponse("map.html", {
                "request": request,
                "map_data": map_record,
                "metadata": metadata,
                "location_data": location_data,
                "config": {
                    "MAPBOX_ACCESS_TOKEN": mapbox_token
                }
            })
        
        # For non-embedded mode, check if we have a published URL
        if map_record['published_url']:
            logger.info(f"Redirecting to published URL: {map_record['published_url']}")
            return RedirectResponse(url=map_record['published_url'])
        else:
            # If no published URL, serve the embedded template instead of trying to create one
            logger.info(f"Map {id} doesn't have published URL, serving embedded template")
            
            # Get location data
            location_data = map_record.get('location_data', [])
            if isinstance(location_data, str):
                try:
                    location_data = json.loads(location_data)
                except:
                    location_data = []
            
            # Get Mapbox token
            mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "")
            
            # Create metadata for the template
            metadata = {
                "metric_id": map_record.get('metadata', {}).get('metric_id', 'Unknown'),
                "district": map_record.get('metadata', {}).get('district', 'All'),
                "period_type": map_record.get('metadata', {}).get('period_type', 'Unknown'),
                "time_periods": map_record.get('metadata', {}).get('time_periods'),
                "anomaly_type": map_record.get('metadata', {}).get('anomaly_type'),
                "anomaly_field_name": map_record.get('metadata', {}).get('anomaly_field_name'),
            }
            
            return templates.TemplateResponse("map.html", {
                "request": request,
                "map_data": map_record,
                "metadata": metadata,
                "location_data": location_data,
                "config": {
                    "MAPBOX_ACCESS_TOKEN": mapbox_token
                }
            })
        
    except Exception as e:
        logger.error(f"Error retrieving map chart: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": f"Error retrieving map chart: {str(e)}"
        }, status_code=500)

@router.get("/api/district-maps")
@router.get("/backend/api/district-maps")  # Add an extra route
async def get_district_maps(metric_id: str = None):
    """
    Retrieve district maps for a specific metric ID.
    
    Args:
        metric_id: The ID of the metric to get maps for
        
    Returns:
        JSON with a list of maps including their IDs and URLs
    """
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info(f"Getting district maps for metric_id={metric_id}")
        
        # Connect to PostgreSQL
        conn = get_postgres_connection()
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query to get maps for the specified metric
        query = """
        SELECT *
        FROM maps
        WHERE type = 'supervisor_district' AND active = TRUE
        """
        
        # Add metric_id filter if provided
        params = []
        if metric_id:
            # First try to use the direct metric_id column
            query = """
            SELECT *
            FROM maps
            WHERE type = 'supervisor_district' 
              AND active = TRUE
              AND (
                  metric_id = %s 
                  OR metadata::jsonb->>'metric_id' = %s
              )
            ORDER BY created_at DESC
            """
            params = [metric_id, metric_id]
        else:
            query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        maps = cursor.fetchall()
        
        # Convert any date objects to ISO format strings for JSON serialization
        for map_item in maps:
            for key, value in map_item.items():
                if hasattr(value, 'isoformat'):
                    map_item[key] = value.isoformat()
                # Parse JSON strings to objects
                elif key in ('metadata', 'location_data') and value:
                    if isinstance(value, str):
                        try:
                            map_item[key] = json.loads(value)
                        except:
                            # If parsing fails, keep as string
                            pass
                    # Ensure the metadata is properly formatted as a string when it's already an object
                    elif isinstance(value, dict) or isinstance(value, list):
                        try:
                            # Convert to JSON string for display
                            map_item[key] = json.dumps(value, indent=2)
                        except:
                            # If conversion fails, use string representation
                            map_item[key] = str(value)
        
        logger.info(f"Found {len(maps)} maps for metric_id={metric_id}")
        
        cursor.close()
        conn.close()
        
        return JSONResponse(
            content={
                "status": "success",
                "metric_id": metric_id,
                "map_count": len(maps),
                "maps": maps
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting district maps: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"Failed to get district maps: {str(e)}"
            }
        )

@router.get("/api/enhanced-queries")
async def get_enhanced_queries_db():
    """Serve enhanced dashboard queries built entirely from database."""
    try:
        from tools.db_utils import get_postgres_connection
        import psycopg2.extras
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query all active metrics from the database
        cursor.execute("""
            SELECT 
                m.id,
                m.metric_name,
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
                d.title as dataset_title_from_datasets,
                d.category as dataset_category_from_datasets,
                d.columns as dataset_columns
            FROM metrics m
            LEFT JOIN datasets d ON m.endpoint = d.endpoint AND d.is_active = true
            WHERE m.is_active = true
            ORDER BY m.category, m.subcategory, m.id
        """)
        
        metrics_rows = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Build the enhanced queries structure
        enhanced_queries = {}
        
        for row in metrics_rows:
            category = row['category']
            subcategory = row['subcategory'] or category
            metric_name = row['metric_name']
            
            # Initialize category if not exists
            if category not in enhanced_queries:
                enhanced_queries[category] = {}
            
            # Initialize subcategory if not exists
            if subcategory not in enhanced_queries[category]:
                enhanced_queries[category][subcategory] = {
                    "queries": {}
                }
            
            # Use dataset info from datasets table if available, otherwise from metrics table
            dataset_title = row['dataset_title_from_datasets'] or row['dataset_title'] or ""
            dataset_category = row['dataset_category_from_datasets'] or row['dataset_category'] or ""
            
            # Build the metric data
            metric_data = {
                "id": row['id'],
                "endpoint": row['endpoint'],
                "summary": row['summary'] or "",
                "definition": row['definition'] or "",
                "data_sf_url": row['data_sf_url'] or "",
                "show_on_dash": "yes" if row['show_on_dash'] else "no",
                "item_noun": row['item_noun'] or "Items",
                "ytd_query": row['ytd_query'] or "",
                "metric_query": row['metric_query'] or "",
                "dataset_title": dataset_title,
                "dataset_category": dataset_category,
                "greendirection": row['greendirection'] or "up",
                "location_fields": row['location_fields'] or [],
                "category_fields": row['category_fields'] or []
            }
            
            # Add metadata if available
            if row['metadata']:
                metric_data["metadata"] = row['metadata']
            
            # Add the metric to the enhanced queries
            enhanced_queries[category][subcategory]["queries"][metric_name] = metric_data
        
        logger.info(f"Built enhanced queries from database with {len(metrics_rows)} metrics")
        return JSONResponse(content=enhanced_queries)
        
    except Exception as e:
        logger.error(f"Error building enhanced queries from database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building enhanced queries: {str(e)}")

@router.get("/get-endpoint-columns/{endpoint}")
async def get_endpoint_columns_db(endpoint: str):
    """Get available columns for an endpoint from database only."""
    try:
        from tools.db_utils import get_postgres_connection
        import psycopg2.extras
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query the datasets table for this endpoint
        cursor.execute("""
            SELECT columns
            FROM datasets 
            WHERE endpoint = %s AND is_active = true
        """, (endpoint,))
        
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if row and row['columns']:
            # Extract column names from the database
            columns = [col["fieldName"] for col in row['columns'] if col.get("fieldName")]
            
            return JSONResponse({
                "status": "success",
                "columns": columns
            })
        else:
            logger.warning(f"No columns found in database for endpoint: {endpoint}")
            raise HTTPException(status_code=404, detail=f"No columns found for endpoint: {endpoint}")
            
    except Exception as e:
        logger.error(f"Error getting columns for endpoint '{endpoint}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting columns: {str(e)}")

@router.get("/get-selected-columns/{endpoint}")
async def get_selected_columns_db(endpoint: str, metric_id: str = None):
    """Get currently selected columns for an endpoint and optionally a specific metric ID from database."""
    try:
        from tools.db_utils import get_postgres_connection
        import psycopg2.extras
        import json
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query the metrics table for this endpoint and metric_id
        if metric_id:
            cursor.execute("""
                SELECT category_fields
                FROM metrics 
                WHERE endpoint = %s AND id = %s AND is_active = true
            """, (endpoint, int(metric_id)))
        else:
            # If no metric_id provided, get the first metric with this endpoint
            cursor.execute("""
                SELECT category_fields
                FROM metrics 
                WHERE endpoint = %s AND is_active = true
                LIMIT 1
            """, (endpoint,))
        
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if row and row['category_fields']:
            # Parse the JSON string if it's a string, otherwise use as-is
            category_fields = row['category_fields']
            if isinstance(category_fields, str):
                try:
                    category_fields = json.loads(category_fields)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing category_fields JSON for endpoint '{endpoint}': {e}")
                    return JSONResponse({
                        "status": "success",
                        "columns": []
                    })
            
            # Extract field names from category_fields
            selected_columns = [field["fieldName"] for field in category_fields if field.get("fieldName")]
            
            logger.info(f"Found selected columns for endpoint {endpoint}" + 
                       (f" and metric_id {metric_id}" if metric_id else "") + 
                       f": {selected_columns}")
            
            return JSONResponse({
                "status": "success",
                "columns": selected_columns
            })
        else:
            logger.info(f"No selected columns found for endpoint {endpoint}" + 
                       (f" and metric_id {metric_id}" if metric_id else ""))
            return JSONResponse({
                "status": "success",
                "columns": []
            })
            
    except Exception as e:
        logger.error(f"Error getting selected columns for endpoint '{endpoint}'" + 
                    (f" and metric_id '{metric_id}'" if metric_id else "") + 
                    f": {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting selected columns: {str(e)}")

@router.post("/update-selected-columns")
async def update_selected_columns_db(request: Request):
    """Update the selected columns for a metric in the database."""
    try:
        from tools.db_utils import get_postgres_connection
        import psycopg2.extras
        
        # Get request data
        data = await request.json()
        endpoint = data.get("endpoint")
        columns = data.get("columns", [])
        metric_id = data.get("metric_id")
        
        if not endpoint or not metric_id:
            raise HTTPException(status_code=400, detail="Both endpoint and metric_id are required")
        
        connection = get_postgres_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build the category_fields JSON structure
        category_fields = [
            {
                "name": col,
                "fieldName": col,
                "description": f"Selected column for metric analysis"
            }
            for col in columns
        ]
        
        # Update the metric in the database
        cursor.execute("""
            UPDATE metrics 
            SET category_fields = %s, updated_at = CURRENT_TIMESTAMP
            WHERE endpoint = %s AND id = %s AND is_active = true
        """, (json.dumps(category_fields), endpoint, int(metric_id)))
        
        if cursor.rowcount == 0:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail=f"No metric found with endpoint: {endpoint} and id: {metric_id}")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully updated category_fields for metric {metric_id} with endpoint {endpoint}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Updated category_fields for metric {metric_id} with endpoint {endpoint}"
        })
        
    except Exception as e:
        logger.error(f"Error updating category_fields: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating selected columns: {str(e)}")

# Import and include evals router
import routes.evals as evals_routes
evals_routes.set_templates(templates)

@router.get("/dashboard")
async def dashboard_page(request: Request):
    """Serve the dashboard page."""
    logger.debug("Dashboard page route called")
    if templates is None:
        logger.error("Templates not initialized in backend router")
        raise RuntimeError("Templates not initialized")
    
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/fetch_metadata")
async def fetch_metadata_route():
    """Fetch SF dataset metadata using fetch_metadata.py and store it in Postgres."""
    logger.debug("Fetch metadata route called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "fetch_metadata.py")

        # Ensure logs directory exists
        logs_dir = os.path.join(script_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        log_file = os.path.join(logs_dir, "fetch_metadata.log")

        # Clear the log file before running
        try:
            with open(log_file, 'w') as f:
                f.write("")
        except Exception:
            # If log file cannot be written, still proceed without halting
            log_file = None

        # Run the script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)

        # Read the log file content if available
        log_content = ""
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
            except Exception as e:
                log_content = f"Error reading log file: {str(e)}"

        if result.returncode == 0:
            logger.info("Metadata fetched and stored successfully.")
            return JSONResponse({
                "status": "success",
                "message": "Metadata fetched and stored successfully.",
                "output": result.stdout,
                "log_content": log_content
            })
        else:
            logger.error(f"Metadata fetch failed: {result.stderr}")
            return JSONResponse({
                "status": "error",
                "message": "Failed to fetch metadata.",
                "output": result.stderr,
                "log_content": log_content
            })
    except Exception as e:
        logger.exception(f"Error fetching metadata: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "log_content": "Error occurred before log file could be read"
        })


@router.get("/refresh_dataset_urls")
async def refresh_dataset_urls_route():
    """Refresh the dataset URLs list using fetch_dataset_urls.py."""
    logger.debug("Refresh dataset URLs route called")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "fetch_dataset_urls.py")

        # Ensure logs directory exists
        logs_dir = os.path.join(script_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        log_file = os.path.join(logs_dir, "fetch_dataset_urls.log")

        # Clear the log file before running
        try:
            with open(log_file, 'w') as f:
                f.write("")
        except Exception:
            # If log file cannot be written, still proceed without halting
            log_file = None

        # Run the script with proper working directory and environment
        result = subprocess.run(
            [sys.executable, script_path],  # Use sys.executable instead of "python"
            capture_output=True, 
            text=True,
            cwd=script_dir,  # Set working directory to script_dir
            env=os.environ.copy()  # Pass current environment variables
        )

        # Read the log file content if available
        log_content = ""
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
            except Exception as e:
                log_content = f"Error reading log file: {str(e)}"

        if result.returncode == 0:
            logger.info("Dataset URLs refreshed successfully.")
            return JSONResponse({
                "status": "success",
                "message": "Dataset URLs refreshed successfully.",
                "output": result.stdout,
                "log_content": log_content
            })
        else:
            logger.error(f"Dataset URLs refresh failed: {result.stderr}")
            return JSONResponse({
                "status": "error",
                "message": "Failed to refresh dataset URLs.",
                "output": result.stderr,
                "log_content": log_content
            })
    except Exception as e:
        logger.exception(f"Error refreshing dataset URLs: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "log_content": "Error occurred before log file could be read"
        })

@router.post("/create_monthly_report")
async def create_monthly_report(request: Request):
    """Create a new monthly report."""
    logger.debug("Create monthly report called")
    try:
        data = await request.json()
        report_name = data.get("report_name", "New Report")
        district = data.get("district", "0")
        period_type = data.get("period_type", "month")
        
        # Import the monthly report functions
        from monthly_report import initialize_monthly_reporting_table
        from tools.db_utils import execute_with_connection
        import os
        from datetime import datetime
        
        # Initialize the monthly reporting table
        init_result = initialize_monthly_reporting_table()
        if not init_result:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to initialize monthly reporting table"}
            )
        
        def create_report_operation(connection):
            cursor = connection.cursor()
            
            # Get current date
            report_date = datetime.now().date()
            
            # Generate filename based on report name
            safe_name = "".join(c for c in report_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"{safe_name}_{report_date.strftime('%Y%m%d_%H%M%S')}.html"
            
            # Create a record in the reports table
            cursor.execute("""
                INSERT INTO reports (
                    max_items, district, period_type, original_filename, revised_filename,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                ) RETURNING id
            """, (
                10,  # Default max_items
                district,
                period_type,
                filename,
                None  # revised_filename starts as null
            ))
            
            report_id = cursor.fetchone()[0]
            connection.commit()
            cursor.close()
            return report_id
        
        # Execute the operation
        result = execute_with_connection(
            operation=create_report_operation,
            db_host=os.getenv("POSTGRES_HOST", "localhost"),
            db_port=int(os.getenv("POSTGRES_PORT", "5432")),
            db_name=os.getenv("POSTGRES_DB", "transparentsf"),
            db_user=os.getenv("POSTGRES_USER", "postgres"),
            db_password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        if result["status"] == "success":
            return JSONResponse({
                "status": "success",
                "message": f"Successfully created new report: {report_name}",
                "report_id": result["result"]
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"Error creating report: {result['message']}"
            }, status_code=500)
            
    except Exception as e:
        error_message = f"Error creating monthly report: {str(e)}"
        logger.error(error_message)
        return JSONResponse({
            "status": "error",
            "message": error_message
        }, status_code=500)

@router.get("/get_report_items/{report_id}")
async def get_report_items(report_id: int):
    """Get items for a specific monthly report."""
    logger.debug(f"Get report items called for report {report_id}")
    try:
        # Import the necessary function
        from monthly_report import get_monthly_reports_list
        from tools.db_utils import execute_with_connection
        import os
        
        def get_report_items_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Get items for this report
            cursor.execute("""
                SELECT id, report_id, item_title, metric_name, metric_id, group_value, 
                       group_field_name, period_type, comparison_mean, recent_mean, 
                       difference, percent_change, rationale, explanation, priority, 
                       district, metadata
                FROM monthly_reporting 
                WHERE report_id = %s
                ORDER BY priority ASC
            """, (report_id,))
            
            items = cursor.fetchall()
            
            # Format the items
            formatted_items = []
            for item in items:
                formatted_items.append({
                    "id": item['id'],
                    "report_id": item['report_id'],
                    "item_title": item['item_title'],
                    "metric_name": item['metric_name'],
                    "metric_id": item['metric_id'],
                    "group_value": item['group_value'],
                    "group_field_name": item['group_field_name'],
                    "period_type": item['period_type'],
                    "comparison_mean": float(item['comparison_mean']) if item['comparison_mean'] and not math.isnan(float(item['comparison_mean'])) else 0,
                    "recent_mean": float(item['recent_mean']) if item['recent_mean'] and not math.isnan(float(item['recent_mean'])) else 0,
                    "difference": float(item['difference']) if item['difference'] and not math.isnan(float(item['difference'])) else 0,
                    "percent_change": float(item['percent_change']) if item['percent_change'] and not math.isnan(float(item['percent_change'])) else 0,
                    "rationale": item['rationale'],
                    "explanation": item['explanation'],
                    "priority": item['priority'],
                    "district": item['district'],
                    "metadata": item['metadata']
                })
            
            cursor.close()
            return formatted_items
        
        # Execute the operation
        result = execute_with_connection(
            operation=get_report_items_operation,
            db_host=os.getenv("POSTGRES_HOST", "localhost"),
            db_port=int(os.getenv("POSTGRES_PORT", "5432")),
            db_name=os.getenv("POSTGRES_DB", "transparentsf"),
            db_user=os.getenv("POSTGRES_USER", "postgres"),
            db_password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        if result["status"] == "success":
            return JSONResponse({
                "status": "success",
                "items": result["result"]
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"Error getting report items: {result['message']}"
            }, status_code=500)
            
    except Exception as e:
        error_message = f"Error getting report items: {str(e)}"
        logger.error(error_message)
        return JSONResponse({
            "status": "error",
            "message": error_message
        }, status_code=500)

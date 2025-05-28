import os
import json
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Templates will be set by the main backend
templates = None

def set_templates(t):
    global templates
    templates = t

# Import evals functionality
from evals import run_and_get_tool_calls, run_all_evals, agent as analyst_agent

@router.get("/run-evals")
async def run_evals_endpoint(query: str):
    """Run evals with the specified query and return results."""
    logger.info(f"Running evals with query: {query}")
    
    try:
        # Run the query through the evals system
        tool_calls = run_and_get_tool_calls(analyst_agent, query)
        
        # Get the log filename that was used
        from evals import log_filename
        
        # Check if the log file exists
        if not os.path.exists(log_filename):
            return JSONResponse({
                "status": "error",
                "message": "Log file not found after running eval"
            }, status_code=500)
        
        # Read the log file content
        with open(log_filename, 'r') as log_file:
            log_content = log_file.read()
        
        return JSONResponse({
            "status": "success",
            "message": f"Eval completed successfully for query: {query}",
            "tool_calls_count": len(tool_calls),
            "log_filename": os.path.basename(log_filename),
            "log_content": log_content
        })
    except Exception as e:
        logger.exception(f"Error running evals: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/run-all-evals")
async def run_all_evals_endpoint():
    """Run all eval test cases and return success/failure counts with details."""
    logger.info("Running all evals")
    
    try:
        # Run all evals
        results = run_all_evals()
        
        return JSONResponse({
            "status": "success",
            "message": f"All evals completed: {results['successful']}/{results['total']} tests passed",
            "results": results
        })
    except Exception as e:
        logger.exception(f"Error running all evals: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/evals-interface")
async def evals_interface(request: Request):
    """Serve the evals interface."""
    logger.debug("Evals interface route called")
    if templates is None:
        logger.error("Templates not initialized in evals router")
        raise RuntimeError("Templates not initialized")
    
    return templates.TemplateResponse("evals.html", {
        "request": request
    })

@router.get("/list-eval-logs")
async def list_eval_logs():
    """List all eval log files."""
    logger.debug("List eval logs route called")
    
    try:
        log_folder = 'logs/evals'
        if not os.path.exists(log_folder):
            return JSONResponse({
                "status": "success",
                "files": []
            })
        
        files = []
        for filename in os.listdir(log_folder):
            if filename.endswith('.log'):
                file_path = os.path.join(log_folder, filename)
                files.append({
                    "name": filename,
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        
        # Sort files by modification time, most recent first
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return JSONResponse({
            "status": "success",
            "files": files
        })
    except Exception as e:
        logger.exception(f"Error listing eval logs: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/eval-log/{filename}")
async def get_eval_log(filename: str):
    """Get the content of an eval log file."""
    logger.debug(f"Get eval log content for {filename}")
    
    try:
        log_folder = 'logs/evals'
        file_path = os.path.join(log_folder, filename)
        
        if not os.path.exists(file_path):
            return JSONResponse({
                "status": "error",
                "message": f"Log file {filename} not found"
            }, status_code=404)
        
        with open(file_path, 'r') as log_file:
            content = log_file.read()
        
        return JSONResponse({
            "status": "success",
            "filename": filename,
            "content": content
        })
    except Exception as e:
        logger.exception(f"Error getting eval log content: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/get-latest-eval-summary")
async def get_latest_eval_summary():
    """Get the latest eval summary results."""
    logger.debug("Get latest eval summary route called")
    
    try:
        log_folder = 'logs/evals'
        if not os.path.exists(log_folder):
            return JSONResponse({
                "status": "success",
                "summary": None
            })
        
        # Find the most recent eval summary file
        summary_files = [f for f in os.listdir(log_folder) if f.startswith('eval_summary_') and f.endswith('.log')]
        if not summary_files:
            return JSONResponse({
                "status": "success",
                "summary": None
            })
        
        # Sort by modification time, most recent first
        summary_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_folder, x)), reverse=True)
        latest_file = summary_files[0]
        
        # Read the summary file and extract the final summary
        file_path = os.path.join(log_folder, latest_file)
        with open(file_path, 'r') as log_file:
            content = log_file.read()
        
        # Parse the log entries to find the final summary
        log_entries = content.split('\n\n')
        summary_data = None
        
        for entry in reversed(log_entries):  # Start from the end
            if entry.strip():
                try:
                    log_entry = json.loads(entry)
                    if log_entry.get('type') == 'eval_run_complete':
                        summary_data = log_entry.get('summary', {})
                        summary_data['timestamp'] = log_entry.get('timestamp')
                        summary_data['filename'] = latest_file
                        break
                except json.JSONDecodeError:
                    continue
        
        return JSONResponse({
            "status": "success",
            "summary": summary_data
        })
    except Exception as e:
        logger.exception(f"Error getting latest eval summary: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500) 
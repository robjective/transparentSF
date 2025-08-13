import os
import json
import math
from datetime import datetime, timedelta, date
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import logging

class DateAwareJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles date, datetime objects, and NaN values."""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)
    
    def encode(self, obj):
        """Override encode to handle NaN and infinity values."""
        return super().encode(self._sanitize_for_json(obj))
    
    def _sanitize_for_json(self, obj):
        """Recursively sanitize an object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None if obj > 0 else None
            return obj
        elif obj is None:
            return None
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return self._sanitize_for_json(obj.item())
        else:
            return obj

def make_json_serializable(obj):
    """Convert an object to JSON serializable format."""
    try:
        return json.loads(json.dumps(obj, cls=DateAwareJSONEncoder))
    except (ValueError, TypeError) as e:
        # If there's still an issue, try a more aggressive sanitization
        logger.warning(f"JSON serialization failed, attempting fallback: {str(e)}")
        sanitized = DateAwareJSONEncoder()._sanitize_for_json(obj)
        return json.loads(json.dumps(sanitized, cls=DateAwareJSONEncoder))

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
from evals import run_all_evals

# Import new eval system
from tools.eval_manager import (
    get_eval_groups, get_evals, get_eval_by_id, create_eval_group, create_eval,
    update_eval_group, delete_eval_group, update_eval, delete_eval,
    save_eval_result, get_eval_results, get_eval_summary
)
from tools.eval_runner import EvalRunner

# Initialize eval runner
eval_runner = EvalRunner()

@router.get("/run-evals")
async def run_evals_endpoint(query: str, model_key: str = None):
    """Run a single eval with the specified query and return results."""
    logger.info(f"Running single eval with query: '{query}' and model: {model_key or 'default'}")
    
    try:
        from evals import run_single_eval_langchain
        
        # Run the query through the new single-eval function
        result = run_single_eval_langchain(query, model_key=model_key)
        
        log_filename = result.get("log_filename")
        if not log_filename or not os.path.exists(log_filename):
            # Correctly locate the log folder path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_folder = os.path.join(script_dir, '..', 'logs', 'evals')
            log_path_check = os.path.join(log_folder, log_filename) if log_filename else ""
            if not os.path.exists(log_path_check):
                return JSONResponse({
                    "status": "error",
                    "message": f"Log file not found after running eval. Checked: {log_filename} and {log_path_check}"
                }, status_code=500)
            log_filename = log_path_check
        
        # Read the log file content
        with open(log_filename, 'r') as log_file:
            log_content = log_file.read()
            
        # The explanation is now in the result from the agent
        explanation = result.get("explanation", "")
        
        return JSONResponse({
            "status": "success",
            "message": f"Eval completed successfully for query: {query}",
            "tool_calls_count": result.get("tool_calls_count", 0),
            "log_filename": os.path.basename(log_filename),
            "log_content": log_content,
            "explanation": explanation
        })
    except Exception as e:
        logger.exception(f"Error running single eval: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

# New database-based eval routes

@router.get("/api/eval-groups")
async def get_eval_groups_endpoint(category: str = None, active_only: bool = True):
    """Get eval groups with optional filtering."""
    try:
        result = get_eval_groups(category=category, active_only=active_only)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error getting eval groups: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/evals")
async def get_evals_endpoint(group_id: int = None, active_only: bool = True):
    """Get evals with optional filtering."""
    try:
        result = get_evals(group_id=group_id, active_only=active_only)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error getting evals: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/evals/{eval_id}")
async def get_eval_by_id_endpoint(eval_id: int):
    """Get a specific eval by ID."""
    try:
        result = get_eval_by_id(eval_id)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error getting eval by ID: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/api/eval-groups")
async def create_eval_group_endpoint(request: Request):
    """Create a new eval group."""
    try:
        body = await request.json()
        name = body.get("name")
        description = body.get("description", "")
        category = body.get("category")
        
        if not name or not category:
            return JSONResponse({
                "status": "error",
                "message": "name and category are required"
            }, status_code=400)
        
        result = create_eval_group(name, description, category)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error creating eval group: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/api/evals")
async def create_eval_endpoint(request: Request):
    """Create a new eval."""
    try:
        body = await request.json()
        group_id = body.get("group_id")
        name = body.get("name")
        description = body.get("description", "")
        prompt = body.get("prompt")
        success_criteria = body.get("success_criteria")
        success_type = body.get("success_type", "binary")
        expected_tool_calls = body.get("expected_tool_calls", [])
        expected_outputs = body.get("expected_outputs", [])
        difficulty_level = body.get("difficulty_level", "medium")
        estimated_time_minutes = body.get("estimated_time_minutes", 5)
        
        if not all([group_id, name, prompt, success_criteria]):
            return JSONResponse({
                "status": "error",
                "message": "group_id, name, prompt, and success_criteria are required"
            }, status_code=400)
        
        result = create_eval(
            group_id=group_id,
            name=name,
            description=description,
            prompt=prompt,
            success_criteria=success_criteria,
            success_type=success_type,
            expected_tool_calls=expected_tool_calls,
            expected_outputs=expected_outputs,
            difficulty_level=difficulty_level,
            estimated_time_minutes=estimated_time_minutes
        )
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error creating eval: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/api/run-eval/{eval_id}")
async def run_eval_endpoint(eval_id: int, request: Request):
    """Run a specific eval and save the result."""
    try:
        body = await request.json() if await request.body() else {}
        model_name = body.get("model_name", "test-model")
        
        result = eval_runner.run_eval(eval_id, model_name)
        # Make the result JSON serializable
        serializable_result = make_json_serializable(result)
        return JSONResponse(serializable_result)
    except Exception as e:
        logger.exception(f"Error running eval: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/api/run-eval-group/{group_id}")
async def run_eval_group_endpoint(group_id: int, request: Request):
    """Run all evals in a group and save the results."""
    try:
        body = await request.json() if await request.body() else {}
        model_name = body.get("model_name", "test-model")
        
        result = eval_runner.run_eval_group(group_id, model_name)
        # Make the result JSON serializable
        serializable_result = make_json_serializable(result)
        return JSONResponse(serializable_result)
    except Exception as e:
        logger.exception(f"Error running eval group: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/eval-results")
async def get_eval_results_endpoint(eval_id: int = None, model_name: str = None, limit: int = 100):
    """Get eval results with optional filtering."""
    try:
        result = get_eval_results(eval_id=eval_id, model_name=model_name, limit=limit)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error getting eval results: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/eval-summary")
async def get_eval_summary_endpoint():
    """Get a summary of eval results."""
    try:
        result = get_eval_summary()
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error getting eval summary: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.put("/api/eval-groups/{group_id}")
async def update_eval_group_endpoint(group_id: int, request: Request):
    """Update an eval group."""
    try:
        body = await request.json()
        result = update_eval_group(group_id, body)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error updating eval group: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.delete("/api/eval-groups/{group_id}")
async def delete_eval_group_endpoint(group_id: int):
    """Delete an eval group."""
    try:
        result = delete_eval_group(group_id)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error deleting eval group: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.put("/api/evals/{eval_id}")
async def update_eval_endpoint(eval_id: int, request: Request):
    """Update an eval."""
    try:
        body = await request.json()
        result = update_eval(eval_id, body)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error updating eval: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.delete("/api/evals/{eval_id}")
async def delete_eval_endpoint(eval_id: int):
    """Delete an eval."""
    try:
        result = delete_eval(eval_id)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error deleting eval: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/run-all-evals")
async def run_all_evals_endpoint(model_key: str = None):
    """Run all eval test cases and return success/failure counts with details."""
    logger.info(f"Running all evals with model: {model_key or 'default'}")
    
    try:
        # Run all evals
        results = run_all_evals(model_key=model_key)
        
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

@router.get("/run-model-comparison")
async def run_model_comparison_endpoint(models: str = None):
    """Run eval test cases across multiple models and return comparison results."""
    if not models:
        return JSONResponse({
            "status": "error",
            "message": "models parameter required (comma-separated list)"
        }, status_code=400)
    
    logger.info(f"Running model comparison with models: {models}")
    
    try:
        model_list = [m.strip() for m in models.split(',')]
        from evals import run_model_comparison
        
        results = run_model_comparison(model_list)
        
        return JSONResponse({
            "status": "success",
            "message": f"Model comparison completed for {len(model_list)} models",
            "results": results
        })
    except Exception as e:
        logger.exception(f"Error running model comparison: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/evals-interface")
async def evals_interface(request: Request):
    """Serve the evals interface HTML page."""
    if not templates:
        raise HTTPException(status_code=500, detail="Templates not configured")
    
    return templates.TemplateResponse("evals.html", {"request": request})

@router.get("/list-eval-logs")
async def list_eval_logs():
    """List all eval log files with metadata."""
    try:
        # Get the logs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(script_dir, '..', 'logs', 'evals')
        
        if not os.path.exists(log_folder):
            return JSONResponse({
                "status": "success",
                "files": []
            })
        
        files = []
        for filename in os.listdir(log_folder):
            if filename.endswith('.log'):
                file_path = os.path.join(log_folder, filename)
                stat = os.stat(file_path)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by modification time, newest first
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
    """Get the content of a specific eval log file."""
    try:
        # Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JSONResponse({
                "status": "error",
                "message": "Invalid filename"
            }, status_code=400)
        
        # Get the logs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(script_dir, '..', 'logs', 'evals')
        file_path = os.path.join(log_folder, filename)
        
        if not os.path.exists(file_path):
            return JSONResponse({
                "status": "error",
                "message": "Log file not found"
            }, status_code=404)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return JSONResponse({
            "status": "success",
            "filename": filename,
            "content": content
        })
    except Exception as e:
        logger.exception(f"Error reading eval log: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/get-latest-eval-summary")
async def get_latest_eval_summary():
    """Get a summary of the latest eval results."""
    try:
        # Get the logs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(script_dir, '..', 'logs', 'evals')
        
        if not os.path.exists(log_folder):
            return JSONResponse({
                "status": "success",
                "summary": {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "success_rate": 0.0,
                    "latest_runs": []
                }
            })
        
        # Get all log files
        log_files = []
        for filename in os.listdir(log_folder):
            if filename.endswith('.log'):
                file_path = os.path.join(log_folder, filename)
                stat = os.stat(file_path)
                log_files.append({
                    "filename": filename,
                    "modified": stat.st_mtime,
                    "path": file_path
                })
        
        # Sort by modification time, newest first
        log_files.sort(key=lambda x: x["modified"], reverse=True)
        
        # Analyze the latest runs
        latest_runs = []
        total_runs = 0
        successful_runs = 0
        failed_runs = 0
        
        for log_file in log_files[:10]:  # Analyze last 10 runs
            try:
                with open(log_file["path"], 'r') as f:
                    content = f.read()
                
                # Simple success/failure detection
                success_indicators = ["SUCCESS", "PASSED", "completed successfully"]
                failure_indicators = ["FAILED", "ERROR", "Exception", "Traceback"]
                
                is_success = any(indicator in content.upper() for indicator in success_indicators)
                is_failure = any(indicator in content.upper() for indicator in failure_indicators)
                
                if is_success and not is_failure:
                    successful_runs += 1
                else:
                    failed_runs += 1
                
                total_runs += 1
                
                latest_runs.append({
                    "filename": log_file["filename"],
                    "timestamp": datetime.fromtimestamp(log_file["modified"]).isoformat(),
                    "success": is_success and not is_failure
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing log file {log_file['filename']}: {str(e)}")
                failed_runs += 1
                total_runs += 1
        
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0.0
        
        return JSONResponse({
            "status": "success",
            "summary": {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": round(success_rate, 2),
                "latest_runs": latest_runs
            }
        })
    except Exception as e:
        logger.exception(f"Error getting eval summary: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/get-eval-history")
async def get_eval_history():
    """Get a detailed history of eval runs."""
    try:
        # Get the logs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(script_dir, '..', 'logs', 'evals')
        
        if not os.path.exists(log_folder):
            return JSONResponse({
                "status": "success",
                "history": []
            })
        
        # Get all log files
        history = []
        for filename in os.listdir(log_folder):
            if filename.endswith('.log'):
                file_path = os.path.join(log_folder, filename)
                stat = os.stat(file_path)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Extract information from log content
                    lines = content.split('\n')
                    query = ""
                    model = "unknown"
                    status = "unknown"
                    
                    for line in lines:
                        if "Query:" in line:
                            query = line.split("Query:")[1].strip()
                        elif "Model:" in line:
                            model = line.split("Model:")[1].strip()
                        elif "SUCCESS" in line.upper():
                            status = "success"
                        elif "FAILED" in line.upper() or "ERROR" in line.upper():
                            status = "failed"
                    
                    history.append({
                        "filename": filename,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "query": query,
                        "model": model,
                        "status": status,
                        "size": stat.st_size
                    })
                    
                except Exception as e:
                    logger.warning(f"Error reading log file {filename}: {str(e)}")
                    history.append({
                        "filename": filename,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "query": "Error reading file",
                        "model": "unknown",
                        "status": "error",
                        "size": stat.st_size
                    })
        
        # Sort by timestamp, newest first
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return JSONResponse({
            "status": "success",
            "history": history
        })
    except Exception as e:
        logger.exception(f"Error getting eval history: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500) 
import os
import json
import math
from datetime import datetime, timedelta, date
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

@router.post("/run-eval-streaming")
async def run_eval_streaming_endpoint(request: Request):
    """
    Run a specific eval with streaming output to show agent thinking in real-time.
    
    Expected JSON payload:
    {
        "eval_id": 5,
        "model_key": "claude-3-7-sonnet"
    }
    """
    try:
        data = await request.json()
        eval_id = data.get("eval_id")
        model_key = data.get("model_key", "claude-3-7-sonnet")
        
        if not eval_id:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No eval_id provided"
                }
            )
        
        # Get the eval details
        eval_result = get_eval_by_id(eval_id)
        if eval_result["status"] != "success":
            return JSONResponse({
                "status": "error",
                "message": f"Failed to get eval: {eval_result['message']}"
            }, status_code=400)
        
        eval_data = eval_result["eval"]
        prompt = eval_data["prompt"]
        
        logger.info(f"Starting streaming eval {eval_id}: {eval_data['name']}")
        logger.info(f"Model: {model_key}")
        logger.info(f"Prompt: {prompt}")
        
        # Create new LangChain agent with session logging enabled
        from agents.langchain_agent.explainer_agent import create_explainer_agent
        from agents.langchain_agent.config.tool_config import ToolGroup
        
        agent = create_explainer_agent(
            model_key=model_key, 
            tool_groups=[ToolGroup.CORE, ToolGroup.ANALYSIS], 
            enable_session_logging=True
        )
        
        async def generate_eval_stream():
            """Generate streaming response for eval with visible agent thinking"""
            try:
                # Send eval start signal
                yield f"data: {json.dumps({'eval_start': True, 'eval_id': eval_id, 'eval_name': eval_data['name'], 'model': model_key})}\n\n"
                
                # Use the agent's streaming method to show thinking with the eval prompt
                async for chunk in agent.explain_change_streaming(prompt, metric_details={}):
                    if chunk:
                        yield chunk
                
                # Get the most recent session to extract tool calls for eval analysis
                if hasattr(agent, 'session_logger') and agent.session_logger:
                    sessions_dir = agent.session_logger.logs_dir
                    if sessions_dir.exists():
                        session_files = sorted(sessions_dir.glob("session_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
                        if session_files:
                            try:
                                with open(session_files[0], 'r') as session_file:
                                    session_data = json.load(session_file)
                                    
                                    # Run the eval through the eval_runner to get proper evaluation
                                    eval_result = eval_runner.run_eval(eval_id, model_key)
                                    execution_result = eval_result.get("execution_result", {})
                                    
                                    # Send comprehensive eval results for database saving
                                    tool_calls = session_data.get("tool_calls", [])
                                    successful_tool_calls = [tc for tc in tool_calls if tc.get('success')]
                                    failed_tool_calls = [tc for tc in tool_calls if not tc.get('success')]
                                    
                                    eval_results = {
                                        'eval_complete': True,
                                        'eval_id': eval_id,
                                        'eval_name': eval_data['name'],
                                        'tool_calls_count': len(tool_calls),
                                        'successful_tool_calls': len(successful_tool_calls),
                                        'failed_tool_calls': len(failed_tool_calls),
                                        'success_score': execution_result.get('success_score', 0.0),
                                        'success': session_data.get("success", False),
                                        'total_execution_time_ms': session_data.get("total_execution_time_ms", 0),
                                        'session_id': session_data.get("session_id"),
                                        'final_response': session_data.get("final_response", ""),
                                        'conversation': session_data.get("conversation", []),
                                        'tool_calls_details': [
                                            {
                                                'tool_name': tc.get('tool_name'),
                                                'arguments': tc.get('arguments', {}),
                                                'success': tc.get('success'),
                                                'execution_time_ms': tc.get('execution_time_ms'),
                                                'result': tc.get('result'),
                                                'error_message': tc.get('error_message')
                                            }
                                            for tc in tool_calls
                                        ],
                                        'success_details': execution_result.get('success_details', {})
                                    }
                                    yield f"data: {json.dumps(eval_results)}\n\n"
                            except Exception as e:
                                logger.error(f"Error reading session file for eval: {e}")
                
                # Send completion signal
                yield f"data: {json.dumps({'completed': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming eval generation: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_eval_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.exception(f"Error setting up streaming eval: {str(e)}")
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

@router.post("/api/save-streaming-eval-result/{eval_id}")
async def save_streaming_eval_result_endpoint(eval_id: int, request: Request):
    """Save results from a streaming eval session without re-running the agent."""
    try:
        body = await request.json()
        model_name = body.get("model_name", "test-model")
        streaming_results = body.get("streaming_results", {})
        
        # Get eval details for prompt
        eval_result = get_eval_by_id(eval_id)
        if eval_result["status"] != "success":
            return JSONResponse({
                "status": "error",
                "message": f"Failed to get eval: {eval_result['message']}"
            }, status_code=400)
        
        eval_data = eval_result["eval"]
        prompt = eval_data["prompt"]
        
        # Extract data from streaming results
        response_text = streaming_results.get("final_response", "")
        tool_calls_details = streaming_results.get("tool_calls_details", [])
        success_score = streaming_results.get("success_score", 0.0)
        execution_time_seconds = streaming_results.get("total_execution_time_ms", 0) / 1000.0
        
        # Convert tool calls to the format expected by save_eval_result
        tool_calls_made = [
            {
                "function": tc.get("tool_name"),
                "arguments": tc.get("arguments", {}),
                "successful": tc.get("success", False),
                "execution_time_ms": tc.get("execution_time_ms", 0),
                "result": tc.get("result"),
                "error_message": tc.get("error_message")
            }
            for tc in tool_calls_details
        ]
        
        # Calculate success score based on streaming results
        if success_score == 0.0:
            # Calculate success score from tool calls if not provided
            total_tools = len(tool_calls_made)
            successful_tools = len([tc for tc in tool_calls_made if tc.get("successful")])
            success_score = (successful_tools / max(total_tools, 1)) if total_tools > 0 else 0.0
        
        # Save the result
        save_result = save_eval_result(
            eval_id=eval_id,
            model_name=model_name,
            prompt_used=prompt,
            response_received=response_text,
            tool_calls_made=tool_calls_made,
            success_score=success_score,
            success_details={
                "streaming_session": True,
                "session_id": streaming_results.get("session_id"),
                "tool_calls_count": len(tool_calls_made),
                "successful_tool_calls": len([tc for tc in tool_calls_made if tc.get("successful")]),
                "failed_tool_calls": len([tc for tc in tool_calls_made if not tc.get("successful")]),
                "total_execution_time_ms": streaming_results.get("total_execution_time_ms", 0)
            },
            execution_time_seconds=execution_time_seconds,
            status="completed",
            conversation_history=streaming_results.get("conversation", [])
        )
        
        if save_result["status"] != "success":
            return JSONResponse({
                "status": "error", 
                "message": f"Failed to save results: {save_result['message']}"
            }, status_code=500)
        
        return JSONResponse({
            "status": "success",
            "eval_id": eval_id,
            "eval_name": eval_data["name"],
            "execution_result": {
                "success_score": success_score,
                "tool_calls_count": len(tool_calls_made),
                "successful_tool_calls": len([tc for tc in tool_calls_made if tc.get("successful")]),
                "failed_tool_calls": len([tc for tc in tool_calls_made if not tc.get("successful")]),
                "response": response_text,
                "status": "completed"
            },
            "result_id": save_result.get("result_id"),
            "source": "streaming_session"
        })
        
    except Exception as e:
        logger.exception(f"Error saving streaming eval result: {str(e)}")
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

@router.get("/api/sessions-summary")
async def get_sessions_summary_endpoint():
    """Get a summary of all sessions from the logs/sessions folder."""
    try:
        # Get the sessions directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sessions_dir = os.path.join(script_dir, '..', 'logs', 'sessions')
        
        if not os.path.exists(sessions_dir):
            return JSONResponse({
                "status": "success",
                "summary": {
                    "total_sessions": 0,
                    "sessions": [],
                    "model_stats": {},
                    "agent_type_stats": {},
                    "initiation_stats": {},
                    "tool_call_stats": {}
                }
            })
        
        sessions = []
        model_stats = {}
        agent_type_stats = {}
        initiation_stats = {}
        tool_call_stats = {}
        
        # Get all session files
        for filename in os.listdir(sessions_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(sessions_dir, filename)
                stat = os.stat(file_path)
                
                try:
                    with open(file_path, 'r') as f:
                        session_data = json.load(f)
                    
                    session_id = session_data.get('session_id', filename.replace('.json', ''))
                    timestamp = session_data.get('timestamp', '')
                    agent_type = session_data.get('agent_type', 'unknown')
                    model = session_data.get('model', 'unknown')
                    user_input = session_data.get('user_input', '')
                    tool_calls = session_data.get('tool_calls', [])
                    success = session_data.get('success', False)
                    total_execution_time_ms = session_data.get('total_execution_time_ms', 0)
                    
                    # Determine initiation type based on user input or session context
                    initiation_type = "UI"
                    user_input_lower = user_input.lower()
                    
                    # Check for eval-related keywords
                    if any(keyword in user_input_lower for keyword in ["eval", "test", "evaluation", "assessment"]):
                        initiation_type = "Eval"
                    # Check for newsletter/report keywords
                    elif any(keyword in user_input_lower for keyword in ["newsletter", "report", "monthly", "weekly", "summary"]):
                        initiation_type = "Newsletter"
                    # Check for anomaly-related keywords
                    elif any(keyword in user_input_lower for keyword in ["anomaly", "anomalies", "unusual", "spike", "outlier"]):
                        initiation_type = "Anomaly"
                    # Check for metric creation/editing
                    elif any(keyword in user_input_lower for keyword in ["create metric", "new metric", "edit metric", "update metric"]):
                        initiation_type = "Metric Management"
                    # Check for data analysis
                    elif any(keyword in user_input_lower for keyword in ["analyze", "analysis", "trend", "chart", "map"]):
                        initiation_type = "Data Analysis"
                    
                    # Count tool calls by type
                    tool_call_count = len(tool_calls)
                    successful_tool_calls = len([tc for tc in tool_calls if tc.get('success')])
                    failed_tool_calls = tool_call_count - successful_tool_calls
                    
                    # Update statistics
                    model_stats[model] = model_stats.get(model, 0) + 1
                    agent_type_stats[agent_type] = agent_type_stats.get(agent_type, 0) + 1
                    initiation_stats[initiation_type] = initiation_stats.get(initiation_type, 0) + 1
                    
                    # Tool call statistics
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('tool_name', 'unknown')
                        tool_call_stats[tool_name] = tool_call_stats.get(tool_name, {
                            'total_calls': 0,
                            'successful_calls': 0,
                            'failed_calls': 0
                        })
                        tool_call_stats[tool_name]['total_calls'] += 1
                        if tool_call.get('success'):
                            tool_call_stats[tool_name]['successful_calls'] += 1
                        else:
                            tool_call_stats[tool_name]['failed_calls'] += 1
                    
                    sessions.append({
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "agent_type": agent_type,
                        "model": model,
                        "user_input": user_input[:60] + "..." if len(user_input) > 60 else user_input,
                        "tool_call_count": tool_call_count,
                        "successful_tool_calls": successful_tool_calls,
                        "failed_tool_calls": failed_tool_calls,
                        "success": success,
                        "total_execution_time_ms": total_execution_time_ms,
                        "initiation_type": initiation_type,
                        "filename": filename
                    })
                    
                except Exception as e:
                    logger.warning(f"Error reading session file {filename}: {str(e)}")
                    sessions.append({
                        "session_id": filename.replace('.json', ''),
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "agent_type": "error",
                        "model": "unknown",
                        "user_input": "Error reading file",
                        "tool_call_count": 0,
                        "successful_tool_calls": 0,
                        "failed_tool_calls": 0,
                        "success": False,
                        "total_execution_time_ms": 0,
                        "initiation_type": "Error",
                        "filename": filename
                    })
        
        # Sort sessions by timestamp, newest first
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Calculate success rates for models based on tool call success
        model_success_rates = {}
        for session in sessions:
            model = session["model"]
            if model not in model_success_rates:
                model_success_rates[model] = {"total_sessions": 0, "total_tool_calls": 0, "successful_tool_calls": 0}
            
            model_success_rates[model]["total_sessions"] += 1
            model_success_rates[model]["total_tool_calls"] += session["tool_call_count"]
            model_success_rates[model]["successful_tool_calls"] += session["successful_tool_calls"]
        
        # Convert to percentages based on tool call success rate
        for model, stats in model_success_rates.items():
            if stats["total_tool_calls"] > 0:
                success_rate = (stats["successful_tool_calls"] / stats["total_tool_calls"] * 100)
            else:
                # If no tool calls, use session success as fallback
                success_rate = 100.0  # Sessions without tool calls are considered successful
            model_success_rates[model] = round(success_rate, 1)
        
        return JSONResponse({
            "status": "success",
            "summary": {
                "total_sessions": len(sessions),
                "sessions": sessions,
                "model_stats": model_stats,
                "model_success_rates": model_success_rates,
                "agent_type_stats": agent_type_stats,
                "initiation_stats": initiation_stats,
                "tool_call_stats": tool_call_stats
            }
        })
    except Exception as e:
        logger.exception(f"Error getting sessions summary: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500) 
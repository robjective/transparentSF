"""
Explainer Chat Routes

This module contains all the explainer agent and chat-related endpoints,
moved from backend.py for better organization.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import uuid
from typing import Dict, Any

# Import the LangChain explainer agent
from agents.langchain_agent.explainer_agent import create_explainer_agent
from agents.langchain_agent.config.tool_config import ToolGroup

# Import the necessary function for available models
from agents.config.models import get_available_models, get_default_model

# Initialize router and logger
router = APIRouter()
logger = logging.getLogger(__name__)

# Templates will be set by main.py
templates = None

def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logger.info("Templates set in explainer chat router")

# Session store for explainer agents
# This stores ExplainerAgent instances by session_id to maintain conversation history
explainer_sessions: Dict[str, Any] = {}

@router.get("/available-models")
async def get_available_models_endpoint():
    """Get available models for frontend dropdowns."""
    try:
        available_models = get_available_models()
        model_list = []
        
        for model_key, model_config in available_models.items():
            model_list.append({
                "key": model_key,
                "name": model_config.full_name,
                "provider": model_config.provider.value,
                "available": model_config.is_available()
            })
        
        # Sort by provider and then by name
        model_list.sort(key=lambda x: (x["provider"], x["name"]))
        
        return JSONResponse({
            "status": "success",
            "models": model_list,
            "default_model": get_default_model()
        })
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Error getting available models: {str(e)}"
        }, status_code=500)

@router.post("/api/clear-explainer-sessions")
async def clear_explainer_sessions():
    """Clear all explainer agent sessions to force fresh agent creation."""
    global explainer_sessions
    explainer_sessions.clear()
    logger.info("Cleared all explainer agent sessions")
    return JSONResponse(content={"status": "success", "message": "All sessions cleared"})

@router.get("/api/agent-config/{session_id}")
async def get_agent_config(session_id: str):
    """Get the configuration of a specific agent session."""
    session_key = f"langchain_{session_id}"
    if session_key in explainer_sessions:
        agent = explainer_sessions[session_key]
        config = agent.get_configuration_info()
        return JSONResponse(content={"status": "success", "config": config})
    else:
        return JSONResponse(content={"status": "error", "message": "Session not found"})

@router.post("/api/explain-change")
async def explain_change_api(request: Request):
    """
    API endpoint to explain data changes using the explainer agent.
    
    Expected JSON payload:
    {
        "prompt": "Explain the change in metric X for district Y",
        "metric_id": 123,
        "district_id": 0,
        "period_type": "month",
        "return_json": true
    }
    """
    try:
        data = await request.json()
        prompt = data.get("prompt")
        metric_id = data.get("metric_id")
        district_id = data.get("district_id", 0)
        period_type = data.get("period_type", "month")
        return_json = data.get("return_json", True)
        
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No prompt provided"
                }
            )
        
        logger.info(f"Explaining change with prompt: {prompt}")
        
        # Create explainer agent
        agent = create_explainer_agent()
        
        # Prepare metric details for the agent
        metric_details = {}
        if metric_id is not None:
            metric_details = {
                "metric_id": metric_id,
                "district_id": district_id,
                "period_type": period_type
            }
            enhanced_prompt = f"""
            {prompt}
            
            Please analyze metric {metric_id} for district {district_id} over the {period_type} period.
            """
        else:
            enhanced_prompt = prompt
        
        # Get explanation using the LangChain agent interface
        result = agent.explain_change_sync(enhanced_prompt, metric_details)
        
        return JSONResponse(
            content={
                "status": "success" if result.get("success", True) else "error",
                "result": result
            }
        )
        
    except Exception as e:
        logger.error(f"Error in explain_change_api: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error explaining change: {str(e)}"
            }
        )

@router.post("/api/explain-change-streaming")
async def explain_change_streaming_api(request: Request):
    """
    Streaming API endpoint for explainer agent chat functionality.
    
    Expected JSON payload:
    {
        "prompt": "Explain the change in metric X for district Y",
        "session_data": {
            "session_id": "unique_session_id"  // Optional, will create new if not provided
        }
    }
    """
    try:
        data = await request.json()
        prompt = data.get("prompt")
        session_data = data.get("session_data", {})
        session_id = session_data.get("session_id")
        
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No prompt provided"
                }
            )
        
        logger.info(f"Starting streaming explanation with prompt: {prompt}")
        
        # Get or create explainer agent for this session
        if session_id and session_id in explainer_sessions:
            agent = explainer_sessions[session_id]
            logger.info(f"Using existing explainer agent for session: {session_id}")
        else:
            # Create new agent and session
            agent = create_explainer_agent()
            if not session_id:
                session_id = str(uuid.uuid4())
            explainer_sessions[session_id] = agent
            logger.info(f"Created new explainer agent for session: {session_id}")
        
        async def generate_stream():
            """Generate streaming response"""
            try:
                # Send session ID first so frontend can track it
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                
                # Prepare metric details for the agent
                metric_details = {}
                if session_data and 'metric_id' in session_data:
                    metric_details = {
                        "metric_id": session_data.get('metric_id'),
                        "district_id": session_data.get('district_id', 0),
                        "period_type": session_data.get('period_type', 'month')
                    }
                
                async for chunk in agent.explain_change_streaming(prompt, metric_details):
                    if chunk:
                        # The agent already yields properly formatted SSE data, so pass it through directly
                        yield chunk
                
                # Send completion signal
                yield f"data: {json.dumps({'completed': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming generation: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in explain_change_streaming_api: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error in streaming explanation: {str(e)}"
            }
        )

@router.post("/api/langchain-explainer-streaming")
async def langchain_explainer_streaming_api(request: Request):
    """
    LangChain-based streaming API endpoint for explainer agent chat functionality.
    
    Expected JSON payload:
    {
        "prompt": "Explain the change in metric X for district Y",
        "model_key": "claude-3-7-sonnet",
        "tool_groups": ["core", "analysis", "metrics"],
        "session_data": {
            "session_id": "unique_session_id"  // Optional, will create new if not provided
        }
    }
    """
    try:
        data = await request.json()
        prompt = data.get("prompt")
        model_key = data.get("model_key", "claude-3-7-sonnet")
        tool_groups = data.get("tool_groups", ["core", "analysis", "visualization"])
        session_data = data.get("session_data", {})
        session_id = session_data.get("session_id")
        
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No prompt provided"
                }
            )
        
        logger.info(f"Starting LangChain streaming explanation with prompt: {prompt}")
        logger.info(f"Model: {model_key}, Tool groups: {tool_groups}")
        logger.info(f"Tool groups type: {type(tool_groups)}, Tool groups content: {tool_groups}")
        
        # Convert tool group strings to ToolGroup enums
        tool_group_enums = []
        for group_name in tool_groups:
            try:
                tool_group_enums.append(ToolGroup(group_name))
            except ValueError:
                logger.warning(f"Unknown tool group: {group_name}")
        
        if not tool_group_enums:
            tool_group_enums = [ToolGroup.CORE, ToolGroup.ANALYSIS, ToolGroup.VISUALIZATION]
        
        # Get or create LangChain explainer agent for this session
        if not session_id:
            session_id = str(uuid.uuid4())
        session_key = f"langchain_{session_id}"
        
        if session_key in explainer_sessions:
            agent = explainer_sessions[session_key]
            # Update agent configuration if needed
            if hasattr(agent, 'model_key') and agent.model_key != model_key:
                agent = create_explainer_agent(model_key=model_key, tool_groups=tool_group_enums, enable_session_logging=True)
                explainer_sessions[session_key] = agent
            elif hasattr(agent, 'tool_groups') and agent.tool_groups != tool_group_enums:
                agent.update_tool_groups(tool_groups)
            logger.info(f"Using existing LangChain explainer agent for session: {session_key}")
        else:
            # Create new LangChain agent and session with session logging enabled
            agent = create_explainer_agent(model_key=model_key, tool_groups=tool_group_enums, enable_session_logging=True)
            explainer_sessions[session_key] = agent
            logger.info(f"Created new LangChain explainer agent for session: {session_key}")
        
        async def generate_stream():
            """Generate streaming response using LangChain agent with session logging"""
            try:
                # Send session ID first so frontend can track it
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                
                # Prepare metric details for the agent
                metric_details = {}
                if session_data and 'metric_id' in session_data:
                    metric_details = {
                        "metric_id": session_data.get('metric_id'),
                        "district_id": session_data.get('district_id', 0),
                        "period_type": session_data.get('period_type', 'month')
                    }
                
                # Use the agent's explain_change_streaming method which includes real-time tool call logging
                async for chunk in agent.explain_change_streaming(prompt, metric_details):
                    if chunk:
                        # The agent already yields properly formatted SSE data, so pass it through directly
                        yield chunk
                
                # Send completion signal
                yield f"data: {json.dumps({'completed': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in LangChain streaming generation: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in langchain_explainer_streaming_api: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error in LangChain streaming explanation: {str(e)}"
            }
        )

# Remove the backend prefix since the entire router is mounted at /backend
# @router.post("/backend/api/langchain-explainer-streaming")
# async def langchain_explainer_streaming_api_backend(request: Request):
#     """Backend prefix version of the LangChain streaming endpoint."""
#     return await langchain_explainer_streaming_api(request)

@router.post("/api/explain-metric-change")
async def explain_metric_change_api(request: Request):
    """
    Convenience API endpoint to explain a specific metric change.
    
    Expected JSON payload:
    {
        "metric_id": 123,
        "district_id": 0,
        "period_type": "month"
    }
    """
    try:
        data = await request.json()
        metric_id = data.get("metric_id")
        district_id = data.get("district_id", 0)
        period_type = data.get("period_type", "month")
        
        if metric_id is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "metric_id is required"
                }
            )
        
        logger.info(f"Explaining metric {metric_id} change for district {district_id}")
        
        # This functionality has been moved to the LangChain-based explainer agent
        # Use the /api/explain-change endpoint instead
        return JSONResponse(
            content={
                "status": "info",
                "message": "This functionality has been moved to the LangChain-based explainer agent. Use the /api/explain-change endpoint instead."
            }
        )
        
    except Exception as e:
        logger.error(f"Error in explain_metric_change_api: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error explaining metric change: {str(e)}"
            }
        )

@router.post("/api/explainer-cancel")
async def cancel_explainer_session(request: Request):
    """Cancel the current explainer session."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if session_id:
            session_key = f"langchain_{session_id}"
            if session_key in explainer_sessions:
                # Remove the session to cancel any ongoing operations
                del explainer_sessions[session_key]
                logger.info(f"Cancelled explainer session: {session_key}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "message": "Session cancelled successfully"
                    }
                )
            else:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "message": "Session not found"
                    }
                )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No session ID provided"
                }
            )
    except Exception as e:
        logger.error(f"Error cancelling explainer session: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error cancelling session: {str(e)}"
            }
        )

# Remove backend prefix since the entire router is mounted at /backend
# @router.post("/backend/api/explainer-cancel") 
# async def cancel_explainer_session_backend(request: Request):
#     """Backend prefix version of the cancel endpoint."""
#     return await cancel_explainer_session(request)

@router.post("/api/explainer-clear-session")
async def clear_explainer_session(request: Request):
    """
    Clear a specific explainer session.
    
    Expected JSON payload:
    {
        "session_id": "session_id_to_clear"
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "session_id is required"
                }
            )
        
        if session_id in explainer_sessions:
            del explainer_sessions[session_id]
            logger.info(f"Cleared explainer session: {session_id}")
            return JSONResponse(content={"status": "success", "message": f"Session {session_id} cleared"})
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }
            )
        
    except Exception as e:
        logger.error(f"Error clearing explainer session: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error clearing session: {str(e)}"
            }
        )

# Remove backend prefix since the entire router is mounted at /backend
# @router.post("/backend/api/explainer-clear-session")
# async def clear_explainer_session_backend(request: Request):
#     """Backend prefix version of the clear session endpoint.""" 
#     return await clear_explainer_session(request)

@router.post("/api/explainer-clear-all-sessions")
async def clear_all_explainer_sessions():
    """Clear all explainer sessions."""
    try:
        session_count = len(explainer_sessions)
        explainer_sessions.clear()
        logger.info(f"Cleared all {session_count} explainer sessions")
        return JSONResponse(content={
            "status": "success", 
            "message": f"Cleared {session_count} sessions"
        })
        
    except Exception as e:
        logger.error(f"Error clearing all explainer sessions: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error clearing sessions: {str(e)}"
            }
        )

@router.get("/api/explainer-sessions")
async def get_explainer_sessions():
    """Get information about active explainer sessions."""
    try:
        sessions_info = {}
        for session_id, agent in explainer_sessions.items():
            sessions_info[session_id] = {
                "message_count": len(agent.get_conversation_history()) if hasattr(agent, 'get_conversation_history') else 0,
                "agent_name": getattr(agent, 'agent', {}).get('name', 'Unknown') if hasattr(agent, 'agent') else 'LangChain Agent',
                "has_context": bool(getattr(agent, 'context_variables', {}))
            }
        
        return JSONResponse(content={
            "status": "success",
            "active_sessions": len(explainer_sessions),
            "sessions": sessions_info
        })
        
    except Exception as e:
        logger.error(f"Error getting explainer sessions: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error getting sessions: {str(e)}"
            }
        )

@router.get("/api/test-explainer-session")
async def test_explainer_session():
    """Test endpoint to verify explainer session management is working."""
    try:
        # Create a test session
        test_session_id = str(uuid.uuid4())
        agent = create_explainer_agent()
        explainer_sessions[test_session_id] = agent
        
        # Add some test messages
        agent.add_message("user", "Hello, this is a test message")
        agent.add_message("assistant", "Hello! I received your test message.")
        
        # Get session info
        session_info = {
            "test_session_id": test_session_id,
            "message_count": len(agent.get_conversation_history()) if hasattr(agent, 'get_conversation_history') else 0,
            "total_sessions": len(explainer_sessions),
            "conversation_history": agent.get_conversation_history() if hasattr(agent, 'get_conversation_history') else []
        }
        
        return JSONResponse(content={
            "status": "success",
            "message": "Session management test completed",
            "session_info": session_info
        })
        
    except Exception as e:
        logger.error(f"Error in test explainer session: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Test failed: {str(e)}"
            }
        )

@router.post("/api/langchain-explainer-continue")
async def langchain_explainer_continue_api(request: Request):
    """
    Continue LangChain-based streaming API endpoint when agent hits stop condition.
    
    Expected JSON payload:
    {
        "session_id": "unique_session_id",  // Required - existing session to continue
        "continuation_prompt": "Additional context or question",  // Optional
        "model_key": "gpt-5",  // Optional - will use existing session's model if not provided
        "tool_groups": ["core", "analysis", "metrics"]  // Optional - will use existing session's tools if not provided
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        continuation_prompt = data.get("continuation_prompt", "")
        model_key = data.get("model_key")
        tool_groups = data.get("tool_groups")
        
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No session_id provided. Cannot continue without existing session."
                }
            )
        
        logger.info(f"Continuing LangChain streaming analysis for session: {session_id}")
        logger.info(f"Continuation prompt: {continuation_prompt}")
        
        # Get existing LangChain explainer agent for this session
        session_key = f"langchain_{session_id}"
        
        if session_key not in explainer_sessions:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Session {session_id} not found. Please start a new conversation."
                }
            )
        
        agent = explainer_sessions[session_key]
        logger.info(f"Found existing LangChain explainer agent for session: {session_key}")
        
        # Update agent configuration if new parameters are provided
        if model_key and hasattr(agent, 'model_key') and agent.model_key != model_key:
            # Convert tool group strings to ToolGroup enums if provided
            tool_group_enums = []
            if tool_groups:
                for group_name in tool_groups:
                    try:
                        tool_group_enums.append(ToolGroup(group_name))
                    except ValueError:
                        logger.warning(f"Unknown tool group: {group_name}")
            else:
                tool_group_enums = getattr(agent, 'tool_groups', [ToolGroup.CORE, ToolGroup.ANALYSIS, ToolGroup.VISUALIZATION])
            
            agent = create_explainer_agent(model_key=model_key, tool_groups=tool_group_enums, enable_session_logging=True)
            explainer_sessions[session_key] = agent
            logger.info(f"Updated agent with new model: {model_key}")
        elif tool_groups and hasattr(agent, 'tool_groups'):
            # Convert tool group strings to ToolGroup enums
            tool_group_enums = []
            for group_name in tool_groups:
                try:
                    tool_group_enums.append(ToolGroup(group_name))
                except ValueError:
                    logger.warning(f"Unknown tool group: {group_name}")
            
            if tool_group_enums and agent.tool_groups != tool_group_enums:
                agent.update_tool_groups(tool_group_enums)
                logger.info(f"Updated agent with new tool groups: {tool_groups}")
        
        async def generate_continue_stream():
            """Generate streaming response for continuation using LangChain agent"""
            try:
                # Send session ID first so frontend can track it
                yield f"data: {json.dumps({'session_id': session_id, 'continued': True})}\n\n"
                
                # Use the agent's continuation streaming method
                async for chunk in agent.continue_analysis_streaming(continuation_prompt, metric_details={}):
                    yield chunk
                
            except Exception as e:
                logger.error(f"Error in LangChain continuation streaming generation: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_continue_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in langchain_explainer_continue_api: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error in LangChain continuation: {str(e)}"
            }
        )

@router.post("/api/test-session-logging")
async def test_session_logging():
    """Test endpoint to verify session logging is working."""
    try:
        # Create a test LangChain agent with session logging enabled
        agent = create_explainer_agent(
            model_key="claude-3-7-sonnet",
            tool_groups=[ToolGroup.CORE, ToolGroup.ANALYSIS, ToolGroup.VISUALIZATION],
            enable_session_logging=True
        )
        
        # Verify the agent has session logging enabled
        session_logging_enabled = getattr(agent, 'enable_session_logging', False)
        session_logger = getattr(agent, 'session_logger', None)
        logs_dir = None
        if session_logger:
            logs_dir = str(session_logger.logs_dir)
        
        # Try to run a simple sync explanation to generate a session log
        test_result = agent.explain_change_sync(
            "Test prompt for session logging verification",
            metric_details={}
        )
        
        # Check if any session files were created
        import os
        from pathlib import Path
        session_logs_dir = Path(__file__).parent.parent / 'logs' / 'sessions'
        session_files = list(session_logs_dir.glob('*.json')) if session_logs_dir.exists() else []
        
        return JSONResponse(content={
            "status": "success",
            "session_logging_enabled": session_logging_enabled,
            "session_logger_available": session_logger is not None,
            "logs_directory": logs_dir,
            "logs_directory_exists": session_logs_dir.exists() if session_logs_dir else False,
            "session_files_count": len(session_files),
            "recent_session_files": [f.name for f in session_files[-5:]] if session_files else [],
            "test_result_session_id": test_result.get('session_id') if isinstance(test_result, dict) else None,
            "agent_type": type(agent).__name__
        })
        
    except Exception as e:
        logger.error(f"Error testing session logging: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Test failed: {str(e)}",
                "error_type": type(e).__name__
            }
        )

import os
import json
import logging
import math
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# Initialize APIRouter
router = APIRouter()

# Templates will be set by the main app
templates = None

def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logging.info("Templates set in conversation router")

def clean_json_for_api(obj):
    """Clean JSON data to ensure it's compatible with FastAPI serialization."""
    if isinstance(obj, dict):
        return {k: clean_json_for_api(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_for_api(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

@router.get("/api/sessions/{session_id}")
async def get_session_data(session_id: str):
    """Get session data for conversation viewer."""
    try:
        sessions_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'sessions')
        session_file = os.path.join(sessions_dir, f'{session_id}.json')
        
        # Direct filename lookup (session files are now named after session_id)
        if os.path.exists(session_file):
            logging.info(f"Found session {session_id}")
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Clean any problematic float values for FastAPI JSON serialization
            cleaned_data = clean_json_for_api(session_data)
            return JSONResponse(cleaned_data)
        
        logging.warning(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        logging.exception(f"Error loading session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation")
async def conversation_viewer(request: Request):
    """Serve the conversation viewer page."""
    return templates.TemplateResponse("conversation_viewer.html", {"request": request})

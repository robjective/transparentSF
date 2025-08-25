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
from tools.vector_query import query_docs
from tools.genChart import generate_time_series_chart
from tools.retirementdata import read_csv_with_encoding

from tools.generate_map import generate_map, get_map_by_id, get_recent_maps
from tools.gen_map_dw import create_datawrapper_map
from tools.notes_manager import get_notes, load_and_combine_notes, initialize_notes
from pathlib import Path
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
from tools.store_anomalies import get_anomaly_details as get_anomaly_details_from_db, get_anomalies
from chart_message import generate_chart_message, generate_anomaly_chart_html
import re
from agents.explainer_agent import ExplainerAgent, create_explainer_agent, PROMPT_SECTIONS
from agents.explainer_prompts import get_complete_instructions
import importlib

# Import the necessary function for available models
from agents.config.models import get_available_models, get_default_model

# ------------------------------
# Configuration and Setup
# ------------------------------

load_dotenv()

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
# Confirm that there is an openai_api_key set
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize APIRouter
router = APIRouter()

# Templates will be set by the main app
templates = None

def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logging.info("Templates set in backend router")

 
import os
import json
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import logging
import traceback
import time
import asyncio
import requests
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm, Agent

from fastapi import APIRouter, Request, HTTPException, FastAPI, Response, status
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import math
import uuid
from typing import Dict, Any, List
import csv
import io

# Import tools
from tools.data_fetcher import set_dataset
from tools.vector_query import query_docs
from tools.anomaly_detection import anomaly_detection
from tools.generateAnomalyCharts import generate_anomalies_summary_with_charts
from tools.genChart import generate_time_series_chart
from webChat import (
    swarm_client, context_variables, client, AGENT_MODEL, load_and_combine_notes,
    query_docs, set_dataset, get_dataset, set_columns, get_data_summary, 
    anomaly_detection, generate_time_series_chart, transfer_to_researcher_agent, 
    generate_chart_message, get_anomaly_details, query_anomalies_db, generate_map
)
from tools.dashboard_metric_tool import get_dashboard_metric
from tools.store_anomalies import get_anomalies, get_anomaly_details as get_anomaly_details_from_db  # Import the new functions

# Configure logging
# log_level = os.getenv("LOG_LEVEL", "INFO") # REMOVED: Configured in main.py
# if log_level.upper() == "DEBUG":
#     log_level = logging.DEBUG
# elif log_level.upper() == "INFO":
#     log_level = logging.INFO
# elif log_level.upper() == "WARNING":
#     log_level = logging.WARNING
# elif log_level.upper() == "ERROR":
#     log_level = logging.ERROR
# elif log_level.upper() == "CRITICAL":
#     log_level = logging.CRITICAL
# else:
#     log_level = logging.INFO

# logging.basicConfig( # REMOVED: Configured in main.py
#     level=log_level,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'anomalyAnalyzer.log')),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Initialize API router
router = APIRouter()

# Templates instance
templates = None

# Set templates function for the main app to use
def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logger.info("Templates set in anomalyAnalyzer router")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize OpenAI client
client = OpenAI()

# Initialize Swarm client
debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
swarm_client = Swarm()

# Set models
EMBEDDING_MODEL = "text-embedding-3-large"
AGENT_MODEL = "gpt-4o"

# Session management
sessions = {}
MAX_HISTORY = 10

# Context variables
context_variables = {
    "dataset": pd.DataFrame(),
    "anomalies": [],
    "explanations": {},
    "current_metric": None,
    "current_period_type": None
}

# Define agent tools
def get_dataset(context_variables, *args, **kwargs):
    """
    Returns the dataset for analysis.
    """
    dataset = context_variables.get("dataset")
    if dataset is None or dataset.empty:
        return {"error": "No dataset loaded. Please use set_dataset first."}
    
    data_info = {
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "data_preview": dataset.head(5).to_dict(orient='records')
    }
    return data_info

def explain_anomaly(context_variables, group_value, group_field, numeric_field, date_field, period_type='month'):
    """
    Tool to explain why an anomaly occurred by analyzing the data from different perspectives.
    """
    dataset = context_variables.get("dataset", pd.DataFrame())
    if dataset.empty:
        return {"error": "No dataset available"}
    
    # Filter dataset to the specific group value
    filtered_data = dataset[dataset[group_field] == group_value]
    
    if filtered_data.empty:
        return {"error": f"No data found for {group_field}={group_value}"}
    
    # Get all available columns except the ones we're already analyzing
    analysis_columns = [col for col in dataset.columns if col not in [group_field, numeric_field, date_field]]
    
    explanations = []
    
    # Analyze each column for potential explanations
    for column in analysis_columns[:5]:  # Limit to first 5 columns to avoid overwhelming
        try:
            # Group by this dimension and see if there are any significant changes
            grouped = filtered_data.groupby([column, date_field])[numeric_field].mean().reset_index()
            # Pivot to have dates as columns
            pivot = grouped.pivot(index=column, columns=date_field, values=numeric_field)
            
            # Compare the first and last period
            if len(pivot.columns) >= 2:
                first_period = pivot.columns[0]
                last_period = pivot.columns[-1]
                
                # Calculate percent change
                pivot['pct_change'] = (pivot[last_period] - pivot[first_period]) / pivot[first_period] * 100
                
                # Find significant changes (over 20%)
                significant_changes = pivot[abs(pivot['pct_change']) > 20].sort_values('pct_change', ascending=False)
                
                if not significant_changes.empty:
                    for idx, row in significant_changes.iterrows():
                        direction = "increased" if row['pct_change'] > 0 else "decreased"
                        explanations.append({
                            "dimension": column,
                            "value": idx,
                            "first_period_value": row[first_period],
                            "last_period_value": row[last_period],
                            "percent_change": row['pct_change'],
                            "explanation": f"The value of {numeric_field} for {group_value} has {direction} by {abs(row['pct_change']):.1f}% when looking at {column}={idx}"
                        })
        except Exception as e:
            logger.error(f"Error analyzing column {column}: {e}")
    
    # Generate summary chart 
    chart_html = generate_time_series_chart(
        filtered_data,
        date_field=date_field,
        numeric_field=numeric_field,
        title=f"Trend for {group_value}",
        y_axis_label=numeric_field
    )
    
    # Store explanations in context
    context_variables["explanations"][group_value] = {
        "explanations": explanations,
        "chart": chart_html["chart_html"]
    }
    
    return {
        "explanations": explanations,
        "chart_html": chart_html["chart_html"]
    }

def get_anomalies(context_variables):
    """
    Tool to retrieve the list of detected anomalies.
    """
    return {"anomalies": context_variables.get("anomalies", [])}

def run_anomaly_detection(context_variables, group_field, numeric_field, date_field, period_type='month', min_diff=2):
    """
    Wrapper tool for anomaly_detection that stores results in context.
    """
    results = anomaly_detection(
        context_variables,
        group_field=group_field,
        numeric_field=numeric_field,
        date_field=date_field,
        period_type=period_type,
        min_diff=min_diff
    )
    
    # Store anomalies in context
    if "anomalies" in results:
        context_variables["anomalies"] = results["anomalies"]
    
    return results

def set_analysis_parameters(context_variables, metric_name, period_type='month'):
    """
    Tool to set current metric and period type for analysis.
    """
    context_variables["current_metric"] = metric_name
    context_variables["current_period_type"] = period_type
    return {"status": "success", "message": f"Set analysis parameters to metric: {metric_name}, period: {period_type}"}

def get_analysis_parameters(context_variables):
    """
    Tool to get current metric and period type.
    """
    return {
        "current_metric": context_variables.get("current_metric"),
        "current_period_type": context_variables.get("current_period_type")
    }

def query_anomalies_db(context_variables, query_type='recent', limit=10, group_filter=None, date_start=None, date_end=None, only_anomalies=True, metric_name=None, district_filter=None):
    """
    Tool to query the anomalies table in the PostgreSQL database.
    
    Args:
        context_variables: The context variables dictionary
        query_type: Type of query to run (recent, by_group, by_date, by_metric)
        limit: Maximum number of records to return (default: 10)
        group_filter: Filter by group value (substring match)
        date_start: Start date for filtering (format: YYYY-MM-DD)
        date_end: End date for filtering (format: YYYY-MM-DD)
        only_anomalies: If True, only return records where out_of_bounds=True
        metric_name: Filter by metric name in metadata
        district_filter: Filter by specific district
    
    Returns:
        Dictionary with query results
    """
    try:
        # Use the imported get_anomalies function
        result = get_anomalies(
            query_type=query_type,
            limit=limit,
            group_filter=group_filter,
            date_start=date_start,
            date_end=date_end,
            only_anomalies=only_anomalies,
            metric_name=metric_name,
            district_filter=district_filter
        )
        
        # If successful, format the results for display
        if result["status"] == "success":
            # Format results for display
            formatted_results = []
            for item in result["results"]:
                # Extract relevant information from metadata
                metadata = item.get("metadata", {})
                object_name = metadata.get("object_name", "Unknown Metric")
                
                # Format the difference for better readability
                difference = item.get("difference", 0)
                if difference > 0:
                    difference_str = f"+{difference:.2f}"
                else:
                    difference_str = f"{difference:.2f}"
                
                # Add formatted item
                formatted_results.append({
                    "id": item.get("id"),
                    "metric": object_name,
                    "group": item.get("group_value", "Unknown"),
                    "recent_mean": round(item.get("recent_mean", 0), 2),
                    "comparison_mean": round(item.get("comparison_mean", 0), 2),
                    "difference": difference_str,
                    "difference_value": difference,  # Used for sorting
                    "std_dev": round(item.get("std_dev", 0), 2),
                    "created_at": item.get("created_at"),
                    "district": item.get("district")
                })
                
            return {
                "status": "success", 
                "count": len(formatted_results),
                "results": formatted_results,
                "query_info": {
                    "query_type": query_type,
                    "filters_applied": {
                        "group_filter": group_filter,
                        "date_start": date_start,
                        "date_end": date_end,
                        "only_anomalies": only_anomalies,
                        "metric_name": metric_name,
                        "district_filter": district_filter
                    }
                }
            }
        else:
            return result  # Return the error result as is
        
    except Exception as e:
        logger.error(f"Error querying anomalies database: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to query anomalies database: {str(e)}"
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
        # Use the imported get_anomaly_details_from_db function
        result = get_anomaly_details_from_db(anomaly_id=anomaly_id)
        
        # If successful, process the results
        if result["status"] == "success":
            item = result["anomaly"]
            
            # Format and structure the returned data
            formatted_details = {
                "id": item.get("id"),
                "group_value": item.get("group_value"),
                "group_field_name": item.get("group_field_name"),
                "field_name": item.get("field_name"),
                "period_type": item.get("period_type"),
                "comparison_mean": item.get("comparison_mean"),
                "recent_mean": item.get("recent_mean"),
                "difference": item.get("difference"),
                "percent_change": round(item.get("difference", 0) / max(0.001, item.get("comparison_mean", 1)) * 100, 2),
                "std_dev": item.get("std_dev"),
                "out_of_bounds": item.get("out_of_bounds"),
                "created_at": item.get("created_at"),
                "metadata": item.get("metadata", {}),
                "recent_date": item.get("recent_date"),
                "comparison_dates": item.get("comparison_dates", []),
                "comparison_counts": item.get("comparison_counts", []),
                "recent_dates": item.get("recent_dates", []),
                "recent_counts": item.get("recent_counts", []),
                "district": item.get("district"),
            }
            
            # Convert date objects to ISO format strings
            if formatted_details["created_at"] and hasattr(formatted_details["created_at"], 'isoformat'):
                formatted_details["created_at"] = formatted_details["created_at"].isoformat()
                
            if formatted_details["recent_date"] and hasattr(formatted_details["recent_date"], 'isoformat'):
                formatted_details["recent_date"] = formatted_details["recent_date"].isoformat()
                
            # Convert dates in comparison_dates and recent_dates
            formatted_comparison_dates = []
            for date in formatted_details["comparison_dates"]:
                if date and hasattr(date, 'isoformat'):
                    formatted_comparison_dates.append(date.isoformat())
                else:
                    formatted_comparison_dates.append(date)
            formatted_details["comparison_dates"] = formatted_comparison_dates
            
            formatted_recent_dates = []
            for date in formatted_details["recent_dates"]:
                if date and hasattr(date, 'isoformat'):
                    formatted_recent_dates.append(date.isoformat())
                else:
                    formatted_recent_dates.append(date)
            formatted_details["recent_dates"] = formatted_recent_dates
            
            # Get the chart data for visualization
            chart_data = {
                "dates": [],
                "values": [],
                "periods": []
            }
            
            # Add comparison period data
            for i in range(len(formatted_details["comparison_dates"])):
                chart_data["dates"].append(formatted_details["comparison_dates"][i])
                chart_data["values"].append(formatted_details["comparison_counts"][i])
                chart_data["periods"].append("comparison")
            
            # Add recent period data
            for i in range(len(formatted_details["recent_dates"])):
                chart_data["dates"].append(formatted_details["recent_dates"][i])
                chart_data["values"].append(formatted_details["recent_counts"][i])
                chart_data["periods"].append("recent")
            
            formatted_details["chart_data"] = chart_data
            
            return {
                "status": "success",
                "anomaly": formatted_details
            }
        else:
            return result  # Return error as is
        
    except Exception as e:
        logger.error(f"Error retrieving anomaly details: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to retrieve anomaly details: {str(e)}"
        }

def get_dataset_columns(context_variables, endpoint=None):
    """
    Tool to get the columns available in a particular dataset endpoint.
    
    Args:
        context_variables: The context variables dictionary
        endpoint: The dataset identifier (e.g., 'ubvf-ztfx')
        
    Returns:
        Dictionary with columns information or error
    """
    import os
    import json
    
    try:
        logger.info(f"Getting columns for endpoint: {endpoint}")
        
        if not endpoint:
            return {"error": "No endpoint provided"}
        
        # Clean the endpoint parameter (remove .json if present)
        endpoint = endpoint.replace('.json', '')
        
        # Look for dataset metadata in data/datasets directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, 'data', 'datasets')
        
        # Check if the directory exists
        if not os.path.exists(datasets_dir):
            return {"error": f"Datasets directory not found: {datasets_dir}"}
        
        # Try to find a metadata file for this endpoint
        metadata_files = [
            os.path.join(datasets_dir, f"{endpoint}.json"),
            os.path.join(datasets_dir, f"{endpoint}_metadata.json"),
            os.path.join(datasets_dir, f"{endpoint}_columns.json")
        ]
        
        metadata_file = None
        for file_path in metadata_files:
            if os.path.exists(file_path):
                metadata_file = file_path
                break
        
        if not metadata_file:
            return {"error": f"No metadata found for endpoint: {endpoint}"}
        
        # Read the metadata file
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract columns information (adapt based on your metadata structure)
        columns = []
        if isinstance(metadata, dict):
            if "columns" in metadata:
                columns = metadata["columns"]
            elif "fields" in metadata:
                columns = metadata["fields"]
        elif isinstance(metadata, list):
            columns = metadata  # Assume it's a list of column objects
        
        # Return simplified column info for agent use
        column_info = []
        for col in columns:
            if isinstance(col, dict):
                info = {
                    "name": col.get("name", col.get("fieldName", "unknown")),
                    "type": col.get("dataTypeName", col.get("type", "unknown")),
                    "description": col.get("description", "")
                }
                column_info.append(info)
        
        return {
            "endpoint": endpoint,
            "column_count": len(column_info),
            "columns": column_info
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset columns: {str(e)}", exc_info=True)
        return {"error": f"Failed to get dataset columns: {str(e)}"}

def explain_metric_change(context_variables, metric_id=None, metric_name=None, district=None, 
                         period_type='month', previous_value=None, recent_value=None, 
                         previous_period=None, recent_period=None, delta=None, percent_change=None):
    """
    Tool to generate an explanation for a specific metric change.
    
    Args:
        context_variables: The context variables dictionary
        metric_id: The ID of the metric
        metric_name: The name of the metric
        district: The district number or 'all'
        period_type: The period type (month, quarter, year)
        previous_value: The value from the previous period
        recent_value: The value from the most recent period
        previous_period: The previous period date
        recent_period: The most recent period date
        delta: The absolute change
        percent_change: The percentage change
        
    Returns:
        Dictionary with explanation and related data
    """
    logger.info(f"""
=== Explaining Metric Change ===
Metric ID: {metric_id}
Metric Name: {metric_name}
District: {district}
Period Type: {period_type}
Change: {previous_value} → {recent_value} (Δ{delta}, {percent_change}%)
Periods: {previous_period} → {recent_period}
""")
    
    try:
        # If metric_id is available, try to get dashboard metric data
        dashboard_data = None
        if metric_id and district is not None:
            try:
                district_num = 0  # Default to citywide
                if district != 'all':
                    district_num = int(district)
                
                # Get metric data from dashboard
                dashboard_data = get_dashboard_metric(context_variables, 
                                                     district_number=district_num, 
                                                     metric_id=metric_id)
                
                logger.info(f"Got dashboard data: {str(dashboard_data)[:200]}...")
            except Exception as e:
                logger.error(f"Error getting dashboard metric: {str(e)}")
        
        # Check for relevant documentation from Qdrant
        docs_results = None
        if metric_name:
            try:
                # Search for documentation about this metric
                docs_results = query_docs(
                    context_variables, 
                    collection_name="SFPublicData", 
                    query=f"metrics {metric_name} analysis explanation {period_type}"
                )
                logger.info(f"Got {len(docs_results.get('results', []))} documentation results")
            except Exception as e:
                logger.error(f"Error querying docs: {str(e)}")
        
        # Try to get anomaly data for this metric
        anomaly_results = None
        try:
            anomaly_query = None
            if metric_id:
                anomaly_query = query_anomalies_db(
                    context_variables,
                    query_type='by_metric',
                    limit=5,
                    metric_name=metric_name,
                    district_filter=district if district != 'all' else None,
                    only_anomalies=True
                )
            
            if anomaly_query and anomaly_query.get("results"):
                # Found some anomalies related to this metric
                logger.info(f"Found {len(anomaly_query['results'])} related anomalies")
                anomaly_results = anomaly_query
            else:
                logger.info("No related anomalies found in database")
        except Exception as e:
            logger.error(f"Error querying anomalies database: {str(e)}")
        
        # Generate explanation based on available data
        direction = "increased" if delta and float(delta) > 0 else "decreased"
        pct_str = f"{abs(float(percent_change)):.1f}%" if percent_change is not None else "an unknown percentage"
        
        explanation = {
            "metric_info": {
                "name": metric_name,
                "id": metric_id,
                "district": district,
                "period_type": period_type
            },
            "change_data": {
                "previous_value": previous_value,
                "recent_value": recent_value,
                "delta": delta,
                "percent_change": percent_change,
                "previous_period": previous_period,
                "recent_period": recent_period,
                "direction": direction
            },
            "summary": f"The value of {metric_name} {direction} by {pct_str} from {previous_period} to {recent_period}.",
            "dashboard_data": dashboard_data,
            "documentation": docs_results,
            "anomalies": anomaly_results
        }
        
        # Generate chart for the trend if we have dashboard data
        if dashboard_data and isinstance(dashboard_data, dict) and "data" in dashboard_data:
            try:
                # TODO: Generate chart based on available data
                # This will depend on the structure of your dashboard data
                logger.info("Could generate chart here based on dashboard data")
            except Exception as e:
                logger.error(f"Error generating chart: {str(e)}")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error explaining metric change: {str(e)}", exc_info=True)
        return {"error": f"Failed to explain metric change: {str(e)}"}

# Define the explainer agent directly like in webChat.py
ANOMALY_EXPLAINER_INSTRUCTIONS = """You are an anomaly explanation agent that specializes in providing deep insights into detected anomalies.

IMPORTANT: You MUST use tools to gather data BEFORE responding. Direct explanations without tool usage are NOT acceptable.

Your task is to:
1. Take an anomaly that has already been identified in dashboard metrics
2. Research and explain why this anomaly might have occurred
3. Analyze the anomaly from multiple perspectives and dimensions
4. Provide clear, comprehensive explanations with supporting evidence
5. Generate visualizations that help illustrate your findings

MANDATORY WORKFLOW (follow this exact sequence):
1. FIRST, use query_docs to find information about the metric in question
2. SECOND, if a metric_id is provided, use get_dashboard_metric to retrieve the actual metric data
3. THIRD, use query_anomalies_db to find related anomalies
4. FOURTH, if the metric is related to a dataset, use get_dataset_columns to understand available data fields
5. FINALLY, use explain_metric_change to compile all the gathered data into an explanation

DO NOT skip these steps. You MUST use at least 3 tools before providing your final response.

TOOLS YOU SHOULD USE:
- get_dashboard_metric: Retrieve dashboard metric data containing anomalies
  USAGE: get_dashboard_metric(context_variables, district_number=0, metric_id="metric_name")
  Use this to get the dashboard metric that contains the anomaly the user wants explained.

- query_anomalies_db: Query anomalies directly from the PostgreSQL database
  USAGE: query_anomalies_db(context_variables, query_type='recent', limit=10, group_filter=None, date_start=None, date_end=None, only_anomalies=True, metric_name=None)
  Use this to find specific anomalies in the database to explain. Search by various criteria like date range or metric name.

- get_anomaly_details: Get detailed information about a specific anomaly by ID
  USAGE: get_anomaly_details(context_variables, anomaly_id=123)
  IMPORTANT: Always pass the anomaly ID using the **anomaly_id=123** keyword. Do NOT pass it positionally or inside an "args" object.
  Use this to get complete information about a specific anomaly, including its time series data and metadata.

- get_dataset: Get information about any dataset that's been loaded
  USAGE: get_dataset(context_variables)
  Use this to see what data is available for further analysis.

- set_dataset: Load a dataset for analysis
  USAGE: set_dataset(context_variables, endpoint="dataset-id", query="your-soql-query")
  Use this to load data for further analysis when needed.

- get_dataset_columns: Get column information for a dataset endpoint
  USAGE: get_dataset_columns(context_variables, endpoint="dataset-id")
  Use this to explore what columns are available in a specific dataset.

- explain_anomaly: Analyze why an anomaly occurred from different perspectives
  USAGE: explain_anomaly(context_variables, group_value="specific_value", group_field="category_column", numeric_field="value_column", date_field="date_column")
  This is your main tool - use it to provide multi-dimensional analysis of anomalies.

- explain_metric_change: Generate an explanation for a specific metric change
  USAGE: explain_metric_change(context_variables, metric_id=metric_id, metric_name=metric_name, district=district, period_type=period_type, previous_value=previous_value, recent_value=recent_value, previous_period=previous_period, recent_period=recent_period, delta=delta, percent_change=percent_change)
  Use this to explain why a specific metric changed between two periods.

- get_anomalies: Get the list of anomalies detected in dashboard metrics
  USAGE: get_anomalies(context_variables)
  Use this to see what anomalies are available to explain.

- query_docs: Search for additional context in documentation
  USAGE: query_docs(context_variables, collection_name="SFPublicData", query="information related to [specific anomaly]")
  Use this to find domain-specific information that might explain the anomaly.

- generate_time_series_chart: Generate visualizations to help explain the anomaly
  USAGE: generate_time_series_chart(data, date_field="date_column", numeric_field="value_column", title="Chart Title")
  Use this for additional visualizations that help explain your findings.

When explaining an anomaly or metric change:
- Consider multiple possible explanations (seasonal patterns, policy changes, economic factors, etc.)
- Compare the anomaly to historical trends and similar metrics
- Quantify the magnitude and significance of the anomaly
- Avoid speculation - stick to what the data and documentation show
- Always include time-series charts to visualize the anomaly
"""

# Create the explainer agent directly (similar to how Researcher_agent is created in webChat.py)
anomaly_explainer_agent = Agent(
    model=AGENT_MODEL,
    name="Explainer",
    instructions=ANOMALY_EXPLAINER_INSTRUCTIONS,
    functions=[
        get_dataset,
        set_dataset,
        explain_anomaly,
        get_anomalies,
        query_docs,
        generate_time_series_chart,
        get_dashboard_metric,
        query_anomalies_db,
        get_anomaly_details,
        get_dataset_columns,
        explain_metric_change
    ],
    context_variables=context_variables,
    debug=debug_mode
)

# Now update the function mapping to refer directly to the functions
function_mapping = {
    'get_dataset': get_dataset,
    'set_dataset': set_dataset,
    'query_docs': query_docs,
    'run_anomaly_detection': run_anomaly_detection,
    'explain_anomaly': explain_anomaly,
    'get_anomalies': get_anomalies,
    'set_analysis_parameters': set_analysis_parameters,
    'get_analysis_parameters': get_analysis_parameters,
    'generate_time_series_chart': generate_time_series_chart,
    'get_dashboard_metric': get_dashboard_metric,
    'query_anomalies_db': query_anomalies_db,
    'get_anomaly_details': get_anomaly_details,
    'get_dataset_columns': get_dataset_columns,
    'explain_metric_change': explain_metric_change
}

# Add logging wrappers around each function
def add_logging_wrappers():
    """Add logging wrappers around each function in the function_mapping."""
    def create_logging_wrapper(f, name):
        def wrapper(*args, **kwargs):
            logger.info(f"BEGIN FUNCTION CALL: {name}")
            try:
                # Log arguments (truncated)
                arg_str = str(args)[:100] + ("..." if len(str(args)) > 100 else "")
                kwarg_str = str(kwargs)[:100] + ("..." if len(str(kwargs)) > 100 else "")
                logger.info(f"Args: {arg_str}, Kwargs: {kwarg_str}")
                
                # Call the original function
                result = f(*args, **kwargs)
                
                # Log result (truncated)
                result_str = str(result)[:100] + ("..." if len(str(result)) > 100 else "")
                logger.info(f"END FUNCTION CALL: {name} - Result: {result_str}")
                
                return result
            except Exception as e:
                logger.error(f"ERROR IN FUNCTION: {name} - {str(e)}")
                raise
        return wrapper
    
    # Replace each function with its wrapped version
    for func_name, func in list(function_mapping.items()):  # Use list() to create a copy
        function_mapping[func_name] = create_logging_wrapper(func, func_name)

# Apply the logging wrappers
add_logging_wrappers()

# Helper functions for response handling
def process_and_print_streaming_response(response):
    """Process streaming response to output progress in real-time."""
    full_response = ""
    for chunk in response:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            chunk_content = chunk.choices[0].delta.content or ""
            full_response += chunk_content
            print(chunk_content, end="", flush=True)
    print("\n")  # End with a newline
    return full_response

def truncate_messages(messages):
    """Truncate message history to keep within token limits."""
    if len(messages) <= MAX_HISTORY:
        return messages
    
    # Keep the first system message
    system_message = next((m for m in messages if m["role"] == "system"), None)
    
    # Get the most recent messages, excluding the system message
    recent_messages = [m for m in messages if m["role"] != "system"][-MAX_HISTORY:]
    
    # Put the system message back at the beginning if it exists
    if system_message:
        return [system_message] + recent_messages
    
    return recent_messages

async def generate_response(user_input, session_data):
    """Generate a streaming response for the anomaly analyzer agent."""
    logger.info(f"""
=== Starting Anomaly Analyzer Response ===
Session ID: {id(session_data)}
Current Agent: {session_data['agent'].name}
Timestamp: {datetime.datetime.now().isoformat()}
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
            debug=debug_mode,
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
                                # Call the function with the context variables and arguments
                                logger.info(f"Executing tool: {current_function_name}")
                                result = function_to_call(context_variables, **arguments_json)
                                
                                logger.info(f"""
=== Tool Result ===
Function: {current_function_name}
Result: {str(result)[:500]}{'...' if len(str(result)) > 500 else ''}
""")
                                
                                # Format the result for display
                                if isinstance(result, dict):
                                    result_message = {
                                        "type": "tool_result",
                                        "sender": assistant_message["sender"],
                                        "function_name": current_function_name,
                                        "result": result
                                    }
                                else:
                                    result_message = {
                                        "type": "tool_result",
                                        "sender": assistant_message["sender"],
                                        "function_name": current_function_name,
                                        "result": str(result)
                                    }
                                
                                yield json.dumps(result_message) + "\n"
                            except Exception as e:
                                logger.error(f"""
=== Tool Error ===
Function: {current_function_name}
Error: {str(e)}
""")
                                error_message = {
                                    "type": "error",
                                    "sender": assistant_message["sender"],
                                    "function_name": current_function_name,
                                    "error": str(e)
                                }
                                yield json.dumps(error_message) + "\n"

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
Timestamp: {datetime.datetime.now().isoformat()}
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
Timestamp: {datetime.datetime.now().isoformat()}
===============================
""")
        error_message = {
            "type": "error",
            "sender": "System",
            "error": f"I encountered an error while processing your request: {str(e)}"
        }
        yield json.dumps(error_message) + "\n"

    logger.info(f"""
=== Chat Interaction Summary ===
Total messages: {len(messages)}
Timestamp: {datetime.datetime.now().isoformat()}
===============================
""")

# API Routes
@router.post("/api/anomaly-chat")
async def anomaly_chat(request: Request):
    """API endpoint for the anomaly chat interface."""
    try:
        data = await request.json()
        user_input = data.get("message", "")
        session_id = data.get("session_id", "default")
        
        logger.info(f"Received chat request - Session: {session_id}, Input: {user_input}")
        
        # Initialize or get session data
        if session_id not in sessions:
            logger.info(f"Creating new session: {session_id}")
            sessions[session_id] = {
                "messages": [],
                "agent": anomaly_explainer_agent,
                "context_variables": context_variables.copy()
            }
        
        session_data = sessions[session_id]
        
        # Create StreamingResponse
        response = StreamingResponse(
            generate_response(user_input, session_data),
            media_type="text/plain"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in anomaly_chat: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@router.get("/")
async def anomaly_analyzer_page(request: Request):
    """Serve the anomaly analyzer interface."""
    from fastapi.templating import Jinja2Templates
    
    logger.info("Anomaly analyzer page requested")
    logger.info(f"Templates instance: {templates}")
    
    # Use the templates instance set by main.py
    if templates is None:
        logger.error("Templates not initialized in anomalyAnalyzer router")
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        logger.info(f"Using local templates from: {templates_dir}")
        templates_local = Jinja2Templates(directory=templates_dir)
        return templates_local.TemplateResponse("anomaly_analyzer.html", {"request": request})
    
    logger.info("Using global templates instance")
    return templates.TemplateResponse("anomaly_analyzer.html", {"request": request})

@router.post("/api/reset-anomaly-chat")
async def reset_anomaly_chat(request: Request):
    """Reset the anomaly chat conversation."""
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")
        
        if session_id in sessions:
            sessions[session_id] = {
                "messages": [],
                "agent": anomaly_explainer_agent,
                "context_variables": context_variables.copy()
            }
        
        return JSONResponse(content={"status": "success", "message": "Conversation reset successfully"})
    
    except Exception as e:
        logger.error(f"Error in reset_anomaly_chat: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@router.get("/api/top-metric-changes")
async def get_top_metric_changes(
    period_type: str = "month",
    limit: int = 10,
    object_id: str = None,   # Now this is optional
    district: str = "0",    # Changed from int to str to support 'all'
    show_both: bool = True,
    show_on_dash: bool = False  # Add show_on_dash parameter
):
    """
    Get metrics with the greatest changes over the specified period.
    
    Args:
        period_type: Type of period to compare ('month', 'quarter', 'year')
        limit: Maximum number of records to return for each direction (top/bottom)
        object_id: Optional - Filter by specific object_id 
        district: Filter by specific district (default: 0)
        show_both: If True, return both top and bottom changes; if False, return only top changes
        show_on_dash: If True, only return metrics that are shown on dashboard
        
    Returns:
        JSON with metrics sorted by delta (highest to lowest)
    """
    try:
        import psycopg2
        import psycopg2.extras
        from datetime import datetime, date
        import calendar
        
        logger.info(f"get_top_metric_changes called with: period_type={period_type}, limit={limit}, object_id={object_id}, district={district}, show_on_dash={show_on_dash}")
        
        # Calculate report date based on period_type
        today = date.today()
        if period_type == 'week':
            # For weekly, we need to find the most recent week that has data available
            # First, let's get the latest available week from the database
            # We'll calculate this after connecting to the database
            report_date = None  # Will be calculated after DB connection
        elif period_type == 'month':
            # For monthly, report date is previous month
            if today.month == 1:
                report_date = date(today.year - 1, 12, 1)
            else:
                report_date = date(today.year, today.month - 1, 1)
        elif period_type == 'quarter':
            # For quarterly, report date is previous quarter
            current_quarter = (today.month - 1) // 3 + 1
            if current_quarter == 1:
                report_date = date(today.year - 1, 10, 1)  # Q4 of previous year
            else:
                report_date = date(today.year, (current_quarter - 1) * 3 - 2, 1)
        else:  # year
            # For annual, report date is previous year
            report_date = date(today.year - 1, 1, 1)
            
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "transparentsf"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # For weekly data, calculate the report date based on available data
        if period_type == 'week' and report_date is None:
            # Get the latest available week from the database
            cursor.execute("""
                SELECT MAX(tsd.time_period) as latest_week
                FROM time_series_data tsd
                JOIN time_series_metadata tsm ON tsd.chart_id = tsm.chart_id
                WHERE tsm.period_type = 'week'
                AND tsm.is_active = TRUE
            """)
            latest_week_result = cursor.fetchone()
            if latest_week_result and latest_week_result['latest_week']:
                # Use the latest available week as the report date
                report_date = latest_week_result['latest_week']
                logger.info(f"Using latest available week as report date: {report_date}")
            else:
                # Fallback to calculating previous week
                days_since_monday = today.weekday()
                if days_since_monday == 0:  # Today is Monday
                    report_date = today - timedelta(days=7)
                else:
                    report_date = today - timedelta(days=days_since_monday + 7)
                logger.info(f"Fallback report date calculated as: {report_date}")
        else:
            logger.info(f"Report date calculated as: {report_date}")
        
        # Create cursor with dictionary-like results
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build query conditions based on provided parameters
        where_conditions = []
        query_params = []
        
        # Always filter by period_type and district
        where_conditions.append("tsm.period_type = %s")
        query_params.append(period_type)
        
        # Only add district filter if not 'all'
        if district != 'all':
            where_conditions.append("tsm.district = %s")
            query_params.append(district)
        else:
            logger.info("No district filter applied - showing metrics from all districts")
        
        # Only add object_id filter if it's provided
        if object_id:
            where_conditions.append("tsm.object_id = %s")
            query_params.append(object_id)
        
        # Filter for only charts where group_field is null
        where_conditions.append("tsm.group_field IS NULL")
        
        # Filter for only active records
        where_conditions.append("tsm.is_active = TRUE")
        
        # Add show_on_dash filter if requested
        if show_on_dash:
            where_conditions.append("m.show_on_dash = TRUE")
        
        # Combine WHERE conditions
        where_clause = " AND ".join(where_conditions)
        
        # Get the chart_ids for the specified filters, including their most recent time_period
        # Now join with metrics table to check show_on_dash
        chart_query = f"""
        WITH latest_periods AS (
            SELECT 
                tsm.chart_id,
                tsm.object_name,
                tsm.group_field,
                tsm.object_id,
                MAX(tsd.time_period) as latest_period
            FROM time_series_metadata tsm
            JOIN time_series_data tsd ON tsm.chart_id = tsd.chart_id
            LEFT JOIN metrics m ON tsm.object_id::integer = m.id
            WHERE {where_clause}
            GROUP BY tsm.chart_id, tsm.object_name, tsm.group_field, tsm.object_id
        )
        SELECT 
            chart_id,
            object_name,
            group_field,
            object_id,
            latest_period
        FROM latest_periods
        WHERE latest_period >= %s
        """
        query_params.append(report_date)
        
        logger.info(f"Chart query: {chart_query} with params: {query_params}")
        cursor.execute(chart_query, query_params)
        charts = cursor.fetchall()
        
        if not charts:
            filter_desc = f"period_type={period_type}, district={district}, group_field=NULL, report_date={report_date}, show_on_dash={show_on_dash}"
            if object_id:
                filter_desc += f", object_id={object_id}"
                
            logger.warning(f"No charts found matching filters: {filter_desc}")
            
            # Let's check if there are any charts at all, regardless of filters
            cursor.execute("SELECT COUNT(*) AS count FROM time_series_metadata")
            total_count = cursor.fetchone()['count']
            logger.info(f"Total charts in time_series_metadata: {total_count}")
            
            # Check if any charts match just the period_type
            cursor.execute("SELECT COUNT(*) AS count FROM time_series_metadata WHERE period_type = %s", [period_type])
            period_count = cursor.fetchone()['count']
            logger.info(f"Charts with period_type={period_type}: {period_count}")
            
            # Check if any charts have group_field IS NULL
            cursor.execute("SELECT COUNT(*) AS count FROM time_series_metadata WHERE group_field IS NULL")
            null_group_count = cursor.fetchone()['count']
            logger.info(f"Charts with group_field IS NULL: {null_group_count}")
            
            # Check show_on_dash if filtering by it
            if show_on_dash:
                cursor.execute("""
                    SELECT COUNT(*) AS count 
                    FROM time_series_metadata tsm 
                    LEFT JOIN metrics m ON tsm.object_id = m.id 
                    WHERE m.show_on_dash = TRUE
                """)
                show_on_dash_count = cursor.fetchone()['count']
                logger.info(f"Charts with show_on_dash=TRUE: {show_on_dash_count}")
            
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"No charts found for {filter_desc}"
                }
            )
        
        # For each chart, find the two most recent periods and compare
        all_results = []
        
        logger.info(f"Found {len(charts)} charts with null group_field and valid report date")
        
        for chart in charts:
            chart_id = chart['chart_id']
            
            # Get the two most recent periods for this chart
            periods_query = """
            SELECT DISTINCT time_period
            FROM time_series_data
            WHERE chart_id = %s
            ORDER BY time_period DESC
            LIMIT 2
            """
            cursor.execute(periods_query, [chart_id])
            periods = cursor.fetchall()
            
            if len(periods) < 2:
                logger.info(f"Skipping chart_id {chart_id} - not enough time periods (found {len(periods)})")
                continue
                
            latest_period = periods[0]['time_period']
            previous_period = periods[1]['time_period']
            
            # Skip if latest period is before report date
            if latest_period < report_date:
                logger.info(f"Skipping chart_id {chart_id} - latest period {latest_period} is before report date {report_date}")
                continue
            
            logger.info(f"Processing chart_id {chart_id} - comparing periods {latest_period} and {previous_period}")
            
            # Check for stale data by comparing most_recent_data_date with recent_period.end
            stale_data_warning = None
            if chart['object_id'] and chart['object_id'] != 'unknown':
                try:
                    # Get the most_recent_data_date from metrics table
                    cursor.execute("""
                        SELECT most_recent_data_date 
                        FROM metrics 
                        WHERE id = %s
                        LIMIT 1
                    """, [chart['object_id']])
                    metric_result = cursor.fetchone()
                    
                    if metric_result and metric_result['most_recent_data_date']:
                        # Get the recent_period.end from time_series_metadata metadata column
                        cursor.execute("""
                            SELECT metadata 
                            FROM time_series_metadata 
                            WHERE chart_id = %s
                            LIMIT 1
                        """, [chart_id])
                        metadata_result = cursor.fetchone()
                        
                        if metadata_result and metadata_result['metadata']:
                            metadata = metadata_result['metadata']
                            if isinstance(metadata, str):
                                import json
                                try:
                                    metadata = json.loads(metadata)
                                except json.JSONDecodeError:
                                    metadata = {}
                            
                            # Look for the end date in filter_conditions (the condition with operator "<=")
                            filter_conditions = metadata.get('filter_conditions', [])
                            expected_end_date = None
                            
                            for condition in filter_conditions:
                                if condition.get('operator') == '<=' and condition.get('is_date', False):
                                    end_date_str = condition.get('value')
                                    if end_date_str:
                                        try:
                                            if 'T' in end_date_str:
                                                expected_end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                                            else:
                                                expected_end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                                            break
                                        except ValueError:
                                            try:
                                                expected_end_date = datetime.strptime(end_date_str, '%Y-%m').date()
                                                break
                                            except ValueError:
                                                continue
                            
                            # Compare dates
                            if expected_end_date and metric_result['most_recent_data_date'] < expected_end_date:
                                days_stale = (expected_end_date - metric_result['most_recent_data_date']).days
                                stale_data_warning = f"Data is {days_stale} days stale (most recent: {metric_result['most_recent_data_date']}, expected: {expected_end_date})"
                                logger.warning(f"Stale data detected for chart {chart_id} ({chart['object_name']}): {stale_data_warning}")
                                    
                except Exception as e:
                    logger.warning(f"Error checking stale data for chart {chart_id}: {e}")
            
            # Get the data for both periods and join them
            data_query = """
            WITH latest AS (
                SELECT group_value, numeric_value AS recent_value, time_period AS recent_period
                FROM time_series_data
                WHERE chart_id = %s AND time_period = %s
                LIMIT 1
            ),
            previous AS (
                SELECT group_value, numeric_value AS previous_value, time_period AS previous_period
                FROM time_series_data
                WHERE chart_id = %s AND time_period = %s
                LIMIT 1
            ),
            combined AS (
                SELECT
                    (SELECT group_value FROM latest) AS group_value,
                    (SELECT recent_value FROM latest) AS recent_value,
                    (SELECT previous_value FROM previous) AS previous_value,
                    (SELECT recent_period FROM latest) AS recent_period,
                    (SELECT previous_period FROM previous) AS previous_period,
                    ((SELECT recent_value FROM latest) - (SELECT previous_value FROM previous)) AS delta,
                    ABS((SELECT recent_value FROM latest) - (SELECT previous_value FROM previous)) AS abs_delta
            )
            SELECT
                group_value,
                recent_value,
                previous_value,
                delta,
                abs_delta,
                recent_period,
                previous_period
            FROM combined
            WHERE recent_value IS NOT NULL AND previous_value IS NOT NULL
            """
            
            cursor.execute(data_query, [chart_id, latest_period, chart_id, previous_period])
            data_results = cursor.fetchall()
            
            if not data_results:
                logger.info(f"No comparison data found for chart_id {chart_id} between periods {latest_period} and {previous_period}")
                continue
                
            logger.info(f"Found {len(data_results)} comparison rows for chart_id {chart_id}")
            
            # Add metadata to each result
            for result in data_results:
                result['chart_id'] = chart_id
                result['object_id'] = chart['object_id']
                result['object_name'] = chart['object_name']
                result['group_field'] = chart['group_field']
                result['report_date'] = report_date.isoformat()
                
                # Log the raw data for debugging
                logger.info(f"Raw data for {chart['object_name']}: recent={result['recent_value']} ({result['recent_period']}), previous={result['previous_value']} ({result['previous_period']}), delta={result['delta']}")
                
                # Use the district from the data if available, otherwise use the selected district
                if 'district' in result and result['district'] is not None:
                    # District is already set in the result
                    pass
                elif district == 'all':
                    # If showing all districts, get the district from the metadata if available
                    cursor.execute("SELECT district FROM time_series_metadata WHERE chart_id = %s", [chart_id])
                    chart_district = cursor.fetchone()
                    if chart_district and chart_district['district'] is not None:
                        result['district'] = chart_district['district']
                    else:
                        result['district'] = 'N/A'
                else:
                    # Use the selected district
                    result['district'] = district
                
                # Calculate percent change
                if result['previous_value'] and float(result['previous_value']) != 0:
                    result['percent_change'] = (float(result['delta']) / float(result['previous_value'])) * 100
                else:
                    result['percent_change'] = None
                
                # Get greendirection from metrics table based on object_id
                result['greendirection'] = None
                if chart['object_id'] and chart['object_id'] != 'unknown':
                    try:
                        # Try to convert to integer to validate
                        object_id_int = int(chart['object_id'])
                        greendirection_query = """
                        SELECT greendirection 
                        FROM metrics 
                        WHERE id = %s
                        LIMIT 1
                        """
                        cursor.execute(greendirection_query, [object_id_int])
                        greendirection_result = cursor.fetchone()
                        
                        if greendirection_result:
                            result['greendirection'] = greendirection_result['greendirection']
                        else:
                            logger.warning(f"No greendirection found for object_id {chart['object_id']}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid object_id format: {chart['object_id']}, skipping greendirection query")
                else:
                    logger.warning(f"Skipping greendirection query for object_id: {chart['object_id']}")
                
                # Get citywide changes data if available
                citywide_changes = ""
                if chart['object_id'] and chart['object_id'] != 'unknown':
                    try:
                        # Convert to integer for validation, but pass as string to match column type
                        object_id_int = int(chart['object_id'])
                        cursor.execute(
                            "SELECT location_data FROM maps WHERE active = true AND metric_id = %s AND metadata->>'has_change_data' = 'true'",
                            [chart['object_id']]  # Pass as string, not integer
                        )
                        map_data = cursor.fetchone()
                        if map_data and map_data["location_data"]:
                            location_data = map_data["location_data"]
                            if location_data.get("type") == "csv" and location_data.get("csv_data"):
                                # Parse CSV data into a more readable format
                                csv_data = location_data["csv_data"]
                                csv_reader = csv.DictReader(io.StringIO(csv_data))
                                district_changes = []
                                for row in csv_reader:
                                    district_changes.append(
                                        f"District {row['district']}: {row['current_value']} (change: {row['delta']}, {float(row['percent_change'])*100:.1f}%)"
                                    )
                                citywide_changes = "\n".join(district_changes)
                                logger.info(f"Found district data for metric_id {chart['object_id']}:\n{citywide_changes}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing citywide changes data: {e}")
                result['citywide_changes'] = citywide_changes
                
                # Add stale data warning if detected
                if stale_data_warning:
                    result['stale_data_warning'] = stale_data_warning
                    logger.warning(f"Adding stale data warning to result: {stale_data_warning}")
                    
                # Log the values for debugging
                logger.info(f"Result for chart {chart_id}: {result['object_name']} - Previous: {result['previous_value']} ({result['previous_period']}), Recent: {result['recent_value']} ({result['recent_period']}), Delta: {result['delta']}, greendirection: {result['greendirection']}, citywide_changes: {citywide_changes}")
                all_results.append(result)
        
        # Sort all results by delta (descending and ascending)
        all_results = [r for r in all_results if r['delta'] is not None]
        logger.info(f"Total valid results after filtering: {len(all_results)}")
        
        if not all_results:
            logger.warning("No valid comparison results found after processing all charts")
            return JSONResponse(
                content={
                    "status": "success",
                    "count": 0,
                    "period_type": period_type,
                    "object_id": object_id,
                    "district": district,
                    "report_date": report_date.isoformat(),
                    "stale_data_warnings": 0,
                    "positive_changes": [],
                    "negative_changes": []
                }
            )
        
        # Log original data before sorting to verify
        logger.info("Unsorted results sample:")
        for i, r in enumerate(all_results[:3]):
            logger.info(f"Result {i}: {r.get('object_name')} - Delta: {r.get('delta')}, Percent Change: {r.get('percent_change')}")
        
        # Categorize changes based on greendirection and percent_change
        positive_changes = []
        negative_changes = []
        
        for r in all_results:
            if r['percent_change'] is None or r['greendirection'] is None:
                continue
                
            percent_change = float(r['percent_change'])
            greendirection = r['greendirection'].lower() if r['greendirection'] else None
            
            # Determine if this is a positive or negative change based on greendirection
            if greendirection == 'up':
                # For metrics where up is good (greendirection=up)
                if percent_change > 0:
                    positive_changes.append(r)  # Increase is positive
                elif percent_change < 0:
                    negative_changes.append(r)  # Decrease is negative
            elif greendirection == 'down':
                # For metrics where down is good (greendirection=down)
                if percent_change > 0:
                    negative_changes.append(r)  # Increase is negative
                elif percent_change < 0:
                    positive_changes.append(r)  # Decrease is positive
            else:
                # If greendirection is not 'up' or 'down', use traditional logic
                if percent_change > 0:
                    positive_changes.append(r)
                elif percent_change < 0:
                    negative_changes.append(r)
        
        logger.info(f"Split results: {len(positive_changes)} positive changes, {len(negative_changes)} negative changes")
        
        # Sort by absolute percent change (highest impact first)
        positive_changes = sorted(positive_changes, 
                                key=lambda x: abs(float(x['percent_change'])), 
                                reverse=True)[:limit]
        
        negative_changes = sorted(negative_changes, 
                                key=lambda x: abs(float(x['percent_change'])), 
                                reverse=True)[:limit]
        
        # Log positive/negative results to verify categorization
        logger.info("Positive changes after categorization:")
        for i, r in enumerate(positive_changes[:3]):
            logger.info(f"Positive {i}: {r.get('object_name')} - Percent Change: {r.get('percent_change')}%, greendirection: {r.get('greendirection')}")
            
        logger.info("Negative changes after categorization:")
        for i, r in enumerate(negative_changes[:3]):
            logger.info(f"Negative {i}: {r.get('object_name')} - Percent Change: {r.get('percent_change')}%, greendirection: {r.get('greendirection')}")
        
        logger.info(f"Found {len(positive_changes)} positive changes and {len(negative_changes)} negative changes")
        
        # Process results to ensure JSON compliance
        def process_results(results):
            formatted = []
            for item in results:
                # Handle date objects
                for key, value in item.items():
                    if hasattr(value, 'isoformat'):  # This catches date, datetime, etc.
                        item[key] = value.isoformat()
                
                # Ensure citywide_changes is included
                if 'citywide_changes' not in item:
                    item['citywide_changes'] = ""
                
                # Ensure stale_data_warning is included
                if 'stale_data_warning' not in item:
                    item['stale_data_warning'] = None
                
                formatted.append(item)
            return formatted
            
        # Format the results
        positive_formatted = process_results(positive_changes)
        negative_formatted = process_results(negative_changes)
        
        # Count stale data warnings
        stale_data_count = 0
        for result in positive_formatted + negative_formatted:
            if result.get('stale_data_warning'):
                stale_data_count += 1
        
        cursor.close()
        conn.close()
        
        return JSONResponse(
            content={
                "status": "success",
                "count": len(positive_formatted) + len(negative_formatted),
                "period_type": period_type,
                "object_id": object_id,
                "district": district,
                "report_date": report_date.isoformat(),
                "stale_data_warnings": stale_data_count,
                "positive_changes": positive_formatted,
                "negative_changes": negative_formatted
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting top metric changes: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to get top metric changes: {str(e)}"}
        )

@router.post("/api/explain-metric-change")
async def explain_metric_change_endpoint(request: Request):
    """API endpoint to explain a specific metric change."""
    try:
        data = await request.json()
        logger.info(f"Received metric change explanation request: {data}")
        
        # Create a new session for this explanation
        session_id = f"explain_{str(uuid.uuid4())}"
        logger.info(f"Created new session: {session_id}")
        
        # Initialize session with the explainer agent
        sessions[session_id] = {
            "messages": [],
            "agent": anomaly_explainer_agent,
            "context_variables": context_variables.copy()
        }
        
        # Get session data
        session_data = sessions[session_id]
        
        # Add a system message to explicitly require tool usage
        system_message = {
            "role": "system", 
            "content": """CRITICAL INSTRUCTION: You MUST use the tools provided to analyze data before responding. 
            
1. You CANNOT respond with a direct answer or explanation without first making tool calls.
2. If you try to respond without making tool calls, your response will be rejected.
3. The correct workflow is: make tool calls first, then analyze the data, then provide an explanation.
4. At minimum, you must use the query_docs, get_dashboard_metric, and explain_metric_change tools.
5. The user cannot see your tool calls, only your final response, so you need to include your analysis and findings.

This requirement is non-negotiable and is essential for providing accurate information."""
        }
        
        # Store the metric data in context variables
        session_data["context_variables"]["current_metric_change"] = data
        logger.info(f"Stored metric data in context variables: {list(session_data['context_variables'].keys())}")
        
        # Create a prompt for the agent
        metric_name = data.get("metric_name", "unknown metric")
        metric_id = data.get("metric_id", "")
        district = data.get("district", "citywide")
        if district == "0":
            district = "citywide"
            district_number = 0
        elif district == "all":
            district = "all districts"
            district_number = None
        else:
            district_number = int(district)
        
        period_type = data.get("period_type", "month")
        previous_period = data.get("previous_period", "previous period")
        recent_period = data.get("recent_period", "recent period")
        previous_value = data.get("previous_value", 0)
        recent_value = data.get("recent_value", 0)
        delta = data.get("delta", 0)
        percent_change = data.get("percent_change", 0)
        
        if percent_change is not None and isinstance(percent_change, (int, float)):
            percent_change_str = f"{percent_change:.2f}%"
        else:
            percent_change_str = "unknown percentage"
        
        direction = "increased" if float(data.get("delta", 0)) > 0 else "decreased"
        
        # Create a more structured prompt that explicitly lists the tools to use
        prompt = f"""I need you to explain a significant change in the metric '{metric_name}' (ID: {metric_id}) for {district}.

METRIC DETAILS:
- Name: {metric_name}
- ID: {metric_id}
- District: {district} (district_number={district_number})
- Period Type: {period_type}
- Previous Period: {previous_period} - Value: {previous_value}
- Recent Period: {recent_period} - Value: {recent_value}
- Change: {direction} by {delta} ({percent_change_str})

INSTRUCTIONS:
Follow these exact steps in order:

1. FIRST, use query_docs(context_variables, collection_name="SFPublicData", query="{metric_name} trends analysis {period_type}")
   This will help you find any existing documentation about this metric.

2. SECOND, use get_dashboard_metric(context_variables, district_number={district_number}, metric_id="{metric_id}")
   This will retrieve the actual metric data from the dashboard.

3. THIRD, use query_anomalies_db(context_variables, query_type="by_metric", limit=5, metric_name="{metric_name}", district_filter="{district}")
   This will find any anomalies associated with this metric.

4. FINALLY, use explain_metric_change with all the parameters to generate an explanation:
   explain_metric_change(
       context_variables,
       metric_id="{metric_id}",
       metric_name="{metric_name}",
       district="{district}",
       period_type="{period_type}",
       previous_value={previous_value},
       recent_value={recent_value},
       previous_period="{previous_period}",
       recent_period="{recent_period}",
       delta={delta},
       percent_change={percent_change}
   )

After running these tools, provide a clear, comprehensive explanation of why this change occurred, focusing on:
- Historical context and trends
- Possible contributing factors
- Similar patterns in related metrics
- Whether this is part of a longer trend

IMPORTANT: You MUST use the tools in the order specified before providing your explanation.
"""
        
        logger.info(f"Generated prompt for agent: {prompt}")
        
        # Add the system message first, then the user message
        session_data["messages"].append(system_message)
        session_data["messages"].append({"role": "user", "content": prompt})
        
        # Get the agent from the session
        agent = session_data["agent"]
        logger.info(f"Using agent {agent.name} for metric change explanation with model: {AGENT_MODEL}")
        logger.info(f"Agent functions available: {[func.__name__ for func in agent.functions]}")
        
        # Process with the agent
        logger.info("Starting swarm_client.run for explanation")
        response_generator = swarm_client.run(
            agent=agent,
            messages=session_data["messages"],
            context_variables=session_data["context_variables"],
            stream=True,
            debug=debug_mode,
        )
        
        # Process and format the response
        full_response = ""
        current_function_name = None
        incomplete_tool_call = None
        tool_calls_made = []
        
        try:
            logger.info("Processing agent response stream")
            for chunk in response_generator:
                logger.debug(f"Received chunk: {str(chunk)[:200]}...")
                
                # Handle tool calls
                if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                    for tool_call in chunk["tool_calls"]:
                        function_info = tool_call.get("function")
                        if not function_info:
                            logger.warning("Received empty function info in tool call")
                            continue
                        
                        if function_info.get("name"):
                            current_function_name = function_info["name"]
                            logger.info(f"Receiving tool call: {current_function_name}")
                        
                        if not current_function_name:
                            logger.warning("Function name is empty or missing")
                            continue
                        
                        arguments_fragment = function_info.get("arguments", "")
                        logger.debug(f"Received arguments fragment: {arguments_fragment[:100]}...")
                        
                        if incomplete_tool_call is None or incomplete_tool_call["function_name"] != current_function_name:
                            incomplete_tool_call = {
                                "type": "tool_call",
                                "sender": "System",
                                "function_name": current_function_name,
                                "arguments": ""
                            }
                            logger.info(f"Started new tool call: {current_function_name}")
                        
                        incomplete_tool_call["arguments"] += arguments_fragment
                        
                        try:
                            full_args = incomplete_tool_call["arguments"]
                            arguments_json = json.loads(full_args)
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
                                    # Call the function with the context variables and arguments
                                    logger.info(f"Executing tool: {current_function_name}")
                                    result = function_to_call(session_data["context_variables"], **arguments_json)
                                    
                                    logger.info(f"""
=== Tool Result ===
Function: {current_function_name}
Result: {str(result)[:500]}{'...' if len(str(result)) > 500 else ''}
""")
                                    
                                    # Format the result for display
                                    if isinstance(result, dict):
                                        result_message = {
                                            "type": "tool_result",
                                            "sender": "System",
                                            "function_name": current_function_name,
                                            "result": result
                                        }
                                    else:
                                        result_message = {
                                            "type": "tool_result",
                                            "sender": "System",
                                            "function_name": current_function_name,
                                            "result": str(result)
                                        }
                                    
                                    yield json.dumps(result_message) + "\n"
                                except Exception as e:
                                    logger.error(f"""
=== Tool Error ===
Function: {current_function_name}
Error: {str(e)}
""")
                                    error_message = {
                                        "type": "error",
                                        "sender": "System",
                                        "function_name": current_function_name,
                                        "error": str(e)
                                    }
                                    yield json.dumps(error_message) + "\n"

                            incomplete_tool_call = None
                            current_function_name = None
                        except json.JSONDecodeError:
                            # Still accumulating arguments
                            pass
                
                # Handle content
                if "content" in chunk and chunk["content"] is not None:
                    content_piece = chunk["content"]
                    full_response += content_piece
                    logger.debug(f"Adding content to response: {content_piece[:50]}...")
        
            logger.info(f"Response generation complete. Tool calls made: {len(tool_calls_made)}")
            for i, call in enumerate(tool_calls_made):
                logger.info(f"Tool call {i+1}: {call['name']}")
            
        except Exception as e:
            err_msg = f"Error running agent: {str(e)}"
            logger.error(err_msg, exc_info=True)
            
            # Log all the information we have
            logger.error(f"Current function name: {current_function_name}")
            if incomplete_tool_call:
                logger.error(f"Incomplete tool call: {incomplete_tool_call['function_name']}")
                logger.error(f"Arguments accumulated so far: {incomplete_tool_call['arguments'][:200]}")
            
            logger.error(f"Most recent tool calls: {tool_calls_made[-3:] if tool_calls_made else 'None'}")
            logger.error(f"Available functions: {list(function_mapping.keys())}")
            
            full_response += f"\n\n**ERROR: I encountered an error while processing your request: {str(e)}**"
        
        # Check if no tool calls were made and add a special message
        if len(tool_calls_made) == 0:
            warning_message = """
### Warning: Analysis Incomplete

The system could not analyze this metric properly because no data tools were used. 

Please try again. If the problem persists, contact support.
"""
            full_response = warning_message + full_response
        
        # Log the final response length
        logger.info(f"Final response length: {len(full_response)}")
        logger.debug(f"Final response first 500 chars: {full_response[:500]}")
        
        # Add assistant response to history
        session_data["messages"].append({"role": "assistant", "content": full_response})
        
        # Yield the final content
        yield full_response
    
    except Exception as e:
        logger.error(f"Error in explain_metric_change_endpoint: {str(e)}", exc_info=True)
        error_message = {"error": f"Internal server error: {str(e)}"}
        yield json.dumps(error_message) + "\n"

@router.get("/api/query-anomalies")
async def query_anomalies_endpoint(
    query_type: str = "recent",
    limit: int = 10,
    group_filter: str = None,
    date_start: str = None,
    date_end: str = None,
    only_anomalies: bool = True,
    metric_name: str = None,
    object_id: str = None,
    period_type: str = None,
    district: str = None,
    only_active: bool = True
):
    """
    API endpoint to query anomalies for a specific metric.
    
    This endpoint allows fetching anomalies with various filters and is used
    by the metric detail view to show anomaly data for a selected metric.
    
    Parameters:
        query_type: Type of query to run (recent, by_group, by_date, by_anomaly_severity, by_district)
        limit: Maximum number of results to return
        group_filter: Filter by group value (substring match)
        date_start: Start date for filtering (format: YYYY-MM-DD)
        date_end: End date for filtering (format: YYYY-MM-DD)
        only_anomalies: If True, only return records where out_of_bounds=True
        metric_name: Filter by metric name in metadata
        object_id: Filter by object_id
        period_type: Filter by period type (month, quarter, year)
        district: Filter by district
        only_active: If True, only return active anomalies (is_active=True)
    """
    try:
        # Create context_variables dict
        context_variables = {}
        
        # Log the request parameters
        logging.info(f"Query anomalies request: metric_name={metric_name}, object_id={object_id}, district={district}, period_type={period_type}, only_active={only_active}")
        
        # Import the function here to make sure it's available
        try:
            from tools.store_anomalies import get_anomalies
        except ImportError:
            logging.error("Failed to import get_anomalies function")
            return {
                "status": "error",
                "message": "The anomaly detection system is not available",
                "anomalies": []
            }
        
        # Set a higher limit if we need to filter by object_id afterwards
        actual_limit = limit
        if object_id:
            actual_limit = limit * 10  # Request more results to account for filtering
        
        # Convert district to string to match database storage format
        if district is not None:
            district = str(district)
        
        # Call the get_anomalies function directly
        try:
            # If period_type is provided, add it to the query
            query_params = {
                "query_type": query_type,
                "limit": actual_limit,
                "group_filter": group_filter,
                "date_start": date_start,
                "date_end": date_end, 
                "only_anomalies": only_anomalies,
                "metric_name": metric_name,
                "district_filter": district
            }
            
            # Need to add a custom query for object_id since it's not a standard filter
            custom_query = ""
            custom_params = []
            
            if object_id:
                # Filter by object_id directly in SQL
                custom_query = " AND object_id = %s"
                custom_params.append(str(object_id))  # Make sure it's a string
                
            # Log query parameters for debugging
            logging.info(f"Calling get_anomalies with parameters: {query_params}")
            logging.info(f"Custom SQL filter for object_id: {custom_query}")
            
            # Use a modified approach to directly filter by object_id in SQL
            # Connect to database
            from tools.store_anomalies import get_postgres_connection
            from psycopg2.extras import RealDictCursor
            
            # Get database connection parameters from environment
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            db_user = os.getenv("POSTGRES_USER", "postgres")
            db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
            db_name = os.getenv("POSTGRES_DB", "transparentsf")
            
            connection = get_postgres_connection(
                host=db_host,
                port=int(db_port),
                dbname=db_name,
                user=db_user,
                password=db_password
            )
            
            if connection is None:
                return {
                    "status": "error",
                    "message": "Failed to connect to database"
                }
            
            # Create cursor
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Build query with standard filters and join with metrics table to get greendirection
            query = """
                SELECT a.*, m.greendirection 
                FROM anomalies a
                LEFT JOIN metrics m ON a.object_id = m.id::text
                WHERE 1=1 
            """
            params = []
            
            # Apply standard filters from get_anomalies logic
            if only_anomalies:
                query += "AND a.out_of_bounds = true "
                
            # Filter for active anomalies if requested
            if only_active:
                query += "AND a.is_active = true "
                
            if group_filter:
                query += "AND a.group_value ILIKE %s "
                params.append(f"%{group_filter}%")
                
            if date_start:
                query += "AND a.created_at >= %s "
                params.append(date_start)
                
            if date_end:
                query += "AND a.created_at <= %s "
                params.append(date_end)
                
            if metric_name:
                # Filter by metric name in the metadata JSON
                query += "AND a.metadata->>'object_name' ILIKE %s "
                params.append(f"%{metric_name}%")
            
            if district:
                # Filter by district
                query += "AND a.district = %s "
                params.append(district)
                
            # Add object_id filter - this is our key addition
            if object_id:
                query += "AND a.object_id = %s "
                params.append(str(object_id))
                
            # Add period_type filter if provided
            if period_type:
                query += "AND a.period_type = %s "
                params.append(period_type)
            
            # Order based on query_type
            if query_type == 'recent':
                query += "ORDER BY a.created_at DESC "
            elif query_type == 'by_group':
                query += "ORDER BY a.group_value ASC, a.created_at DESC "
            elif query_type == 'by_date':
                query += "ORDER BY a.created_at ASC "
            elif query_type == 'by_anomaly_severity':
                query += "ORDER BY ABS(a.difference) DESC "
            elif query_type == 'by_district':
                query += "ORDER BY a.district ASC, a.created_at DESC "
            
            # Add limit
            query += "LIMIT %s"
            params.append(limit)
            
            # Log the final query for debugging
            logging.info(f"Executing SQL query: {query}")
            logging.info(f"With parameters: {params}")
            
            # Execute query
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Convert to list of dictionaries and handle datetime objects
            result_list = []
            for row in results:
                row_dict = dict(row)
                # Format created_at for display
                if row_dict.get('created_at') and isinstance(row_dict['created_at'], datetime):
                    row_dict['created_at'] = row_dict['created_at'].isoformat()
                result_list.append(row_dict)
            
            cursor.close()
            connection.close()
            
            anomalies_result = {
                "status": "success",
                "count": len(result_list),
                "results": result_list
            }
            
        except Exception as e:
            logging.error(f"Error querying anomalies: {e}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Error accessing anomaly data: {str(e)}",
                "anomalies": []
            }
        
        # Process results to match the frontend's expected format and categorize by greendirection
        positive_anomalies = []
        negative_anomalies = []
        
        if anomalies_result.get("status") == "success":
            logging.info(f"Got {len(anomalies_result.get('results', []))} results from query")
            
            for item in anomalies_result.get("results", []):
                try:
                    # SQL results structure is different from before
                    # Now we have direct access to object_id and greendirection
                    
                    # Calculate percent change
                    comparison_mean = item.get("comparison_mean", 0)
                    percent_change = 0
                    if comparison_mean and comparison_mean != 0:
                        percent_change = (item.get("difference", 0) / comparison_mean) * 100
                    
                    # Get the appropriate date field
                    period_date = item.get("recent_date")
                    
                    # Format period date based on period type
                    item_period_type = item.get("period_type", "")
                    formatted_date = period_date
                    if period_date:
                        if isinstance(period_date, str):
                            try:
                                # Convert string to datetime if needed
                                period_date = datetime.fromisoformat(period_date.replace('Z', '+00:00'))
                            except Exception as e:
                                logging.error(f"Error parsing date: {e}")
                                
                        if item_period_type == 'month':
                            formatted_date = period_date.strftime('%b %Y')
                        elif item_period_type == 'year':
                            formatted_date = period_date.strftime('%Y')
                        elif item_period_type == 'quarter':
                            quarter = (period_date.month - 1) // 3 + 1
                            formatted_date = f"Q{quarter} {period_date.year}"
                        else:
                            formatted_date = period_date.strftime('%Y-%m-%d')
                    
                    # Extract data from the result
                    anomaly_data = {
                        "id": item.get("id"),
                        "metric_name": item.get("object_name", ""),  # Use object_name directly
                        "object_id": item.get("object_id", ""),      # Use object_id directly
                        "district": item.get("district", "0"),
                        "group_value": item.get("group_value", ""),
                        "group_field_name": item.get("group_field_name", ""),  # Add group_field_name
                        "recent_mean": item.get("recent_mean", 0),
                        "comparison_mean": comparison_mean,
                        "difference": item.get("difference", 0),
                        "percent_change": percent_change,
                        "std_dev": item.get("std_dev", 0),
                        "period_date": formatted_date,
                        "period_type": item_period_type,
                        "explanation": item.get("explanation", ""),
                        "is_active": item.get("is_active", True),
                        "greendirection": item.get("greendirection")  # Include greendirection from metrics table
                    }
                    
                    # Categorize anomaly as positive or negative based on greendirection
                    if percent_change is not None and item.get("greendirection"):
                        greendirection = item.get("greendirection").lower()
                        
                        # Determine if this is a positive or negative anomaly based on greendirection
                        if greendirection == 'up':
                            # For metrics where up is good (greendirection=up)
                            if percent_change > 0:
                                positive_anomalies.append(anomaly_data)  # Increase is positive
                            elif percent_change < 0:
                                negative_anomalies.append(anomaly_data)  # Decrease is negative
                        elif greendirection == 'down':
                            # For metrics where down is good (greendirection=down)
                            if percent_change > 0:
                                negative_anomalies.append(anomaly_data)  # Increase is negative
                            elif percent_change < 0:
                                positive_anomalies.append(anomaly_data)  # Decrease is positive
                        else:
                            # If greendirection is not 'up' or 'down', use traditional logic
                            if percent_change > 0:
                                positive_anomalies.append(anomaly_data)
                            elif percent_change < 0:
                                negative_anomalies.append(anomaly_data)
                    else:
                        # Fallback to traditional logic if no greendirection or percent_change
                        if percent_change > 0:
                            positive_anomalies.append(anomaly_data)
                        elif percent_change < 0:
                            negative_anomalies.append(anomaly_data)
                        
                except Exception as item_error:
                    logging.error(f"Error processing anomaly item: {item_error}")
                    logging.error(f"Problem item: {item}")
            
            # Sort by absolute percent change (highest impact first)
            positive_anomalies = sorted(positive_anomalies, 
                                      key=lambda x: abs(float(x['percent_change'])), 
                                      reverse=True)[:limit]
            
            negative_anomalies = sorted(negative_anomalies, 
                                      key=lambda x: abs(float(x['percent_change'])), 
                                      reverse=True)[:limit]
            
            logging.info(f"Categorized anomalies: {len(positive_anomalies)} positive, {len(negative_anomalies)} negative")
            
            return {
                "status": "success",
                "count": len(positive_anomalies) + len(negative_anomalies),
                "positive_anomalies": positive_anomalies,
                "negative_anomalies": negative_anomalies
            }
        else:
            # Return error from the anomalies result
            return {
                "status": "error",
                "message": anomalies_result.get("message", "Unknown error occurred"),
                "anomalies": []
            }
    
    except Exception as e:
        logging.error(f"Error in query_anomalies_endpoint: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to query anomalies: {str(e)}",
            "anomalies": []
        }

@router.get("/api/anomaly-details/{anomaly_id}")
async def get_anomaly_details_endpoint(anomaly_id: int):
    """
    API endpoint to get detailed information about a specific anomaly by ID.
    
    Args:
        anomaly_id: The ID of the anomaly to retrieve
        
    Returns:
        JSON with detailed anomaly information
    """
    try:
        # Create context_variables dict
        context_variables = {}
        
        # Log the request
        logger.info(f"Fetching detailed anomaly data for ID: {anomaly_id}")
        
        # Use the get_anomaly_details function
        result = get_anomaly_details(context_variables, anomaly_id)
        
        # Return the result
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error fetching anomaly details: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to fetch anomaly details: {str(e)}"
            }
        )

@router.get("/anomaly-chart")
async def anomaly_chart_page(request: Request):
    """
    Serve the dedicated anomaly chart page.
    
    Query Parameters:
        id: ID of the anomaly to display
        format: Set to 'image' to return a static image instead of HTML
    """
    # Check if format=image is in the query parameters
    format_param = request.query_params.get('format')
    
    if format_param == 'image':
        try:
            # Extract parameters for the chart
            anomaly_id = request.query_params.get('id')
            
            if not anomaly_id:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "id is required for image export"}
                )
            
            # Get anomaly details
            context_variables = {}
            anomaly_data = get_anomaly_details(context_variables, int(anomaly_id))
            
            if anomaly_data.get('status') != 'success' or not anomaly_data.get('anomaly'):
                return JSONResponse(
                    status_code=404,
                    content={"detail": "No anomaly found with the specified ID"}
                )
            
            # Use Plotly to create the chart image
            import plotly.graph_objects as go
            import plotly.io as pio
            from io import BytesIO
            import base64
            from datetime import datetime
            
            # Create the figure
            fig = go.Figure()
            
            # Get anomaly data
            anomaly = anomaly_data.get('anomaly', {})
            chart_data = anomaly.get('chart_data', {})
            metadata = anomaly.get('metadata', {})
            
            # Process chart data
            recent_dates = []
            recent_values = []
            comparison_dates = []
            comparison_values = []
            
            if chart_data.get('dates') and chart_data.get('values') and chart_data.get('periods'):
                for i, date_str in enumerate(chart_data.get('dates', [])):
                    value = chart_data.get('values', [])[i] if i < len(chart_data.get('values', [])) else None
                    period = chart_data.get('periods', [])[i] if i < len(chart_data.get('periods', [])) else None
                    
                    if value is None or period is None:
                        continue
                    
                    # Convert date string to datetime
                    try:
                        # Handle different date formats
                        if len(date_str.split('-')) == 2:  # Format YYYY-MM
                            year, month = date_str.split('-')
                            date_obj = datetime(int(year), int(month), 1)
                        else:  # Full date
                            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        
                        # Add to appropriate array based on period
                        if period == 'recent':
                            recent_dates.append(date_obj)
                            recent_values.append(value)
                        elif period == 'comparison':
                            comparison_dates.append(date_obj)
                            comparison_values.append(value)
                    except Exception as date_error:
                        logger.error(f"Error parsing date {date_str}: {date_error}")
            
            # Sort dates chronologically
            recent_data = sorted(zip(recent_dates, recent_values), key=lambda x: x[0])
            comparison_data = sorted(zip(comparison_dates, comparison_values), key=lambda x: x[0])
            
            # Unzip after sorting
            recent_dates = [d for d, v in recent_data] if recent_data else []
            recent_values = [v for d, v in recent_data] if recent_data else []
            comparison_dates = [d for d, v in comparison_data] if comparison_data else []
            comparison_values = [v for d, v in comparison_data] if comparison_data else []
            
            # Calculate normal range
            comparison_mean = anomaly.get('comparison_mean', 0)
            std_dev = anomaly.get('std_dev', 0)
            
            # Add normal range area
            if comparison_mean is not None and std_dev is not None:
                all_dates = sorted(comparison_dates + recent_dates)
                if all_dates:
                    upper_bound = [comparison_mean + 2 * std_dev] * len(all_dates)
                    lower_bound = [max(comparison_mean - 2 * std_dev, 0)] * len(all_dates)
                    
                    # Add lower bound trace (invisible)
                    fig.add_trace(go.Scatter(
                        x=all_dates,
                        y=lower_bound,
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=False
                    ))
                    
                    # Add normal range area
                    fig.add_trace(go.Scatter(
                        x=all_dates,
                        y=upper_bound,
                        mode='none',
                        fill='tonexty',
                        fillcolor='rgba(74, 116, 99, 0.15)',  # Spruce Green with transparency
                        name='Normal Range',
                        showlegend=True
                    ))
            
            # Add comparison period trace
            if comparison_dates and comparison_values:
                fig.add_trace(go.Scatter(
                    x=comparison_dates,
                    y=comparison_values,
                    mode='lines+markers',
                    name='Time Series',
                    line=dict(color='#ad35fa', width=2),  # Bright Blue
                    marker=dict(color='#ad35fa', size=6)
                ))
            
            # Add recent period trace
            if recent_dates and recent_values:
                fig.add_trace(go.Scatter(
                    x=recent_dates,
                    y=recent_values,
                    mode='lines+markers',
                    name='Time Series',
                    line=dict(color='#ad35fa', width=2),  # Bright Blue
                    marker=dict(color='#ad35fa', size=6),
                    showlegend=False  # Hide duplicate legend entry
                ))
            
            # Add connector line between comparison and recent periods
            if comparison_dates and comparison_values and recent_dates and recent_values:
                fig.add_trace(go.Scatter(
                    x=[comparison_dates[-1], recent_dates[0]],
                    y=[comparison_values[-1], recent_values[0]],
                    mode='lines',
                    line=dict(color='#ad35fa', width=2),  # Bright Blue
                    showlegend=False
                ))
            
            # Get the metric name from metadata
            object_name = metadata.get('object_name', anomaly.get('field_name', 'Value'))
            
            # Create subtitle with district if present
            group_field_name = anomaly.get('group_field_name', 'Group')
            group_value = anomaly.get('group_value', 'Value')
            subtitle = f"{group_field_name}: {group_value}"
            
            # Update layout
            fig.update_layout(
                title={
                    'text': object_name,
                    'y': 0.95,
                    'font': {'size': 16, 'color': '#222222'}
                },
                annotations=[
                    dict(
                        text=f"{'Spike' if anomaly.get('percent_change', 0) > 0 else 'Drop'} in {subtitle}",
                        showarrow=False,
                        xref="paper", 
                        yref="paper",
                        x=0.5, 
                        y=0.9,
                        font={'size': 14, 'color': '#222222'}
                    )
                ],
                xaxis={
                    'title': '',
                    'showgrid': False,
                    'showline': True,
                    'linecolor': '#e5e7eb',
                    'linewidth': 1,
                    'tickmode': 'auto',
                    'tickformat': '%b %Y',
                    'tickangle': 0
                },
                yaxis={
                    'title': metadata.get('y_axis_label', object_name),
                    'showgrid': True,
                    'gridcolor': 'rgba(232, 233, 235, 0.5)',
                    'zeroline': False,
                    'rangemode': 'tozero'
                },
                legend={
                    'orientation': 'h',
                    'x': 0.5,
                    'y': -0.12,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'bgcolor': 'rgba(246, 241, 234, 0.7)'
                },
                autosize=False,
                width=1000,
                height=600,
                margin=dict(l=80, r=80, t=120, b=100),
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(255,255,255,0.9)'
            )
            
            # Add annotation for most recent value if available
            if recent_dates and recent_values:
                fig.add_annotation(
                    x=recent_dates[-1],
                    y=recent_values[-1],
                    text=f"{recent_dates[-1].strftime('%b %Y')}: {recent_values[-1]:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=-50,
                    ay=30,
                    bgcolor='rgba(0, 123, 255, 0.7)',
                    bordercolor='#ad35fa',
                    borderwidth=1
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
                    "Content-Disposition": f"attachment; filename=anomaly_{anomaly_id}.png"
                }
            )
        
        except Exception as e:
            logger.error(f"Error generating anomaly chart image: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error generating chart image: {str(e)}"}
            )
    
    # Default behavior - return HTML page
    # Use the templates instance set by main.py
    if templates is None:
        logger.error("Templates not initialized in anomalyAnalyzer router")
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        templates_local = Jinja2Templates(directory=templates_dir)
        return templates_local.TemplateResponse("anomaly_chart.html", {"request": request})
    
    return templates.TemplateResponse("anomaly_chart.html", {"request": request})

@router.post("/api/add-to-monthly-report")
async def add_to_monthly_report(request: Request):
    """
    Add an item from the anomaly analyzer to a monthly report.
    Creates a new monthly report if one doesn't exist for the current month.
    """
    try:
        data = await request.json()
        item_data = data.get("item_data", {})
        report_title = data.get("report_title", "")
        district = data.get("district", "0")
        period_type = data.get("period_type", "month")
        
        logger.info(f"Adding item to monthly report: {item_data.get('object_name', 'Unknown')}")
        
        # Import the monthly report functions
        from .monthly_report import (
            initialize_monthly_reporting_table, 
            store_prioritized_items,
            get_monthly_reports_list
        )
        
        # Initialize the monthly reporting table
        init_result = initialize_monthly_reporting_table()
        if not init_result:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to initialize monthly reporting table"}
            )
        
        # Get current date
        from datetime import datetime
        report_date = datetime.now().date()
        
        # Check if a report already exists for this month and district
        existing_reports = get_monthly_reports_list()
        existing_report = None
        
        for report in existing_reports:
            if (report.get("district") == district and 
                report.get("period_type") == period_type):
                # Check if it's from the current month
                report_date_str = report.get("report_date", "")
                if report_date_str:
                    try:
                        report_datetime = datetime.fromisoformat(report_date_str.replace('Z', '+00:00'))
                        if (report_datetime.year == report_date.year and 
                            report_datetime.month == report_date.month):
                            existing_report = report
                            break
                    except:
                        continue
        
        # Create the item data in the format expected by store_prioritized_items
        prioritized_item = {
            "metric": item_data.get("object_name", "Unknown Metric"),
            "metric_id": item_data.get("id", "Unknown"),
            "group": item_data.get("group_value", "All"),
            "recent_mean": float(item_data.get("recent_value", 0)),
            "comparison_mean": float(item_data.get("previous_value", 0)),
            "difference_value": float(item_data.get("delta", 0)),
            "difference": float(item_data.get("delta", 0)),
            "district": district,
            "change_type": "positive" if float(item_data.get("delta", 0)) > 0 else "negative",
            "percent_change": float(item_data.get("percent_change", 0)),
            "greendirection": "up",  # Default, could be enhanced to get from metrics table
            "citywide_changes": "",
            "priority": 1,  # High priority since user manually selected it
            "explanation": f"Manually added from anomaly analyzer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "trend_analysis": "",
            "follow_up": ""
        }
        
        # Add anomaly-specific metadata if this is an anomaly
        if item_data.get("type") == "anomaly":
            prioritized_item["metadata"] = {
                "anomaly_id": item_data.get("anomaly_id"),
                "source": "anomaly_analyzer",
                "added_manually": True,
                "added_at": datetime.now().isoformat()
            }
        else:
            prioritized_item["metadata"] = {
                "source": "anomaly_analyzer",
                "added_manually": True,
                "added_at": datetime.now().isoformat()
            }
        
        if existing_report:
            # Add to existing report
            logger.info(f"Adding item to existing report: {existing_report.get('id')}")
            
            # Import database utilities
            from .tools.db_utils import execute_with_connection
            import os
            
            def add_to_existing_report_operation(connection):
                cursor = connection.cursor()
                
                # Get the next priority number for this report
                cursor.execute("""
                    SELECT COALESCE(MAX(priority), 0) + 1 
                    FROM monthly_reporting 
                    WHERE report_id = %s
                """, (existing_report["id"],))
                
                next_priority = cursor.fetchone()[0]
                
                # Create item title
                metric_name = prioritized_item["metric"]
                group_value = prioritized_item["group"]
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
                    existing_report["id"],
                    report_date,
                    item_title,
                    prioritized_item["metric"],
                    prioritized_item["metric_id"],
                    prioritized_item["group"],
                    "group",
                    period_type,
                    prioritized_item["comparison_mean"],
                    prioritized_item["recent_mean"],
                    prioritized_item["difference"],
                    prioritized_item["percent_change"],
                    prioritized_item["explanation"],
                    "",  # explanation: will be filled in by generate_explanations
                    next_priority,
                    district,
                    json.dumps(prioritized_item["metadata"])
                ))
                
                inserted_id = cursor.fetchone()[0]
                connection.commit()
                cursor.close()
                return inserted_id
            
            # Execute the operation
            result = execute_with_connection(
                operation=add_to_existing_report_operation,
                db_host=os.getenv("POSTGRES_HOST", "localhost"),
                db_port=int(os.getenv("POSTGRES_PORT", "5432")),
                db_name=os.getenv("POSTGRES_DB", "transparentsf"),
                db_user=os.getenv("POSTGRES_USER", "postgres"),
                db_password=os.getenv("POSTGRES_PASSWORD", "postgres")
            )
            
            if result["status"] != "success":
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Failed to add item to existing report: {result['message']}"}
                )
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Successfully added '{prioritized_item['metric']}' to existing monthly report",
                "report_id": existing_report["id"],
                "item_id": result["result"]
            })
            
        else:
            # Create a new report
            logger.info("Creating new monthly report")
            
            # Store the prioritized item (this will create a new report)
            store_result = store_prioritized_items(
                [prioritized_item],
                period_type=period_type,
                district=district
            )
            
            if store_result.get("status") != "success":
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Failed to create new report: {store_result.get('message')}"}
                )
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Successfully created new monthly report and added '{prioritized_item['metric']}'",
                "report_id": store_result.get("report_id"),
                "item_id": store_result.get("inserted_ids", [None])[0]
            })
            
    except Exception as e:
        logger.error(f"Error adding item to monthly report: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to add item to monthly report: {str(e)}"}
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000) 
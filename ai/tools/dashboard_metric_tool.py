"""
Dashboard Metric Tool - Retrieves dashboard metric data and analysis files
"""

import json
import logging
import os
from pathlib import Path
import re

# Configure logging
logger = logging.getLogger(__name__)


def get_dashboard_metric(context_variables, district_number=0, metric_id=None):
    """
    Retrieves dashboard metric data for a specific district and metric.
    
    Args:
        context_variables: The context variables dictionary
        district_number: The district number (0 for citywide, 1-11 for specific districts). Can be int or string.
        metric_id: The specific metric ID to retrieve. If None, returns the top_level.json file.
    
    Returns:
        A dictionary containing the metric data or error message
    """
    try:
        # Convert district_number to int if it's a string
        try:
            district_number = int(district_number)
        except (TypeError, ValueError):
            return {"error": f"Invalid district number format: {district_number}. Must be a number between 0 and 11."}
        
        # Validate district number range
        if district_number < 0 or district_number > 11:
            return {"error": f"Invalid district number: {district_number}. Must be between 0 (citywide) and 11."}
        
        # Construct the base path - looking one level up from the script
        script_dir = Path(__file__).parent.parent
        dashboard_dir = script_dir / 'output' / 'dashboard'
        
        logger.info(f"Looking for dashboard data in: {dashboard_dir}")
        
        # If metric_id is None, return the top-level district summary
        if metric_id is None:
            file_path = dashboard_dir / f"district_{district_number}.json"
            logger.info(f"Fetching top-level dashboard data for district {district_number} from {file_path}")
        else:
            # Metric ID is provided, look in the district-specific folder
            district_folder = dashboard_dir / str(district_number)
            
            # Convert metric_id to string and handle .json extension
            metric_id_str = str(metric_id)
            if not metric_id_str.endswith('.json'):
                metric_id_str = f"{metric_id_str}.json"
                
            file_path = district_folder / metric_id_str
            logger.info(f"Fetching specific metric '{metric_id_str}' for district {district_number} from {file_path}")
        
        # Check if the file exists
        if not file_path.exists():
            # If specific metric file doesn't exist, list available metrics
            if metric_id is not None:
                available_metrics = []
                district_folder = dashboard_dir / str(district_number)
                if district_folder.exists():
                    available_metrics = [f.name for f in district_folder.glob('*.json')]
                
                return {
                    "error": f"Metric '{metric_id_str}' not found for district {district_number}",
                    "available_metrics": available_metrics
                }
            else:
                return {"error": f"Dashboard data not found for district {district_number}"}
        
        # Read and parse the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add metadata about the source
        result = {
            "source": str(file_path),
            "district": district_number,
            "metric_id": metric_id_str,
            "data": data
        }
        
        # Add endpoint from metric data if available
        if isinstance(data, dict) and "endpoint" in data:
            result["endpoint"] = data["endpoint"]
            logger.info(f"Found endpoint in metric data: {data['endpoint']}")
        
        # Try to find and add corresponding analysis files
        analysis_content = {}
        total_analysis_length = 0
        max_tokens = 100000  # Approximate max tokens we want to allow
        
        # Get metric ID for analysis files
        metric_id_number = None
        
        # Clean the metric_id to use as base filename
        if metric_id_str:
            base_metric_name = metric_id_str.replace('.json', '')
            
            # Try to extract the ID number from the data if we have a named metric
            if isinstance(data, dict) and "id" in data:
                # If the metric has an ID in its data, use that for analysis files
                metric_id_number = str(data["id"])
                logger.info(f"Found metric ID number {metric_id_number} from data")
            elif base_metric_name.isdigit():
                # If the metric_id itself is a number, use it directly
                metric_id_number = base_metric_name
                logger.info(f"Using metric_id as a number: {metric_id_number}")
            else:
                # Try to find the ID from the dashboard_queries_enhanced.json file
                queries_file = script_dir / 'data' / 'dashboard' / 'dashboard_queries_enhanced.json'
                if queries_file.exists():
                    try:
                        with open(queries_file, 'r', encoding='utf-8') as f:
                            queries_data = json.load(f)
                        
                        # Look for the metric name in the queries data
                        for category in queries_data.values():
                            for subcategory in category.values():
                                if "queries" in subcategory:
                                    for query_name, query_data in subcategory["queries"].items():
                                        # Check if the name matches our metric name (with or without emoji)
                                        item_name = query_name.lower()
                                        clean_metric_name = base_metric_name.lower()
                                        
                                        # Remove emojis and special characters for comparison
                                        clean_item_name = re.sub(r'[^\w\s]', '', item_name).strip()
                                        clean_base_name = re.sub(r'[^\w\s]', '', clean_metric_name).strip()
                                        
                                        if clean_item_name == clean_base_name or item_name == clean_metric_name:
                                            metric_id_number = str(query_data["id"])
                                            # Add query information to result
                                            result["endpoint"] = query_data.get("endpoint")
                                            result["ytd_query"] = query_data.get("ytd_query")
                                            result["metric_query"] = query_data.get("metric_query")
                                            logger.info(f"Found metric ID {metric_id_number} from dashboard_queries_enhanced.json")
                                            break
                        
                        if not metric_id_number:
                            logger.info(f"Could not find metric ID for '{base_metric_name}' in dashboard_queries_enhanced.json")
                    except Exception as e:
                        logger.error(f"Error reading dashboard_queries_enhanced.json: {str(e)}")
                else:
                    logger.info(f"dashboard_queries_enhanced.json not found at {queries_file}")
                
                if not metric_id_number:
                    logger.info(f"Could not determine metric ID number from {base_metric_name}")
            
            # Look for analysis files in monthly, annual, and weekly folders
            monthly_dir = script_dir / 'output' / 'monthly'
            annual_dir = script_dir / 'output' / 'annual'
            weekly_dir = script_dir / 'output' / 'weekly'
            
            # Only proceed if we have a metric ID number
            if metric_id_number:
                # Paths for analysis files using the ID number
                monthly_analysis_path = monthly_dir / f"{district_number}/{metric_id_number}.md"
                annual_analysis_path = annual_dir / f"{district_number}/{metric_id_number}.md"
                weekly_analysis_path = weekly_dir / f"{district_number}/{metric_id_number}.md"
                
                logger.info(f"Looking for monthly analysis at: {monthly_analysis_path}")
                logger.info(f"Looking for annual analysis at: {annual_analysis_path}")
                logger.info(f"Looking for weekly analysis at: {weekly_analysis_path}")
                
                # Read monthly analysis if it exists
                if monthly_analysis_path.exists():
                    try:
                        with open(monthly_analysis_path, 'r', encoding='utf-8') as f:
                            monthly_content = f.read()
                            total_analysis_length += len(monthly_content.split())
                            analysis_content["monthly_analysis"] = monthly_content
                            logger.info(f"Found monthly analysis file ({len(monthly_content)} chars)")
                    except Exception as e:
                        logger.error(f"Error reading monthly analysis: {str(e)}")
                
                # Read annual analysis if it exists
                if annual_analysis_path.exists():
                    try:
                        with open(annual_analysis_path, 'r', encoding='utf-8') as f:
                            annual_content = f.read()
                            total_analysis_length += len(annual_content.split())
                            analysis_content["annual_analysis"] = annual_content
                            logger.info(f"Found annual analysis file ({len(annual_content)} chars)")
                    except Exception as e:
                        logger.error(f"Error reading annual analysis: {str(e)}")
                
                # Read weekly analysis if it exists
                if weekly_analysis_path.exists():
                    try:
                        with open(weekly_analysis_path, 'r', encoding='utf-8') as f:
                            weekly_content = f.read()
                            total_analysis_length += len(weekly_content.split())
                            analysis_content["weekly_analysis"] = weekly_content
                            logger.info(f"Found weekly analysis file ({len(weekly_content)} chars)")
                    except Exception as e:
                        logger.error(f"Error reading weekly analysis: {str(e)}")
            else:
                logger.info("No metric ID number available for finding analysis files")
            
            # Check if we need to summarize (approximating tokens as words/0.75)
            estimated_tokens = total_analysis_length / 0.75
            if estimated_tokens > max_tokens and analysis_content:
                logger.info(f"Analysis content too large (~{estimated_tokens:.0f} tokens). Summarizing...")
                
                # Create a simple summary by truncating and adding a note
                for key in analysis_content:
                    original_length = len(analysis_content[key])
                    # Calculate how much to keep (proportional to original length)
                    proportion = len(analysis_content[key]) / total_analysis_length
                    max_chars = int((max_tokens * 0.75) * proportion * 4)  # Rough char estimate
                    
                    if len(analysis_content[key]) > max_chars:
                        analysis_content[key] = (
                            f"{analysis_content[key][:max_chars]}\n\n"
                            f"[Note: Analysis truncated due to length. Original size: {original_length} characters]"
                        )
                        logger.info(f"Truncated {key} from {original_length} to {len(analysis_content[key])} chars")
            
            # Add analysis content to result if any was found
            if analysis_content:
                result["analysis"] = analysis_content
                logger.info(f"Added analysis content with keys: {list(analysis_content.keys())}")
        
        logger.info(f"Successfully retrieved dashboard metric data from {file_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard metric: {str(e)}", exc_info=True)
        return {"error": f"Error retrieving dashboard metric: {str(e)}"} 
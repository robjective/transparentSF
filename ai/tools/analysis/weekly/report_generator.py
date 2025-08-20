"""
Report Generator Module for Weekly Analysis

This module handles saving analysis results to files and generating
weekly newsletters.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables to get LOG_LEVEL
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Get log level from environment variable
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_level_map.get(log_level_str, logging.INFO)

# Configure logging for this module
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create module logger
logger = logging.getLogger(__name__)

# Configure file handler for weekly analysis
weekly_log_file = os.path.join(logs_dir, 'weekly_metric_analysis.log')
file_handler = logging.FileHandler(weekly_log_file, mode='a', encoding='utf-8')
file_handler.setLevel(log_level)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger if it doesn't already have one
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.setLevel(log_level)

logger.info(f"Weekly report generator logging initialized with level: {log_level_str}. Log file: {weekly_log_file}")

# Get script directory and ensure output directories exist
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(script_dir, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectory for weekly analysis
WEEKLY_DIR = os.path.join(OUTPUT_DIR, 'weekly')
os.makedirs(WEEKLY_DIR, exist_ok=True)

def save_weekly_analysis(result, metric_id, district=None):
    """Save weekly analysis results to markdown files following the directory structure used
    in generate_metric_analysis.py."""
    # Create output directory if it doesn't exist
    os.makedirs(WEEKLY_DIR, exist_ok=True)
    
    # Use the numeric metric_id if available
    # First try from the input parameter
    file_metric_id = metric_id
    
    # Ensure we have a valid metric_id
    if not file_metric_id or (isinstance(file_metric_id, str) and file_metric_id.strip() == ''):
        # Try to get it from the result object
        file_metric_id = result.get('metric_id', '')
        
        if not file_metric_id or (isinstance(file_metric_id, str) and file_metric_id.strip() == ''):
            # As a last resort, generate a sanitized metric ID from the query name
            query_name = result.get('name', '')
            if query_name:
                file_metric_id = query_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            else:
                file_metric_id = "unknown_metric"
            logger.warning(f"Missing metric_id for {query_name}, using sanitized query name: {file_metric_id}")
    
    # Create district subfolder using the district number - 0 for citywide
    if district is not None:
        # Convert district to integer if it's a decimal for proper folder naming
        try:
            if isinstance(district, str):
                # Try to convert string to float first, then to int
                district_folder = str(int(float(district)))
            else:
                # Handle numeric types (float, int, etc.)
                district_folder = str(int(float(district)))
        except (ValueError, TypeError):
            # If conversion fails, use the original value as string
            district_folder = str(district)
        
        district_dir = os.path.join(WEEKLY_DIR, district_folder)
        os.makedirs(district_dir, exist_ok=True)
        output_path = district_dir
    else:
        # Default to folder 0 if no district specified (treating as citywide)
        district_dir = os.path.join(WEEKLY_DIR, "0")
        os.makedirs(district_dir, exist_ok=True)
        output_path = district_dir
    
    # Generate filename - just metric_id.md (no date needed as it will be overwritten each week)
    md_filename = f"{file_metric_id}.md"
    md_path = os.path.join(output_path, md_filename)
    
    # Check for other files with similar metric names and remove them
    # This helps avoid duplicate files when names change
    query_name = result.get('name', '')
    if query_name:
        # Create a cleaned version of the name for comparison
        cleaned_name = query_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        # Look for files matching this pattern, except our target files
        for filename in os.listdir(output_path):
            file_path = os.path.join(output_path, filename)
            # Check if it might be a previously saved version with different naming
            if (cleaned_name in filename and 
                filename != md_filename and
                not filename.startswith(".")):  # Skip hidden files
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old file with different naming: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove old file {file_path}: {str(e)}")
    
    # Log the file path being used
    logger.info(f"Saving weekly analysis to: {md_path}")
    
    # Get the markdown and HTML content
    markdown_content = result.get('content', '')
    # Build "data current as of" header if info available
    header = ""
    data_as_of = result.get('data_as_of')
    date_field_name = result.get('date_field')
    if data_as_of and date_field_name:
        header = f"**Data current as of {data_as_of} (date field: {date_field_name})**\n\n"
    elif data_as_of:
        header = f"**Data current as of {data_as_of}**\n\n"
    # Prepend header if not already present
    if header:
        markdown_content = header + markdown_content
    html_content = result.get('html_contents', [])
    
    try:
        # Write markdown file
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        logger.info(f"Successfully wrote markdown file ({len(markdown_content)} chars) to {md_path}")
    except Exception as e:
        logger.error(f"Error writing markdown file to {md_path}: {str(e)}")
    
    # Get district description based on district value
    if district == 0 or district is None:
        district_info = " for Citywide"
    else:
        district_info = f" for District {district}"
        
    logger.info(f"Saved weekly analysis for {file_metric_id}{district_info} to {md_path}")
    
    return {
        'md_path': md_path
    }

def generate_weekly_newsletter(results):
    """Generate a weekly newsletter based on the analysis results."""
    start_time = datetime.now()
    logger.info(f"========== STARTING NEWSLETTER GENERATION: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    
    if not results:
        logger.warning("No results to include in newsletter - skipping generation")
        return None
    
    logger.info(f"Generating newsletter with {len(results)} metric results")
    
    today = datetime.now().strftime('%Y-%m-%d')
    newsletter_title = f"SF Data Weekly Trends - {today}"
    logger.info(f"Newsletter title: {newsletter_title}")
    
    # Start building the newsletter content
    newsletter_content = f"# {newsletter_title}\n\n"
    newsletter_content += "## This Week's Data Trends\n\n"
    
    # Add each analysis result to the newsletter
    for i, result in enumerate(results):
        metric_id = result.get('metric_id', '')
        query_name = result.get('name', '')
        file_path = result.get('md_path', '')
        
        logger.info(f"Adding result {i+1}/{len(results)} to newsletter: {metric_id} - {query_name}")
        
        # Add a section for this metric
        newsletter_content += f"### {query_name}\n\n"
        
        # Include links to the full analysis
        if file_path:
            md_relative_path = os.path.relpath(file_path, OUTPUT_DIR)
            newsletter_content += f"[View full analysis (Markdown)]({md_relative_path})\n\n"
            logger.info(f"Added link to markdown analysis file: {md_relative_path}")
        else:
            logger.warning(f"No markdown file path for {metric_id} - {query_name}")
        
        # Include a summary of key findings (this would need to be extracted from the analysis)
        newsletter_content += "Key findings:\n"
        # This is a placeholder - in a real implementation, you'd parse the analysis
        # to extract important trends, anomalies, etc.
        newsletter_content += "- Weekly data analysis completed\n"
        newsletter_content += "- Check the full report for detailed trends\n\n"
    
    # Save the newsletter
    newsletter_path = os.path.join(WEEKLY_DIR, f"weekly_newsletter_{today}.md")
    logger.info(f"Saving newsletter to: {newsletter_path}")
    
    try:
        with open(newsletter_path, 'w') as f:
            f.write(newsletter_content)
        logger.info(f"Successfully saved newsletter ({len(newsletter_content)} characters)")
    except Exception as e:
        logger.error(f"Error saving newsletter to {newsletter_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"========== NEWSLETTER GENERATION COMPLETE: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    logger.info(f"Total duration: {duration:.2f} seconds")
    logger.info(f"Newsletter saved to: {newsletter_path}")
    
    return newsletter_path 
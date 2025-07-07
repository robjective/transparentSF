"""
Scheduler Module for Weekly Analysis

This module handles scheduled weekly tasks and automation.
"""

import logging
import traceback
from datetime import datetime

from .analysis_engine import run_weekly_analysis
# from .report_generator import generate_weekly_newsletter

# Configure logging for this module
import os
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

logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create module logger
logger = logging.getLogger(__name__)

# Configure file handler for weekly analysis
weekly_log_file = os.path.join(logs_dir, 'weekly_metric_analysis.log')
file_handler = logging.FileHandler(weekly_log_file, mode='a', encoding='utf-8')
file_handler.setLevel(log_level)  # Use environment variable instead of hardcoded INFO

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger if it doesn't already have one
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.setLevel(log_level)  # Use environment variable instead of hardcoded INFO

logger.info(f"Weekly scheduler logging initialized with level: {log_level_str}. Log file: {weekly_log_file}")

def scheduled_weekly_task():
    """Task to run weekly analysis and generate newsletter."""
    start_time = datetime.now()
    logger.info(f"========== STARTING SCHEDULED WEEKLY TASK: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    
    # Run analysis for all metrics
    logger.info("Running weekly analysis with district processing enabled")
    try:
        results = run_weekly_analysis(process_districts=True)
        
        if results:
            logger.info(f"Weekly analysis completed successfully with {len(results)} metrics")
        else:
            logger.warning("Weekly analysis completed but returned no results")
            
        # Newsletter generation temporarily disabled
        # logger.info("Proceeding to newsletter generation")
        # newsletter_path = generate_weekly_newsletter(results)
        
        # if newsletter_path:
        #     logger.info(f"Weekly task completed successfully, newsletter saved to {newsletter_path}")
        # else:
        #     logger.error("Weekly task did not generate a newsletter")
        
        logger.info("Weekly task completed successfully (newsletter generation disabled)")
    except Exception as e:
        logger.error(f"Exception in scheduled weekly task: {str(e)}")
        logger.error(traceback.format_exc())
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"========== SCHEDULED WEEKLY TASK COMPLETE: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    logger.info(f"Total task duration: {duration:.2f} seconds ({duration/60:.2f} minutes)") 
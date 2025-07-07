"""
Time Utilities Module for Weekly Analysis

This module handles time period calculations and date range generation
for weekly analysis.
"""

import logging
from datetime import date, timedelta

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

logger.info(f"Weekly time utils logging initialized with level: {log_level_str}. Log file: {weekly_log_file}")

def get_weekly_time_ranges(reference_date=None):
    """
    Calculate recent and comparison periods for weekly analysis using ISO weeks (Monday-Sunday).
    The recent period is the last fully completed ISO week before the reference date.
    The comparison period is the same ISO week number in the previous year.
    """
    if reference_date is not None:
        if not isinstance(reference_date, date):
            raise TypeError("reference_date must be a datetime.date instance")
        anchor = reference_date
    else:
        anchor = date.today()

    # Find the last Sunday before (or on) the anchor date
    days_since_sunday = (anchor.weekday() + 1) % 7  # Monday=0, Sunday=6
    last_sunday = anchor - timedelta(days=days_since_sunday)
    # The week starts on the previous Monday
    last_monday = last_sunday - timedelta(days=6)

    # This is the most recent fully completed ISO week
    recent_period = {
        'start': last_monday,
        'end': last_sunday
    }

    # Get ISO week number and year
    iso_year, iso_week, _ = last_sunday.isocalendar()
    # Get the Monday and Sunday of the same ISO week last year
    try:
        comparison_monday = date.fromisocalendar(iso_year - 1, iso_week, 1)
        comparison_sunday = date.fromisocalendar(iso_year - 1, iso_week, 7)
    except ValueError:
        # Handle edge case for week 53 not existing in previous year
        comparison_monday = date.fromisocalendar(iso_year - 1, 52, 1)
        comparison_sunday = date.fromisocalendar(iso_year - 1, 52, 7)

    comparison_period = {
        'start': comparison_monday,
        'end': comparison_sunday
    }

    logger.info(f"Recent period (ISO week): {recent_period['start']} to {recent_period['end']}")
    logger.info(f"Comparison period (ISO week last year): {comparison_period['start']} to {comparison_period['end']}")

    return recent_period, comparison_period 
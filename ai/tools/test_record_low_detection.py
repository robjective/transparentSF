#!/usr/bin/env python3
"""
Test Script for Record Low Detection

This script demonstrates the record low detection functionality by testing it
on a specific metric or by running a quick analysis.
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
import pandas as pd

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .detect_record_lows import (
    get_all_active_metrics,
    get_time_series_data_for_metric,
    detect_record_low,
    create_record_low_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_record_low_detection():
    """Test the record low detection functionality."""
    logger.info("Starting record low detection test")
    
    # Get all active metrics
    metrics = get_all_active_metrics()
    
    if not metrics:
        logger.error("No active metrics found")
        return
    
    logger.info(f"Found {len(metrics)} active metrics")
    
    # Test with the first few metrics
    test_metrics = metrics[:3]  # Test first 3 metrics
    
    for metric in test_metrics:
        logger.info(f"\nTesting metric: {metric['metric_name']} (ID: {metric['id']})")
        
        # Get time series data for monthly periods
        time_series_df = get_time_series_data_for_metric(metric['id'], 'month')
        
        if time_series_df is None or time_series_df.empty:
            logger.warning(f"No time series data available for metric {metric['id']}")
            continue
        
        logger.info(f"Retrieved {len(time_series_df)} time series data points")
        
        # Detect record low
        record_low_info = detect_record_low(time_series_df)
        
        if record_low_info is None:
            logger.info("No record low analysis possible")
            continue
        
        if record_low_info['is_record_low']:
            logger.info("RECORD LOW DETECTED!")
            logger.info(f"Current value: {record_low_info['current_value']:,.0f}")
            logger.info(f"Historical min: {record_low_info['historical_min']:,.0f}")
            logger.info(f"Most recent period: {record_low_info['most_recent_period']}")
            logger.info(f"Last low period: {record_low_info['last_low_period']}")
            logger.info(f"Years since last low: {record_low_info['years_since_last_low']:.1f}")
            
            # Create summary
            summary = create_record_low_summary(metric, record_low_info, 'month')
            if summary:
                logger.info(f"Summary:\n{summary}")
        else:
            logger.info("No record low detected")
            logger.info(f"Current value: {record_low_info['current_value']:,.0f}")
            logger.info(f"Historical min: {record_low_info['historical_min']:,.0f}")

def test_specific_metric(metric_id):
    """Test record low detection for a specific metric."""
    logger.info(f"Testing record low detection for metric ID: {metric_id}")
    
    # Get all active metrics
    metrics = get_all_active_metrics()
    
    # Find the specific metric
    metric = None
    for m in metrics:
        if str(m['id']) == str(metric_id):
            metric = m
            break
    
    if not metric:
        logger.error(f"Metric ID {metric_id} not found")
        return
    
    logger.info(f"Found metric: {metric['metric_name']}")
    
    # Get time series data
    time_series_df = get_time_series_data_for_metric(metric['id'], 'month')
    
    if time_series_df is None or time_series_df.empty:
        logger.error(f"No time series data available for metric {metric['id']}")
        return
    
    logger.info(f"Retrieved {len(time_series_df)} time series data points")
    
    # Show some data statistics
    logger.info(f"Data range: {time_series_df['time_period'].min()} to {time_series_df['time_period'].max()}")
    logger.info(f"Value range: {time_series_df['numeric_value'].min():,.0f} to {time_series_df['numeric_value'].max():,.0f}")
    
    # Detect record low
    record_low_info = detect_record_low(time_series_df)
    
    if record_low_info is None:
        logger.info("No record low analysis possible")
        return
    
    if record_low_info['is_record_low']:
        logger.info("RECORD LOW DETECTED!")
        logger.info(f"Current value: {record_low_info['current_value']:,.0f}")
        logger.info(f"Historical min: {record_low_info['historical_min']:,.0f}")
        logger.info(f"Most recent period: {record_low_info['most_recent_period']}")
        logger.info(f"Last low period: {record_low_info['last_low_period']}")
        logger.info(f"Years since last low: {record_low_info['years_since_last_low']:.1f}")
        
        # Create summary
        summary = create_record_low_summary(metric, record_low_info, 'month')
        if summary:
            logger.info(f"Summary:\n{summary}")
    else:
        logger.info("No record low detected")
        logger.info(f"Current value: {record_low_info['current_value']:,.0f}")
        logger.info(f"Historical min: {record_low_info['historical_min']:,.0f}")

def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test record low detection')
    parser.add_argument('--metric-id', type=str, help='Test specific metric ID')
    
    args = parser.parse_args()
    
    if args.metric_id:
        test_specific_metric(args.metric_id)
    else:
        test_record_low_detection()

if __name__ == "__main__":
    main() 
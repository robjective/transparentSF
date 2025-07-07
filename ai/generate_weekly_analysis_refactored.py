"""
Refactored Weekly Analysis Generator

This is the main entry point for weekly analysis generation, now using
modular components from tools/analysis/weekly/.
"""

import os
import logging
import traceback
import argparse
import sys
from datetime import datetime
import schedule
import time

# Import from our modular components
from tools.analysis.weekly import (
    # Data processing
    load_json_file,
    find_metric_in_queries,
    extract_date_field_from_query,
    transform_query_for_weekly,
    detect_avg_aggregation,
    
    # Time utilities
    get_weekly_time_ranges,
    
    # Analysis engine
    process_weekly_analysis,
    run_weekly_analysis,
    
    # Report generation
    save_weekly_analysis,
    # generate_weekly_newsletter,
    
    # Scheduling
    scheduled_weekly_task
)

# Get script directory and ensure logs directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create module logger
logger = logging.getLogger(__name__)

# Log a message to confirm logging is set up
logger.info("==========================================================")
logger.info("Logging configured for generate_weekly_analysis_refactored.py")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info("==========================================================")

def main():
    """Main function to generate weekly metric analysis."""
    start_time = datetime.now()
    logger.info(f"========== WEEKLY ANALYSIS APP STARTING: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Parse command line arguments
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser(description='Generate weekly analysis for SF data metrics')
    parser.add_argument('--metric_id', help='ID of a specific metric to analyze', default=None)
    parser.add_argument('--metrics', help='Comma-separated list of metric IDs to analyze', default=None)
    parser.add_argument('--schedule', action='store_true', help='Run as a scheduled task every Thursday at 11am')
    parser.add_argument('--process-districts', action='store_true', 
                        help='Process and generate separate reports for each supervisor district if available')
    args = parser.parse_args()
    
    # Log parsed arguments
    logger.info(f"Arguments: metric_id={args.metric_id}, metrics={args.metrics}, schedule={args.schedule}, process_districts={args.process_districts}")
    
    if args.schedule:
        logger.info("Setting up scheduled task for weekly analysis (Thursdays at 11am)")
        
        # Schedule the job to run every Thursday at 11am
        schedule.every().thursday.at("11:00").do(scheduled_weekly_task)
        
        # Keep the script running
        logger.info("Scheduler is running, press Ctrl+C to exit")
        
        try:
            while True:
                pending_jobs = len(schedule.get_jobs())
                logger.debug(f"Checking pending jobs: {pending_jobs}")
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        # Run immediately for specified metrics
        logger.info("Running in immediate execution mode")
        metrics_list = []
        
        if args.metric_id:
            metrics_list.append(args.metric_id)
            logger.info(f"Added single metric from --metric_id: {args.metric_id}")
        
        if args.metrics:
            added_metrics = args.metrics.split(',')
            metrics_list.extend(added_metrics)
            logger.info(f"Added {len(added_metrics)} metrics from --metrics parameter")
        
        # If no metrics specified, use default set in run_weekly_analysis
        if not metrics_list:
            logger.info("No specific metrics requested, using default set")
        else:
            logger.info(f"Prepared to analyze {len(metrics_list)} metrics: {', '.join(metrics_list)}")
        
        try:
            # Run the analysis
            process_start = datetime.now()
            logger.info(f"Starting analysis process at {process_start.strftime('%H:%M:%S')}")
            
            results = run_weekly_analysis(
                metrics_list=metrics_list if metrics_list else None,
                process_districts=args.process_districts
            )
            
            # Generate a newsletter
            if results:
                logger.info(f"Analysis completed with {len(results)} results")
                # Newsletter generation temporarily disabled
                # logger.info(f"Analysis completed with {len(results)} results, generating newsletter")
                # newsletter_path = generate_weekly_newsletter(results)
                
                # if newsletter_path:
                #     logger.info(f"Analysis completed, newsletter saved to {newsletter_path}")
                # else:
                #     logger.warning("Analysis completed but no newsletter was generated")
                logger.info("Analysis completed successfully (newsletter generation disabled)")
            else:
                logger.warning("Analysis completed but returned no results")
                # logger.warning("Analysis completed but returned no results, skipping newsletter generation")
                
        except Exception as e:
            logger.error(f"Error during analysis execution: {str(e)}")
            logger.error(traceback.format_exc())
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"========== WEEKLY ANALYSIS APP COMPLETE: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    logger.info(f"Total app execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

if __name__ == "__main__":
    main() 
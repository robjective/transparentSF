"""
Weekly Analysis Module

This module provides tools for generating weekly analysis reports for SF data metrics.
It includes data processing, chart generation, anomaly detection, and report creation.
"""

from .data_processor import (
    find_metric_in_queries,
    extract_date_field_from_query,
    transform_query_for_weekly,
    detect_avg_aggregation
)

from .time_utils import (
    get_weekly_time_ranges
)

from .analysis_engine import (
    process_weekly_analysis,
    run_weekly_analysis
)

from .report_generator import (
    save_weekly_analysis,
    # generate_weekly_newsletter
)

# Weekly scheduler removed - no longer automatically scheduled

__all__ = [
    # Data processing
    'find_metric_in_queries', 
    'extract_date_field_from_query',
    'transform_query_for_weekly',
    'detect_avg_aggregation',
    
    # Time utilities
    'get_weekly_time_ranges',
    
    # Analysis engine
    'process_weekly_analysis',
    'run_weekly_analysis',
    
    # Report generation
    'save_weekly_analysis',
    # 'generate_weekly_newsletter',
    
    # Scheduling - removed
] 
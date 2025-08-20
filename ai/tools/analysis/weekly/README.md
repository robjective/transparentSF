# Weekly Analysis Module

This module provides a refactored, modular approach to generating weekly analysis reports for SF data metrics. The code has been broken down into logical components for better maintainability and consistency.

## Module Structure

```
ai/tools/analysis/weekly/
├── __init__.py              # Main module exports
├── data_processor.py        # Data loading and query processing
├── time_utils.py           # Time period calculations
├── analysis_engine.py      # Core analysis logic
├── report_generator.py     # File saving and newsletter generation
├── scheduler.py            # Scheduled task handling
└── README.md              # This file
```

## Components

### 1. Data Processor (`data_processor.py`)
Handles data loading, metric discovery, and query transformation:
- `load_json_file()` - Load JSON configuration files
- `find_metric_in_queries()` - Find metrics in the database
- `extract_date_field_from_query()` - Extract date fields from SQL queries
- `transform_query_for_weekly()` - Transform queries for weekly analysis
- `detect_avg_aggregation()` - Detect AVG() aggregation in queries

### 2. Time Utilities (`time_utils.py`)
Manages time period calculations:
- `get_weekly_time_ranges()` - Calculate recent and comparison periods for weekly analysis

### 3. Analysis Engine (`analysis_engine.py`)
Contains the core analysis processing logic:
- `process_weekly_analysis()` - Process analysis for a single metric
- `run_weekly_analysis()` - Run analysis for multiple metrics

### 4. Report Generator (`report_generator.py`)
Handles saving results and generating newsletters:
- `save_weekly_analysis()` - Save analysis results to files
- `generate_weekly_newsletter()` - Generate weekly newsletter (temporarily disabled)

### 5. Scheduler (`scheduler.py`)
Manages scheduled tasks:
- `scheduled_weekly_task()` - Scheduled weekly analysis task

## Usage

### Main Entry Point
The refactored main file is `ai/generate_weekly_analysis_refactored.py`:

```python
from tools.analysis.weekly import run_weekly_analysis
# from tools.analysis.weekly import generate_weekly_newsletter

# Run analysis for specific metrics
results = run_weekly_analysis(metrics_list=["1", "2", "3"], process_districts=True)

# Generate newsletter (temporarily disabled)
# newsletter_path = generate_weekly_newsletter(results)
```

### Command Line Usage
```bash
# Run for specific metrics
python ai/generate_weekly_analysis_refactored.py --metric_id 1 --process-districts

# Run for multiple metrics
python ai/generate_weekly_analysis_refactored.py --metrics "1,2,3,4" --process-districts

# Run as scheduled task
python ai/generate_weekly_analysis_refactored.py --schedule
```

## Benefits of Refactoring

1. **Modularity**: Each component has a single responsibility
2. **Maintainability**: Easier to locate and fix issues
3. **Testability**: Individual components can be tested in isolation
4. **Reusability**: Components can be imported and used in other modules
5. **Consistency**: Follows the same pattern as other analysis modules
6. **Documentation**: Each module is well-documented with clear purposes

## Migration from Original

The original `generate_weekly_analysis.py` file has been refactored into:
- **1598 lines** → **6 modular files** (~200-400 lines each)
- **Monolithic structure** → **Logical separation of concerns**
- **Hard to maintain** → **Easy to extend and modify**

## Next Steps

This refactoring pattern can be applied to:
- `generate_metric_analysis.py` → `tools/analysis/metric/`
- `monthly_report.py` → `tools/analysis/monthly/`

## Dependencies

The module depends on:
- `tools.data_fetcher` - For dataset operations
- `tools.genChart` - For chart generation
- `tools.anomaly_detection` - For anomaly detection
- `tools.db_utils` - For database connections

## Configuration

Default metrics are defined in `analysis_engine.py`:
```python
DEFAULT_METRICS = [
    "1",   # Total Police Incidents 
    "2",   # Arrests Presented
    "3",   # Arrests Booked
    "4",   # Police Response Times
    "5",   # 311 Cases
    "6",   # DPW Service Requests
    "8",   # Building Permits
    "14"   # Public Works Projects
]
```

## Output Structure

Analysis results are saved to:
```
ai/output/weekly/
├── 0/                    # Citywide analysis
│   ├── 1.md
│   └── ...
├── 1/                    # District 1 analysis
│   ├── 1.md
│   └── ...
└── weekly_newsletter_YYYY-MM-DD.md
``` 
# TransparentSF Metrics Management System

This document describes the new metrics management system for TransparentSF, which allows the explainer agent to query, edit, and add new metrics to the database.

## Overview

The metrics system consists of several components:

1. **Database Table**: `metrics` table in PostgreSQL
2. **Migration Script**: `migrate_dashboard_to_metrics.py` - imports existing dashboard queries
3. **Core Manager**: `metrics_manager.py` - low-level database operations
4. **Agent Interface**: `explainer_metrics_tools.py` - simplified interface for the explainer agent
5. **Database Schema**: Updated `init_postgres_db.py` with metrics table

### District-Level Data Requirements

When creating metrics that should have district-level data (broken down by supervisor district), you MUST include `supervisor_district` in both the `metric_query` and `ytd_query`:

1. **For metric_query**: Add `, supervisor_district` to the SELECT clause and `GROUP BY supervisor_district` at the end
2. **For ytd_query**: Add `, supervisor_district` to the SELECT clause and `, supervisor_district` to the GROUP BY clause

**CRITICAL**: If you include `supervisor_district` in `location_fields` but don't include it in the queries, the metric will only generate citywide data (district 0) instead of district-level data for all 12 districts.

**Example of CORRECT district-level queries:**
```sql
-- metric_query
SELECT 'Metric Name' as label, COUNT(*) as this_year, supervisor_district GROUP BY supervisor_district

-- ytd_query  
SELECT date_trunc_ymd(date_field) as date, COUNT(*) as value, supervisor_district WHERE date_field >= last_year_start GROUP BY date, supervisor_district ORDER BY date
```

**Example of INCORRECT queries (will only generate citywide data):**
```sql
-- metric_query ❌ Missing supervisor_district
SELECT 'Metric Name' as label, COUNT(*) as this_year GROUP BY label

-- ytd_query ❌ Missing supervisor_district
SELECT date_trunc_ymd(date_field) as date, COUNT(*) as value WHERE date_field >= last_year_start GROUP BY date ORDER BY date
```

The system checks for `'supervisor_district'` in the query results to determine if district-level data is available. Without it in the queries, the system will log "Query has district data: False" and only create citywide metrics.

## Database Schema

The `metrics` table contains the following fields:

```sql
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL,                    -- Display name (e.g., "🚨 Violent Crime Incidents")
    metric_key TEXT UNIQUE NOT NULL,              -- Unique identifier (e.g., "violent_crime_incidents")
    category TEXT NOT NULL,                       -- Main category (e.g., "crime", "safety", "economy")
    subcategory TEXT,                             -- Subcategory (e.g., "Crime", "fire", "housing")
    endpoint TEXT NOT NULL,                       -- DataSF endpoint ID
    summary TEXT,                                 -- Brief description
    definition TEXT,                              -- Detailed definition
    data_sf_url TEXT,                            -- URL to DataSF dataset
    ytd_query TEXT,                              -- Year-to-date query for time series
    metric_query TEXT,                           -- Query for metric calculations
    dataset_title TEXT,                          -- Title of source dataset
    dataset_category TEXT,                       -- Category of source dataset
    show_on_dash BOOLEAN DEFAULT TRUE,           -- Whether to show on dashboard
    item_noun TEXT DEFAULT 'Items',              -- Noun for items (e.g., "Cases", "Incidents")
    location_fields JSONB DEFAULT '[]'::jsonb,   -- Location field definitions
    category_fields JSONB DEFAULT '[]'::jsonb,   -- Category field definitions
    metadata JSONB DEFAULT '{}'::jsonb,          -- Additional metadata
    is_active BOOLEAN DEFAULT TRUE,              -- Whether metric is active
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Setup and Migration

### 1. Initialize Database

First, run the database initialization script to create all tables including the new metrics table:

```bash
cd ai/tools
python init_postgres_db.py
```

### 2. Migrate Existing Dashboard Queries

Run the migration script to import all existing dashboard queries into the metrics table:

```bash
python migrate_dashboard_to_metrics.py
```

This will:
- Load data from `ai/data/dashboard/dashboard_queries_enhanced.json`
- Parse and flatten the nested structure
- Insert 19 metrics into the database
- Create unique keys for each metric

## Usage for Explainer Agent

The explainer agent should use the functions in `explainer_metrics_tools.py`. Here are the main functions:

### Querying Metrics

```python
from explainer_metrics_tools import query_metrics, get_metric_details

# Get all crime metrics
crime_metrics = query_metrics(category="crime")

# Search for police-related metrics
police_metrics = query_metrics(search_term="police")

# Get all dashboard metrics
dashboard_metrics = query_metrics(dashboard_only=True)

# Get metric details by ID or key
metric = get_metric_details(1)  # By ID
metric = get_metric_details("total_police_incidents")  # By key
```

### Managing Metrics

```python
from explainer_metrics_tools import create_new_metric, edit_metric, disable_metric

# Create a new metric
result = create_new_metric(
    name="🚗 Vehicle Thefts",
    key="vehicle_thefts",
    category="crime",
    subcategory="property",
    endpoint="wg3w-h783",
    summary="Count of reported vehicle theft incidents",
    definition="Vehicle thefts include all reported incidents...",
    show_on_dash=True
)

# Edit an existing metric
edit_metric(1, {
    "summary": "Updated summary text",
    "show_on_dash": False
})

# Disable a metric (soft delete)
disable_metric("old_metric_key")
```

### Getting Overviews

```python
from explainer_metrics_tools import get_metrics_overview, list_categories

# Get system overview
overview = get_metrics_overview()
print(f"Total metrics: {overview['summary']['total_metrics']}")

# List all categories
categories = list_categories()
for cat, data in categories["categories"].items():
    print(f"{cat}: {data['total_metrics']} metrics")
```

## Available Functions

### Core Query Functions
- `query_metrics()` - Main function for finding metrics with filters
- `get_metric_details()` - Get detailed info about a specific metric
- `list_categories()` - Get all categories and subcategories
- `get_dashboard_metrics()` - Get metrics shown on dashboard
- `get_metrics_overview()` - Get system statistics

### Management Functions
- `create_new_metric()` - Add a new metric
- `edit_metric()` - Update existing metric
- `disable_metric()` - Deactivate metric (soft delete)
- `enable_metric()` - Reactivate metric

### Utility Functions
- `find_metrics_by_endpoint()` - Find metrics using specific DataSF endpoint
- `get_crime_metrics()` - Get all crime metrics
- `get_safety_metrics()` - Get all safety metrics
- `get_economy_metrics()` - Get all economy metrics
- `format_metric_info()` - Format metric for display
- `display_metrics_list()` - Format list of metrics

## Current Metrics

After migration, the system contains 19 metrics across 3 main categories:

### Crime (7 metrics)
- 👮 Total Police Incidents
- 🚨 Violent Crime Incidents  
- 🏠 Property Crime Incidents
- 💊 Drug Crime Incidents
- 🔒 Jail Bookings
- 🚓 Arrests Presented to DA
- ⚖️ DA Convictions

### Safety (7 metrics)
- 🚒 Fire Incidents YTD
- 💔 Fire Fatalities YTD
- 🚑 911 Response (minutes) - Danger to life
- 🚑 911 Response (minutes) - Danger to property
- 🚑 911 Response (minutes) - No danger to life or property
- 📞 311 Cases
- ⛺ Encampment Calls

### Economy (5 metrics)
- 🏠 New Housing Units Completed
- 🏢 New Business Registrations
- 🚫 Business Closures
- 🛍️ New Retail Registrations
- 🏪 Retail Closures

## Testing

Run the comprehensive test suite to verify functionality:

```bash
python test_metrics_system.py
```

This tests:
- Basic querying and filtering
- Search functionality
- Metric details retrieval
- Endpoint-based searches
- Metric creation and editing
- Formatting functions

## Integration with Existing Systems

The metrics system integrates with existing TransparentSF components:

1. **Database**: Uses existing PostgreSQL connection utilities in `db_utils.py`
2. **Dashboard**: Metrics marked with `show_on_dash=True` can be used by dashboard
3. **Anomaly Detection**: Metrics provide metadata for anomaly detection queries
4. **Reporting**: Metrics can be referenced in monthly reports

## Best Practices

### For the Explainer Agent

1. **Always check return status**: All functions return `{"status": "success/error", ...}`
2. **Use descriptive keys**: When creating metrics, use clear, unique keys
3. **Include proper metadata**: Add summaries, definitions, and field information
4. **Test before deploying**: Use the test functions to verify new metrics work

### For Metric Creation

1. **Unique keys**: Ensure metric keys are unique and descriptive
2. **Proper categorization**: Use existing categories when possible
3. **Complete information**: Include summaries, definitions, and DataSF URLs
4. **Field definitions**: Add location_fields and category_fields for enhanced queries

## Troubleshooting

### Common Issues

1. **Database connection errors**: Check environment variables for PostgreSQL connection
2. **Duplicate key errors**: Ensure metric keys are unique when creating new metrics
3. **Missing metrics**: Run migration script if metrics table is empty
4. **Permission errors**: Ensure database user has proper permissions

### Environment Variables

The system uses these environment variables:
- `POSTGRES_HOST` (default: localhost)
- `POSTGRES_PORT` (default: 5432)
- `POSTGRES_DB` (default: transparentsf)
- `POSTGRES_USER` (default: postgres)
- `POSTGRES_PASSWORD` (default: postgres)

## Future Enhancements

Potential improvements to the metrics system:

1. **Versioning**: Track changes to metric definitions over time
2. **Dependencies**: Track relationships between metrics
3. **Validation**: Validate queries against DataSF endpoints
4. **Caching**: Cache frequently accessed metrics
5. **API Integration**: Direct integration with DataSF API for validation
6. **Bulk Operations**: Support for bulk metric updates
7. **Export/Import**: Export metrics to JSON for backup/sharing

## Files Overview

- `init_postgres_db.py` - Database initialization with metrics table
- `migrate_dashboard_to_metrics.py` - One-time migration script
- `metrics_manager.py` - Core database operations
- `explainer_metrics_tools.py` - Simplified interface for agents
- `test_metrics_system.py` - Comprehensive test suite
- `METRICS_SYSTEM_README.md` - This documentation

The metrics system provides a robust foundation for managing and querying metrics in TransparentSF, enabling the explainer agent to dynamically work with city data metrics. 
# Stale Data Detection Implementation

## Overview
Added stale data detection to the anomaly analyzer to flag metrics that have a `most_recent_data_date` that is lower than the last day of the "recent_data" period available in the metadata column of `time_series_metadata`.

## Changes Made

### 1. Modified `ai/anomalyAnalyzer.py`

#### Added Stale Data Check Logic
- **Location**: `get_top_metric_changes` function, after processing each chart's periods
- **Logic**: 
  - Query the `most_recent_data_date` from the `metrics` table using the chart's `object_id`
  - Query the `metadata` column from `time_series_metadata` table to get the `recent_period.end` date
  - Compare the two dates to detect stale data
  - Calculate the number of days the data is stale

#### Date Parsing
- **String to Date Conversion**: Handles both `YYYY-MM-DD` and `YYYY-MM` formats
- **JSON Metadata Parsing**: Safely extracts `recent_period.end` from the JSONB metadata column
- **Error Handling**: Graceful fallback if date parsing fails

#### Warning Generation
- **Format**: `"Data is {days_stale} days stale (most recent: {most_recent_data_date}, expected: {recent_period_end})"`
- **Logging**: Warning messages logged for each detected stale data case
- **Result Addition**: Stale data warning added to each result object

#### Response Enhancement
- **Individual Results**: Each metric result includes a `stale_data_warning` field (null if no warning)
- **Summary Count**: Response includes a `stale_data_warnings` count field
- **JSON Compliance**: Ensured all date objects are properly serialized

### 2. Database Schema Requirements

The implementation relies on existing database schema:

#### `metrics` table
- `most_recent_data_date` (DATE): The most recent data date available for each metric
- `id` (INTEGER): Primary key used to link with charts

#### `time_series_metadata` table  
- `metadata` (JSONB): Contains period information including `recent_period.end`
- `chart_id` (INTEGER): Primary key for linking with time series data
- `object_id` (TEXT): Links to the metrics table

### 3. API Response Changes

#### New Fields Added
```json
{
  "status": "success",
  "count": 10,
  "period_type": "month",
  "object_id": "123",
  "district": "0",
  "report_date": "2024-12-01",
  "stale_data_warnings": 2,  // NEW: Count of metrics with stale data
  "positive_changes": [
    {
      "metric_name": "Police Incidents",
      "percent_change": 15.5,
      "stale_data_warning": "Data is 30 days stale (most recent: 2024-11-01, expected: 2024-12-01)",  // NEW
      // ... other fields
    }
  ],
  "negative_changes": [
    // ... similar structure
  ]
}
```

## Usage

### Detecting Stale Data
The system automatically detects stale data when:
1. A metric's `most_recent_data_date` is older than the `recent_period.end` date
2. The difference is calculated in days
3. A warning message is generated and included in the response

### Example Scenarios

#### Scenario 1: Fresh Data
- `most_recent_data_date`: 2024-12-31
- `recent_period.end`: 2024-12-01
- **Result**: No warning (data is current)

#### Scenario 2: Stale Data
- `most_recent_data_date`: 2024-11-15  
- `recent_period.end`: 2024-12-31
- **Result**: Warning: "Data is 46 days stale (most recent: 2024-11-15, expected: 2024-12-31)"

#### Scenario 3: Same Date
- `most_recent_data_date`: 2024-12-15
- `recent_period.end`: 2024-12-15
- **Result**: No warning (data is current)

## Benefits

1. **Data Quality Awareness**: Users can identify when metrics are based on outdated data
2. **Transparency**: Clear indication of data freshness in API responses
3. **Monitoring**: Easy to track which metrics need data updates
4. **Decision Making**: Helps users understand the reliability of metric changes

## Error Handling

- **Missing Data**: Gracefully handles cases where `most_recent_data_date` or metadata is null
- **Date Parsing**: Robust parsing of different date formats with fallback handling
- **Database Errors**: Logs warnings but continues processing other metrics
- **Invalid Object IDs**: Skips stale data check for invalid or missing object IDs

## Testing

The implementation includes comprehensive error handling and has been tested with various scenarios:
- Fresh data detection
- Stale data detection  
- Different date formats
- Missing or invalid data
- JSON metadata parsing

## Future Enhancements

Potential improvements could include:
1. Configurable stale data thresholds
2. Different warning levels based on staleness duration
3. Automatic data refresh triggers
4. Dashboard indicators for stale data
5. Email alerts for severely stale data 
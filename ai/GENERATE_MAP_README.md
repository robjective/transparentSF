# Generate Map Functionality

## Overview

The generate map functionality allows users to create interactive maps from anomaly data in the anomaly analyzer. When a user clicks on an anomaly and then clicks the "Generate Map" button, the system will:

1. Analyze the data points available for the metric
2. Determine the appropriate map type based on the number of data points
3. Generate either a locator map (≤100 points) or symbol chart (>100 points)
4. Display the map with gradient colors based on date

## Implementation Details

### API Endpoint

**POST** `/anomaly-analyzer/api/generate-map`

**Request Body:**
```json
{
    "metric_id": "string",
    "district": "string", 
    "period_type": "string",
    "anomaly_id": "string" (optional)
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Successfully generated map for [metric_name]",
    "map_id": "string",
    "edit_url": "string",
    "publish_url": "string", 
    "data_points": "number",
    "map_type": "locator|symbol"
}
```

### Map Type Decision Logic

- **≤100 data points**: Creates a locator map with actual data points
- **>100 data points**: Creates a symbol chart with gradient colors based on date

### Data Processing

The system:

1. **Fetches metric information** from the database using the metric ID
2. **Extracts the query URL** from the metric
3. **Modifies the query** to get detailed data for the last two periods (e.g., last month + this month)
4. **Fetches the data** from DataSF API
5. **Processes location data** in various formats:
   - DataSF location format: `{"type": "Point", "coordinates": [lon, lat]}`
   - Intersection data: `{"type": "Point", "coordinates": [lon, lat]}`
   - Direct coordinates: `lat`/`lon` or `latitude`/`longitude` fields
   - Point field: `point` field with coordinate data
6. **Validates coordinates** to ensure they're in the San Francisco area
7. **Creates tooltips** with date and value information
8. **Generates the map** using the existing `generate_map` function

### Location Data Formats Supported

The system can handle various location data formats:

```python
# DataSF location format
{"location": {"type": "Point", "coordinates": [lon, lat]}}

# Intersection format  
{"intersection": {"type": "Point", "coordinates": [lon, lat]}}

# Direct coordinates
{"lat": 37.7749, "lon": -122.4194}

# Latitude/longitude fields
{"latitude": 37.7749, "longitude": -122.4194}

# Point field
{"point": {"type": "Point", "coordinates": [lon, lat]}}

# Direct coordinates array
{"coordinates": [lon, lat]}
```

### Date Field Detection

The system automatically detects date fields in queries using regex patterns:

1. **Primary pattern**: `date_trunc('period', field_name)`
2. **Fallback pattern**: `date_trunc('field_name')`
3. **Common field names**: `date`, `datetime`, `created_date`, `incident_date`, `report_date`

### Coordinate Validation

All coordinates are validated to ensure they're within the San Francisco area:
- Latitude: 37.6° to 37.9°
- Longitude: -122.6° to -122.2°

### Color Coding

Maps use sequential color palettes to show date progression:
- **Locator maps**: Points colored by date using sequential colors
- **Symbol charts**: Symbols colored by date using sequential colors

## Frontend Integration

### JavaScript Implementation

The generate map button is implemented in the anomaly analyzer template:

```javascript
// Generate map button click handler
newGenerateMapBtn.addEventListener('click', function() {
    // Show loading state
    const originalText = newGenerateMapBtn.innerHTML;
    newGenerateMapBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Map...';
    newGenerateMapBtn.disabled = true;
    
    // Prepare request data
    const requestData = {
        metric_id: data.id || data.object_id,
        district: data.district || '0',
        period_type: periodSelect.value || 'month'
    };
    
    // Add anomaly_id if this is an anomaly
    if (data.anomaly_id) {
        requestData.anomaly_id = data.anomaly_id;
    }
    
    // Make API call
    fetch('/anomaly-analyzer/api/generate-map', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(result => {
        if (result.status === 'success') {
            // Show success message
            const mapType = result.map_type === 'locator' ? 'Locator Map' : 'Symbol Chart';
            const message = `Successfully generated ${mapType} with ${result.data_points} data points!`;
            alert(message);
            
            // Optionally open map in new tab
            if (result.publish_url) {
                const openMap = confirm('Would you like to open the map in a new tab?');
                if (openMap) {
                    window.open(result.publish_url, '_blank');
                }
            }
        } else {
            alert(`Error generating map: ${result.message}`);
        }
    })
    .catch(error => {
        alert(`Error generating map: ${error.message}`);
    })
    .finally(() => {
        // Reset button state
        newGenerateMapBtn.innerHTML = originalText;
        newGenerateMapBtn.disabled = false;
    });
});
```

## Testing

A test script is provided at `test_generate_map.py` to verify the functionality:

```bash
# Run the test script
python test_generate_map.py
```

The test script:
1. Tests metric retrieval functionality
2. Tests the generate map API endpoint with various parameters
3. Provides detailed output for debugging

## Error Handling

The system includes comprehensive error handling:

- **Missing metric ID**: Returns 400 error
- **Metric not found**: Returns 404 error  
- **No query URL**: Returns 400 error
- **Date field detection failure**: Returns 400 error
- **Data fetching failure**: Returns 500 error
- **No location data**: Returns 400 error
- **Map generation failure**: Returns 500 error

## Dependencies

The implementation depends on:

- `tools.metrics_manager`: For retrieving metric information
- `tools.data_fetcher`: For fetching data from DataSF API
- `tools.generate_map`: For creating the actual maps
- FastAPI: For the web framework
- Datawrapper API: For map generation

## Configuration

The system uses environment variables for configuration:

- `DATAWRAPPER_API_KEY`: Required for map generation
- `POSTGRES_HOST`, `POSTGRES_PORT`, etc.: For database access

## Future Enhancements

Potential improvements:

1. **Caching**: Cache generated maps to avoid regenerating identical maps
2. **Batch processing**: Allow generating multiple maps at once
3. **Custom date ranges**: Allow users to specify custom date ranges
4. **Advanced filtering**: Add more filtering options (time of day, etc.)
5. **Map templates**: Allow users to choose from different map styles
6. **Export options**: Add export to PDF, image, etc. 
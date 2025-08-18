# Map Generator for TransparentSF

This module provides a new map generation interface that uses Mapbox instead of Datawrapper for map visualizations.

## Features

- **Interactive Map Interface**: Modern web interface with dropdown controls
- **Mapbox Integration**: Uses Mapbox GL JS for high-quality, interactive maps
- **Multiple Map Types**: Supports district maps and point maps
- **Real-time Data**: Connects to your existing TransparentSF database
- **Responsive Design**: Works on desktop and mobile devices

## Setup

### 1. Environment Variables

Create a `.env` file in the `ai/` directory with the following variables:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=transparentsf
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Mapbox Configuration (Required for new maps)
MAPBOX_ACCESS_TOKEN=pk.your_mapbox_access_token_here

# Datawrapper Configuration (for existing maps)
DATAWRAPPER_API_KEY=your_datawrapper_api_key_here
DATAWRAPPER_REFERENCE_CHART=j5vON

# Logging Configuration
LOG_LEVEL=INFO
```

### 2. Get a Mapbox Access Token

1. Go to [Mapbox](https://www.mapbox.com/) and create an account
2. Navigate to your account dashboard
3. Create a new access token or use the default public token
4. Copy the token and add it to your `.env` file

### 3. Database Setup

The map generator uses the existing `maps` table in your database. Make sure it has the following structure:

```sql
CREATE TABLE IF NOT EXISTS maps (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    location_data JSONB,
    metadata JSONB,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Usage

### Accessing the Map Generator

1. Start your TransparentSF application
2. Navigate to `/map-generator` in your browser
3. Use the dropdown controls to configure your map:
   - **Metric**: Select from available metrics in your database
   - **Anomaly Type**: Choose specific anomaly types (populated after metric selection)
   - **District**: Select citywide, all districts, or specific districts
   - **Period Type**: Choose monthly, weekly, yearly, or year-to-date

### Generating Maps

1. Configure your map settings using the dropdowns
2. Click "Generate Map" to create a new visualization
3. The map will be stored in your database and displayed
4. Use the action buttons to view, edit, or share your map

### Viewing Maps

- Maps are accessible at `/map/{map_id}`
- Each map has an interactive interface with:
  - Navigation controls
  - Fullscreen mode
  - Information overlay
  - Legend (if applicable)
  - Popup details on click

## API Endpoints

### GET `/map-generator`
Serves the map generator interface.

### GET `/api/metrics`
Returns all available metrics for the dropdown.

### GET `/api/anomalies/{metric_id}`
Returns anomalies for a specific metric.

### POST `/api/generate-map`
Generates a new map with the provided parameters.

### GET `/map/{map_id}`
Displays a specific map by ID.

### GET `/api/maps`
Returns recent maps, optionally filtered by metric.

## Map Types

### District Maps (`supervisor_district`)
- Shows data by San Francisco supervisor districts
- Color-coded based on values
- Interactive hover and click effects

### Point Maps (`point`)
- Shows individual data points on the map
- Supports custom markers and colors
- Popup information on click

## Data Format

The map generator expects data in the following formats:

### District Data
```json
[
  {"district": "1", "value": 120, "title": "District 1"},
  {"district": "2", "value": 85, "title": "District 2"}
]
```

### Point Data
```json
[
  {
    "lat": 37.7749,
    "lon": -122.4194,
    "title": "Location Name",
    "value": 100,
    "description": "Additional information"
  }
]
```

## Customization

### Styling
- Maps use the TransparentSF color scheme
- Dark mode support is included
- Responsive design for mobile devices

### Colors
- District maps use a green color scale by default
- Point maps can use custom colors or series-based coloring
- Colors are defined in CSS custom properties

### Map Styles
- Uses Mapbox Light style for better data visibility
- Centered on San Francisco with appropriate zoom level
- Navigation and fullscreen controls included

## Troubleshooting

### Map Not Loading
1. Check that your Mapbox access token is valid
2. Verify the token is set in your `.env` file
3. Check browser console for JavaScript errors

### No Data Displayed
1. Verify your database connection
2. Check that the selected metric has data
3. Review the map generation logs

### Performance Issues
1. Limit the number of data points for large datasets
2. Use appropriate map types for your data
3. Consider data aggregation for district maps

## Migration from Datawrapper

The new map generator is designed to work alongside existing Datawrapper maps:

- Existing maps continue to work unchanged
- New maps use Mapbox for better performance and features
- Both systems share the same database table
- Gradual migration is supported

## Development

### Adding New Map Types
1. Update the `process_dataset_for_map` function
2. Add new map type handling in the Mapbox JavaScript
3. Update the UI to support new options

### Custom Styling
1. Modify the CSS in `map_generator.html` and `map.html`
2. Update color schemes in the JavaScript
3. Add new Mapbox layers as needed

### Database Schema Changes
1. Update the `maps` table structure if needed
2. Modify the `generate_mapbox_map` function
3. Update the map display logic

## Support

For issues or questions:
1. Check the application logs in `ai/logs/`
2. Verify your environment configuration
3. Test with the provided example data
4. Review the Mapbox documentation for advanced features

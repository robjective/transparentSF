# Enhanced Map Generation with Series Support

## Overview

The map generation system has been enhanced to support **different colored series**, allowing for better data visualization and categorization on maps. This enhancement enables you to group markers by categories and automatically assign distinct colors to each group, complete with legends and enhanced tooltips.

## Key Features

### 🎨 Multiple Colored Series
- Group markers into different categories with distinct colors
- Automatic color assignment based on data fields
- Consistent color mapping across map updates

### 🎯 Predefined Color Palettes
- **Categorical**: 12 distinct colors for general categorization
- **Status**: 5 colors for status indication (Green, Amber, Red, Blue, Purple)
- **Priority**: 4 colors for priority levels (Red, Orange, Green, Grey)
- **Sequential**: 9 colors for graduated/sequential data

### 🎨 Custom Color Support
- Define your own color palettes
- Full hex color code support
- Automatic cycling for more series than colors

### 📊 Automatic Legend Generation
- Legends automatically created for series maps
- Shows all series with their assigned colors
- Positioned optimally on the map

### 💬 Enhanced Tooltips
- Series information automatically added to tooltips
- Maintains existing tooltip content
- Clear category identification

## Usage Examples

### Basic Series Map

```python
from ai.tools.generate_map import generate_map

# Sample data with series field
location_data = [
    {
        "title": "Police Station - Central",
        "lat": 37.7749,
        "lon": -122.4194,
        "tooltip": "Central Police Station",
        "series": "Police Stations"
    },
    {
        "title": "Fire Station - Engine 1", 
        "lat": 37.7849,
        "lon": -122.4094,
        "tooltip": "Fire Station Engine 1",
        "series": "Fire Stations"
    },
    {
        "title": "Hospital - UCSF",
        "lat": 37.7627,
        "lon": -122.4581,
        "tooltip": "UCSF Medical Center", 
        "series": "Hospitals"
    }
]

# Generate map with series
result = generate_map(
    context_variables={},
    map_title="SF Public Services",
    map_type="point",
    location_data=location_data,
    series_field="series",           # Field containing series data
    color_palette="categorical"      # Use predefined categorical colors
)
```

### Status-Based Coloring

```python
status_data = [
    {"title": "Active Site", "lat": 37.7749, "lon": -122.4194, "series": "Active"},
    {"title": "Pending Site", "lat": 37.7849, "lon": -122.4094, "series": "Pending"}, 
    {"title": "Inactive Site", "lat": 37.7627, "lon": -122.4581, "series": "Inactive"}
]

result = generate_map(
    context_variables={},
    map_title="Site Status Map",
    map_type="point",
    location_data=status_data,
    series_field="series",
    color_palette="status"  # Green, Amber, Red, Blue, Purple
)
```

### Custom Color Palette

```python
# Define custom colors
custom_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]  # Red, Green, Blue, Yellow

result = generate_map(
    context_variables={},
    map_title="Custom Colors Map",
    map_type="point",
    location_data=location_data,
    series_field="series",
    color_palette=custom_colors
)
```

### Address-Based Series Map

```python
address_data = [
    {
        "title": "City Hall",
        "address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA",
        "series": "Government",
        "tooltip": "San Francisco City Hall"
    },
    {
        "title": "SFPD Central Station", 
        "address": "766 Vallejo St, San Francisco, CA",
        "series": "Police",
        "tooltip": "Central Police Station"
    }
]

result = generate_map(
    context_variables={},
    map_title="SF Key Locations",
    map_type="address",  # Uses address geocoding
    location_data=address_data,
    series_field="series",
    color_palette="categorical"
)
```

## Function Parameters

### Enhanced `generate_map()` Function

```python
def generate_map(
    context_variables,
    map_title,
    map_type,
    location_data,
    map_metadata=None,
    reference_chart_id=None,
    metric_id=None,
    group_field=None,
    series_field=None,        # NEW: Field name for series grouping
    color_palette=None        # NEW: Color palette for series
):
```

#### New Parameters

- **`series_field`** (str, optional): Field name in your data to use for series grouping (e.g., 'series', 'category', 'type')
- **`color_palette`** (list or str, optional): 
  - String: Name of predefined palette ('categorical', 'status', 'priority', 'sequential')
  - List: Custom list of hex color codes

## Available Color Palettes

### Categorical (Default)
12 distinct colors for general categorization:
- `#ad35fa` (Bright Purple - primary brand color)
- `#FF6B5A` (Warm Coral)
- `#4A7463` (Spruce Green)
- `#71B2CA` (Sky Blue)
- `#FFC107` (Amber)
- `#9C27B0` (Purple)
- `#2196F3` (Light Blue)
- `#E91E63` (Pink)
- `#4CAF50` (Green)
- `#FF5722` (Deep Orange)
- `#795548` (Brown)
- `#607D8B` (Blue Grey)

### Status
5 colors for status indication:
- `#4CAF50` (Green - Good/Active)
- `#FFC107` (Amber - Warning/Pending)
- `#F44336` (Red - Error/Inactive)
- `#2196F3` (Blue - Info/Processing)
- `#9C27B0` (Purple - Special)

### Priority
4 colors for priority levels:
- `#F44336` (Red - High Priority)
- `#FF9800` (Orange - Medium Priority)
- `#4CAF50` (Green - Low Priority)
- `#9E9E9E` (Grey - No Priority)

### Sequential
9 colors for graduated/sequential data (light to dark progression):
- `#f7fcf0` to `#084081` (Light Green to Dark Blue progression)

## Supported Map Types

Series functionality is available for **locator maps** only:

✅ **Supported:**
- `point`: Coordinate-based markers
- `address`: Address-based markers (geocoded automatically)
- `intersection`: Street intersection markers

❌ **Not Supported:**
- `supervisor_district`: Uses choropleth styling
- `police_district`: Uses choropleth styling

## Data Format Requirements

Your location data should include a field for series grouping:

```python
location_data = [
    {
        "title": "Location Name",           # Required
        "lat": 37.7749,                    # Required (or use coordinates/address)
        "lon": -122.4194,                  # Required (or use coordinates/address)
        "series": "Category Name",         # Required for series functionality
        "tooltip": "Description",          # Optional
        "markerColor": "#FF0000"           # Optional (overrides series color)
    }
]
```

## Automatic Features

When using series functionality, the system automatically:

1. **Assigns Colors**: Consistent colors to each unique series value
2. **Creates Legend**: Shows all series with their colors
3. **Enhances Tooltips**: Adds series information to existing tooltips
4. **Maintains Consistency**: Same series always gets same color

## Testing

Run the test script to see the functionality in action:

```bash
python test_series_maps.py
```

This will create several example maps demonstrating:
- Categorical color series
- Status-based coloring
- Custom color palettes
- Address-based series maps

## Migration Guide

### Existing Maps
Existing maps without series functionality will continue to work unchanged. No breaking changes have been introduced.

### Adding Series to Existing Maps
To add series functionality to existing maps:

1. Add a series field to your location data
2. Add `series_field` parameter to your `generate_map()` call
3. Optionally specify a `color_palette`

### Example Migration

**Before:**
```python
generate_map(
    context_variables={},
    map_title="My Map",
    map_type="point",
    location_data=data
)
```

**After:**
```python
generate_map(
    context_variables={},
    map_title="My Map", 
    map_type="point",
    location_data=data,
    series_field="category",      # Add series field
    color_palette="categorical"   # Add color palette
)
```

## Technical Implementation

### Core Functions Enhanced

1. **`_prepare_locator_marker_data()`**: Enhanced to support series grouping and color assignment
2. **`_create_and_configure_locator_map()`**: Enhanced to create legends for series
3. **`create_datawrapper_chart()`**: Enhanced to pass series parameters
4. **`generate_map()`**: Enhanced with new series parameters

### Color Assignment Logic

1. Extract unique series values from data
2. Sort series for consistent assignment
3. Assign colors from palette (cycling if needed)
4. Apply colors to markers
5. Generate legend with color mapping

### Legend Generation

- Automatically enabled when series are detected
- Positioned at top-right of map
- Shows all series with their assigned colors
- Maintains Datawrapper styling consistency

## Benefits

### For Users
- **Better Visualization**: Clear categorization through colors
- **Improved Understanding**: Legends help identify categories
- **Enhanced Tooltips**: More informative marker details

### For Developers
- **Easy Implementation**: Simple parameter additions
- **Flexible Palettes**: Predefined and custom options
- **Backward Compatible**: No breaking changes
- **Automatic Features**: Legends and tooltips handled automatically

## Future Enhancements

Potential future improvements:
- Icon-based series (different shapes per series)
- Size-based series (different marker sizes)
- Pattern-based series (different fill patterns)
- Interactive legend filtering
- Series-based clustering

---

This enhancement significantly improves the map generation capabilities while maintaining simplicity and backward compatibility. 
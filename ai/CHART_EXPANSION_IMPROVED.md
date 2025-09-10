# Improved Chart Expansion System

## Overview

The improved chart expansion system provides a cleaner, more maintainable approach to handling charts in monthly newsletters. It separates concerns and provides different chart expansion strategies for different use cases.

## Workflow

### 1. Original Version (Immediate Viewing)
- **Purpose**: Users can immediately view the newsletter with working charts
- **Charts**: TransparentSF charts only (iframes)
- **Function**: `expand_charts_for_original_viewing()`
- **When**: Right after newsletter generation
- **Result**: Clean, simple charts that work immediately

### 2. Proofreading Version
- **Purpose**: AI proofreading with clear chart context
- **Charts**: Placeholder boxes showing chart information
- **Function**: `keep_placeholders_for_proofreading()`
- **When**: Before AI proofreading step
- **Result**: AI can understand chart context without heavy HTML

### 3. Final Web Version (Post-Proofreading)
- **Purpose**: Full-featured web version with both chart types
- **Charts**: Tabbed interface with TransparentSF and DataWrapper charts
- **Function**: `expand_charts_with_tabs_final()`
- **When**: After proofreading and revision
- **Result**: Users can switch between chart types

### 4. Email Version
- **Purpose**: Clean URLs for email clients
- **Charts**: DataWrapper URLs in copy-pasteable format
- **Function**: `expand_charts_for_email()`
- **When**: Final step for email distribution
- **Result**: Simple, clean URLs ready for email

## Chart Types Supported

### Time Series Charts
- **Pattern**: `[CHART:time_series:metric_id:district:period_type]`
- **TransparentSF**: `/backend/time-series-chart?metric_id=X&district=Y&period_type=Z`
- **DataWrapper**: Generated via `request_chart_image()`

### Time Series ID Charts
- **Pattern**: `[CHART:time_series_id:chart_id]`
- **TransparentSF**: `/backend/time-series-chart?chart_id=X`
- **DataWrapper**: Not available (TransparentSF only)

### Anomaly Charts
- **Pattern**: `[CHART:anomaly:anomaly_id]`
- **TransparentSF**: `/anomaly-analyzer/anomaly-chart?id=X`
- **DataWrapper**: Generated via `generate_anomaly_chart_from_id()`

### Map Charts
- **Pattern**: `[CHART:map:map_id]`
- **TransparentSF**: `/backend/map-chart?id=X`
- **DataWrapper**: Retrieved via `get_existing_map_url()` or created via `create_datawrapper_map()`

## Implementation Details

### File Structure
```
ai/tools/chart_expansion_improved.py  # New improved system
ai/tools/chart_expansion.py          # Original system (kept for fallback)
ai/static/chart-tabs.css             # Tab styling
ai/static/chart-tabs.js              # Tab functionality
```

### Integration Points
- **Monthly Report Process**: `ai/monthly_report.py`
- **Email Generation**: `generate_email_compatible_report()`
- **Chart Generation**: Various tools in `ai/tools/`

### Error Handling
- Each function has comprehensive error handling
- Fallback to original system if improved system fails
- Graceful degradation if DataWrapper charts can't be generated

## Benefits

### 1. Separation of Concerns
- Each function has a single, clear purpose
- Easier to maintain and debug
- Better error isolation

### 2. Improved User Experience
- Original version works immediately
- Proofreading is more effective with placeholders
- Final version has full functionality
- Email version is clean and simple

### 3. Better Maintainability
- Clear function names and purposes
- Consistent error handling
- Modular design
- Easy to extend

### 4. Performance
- No unnecessary chart generation during proofreading
- Efficient placeholder processing
- Lazy DataWrapper chart generation

## Usage Examples

### Basic Usage
```python
from tools.chart_expansion_improved import (
    expand_charts_for_original_viewing,
    keep_placeholders_for_proofreading,
    expand_charts_with_tabs_final,
    expand_charts_for_email
)

# Original version
expand_charts_for_original_viewing("newsletter.html")

# Proofreading version
keep_placeholders_for_proofreading("newsletter.html")

# Final web version
expand_charts_with_tabs_final("newsletter.html")

# Email version
expand_charts_for_email("newsletter.html")
```

### Integration in Monthly Report Process
```python
# Step 1: Generate newsletter with placeholders
newsletter_result = generate_monthly_report(...)

# Step 2: Create original viewing version
expand_charts_for_original_viewing(newsletter_result["report_path"])

# Step 3: Prepare for proofreading
keep_placeholders_for_proofreading(newsletter_result["report_path"])

# Step 4: Proofreading
revised_result = proofread_and_revise_report(...)

# Step 5: Create final web version
expand_charts_with_tabs_final(revised_result["revised_report_path"])

# Step 6: Create email version
expand_charts_for_email(revised_result["revised_report_path"])
```

## Migration from Old System

The improved system is designed to be a drop-in replacement for the old system:

1. **Backward Compatibility**: Old functions are still available
2. **Gradual Migration**: Can be adopted incrementally
3. **Fallback Support**: Falls back to old system if new system fails
4. **Same Interface**: Functions have similar signatures

## Future Enhancements

### Potential Improvements
1. **Caching**: Cache DataWrapper chart URLs to avoid regeneration
2. **Async Processing**: Generate DataWrapper charts asynchronously
3. **Chart Validation**: Validate chart URLs before including them
4. **Custom Styling**: Allow custom CSS for different chart types
5. **Analytics**: Track which chart types are used most

### Configuration Options
1. **Chart Sizes**: Configurable chart dimensions
2. **Default Chart Type**: Choose default tab (TransparentSF vs DataWrapper)
3. **Error Handling**: Configurable fallback behavior
4. **Logging**: Configurable logging levels

## Troubleshooting

### Common Issues

1. **DataWrapper Charts Not Generating**
   - Check API credentials
   - Verify chart data is available
   - Check network connectivity

2. **Tab Interface Not Working**
   - Verify CSS and JS files are loaded
   - Check browser console for errors
   - Ensure proper HTML structure

3. **Placeholders Not Showing**
   - Check placeholder patterns match
   - Verify regex patterns are correct
   - Check file encoding

### Debug Mode
Enable debug logging to see detailed information about chart expansion:
```python
import logging
logging.getLogger('tools.chart_expansion_improved').setLevel(logging.DEBUG)
```

## Testing

### Test Cases
1. **Original Viewing**: Verify TransparentSF charts load
2. **Proofreading**: Verify placeholders are clear and informative
3. **Final Web**: Verify tab switching works
4. **Email**: Verify DataWrapper URLs are clean and copy-pasteable

### Test Data
Use test newsletters with various chart types to verify all functionality works correctly.

## Conclusion

The improved chart expansion system provides a more robust, maintainable, and user-friendly approach to handling charts in monthly newsletters. It separates concerns, improves performance, and provides better error handling while maintaining backward compatibility.



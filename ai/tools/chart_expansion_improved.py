"""
Improved Chart Expansion System for Monthly Reports

This module provides a cleaner, more maintainable approach to chart expansion
with separate functions for different use cases:

1. Original version: TransparentSF charts only (for immediate viewing)
2. Proofreading version: Keep placeholders intact
3. Final web version: Tabbed interface with both TransparentSF and DataWrapper
4. Email version: Clean DataWrapper URLs for copy-paste
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def expand_charts_for_original_viewing(report_path):
    """
    Expand chart placeholders to TransparentSF charts only.
    This is for the original version that users see immediately.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding charts for original viewing in: {report_path}")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define chart placeholder patterns
        patterns = {
            'time_series': r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*',
            'time_series_id': r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*',
            'anomaly': r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*',
            'map': r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        }
        
        def replace_time_series(match):
            metric_id, district, period_type = match.groups()
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <iframe src="/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}"
            style="width: 100%; height: 600px; border: none;" 
            frameborder="0" 
            scrolling="no"
            title="Time Series Chart - Metric {metric_id}">
    </iframe>
</div>'''
        
        def replace_time_series_id(match):
            chart_id = match.group(1)
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <iframe src="/backend/time-series-chart?chart_id={chart_id}"
            style="width: 100%; height: 600px; border: none;" 
            frameborder="0" 
            scrolling="no"
            title="Time Series Chart - ID {chart_id}">
    </iframe>
</div>'''
        
        def replace_anomaly(match):
            anomaly_id = match.group(1)
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section"
            style="width: 100%; height: 600px; border: none;" 
            frameborder="0" 
            scrolling="no"
            title="Anomaly Analysis - ID {anomaly_id}">
    </iframe>
</div>'''
        
        def replace_map(match):
            map_id = match.group(1)
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <iframe src="/backend/map-chart?map_id={map_id}"
            style="width: 100%; height: 400px; border: none;" 
            frameborder="0" 
            scrolling="no"
            title="Map - ID {map_id}">
    </iframe>
</div>'''
        
        # Apply replacements
        content = re.sub(patterns['time_series'], replace_time_series, content)
        content = re.sub(patterns['time_series_id'], replace_time_series_id, content)
        content = re.sub(patterns['anomaly'], replace_anomaly, content)
        content = re.sub(patterns['map'], replace_map, content)
        
        # Write back to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully expanded charts for original viewing: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error expanding charts for original viewing: {str(e)}")
        return report_path

def keep_placeholders_for_proofreading(report_path):
    """
    Keep chart placeholders intact for AI proofreading.
    This function ensures placeholders are properly formatted but not expanded.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Preparing placeholders for proofreading in: {report_path}")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define patterns for chart placeholders
        patterns = {
            'time_series': r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*',
            'time_series_id': r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*',
            'anomaly': r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*',
            'map': r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        }
        
        def format_placeholder(match, chart_type, chart_id):
            return f'''
<div class="chart-placeholder" style="border: 2px dashed #ccc; padding: 20px; margin: 20px 0; text-align: center; background-color: #f9f9f9;">
    <h4>📊 {chart_type} Chart</h4>
    <p><strong>Chart ID:</strong> {chart_id}</p>
    <p><em>Chart will be rendered here after proofreading</em></p>
</div>'''
        
        def replace_time_series(match):
            metric_id, district, period_type = match.groups()
            return format_placeholder(match, "Time Series", f"{metric_id} (District {district}, {period_type})")
        
        def replace_time_series_id(match):
            chart_id = match.group(1)
            return format_placeholder(match, "Time Series", chart_id)
        
        def replace_anomaly(match):
            anomaly_id = match.group(1)
            return format_placeholder(match, "Anomaly Analysis", anomaly_id)
        
        def replace_map(match):
            map_id = match.group(1)
            return format_placeholder(match, "Map", map_id)
        
        # Apply replacements
        content = re.sub(patterns['time_series'], replace_time_series, content)
        content = re.sub(patterns['time_series_id'], replace_time_series_id, content)
        content = re.sub(patterns['anomaly'], replace_anomaly, content)
        content = re.sub(patterns['map'], replace_map, content)
        
        # Write back to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully prepared placeholders for proofreading: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error preparing placeholders for proofreading: {str(e)}")
        return report_path

def expand_charts_with_tabs_final(report_path):
    """
    Expand chart placeholders to tabbed interface with both TransparentSF and DataWrapper charts.
    This is the final web version after proofreading.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding charts with tabs for final version in: {report_path}")
    
    try:
        from monthly_report import request_chart_image, get_existing_map_url
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define patterns
        patterns = {
            'time_series': r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*',
            'time_series_id': r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*',
            'anomaly': r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*',
            'map': r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        }
        
        def replace_time_series(match):
            metric_id, district, period_type = match.groups()
            chart_id = f"chart_{metric_id}_{district}_{period_type}"
            
            # Try to get DataWrapper chart
            dw_url = None
            try:
                dw_url = request_chart_image('time_series', {
                    'metric_id': metric_id,
                    'district': district,
                    'period_type': period_type
                })
                if dw_url and 'datawrapper' not in dw_url:
                    dw_url = None
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for time series {metric_id}: {e}")
            
            # Build tabbed interface
            dw_tab = ""
            dw_button = ""
            if dw_url:
                dw_tab = f'''
    <div id="{chart_id}_dw" class="chart-panel" style="display: none;">
        <iframe src="{dw_url}"
                title="Time Series Chart - Metric {metric_id}"
                style="width: 100%; height: 600px; border: none; display: block; margin: 0 auto;" 
                frameborder="0">
        </iframe>
    </div>'''
                dw_button = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <div id="{chart_id}_local" class="chart-panel active">
        <iframe src="/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}"
                style="width: 100%; height: 600px; border: none; display: block; margin: 0 auto;" 
                frameborder="0" 
                scrolling="no"
                title="Time Series Chart - Metric {metric_id}">
        </iframe>
    </div>
    {dw_tab}
    <div style="margin-top: 10px; text-align: center;">
        <button onclick="toggleChartView('{chart_id}')" style="
            background: #f8f9fa; 
            border: 1px solid #ddd; 
            padding: 5px 15px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 12px;
            color: #666;
        ">Switch to DataWrapper</button>
    </div>
</div>'''
        
        def replace_time_series_id(match):
            chart_id = match.group(1)
            chart_container_id = f"chart_{chart_id}"
            
            # Try to get DataWrapper chart by looking up the chart metadata
            dw_url = None
            try:
                # Get chart metadata to find the actual metric_id (object_id)
                from tools.store_time_series import get_time_series_metadata
                metadata = get_time_series_metadata(chart_id=chart_id)
                
                if metadata is not None and not metadata.empty:
                    # Get the first row of metadata
                    chart_meta = metadata.iloc[0]
                    object_id = chart_meta.get('object_id')
                    district = chart_meta.get('district', 0)
                    period_type = chart_meta.get('period_type', 'month')
                    
                    if object_id:
                        logger.info(f"Found chart metadata for chart_id {chart_id}: object_id={object_id}, district={district}, period_type={period_type}")
                        
                        # Generate DataWrapper chart using the object_id as metric_id
                        dw_url = request_chart_image('time_series', {
                            'metric_id': object_id,
                            'district': district,
                            'period_type': period_type
                        })
                        
                        if dw_url and 'datawrapper' not in dw_url:
                            dw_url = None
                        elif dw_url:
                            logger.info(f"Successfully generated DataWrapper chart for chart_id {chart_id}: {dw_url}")
                else:
                    logger.warning(f"No metadata found for chart_id {chart_id}")
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for chart_id {chart_id}: {e}")
            
            # Build tabbed interface
            dw_tab = ""
            dw_button = ""
            if dw_url:
                dw_tab = f'''
    <div id="{chart_container_id}_dw" class="chart-panel" style="display: none;">
        <iframe src="{dw_url}"
                title="Time Series Chart - ID {chart_id}"
                style="width: 100%; height: 600px; border: none; display: block; margin: 0 auto;" 
                frameborder="0">
        </iframe>
    </div>'''
                dw_button = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_container_id}\', \'dw\')">DataWrapper</button>'
            
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <div id="{chart_container_id}_local" class="chart-panel active">
        <iframe src="/backend/time-series-chart?chart_id={chart_id}"
                style="width: 100%; height: 600px; border: none; display: block; margin: 0 auto;" 
                frameborder="0" 
                scrolling="no"
                title="Time Series Chart - ID {chart_id}">
        </iframe>
    </div>
    {dw_tab}
    <div style="margin-top: 10px; text-align: center;">
        <button onclick="toggleChartView('{chart_container_id}')" style="
            background: #f8f9fa; 
            border: 1px solid #ddd; 
            padding: 5px 15px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 12px;
            color: #666;
        ">Switch to DataWrapper</button>
    </div>
</div>'''
        
        def replace_anomaly(match):
            anomaly_id = match.group(1)
            chart_id = f"anomaly_{anomaly_id}"
            
            # Try to get DataWrapper chart
            dw_url = None
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                dw_url = generate_anomaly_chart_from_id(anomaly_id)
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for anomaly {anomaly_id}: {e}")
            
            # Build tabbed interface
            dw_tab = ""
            dw_button = ""
            if dw_url:
                dw_tab = f'''
    <div id="{chart_id}_dw" class="chart-panel" style="display: none;">
        <iframe src="{dw_url}"
                title="Anomaly {anomaly_id}: Trend Analysis"
                style="width: 100%; height: 600px; border: none; display: block; margin: 0 auto;" 
                frameborder="0">
        </iframe>
    </div>'''
                dw_button = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <div id="{chart_id}_local" class="chart-panel active">
        <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section"
                style="width: 100%; height: 600px; border: none; display: block; margin: 0 auto;" 
                frameborder="0" 
                scrolling="no"
                title="Anomaly Analysis - ID {anomaly_id}">
        </iframe>
    </div>
    {dw_tab}
    <div style="margin-top: 10px; text-align: center;">
        <button onclick="toggleChartView('{chart_id}')" style="
            background: #f8f9fa; 
            border: 1px solid #ddd; 
            padding: 5px 15px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 12px;
            color: #666;
        ">Switch to DataWrapper</button>
    </div>
</div>'''
        
        def replace_map(match):
            map_id = match.group(1)
            chart_id = f"map_{map_id}"
            
            # Try to get DataWrapper map
            dw_url = get_existing_map_url(map_id)
            if not dw_url:
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    dw_url = create_datawrapper_map(map_id)
                except Exception as e:
                    logger.warning(f"Failed to create DataWrapper map for {map_id}: {e}")
            
            # Build tabbed interface
            dw_tab = ""
            dw_button = ""
            if dw_url:
                dw_tab = f'''
    <div id="{chart_id}_dw" class="chart-panel" style="display: none;">
        <iframe src="{dw_url}"
                title="Map {map_id}"
                style="width: 100%; max-width: 600px; height: 400px; border: none; display: block; margin: 0 auto;" 
                frameborder="0">
        </iframe>
    </div>'''
                dw_button = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            return f'''
<div class="chart-container" style="margin: 20px 0; text-align: center;">
    <div id="{chart_id}_local" class="chart-panel active">
        <iframe src="/backend/map-chart?map_id={map_id}"
                style="width: 100%; max-width: 600px; height: 400px; border: none; display: block; margin: 0 auto;" 
                frameborder="0" 
                scrolling="no"
                title="Map - ID {map_id}">
        </iframe>
    </div>
    {dw_tab}
    <div style="margin-top: 10px; text-align: center;">
        <button onclick="toggleChartView('{chart_id}')" style="
            background: #f8f9fa; 
            border: 1px solid #ddd; 
            padding: 5px 15px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 12px;
            color: #666;
        ">Switch to DataWrapper</button>
    </div>
</div>'''
        
        # Apply replacements
        content = re.sub(patterns['time_series'], replace_time_series, content)
        content = re.sub(patterns['time_series_id'], replace_time_series_id, content)
        content = re.sub(patterns['anomaly'], replace_anomaly, content)
        content = re.sub(patterns['map'], replace_map, content)
        
        # Add CSS and JS references
        css_js = '''
<link rel="stylesheet" href="/static/chart-tabs.css">
<script src="/static/chart-tabs.js"></script>
<script>
function toggleChartView(chartId) {
    const localPanel = document.getElementById(chartId + '_local');
    const dwPanel = document.getElementById(chartId + '_dw');
    const button = event.target;
    
    if (localPanel && dwPanel) {
        if (localPanel.style.display !== 'none') {
            // Switch to DataWrapper
            localPanel.style.display = 'none';
            dwPanel.style.display = 'block';
            button.textContent = 'Switch to Transparent SF';
        } else {
            // Switch to Transparent SF
            localPanel.style.display = 'block';
            dwPanel.style.display = 'none';
            button.textContent = 'Switch to DataWrapper';
        }
    }
}
</script>
'''
        
        # Insert CSS and JS in head section
        if '<head>' in content:
            content = content.replace('</head>', css_js + '</head>')
        else:
            content = css_js + content
        
        # Write back to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully expanded charts with tabs for final version: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error expanding charts with tabs: {str(e)}")
        return report_path

def expand_charts_for_email(report_path):
    """
    Expand chart placeholders to clean DataWrapper URLs for email use.
    This creates a simple, copy-pasteable format for email clients.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding charts for email format in: {report_path}")
    
    try:
        from monthly_report import request_chart_image, get_existing_map_url
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define patterns
        patterns = {
            'time_series': r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*',
            'time_series_id': r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*',
            'anomaly': r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*',
            'map': r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        }
        
        def replace_time_series(match):
            metric_id, district, period_type = match.groups()
            
            # Try to get DataWrapper chart
            dw_url = None
            try:
                dw_url = request_chart_image('time_series', {
                    'metric_id': metric_id,
                    'district': district,
                    'period_type': period_type
                })
                if dw_url and 'datawrapper' not in dw_url:
                    dw_url = None
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for time series {metric_id}: {e}")
            
            if dw_url:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Time Series Chart - Metric {metric_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_url}
    </div>
</div>'''
            else:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Time Series Chart - Metric {metric_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>'''
        
        def replace_time_series_id(match):
            chart_id = match.group(1)
            
            # Try to get DataWrapper chart by looking up the chart metadata
            dw_url = None
            try:
                # Get chart metadata to find the actual metric_id (object_id)
                from tools.store_time_series import get_time_series_metadata
                metadata = get_time_series_metadata(chart_id=chart_id)
                
                if metadata is not None and not metadata.empty:
                    # Get the first row of metadata
                    chart_meta = metadata.iloc[0]
                    object_id = chart_meta.get('object_id')
                    district = chart_meta.get('district', 0)
                    period_type = chart_meta.get('period_type', 'month')
                    
                    if object_id:
                        logger.info(f"Found chart metadata for chart_id {chart_id}: object_id={object_id}, district={district}, period_type={period_type}")
                        
                        # Generate DataWrapper chart using the object_id as metric_id
                        dw_url = request_chart_image('time_series', {
                            'metric_id': object_id,
                            'district': district,
                            'period_type': period_type
                        })
                        
                        if dw_url and 'datawrapper' not in dw_url:
                            dw_url = None
                else:
                    logger.warning(f"No metadata found for chart_id {chart_id}")
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for chart_id {chart_id}: {e}")
            
            if dw_url:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Time Series Chart - ID {chart_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_url}
    </div>
</div>'''
            else:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Time Series Chart - ID {chart_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>'''
        
        def replace_anomaly(match):
            anomaly_id = match.group(1)
            
            # Try to get DataWrapper chart
            dw_url = None
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                dw_url = generate_anomaly_chart_from_id(anomaly_id)
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for anomaly {anomaly_id}: {e}")
            
            if dw_url:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Anomaly Analysis - ID {anomaly_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_url}
    </div>
</div>'''
            else:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Anomaly Analysis - ID {anomaly_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>'''
        
        def replace_map(match):
            map_id = match.group(1)
            
            # Try to get DataWrapper map
            dw_url = get_existing_map_url(map_id)
            if not dw_url:
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    dw_url = create_datawrapper_map(map_id)
                except Exception as e:
                    logger.warning(f"Failed to create DataWrapper map for {map_id}: {e}")
            
            if dw_url:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Map - ID {map_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_url}
    </div>
</div>'''
            else:
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Map - ID {map_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>'''
        
        # Apply replacements
        content = re.sub(patterns['time_series'], replace_time_series, content)
        content = re.sub(patterns['time_series_id'], replace_time_series_id, content)
        content = re.sub(patterns['anomaly'], replace_anomaly, content)
        content = re.sub(patterns['map'], replace_map, content)
        
        # Write back to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully expanded charts for email format: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error expanding charts for email: {str(e)}")
        return report_path



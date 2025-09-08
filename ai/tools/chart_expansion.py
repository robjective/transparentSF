"""
Chart expansion utilities for monthly reports.

This module provides two different chart expansion strategies:
1. Local/Internal expansion: For web sharing (uses internal iframes)
2. DataWrapper (DW) expansion: For email/newsletter compatibility (uses DataWrapper charts)
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def expand_chart_references_local(report_path):
    """
    Process a report file and replace simplified chart references with local/internal iframes.
    This is good for web sharing where the charts are served from the local backend.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references to local/internal charts in report: {report_path}")
    
    try:
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*'
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*'
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        
        # Replace time series chart references with local iframes
        def replace_time_series_local(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Using local chart for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            iframe_html = (
                f'<div class="chart-container">\n'
                f'    <iframe src="/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}"\n'
                f'            style="width: 100%; height: 600px; border: none;" \n'
                f'            frameborder="0" \n'
                f'            scrolling="yes"\n'
                f'            title="Time Series Chart - Metric {metric_id}">\n'
                f'    </iframe>\n'
                f'</div>'
            )
            return iframe_html
        
        # Replace anomaly chart references with local iframes
        def replace_anomaly_local(match):
            anomaly_id = match.group(1)
            
            logger.info(f"Using local chart for anomaly ID: {anomaly_id}")
            
            iframe_html = (
                f'<div class="chart-container">\n'
                f'    <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section"\n'
                f'            style="width: 100%; height: 600px; border: none;" \n'
                f'            frameborder="0" \n'
                f'            scrolling="yes"\n'
                f'            title="Anomaly Analysis - ID {anomaly_id}">\n'
                f'    </iframe>\n'
                f'</div>'
            )
            return iframe_html
        
        # Replace map chart references with local iframes
        def replace_map_local(match):
            map_id = match.group(1)
            
            logger.info(f"Using local chart for map ID: {map_id}")
            
            iframe_html = (
                f'<div class="chart-container">\n'
                f'    <iframe src="/backend/map-chart?map_id={map_id}"\n'
                f'            style="width: 100%; height: 600px; border: none;" \n'
                f'            frameborder="0" \n'
                f'            scrolling="yes"\n'
                f'            title="Map - ID {map_id}">\n'
                f'    </iframe>\n'
                f'</div>'
            )
            return iframe_html
        
        # Replace time series ID chart references with local iframes
        def replace_time_series_id_local(match):
            chart_id = match.group(1)
            
            logger.info(f"Using local chart for time_series_id: {chart_id}")
            
            iframe_html = (
                f'<div class="chart-container">\n'
                f'    <iframe src="/backend/time-series-chart?chart_id={chart_id}"\n'
                f'            style="width: 100%; height: 600px; border: none;" \n'
                f'            frameborder="0" \n'
                f'            scrolling="yes"\n'
                f'            title="Time Series Chart - ID {chart_id}">\n'
                f'    </iframe>\n'
                f'</div>'
            )
            return iframe_html
        
        # Apply all replacements
        report_html = re.sub(time_series_pattern, replace_time_series_local, report_html)
        report_html = re.sub(anomaly_pattern, replace_anomaly_local, report_html)
        report_html = re.sub(map_pattern, replace_map_local, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id_local, report_html)
        
        # Write the updated report back to the file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references to local charts in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references_local: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def expand_chart_references_dw(report_path):
    """
    Process a report file and replace simplified chart references with DataWrapper charts.
    This is good for email/newsletter compatibility where external chart services are needed.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references to DataWrapper charts in report: {report_path}")
    
    try:
        # Import the request_chart_image function from monthly_report
        from monthly_report import request_chart_image, get_existing_map_url
        
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*'
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*'
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        
        # Define pattern for direct image references
        image_pattern_with_alt = r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]+)"[^>]*>'
        image_pattern_without_alt = r'<img[^>]*src="([^"]+)"[^>]*>'
        
        # Replace time series chart references with DataWrapper charts
        def replace_time_series_dw(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Generating DataWrapper chart for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            # Try to generate a DataWrapper chart first
            chart_url = request_chart_image('time_series', {
                'metric_id': metric_id,
                'district': district,
                'period_type': period_type
            })
            
            if chart_url and 'datawrapper' in chart_url:
                # Use DataWrapper chart
                iframe_html = (
                    f'<div class="chart-container">\n'
                    f'    <div class="datawrapper-chart-embed">\n'
                    f'        <iframe src="{chart_url}"\n'
                    f'                title="Time Series Chart - Metric {metric_id}"\n'
                    f'                style="width: 100%; border: none;" \n'
                    f'                height="400"\n'
                    f'                frameborder="0">\n'
                    f'        </iframe>\n'
                    f'    </div>\n'
                    f'</div>'
                )
                return iframe_html
            else:
                # Fallback to local iframe
                logger.warning(f"DataWrapper chart generation failed, using local fallback for metric_id: {metric_id}")
                iframe_html = (
                    f'<div class="chart-container">\n'
                    f'    <iframe src="/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}"\n'
                    f'            style="width: 100%; height: 600px; border: none;" \n'
                    f'            frameborder="0" \n'
                    f'            scrolling="yes"\n'
                    f'            title="Time Series Chart - Metric {metric_id}">\n'
                    f'    </iframe>\n'
                    f'</div>'
                )
                return iframe_html
        
        # Replace anomaly chart references with DataWrapper charts
        def replace_anomaly_dw(match):
            anomaly_id = match.group(1)
            
            logger.info(f"Generating DataWrapper chart for anomaly ID: {anomaly_id}")
            
            # Try to use Datawrapper for this anomaly
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                chart_url = generate_anomaly_chart_from_id(anomaly_id)
                
                if chart_url:
                    logger.info(f"Successfully generated Datawrapper chart for anomaly {anomaly_id}: {chart_url}")
                    iframe_html = (
                        f'<div class="chart-container">\n'
                        f'    <div class="datawrapper-chart-embed">\n'
                        f'        <iframe src="{chart_url}"\n'
                        f'                title="Anomaly {anomaly_id}: Trend Analysis"\n'
                        f'                style="width: 100%; border: none;" \n'
                        f'                height="400"\n'
                        f'                frameborder="0">\n'
                        f'        </iframe>\n'
                        f'    </div>\n'
                        f'</div>'
                    )
                    return iframe_html
            except Exception as e:
                logger.error(f"Error generating Datawrapper chart for anomaly {anomaly_id}: {str(e)}")
            
            # Fallback to local iframe
            logger.warning(f"DataWrapper chart generation failed, using local fallback for anomaly_id: {anomaly_id}")
            iframe_html = (
                f'<div class="chart-container">\n'
                f'    <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section"\n'
                f'            style="width: 100%; height: 600px; border: none;" \n'
                f'            frameborder="0" \n'
                f'            scrolling="yes"\n'
                f'            title="Anomaly Analysis - ID {anomaly_id}">\n'
                f'    </iframe>\n'
                f'</div>'
            )
            return iframe_html
        
        # Replace map chart references with DataWrapper charts
        def replace_map_dw(match):
            map_id = match.group(1)
            
            logger.info(f"Processing DataWrapper map reference: {map_id}")
            
            # Try to get existing map URL first
            map_url = get_existing_map_url(map_id)
            
            if map_url:
                logger.info(f"Found existing DataWrapper map URL for {map_id}: {map_url}")
                iframe_html = (
                    f'<div class="chart-container">\n'
                    f'    <div class="datawrapper-chart-embed">\n'
                    f'        <iframe src="{map_url}"\n'
                    f'                title="Map {map_id}"\n'
                    f'                style="width: 100%; border: none;" \n'
                    f'                height="400"\n'
                    f'                frameborder="0">\n'
                    f'        </iframe>\n'
                    f'    </div>\n'
                    f'</div>'
                )
                return iframe_html
            else:
                # If no existing map URL found, try to create/publish the map
                logger.info(f"No existing map URL found for map {map_id}, attempting to create it")
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    map_url = create_datawrapper_map(map_id)
                    
                    if map_url:
                        logger.info(f"Successfully created Datawrapper map for map {map_id}: {map_url}")
                        iframe_html = (
                            f'<div class="chart-container">\n'
                            f'    <div class="datawrapper-chart-embed">\n'
                            f'        <iframe src="{map_url}"\n'
                            f'                title="Map {map_id}"\n'
                            f'                style="width: 100%; border: none;" \n'
                            f'                height="400"\n'
                            f'                frameborder="0">\n'
                            f'        </iframe>\n'
                            f'    </div>\n'
                            f'</div>'
                        )
                        return iframe_html
                    else:
                        logger.warning(f"Failed to create Datawrapper map for {map_id}")
                        return f'<div class="chart-container">\n<p>Map {map_id} could not be loaded</p>\n</div>'
                except Exception as e:
                    logger.error(f"Error creating Datawrapper map for {map_id}: {str(e)}")
                    return f'<div class="chart-container">\n<p>Map {map_id} could not be loaded</p>\n</div>'
        
        # Replace time series ID chart references with DataWrapper charts
        def replace_time_series_id_dw(match):
            chart_id = match.group(1)
            
            logger.info(f"Using local chart for time_series_id: {chart_id} (no DataWrapper equivalent)")
            
            # For time_series_id, we don't have a DataWrapper equivalent, so use local
            iframe_html = (
                f'<div class="chart-container">\n'
                f'    <iframe src="/backend/time-series-chart?chart_id={chart_id}"\n'
                f'            style="width: 100%; height: 600px; border: none;" \n'
                f'            frameborder="0" \n'
                f'            scrolling="yes"\n'
                f'            title="Time Series Chart - ID {chart_id}">\n'
                f'    </iframe>\n'
                f'</div>'
            )
            return iframe_html
        
        # Replace direct image references with DataWrapper charts
        def replace_image_with_alt_dw(match):
            img_tag = match.group(0)
            img_src = match.group(1)
            img_alt = match.group(2)
            
            # Check if it's an anomaly chart image
            if img_src.startswith("anomaly_"):
                # Extract anomaly ID from the filename
                parts = img_src.split("_")
                if len(parts) >= 2:
                    anomaly_id = parts[1].split(".")[0]  # Remove file extension
                    
                    # Try to use Datawrapper for this anomaly image
                    try:
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        logger.info(f"Generating DataWrapper chart for anomaly image ID: {anomaly_id}")
                        
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly image {anomaly_id}: {chart_url}")
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{chart_url}"\n'
                                f'                title="Anomaly {anomaly_id}: {img_alt}"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="400"\n'
                                f'                frameborder="0">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as e:
                        logger.error(f"Error generating Datawrapper chart for anomaly image {anomaly_id}: {str(e)}")
                
                # Fallback to original image
                return img_tag
            else:
                # For non-anomaly images, return as-is
                return img_tag
        
        def replace_image_without_alt_dw(match):
            img_tag = match.group(0)
            img_src = match.group(1)
            
            # Check if it's an anomaly chart image
            if img_src.startswith("anomaly_"):
                # Extract anomaly ID from the filename
                parts = img_src.split("_")
                if len(parts) >= 2:
                    anomaly_id = parts[1].split(".")[0]  # Remove file extension
                    
                    # Try to use Datawrapper for this anomaly image
                    try:
                        from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                        logger.info(f"Generating DataWrapper chart for anomaly image ID (no alt): {anomaly_id}")
                        
                        chart_url = generate_anomaly_chart_from_id(anomaly_id)
                        
                        if chart_url:
                            logger.info(f"Successfully generated Datawrapper chart for anomaly image {anomaly_id}: {chart_url}")
                            iframe_html = (
                                f'<div class="chart-container">\n'
                                f'    <div class="datawrapper-chart-embed">\n'
                                f'        <iframe src="{chart_url}"\n'
                                f'                title="Anomaly {anomaly_id}: Trend Analysis"\n'
                                f'                style="width: 100%; border: none;" \n'
                                f'                height="400"\n'
                                f'                frameborder="0">\n'
                                f'        </iframe>\n'
                                f'    </div>\n'
                                f'</div>'
                            )
                            return iframe_html
                    except Exception as e:
                        logger.error(f"Error generating Datawrapper chart for anomaly image {anomaly_id}: {str(e)}")
                
                # Fallback to original image
                return img_tag
            else:
                # For non-anomaly images, return as-is
                return img_tag
        
        # Apply all replacements
        report_html = re.sub(time_series_pattern, replace_time_series_dw, report_html)
        report_html = re.sub(anomaly_pattern, replace_anomaly_dw, report_html)
        report_html = re.sub(map_pattern, replace_map_dw, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id_dw, report_html)
        report_html = re.sub(image_pattern_with_alt, replace_image_with_alt_dw, report_html)
        report_html = re.sub(image_pattern_without_alt, replace_image_without_alt_dw, report_html)
        
        # Write the updated report back to the file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references to DataWrapper charts in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references_dw: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def expand_chart_references_for_proofreader(report_path):
    """
    Process a report file and replace chart references with simple placeholders
    for proofreading. This version doesn't include the heavy CSS/JS overhead.
    """
    try:
        logger.info(f"Expanding chart references for proofreader in: {report_path}")
        
        # Read the report content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace time series chart references with simple placeholders
        def replace_time_series_placeholder(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Creating placeholder for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            placeholder_html = f'''
<div class="chart-placeholder" style="border: 2px dashed #ccc; padding: 20px; margin: 20px 0; text-align: center; background-color: #f9f9f9;">
    <h4>📊 Time Series Chart</h4>
    <p><strong>Metric ID:</strong> {metric_id} | <strong>District:</strong> {district} | <strong>Period:</strong> {period_type}</p>
    <p><em>Chart will be rendered here with tabbed interface (Transparent SF + DataWrapper)</em></p>
</div>
'''
            return placeholder_html
        
        # Replace anomaly chart references with simple placeholders
        def replace_anomaly_placeholder(match):
            anomaly_id = match.group(1)
            
            logger.info(f"Creating placeholder for anomaly ID: {anomaly_id}")
            
            placeholder_html = f'''
<div class="chart-placeholder" style="border: 2px dashed #ccc; padding: 20px; margin: 20px 0; text-align: center; background-color: #f9f9f9;">
    <h4>📈 Anomaly Chart</h4>
    <p><strong>Anomaly ID:</strong> {anomaly_id}</p>
    <p><em>Chart will be rendered here with tabbed interface (Transparent SF + DataWrapper)</em></p>
</div>
'''
            return placeholder_html
        
        # Replace map chart references with simple placeholders
        def replace_map_placeholder(match):
            map_id = match.group(1)
            
            logger.info(f"Creating placeholder for map ID: {map_id}")
            
            placeholder_html = f'''
<div class="chart-placeholder" style="border: 2px dashed #ccc; padding: 20px; margin: 20px 0; text-align: center; background-color: #f9f9f9;">
    <h4>🗺️ Map Chart</h4>
    <p><strong>Map ID:</strong> {map_id}</p>
    <p><em>Map will be rendered here with tabbed interface (Local + DataWrapper)</em></p>
</div>
'''
            return placeholder_html
        
        # Replace time series ID chart references with simple placeholders
        def replace_time_series_id_placeholder(match):
            chart_id = match.group(1)
            
            logger.info(f"Creating placeholder for chart ID: {chart_id}")
            
            placeholder_html = f'''
<div class="chart-placeholder" style="border: 2px dashed #ccc; padding: 20px; margin: 20px 0; text-align: center; background-color: #f9f9f9;">
    <h4>📊 Time Series Chart</h4>
    <p><strong>Chart ID:</strong> {chart_id}</p>
    <p><em>Chart will be rendered here</em></p>
</div>
'''
            return placeholder_html
        
        # Apply replacements
        content = re.sub(r'\[CHART:time_series:(\d+):(\d+):(\w+)\]', replace_time_series_placeholder, content)
        content = re.sub(r'\[CHART:anomaly:(\d+)\]', replace_anomaly_placeholder, content)
        content = re.sub(r'\[CHART:map:(\d+)\]', replace_map_placeholder, content)
        content = re.sub(r'\[CHART:time_series_id:(\d+)\]', replace_time_series_id_placeholder, content)
        
        # Write the updated content back to the file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully expanded chart references for proofreader in: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references_for_proofreader: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def expand_chart_references_with_tabs(report_path):
    """
    Process a report file and replace simplified chart references with switchable tabs
    showing both local/internal charts and DataWrapper charts.
    This provides the best of both worlds - local charts for web viewing and DataWrapper for email compatibility.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references with switchable tabs in report: {report_path}")
    
    try:
        # Import the request_chart_image function from monthly_report
        from monthly_report import request_chart_image, get_existing_map_url
        
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*'
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*'
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        
        # Define pattern for direct image references
        image_pattern_with_alt = r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]+)"[^>]*>'
        image_pattern_without_alt = r'<img[^>]*src="([^"]+)"[^>]*>'
        
        # Replace time series chart references with switchable tabs
        def replace_time_series_tabs(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Creating switchable tabs for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            # Try to generate a DataWrapper chart
            dw_chart_url = None
            try:
                dw_chart_url = request_chart_image('time_series', {
                    'metric_id': metric_id,
                    'district': district,
                    'period_type': period_type
                })
                if dw_chart_url and 'datawrapper' not in dw_chart_url:
                    dw_chart_url = None
            except Exception as e:
                logger.warning(f"Failed to generate DataWrapper chart for metric_id {metric_id}: {str(e)}")
            
            # Create the tabbed interface
            chart_id = f"chart_{metric_id}_{district}_{period_type}"
            
            # Build the DataWrapper tab HTML if URL exists
            dw_tab_html = ""
            if dw_chart_url:
                dw_tab_html = f'''
            <div id="{chart_id}_dw" class="chart-tab-panel">
                <div class="datawrapper-chart-embed" style="display: flex; justify-content: center;">
                    <iframe src="{dw_chart_url}"
                            title="Time Series Chart - Metric {metric_id}"
                            style="width: 1000px; height: 600px; border: none; display: block;" 
                            frameborder="0">
                    </iframe>
                </div>
            </div>
            '''
            
            # Build the DataWrapper button HTML if URL exists
            dw_button_html = ""
            if dw_chart_url:
                dw_button_html = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            tabs_html = f'''
<div class="chart-container" style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="chart-tabs-container" id="{chart_id}_container" style="max-width: 1000px; width: 100%;">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('{chart_id}', 'local')">Transparent SF</button>
            {dw_button_html}
        </div>
        <div class="chart-tab-content">
            <div id="{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}"
                        style="width: 1000px; height: 600px; border: none; display: block; margin: 0 auto;" 
                        frameborder="0" 
                        scrolling="no"
                        title="Time Series Chart - Metric {metric_id}">
                </iframe>
            </div>
            {dw_tab_html}
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Replace anomaly chart references with switchable tabs
        def replace_anomaly_tabs(match):
            anomaly_id = match.group(1)
            
            logger.info(f"Creating switchable tabs for anomaly ID: {anomaly_id}")
            
            # Try to generate a DataWrapper chart
            dw_chart_url = None
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                dw_chart_url = generate_anomaly_chart_from_id(anomaly_id)
                if not dw_chart_url:
                    logger.warning(f"DataWrapper chart generation returned None for anomaly {anomaly_id}")
            except Exception as e:
                logger.warning(f"Failed to generate DataWrapper chart for anomaly {anomaly_id}: {str(e)}")
                # Continue without DataWrapper chart - the local chart will still work
            
            # Create the tabbed interface
            chart_id = f"anomaly_{anomaly_id}"
            
            # Build the DataWrapper tab HTML if URL exists
            dw_tab_html = ""
            if dw_chart_url:
                dw_tab_html = f'''
            <div id="{chart_id}_dw" class="chart-tab-panel">
                <div class="datawrapper-chart-embed" style="display: flex; justify-content: center;">
                    <iframe src="{dw_chart_url}"
                            title="Anomaly {anomaly_id}: Trend Analysis"
                            style="width: 1000px; height: 600px; border: none; display: block;" 
                            frameborder="0">
                    </iframe>
                </div>
            </div>
            '''
            
            # Build the DataWrapper button HTML if URL exists
            dw_button_html = ""
            if dw_chart_url:
                dw_button_html = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            tabs_html = f'''
<div class="chart-container" style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="chart-tabs-container" id="{chart_id}_container" style="max-width: 1000px; width: 100%;">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('{chart_id}', 'local')">Transparent SF</button>
            {dw_button_html}
        </div>
        <div class="chart-tab-content">
            <div id="{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section"
                        style="width: 1000px; height: 600px; border: none; display: block; margin: 0 auto;" 
                        frameborder="0" 
                        scrolling="no"
                        title="Anomaly Analysis - ID {anomaly_id}">
                </iframe>
            </div>
            {dw_tab_html}
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Replace map chart references with switchable tabs
        def replace_map_tabs(match):
            map_id = match.group(1)
            
            logger.info(f"Creating switchable tabs for map ID: {map_id}")
            
            # Try to get existing map URL first
            dw_chart_url = get_existing_map_url(map_id)
            
            if not dw_chart_url:
                # If no existing map URL found, try to create/publish the map
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    dw_chart_url = create_datawrapper_map(map_id)
                except Exception as e:
                    logger.warning(f"Failed to create DataWrapper map for {map_id}: {str(e)}")
            
            # Create the tabbed interface
            chart_id = f"map_{map_id}"
            
            # Build the DataWrapper tab HTML if URL exists
            dw_tab_html = ""
            if dw_chart_url:
                dw_tab_html = f'''
            <div id="{chart_id}_dw" class="chart-tab-panel">
                <div class="datawrapper-chart-embed" style="display: flex; justify-content: center;">
                    <iframe src="{dw_chart_url}"
                            title="Map {map_id}"
                            style="width: 600px; height: 400px; border: none; display: block;" 
                            frameborder="0">
                    </iframe>
                </div>
            </div>
            '''
            
            # Build the DataWrapper button HTML if URL exists
            dw_button_html = ""
            if dw_chart_url:
                dw_button_html = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            tabs_html = f'''
<div class="chart-container" style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="chart-tabs-container" id="{chart_id}_container" style="max-width: 600px; width: 100%;">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('{chart_id}', 'local')">Transparent SF</button>
            {dw_button_html}
        </div>
        <div class="chart-tab-content">
            <div id="{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/backend/map-chart?map_id={map_id}"
                        style="width: 600px; height: 400px; border: none; display: block; margin: 0 auto;" 
                        frameborder="0" 
                        scrolling="no"
                        title="Map - ID {map_id}">
                </iframe>
            </div>
            {dw_tab_html}
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Replace time series ID chart references with switchable tabs
        def replace_time_series_id_tabs(match):
            chart_id = match.group(1)
            
            logger.info(f"Creating switchable tabs for time_series_id: {chart_id}")
            
            # For time_series_id, we don't have a DataWrapper equivalent, so just show local
            tabs_html = f'''
<div class="chart-container">
    <div class="chart-tabs-container" id="chart_{chart_id}_container">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('chart_{chart_id}', 'local')">Transparent SF</button>
        </div>
        <div class="chart-tab-content">
            <div id="chart_{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/backend/time-series-chart?chart_id={chart_id}"
                        style="width: 100%; height: 600px; border: none;" 
                        frameborder="0" 
                        scrolling="yes"
                        title="Time Series Chart - ID {chart_id}">
                </iframe>
            </div>
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Apply all replacements
        report_html = re.sub(time_series_pattern, replace_time_series_tabs, report_html)
        report_html = re.sub(anomaly_pattern, replace_anomaly_tabs, report_html)
        report_html = re.sub(map_pattern, replace_map_tabs, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id_tabs, report_html)
        
        # Add references to external CSS and JavaScript files
        css_and_js = '''
<link rel="stylesheet" href="/static/chart-tabs.css">
<script src="/static/chart-tabs.js"></script>
'''
        
        # Insert the CSS and JavaScript at the end of the head section
        if '<head>' in report_html:
            report_html = report_html.replace('</head>', css_and_js + '</head>')
        else:
            # If no head section, add it at the beginning
            report_html = css_and_js + report_html
        
        # Write the updated report back to the file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references with switchable tabs in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references_with_tabs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def expand_chart_references_with_auto_dw_generation(report_path):
    """
    Process a report file and replace simplified chart references with switchable tabs
    showing both local/internal charts and DataWrapper charts.
    This version auto-generates DataWrapper charts for any charts that don't already have DW URLs.
    This is used in the revision step to ensure all charts have DataWrapper versions.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references with auto DataWrapper generation in report: {report_path}")
    
    try:
        # Import the request_chart_image function from monthly_report
        from monthly_report import request_chart_image, get_existing_map_url
        
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*'
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*'
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        
        # Replace time series chart references with switchable tabs (auto-generate DW if needed)
        def replace_time_series_auto_dw(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Creating switchable tabs with auto DW generation for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            # Always try to generate a DataWrapper chart
            dw_chart_url = None
            try:
                dw_chart_url = request_chart_image('time_series', {
                    'metric_id': metric_id,
                    'district': district,
                    'period_type': period_type
                })
                if dw_chart_url and 'datawrapper' not in dw_chart_url:
                    dw_chart_url = None
                elif dw_chart_url:
                    logger.info(f"Successfully generated/retrieved DataWrapper chart for metric_id {metric_id}: {dw_chart_url}")
            except Exception as e:
                logger.warning(f"Failed to generate DataWrapper chart for metric_id {metric_id}: {str(e)}")
            
            # Create the tabbed interface
            chart_id = f"chart_{metric_id}_{district}_{period_type}"
            
            # Build the DataWrapper tab HTML if URL exists
            dw_tab_html = ""
            if dw_chart_url:
                dw_tab_html = f'''
            <div id="{chart_id}_dw" class="chart-tab-panel">
                <div class="datawrapper-chart-embed" style="display: flex; justify-content: center;">
                    <iframe src="{dw_chart_url}"
                            title="Time Series Chart - Metric {metric_id}"
                            style="width: 1000px; height: 600px; border: none; display: block;" 
                            frameborder="0">
                    </iframe>
                </div>
            </div>
            '''
            
            # Build the DataWrapper button HTML if URL exists
            dw_button_html = ""
            if dw_chart_url:
                dw_button_html = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            tabs_html = f'''
<div class="chart-container" style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="chart-tabs-container" id="{chart_id}_container" style="max-width: 1000px; width: 100%;">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('{chart_id}', 'local')">Transparent SF</button>
            {dw_button_html}
        </div>
        <div class="chart-tab-content">
            <div id="{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period_type}"
                        style="width: 1000px; height: 600px; border: none; display: block; margin: 0 auto;" 
                        frameborder="0" 
                        scrolling="no"
                        title="Time Series Chart - Metric {metric_id}">
                </iframe>
            </div>
            {dw_tab_html}
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Replace anomaly chart references with switchable tabs (auto-generate DW if needed)
        def replace_anomaly_auto_dw(match):
            anomaly_id = match.group(1)
            
            logger.info(f"Creating switchable tabs with auto DW generation for anomaly ID: {anomaly_id}")
            
            # Always try to generate a DataWrapper chart
            dw_chart_url = None
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                dw_chart_url = generate_anomaly_chart_from_id(anomaly_id)
                if dw_chart_url:
                    logger.info(f"Successfully generated DataWrapper chart for anomaly {anomaly_id}: {dw_chart_url}")
            except Exception as e:
                logger.warning(f"Failed to generate DataWrapper chart for anomaly {anomaly_id}: {str(e)}")
            
            # Create the tabbed interface
            chart_id = f"anomaly_{anomaly_id}"
            
            # Build the DataWrapper tab HTML if URL exists
            dw_tab_html = ""
            if dw_chart_url:
                dw_tab_html = f'''
            <div id="{chart_id}_dw" class="chart-tab-panel">
                <div class="datawrapper-chart-embed" style="display: flex; justify-content: center;">
                    <iframe src="{dw_chart_url}"
                            title="Anomaly {anomaly_id}: Trend Analysis"
                            style="width: 1000px; height: 600px; border: none; display: block;" 
                            frameborder="0">
                    </iframe>
                </div>
            </div>
            '''
            
            # Build the DataWrapper button HTML if URL exists
            dw_button_html = ""
            if dw_chart_url:
                dw_button_html = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            tabs_html = f'''
<div class="chart-container" style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="chart-tabs-container" id="{chart_id}_container" style="max-width: 1000px; width: 100%;">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('{chart_id}', 'local')">Transparent SF</button>
            {dw_button_html}
        </div>
        <div class="chart-tab-content">
            <div id="{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/anomaly-analyzer/anomaly-chart?id={anomaly_id}#chart-section"
                        style="width: 1000px; height: 600px; border: none; display: block; margin: 0 auto;" 
                        frameborder="0" 
                        scrolling="no"
                        title="Anomaly Analysis - ID {anomaly_id}">
                </iframe>
            </div>
            {dw_tab_html}
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Replace map chart references with switchable tabs (auto-generate DW if needed)
        def replace_map_auto_dw(match):
            map_id = match.group(1)
            
            logger.info(f"Creating switchable tabs with auto DW generation for map ID: {map_id}")
            
            # Always try to get or create a DataWrapper map
            dw_chart_url = get_existing_map_url(map_id)
            
            if not dw_chart_url:
                # If no existing map URL found, try to create/publish the map
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    dw_chart_url = create_datawrapper_map(map_id)
                    if dw_chart_url:
                        logger.info(f"Successfully created DataWrapper map for {map_id}: {dw_chart_url}")
                except Exception as e:
                    logger.warning(f"Failed to create DataWrapper map for {map_id}: {str(e)}")
            else:
                logger.info(f"Using existing DataWrapper map for {map_id}: {dw_chart_url}")
            
            # Create the tabbed interface
            chart_id = f"map_{map_id}"
            
            # Build the DataWrapper tab HTML if URL exists
            dw_tab_html = ""
            if dw_chart_url:
                dw_tab_html = f'''
            <div id="{chart_id}_dw" class="chart-tab-panel">
                <div class="datawrapper-chart-embed" style="display: flex; justify-content: center;">
                    <iframe src="{dw_chart_url}"
                            title="Map {map_id}"
                            style="width: 600px; height: 400px; border: none; display: block;" 
                            frameborder="0">
                    </iframe>
                </div>
            </div>
            '''
            
            # Build the DataWrapper button HTML if URL exists
            dw_button_html = ""
            if dw_chart_url:
                dw_button_html = f'<button class="chart-tab-btn" onclick="switchChartTab(\'{chart_id}\', \'dw\')">DataWrapper</button>'
            
            tabs_html = f'''
<div class="chart-container" style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="chart-tabs-container" id="{chart_id}_container" style="max-width: 600px; width: 100%;">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('{chart_id}', 'local')">Transparent SF</button>
            {dw_button_html}
        </div>
        <div class="chart-tab-content">
            <div id="{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/backend/map-chart?map_id={map_id}"
                        style="width: 600px; height: 400px; border: none; display: block; margin: 0 auto;" 
                        frameborder="0" 
                        scrolling="no"
                        title="Map - ID {map_id}">
                </iframe>
            </div>
            {dw_tab_html}
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Replace time series ID chart references with switchable tabs
        def replace_time_series_id_auto_dw(match):
            chart_id = match.group(1)
            
            logger.info(f"Creating switchable tabs for time_series_id: {chart_id}")
            
            # For time_series_id, we don't have a DataWrapper equivalent, so just show local
            tabs_html = f'''
<div class="chart-container">
    <div class="chart-tabs-container" id="chart_{chart_id}_container">
        <div class="chart-tabs-header">
            <button class="chart-tab-btn active" onclick="switchChartTab('chart_{chart_id}', 'local')">Transparent SF</button>
        </div>
        <div class="chart-tab-content">
            <div id="chart_{chart_id}_local" class="chart-tab-panel active">
                <iframe src="/backend/time-series-chart?chart_id={chart_id}"
                        style="width: 100%; height: 600px; border: none;" 
                        frameborder="0" 
                        scrolling="yes"
                        title="Time Series Chart - ID {chart_id}">
                </iframe>
            </div>
        </div>
    </div>
</div>
'''
            return tabs_html
        
        # Apply all replacements
        report_html = re.sub(time_series_pattern, replace_time_series_auto_dw, report_html)
        report_html = re.sub(anomaly_pattern, replace_anomaly_auto_dw, report_html)
        report_html = re.sub(map_pattern, replace_map_auto_dw, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id_auto_dw, report_html)
        
        # Add references to external CSS and JavaScript files
        css_and_js = '''
<link rel="stylesheet" href="/static/chart-tabs.css">
<script src="/static/chart-tabs.js"></script>
'''
        
        # Insert the CSS and JavaScript at the end of the head section
        if '<head>' in report_html:
            report_html = report_html.replace('</head>', css_and_js + '</head>')
        else:
            # If no head section, add it at the beginning
            report_html = css_and_js + report_html
        
        # Write the updated report back to the file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references with auto DataWrapper generation in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references_with_auto_dw_generation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

def expand_chart_references_for_email(report_path):
    """
    Process a report file and replace chart references with DataWrapper URLs
    ready for copy-paste in email/newsletter format.
    This version shows only the DataWrapper URLs in a clean format for email use.
    
    Args:
        report_path: Path to the report file to process
        
    Returns:
        Path to the processed report file
    """
    logger.info(f"Expanding chart references for email format in report: {report_path}")
    
    try:
        # Import the request_chart_image function from monthly_report
        from monthly_report import request_chart_image, get_existing_map_url
        
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Define patterns for simplified chart references (original format)
        time_series_pattern = r'\[CHART:time_series:(\d+):(\d+):(\w+)\]\s*[.,;:]*\s*'
        time_series_id_pattern = r'\[CHART:time_series_id:(\d+)\]\s*[.,;:]*\s*'
        anomaly_pattern = r'\[CHART:anomaly:([a-zA-Z0-9]+)\]\s*[.,;:]*\s*'
        map_pattern = r'\[CHART:map:([a-zA-Z0-9\-]+)\]\s*[.,;:]*\s*'
        
        # Define patterns for already-expanded HTML charts (more comprehensive)
        expanded_time_series_pattern = r'<div class="chart-container"[^>]*>.*?<iframe[^>]*src="/backend/time-series-chart\?chart_id=(\d+)"[^>]*>.*?</div>'
        expanded_anomaly_pattern = r'<div class="chart-container"[^>]*>.*?<iframe[^>]*src="/anomaly-analyzer/anomaly-chart\?id=([a-zA-Z0-9]+)"[^>]*>.*?</div>'
        expanded_map_pattern = r'<div class="chart-container"[^>]*>.*?<iframe[^>]*src="/backend/map-chart\?map_id=([a-zA-Z0-9\-]+)"[^>]*>.*?</div>'
        
        # More specific patterns for the nested structure
        expanded_time_series_nested_pattern = r'<div class="chart-container"[^>]*>.*?<div class="chart-tabs-container"[^>]*>.*?<iframe[^>]*src="/backend/time-series-chart\?chart_id=(\d+)"[^>]*>.*?</div>'
        expanded_anomaly_nested_pattern = r'<div class="chart-container"[^>]*>.*?<div class="chart-tabs-container"[^>]*>.*?<iframe[^>]*src="/anomaly-analyzer/anomaly-chart\?id=([a-zA-Z0-9]+)"[^>]*>.*?</div>'
        expanded_map_nested_pattern = r'<div class="chart-container"[^>]*>.*?<div class="chart-tabs-container"[^>]*>.*?<iframe[^>]*src="/backend/map-chart\?map_id=([a-zA-Z0-9\-]+)"[^>]*>.*?</div>'
        
        # Also try to match the entire chart container structure
        expanded_chart_container_pattern = r'<div class="chart-container"[^>]*>.*?</div>'
        
        # Replace time series chart references with DataWrapper URLs for email
        def replace_time_series_email(match):
            metric_id = match.group(1)
            district = match.group(2)
            period_type = match.group(3)
            
            logger.info(f"Creating email format for metric_id: {metric_id}, district: {district}, period: {period_type}")
            
            # Try to get DataWrapper chart URL
            dw_chart_url = None
            try:
                dw_chart_url = request_chart_image('time_series', {
                    'metric_id': metric_id,
                    'district': district,
                    'period_type': period_type
                })
                if dw_chart_url and 'datawrapper' not in dw_chart_url:
                    dw_chart_url = None
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for metric_id {metric_id}: {str(e)}")
            
            if dw_chart_url:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Time Series Chart - Metric {metric_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
            else:
                # Fallback - no DataWrapper URL available
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Time Series Chart - Metric {metric_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            return email_html
        
        # Replace anomaly chart references with DataWrapper URLs for email
        def replace_anomaly_email(match):
            anomaly_id = match.group(1)
            
            logger.info(f"Creating email format for anomaly ID: {anomaly_id}")
            
            # Try to get DataWrapper chart URL
            dw_chart_url = None
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                dw_chart_url = generate_anomaly_chart_from_id(anomaly_id)
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for anomaly {anomaly_id}: {str(e)}")
            
            if dw_chart_url:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Anomaly Analysis - ID {anomaly_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
            else:
                # Fallback - no DataWrapper URL available
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Anomaly Analysis - ID {anomaly_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            return email_html
        
        # Replace map chart references with DataWrapper URLs for email
        def replace_map_email(match):
            map_id = match.group(1)
            
            logger.info(f"Creating email format for map ID: {map_id}")
            
            # Try to get DataWrapper map URL
            dw_chart_url = get_existing_map_url(map_id)
            
            if not dw_chart_url:
                # If no existing map URL found, try to create/publish the map
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    dw_chart_url = create_datawrapper_map(map_id)
                except Exception as e:
                    logger.warning(f"Failed to create DataWrapper map for {map_id}: {str(e)}")
            
            if dw_chart_url:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Map - ID {map_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
            else:
                # Fallback - no DataWrapper URL available
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Map - ID {map_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            return email_html
        
        # Replace time series ID chart references with local charts for email
        def replace_time_series_id_email(match):
            chart_id = match.group(1)
            
            logger.info(f"Creating email format for time_series_id: {chart_id}")
            
            # For time_series_id, we don't have a DataWrapper equivalent, so show note only
            email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Time Series Chart - ID {chart_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart type.</p>
</div>
'''
            return email_html
        
        # Replace expanded HTML charts with email format
        def replace_expanded_time_series_email(match):
            chart_id = match.group(1)
            logger.info(f"Creating email format for expanded time_series_id: {chart_id}")
            
            # For time_series_id, we don't have a DataWrapper equivalent, so show note only
            email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Time Series Chart - ID {chart_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart type.</p>
</div>
'''
            return email_html
        
        def replace_expanded_anomaly_email(match):
            anomaly_id = match.group(1)
            logger.info(f"Creating email format for expanded anomaly ID: {anomaly_id}")
            
            # Try to get DataWrapper chart URL
            dw_chart_url = None
            try:
                from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                dw_chart_url = generate_anomaly_chart_from_id(anomaly_id)
            except Exception as e:
                logger.warning(f"Failed to get DataWrapper chart for anomaly {anomaly_id}: {str(e)}")
            
            if dw_chart_url:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Anomaly Analysis - ID {anomaly_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
            else:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Anomaly Analysis - ID {anomaly_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            return email_html
        
        def replace_expanded_map_email(match):
            map_id = match.group(1)
            logger.info(f"Creating email format for expanded map ID: {map_id}")
            
            # Try to get DataWrapper map URL
            dw_chart_url = get_existing_map_url(map_id)
            
            if not dw_chart_url:
                # If no existing map URL found, try to create/publish the map
                try:
                    from tools.gen_map_dw import create_datawrapper_map
                    dw_chart_url = create_datawrapper_map(map_id)
                except Exception as e:
                    logger.warning(f"Failed to create DataWrapper map for {map_id}: {str(e)}")
            
            if dw_chart_url:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Map - ID {map_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
            else:
                email_html = f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Map - ID {map_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            return email_html
        
        # Fallback function to handle any remaining chart containers
        def replace_any_chart_container(match):
            container_html = match.group(0)
            logger.info(f"Processing chart container: {container_html[:100]}...")
            
            # Try to extract chart information from the container
            time_series_match = re.search(r'/backend/time-series-chart\?chart_id=(\d+)', container_html)
            anomaly_match = re.search(r'/anomaly-analyzer/anomaly-chart\?id=([a-zA-Z0-9]+)', container_html)
            map_match = re.search(r'/backend/map-chart\?map_id=([a-zA-Z0-9\-]+)', container_html)
            
            if time_series_match:
                chart_id = time_series_match.group(1)
                return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Time Series Chart - ID {chart_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart type.</p>
</div>
'''
            elif anomaly_match:
                anomaly_id = anomaly_match.group(1)
                # Try to get DataWrapper chart URL
                dw_chart_url = None
                try:
                    from tools.gen_anomaly_chart_dw import generate_anomaly_chart_from_id
                    dw_chart_url = generate_anomaly_chart_from_id(anomaly_id)
                except Exception as e:
                    logger.warning(f"Failed to get DataWrapper chart for anomaly {anomaly_id}: {str(e)}")
                
                if dw_chart_url:
                    return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Anomaly Analysis - ID {anomaly_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
                else:
                    return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Anomaly Analysis - ID {anomaly_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            elif map_match:
                map_id = map_match.group(1)
                # Try to get DataWrapper map URL
                dw_chart_url = get_existing_map_url(map_id)
                
                if not dw_chart_url:
                    try:
                        from tools.gen_map_dw import create_datawrapper_map
                        dw_chart_url = create_datawrapper_map(map_id)
                    except Exception as e:
                        logger.warning(f"Failed to create DataWrapper map for {map_id}: {str(e)}")
                
                if dw_chart_url:
                    return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">Map - ID {map_id}</h4>
    <div style="margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <strong>DataWrapper URL:</strong><br>
        {dw_chart_url}
    </div>
</div>
'''
                else:
                    return f'''
<div class="chart-container" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
    <h4 style="margin: 0 0 10px 0; color: #856404;">Map - ID {map_id}</h4>
    <p style="margin: 0; color: #856404; font-size: 14px;">DataWrapper version not available for this chart.</p>
</div>
'''
            else:
                # If we can't identify the chart type, just remove the container
                logger.warning("Could not identify chart type in container, removing it")
                return ""
        
        # Apply all replacements - try nested patterns first, then simple patterns, then original patterns, then fallback
        report_html = re.sub(expanded_time_series_nested_pattern, replace_expanded_time_series_email, report_html, flags=re.DOTALL)
        report_html = re.sub(expanded_anomaly_nested_pattern, replace_expanded_anomaly_email, report_html, flags=re.DOTALL)
        report_html = re.sub(expanded_map_nested_pattern, replace_expanded_map_email, report_html, flags=re.DOTALL)
        
        # Try simple patterns as fallback
        report_html = re.sub(expanded_time_series_pattern, replace_expanded_time_series_email, report_html, flags=re.DOTALL)
        report_html = re.sub(expanded_anomaly_pattern, replace_expanded_anomaly_email, report_html, flags=re.DOTALL)
        report_html = re.sub(expanded_map_pattern, replace_expanded_map_email, report_html, flags=re.DOTALL)
        
        # Also try original patterns in case some charts weren't expanded
        report_html = re.sub(time_series_pattern, replace_time_series_email, report_html)
        report_html = re.sub(anomaly_pattern, replace_anomaly_email, report_html)
        report_html = re.sub(map_pattern, replace_map_email, report_html)
        report_html = re.sub(time_series_id_pattern, replace_time_series_id_email, report_html)
        
        # Fallback: catch any remaining chart containers
        report_html = re.sub(expanded_chart_container_pattern, replace_any_chart_container, report_html, flags=re.DOTALL)
        
        # Write the updated report back to the file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Expanded chart references for email format in report: {report_path}")
        return report_path
        
    except Exception as e:
        error_msg = f"Error in expand_chart_references_for_email: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return report_path

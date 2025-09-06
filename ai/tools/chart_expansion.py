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

"""
Chart Placeholder Processor for Server-Side Processing

This module provides server-side chart placeholder processing capabilities
for use in Flask routes and other backend processing.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ChartProcessor:
    """
    Server-side chart placeholder processor
    """
    
    def __init__(self, options: Optional[Dict] = None):
        """
        Initialize the chart processor with options
        
        Args:
            options: Dictionary of configuration options
        """
        self.options = {
            'default_chart_height': '400px',
            'default_chart_width': '100%',
            'chart_container_class': 'chart-container',
            'chart_container_style': 'margin: 10px 0;',
            'base_url': '',  # Can be set for absolute URLs
            **(options or {})
        }
    
    def process_placeholders(self, content: str) -> str:
        """
        Process chart placeholders in content and replace with iframe HTML
        
        Args:
            content: The content containing chart placeholders
            
        Returns:
            Content with chart placeholders replaced by iframes
        """
        if not content:
            return content
        
        # Regular expression to match chart placeholders: [CHART:type:param1:param2:param3]
        chart_regex = r'\[CHART:([^:]+):([^\]]+)\]'
        
        def replace_chart(match):
            chart_type = match.group(1)
            params = match.group(2)
            return self.create_chart_element(chart_type, params)
        
        return re.sub(chart_regex, replace_chart, content)
    
    def create_chart_element(self, chart_type: str, params: str) -> str:
        """
        Create chart element HTML based on chart type and parameters
        
        Args:
            chart_type: The type of chart (anomaly, time_series, time_series_id, map)
            params: Colon-separated parameters for the chart
            
        Returns:
            HTML for the chart iframe
        """
        param_parts = params.split(':')
        
        if chart_type == 'anomaly':
            return self.create_anomaly_chart(param_parts)
        elif chart_type == 'time_series':
            return self.create_time_series_chart(param_parts)
        elif chart_type == 'time_series_id':
            return self.create_time_series_chart_by_id(param_parts)
        elif chart_type == 'map':
            return self.create_map_chart(param_parts)
        else:
            logger.warning(f"Unknown chart type: {chart_type}")
            return f'[CHART:{chart_type}:{params}]'  # Return original placeholder
    
    def create_anomaly_chart(self, param_parts: List[str]) -> str:
        """
        Create anomaly chart iframe
        
        Args:
            param_parts: [anomaly_id]
            
        Returns:
            HTML for anomaly chart iframe
        """
        if not param_parts or not param_parts[0]:
            logger.error('Missing anomaly_id for anomaly chart')
            return '[CHART:anomaly:MISSING_ID]'
        
        anomaly_id = param_parts[0]
        src = f"{self.options['base_url']}/anomaly-analyzer/anomaly-chart?id={anomaly_id}"
        title = f"Anomaly Chart {anomaly_id}"
        
        return self.create_iframe_wrapper(src, title)
    
    def create_time_series_chart(self, param_parts: List[str]) -> str:
        """
        Create time series chart iframe using metric parameters
        
        Args:
            param_parts: [metric_id, district, period]
            
        Returns:
            HTML for time series chart iframe
        """
        if not param_parts or not param_parts[0]:
            logger.error('Missing metric_id for time series chart')
            return '[CHART:time_series:MISSING_METRIC_ID]'
        
        metric_id = param_parts[0]
        district = param_parts[1] if len(param_parts) > 1 else '0'
        period = param_parts[2] if len(param_parts) > 2 else 'month'
        
        src = f"{self.options['base_url']}/backend/time-series-chart?metric_id={metric_id}&district={district}&period_type={period}"
        title = f"Time Series Chart - Metric {metric_id}"
        
        return self.create_iframe_wrapper(src, title)
    
    def create_time_series_chart_by_id(self, param_parts: List[str]) -> str:
        """
        Create time series chart iframe using chart ID
        
        Args:
            param_parts: [chart_id]
            
        Returns:
            HTML for time series chart iframe
        """
        if not param_parts or not param_parts[0]:
            logger.error('Missing chart_id for time series chart')
            return '[CHART:time_series_id:MISSING_CHART_ID]'
        
        chart_id = param_parts[0]
        src = f"{self.options['base_url']}/backend/time-series-chart?chart_id={chart_id}"
        title = f"Time Series Chart {chart_id}"
        
        return self.create_iframe_wrapper(src, title)
    
    def create_map_chart(self, param_parts: List[str]) -> str:
        """
        Create map chart iframe
        
        Args:
            param_parts: [map_id]
            
        Returns:
            HTML for map chart iframe
        """
        if not param_parts or not param_parts[0]:
            logger.error('Missing map_id for map chart')
            return '[CHART:map:MISSING_MAP_ID]'
        
        map_id = param_parts[0]
        # Note: Update this URL based on your actual map chart endpoint
        src = f"{self.options['base_url']}/backend/map-chart?id={map_id}"
        title = f"Map Chart {map_id}"
        
        return self.create_iframe_wrapper(src, title)
    
    def create_iframe_wrapper(self, src: str, title: str) -> str:
        """
        Create iframe wrapper HTML
        
        Args:
            src: The iframe source URL
            title: The iframe title for accessibility
            
        Returns:
            Complete iframe HTML with wrapper
        """
        return f'''<div class="{self.options['chart_container_class']}" style="{self.options['chart_container_style']}">
    <iframe src="{src}" 
            title="{title}"
            style="width: {self.options['default_chart_width']}; height: {self.options['default_chart_height']}; border: none;" 
            frameborder="0" 
            scrolling="no"
            allowfullscreen>
    </iframe>
</div>'''
    
    def extract_placeholders(self, content: str) -> List[Dict]:
        """
        Extract chart placeholders from content (useful for analysis)
        
        Args:
            content: The content to analyze
            
        Returns:
            List of chart placeholder dictionaries
        """
        if not content:
            return []
        
        chart_regex = r'\[CHART:([^:]+):([^\]]+)\]'
        placeholders = []
        
        for match in re.finditer(chart_regex, content):
            placeholders.append({
                'full_match': match.group(0),
                'chart_type': match.group(1),
                'params': match.group(2).split(':'),
                'position': match.start()
            })
        
        return placeholders
    
    def validate_placeholders(self, content: str) -> Dict:
        """
        Validate chart placeholders in content
        
        Args:
            content: The content to validate
            
        Returns:
            Dictionary with validation results (valid, errors, warnings)
        """
        placeholders = self.extract_placeholders(content)
        results = {
            'valid': [],
            'errors': [],
            'warnings': []
        }
        
        for placeholder in placeholders:
            chart_type = placeholder['chart_type']
            params = placeholder['params']
            full_match = placeholder['full_match']
            
            if chart_type == 'anomaly':
                if not params or not params[0]:
                    results['errors'].append(f"Missing anomaly_id in: {full_match}")
                else:
                    results['valid'].append(full_match)
            
            elif chart_type == 'time_series':
                if not params or not params[0]:
                    results['errors'].append(f"Missing metric_id in: {full_match}")
                else:
                    results['valid'].append(full_match)
                    if len(params) > 3:
                        results['warnings'].append(f"Extra parameters in: {full_match}")
            
            elif chart_type == 'time_series_id':
                if not params or not params[0]:
                    results['errors'].append(f"Missing chart_id in: {full_match}")
                else:
                    results['valid'].append(full_match)
            
            elif chart_type == 'map':
                if not params or not params[0]:
                    results['errors'].append(f"Missing map_id in: {full_match}")
                else:
                    results['valid'].append(full_match)
            
            else:
                results['warnings'].append(f"Unknown chart type: {full_match}")
        
        return results


# Create default instance
default_chart_processor = ChartProcessor()


def process_chart_placeholders(content: str, options: Optional[Dict] = None) -> str:
    """
    Convenience function to process chart placeholders with optional custom options
    
    Args:
        content: Content containing chart placeholders
        options: Optional configuration options
        
    Returns:
        Content with chart placeholders replaced by iframes
    """
    if options:
        processor = ChartProcessor(options)
        return processor.process_placeholders(content)
    else:
        return default_chart_processor.process_placeholders(content)


def validate_chart_placeholders(content: str) -> Dict:
    """
    Convenience function to validate chart placeholders
    
    Args:
        content: Content to validate
        
    Returns:
        Validation results dictionary
    """
    return default_chart_processor.validate_placeholders(content)


# Usage examples:
if __name__ == "__main__":
    # Example usage
    sample_content = """
    Here's an anomaly: [CHART:anomaly:12345]
    
    And a time series: [CHART:time_series:1:0:month]
    
    Chart by ID: [CHART:time_series_id:67890]
    
    A map: [CHART:map:map123]
    """
    
    processor = ChartProcessor()
    processed = processor.process_placeholders(sample_content)
    print("Processed content:")
    print(processed)
    
    validation = processor.validate_placeholders(sample_content)
    print("\nValidation results:")
    print(validation) 
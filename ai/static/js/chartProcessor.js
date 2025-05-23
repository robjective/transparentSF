/**
 * Chart Placeholder Processor Utility
 * 
 * This module provides functions to process chart placeholders in content
 * and replace them with actual chart iframes.
 * 
 * Supported placeholder formats:
 * - [CHART:anomaly:anomaly_id]
 * - [CHART:time_series:metric_id:district:period]
 * - [CHART:time_series_id:chart_id]
 * - [CHART:map:map_id]
 */

class ChartProcessor {
    constructor(options = {}) {
        this.options = {
            defaultChartHeight: '400px',
            defaultChartWidth: '100%',
            chartContainerClass: 'chart-container',
            chartContainerStyle: 'margin: 10px 0;',
            ...options
        };
    }

    /**
     * Process chart placeholders in content and replace with iframe HTML
     * @param {string} content - The content containing chart placeholders
     * @returns {string} - Content with chart placeholders replaced by iframes
     */
    processPlaceholders(content) {
        if (!content) return content;
        
        // Regular expression to match chart placeholders: [CHART:type:param1:param2:param3]
        const chartRegex = /\[CHART:([^:]+):([^\]]+)\]/g;
        
        return content.replace(chartRegex, (match, chartType, params) => {
            return this.createChartElement(chartType, params);
        });
    }

    /**
     * Create chart element HTML based on chart type and parameters
     * @param {string} chartType - The type of chart (anomaly, time_series, time_series_id, map)
     * @param {string} params - Colon-separated parameters for the chart
     * @returns {string} - HTML for the chart iframe
     */
    createChartElement(chartType, params) {
        const paramParts = params.split(':');
        
        switch (chartType) {
            case 'anomaly':
                return this.createAnomalyChart(paramParts);
            
            case 'time_series':
                return this.createTimeSeriesChart(paramParts);
            
            case 'time_series_id':
                return this.createTimeSeriesChartById(paramParts);
            
            case 'map':
                return this.createMapChart(paramParts);
            
            default:
                console.warn(`Unknown chart type: ${chartType}`);
                return `[CHART:${chartType}:${params}]`; // Return original placeholder
        }
    }

    /**
     * Create anomaly chart iframe
     * @param {string[]} paramParts - [anomaly_id]
     * @returns {string} - HTML for anomaly chart iframe
     */
    createAnomalyChart(paramParts) {
        const anomalyId = paramParts[0];
        if (!anomalyId) {
            console.error('Missing anomaly_id for anomaly chart');
            return '[CHART:anomaly:MISSING_ID]';
        }

        return this.createIframeWrapper(
            `/anomaly-analyzer/anomaly-chart?id=${anomalyId}`,
            `Anomaly Chart ${anomalyId}`
        );
    }

    /**
     * Create time series chart iframe using metric parameters
     * @param {string[]} paramParts - [metric_id, district, period]
     * @returns {string} - HTML for time series chart iframe
     */
    createTimeSeriesChart(paramParts) {
        const metricId = paramParts[0];
        const district = paramParts[1] || '0';
        const period = paramParts[2] || 'month';

        if (!metricId) {
            console.error('Missing metric_id for time series chart');
            return '[CHART:time_series:MISSING_METRIC_ID]';
        }

        return this.createIframeWrapper(
            `/backend/time-series-chart?metric_id=${metricId}&district=${district}&period_type=${period}`,
            `Time Series Chart - Metric ${metricId}`
        );
    }

    /**
     * Create time series chart iframe using chart ID
     * @param {string[]} paramParts - [chart_id]
     * @returns {string} - HTML for time series chart iframe
     */
    createTimeSeriesChartById(paramParts) {
        const chartId = paramParts[0];
        if (!chartId) {
            console.error('Missing chart_id for time series chart');
            return '[CHART:time_series_id:MISSING_CHART_ID]';
        }

        return this.createIframeWrapper(
            `/backend/time-series-chart?chart_id=${chartId}`,
            `Time Series Chart ${chartId}`
        );
    }

    /**
     * Create map chart iframe
     * @param {string[]} paramParts - [map_id]
     * @returns {string} - HTML for map chart iframe
     */
    createMapChart(paramParts) {
        const mapId = paramParts[0];
        if (!mapId) {
            console.error('Missing map_id for map chart');
            return '[CHART:map:MISSING_MAP_ID]';
        }

        // Note: Update this URL based on your actual map chart endpoint
        return this.createIframeWrapper(
            `/backend/map-chart?id=${mapId}`,
            `Map Chart ${mapId}`
        );
    }

    /**
     * Create iframe wrapper HTML
     * @param {string} src - The iframe source URL
     * @param {string} title - The iframe title for accessibility
     * @returns {string} - Complete iframe HTML with wrapper
     */
    createIframeWrapper(src, title) {
        return `<div class="${this.options.chartContainerClass}" style="${this.options.chartContainerStyle}">
            <iframe src="${src}" 
                    title="${title}"
                    style="width: ${this.options.defaultChartWidth}; height: ${this.options.defaultChartHeight}; border: none;" 
                    frameborder="0" 
                    scrolling="no"
                    allowfullscreen>
            </iframe>
        </div>`;
    }

    /**
     * Extract chart placeholders from content (useful for analysis)
     * @param {string} content - The content to analyze
     * @returns {Array} - Array of chart placeholder objects
     */
    extractPlaceholders(content) {
        if (!content) return [];
        
        const chartRegex = /\[CHART:([^:]+):([^\]]+)\]/g;
        const placeholders = [];
        let match;
        
        while ((match = chartRegex.exec(content)) !== null) {
            placeholders.push({
                fullMatch: match[0],
                chartType: match[1],
                params: match[2].split(':'),
                position: match.index
            });
        }
        
        return placeholders;
    }

    /**
     * Validate chart placeholders in content
     * @param {string} content - The content to validate
     * @returns {Object} - Validation results with errors and warnings
     */
    validatePlaceholders(content) {
        const placeholders = this.extractPlaceholders(content);
        const results = {
            valid: [],
            errors: [],
            warnings: []
        };
        
        placeholders.forEach(placeholder => {
            const { chartType, params, fullMatch } = placeholder;
            
            switch (chartType) {
                case 'anomaly':
                    if (!params[0]) {
                        results.errors.push(`Missing anomaly_id in: ${fullMatch}`);
                    } else {
                        results.valid.push(fullMatch);
                    }
                    break;
                
                case 'time_series':
                    if (!params[0]) {
                        results.errors.push(`Missing metric_id in: ${fullMatch}`);
                    } else {
                        results.valid.push(fullMatch);
                        if (params.length > 3) {
                            results.warnings.push(`Extra parameters in: ${fullMatch}`);
                        }
                    }
                    break;
                
                case 'time_series_id':
                    if (!params[0]) {
                        results.errors.push(`Missing chart_id in: ${fullMatch}`);
                    } else {
                        results.valid.push(fullMatch);
                    }
                    break;
                
                case 'map':
                    if (!params[0]) {
                        results.errors.push(`Missing map_id in: ${fullMatch}`);
                    } else {
                        results.valid.push(fullMatch);
                    }
                    break;
                
                default:
                    results.warnings.push(`Unknown chart type: ${fullMatch}`);
            }
        });
        
        return results;
    }
}

// Create default instance
const defaultChartProcessor = new ChartProcessor();

// Export for use in modules or direct access
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChartProcessor, defaultChartProcessor };
}

// Global access for browser environments
if (typeof window !== 'undefined') {
    window.ChartProcessor = ChartProcessor;
    window.chartProcessor = defaultChartProcessor;
    
    // Convenience global function
    window.processChartPlaceholders = function(content) {
        return defaultChartProcessor.processPlaceholders(content);
    };
} 
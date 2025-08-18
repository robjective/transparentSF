-- Migration script to add improved map generation columns to metrics table
-- This addresses the complex query transformation issues by storing structured map data

BEGIN;

-- Add new columns for improved map generation
ALTER TABLE metrics 
ADD COLUMN map_query TEXT,
ADD COLUMN map_filters JSONB DEFAULT '{}'::jsonb,
ADD COLUMN map_config JSONB DEFAULT '{}'::jsonb;

-- Add comments to explain the new columns
COMMENT ON COLUMN metrics.map_query IS 'Simplified base query for map generation without complex transformations';
COMMENT ON COLUMN metrics.map_filters IS 'Structured filters that can be applied programmatically (geometry, date ranges, etc.)';
COMMENT ON COLUMN metrics.map_config IS 'Map-specific configuration including date fields, location fields, and rendering options';

-- Add indexes for the new JSONB columns
CREATE INDEX metrics_map_filters_idx ON metrics USING GIN (map_filters);
CREATE INDEX metrics_map_config_idx ON metrics USING GIN (map_config);

-- Example of the new structure for documentation:
-- map_query: "SELECT *, location FROM endpoint_table"
-- map_filters: {
--   "geometry": {
--     "type": "within_polygon", 
--     "field": "location",
--     "value": "MULTIPOLYGON(...)"
--   },
--   "date_range": {
--     "field": "dba_start_date",
--     "fallback_field": "location_start_date",
--     "fallback_condition": "dba_start_date < '2023-07-01'"
--   },
--   "static_filters": [
--     {"field": "status", "operator": "=", "value": "active"}
--   ]
-- }
-- map_config: {
--   "date_field": "dba_start_date",
--   "location_field": "location",
--   "title_template": "{metric_name}",
--   "supports_districts": true,
--   "supports_time_periods": true,
--   "chart_type_preference": "symbol",
--   "data_point_threshold": 100
-- }

COMMIT;

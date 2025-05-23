-- Add new columns to the maps table
ALTER TABLE maps ADD COLUMN IF NOT EXISTS metric_id TEXT;
ALTER TABLE maps ADD COLUMN IF NOT EXISTS group_field TEXT;
ALTER TABLE maps ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE;

-- Set all existing maps to active
UPDATE maps SET active = TRUE WHERE active IS NULL;

-- Try to populate metric_id column from metadata where possible
UPDATE maps 
SET metric_id = metadata::jsonb->>'metric_id'
WHERE metric_id IS NULL 
  AND metadata IS NOT NULL 
  AND metadata::jsonb->>'metric_id' IS NOT NULL;

-- Set supervisor_district as group_field for existing district maps
UPDATE maps
SET group_field = 'supervisor_district'
WHERE type = 'supervisor_district' AND group_field IS NULL;

-- Log summary of updated rows
SELECT 
  COUNT(*) AS total_maps,
  SUM(CASE WHEN metric_id IS NOT NULL THEN 1 ELSE 0 END) AS maps_with_metric_id,
  SUM(CASE WHEN group_field IS NOT NULL THEN 1 ELSE 0 END) AS maps_with_group_field,
  SUM(CASE WHEN active = TRUE THEN 1 ELSE 0 END) AS active_maps
FROM maps; 
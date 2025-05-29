-- Add new display_order column to metrics table
--
-- 1. Add the column if it does not already exist.
-- 2. Populate the column for existing rows so that the ordering matches the current order
--    (based on id ascending, which reflects the order the metrics are currently stored).
-- 3. Create an index to speed up ordering queries (optional but helpful).

-- Step 1: add the column
ALTER TABLE metrics
ADD COLUMN IF NOT EXISTS display_order INTEGER;

-- Step 2: populate display_order for existing rows using the current order
WITH ordered AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn
    FROM metrics
)
UPDATE metrics m
SET display_order = o.rn
FROM ordered o
WHERE m.id = o.id;

-- Step 3: create an index on the new column
CREATE INDEX IF NOT EXISTS metrics_display_order_idx ON metrics (display_order); 
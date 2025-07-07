-- Add most_recent_data_date column to metrics table
--
-- This column will track the most recent data date for each metric,
-- which will be updated daily when the dashboard metrics are generated.

-- Step 1: Add the column if it does not already exist
ALTER TABLE metrics
ADD COLUMN IF NOT EXISTS most_recent_data_date DATE;

-- Step 2: Create an index to speed up queries on this column
CREATE INDEX IF NOT EXISTS metrics_most_recent_data_date_idx ON metrics (most_recent_data_date);

-- Step 3: Add a comment to document the column's purpose
COMMENT ON COLUMN metrics.most_recent_data_date IS 'The most recent data date available for this metric, updated daily during dashboard generation'; 
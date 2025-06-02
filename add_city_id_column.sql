-- Add city_id column to metrics table and set up San Francisco as default city
--
-- 1. Insert San Francisco into cities table if it doesn't exist
-- 2. Add the city_id column to metrics table
-- 3. Set all existing metrics to reference San Francisco
-- 4. Create an index on the new column

-- Step 1: Insert San Francisco into cities table if it doesn't exist
INSERT INTO cities (city, state_code, place_code, full_name, population)
SELECT 'San Francisco', 'CA', '6075000', 'San Francisco, California', 873965
WHERE NOT EXISTS (
    SELECT 1 FROM cities WHERE city = 'San Francisco' AND state_code = 'CA'
);

-- Step 2: Add the city_id column to metrics table
ALTER TABLE metrics
ADD COLUMN IF NOT EXISTS city_id INTEGER;

-- Step 3: Set all existing metrics to reference San Francisco
UPDATE metrics
SET city_id = (SELECT id FROM cities WHERE city = 'San Francisco' AND state_code = 'CA')
WHERE city_id IS NULL;

-- Step 4: Add foreign key constraint (only if it doesn't exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_metrics_city_id' AND table_name = 'metrics'
    ) THEN
        ALTER TABLE metrics
        ADD CONSTRAINT fk_metrics_city_id
        FOREIGN KEY (city_id) REFERENCES cities(id);
    END IF;
END $$;

-- Step 5: Create indexes on the new column
CREATE INDEX IF NOT EXISTS metrics_city_id_idx ON metrics (city_id);
CREATE INDEX IF NOT EXISTS metrics_city_id_display_order_idx ON metrics (city_id, display_order); 
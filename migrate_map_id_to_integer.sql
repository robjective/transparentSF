-- Migration script to change maps.id from UUID to SERIAL (auto-incrementing integer)
-- This will make the map_id smaller and more manageable

-- Step 1: Create a new temporary column with SERIAL type
ALTER TABLE maps ADD COLUMN new_id SERIAL;

-- Step 2: Update any foreign key references (if they exist)
-- Note: Check if there are any tables that reference maps.id before running this

-- Step 3: Drop the old UUID primary key constraint
ALTER TABLE maps DROP CONSTRAINT maps_pkey;

-- Step 4: Drop the old id column
ALTER TABLE maps DROP COLUMN id;

-- Step 5: Rename the new column to id
ALTER TABLE maps RENAME COLUMN new_id TO id;

-- Step 6: Add the new primary key constraint
ALTER TABLE maps ADD PRIMARY KEY (id);

-- Step 7: Create an index for better performance (optional, since PRIMARY KEY creates one)
-- CREATE INDEX IF NOT EXISTS maps_id_idx ON maps (id);

-- Verify the changes
SELECT 
    column_name, 
    data_type, 
    is_nullable, 
    column_default
FROM information_schema.columns 
WHERE table_name = 'maps' 
AND column_name = 'id';

-- Show sample of updated records
SELECT id, title, type, created_at 
FROM maps 
ORDER BY id 
LIMIT 5; 
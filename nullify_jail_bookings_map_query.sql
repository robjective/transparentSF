-- Nullify map_query for jail booking metrics
-- This removes them from the map generator dropdown

UPDATE metrics 
SET map_query = NULL 
WHERE id IN (5, 26, 27, 28);

-- Verify the update
SELECT id, metric_name, map_query 
FROM metrics 
WHERE id IN (5, 26, 27, 28);






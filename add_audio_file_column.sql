-- Add audio_file column to the reports table
ALTER TABLE reports ADD COLUMN IF NOT EXISTS audio_file TEXT;

-- Log summary of the table after adding the column
SELECT 
  column_name, 
  data_type, 
  is_nullable
FROM information_schema.columns 
WHERE table_name = 'reports' 
ORDER BY ordinal_position; 
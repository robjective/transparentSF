-- Add conversation_history column to eval_results table
-- This migration adds support for storing the full conversation history from the AI agent

ALTER TABLE eval_results 
ADD COLUMN conversation_history JSONB DEFAULT '[]'::jsonb;

-- Add index for the new column
CREATE INDEX eval_results_conversation_history_idx ON eval_results USING GIN (conversation_history); 
-- Add evaluation tables to the database
-- This script should be run once to add the new eval tables

-- Drop existing tables if they exist (in reverse order of dependencies)
DROP TABLE IF EXISTS eval_results CASCADE;
DROP TABLE IF EXISTS evals CASCADE;
DROP TABLE IF EXISTS eval_groups CASCADE;

-- Create eval_groups table
CREATE TABLE eval_groups (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL, -- e.g., 'metrics', 'system', 'features'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create evals table
CREATE TABLE evals (
    id SERIAL PRIMARY KEY,
    group_id INTEGER REFERENCES eval_groups(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    prompt TEXT NOT NULL,
    success_criteria TEXT NOT NULL,
    success_type TEXT NOT NULL, -- 'binary', 'scored', 'multi_step'
    expected_tool_calls JSONB DEFAULT '[]'::jsonb,
    expected_outputs JSONB DEFAULT '[]'::jsonb,
    difficulty_level TEXT DEFAULT 'medium', -- 'easy', 'medium', 'hard'
    estimated_time_minutes INTEGER DEFAULT 5,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create eval_results table
CREATE TABLE eval_results (
    id SERIAL PRIMARY KEY,
    eval_id INTEGER REFERENCES evals(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    prompt_used TEXT NOT NULL,
    response_received TEXT,
    tool_calls_made JSONB DEFAULT '[]'::jsonb,
    success_score FLOAT, -- 0.0 to 1.0
    success_details JSONB DEFAULT '{}'::jsonb,
    execution_time_seconds FLOAT,
    error_message TEXT,
    status TEXT DEFAULT 'completed', -- 'running', 'completed', 'failed', 'timeout'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX eval_groups_category_idx ON eval_groups (category);
CREATE INDEX eval_groups_active_idx ON eval_groups (is_active);
CREATE INDEX evals_group_id_idx ON evals (group_id);
CREATE INDEX evals_active_idx ON evals (is_active);
CREATE INDEX evals_success_type_idx ON evals (success_type);
CREATE INDEX evals_difficulty_idx ON evals (difficulty_level);
CREATE INDEX eval_results_eval_id_idx ON eval_results (eval_id);
CREATE INDEX eval_results_model_name_idx ON eval_results (model_name);
CREATE INDEX eval_results_status_idx ON eval_results (status);
CREATE INDEX eval_results_created_at_idx ON eval_results (created_at);

-- Create trigger for updating timestamps
CREATE OR REPLACE FUNCTION update_eval_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_eval_groups_updated_at
    BEFORE UPDATE ON eval_groups
    FOR EACH ROW
    EXECUTE FUNCTION update_eval_updated_at_column();

CREATE TRIGGER update_evals_updated_at
    BEFORE UPDATE ON evals
    FOR EACH ROW
    EXECUTE FUNCTION update_eval_updated_at_column();

-- Insert some initial eval groups
INSERT INTO eval_groups (name, description, category) VALUES
('Metrics Management', 'Tests for creating and updating metrics', 'metrics'),
('System Integration', 'Tests for overall system functionality', 'system'),
('Data Querying', 'Tests for data retrieval and analysis', 'features'),
('Anomaly Detection', 'Tests for anomaly detection capabilities', 'features');

-- Insert some initial evals for metrics management
INSERT INTO evals (group_id, name, description, prompt, success_criteria, success_type, expected_tool_calls, difficulty_level) 
SELECT 
    eg.id,
    'Create Castro Violent Crime Metric',
    'Create a metric similar to violent crime but focused on Castro neighborhood',
    'Create a metric similar to our violent crime metric, but focused only on the Neighborhood of the Castro, and we will call it Castro Violent Crime.',
    '1. Successfully follow directions\n2. Successfully find the correct endpoint\n3. Successfully test the queries\n4. Successfully set the metric',
    'multi_step',
    '[{"function": "create_new_metric", "required": true}]',
    'medium'
FROM eval_groups eg 
WHERE eg.name = 'Metrics Management';

INSERT INTO evals (group_id, name, description, prompt, success_criteria, success_type, expected_tool_calls, difficulty_level) 
SELECT 
    eg.id,
    'Update Existing Metric',
    'Update an existing metric with new parameters',
    'Update the Castro Violent Crime metric to include data from the last 3 years instead of just the current year.',
    '1. Successfully identify the existing metric\n2. Successfully update the metric with new parameters\n3. Verify the update was successful',
    'multi_step',
    '[{"function": "get_metric_details", "required": true}, {"function": "edit_metric", "required": true}]',
    'easy'
FROM eval_groups eg 
WHERE eg.name = 'Metrics Management'; 
#!/usr/bin/env python3
"""
PostgreSQL database initialization script for TransparentSF anomaly detection.
This script creates the database and necessary tables if they don't exist.
"""

import os
import logging
from db_utils import get_postgres_connection, execute_with_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """
    Initialize the PostgreSQL database with required tables and schemas.
    """
    connection = None
    try:
        # Get database connection
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to establish database connection")
            return False

        # Create tables
        cursor = connection.cursor()
        
        # Enable required extensions
        logger.info("Enabling required PostgreSQL extensions...")
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS hstore")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            logger.info("Successfully enabled PostgreSQL extensions")
        except Exception as e:
            logger.error(f"Error enabling extensions: {str(e)}")
            return False

        # Drop existing tables in reverse order of dependencies
        logger.info("Dropping existing tables...")
        try:
            cursor.execute("""
                DROP TABLE IF EXISTS monthly_reporting CASCADE;
                DROP TABLE IF EXISTS reports CASCADE;
                DROP TABLE IF EXISTS charts CASCADE;
                DROP TABLE IF EXISTS anomalies CASCADE;
                DROP TABLE IF EXISTS time_series_data CASCADE;
                DROP TABLE IF EXISTS time_series_metadata CASCADE;
                DROP TABLE IF EXISTS metrics CASCADE;
                DROP TABLE IF EXISTS datasets CASCADE;
                DROP TABLE IF EXISTS cities CASCADE;
            """)
            logger.info("Successfully dropped existing tables")
        except Exception as e:
            logger.error(f"Error dropping tables: {str(e)}")
            return False

        # Create trigger function for updating timestamps
        logger.info("Creating timestamp update trigger function...")
        try:
            cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
            logger.info("Successfully created timestamp update trigger function")
        except Exception as e:
            logger.error(f"Error creating trigger function: {str(e)}")
            return False

        # Create cities table first (no dependencies)
        logger.info("Creating cities table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cities (
                    id SERIAL PRIMARY KEY,
                    city TEXT NOT NULL,
                    state_code TEXT NOT NULL,
                    place_code TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    population INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS cities_city_state_idx ON cities (city, state_code)
            """)
            logger.info("Successfully created cities table")
        except Exception as e:
            logger.error(f"Error creating cities table: {str(e)}")
            return False

        # Create metrics table (depends on cities)
        logger.info("Creating metrics table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_key TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    endpoint TEXT NOT NULL,
                    summary TEXT,
                    definition TEXT,
                    data_sf_url TEXT,
                    ytd_query TEXT,
                    metric_query TEXT,
                    dataset_title TEXT,
                    dataset_category TEXT,
                    show_on_dash BOOLEAN DEFAULT TRUE,
                    item_noun TEXT DEFAULT 'Items',
                    greendirection TEXT DEFAULT 'up',
                    location_fields JSONB DEFAULT '[]'::jsonb,
                    category_fields JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    city_id INTEGER REFERENCES cities(id),
                    display_order INTEGER DEFAULT 1000,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for metrics table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS metrics_category_idx ON metrics (category);
                CREATE INDEX IF NOT EXISTS metrics_subcategory_idx ON metrics (subcategory);
                CREATE INDEX IF NOT EXISTS metrics_endpoint_idx ON metrics (endpoint);
                CREATE INDEX IF NOT EXISTS metrics_active_idx ON metrics (is_active);
                CREATE INDEX IF NOT EXISTS metrics_show_on_dash_idx ON metrics (show_on_dash);
                CREATE INDEX IF NOT EXISTS metrics_city_id_idx ON metrics (city_id);
                CREATE INDEX IF NOT EXISTS metrics_display_order_idx ON metrics (display_order);
                CREATE INDEX IF NOT EXISTS metrics_category_display_order_idx ON metrics (category, display_order);
            """)
            logger.info("Successfully created metrics table and indexes")
        except Exception as e:
            logger.error(f"Error creating metrics table: {str(e)}")
            return False

        # Create anomalies table
        logger.info("Creating anomalies table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id SERIAL PRIMARY KEY,
                    group_value TEXT,
                    group_field_name TEXT,  
                    period_type TEXT,
                    comparison_mean FLOAT,
                    recent_mean FLOAT,
                    difference FLOAT,
                    std_dev FLOAT,
                    out_of_bounds BOOLEAN,
                    recent_date DATE,
                    comparison_dates JSONB,
                    comparison_counts JSONB,
                    recent_dates JSONB,
                    recent_counts JSONB,
                    metadata JSONB,
                    field_name TEXT,
                    object_type TEXT,
                    object_id TEXT,
                    object_name TEXT,
                    recent_data JSONB,
                    comparison_data JSONB,
                    district TEXT,
                    executed_query_url TEXT,
                    caption TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for anomalies
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS anomalies_created_at_idx ON anomalies (created_at);
                CREATE INDEX IF NOT EXISTS anomalies_out_of_bounds_idx ON anomalies (out_of_bounds);
                CREATE INDEX IF NOT EXISTS anomalies_is_active_idx ON anomalies (is_active);
                CREATE INDEX IF NOT EXISTS anomalies_district_idx ON anomalies (district);
            """)
            logger.info("Successfully created anomalies table and indexes")
        except Exception as e:
            logger.error(f"Error creating anomalies table: {str(e)}")
            return False

        # Create time_series_metadata table
        logger.info("Creating time_series_metadata table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS time_series_metadata (
                    chart_id SERIAL PRIMARY KEY,
                    object_type TEXT,
                    object_id TEXT,
                    object_name TEXT,
                    field_name TEXT,
                    y_axis_label TEXT,
                    period_type TEXT,
                    chart_title TEXT,
                    metadata JSONB,
                    filter_conditions JSONB,
                    district INTEGER DEFAULT 0,
                    group_field TEXT,
                    executed_query_url TEXT,
                    caption TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for time_series_metadata
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS time_series_metadata_object_type_id_idx ON time_series_metadata (object_type, object_id);
                CREATE INDEX IF NOT EXISTS time_series_metadata_field_name_idx ON time_series_metadata (field_name);
                CREATE INDEX IF NOT EXISTS time_series_metadata_district_idx ON time_series_metadata (district);
            """)
            logger.info("Successfully created time_series_metadata table and indexes")
        except Exception as e:
            logger.error(f"Error creating time_series_metadata table: {str(e)}")
            return False

        # Create time_series_data table
        logger.info("Creating time_series_data table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS time_series_data (
                    id SERIAL PRIMARY KEY,
                    chart_id INTEGER REFERENCES time_series_metadata(chart_id),
                    time_period DATE,
                    group_value TEXT,
                    numeric_value FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for time_series_data
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS time_series_data_chart_id_idx ON time_series_data (chart_id);
                CREATE INDEX IF NOT EXISTS time_series_data_time_period_idx ON time_series_data (time_period);
                CREATE INDEX IF NOT EXISTS time_series_data_group_value_idx ON time_series_data (group_value);
                CREATE INDEX IF NOT EXISTS time_series_data_chart_time_idx ON time_series_data (chart_id, time_period);
            """)
            logger.info("Successfully created time_series_data table and indexes")
        except Exception as e:
            logger.error(f"Error creating time_series_data table: {str(e)}")
            return False

        # Create reports table
        logger.info("Creating reports table...")
        try:
            cursor.execute("""
                CREATE TABLE reports (
                    id SERIAL PRIMARY KEY,
                    district VARCHAR(50) NOT NULL,
                    period_type VARCHAR(20) NOT NULL,
                    max_items INTEGER NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    revised_filename VARCHAR(255),
                    proofread_feedback TEXT,
                    published_url TEXT,
                    headlines JSONB,
                    audio_file TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT valid_period_type CHECK (period_type IN ('month', 'quarter', 'year')),
                    CONSTRAINT valid_max_items CHECK (max_items > 0)
                )
            """)
            logger.info("Successfully created reports table")
        except Exception as e:
            logger.error(f"Error creating reports table: {str(e)}")
            return False

        # Create monthly_reporting table
        logger.info("Creating monthly_reporting table...")
        try:
            cursor.execute("""
                CREATE TABLE monthly_reporting (
                    id SERIAL PRIMARY KEY,
                    report_id INTEGER NOT NULL,
                    report_date DATE DEFAULT CURRENT_DATE,
                    metric_name TEXT,
                    metric_id TEXT,
                    group_value TEXT,
                    group_field_name TEXT,
                    period_type TEXT DEFAULT 'month',
                    comparison_mean FLOAT,
                    recent_mean FLOAT,
                    difference FLOAT,
                    std_dev FLOAT,
                    percent_change FLOAT,
                    rationale TEXT,
                    explanation TEXT,
                    priority INTEGER,
                    report_text TEXT,
                    district TEXT,
                    chart_data JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_report
                        FOREIGN KEY(report_id) 
                        REFERENCES reports(id)
                        ON DELETE CASCADE
                )
            """)
            
            # Create indexes for monthly_reporting
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS monthly_reporting_report_date_idx ON monthly_reporting (report_date);
                CREATE INDEX IF NOT EXISTS monthly_reporting_district_idx ON monthly_reporting (district);
                CREATE INDEX IF NOT EXISTS monthly_reporting_priority_idx ON monthly_reporting (priority);
                CREATE INDEX IF NOT EXISTS monthly_reporting_report_id_idx ON monthly_reporting (report_id);
            """)
            logger.info("Successfully created monthly_reporting table and indexes")
        except Exception as e:
            logger.error(f"Error creating monthly_reporting table: {str(e)}")
            return False

        # Create datasets table
        logger.info("Creating datasets table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id SERIAL PRIMARY KEY,
                    endpoint TEXT UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    publishing_department TEXT,
                    rows_updated_at TIMESTAMP,
                    columns JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for datasets
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS datasets_endpoint_idx ON datasets (endpoint);
                CREATE INDEX IF NOT EXISTS datasets_category_idx ON datasets (category);
                CREATE INDEX IF NOT EXISTS datasets_title_idx ON datasets (title);
                CREATE INDEX IF NOT EXISTS datasets_active_idx ON datasets (is_active);
                CREATE INDEX IF NOT EXISTS datasets_updated_at_idx ON datasets (rows_updated_at);
            """)
            logger.info("Successfully created datasets table and indexes")
        except Exception as e:
            logger.error(f"Error creating datasets table: {str(e)}")
            return False

        # Create triggers for updating timestamps
        logger.info("Creating update triggers...")
        try:
            cursor.execute("""
                CREATE TRIGGER update_reports_updated_at
                    BEFORE UPDATE ON reports
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();

                CREATE TRIGGER update_monthly_reporting_updated_at
                    BEFORE UPDATE ON monthly_reporting
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();

                CREATE TRIGGER update_metrics_updated_at
                    BEFORE UPDATE ON metrics
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();

                CREATE TRIGGER update_datasets_updated_at
                    BEFORE UPDATE ON datasets
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """)
            logger.info("Successfully created update triggers")
        except Exception as e:
            logger.error(f"Error creating update triggers: {str(e)}")
            return False

        connection.commit()
        cursor.close()
        logger.info("Successfully initialized all database tables")
        return True    

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if connection:
            connection.rollback()
        return False
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    init_database() 
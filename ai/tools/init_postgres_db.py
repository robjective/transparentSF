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
        
        # Drop existing tables in reverse order of dependencies
        cursor.execute("""
            DROP TABLE IF EXISTS monthly_reporting CASCADE;
            DROP TABLE IF EXISTS reports CASCADE;
            DROP TABLE IF EXISTS charts CASCADE;
            DROP TABLE IF EXISTS anomalies CASCADE;
            DROP TABLE IF EXISTS metrics CASCADE;
        """)

        # Create trigger function for updating timestamps
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)

        # Create metrics table
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
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for metrics table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS metrics_category_idx ON metrics (category)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS metrics_subcategory_idx ON metrics (subcategory)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS metrics_endpoint_idx ON metrics (endpoint)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS metrics_active_idx ON metrics (is_active)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS metrics_show_on_dash_idx ON metrics (show_on_dash)
        """)

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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create an index on the created_at column for faster querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS anomalies_created_at_idx ON anomalies (created_at)
        """)
        
        # Create an index on the out_of_bounds column for faster filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS anomalies_out_of_bounds_idx ON anomalies (out_of_bounds)
        """)
        
        # Create an index on the district column for faster filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS anomalies_district_idx ON anomalies (district)
        """)
        
        # Create time_series_metadata table for time series data
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
            CREATE INDEX IF NOT EXISTS time_series_metadata_object_type_id_idx ON time_series_metadata (object_type, object_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS time_series_metadata_field_name_idx ON time_series_metadata (field_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS time_series_metadata_district_idx ON time_series_metadata (district)
        """)
        
        # Create time_series_data table
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
            CREATE INDEX IF NOT EXISTS time_series_data_chart_id_idx ON time_series_data (chart_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS time_series_data_time_period_idx ON time_series_data (time_period)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS time_series_data_group_value_idx ON time_series_data (group_value)
        """)
        
        # Create a composite index for common query patterns
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS time_series_data_chart_time_idx ON time_series_data (chart_id, time_period)
        """)
        
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT valid_period_type CHECK (period_type IN ('month', 'quarter', 'year')),
                CONSTRAINT valid_max_items CHECK (max_items > 0)
            )
        """)

        # Create monthly_reporting table (child table)
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

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS monthly_reporting_report_date_idx ON monthly_reporting (report_date)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS monthly_reporting_district_idx ON monthly_reporting (district)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS monthly_reporting_priority_idx ON monthly_reporting (priority)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS monthly_reporting_report_id_idx ON monthly_reporting (report_id)
        """)

        # Create triggers for updating timestamps
        cursor.execute("""
            CREATE TRIGGER update_reports_updated_at
                BEFORE UPDATE ON reports
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)

        cursor.execute("""
            CREATE TRIGGER update_monthly_reporting_updated_at
                BEFORE UPDATE ON monthly_reporting
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)

        # Create trigger for metrics table
        cursor.execute("""
            CREATE TRIGGER update_metrics_updated_at
                BEFORE UPDATE ON metrics
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)

        connection.commit()
        cursor.close()
        logger.info("Successfully initialized database tables")
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
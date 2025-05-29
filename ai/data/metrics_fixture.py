# ai/data/metrics_fixture.py

# This file should contain the hardcoded metrics data to be loaded into the database.
# Populate this list with dictionaries, where each dictionary represents a row
# in the 'metrics' table. Ensure all necessary columns are included and data types
# are appropriate for the target database schema.

# Example structure:
# METRICS_DATA = [
#     {
#         'metric_key': 'example_metric_1',
#         'metric_name': 'Example Metric One 📊',
#         'category': 'Examples',
#         'subcategory': 'General Examples',
#         'endpoint': '/data/example1.json',
#         'summary': 'This is a summary for example metric one.',
#         'definition': 'The detailed definition of what example metric one measures.',
#         'data_sf_url': 'https://data.sfgov.org/resource/example1.json',
#         'ytd_query': 'SELECT count(*) FROM table WHERE year = current_year;',
#         'metric_query': 'SELECT count(*) FROM table WHERE date = specific_date;',
#         'dataset_title': 'Example Dataset One',
#         'dataset_category': 'Public Data',
#         'show_on_dash': True,  # Boolean
#         'item_noun': 'Entries',
#         'greendirection': 'down', # 'up', 'down', or 'neutral'
#         'location_fields': [ # List of dicts, or None/empty list
#             {'name': 'District', 'fieldName': 'supervisor_district', 'description': 'Supervisorial District'}
#         ],
#         'category_fields': [ # List of dicts, or None/empty list
#             {'name': 'Type', 'fieldName': 'incident_type', 'description': 'Type of incident'}
#         ],
#         'metadata': { # Dict, or None/empty dict
#             'original_id': 101,
#             'source_notes': 'Data extracted from legacy system.'
#         },
#         # ... add any other columns present in your 'metrics' table
#         # 'another_column_text': 'some text value',
#         # 'another_column_number': 123.45,
#         # 'another_column_date': '2023-10-26', # Dates might need specific formatting or be Python date objects
#         # 'another_column_bool': False,
#     },
#     # ... more metric dictionaries
# ]

# --- POPULATE METRICS_DATA BELOW --- 
METRICS_DATA = []

# If METRICS_DATA is empty, the migration script will still clear the target 'metrics' table
# but will not insert any new data. A warning will be logged in this case. 
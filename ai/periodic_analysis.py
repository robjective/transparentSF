import os
import json
from tools.data_fetcher import set_dataset
from tools.genChart import generate_time_series_chart
from tools.anomaly_detection import anomaly_detection
import datetime
from swarm import Swarm
from pathlib import Path
import logging
import sys
import pytz

# ------------------------------
# Initialization
# ------------------------------
# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

client = Swarm()

# GPT_MODEL = 'gpt-3.5-turbo-16k'
GPT_MODEL = 'gpt-4.1'

# ------------------------------
# Helper Functions
# ------------------------------

def load_combined_data(datasets_folder, log_file):
    """
    Load combined data from database instead of files.
    TODO: This function needs to be updated to load the appropriate data structure from the database.
    For now, returning empty list to prevent errors.
    """
    print("WARNING: load_combined_data called but datasets have been moved to database.")
    print("This function needs to be updated to load data from the database instead of files.")
    print("Returning empty list to prevent errors.")
    
    # TODO: Update this function to load data from the database
    # The function should query the appropriate tables and return data in the expected format
    # that the analysis functions expect.
    
    return []

def save_combined_html(combined_data, output_folder):
    output_file = os.path.join(output_folder, 'all_data.html')
    with open(output_file, 'w', encoding='utf-8') as html_file:
        html_file.write("<table border='1'><tr><th>Index</th><th>Category</th><th>Title</th><th>Usefulness</th></tr>")
        for index, entry in enumerate(combined_data):
            category = entry.get('report_category', 'Unknown')
            title = entry.get('title', 'Unknown')
            periodic = entry.get('periodic', 'Unknown')
            district_level = entry.get('district_level', 'Unknown')
            html_file.write(f"<tr><td>{index}</td><td>{category}</td><td>{title}</td><td>{periodic}</td><td>{district_level}</td></tr>")
        html_file.write("</table>")
    print(f"Combined data saved to {output_file}")

def get_time_ranges(period_type):
    """
    Calculate recent and comparison periods based on period type.
    
    Args:
        period_type (str): One of 'year', 'month', 'day', or 'ytd'
    
    Returns:
        tuple: (recent_period, comparison_period) each containing start and end dates
    """
    today = datetime.date.today()
    
    if period_type == 'ytd':
        # Current year from Jan 1 to yesterday
        recent_period = {
            'start': datetime.date(today.year, 1, 1),
            'end': today - datetime.timedelta(days=1)
        }
        # Same days last year
        comparison_period = {
            'start': datetime.date(today.year - 1, 1, 1),
            'end': datetime.date(today.year - 1, today.month, today.day) - datetime.timedelta(days=1)
        }
    elif period_type == 'year':
        # Last complete year
        last_year = today.year - 1 
        recent_period = {
            'start': datetime.date(last_year, 1, 1),
            'end': datetime.date(last_year, 12, 31)
        }
        comparison_period = {
            'start': datetime.date(last_year - 10, 1, 1),
            'end': datetime.date(last_year - 1, 12, 31)
        }
    elif period_type == 'month':
        # For monthly analysis, always use the previous month as it's complete
        last_complete_month = today.replace(day=1) - datetime.timedelta(days=1)
        last_complete_month = last_complete_month.replace(day=1)  # First day of the month
        
        recent_period = {
            'start': last_complete_month,
            'end': (last_complete_month.replace(day=1) + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
        }
        
        # Calculate start of comparison (24 months before)
        comparison_start = last_complete_month - datetime.timedelta(days=730)  # Roughly 24 months
        comparison_start = comparison_start.replace(day=1)  # First day of that month
        
        comparison_period = {
            'start': comparison_start,
            'end': last_complete_month - datetime.timedelta(days=1)
        }
    else:  # day
        # Last complete day
        yesterday = today - datetime.timedelta(days=1)
        recent_period = {
            'start': yesterday,
            'end': yesterday
        }
        comparison_period = {
            'start': yesterday - datetime.timedelta(days=28),
            'end': yesterday - datetime.timedelta(days=1)
        }
    
    return recent_period, comparison_period

def log_analysis(analysis_info):
    """
    Write analysis information to a single log file in JSONL format.
    
    Args:
        analysis_info (dict): Dictionary containing analysis information
    """
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'analysis.log')
    
    # Create a concise log entry
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'endpoint': analysis_info.get('endpoint', ''),
        'period_type': analysis_info.get('period_type', ''),
        'recent_period': analysis_info.get('recent_period', {}),
        'comparison_period': analysis_info.get('comparison_period', {}),
        'location': analysis_info.get('location', 'citywide'),
        'fields_analyzed': analysis_info.get('fields_analyzed', []),
        'charts_generated': analysis_info.get('charts_generated', [])
    }
    
    # Write to log file in JSONL format (append mode)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

def process_entry(index, data_entry, output_folder, log_file, script_dir, period_type='year'):
    title = data_entry.get('title', 'Unknown')
    noun = data_entry.get('item_noun', data_entry.get('table_metadata', {}).get('item_noun', 'Unknown'))
    category = data_entry.get('report_category', 'Unknown')
    endpoint = data_entry.get('endpoint', None)
    
    # Initialize analysis info
    analysis_info = {
        'endpoint': endpoint,
        'period_type': period_type,
        'fields_analyzed': [],
        'charts_generated': []
    }
    
    # Log the period type and available queries
    queries = data_entry.get('queries', {})
    log_file.write(f"\n{index}: {title} ({endpoint}) - Processing with period_type: {period_type}\n")
    log_file.write(f"Available queries: {list(queries.keys())}\n")
    
    # Select query based on period_type
    query = None
    if period_type == 'year':
        query = queries.get('Yearly')
        log_file.write(f"Selected Yearly query: {query is not None}\n")
    elif period_type == 'month':
        query = queries.get('Monthly')
        log_file.write(f"Selected Monthly query: {query is not None}\n")
    elif period_type == 'day' or period_type == 'ytd':  # Use Daily query for both day and ytd
        query = queries.get('Daily')
        log_file.write(f"Selected Daily query: {query is not None}\n")
    else:
        log_file.write(f"Unknown period_type: {period_type}\n")
    
    # Log the selected query
    if query:
        log_file.write(f"Using query: {query[:100]}...\n")
    else:
        # If Daily query not found for YTD, try Monthly as fallback
        if period_type == 'ytd':
            query = queries.get('Monthly')
            log_file.write(f"Daily query not found for YTD, using Monthly query as fallback: {query is not None}\n")
        if not query:
            log_file.write(f"No query found for period_type: {period_type}\n")

    date_fields = data_entry.get('DateFields', [])
    numeric_fields = data_entry.get('NumericFields', [])
    category_field = data_entry.get('CategoryFields', [])
    location_field = data_entry.get('LocationFields', [])   
    usefulness = data_entry.get('usefulness', data_entry.get('table_metadata', {}).get('usefulness', 0))
    description = data_entry.get('description', data_entry.get('table_metadata', {}).get('description', ''))
    
    # Add fields to analysis info
    analysis_info['fields_analyzed'].extend(date_fields)
    analysis_info['fields_analyzed'].extend(numeric_fields)
    analysis_info['fields_analyzed'].extend(category_field)
    analysis_info['fields_analyzed'].extend(location_field)
    
    # Check usefulness
    if usefulness == 0:
        log_file.write(f"{index}: {title}({endpoint}) - Skipped due to zero usefulness.\n")
        return

    # Skip if mandatory fields are missing
    if not query or not endpoint or not date_fields or not numeric_fields:
        log_file.write(f"{index}: {title}({endpoint}) - Missing required fields. Skipping.\n")
        return
    
    # Extract table and column metadata
    column_metadata = data_entry.get('columns', [])

    # Get time ranges based on period type
    recent_period, comparison_period = get_time_ranges(period_type)
    analysis_info['recent_period'] = {
        'start': recent_period['start'].isoformat(),
        'end': recent_period['end'].isoformat()
    }
    analysis_info['comparison_period'] = {
        'start': comparison_period['start'].isoformat(),
        'end': comparison_period['end'].isoformat()
    }
    
    # Calculate the number of periods looking back
    periods_lookback = {
        'year': comparison_period['end'].year - comparison_period['start'].year + 1,
        'month': (comparison_period['end'].year - comparison_period['start'].year) * 12 + 
                 comparison_period['end'].month - comparison_period['start'].month + 1,
        'day': (comparison_period['end'] - comparison_period['start']).days + 1,
        'ytd': 1  # YTD always compares to previous year's same period
    }[period_type]

    # Create period description for titles
    period_names = {
        'year': 'Annual',
        'month': 'Monthly',
        'day': 'Daily',
        'ytd': 'Year-to-Date'
    }
    period_desc = f"{period_names[period_type]} ({periods_lookback} {period_type}s lookback)"

    # Modify filter conditions based on period type
    # Determine the appropriate date field name
    date_field_name = date_fields[0]
    if period_type == 'ytd':
        # For YTD using Daily query, we need to use 'day' as that's what the query outputs
        if query and 'date_trunc_ymd' in query:
            date_field_name = 'day'
        elif query and 'date_trunc_ym' in query:
            date_field_name = 'month'
        elif query and 'date_trunc_y' in query:
            date_field_name = 'year'
    elif date_field_name in ['year', 'month', 'day'] and date_field_name != period_type:
        date_field_name = period_type

    filter_conditions = [
        {'field': date_field_name, 'operator': '<=', 'value': recent_period['end'].isoformat()},
        {'field': date_field_name, 'operator': '>=', 'value': comparison_period['start'].isoformat()},
    ]
    # Add filter conditions to log
    log_file.write(f"Filter conditions:\n")
    for condition in filter_conditions:
        log_file.write(f"  - {condition['field']} {condition['operator']} {condition['value']}\n")
    
    # Determine if supervisor_district is present
    has_supervisor_district = 'supervisor_district' in location_field

    try:
        context_variables = {}  # Initialize context_variables for each iteration
        # Set the dataset
        # Use the comparison period start date instead of hardcoding 10 years ago
        query_modified = query.replace(' start_date', f"'{comparison_period['start']}'")
        result = set_dataset(context_variables=context_variables, endpoint=endpoint, query=query_modified, filter_conditions=filter_conditions)
        if 'error' in result:
            log_file.write(f"{index}: {title} ({endpoint}) - Error setting dataset: {result['error']}\n")
            return
        # Get the dataset from context_variables
        if 'dataset' in context_variables:
            dataset = context_variables['dataset']
        else:
            log_file.write(f"{index}: {title} ({endpoint}) - No dataset found in context.\n")
            return
        
        if 'queryURL' in result:
            log_file.write(f"{index}: {title} ({endpoint}) - Query URL: {result['queryURL']}\n")
            query_url = result['queryURL']
            
        all_markdown_contents = []
        all_html_contents = []
        print(f"Date fields from JSON: {date_fields}")
        print(f"Numeric fields from JSON: {numeric_fields}")
        print(f"Category field from JSON: {category_field}")
        print(f"Location field from JSON: {location_field}")
        print(f"Available columns in dataset: {context_variables['dataset'].columns.tolist()}")

        # Function to generate charts and anomalies
        def generate_reports(current_dataset, current_filter_conditions, current_output_folder, current_title_suffix, metadata):
            nonlocal all_markdown_contents, all_html_contents, analysis_info
            # Generate charts for each combination of date and numeric fields

            for date_field in date_fields:
                for numeric_field in numeric_fields:
                    # Update chart title for this combination
                    chart_title = f"{title} <br> {'count' if numeric_field == 'item_count' else numeric_field.replace('_', ' ')} by {date_field}"
                    context_variables['chart_title'] = chart_title
                    context_variables['noun'] = f"{title}"
                    
                    # Track chart being generated
                    analysis_info['charts_generated'].append(chart_title)
                    
                    # Generate the chart
                    chart_result = generate_time_series_chart(
                        context_variables=context_variables,
                        time_series_field=date_field_name,
                        numeric_fields=numeric_field,
                        aggregation_period=period_type,
                        max_legend_items=10,
                        filter_conditions=current_filter_conditions,
                        show_average_line=True,
                        return_html=True,
                        output_dir=current_output_folder
                    )
                    # Ensure we're adding strings, not tuples or dicts
                    if isinstance(chart_result, tuple):
                        markdown_content, html_content = chart_result
                        all_markdown_contents.append(str(markdown_content))
                        all_html_contents.append(str(html_content))
                    elif isinstance(chart_result, dict):
                        all_html_contents.append(str(chart_result.get('html', '')))
                    else:
                        all_html_contents.append(str(chart_result))

            # Loop through each category field to detect anomalies
            for cat_field in category_field:
                try:
                    chart_title = f"{title} <br> {'count' if numeric_fields[0] == 'item_count' else numeric_fields[0].replace('_', ' ')} by {date_fields[0]} by {cat_field}"
                    context_variables['chart_title'] = chart_title
                    analysis_info['charts_generated'].append(chart_title)
                    
                    chart_result = generate_time_series_chart(
                        context_variables=context_variables,
                        time_series_field=date_field_name,
                        numeric_fields=numeric_fields[0],
                        aggregation_period=period_type,
                        max_legend_items=10,
                        group_field=cat_field,
                        filter_conditions=current_filter_conditions,
                        show_average_line=False,
                        return_html=True,
                        output_dir=current_output_folder
                    )
                    if isinstance(chart_result, tuple):
                        markdown_content, html_content = chart_result
                        all_markdown_contents.append(str(markdown_content))
                        all_html_contents.append(str(html_content))
                    elif isinstance(chart_result, dict):
                        all_html_contents.append(str(chart_result.get('html', '')))
                    else:
                        all_html_contents.append(str(chart_result))

                    print(f"Detecting anomalies for category field: {cat_field}")

                    # Check anomalies for each numeric field
                    for numeric_field in numeric_fields:
                        print(f"Detecting anomalies for numeric field: {numeric_field}")

                        anomalies_result = anomaly_detection(
                            context_variables=context_variables,
                            group_field=cat_field,
                            filter_conditions=current_filter_conditions,
                            min_diff=2,
                            recent_period=recent_period,
                            comparison_period=comparison_period,
                            date_field=date_field_name,
                            numeric_field=numeric_field,
                            y_axis_label=numeric_field,
                            title=context_variables['chart_title'],
                            period_type=period_type,
                            object_type='period_analysis',
                            object_id=index,
                            object_name=title
                        )

                        if anomalies_result and 'anomalies' in anomalies_result:
                            anomalies_html = anomalies_result['anomalies']
                            all_html_contents.append(anomalies_html)
                            anomalies_markdown = anomalies_result.get('anomalies_markdown', '')
                            all_markdown_contents.append(anomalies_markdown)

                except Exception as e:
                    print(f"Error detecting anomalies for category {cat_field} in entry at index {index}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    # Log the error and move on
                    log_file.write(f"{index}: {title} - Error detecting anomalies: {str(e)}\n")
                    continue

            # After generating all charts and anomalies, save the files
            # Prepare the HTML content
            processed_html_contents = []
            for content in all_html_contents:
                if content is None:
                    continue
                if isinstance(content, dict):
                    if 'html' in content:
                        processed_html_contents.append(str(content['html']))
                    else:
                        processed_html_contents.append(str(content))
                else:
                    processed_html_contents.append(str(content))

            combined_html = "\n\n".join(processed_html_contents)

            # Prepare metadata for HTML
            metadata_html = f"""<head>
                <title>{metadata['noun']} {current_title_suffix} - {period_desc}</title>
            </head>
            <body>
                <h1>{periods_lookback} {period_type} look-back for {metadata['noun']} {current_title_suffix}</h1>
                <p><strong>Analysis Type:</strong> {period_desc}</p>
                <p><strong>Query URL:</strong> <a href='{query_url}'>LINK</a></p>
                <p><strong>{metadata['description']}</strong></p>
                <p><strong>{metadata['endpoint']}</strong></p>
                <p><strong>{metadata['title']}</strong></p>"""
            metadata_html += "<h2>Column Metadata</h2><table border='1'><tr><th>Field Name</th><th>Description</th><th>Data Type</th></tr>"
            for column in metadata['column_metadata']:
                row = "<tr>"
                if isinstance(column, dict):
                    # Assume fieldName, description, dataTypeName keys exist
                    row += f"<td>{column.get('fieldName', 'Unknown')}</td>"
                    row += f"<td>{column.get('description', 'No description')}</td>"
                    row += f"<td>{column.get('dataTypeName', 'Unknown')}</td>"
                else:
                    # If column is a string
                    row += f"<td>{column}</td><td>No description</td><td>Unknown</td>"
                row += "</tr>"
                metadata_html += row
            metadata_html += "</table>"

            full_html_content = metadata_html + "\n\n" + combined_html

            # Sanitize title for filename
            sanitized_title = endpoint if endpoint.endswith('.json') else endpoint + '.json'
            if current_title_suffix != "City Wide":
                sanitized_title = f"{sanitized_title.replace('.json', '')}_{current_title_suffix.lower().replace(' ', '_')}.json"

            # We no longer save HTML files
            html_filename = os.path.join(current_output_folder, f"{sanitized_title}.html")
            
            # Process and save markdown content
            processed_markdown_contents = []
            for content in all_markdown_contents:
                if content is None:
                    continue
                if isinstance(content, dict):
                    if 'markdown' in content:
                        processed_markdown_contents.append(str(content['markdown']))
                    else:
                        processed_markdown_contents.append(str(content))
                else:
                    processed_markdown_contents.append(str(content))

            combined_markdown = "\n\n".join(processed_markdown_contents)
            metadata_md = f"# {metadata['noun']} {current_title_suffix}\n\n"
            metadata_md += f"**Query URL:** {query_url}\n\n"
            metadata_md += f"**Description:** {metadata['description']}\n\n"
            metadata_md += "## Column Metadata\n\n| Field Name | Description | Data Type |\n|------------|-------------|-----------|\n"
            for column in metadata['column_metadata']:
                if isinstance(column, dict):
                    field_name = column.get('fieldName', 'Unknown')
                    description = column.get('description', 'No description')
                    data_type = column.get('dataTypeName', 'Unknown')
                    metadata_md += f"| {field_name} | {description} | {data_type} |\n"
                else:
                    metadata_md += f"| {column} | No description | Unknown |\n"

            full_markdown_content = metadata_md + "\n\n" + combined_markdown

            # Save markdown file
            markdown_filename = os.path.join(current_output_folder, f"{sanitized_title}.md")
            if os.path.exists(markdown_filename):
                os.remove(markdown_filename)
            with open(markdown_filename, 'w', encoding='utf-8') as f:
                f.write(full_markdown_content)

            # Clear the contents for the next iteration
            all_markdown_contents.clear()
            all_html_contents.clear()

            # Return the html filename for logging
            return html_filename

        # Create metadata dictionary
        metadata = {
            'title': title,
            'noun': noun,
            'description': description,
            'endpoint': endpoint,
            'column_metadata': column_metadata
        }

        # Process unfiltered data (citywide)
        analysis_info['location'] = 'citywide'
        main_html_file = generate_reports(dataset, filter_conditions, output_folder, "City Wide", metadata)
        
        # Log citywide analysis
        log_analysis(analysis_info)

        # If supervisor_district is present, process for each district
        if has_supervisor_district:
            for district in range(1, 12):
                district_analysis_info = analysis_info.copy()
                district_analysis_info['location'] = f'district_{district}'
                
                district_output = os.path.join(output_folder, 'districts', f'district_{district}')
                os.makedirs(district_output, exist_ok=True)
                
                district_filter_conditions = filter_conditions + [
                    {'field': 'supervisor_district', 'operator': '=', 'value': district}
                ]
                
                generate_reports(dataset, district_filter_conditions, district_output, f"District {district}", metadata)
                
                # Log district analysis
                log_analysis(district_analysis_info)

        # Log success using the main HTML file path
        relative_html_path = os.path.relpath(main_html_file, start=script_dir)
        log_file.write(f"{index}: {title} ({endpoint}) - Success. Output HTML: {relative_html_path}\n")

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error processing data from entry at index {index}: {e}")
        print(traceback_str)
        log_file.write(f"{index}: {title}({endpoint}) - Error: {str(e)}\n")

def process_entries(combined_data, num_start, num_end, output_folder, log_file_path, script_dir, period_type='year'):
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        for index in range(num_start, num_end):
            if index >= len(combined_data):
                break
            data_entry = combined_data[index]
            # Process each entry once; district handling is managed within process_entry
            process_entry(index, data_entry, output_folder, log_file, script_dir, period_type)

def export_for_endpoint(endpoint, period_type='year', output_folder=None, 
                       log_file_path=os.path.join('logs', 'analysis_process.log')):
    """
    Export processing for a specific endpoint with specified period type.
    
    Args:
        endpoint (str): The endpoint to process
        period_type (str): One of 'year', 'month', or 'day'. Defaults to 'year'
        output_folder (str, optional): Custom output folder path
        log_file_path (str, optional): Path to log file
    """
    # Add logging for the function entry
    print(f"export_for_endpoint called with: endpoint={endpoint}, period_type={period_type}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    datasets_folder = os.path.join(data_folder, 'datasets/fixed')
    
    if not output_folder:
        # Update output folder based on period_type
        period_folder = {'year': 'annual', 'month': 'monthly', 'day': 'daily', 'ytd': 'ytd'}[period_type]
        output_folder = os.path.join(script_dir, 'output', period_folder)
        print(f"Using output folder: {output_folder}")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Create a log file to track the processing results
    if not log_file_path:
        log_file_path = os.path.join(output_folder, "processing_log.txt")
    
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"Processing Export for Endpoint: {endpoint}\n")
        log_file.write(f"Period Type: {period_type}\n")
        log_file.write(f"Output Folder: {output_folder}\n")
        log_file.write(f"Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"{'='*50}\n")
    
    print(f"Processing Export for Endpoint: {endpoint} at {datetime.datetime.now()}")
    # Load combined data
    combined_data = load_combined_data(datasets_folder, log_file_path)

    # Find records matching the endpoint
    matching_records = [entry for entry in combined_data if entry.get('endpoint') == endpoint or entry.get('endpoint') == endpoint + '.json']

    if not matching_records:
        print(f"No records found for endpoint: {endpoint}")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"No records found for endpoint: {endpoint}\n")
        return

    print(f"Found {len(matching_records)} record(s) for endpoint: {endpoint}")

    # Process each matching record
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for record in matching_records:
            index = combined_data.index(record)
            process_entry(index, record, output_folder, log_file, script_dir, period_type)

    print(f"\nExport processing complete for endpoint: {endpoint}.")
    print(f"Log file location: {log_file_path}")

# ------------------------------
# Main Function
# ------------------------------

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    datasets_folder = os.path.join(data_folder, 'datasets/fixed')
    output_folder = os.path.join(script_dir, 'output', 'annual')

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Set the range of entries to process
    num_start = 120
    num_end = 122

    # Create a log file to track the processing results
    log_file_path = os.path.join(output_folder, "processing_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Processing Log:\n")

    # Load combined data
    combined_data = load_combined_data(datasets_folder, log_file_path)

    # Save combined data as HTML
    if combined_data:
        save_combined_html(combined_data, output_folder)
    else:
        print("No valid data found to save.")

    # Add period_type parameter
    period_type = 'year'  # Default to year, but could be passed as command line argument
    
    # Process entries with period_type
    process_entries(combined_data, num_start, num_end, output_folder, log_file_path, script_dir, period_type)

    print(f"\nProcessing complete for entries from index {num_start} to {num_end - 1}.")
    print(f"Log file location: {log_file_path}")

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == '__main__':
    main()

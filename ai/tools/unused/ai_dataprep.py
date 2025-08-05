import os
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from dateutil.parser import parse as parse_date
import pytz  # Add this for timezone handling
from dateutil.parser import parse as parse_date
import traceback

# Set key variables 
# Load environment variables
load_dotenv()
# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")
GPT_MODEL = 'gpt-4o'

# ------------------------------
# Utility Functions
# ------------------------------

def sanitize_filename(filename):
    """Sanitize the filename by removing or replacing invalid characters."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    sanitized = sanitized.strip()
    return sanitized

def create_analysis_map(datasets_folder, output_folder):
    """
    Re-creates the analysis map from individual result files.
    
    Args:
        datasets_folder (str): Path to folder containing dataset files
        output_folder (str): Path to output folder for analysis map
        
    Returns:
        str: Path to created analysis map file
    """
    # List all JSON files in the datasets directory
    article_list = [f for f in os.listdir(datasets_folder) if f.endswith('.json')]
    if not article_list:
        print(f"No JSON files found in {datasets_folder}")
        return None
        
    print(f"Found {len(article_list)} JSON files to process:")
    # for idx, filename in enumerate(article_list):
    #     print(f"{idx}. {filename}")
        
    # Create output file for analysis map
    # output_filename = f"analysis_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_filename = f"analysis_map.json"
    output_path = os.path.join(output_folder, output_filename)
    # Initialize list to store all results
    all_results = []
    
    # Load and combine individual result files
    individual_results_dir = os.path.join(output_folder, 'individual_results')
    if os.path.exists(individual_results_dir):
        result_files = [f for f in os.listdir(individual_results_dir) if f.endswith('.json')]
        for result_file in result_files:
            result_path = os.path.join(individual_results_dir, result_file)
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    all_results.append(result_data)
            except Exception as e:
                print(f"Error reading result file {result_file}: {e}")
                continue
    
    # Write combined results to analysis map file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Analysis map created with {len(all_results)} entries at: {output_path}")
    return output_path

def format_columns(columns):
    """Format the columns information into a readable string."""
    if not columns:
        return ""
    formatted = ""
    for col in columns:
        formatted += f"- {col['fieldName']} ({col['dataTypeName']}): {col['description']}\n"
    return formatted

def process_single_file(filename, datasets_folder, output_folder, threshold_date, error_log):
    """
    Process a single file from the datasets folder.
    """
    try:
        individual_results_dir = os.path.join(output_folder)
        os.makedirs(individual_results_dir, exist_ok=True)

        article_path = os.path.join(datasets_folder, filename)
        print(f"\nProcessing file: {filename}")

        # Load JSON data to extract 'endpoint' before determining the result filename
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in input file {filename}: {str(e)}"
            print(error_msg)
            error_log.append({'filename': filename, 'error': error_msg, 'phase': 'input_file_reading'})
            return
        except Exception as e:
            error_msg = f"Error reading input file {filename}: {str(e)}"
            print(error_msg)
            error_log.append({'filename': filename, 'error': error_msg, 'phase': 'input_file_reading'})
            return

        # Extract 'endpoint' field
        endpoint = data.get('endpoint', None)
        if not endpoint:
            print(f"'endpoint' field missing in file {filename}. Skipping file.")
            error_log.append({'filename': filename, 'error': "'endpoint' field missing."})
            return

        # Sanitize the 'endpoint' to create a safe filename
        sanitized_endpoint = sanitize_filename(endpoint)
        if not sanitized_endpoint:
            print(f"Sanitized 'endpoint' is empty for file {filename}. Skipping file.")
            error_log.append({'filename': filename, 'error': "Sanitized 'endpoint' is empty."})
            return

        # Define the result filename based on the sanitized 'endpoint'
        result_filename = f"{sanitized_endpoint}.json"
        result_path = os.path.join(individual_results_dir, result_filename)

        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except OSError as e:
                print(f"Error removing existing file {result_path}: {e}")
                return

        # Continue processing the file as before
        # Check 'rows_updated_at' and threshold
        rows_updated_at = data.get("rows_updated_at", None)

        try:
            updated_at_date = parse_date(rows_updated_at)
            if updated_at_date.tzinfo is None:
                updated_at_date = updated_at_date.replace(tzinfo=pytz.UTC)

            # if updated_at_date <= threshold_date:
            #     print(f"File '{filename}' skipped: 'rows_updated_at' is not after {threshold_date}.")
            #     return
        except Exception as e:
            print(f"Error parsing 'rows_updated_at' for file '{filename}': {e}")
            error_log.append({'filename': filename, 'error': str(e)})
            return

        # Extract dataset details
        title = data.get('title', 'Untitled')
        description = data.get('description', '')
        columns = [{
            'fieldName': col['fieldName'],
            'description': col.get('description', ''),
            'dataTypeName': col.get('dataTypeName', '')
        } for col in data.get('columns', []) if 'fieldName' in col]
        columns_formatted = format_columns(columns)

        # Prepare AI prompt
        user_message = f"""
        Here is the dataset information:
        Title: {title}
        Endpoint: {endpoint}
        Description:
        {description}
        Columns:
        {columns_formatted}
        """
        # Construct the prompt
        system_message = """
              You are an AI assistant that generates working API queries and categorizes datasets for analysis. We are looking to find anomalous trends within the dataset. Trends are spotted by comparing sums or averages over time by category. So we are looking for data that updates regularly, has at least one time series variable and at least one numeric variable. Often with city datasets, the numeric variable we need is actually just a count of the number of items by month (like for police or fire reports).

                Your task is:

                - Given dataset information (title, description, columns), generate a set of working API queries, one for monthly analysis, one for annual with an endpoint and query that will work.
                  
                  IMPORTANT QUERY SYNTAX RULES:
                  - Never use type casting with :: syntax
                  - For numeric conversions, use to_number() function instead
                  - Date functions available: date_trunc_y(), date_trunc_ym(), date_trunc_ymd()
                  - Aggregation functions available: sum(), avg(), min(), max(), count()
                  - Never include a FROM clause
                  - Always include proper WHERE clause with start_date
                  - Use proper field names from fieldName property only
                  
                  The field names you will select and return are the ones that are the output of the query, not the input, so if you use a date_field called _added_date, but select it as a trunc_ym(_added_date) as month, then you would use the fieldname month.
                  
                  Required query components:
                  - Date Fields: At least one date field, aggregated by month or year
                  - Numeric Fields: At least one numeric field using sum() or count(*)
                  - Category Fields: Include useful grouping fields (prefer names over IDs)
                  - Location Fields: Include geographic/district fields when available
                  - Where clause: Filter for data from "start_date" forward
                  
                - report_category: Categorize the dataset into one of the following categories: Safety, Health, Economy, Housing, Education, Transportation, Other.
                - usefulness: Estimate the usefullness on a scale of 1-3 (1=least useful, 3=most useful) for an monthly analysis of trends happening in San Francisco.
                - periodic: Boolean: yes if this is the kind of data with constant new entries like police reports, or is it a lookup table like a list of departments or a rarely changing stock, like stop sign locations or wireless cariiers.  
                - item_noun - Rows in this table are what?  In the example abovethey are Police Incident Reports. 
                - district_level: Boolean: yes if this is the kind of data with constant new entries like police reports, or is it a lookup table like a list of departments or a rarely changing stock, like stop sign locations or wireless cariiers.  
                - whom_it_may_interest: Explain who would be interested in this data and why.
                - data_validated: Boolean: yes if the data has been validated by the city, no if it is raw data.
                - queries: A dictionary with two keys, 'Monthly' and 'Yearly', each with a SoQL query as the value.

                Please provide the output in JSON format, with keys 'DateFields' 'NumericFields' 'CategoryFields' 'LocationFields' 'endpoint', 'query',  'report_category',  'usefulness', 'periodic', 'district_level', 'item_noun', 'whom_it_may_interest'.  The 'DateFields', 'NumericFields', and 'CategoryFields' keys should contain lists of OUTPUT field names, meaning the ones that the query will return. 
                The 'endpoint' and 'query' keys should contain the API endpoint and the SoQL query, respectively.  The 'report_category' key should contain the category name.  The 'usefulness' key should contain the usefulness estimate.

                Ensure that the query is a valid SoQL query, and that the endpoint is correct (the dataset identifier).
                Remember no from clause. 
                
                Here's an example: 
                {"usefulness": 3,
                "report_category": "Safety",
                "NumericFields": [
                    "incident_count"
                ],
                "CategoryFields": [
                    "grouped_category",
                    "Incident_Category",
                    "Incident_Subcategory",
                    "Police_District",
                    "supervisor_district"
                ],
                "DateFields": [
                    "month"
                ],
                "LocationFields": [
                    "supervisor_district"
                ],
                "periodic": true,
                "district_level": true,
                "whom_it_may_interest": "Data Analysts, Public Safety Officials, Policy Makers, Researchers studying crime patterns and public safety trends, and Citizens concerned with neighborhood safety",
                "data_validated": true,
                "item_noun": "police incident reports",
                "queries": {
                    "Monthly": "SELECT Incident_Category, Incident_Subcategory, supervisor_district, CASE WHEN Incident_Category IN ('Assault', 'Homicide', 'Rape', 'Robbery', 'Human Trafficking (A), Commercial Sex Acts', 'Human Trafficking, Commercial Sex Acts', 'Human Trafficking (B), Involuntary Servitude', 'Offences Against The Family And Children', 'Weapons Carrying Etc', 'Weapons Offense', 'Weapons Offence') THEN 'Violent Crime' WHEN Incident_Category IN ('Arson', 'Burglary', 'Forgery And Counterfeiting', 'Fraud', 'Larceny Theft', 'Motor Vehicle Theft', 'Motor Vehicle Theft?', 'Stolen Property', 'Vandalism', 'Embezzlement', 'Recovered Vehicle', 'Vehicle Impounded', 'Vehicle Misplaced') THEN 'Property Crime' WHEN Incident_Category IN ('Drug Offense', 'Drug Violation') THEN 'Drug Crimes' ELSE 'Other Crimes' END AS grouped_category, Report_Type_Description, Police_District, date_trunc_ym(Report_Datetime) AS month, COUNT(*) AS incident_count WHERE Report_Datetime >= start_date GROUP BY supervisor_district, grouped_category, Report_Type_Description, Police_District, Incident_Category, Incident_Subcategory, month ORDER BY month, grouped_category",
                    "Yearly": "SELECT Incident_Category, Incident_Subcategory, supervisor_district, CASE WHEN Incident_Category IN ('Assault', 'Homicide', 'Rape', 'Robbery', 'Human Trafficking (A), Commercial Sex Acts', 'Human Trafficking, Commercial Sex Acts', 'Human Trafficking (B), Involuntary Servitude', 'Offences Against The Family And Children', 'Weapons Carrying Etc', 'Weapons Offense', 'Weapons Offence') THEN 'Violent Crime' WHEN Incident_Category IN ('Arson', 'Burglary', 'Forgery And Counterfeiting', 'Fraud', 'Larceny Theft', 'Motor Vehicle Theft', 'Motor Vehicle Theft?', 'Stolen Property', 'Vandalism', 'Embezzlement', 'Recovered Vehicle', 'Vehicle Impounded', 'Vehicle Misplaced') THEN 'Property Crime' WHEN Incident_Category IN ('Drug Offense', 'Drug Violation') THEN 'Drug Crimes' ELSE 'Other Crimes' END AS grouped_category, Report_Type_Description, Police_District, date_trunc_y(Report_Datetime) AS year, COUNT(*) AS incident_count WHERE Report_Datetime >= start_date GROUP BY supervisor_district, grouped_category, Report_Type_Description, Police_District, Incident_Category, Incident_Subcategory, year ORDER BY year, grouped_category"
                }
                }
                
                Ensure the output is strictly formatted as valid JSON.  No operators or additioonal characters, Do not include any additional text or explanations outside the JSON block.
            """

        # This is a placeholder for the removed Swarm functionality
        response_content = {}

        # The rest of the processing logic expects response_content to be a dictionary
        if isinstance(response_content, dict):
            # Merge original data with AI-generated data
            merged_data = {**data, **response_content}
            
            # Save the merged result
            try:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, indent=4)
                print(f"Successfully processed and saved result for '{filename}' at '{result_path}'")
            except Exception as e:
                error_msg = f"Error saving result file {result_filename}: {str(e)}"
                print(error_msg)
                error_log.append({'filename': filename, 'error': error_msg, 'phase': 'result_saving'})
        else:
            error_msg = f"AI response for {filename} was not a valid dictionary."
            print(error_msg)
            error_log.append({'filename': filename, 'error': error_msg, 'phase': 'ai_processing'})

    except Exception as e:
        # Detailed error logging
        tb_str = traceback.format_exc()
        error_msg = f"An unexpected error occurred while processing file '{filename}': {str(e)}\n{tb_str}"
        print(error_msg)
        error_log.append({'filename': filename, 'error': error_msg, 'phase': 'unknown'})


def main(filename=None):
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    datasets_folder = os.path.join(data_folder, 'datasets')
    output_folder = os.path.join(data_folder, 'datasets/fixed')
    error_log_path = os.path.join(output_folder, 'error_log.json')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize error log
    error_log = []

    # Threshold date
    threshold_date = datetime(2024, 9, 1, tzinfo=pytz.UTC)

    if filename:
        process_single_file(filename, datasets_folder, output_folder, threshold_date, error_log)
    else:
        # Process all files if no specific filename is provided
        article_list = [f for f in os.listdir(datasets_folder) if f.endswith('.json')]
        for idx, filename in enumerate(article_list):
            process_single_file(filename, datasets_folder, output_folder, threshold_date, error_log)

    # Write error log
    with open(error_log_path, 'w', encoding='utf-8') as err_f:
        json.dump(error_log, err_f, indent=4)


if __name__ == "__main__":
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    datasets_folder = os.path.join(data_folder, 'datasets')
    output_folder = os.path.join(data_folder, 'datasets/fixed')
    # Check for a filename argument
    filename_arg = sys.argv[1] if len(sys.argv) > 1 else None
    # main(filename_arg)

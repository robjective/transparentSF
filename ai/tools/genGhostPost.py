import requests
import json
import os

node_service_url = "http://localhost:3000"

def generate_ghost_post(context_variables: dict, content, title=None):
    doc_title = title or context_variables.get("doc_title", "My Analysis Chart")

    # Ensure mobiledoc is a JSON string
    if isinstance(content, dict):
        try:
            content = json.dumps(content)  # Convert dictionary to a JSON string
        except (TypeError, ValueError) as e:
            print(f"Error encoding content: {e}")
            return f"Error: Invalid content format: {e}"

    # Validate that content is a string
    if not isinstance(content, str):
        print("Error: content is not a string.")
        return "Error: content must be a string."

    # Send POST request to Node.js service
    try:
        response = requests.post(
            f"{node_service_url}/create-post",
            json={
                "title": doc_title,
                "content": content  # Properly escaped JSON string
            }
        )
        response.raise_for_status()
        print("Post created successfully:", response.json()["post"]["url"])
        return response.json()["post"]["url"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error: " + str(e)

def publish_newsletter_to_ghost(filename, title=None):
    """
    Fetches the HTML content from a newsletter file and publishes it to Ghost.
    
    Args:
        filename (str): The filename of the newsletter HTML file to publish
        title (str, optional): The title to use for the Ghost post. If None, extracts from filename
        
    Returns:
        str: URL of the published post or error message
    """
    try:
        # Construct the file path using the same approach as in backend.py
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to /ai directory
        reports_dir = os.path.join(script_dir, "output", "reports")
        file_path = os.path.join(reports_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return f"Error: File not found: {filename}"
        
        # Check if an email-compatible version exists
        email_filename = filename.replace('.html', '_email.html')
        email_file_path = os.path.join(reports_dir, email_filename)
        
        if os.path.exists(email_file_path):
            print(f"Found email-compatible version: {email_file_path}")
            file_path = email_file_path
            filename = email_filename
        else:
            # Try to create an email-compatible version
            try:
                from ..monthly_report import generate_email_compatible_report
                email_compatible_path = generate_email_compatible_report(file_path)
                if email_compatible_path and os.path.exists(email_compatible_path):
                    print(f"Generated email-compatible version: {email_compatible_path}")
                    file_path = email_compatible_path
                    filename = os.path.basename(email_compatible_path)
            except Exception as e:
                print(f"Error generating email-compatible version: {e}")
                # Continue with original file if email version generation fails
        
        # Read the HTML content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract title from filename if not provided
        if title is None:
            # Format: monthly_report_0_20241101.html -> San Francisco Crime Report - November 2024
            try:
                parts = filename.split('_')
                date_part = parts[3].split('.')[0]  # Extract date part without extension
                month = date_part[4:6]
                year = date_part[0:4]
                
                months = {
                    "01": "January", "02": "February", "03": "March", "04": "April",
                    "05": "May", "06": "June", "07": "July", "08": "August",
                    "09": "September", "10": "October", "11": "November", "12": "December"
                }
                
                month_name = months.get(month, month)
                title = f"San Francisco Crime Report - {month_name} {year}"
            except Exception as e:
                print(f"Error parsing filename for title: {e}")
                title = "San Francisco Crime Report"
        
        # Create empty context variables dict
        context_variables = {"doc_title": title}
        
        # Call the Ghost post generation function
        return generate_ghost_post(context_variables, content, title)
        
    except Exception as e:
        print(f"Error publishing newsletter to Ghost: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    context_variables = {
        "doc_title": "Surprising Crime Statistics for November 2024"
    }
    content = "<p><img src='../static/chart_89b60b30.png' alt='Incident Count by Category for November 2024 compared to historical averages.' /><br>In a surprising development, November 2024 saw a significant reduction in crime rates across San Francisco. This shift marks a pivotal moment for the city with several key data points highlighting notable decreases in various crime categories.</p>"

    title = "Surprising Crime Statistics for November 2024"
    generate_ghost_post(context_variables, content, title)

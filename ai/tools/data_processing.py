import datetime
from urllib.parse import urlparse

def format_columns(columns):
    """Format the columns information into a readable string for embedding."""
    if not columns:
        return ""
    formatted = "Columns Information:\n"
    for col in columns:
        formatted += f"- **{col['name']}** ({col['dataTypeName']}): {col['description']}\n"
    return formatted

def serialize_columns(columns):
    """Serialize the columns into a structured dictionary for payload."""
    if not columns:
        return {}
    serialized = {}
    for col in columns:
        serialized[col['name']] = {
            "fieldName": col.get('fieldName', ''),
            "dataTypeName": col.get('dataTypeName', ''),
            "description": col.get('description', ''),
            "position": col.get('position', ''),
            "renderTypeName": col.get('renderTypeName', ''),
            "tableColumnId": col.get('tableColumnId', '')
        }
    return serialized

def extract_endpoint(url):
    """Extract the Socrata endpoint from the given URL."""
    parsed_url = urlparse(url)
    endpoint = parsed_url.path
    if parsed_url.query:
        endpoint += f"?{parsed_url.query}"
    return endpoint

def convert_to_timestamp(date_input):
    """Convert ISO date string or datetime object to Unix timestamp."""
    if not date_input:
        return 0
    try:
        # If it's already a datetime object, use it directly
        if isinstance(date_input, datetime.datetime):
            return int(date_input.timestamp())
        # If it's a string, parse it
        elif isinstance(date_input, str):
            dt = datetime.datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            return int(dt.timestamp())
        else:
            return 0
    except (ValueError, AttributeError):
        return 0 
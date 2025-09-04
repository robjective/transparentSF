import os
import json
import time
import requests
from datetime import datetime

# Directory to save dataset URLs
data_directory = "data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

def fetch_all_datasets():
    """Fetch all dataset metadata from SF's Socrata API."""
    print("Fetching all datasets from SF's Socrata API...")
    
    # API endpoint for all datasets
    api_url = "https://data.sfgov.org/api/views.json"
    
    try:
        print(f"Making request to: {api_url}")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        datasets = response.json()
        print(f"Successfully fetched {len(datasets)} datasets from API")
        
        # Convert to the format we need
        dataset_urls = []
        for dataset in datasets:
            dataset_id = dataset.get('id')
            if dataset_id:
                # Create the full URL in the format we want
                dataset_url = f"https://data.sfgov.org/d/{dataset_id}"
                dataset_urls.append(dataset_url)
        
        print(f"Generated {len(dataset_urls)} dataset URLs")
        
        # Save to file
        file_path = os.path.join(data_directory, "dataset_urls.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_urls, f, ensure_ascii=False, indent=4)
        
        print(f"Saved {len(dataset_urls)} dataset URLs to {file_path}")
        
        # Also save the full metadata for reference
        metadata_file = os.path.join(data_directory, "dataset_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, ensure_ascii=False, indent=4)
        
        print(f"Saved full metadata to {metadata_file}")
        
        return dataset_urls
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching datasets from API: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def scrape_dataset_urls():
    """Legacy function - now uses API instead of web scraping."""
    print("Using API approach instead of web scraping...")
    return fetch_all_datasets()

if __name__ == '__main__':
    scrape_dataset_urls()

import os
import json
import logging
from pathlib import Path


def save_notes_to_file(notes_text, filename="combined_notes.txt"):
    """
    Saves the combined notes to a file in the output/notes directory.
    """
    logger = logging.getLogger(__name__)
    
    script_dir = Path(__file__).parent.parent  # Go up from tools/ to ai/
    notes_dir = script_dir / 'output' / 'notes'
    
    # Create notes directory if it doesn't exist
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = notes_dir / filename
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(notes_text)
        logger.info(f"Successfully saved notes to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving notes to file: {e}")
        return False


def load_and_combine_notes():
    """
    Loads and combines notes from dashboard data files.
    """
    logger = logging.getLogger(__name__)
    
    script_dir = Path(__file__).parent.parent  # Go up from tools/ to ai/
    dashboard_dir = script_dir / 'output' / 'dashboard'
    combined_text = ""
    districts_processed = 0
    
    # Process each district folder (0-11)
    for district_num in range(12):  # Include all districts 0-11
        district_dir = dashboard_dir / str(district_num)
        top_level_file = district_dir / 'top_level.json'
        
        if top_level_file.exists():
            try:
                with open(top_level_file, 'r', encoding='utf-8') as f:
                    district_data = json.load(f)
                
                district_name = district_data.get("name", f"District {district_num}")
                combined_text += f"\n{'='*80}\n{district_name} Metrics Summary\n{'='*80}\n\n"
                
                # Process each category
                for category in district_data.get("categories", []):
                    category_name = category.get("category", "")
                    combined_text += f"\n{category_name}:\n"
                    
                    # Process all metrics in the category
                    metrics = category.get("metrics", [])
                    for metric in metrics:
                        name = metric.get("name", "")
                        this_year = metric.get("thisYear", 0)
                        last_year = metric.get("lastYear", 0)
                        last_date = metric.get("lastDataDate", "")
                        metric_id = metric.get("numeric_id", metric.get("id", ""))
                        
                        # Calculate percent change
                        if last_year != 0:
                            pct_change = ((this_year - last_year) / last_year) * 100
                            change_text = f"({pct_change:+.1f}% vs last year)"
                        else:
                            change_text = "(no prior year data)"
                        
                        combined_text += f"- {name} (ID: {metric_id}): {this_year:,} {change_text} as of {last_date}\n"
                
                districts_processed += 1
            except Exception as e:
                logger.error(f"Error processing district {district_num} top-level metrics: {e}")
    
    logger.info(f"""
Notes loading complete:
Districts processed: {districts_processed}
Total combined length: {len(combined_text)} characters
First 100 characters: {combined_text[:100]}
""")
    
    # Save the combined notes to a file
    save_notes_to_file(combined_text)
    
    return combined_text


def get_notes(context_variables, *args, **kwargs):
    """
    Returns the notes from context variables, with length checking and logging.
    If notes are not in context variables, loads them from the dashboard data.
    """
    logger = logging.getLogger(__name__)
    
    # Define the maximum message length (OpenAI's limit)
    MAX_MESSAGE_LENGTH = 1048576  # 1MB in characters
    
    try:
        notes = context_variables.get("notes", "").strip()
        
        # If notes are empty in context, try to load them
        if not notes:
            logger.info("Notes not found in context variables, loading from dashboard data...")
            notes = load_and_combine_notes()
            # Update context variables with loaded notes
            context_variables["notes"] = notes
        
        total_length = len(notes)
        
        logger.info(f"""
=== get_notes called ===
Total length: {total_length} characters
Approximate tokens: {len(notes.split())}
First 100 chars: {notes[:100]}
Number of lines: {len(notes.splitlines())}
""")
        
        # If notes exceed OpenAI's limit, truncate them
        if total_length > MAX_MESSAGE_LENGTH:
            logger.warning(f"""
Notes exceed maximum length:
Current length: {total_length}
Maximum allowed: {MAX_MESSAGE_LENGTH}
Difference: {total_length - MAX_MESSAGE_LENGTH}
""")
            # Keep the first part and last part with a message in between
            keep_length = (MAX_MESSAGE_LENGTH // 2) - 100  # Leave room for the truncation message
            truncation_message = "\n\n[CONTENT TRUNCATED DUE TO LENGTH]\n\n"
            notes = notes[:keep_length] + truncation_message + notes[-keep_length:]
            logger.info(f"""
Notes truncated:
New length: {len(notes)}
Truncation point: {keep_length}
""")
        
        if notes:
            return {"notes": notes}
        else:
            logger.error("No notes found or notes are empty")
            return {"error": "No notes found or notes are empty"}
            
    except Exception as e:
        logger.error(f"Error in get_notes: {str(e)}")
        return {"error": f"Error processing notes: {str(e)}"}


def initialize_notes():
    """
    Initialize and return the combined notes. 
    This can be called at module import time to pre-load the notes.
    """
    return load_and_combine_notes() 
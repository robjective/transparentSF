"""
Robust prompts loading utility for TransparentSF.
Provides shared functionality for loading prompts.json with error handling and retry logic.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Module-level cache to ensure prompts are only loaded once per process
_PROMPTS_CACHE = None

logger = logging.getLogger(__name__)

def get_fallback_prompts() -> Dict[str, Any]:
    """
    Return minimal fallback prompts in case the main prompts file cannot be loaded.
    This ensures the application can continue to function with basic functionality.
    
    Returns:
        Dictionary containing minimal prompts
    """
    return {
        "monthly_report": {
            "generate_report_text": {
                "system": "You are a helpful assistant for assembling newsletter stories.",
                "prompt": "Review the following content and assemble it into a 3-5 paragraph news story. Keep it factual and clear. Content: {rationale} {explanation} {trend_analysis} {charts} {citations} {perplexity_context} {citywide_changes}"
            },
            "prioritize_deltas": {
                "system": "You are a data analyst helping prioritize important changes.",
                "prompt": "Identify the {max_items} most important changes from the following data: {changes_text} {notes_text_short}. Return JSON with items array containing index, metric, metric_id, group, priority, and explanation fields."
            },
            "generate_report": {
                "system": "You are a professional data scientist and newsletter writer.",
                "prompt": "Create a comprehensive monthly newsletter with the provided data. Use clear, factual language."
            },
            "proofread": {
                "system": "You are a professional editor and proofreader.",
                "prompt": "Review and improve the following newsletter content for clarity and accuracy."
            },
            "context_enrichment": {
                "system": "You are an information retrieval expert and analyst.",
                "prompt": "Provide additional context and recent news about the topics mentioned."
            },
            "audio_transformation": {
                "system": "You are a professional script writer specializing in audio content.",
                "prompt": "Transform the following newsletter content into an optimized audio script for text-to-speech narration."
            }
        }
    }

def load_prompts_with_retry(prompts_path: Path = None, use_cache: bool = True, max_retries: int = 3) -> Dict[str, Any]:
    """
    Load prompts from the JSON file with robust error handling and retry logic.
    
    Args:
        prompts_path: Path to the prompts.json file. If None, uses default location.
        use_cache: Whether to use cached prompts if already loaded
        max_retries: Maximum number of retry attempts
    
    Returns:
        Dictionary containing all prompts
    """
    global _PROMPTS_CACHE
    
    # Return cached prompts if available and caching is enabled
    if use_cache and _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    
    # Determine prompts path
    if prompts_path is None:
        prompts_path = Path(__file__).parent.parent / 'data' / 'prompts.json'
    
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if file exists and is readable
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
            
            if not prompts_path.is_file():
                raise ValueError(f"Prompts path is not a file: {prompts_path}")
            
            # Check file size (should be reasonable for a JSON file)
            file_size = prompts_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"Prompts file is empty: {prompts_path}")
            
            if file_size < 100:  # Suspiciously small for our prompts
                raise ValueError(f"Prompts file is too small ({file_size} bytes), may be corrupted: {prompts_path}")
            
            logger.info(f"Loading prompts from {prompts_path} (attempt {attempt + 1}/{max_retries}, file size: {file_size} bytes)")
            
            # Try to read and parse the file with specific error handling
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    
                # Validate that we got content
                if not file_content.strip():
                    raise ValueError("File content is empty after reading")
                
                # Parse JSON with better error reporting
                try:
                    prompts = json.loads(file_content)
                except json.JSONDecodeError as json_err:
                    raise ValueError(f"JSON parsing error at line {json_err.lineno}, column {json_err.colno}: {json_err.msg}")
                    
            except OSError as io_err:
                # Handle specific I/O errors
                if io_err.errno == 5:  # Input/output error
                    raise IOError(f"I/O error reading file (errno 5): {io_err}. This may indicate disk issues or file corruption.")
                else:
                    raise IOError(f"File I/O error (errno {io_err.errno}): {io_err}")
            
            # Validate the loaded prompts structure
            if not isinstance(prompts, dict):
                raise ValueError(f"Loaded prompts is not a dictionary: {type(prompts)}")
            
            if 'monthly_report' not in prompts:
                raise ValueError("Missing 'monthly_report' key in prompts")
            
            logger.info(f"Successfully loaded prompts with keys: {list(prompts.keys())}")
            
            # Cache the successfully loaded prompts
            if use_cache:
                _PROMPTS_CACHE = prompts
                
            return prompts
            
        except (FileNotFoundError, ValueError, IOError) as e:
            logger.error(f"Error loading prompts (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Failed to load prompts after {max_retries} attempts")
                logger.warning("Using minimal fallback prompts due to persistent loading failures")
                fallback = get_fallback_prompts()
                if use_cache:
                    _PROMPTS_CACHE = fallback
                return fallback
            else:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        except Exception as e:
            # Unexpected error, don't retry
            logger.error(f"Unexpected error loading prompts: {str(e)}")
            
            if attempt == max_retries - 1:  # Last attempt, try fallback
                logger.warning("Using minimal fallback prompts due to persistent loading failures")
                fallback = get_fallback_prompts()
                if use_cache:
                    _PROMPTS_CACHE = fallback
                return fallback
            else:
                raise RuntimeError(f"Unexpected error loading prompts: {str(e)}")

def clear_prompts_cache():
    """Clear the cached prompts, forcing a reload on next access."""
    global _PROMPTS_CACHE
    _PROMPTS_CACHE = None

def save_prompts(prompts: Dict[str, Any], prompts_path: Path = None, backup: bool = True) -> bool:
    """
    Save prompts to the JSON file with backup and validation.
    
    Args:
        prompts: Dictionary containing prompts to save
        prompts_path: Path to save the prompts file. If None, uses default location.
        backup: Whether to create a backup before saving
    
    Returns:
        True if successful, False otherwise
    """
    if prompts_path is None:
        prompts_path = Path(__file__).parent.parent / 'data' / 'prompts.json'
    
    try:
        # Validate prompts structure
        if not isinstance(prompts, dict):
            raise ValueError(f"Prompts must be a dictionary, got {type(prompts)}")
        
        if 'monthly_report' not in prompts:
            raise ValueError("Missing 'monthly_report' key in prompts")
        
        # Create backup if requested and file exists
        if backup and prompts_path.exists():
            backup_path = prompts_path.with_suffix(f'.json.backup.{int(time.time())}')
            import shutil
            shutil.copy2(prompts_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Save prompts with atomic write (write to temp file, then rename)
        temp_path = prompts_path.with_suffix('.tmp')
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_path.replace(prompts_path)
        
        logger.info(f"Successfully saved prompts to {prompts_path}")
        
        # Clear cache to force reload on next access
        clear_prompts_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving prompts: {str(e)}")
        return False

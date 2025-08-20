import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import json
import logging
import time

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent
sys.path.insert(0, str(ai_dir))

try:
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Import swarm components
    from swarm import Swarm, Agent
    from openai import OpenAI
    
    # Import tools from tools directory
    from tools.vector_query import query_docs
    from tools.data_fetcher import set_dataset
    from tools.anomaly_detection import anomaly_detection
    from tools.generate_map import generate_map, get_map_by_id, get_recent_maps
    from tools.store_anomalies import store_anomaly_data, get_anomalies, get_anomaly_details as get_anomaly_details_from_db
    from tools.genChart import generate_time_series_chart
    from tools.notes_manager import get_notes, load_and_combine_notes, initialize_notes
    from tools.get_charts_for_review import get_charts_for_review
    
    # Import metrics management tools
    from tools.explainer_metrics_tools import (
        query_metrics, get_metric_details, 
        create_new_metric, edit_metric, disable_metric, enable_metric,
        get_metrics_overview, find_metrics_by_endpoint
    )
    
    # Import prompts from the modular structure
    from .explainer_prompts import get_complete_instructions, PROMPT_SECTIONS
    
    # Verify we got the real functions, not dummy ones
    # (Will log this after logger is initialized)
    
    # Import other utility functions from webChat (only those not in notes_manager)
    from webChat import (
        get_dataset,
        query_anomalies_db,
        get_dataset_columns,
        generate_chart_message,
        get_columns,
        get_data_summary,
        format_table,
        format_table_page,
        transfer_to_analyst_agent,
        transfer_to_researcher_agent
    )
    from tools.dashboard_metric_tool import get_dashboard_metric
    
except ImportError as e:
    print(f"Warning: Import error in explainer_agent: {e}")
    # For testing purposes, provide dummy functions
    def dummy_func(*args, **kwargs):
        return {"success": True, "content": "Test mode"}
    
    get_notes = get_dataset = get_dashboard_metric = dummy_func
    query_anomalies_db = get_anomaly_details_from_db = get_dataset_columns = dummy_func
    generate_chart_message = get_columns = get_data_summary = dummy_func
    format_table = format_table_page = transfer_to_analyst_agent = dummy_func
    transfer_to_researcher_agent = query_docs = anomaly_detection = dummy_func
    set_dataset = generate_map = get_map_by_id = get_recent_maps = dummy_func
    store_anomaly_data = generate_time_series_chart = dummy_func
    load_and_combine_notes = initialize_notes = get_charts_for_review = dummy_func
    
    # Metrics tools dummy functions - only if import failed
    if 'query_metrics' not in locals():
        query_metrics = get_metric_details = dummy_func
        create_new_metric = edit_metric = disable_metric = enable_metric = dummy_func
        get_metrics_overview = find_metrics_by_endpoint = dummy_func
        print("Warning: Using dummy metrics functions due to import failure")

# Function mapping for tool execution (similar to webChat)
function_mapping = {
    'transfer_to_analyst_agent': transfer_to_analyst_agent,
    'transfer_to_researcher_agent': transfer_to_researcher_agent,
    'get_dataset': get_dataset,
    'get_notes': get_notes,
    'get_columns': get_columns,
    'get_data_summary': get_data_summary,
    'anomaly_detection': anomaly_detection,
    'query_docs': query_docs,
    'set_dataset': set_dataset,
    'generate_time_series_chart': generate_time_series_chart,
    'get_dashboard_metric': get_dashboard_metric,
    'format_table': format_table,
    'format_table_page': format_table_page,
    'generate_chart_message': generate_chart_message,
    'generate_map': generate_map,
    'get_map_by_id': get_map_by_id,
    'get_recent_maps': get_recent_maps,
    'query_anomalies_db': query_anomalies_db,
    'get_anomaly_details': get_anomaly_details_from_db,
    'get_dataset_columns': get_dataset_columns,
    'get_charts_for_review': get_charts_for_review,
    
    # Metrics management tools
    'query_metrics': query_metrics,
    'get_metric_details': get_metric_details,
    'create_new_metric': create_new_metric,
    'edit_metric': edit_metric,
    'disable_metric': disable_metric,
    'enable_metric': enable_metric,
    'get_metrics_overview': get_metrics_overview,
    'find_metrics_by_endpoint': find_metrics_by_endpoint,
}

# Load environment variables
from .config.models import get_default_model
# Use claude-3-7-sonnet as the default model for explainer agent
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-3-7-sonnet")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Configure logging to match monthly_report.py configuration
# Get logging level from environment or default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Create logs directory if it doesn't exist
script_dir = Path(__file__).parent
ai_dir = script_dir.parent
logs_dir = ai_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure file handler with absolute path to monthly_report.log
log_file = logs_dir / 'monthly_report.log'
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(log_level)

# Configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Initialize logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Remove any existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add handlers to this logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to avoid duplicate logs if root logger is also configured
logger.propagate = False

# Add Seymour's conversation log handler - always logs at INFO level
seymour_log_file = logs_dir / 'seymour.log'
seymour_handler = logging.FileHandler(seymour_log_file, mode='a', encoding='utf-8')
seymour_handler.setLevel(logging.INFO)  # Always log at INFO level
seymour_formatter = logging.Formatter('%(asctime)s - %(message)s')  # Simpler format for conversation
seymour_handler.setFormatter(seymour_formatter)
logger.addHandler(seymour_handler)

logger.info(f"Explainer agent logging initialized with level: {log_level_name}")
logger.info(f"Explainer agent log file location: {log_file}")
logger.info(f"Seymour conversation log file location: {seymour_log_file}")

# Log successful metrics import
try:
    logger.info(f"Successfully imported metrics functions from {query_metrics.__module__}")
except NameError:
    logger.warning("Metrics functions not available - using dummy functions")

# Initialize Swarm client
swarm_client = Swarm()

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def create_explainer_agent(context_variables: Optional[Dict[str, Any]] = None, model_key: Optional[str] = None) -> 'ExplainerAgent':
    """
    Factory function to create a new explainer agent instance.
    
    Args:
        context_variables: Optional context variables dictionary. If None, creates a new one.
        model_key: Optional model key to use (e.g., "gpt-4o", "claude-3-sonnet")
        
    Returns:
        ExplainerAgent instance
    """
    return ExplainerAgent(context_variables, model_key)


class ExplainerAgent:
    """
    A clean, reusable explainer agent that can be instantiated from anywhere in the app.
    Supports both programmatic JSON responses and streaming chat responses.
    """
    
    def __init__(self, context_variables: Optional[Dict[str, Any]] = None, model_key: Optional[str] = None):
        """
        Initialize the explainer agent.
        
        Args:
            context_variables: Optional context variables dictionary. If None, creates a new one with notes.
            model_key: Optional model key to use (e.g., "gpt-4o", "claude-3-sonnet")
        """
        # Initialize context variables with notes if not provided
        if context_variables is None:
            self.context_variables = {
                "dataset": pd.DataFrame(),
                "notes": initialize_notes()  # Load notes using the notes_manager tool
            }
        else:
            self.context_variables = context_variables
            # Ensure notes are available in context
            if "notes" not in self.context_variables or not self.context_variables["notes"]:
                self.context_variables["notes"] = initialize_notes()
        
        # Store model key
        self.model_key = model_key or AGENT_MODEL
        
        # Initialize conversation history
        self.messages = []
        
        self.swarm_client = swarm_client
        self.logger = logger
        
        # Create the agent using model factory
        from .model_factory import model_factory
        
        self.agent = model_factory.create_agent(
            model_key=self.model_key,
            name="Explainer",
            instructions=get_complete_instructions(),
            functions=[
                get_notes,
                get_dataset,
                set_dataset,
                query_docs,
                query_anomalies_db,
                get_dashboard_metric,
                get_anomaly_details_from_db,
                get_dataset_columns,
                generate_map,
                get_map_by_id,
                get_recent_maps,
                generate_time_series_chart,
                get_charts_for_review,
                
                # Metrics management tools
                query_metrics,
                get_metric_details,
                create_new_metric,
                edit_metric,
                disable_metric,
                enable_metric,
                get_metrics_overview,
                find_metrics_by_endpoint,
            ],
            context_variables=self.context_variables,
            debug=False,
        )
    
    def get_conversation_history(self) -> list:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()
    
    def set_conversation_history(self, messages: list):
        """
        Set the conversation history.
        
        Args:
            messages: List of message dictionaries
        """
        self.messages = messages.copy() if messages else []
    
    def add_message(self, role: str, content: str, sender: Optional[str] = None):
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            sender: Optional sender name (for assistant messages)
        """
        message = {"role": role, "content": content}
        if sender and role == "assistant":
            message["sender"] = sender
        self.messages.append(message)
        
        # Log conversation to Seymour's log (but not assistant messages to avoid log artifacts in response)
        if role == 'user':
            sender_name = sender if sender else role
            self.logger.info(f"{sender_name}: {content}")
        elif role == 'assistant':
            # Don't log assistant messages to avoid log artifacts appearing in the response
            pass
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.messages = []
    
    def reload_prompts(self):
        """
        Reload prompts from the modular structure and re-instantiate the agent.
        This is called when prompt sections are updated through the settings interface.
        """
        try:
            # Reload the prompts module to get latest changes
            import importlib
            from . import explainer_prompts
            importlib.reload(explainer_prompts)
            
            # Re-import the updated functions
            from .explainer_prompts import get_complete_instructions
            
            # Re-create the agent with updated instructions using model factory
            from .model_factory import model_factory
            self.agent = model_factory.create_agent(
                model_key=self.model_key,
                name="Explainer",
                instructions=get_complete_instructions(),
                functions=[
                    get_notes,
                    get_dataset,
                    set_dataset,
                    query_docs,
                    query_anomalies_db,
                    get_dashboard_metric,
                    get_anomaly_details_from_db,
                    get_dataset_columns,
                    generate_map,
                    get_map_by_id,
                    get_recent_maps,
                    generate_time_series_chart,
                    get_charts_for_review,
                    # Metrics management tools
                    query_metrics,
                    get_metric_details,
                    create_new_metric,
                    edit_metric,
                    disable_metric,
                    enable_metric,
                    get_metrics_overview,
                    find_metrics_by_endpoint,
                ],
                context_variables=self.context_variables,
                debug=False,
            )
            self.logger.info("Explainer agent prompts reloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reloading prompts: {str(e)}")
            return False
    
    def explain_change_sync(self, prompt: str, return_json: bool = False) -> Dict[str, Any]:
        """
        Synchronously explain a data change or anomaly.
        
        Args:
            prompt: The explanation prompt/question
            return_json: If True, attempts to parse response as JSON
            
        Returns:
            Dictionary containing the explanation and any errors
        """
        try:
            self.logger.info(f"Running explainer agent with prompt: {prompt}")
            
            # Clear conversation history to prevent token limit issues
            self.clear_conversation_history()
            
            # Add user message to conversation history
            self.add_message("user", prompt)
            
            # Set a maximum number of tool calls to prevent token accumulation
            max_tool_calls = 5
            tool_call_count = 0
            
            # Use the stored conversation history
            messages = self.messages.copy()
            
            response = self.swarm_client.run(
                agent=self.agent,
                messages=messages,
                context_variables=self.context_variables,
                debug=False,
            )
            
            # Extract the response content and update conversation history
            if hasattr(response, 'messages') and response.messages:
                # Find the last assistant message from the response
                content = ""
                for msg in reversed(response.messages):
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        content = msg.get('content', '')
                        break
                
                # Add assistant response to our conversation history
                if content:
                    self.add_message("assistant", content, self.agent.name)
                else:
                    # Fallback to the last message's content
                    content = response.messages[-1].get('content', '') if response.messages else ''
                    if content:
                        self.add_message("assistant", content, self.agent.name)
                        
            elif hasattr(response, 'content'):
                content = response.content
                # Add assistant response to conversation history
                self.add_message("assistant", content, self.agent.name)
            else:
                content = str(response)
                # Add assistant response to conversation history
                self.add_message("assistant", content, self.agent.name)
            
            result = {
                'success': True,
                'content': content,
                'context_variables': self.context_variables,
                'conversation_history': self.get_conversation_history()
            }
            
            # Try to parse as JSON if requested
            if return_json and content:
                try:
                    # Look for JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed_json = json.loads(json_match.group())
                        result['parsed_json'] = parsed_json
                except (json.JSONDecodeError, AttributeError) as e:
                    self.logger.warning(f"Failed to parse response as JSON: {e}")
                    result['json_parse_error'] = str(e)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in explain_change_sync: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': '',
                'context_variables': self.context_variables,
                'conversation_history': self.get_conversation_history()
            }
    
    async def explain_change_streaming(self, prompt: str, session_data: Optional[Dict] = None):
        """
        Asynchronously explain a data change with streaming response.
        
        Args:
            prompt: The explanation prompt/question
            session_data: Optional session data for maintaining conversation state
            
        Yields:
            String chunks of the response as they're generated
        """
        try:
            self.logger.info(f"Running streaming explainer agent with prompt: {prompt}")
            
            # Use provided session data or stored conversation history
            if session_data and "messages" in session_data:
                messages = session_data["messages"].copy()
                context_variables = session_data.get("context_variables", self.context_variables)
            else:
                messages = self.messages.copy()
                context_variables = self.context_variables
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Run streaming response
            response_generator = self.swarm_client.run(
                agent=self.agent,
                messages=messages,
                context_variables=context_variables,
                stream=True,
                debug=False,
            )
            
            # Initialize variables for tracking the conversation
            current_function_name = None
            incomplete_tool_call = None
            assistant_message = {"role": "assistant", "content": "", "sender": self.agent.name}
            
            for chunk in response_generator:
                # Handle tool calls
                if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                    for tool_call in chunk["tool_calls"]:
                        function_info = tool_call.get("function")
                        if not function_info:
                            continue
                        
                        if function_info.get("name"):
                            current_function_name = function_info["name"]
                            self.logger.info(f"Receiving tool call: {current_function_name}")
                        
                        if not current_function_name:
                            continue
                        
                        arguments_fragment = function_info.get("arguments", "")
                        
                        if incomplete_tool_call is None or incomplete_tool_call["function_name"] != current_function_name:
                            incomplete_tool_call = {
                                "type": "tool_call",
                                "sender": "Explainer",
                                "function_name": current_function_name,
                                "arguments": ""
                            }
                        
                        incomplete_tool_call["arguments"] += arguments_fragment
                        
                        try:
                            arguments_json = json.loads(incomplete_tool_call["arguments"])
                            self.logger.info(f"Tool Call - Function: {current_function_name}, Arguments: {json.dumps(arguments_json, indent=2)}")

                            # Send tool call start notification as structured data
                            tool_id = f'tool-{current_function_name}-{int(time.time())}'
                            tool_start_data = {'tool_call_start': current_function_name, 'tool_id': tool_id}
                            yield f"data: {json.dumps(tool_start_data)}\n\n"
                            
                            # Send tool call arguments
                            tool_args_data = {'tool_call_args': current_function_name, 'tool_id': tool_id, 'arguments': arguments_json}
                            yield f"data: {json.dumps(tool_args_data)}\n\n"

                            # Process the function call
                            function_to_call = function_mapping.get(current_function_name)
                            if function_to_call:
                                try:
                                    # Handle different function call patterns (same as webChat)
                                    if current_function_name == 'generate_chart_message':
                                        chart_data = arguments_json.get('chart_data')
                                        chart_type = arguments_json.get('chart_type', 'anomaly')
                                        result = function_to_call(chart_data=chart_data, chart_type=chart_type)
                                    else:
                                        # Standard function call with context_variables
                                        if 'context_variables' in arguments_json:
                                            del arguments_json['context_variables']
                                        result = function_to_call(context_variables, **arguments_json)
                                    
                                    self.logger.info(f"Tool result: {str(result)[:200]}...")

                                    # Determine success by inspecting the result content
                                    success_flag = True
                                    error_message = None
                                    try:
                                        if isinstance(result, dict):
                                            status_val = str(result.get('status', '')).lower()
                                            if 'error' in result:
                                                success_flag = False
                                                error_message = str(result.get('error'))
                                            elif status_val and status_val != 'success':
                                                success_flag = False
                                                error_message = f"status={result.get('status')}"
                                            elif 'error_type' in result:
                                                success_flag = False
                                                error_message = str(result.get('error_type'))
                                        elif isinstance(result, str):
                                            # Attempt to parse a dict-like string
                                            try:
                                                parsed = json.loads(result)
                                            except Exception:
                                                try:
                                                    import ast
                                                    parsed = ast.literal_eval(result)
                                                except Exception:
                                                    parsed = None
                                            if isinstance(parsed, dict):
                                                status_val = str(parsed.get('status', '')).lower()
                                                if 'error' in parsed:
                                                    success_flag = False
                                                    error_message = str(parsed.get('error'))
                                                elif status_val and status_val != 'success':
                                                    success_flag = False
                                                    error_message = f"status={parsed.get('status')}"
                                                elif 'error_type' in parsed:
                                                    success_flag = False
                                                    error_message = str(parsed.get('error_type'))
                                            else:
                                                lower = result.lower()
                                                indicators = [
                                                    'error', 'exception', 'failed', 'failure', 'invalid',
                                                    'no-such-column', 'query coordinator error', 'soql error',
                                                    '400 client error', 'bad request', 'could not parse',
                                                    'expected', 'table identifier'
                                                ]
                                                if any(ind in lower for ind in indicators):
                                                    success_flag = False
                                                    error_message = result
                                    except Exception as e:
                                        self.logger.warning(f"Could not determine tool success reliably: {e}")

                                    # Send tool call completion notification as structured data
                                    tool_complete_data = {
                                        'tool_call_complete': current_function_name,
                                        'tool_id': tool_id,
                                        'success': success_flag,
                                        'response': make_json_serializable(result)
                                    }
                                    if not success_flag and error_message:
                                        tool_complete_data['error'] = error_message
                                    yield f"data: {json.dumps(tool_complete_data)}\n\n"
                                    
                                    # Check if this is an agent transfer function
                                    if current_function_name in ['transfer_to_analyst_agent', 'transfer_to_researcher_agent']:
                                        # Update the current agent
                                        self.agent = result
                                        transfer_content = f'**Transferred to {result.name} Agent**\n\n'
                                        content_data = {'content': transfer_content}
                                        yield f"data: {json.dumps(content_data)}\n\n"
                                    # If the result has content (like from format_table), send it as a message
                                    elif isinstance(result, dict) and "content" in result:
                                        result_content = f'{result["content"]}\n\n'
                                        content_data = {'content': result_content}
                                        yield f"data: {json.dumps(content_data)}\n\n"
                                    # Handle chart messages
                                    elif isinstance(result, dict) and result.get("type") == "chart":
                                        chart_content = f'**Chart generated: {result.get("chart_id")}**\n\n'
                                        content_data = {'content': chart_content}
                                        yield f"data: {json.dumps(content_data)}\n\n"
                                        
                                except Exception as tool_error:
                                    self.logger.error(f"Error executing tool {current_function_name}: {str(tool_error)}")
                                    # Send tool call error notification as structured data
                                    tool_error_data = {
                                        'tool_call_complete': current_function_name, 
                                        'tool_id': tool_id, 
                                        'success': False, 
                                        'error': str(tool_error),
                                        'response': make_json_serializable({'error': str(tool_error)})
                                    }
                                    yield f"data: {json.dumps(tool_error_data)}\n\n"
                            else:
                                self.logger.warning(f"Unknown tool: {current_function_name}")
                                # Send tool call error notification as structured data
                                unknown_tool_error_data = {
                                    'tool_call_complete': current_function_name, 
                                    'tool_id': tool_id, 
                                    'success': False, 
                                    'error': f'Unknown tool: {current_function_name}',
                                    'response': make_json_serializable({'error': f'Unknown tool: {current_function_name}'})
                                }
                                yield f"data: {json.dumps(unknown_tool_error_data)}\n\n"

                            incomplete_tool_call = None
                            current_function_name = None
                        except json.JSONDecodeError:
                            # Still accumulating arguments
                            pass

                # Handle content
                elif "content" in chunk and chunk["content"] is not None:
                    content_piece = chunk["content"]
                    assistant_message["content"] += content_piece
                    # Send content as SSE data for consistency
                    content_data = {'content': content_piece}
                    yield f"data: {json.dumps(content_data)}\n\n"

                # Handle delim (end of message/agent response)
                if "delim" in chunk and chunk["delim"] == "end":
                    # Always append assistant message if it has content
                    if assistant_message["content"]:
                        messages.append(assistant_message)
                        self.logger.info(f"Agent Response Complete: {assistant_message['sender']}")
                    # Reset for next message
                    assistant_message = {"role": "assistant", "content": "", "sender": self.agent.name}

                # Handle response object (final response)
                if "response" in chunk:
                    self.logger.info("Received final response object, ending stream")
                    break

            # Update stored conversation history
            self.messages = messages
            
            self.logger.info(f"Streaming complete. Total messages: {len(messages)}")

        except Exception as e:
            self.logger.error(f"Error in explain_change_streaming: {str(e)}")
            yield f"\n\n❌ **Error: {str(e)}**\n"


def explain_metric_change(
    metric_id: int,
    district_id: int = 0,
    period_type: str = "month",
    context_variables: Optional[Dict[str, Any]] = None,
    return_json: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to explain a specific metric change.
    
    Args:
        metric_id: The metric ID to explain
        district_id: District ID (0 for citywide, 1-11 for specific districts)
        period_type: Period type (month, year, etc.)
        context_variables: Optional context variables
        return_json: Whether to return structured JSON response
        
    Returns:
        Dictionary containing the explanation
    """
    agent = create_explainer_agent(context_variables)
    
    prompt = f"""
    Please explain the recent changes in metric {metric_id} for district {district_id} 
    over the {period_type} period. Provide a comprehensive analysis including:
    
    1. What specifically changed in the data
    2. Any anomalies that might explain the change
    3. Historical context for this change
    4. Geographic patterns if relevant
    
    Please follow your standard workflow and provide charts and supporting evidence.
    """
    
    return agent.explain_change_sync(prompt, return_json=return_json)


def make_json_serializable(obj):
    """
    Convert objects to JSON serializable format.
    Handles pandas DataFrames, numpy arrays, and other common non-serializable types.
    """
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": obj.shape,
            "columns": obj.columns.tolist(),
            "head": obj.head().to_dict('records') if not obj.empty else [],
            "dtypes": obj.dtypes.astype(str).to_dict(),
            "summary": f"DataFrame with {len(obj)} rows and {len(obj.columns)} columns"
        }
    elif isinstance(obj, pd.Series):
        return {
            "type": "Series",
            "name": obj.name,
            "length": len(obj),
            "head": obj.head().tolist() if not obj.empty else [],
            "dtype": str(obj.dtype)
        }
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return {
            "type": "array",
            "data": obj.tolist()
        }
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)  # Convert to string as fallback


# Export main classes and functions
__all__ = [
    'ExplainerAgent',
    'create_explainer_agent', 
    'explain_metric_change',
    'get_complete_instructions',
    'PROMPT_SECTIONS'
] 
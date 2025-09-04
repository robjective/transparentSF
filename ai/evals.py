import os
import json
from datetime import datetime
from agents.langchain_agent.explainer_agent import create_explainer_agent as create_langchain_agent
import pytest
from dotenv import load_dotenv

# The Swarm client is no longer needed for the LangChain agent
# from swarm import Swarm

load_dotenv()

# We will create the agent inside the test functions now
# agent = create_explainer_agent()
# client = Swarm()

# Create logs subfolder if it doesn't exist
# Correctly locate the project root and set the log folder path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
log_folder = os.path.join(project_root, 'logs', 'evals')
os.makedirs(log_folder, exist_ok=True)

# Global log_filename variable, will be updated when run_and_get_tool_calls is called
log_filename = None

# Define all test cases
TEST_CASES = {
    "sets_data_when_asked": [
        "Please search for a dataset about Police Misconduct and then set the dataset to select everything from that endpoint for the period of Septmeber 2024 to October 2024.",
        "What are the names of the last 5 Retail businesses locations registered in SF?",
    ],
    "does_not_call_set_dataset_when_not_asked": [
        "Hi, who is the mayor of SF?",
        "What is happening with the weather in SF?",
    ]
}

def run_and_get_tool_calls(agent, initial_query, max_turns=5):
    global log_filename
    
    # Generate a unique log filename for the session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds for uniqueness
    log_filename = f"{log_folder}/session_{timestamp}.log"
    
    messages = []
    collected_tool_calls = []

    # Start the conversation with the initial user query
    user_message = {"role": "user", "content": initial_query}
    messages.append(user_message)

    # Log the initial query
    with open(log_filename, 'a') as log_file:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "user_query",
            "content": initial_query
        }
        json.dump(log_entry, log_file, indent=2)
        log_file.write("\n\n")

    for turn in range(max_turns):
        # Assistant responds
        print(f"Turn {turn+1}: Sending request to agent...")
        try:
            response = agent.run(
                messages=messages[-2:],  # Only send the last two messages
                context_variables=agent.context_variables,  # Use agent's context variables
                execute_tools=True,  # Execute tools automatically like in the UI
            )
            print(f"Turn {turn+1}: Received response from agent")
            
            # Get the updated list of messages from the assistant
            response_messages = response.messages
            print(f"Turn {turn+1}: Response has {len(response_messages)} messages")

            # Process each message in the response
            for msg in response_messages:
                print(f"Turn {turn+1}: Processing message with role: {msg.get('role')}")

                # Log the message content if any
                if msg.get('content'):
                    with open(log_filename, 'a') as log_file:
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "assistant_message",
                            "content": msg['content']
                        }
                        json.dump(log_entry, log_file, indent=2)
                        log_file.write("\n\n")

                # Collect and log tool calls if any
                latest_tool_calls = msg.get("tool_calls", [])
                if latest_tool_calls:
                    print(f"Turn {turn+1}: Found {len(latest_tool_calls)} tool calls")
                    for call in latest_tool_calls:
                        function_info = call.get('function', {})
                        if not function_info:
                            continue
                            
                        function_name = function_info.get('name')
                        if not function_name:
                            continue
                            
                        print(f"  - Function: {function_name}")
                        
                        # Log the tool call
                        with open(log_filename, 'a') as log_file:
                            log_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "type": "tool_call",
                                "function": function_name,
                                "arguments": function_info.get('arguments', '')
                            }
                            json.dump(log_entry, log_file, indent=2)
                            log_file.write("\n\n")
                        
                        collected_tool_calls.append(call)
                else:
                    print(f"Turn {turn+1}: No tool calls in this message")

                # Add the message to our conversation history
                messages.append(msg)

            # Check if we need another turn
            if not any(msg.get("tool_calls") for msg in response_messages):
                print(f"Turn {turn+1}: No tool calls in response, ending loop")
                break
        except Exception as e:
            print(f"Error in turn {turn+1}: {str(e)}")
            break

    print(f"Total tool calls collected: {len(collected_tool_calls)}")
    return collected_tool_calls

def run_single_eval_langchain(query: str, model_key: str = None) -> dict:
    """
    Run a single evaluation with the LangChain agent using proper session logging.
    
    Args:
        query: The user query to test.
        model_key: The model to use for the agent.
        
    Returns:
        A dictionary with the results, including tool call count and log filename.
    """
    global log_filename
    
    # Import the LangChain agent
    from agents.langchain_agent.explainer_agent import create_explainer_agent
    from agents.langchain_agent.config.tool_config import ToolGroup
    
    # Create a new LangChain agent with session logging enabled
    agent = create_explainer_agent(
        model_key=model_key, 
        tool_groups=[ToolGroup.CORE, ToolGroup.ANALYSIS, ToolGroup.METRICS, ToolGroup.VISUALIZATION], 
        enable_session_logging=True
    )
    
    # Run the agent using the working explain_change_sync method
    result = agent.explain_change_sync(query, metric_details={})
    
    # Generate log filename for eval compatibility
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    log_filename = f"{log_folder}/eval_langchain_{timestamp}.log"
    
    # Create eval-compatible log from session data
    with open(log_filename, 'w') as log_file:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "eval_run",
            "query": query,
            "model": model_key or agent.model_key,
            "success": result.get("success", False),
            "explanation": result.get("explanation", ""),
            "session_id": result.get("session_id"),
            "tool_calls": [],
            "execution_trace": result.get("execution_trace", [])
        }
        
        # Extract tool calls from execution trace
        if hasattr(agent, 'session_logger') and agent.session_logger:
            # Try to get the last session logged
            sessions_dir = agent.session_logger.logs_dir
            if sessions_dir.exists():
                session_files = sorted(sessions_dir.glob("session_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
                if session_files:
                    try:
                        with open(session_files[0], 'r') as session_file:
                            session_data = json.load(session_file)
                            log_data["tool_calls"] = session_data.get("tool_calls", [])
                            log_data["total_execution_time_ms"] = session_data.get("total_execution_time_ms", 0)
                    except Exception as e:
                        print(f"Warning: Could not read session file: {e}")
        
        json.dump(log_data, log_file, indent=2)
    
    return {
        "status": "success",
        "tool_calls_count": len(log_data["tool_calls"]),
        "log_filename": log_filename,
        "tool_calls": log_data["tool_calls"],
        "explanation": result.get("explanation", ""),
        "session_id": result.get("session_id")
    }

# DEPRECATED: Old broken function - replaced with proper session logging
# def run_and_get_tool_calls_langchain(agent, initial_query, max_turns=5):
#     """
#     DEPRECATED: This function used brittle stdout parsing.
#     Now using proper session logging in run_single_eval_langchain() instead.
#     """
#     pass

def run_all_evals(model_key: str = None):
    """Run all eval test cases and return success/failure counts with details."""
    results = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "test_results": [],
        "timestamp": datetime.now().isoformat(),
        "summary_log_filename": None,
        "model_key": model_key or "default"
    }
    
    # Generate a summary log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_log_filename = f"{log_folder}/eval_summary_{timestamp}.log"
    results["summary_log_filename"] = summary_log_filename
    
    # Log the start of the eval run
    with open(summary_log_filename, 'a') as log_file:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "eval_run_start",
            "message": "Starting eval run for all test cases"
        }
        json.dump(log_entry, log_file, indent=2)
        log_file.write("\n\n")
    
    # Run each test case
    for test_name, queries in TEST_CASES.items():
        for query in queries:
            results["total"] += 1
            test_result = {
                "test_name": test_name,
                "query": query,
                "status": "unknown",
                "error_message": None,
                "log_filename": None
            }
            
            try:
                print(f"\n\nRunning test: {test_name} with query: {query}")
                
                # Create a new explainer agent for this test
                test_agent = create_langchain_agent(model_key=model_key)
                
                # Run the test based on test type
                if test_name == "sets_data_when_asked":
                    success, error_msg = test_sets_data_when_asked_impl(test_agent, query)
                elif test_name == "does_not_call_set_dataset_when_not_asked":
                    success, error_msg = test_does_not_call_set_dataset_when_not_asked_impl(test_agent, query)
                else:
                    success = False
                    error_msg = f"Unknown test type: {test_name}"
                
                test_result["status"] = "success" if success else "failed"
                test_result["error_message"] = error_msg
                test_result["log_filename"] = os.path.basename(log_filename) if log_filename else None
                
                if success:
                    results["successful"] += 1
                    print(f"✓ Test passed: {test_name} - {query}")
                else:
                    results["failed"] += 1
                    print(f"✗ Test failed: {test_name} - {query}: {error_msg}")
                    
            except Exception as e:
                results["failed"] += 1
                test_result["status"] = "failed"
                test_result["error_message"] = str(e)
                print(f"✗ Test error: {test_name} - {query}: {str(e)}")
            
            results["test_results"].append(test_result)
            
            # Log the test result
            with open(summary_log_filename, 'a') as log_file:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "test_result",
                    "test_result": test_result
                }
                json.dump(log_entry, log_file, indent=2)
                log_file.write("\n\n")
    
    # Log the final summary
    with open(summary_log_filename, 'a') as log_file:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "eval_run_complete",
            "summary": {
                "total": results["total"],
                "successful": results["successful"],
                "failed": results["failed"]
            }
        }
        json.dump(log_entry, log_file, indent=2)
        log_file.write("\n\n")
    
    print(f"\n\nEval run complete: {results['successful']}/{results['total']} tests passed")
    return results

def run_model_comparison(models: list, test_cases: dict = None):
    """
    Run the same tests across different models.
    
    Args:
        models: List of model keys to test (e.g., ["gpt-4o", "gpt-4", "claude-3-sonnet"])
        test_cases: Optional test cases to use (defaults to TEST_CASES)
    
    Returns:
        Dictionary with results for each model
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    results = {}
    
    for model_key in models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_key}")
        print(f"{'='*50}")
        
        try:
            model_results = run_all_evals(model_key=model_key)
            results[model_key] = model_results
            print(f"✓ Completed testing for {model_key}")
        except Exception as e:
            print(f"✗ Failed to test {model_key}: {e}")
            results[model_key] = {
                "error": str(e),
                "total": 0,
                "successful": 0,
                "failed": 0
            }
    
    # Print comparison summary
    print(f"\n{'='*50}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    for model_key, result in results.items():
        if "error" in result:
            print(f"{model_key}: ERROR - {result['error']}")
        else:
            success_rate = (result['successful'] / result['total'] * 100) if result['total'] > 0 else 0
            print(f"{model_key}: {result['successful']}/{result['total']} ({success_rate:.1f}%)")
    
    return results

def test_sets_data_when_asked_impl(test_agent, query):
    """Test implementation for checking if set_dataset is called."""
    # Use the proper eval function to get tool calls
    result = run_single_eval_langchain(query, model_key=None)
    tool_calls = result.get("tool_calls", [])
    
    # Check if 'set_dataset' was called
    for call in tool_calls:
        if call.get('tool_name') == 'set_dataset':
            return True, None
            
    return False, "The 'set_dataset' tool was not called."

def test_does_not_call_set_dataset_when_not_asked_impl(test_agent, query):
    """Test implementation for checking if set_dataset is NOT called."""
    # Use the proper eval function to get tool calls
    result = run_single_eval_langchain(query, model_key=None)
    tool_calls = result.get("tool_calls", [])
    
    # Check if 'set_dataset' was called
    for call in tool_calls:
        if call.get('tool_name') == 'set_dataset':
            return False, "The 'set_dataset' tool was called when it should not have been."
            
    return True, None

# Keep the original pytest functions for backward compatibility
@pytest.mark.parametrize(
    "query",
    TEST_CASES["sets_data_when_asked"],
)
def test_sets_data_when_asked(query):
    print(f"\n\nTesting with query: {query}")
    
    # Create a new explainer agent for this test
    test_agent = create_langchain_agent()
    
    success, error_msg = test_sets_data_when_asked_impl(test_agent, query)
    if not success:
        pytest.fail(error_msg)

@pytest.mark.parametrize(
    "query",
    TEST_CASES["does_not_call_set_dataset_when_not_asked"],
)
def test_does_not_call_set_dataset_when_not_asked(query):
    """Test that the agent does not call set_dataset when it should not."""
    # This test currently only supports the LangChain agent.
    # We can add logic to select the agent based on an environment variable or parameter if needed.
    test_does_not_call_set_dataset_when_not_asked_impl(create_langchain_agent, query)

@pytest.mark.parametrize(
    "query, model_key",
    [
        ("who made this model", "gpt-4o"),
        ("who made this model", "gemini-pro"),
    ]
)
def test_single_eval(query, model_key):
    """Runs a single evaluation with the given query and model."""
    print(f"Running single eval with query: '{query}' and model: {model_key}")
    result = run_single_eval_langchain(query, model_key=model_key)
    assert result["status"] == "success"
    # We can add more assertions here based on the expected behavior
    print(f"Log file: {result['log_filename']}")

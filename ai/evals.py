import os
import json
from datetime import datetime
from agents.explainer_agent import create_explainer_agent
import pytest
from dotenv import load_dotenv
from swarm import Swarm

load_dotenv()

# Create explainer agent instance
agent = create_explainer_agent()
client = Swarm()

# Create logs subfolder if it doesn't exist
log_folder = 'logs/evals'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

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
            response = client.run(
                agent=agent.agent,  # Use the agent property of the ExplainerAgent
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

def run_all_evals():
    """Run all eval test cases and return success/failure counts with details."""
    results = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "test_results": [],
        "timestamp": datetime.now().isoformat(),
        "summary_log_filename": None
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
                test_agent = create_explainer_agent()
                
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

def test_sets_data_when_asked_impl(test_agent, query):
    """Implementation of the set_dataset test that returns success/failure instead of asserting."""
    try:
        # Run the agent with the query
        run_and_get_tool_calls(test_agent, query)
        
        # Check the log file for tool calls
        with open(log_filename, 'r') as log_file:
            log_content = log_file.read()
            
        # Check if set_dataset was called
        if "set_dataset" not in log_content:
            return False, f"Expected to find 'set_dataset' in log for query: {query}"
        
        # Check if the set_dataset call was successful (handle escaped JSON)
        success_patterns = ['"status": "success"', '\\"status\\": \\"success\\"']
        if not any(pattern in log_content for pattern in success_patterns):
            return False, f"Expected to find successful set_dataset response in log for query: {query}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def test_does_not_call_set_dataset_when_not_asked_impl(test_agent, query):
    """Implementation of the no set_dataset test that returns success/failure instead of asserting."""
    try:
        tool_calls = run_and_get_tool_calls(test_agent, query)
        
        # Check that set_dataset specifically was not called, even though other tools may be called
        set_dataset_calls = [call for call in tool_calls if call.get('function', {}).get('name') == 'set_dataset']
        if len(set_dataset_calls) > 0:
            return False, f"Expected no set_dataset calls, but found {len(set_dataset_calls)} for query: {query}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)

# Keep the original pytest functions for backward compatibility
@pytest.mark.parametrize(
    "query",
    TEST_CASES["sets_data_when_asked"],
)
def test_sets_data_when_asked(query):
    print(f"\n\nTesting with query: {query}")
    
    # Create a new explainer agent for this test
    test_agent = create_explainer_agent()
    
    success, error_msg = test_sets_data_when_asked_impl(test_agent, query)
    if not success:
        pytest.fail(error_msg)

@pytest.mark.parametrize(
    "query",
    TEST_CASES["does_not_call_set_dataset_when_not_asked"],
)
def test_does_not_call_set_dataset_when_not_asked(query):
    print(f"\n\nTesting with query: {query}")
    
    # Create a new explainer agent for this test
    test_agent = create_explainer_agent()
    
    success, error_msg = test_does_not_call_set_dataset_when_not_asked_impl(test_agent, query)
    if not success:
        pytest.fail(error_msg)

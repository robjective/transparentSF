#!/usr/bin/env python3
"""
Evaluation Runner for TransparentSF
Executes evals from the database and saves results.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .eval_manager import get_eval_by_id, save_eval_result
from .explainer_metrics_tools import (
    query_metrics, get_metric_details, get_dashboard_metrics,
    create_new_metric, edit_metric, disable_metric, enable_metric,
    get_metrics_overview, find_metrics_by_endpoint
)

# Import the real LangChain agent
try:
    from agents.langchain_agent.explainer_agent import create_explainer_agent
    from agents.langchain_agent.config.tool_config import ToolGroup
    AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import LangChain explainer agent: {e}")
    AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class EvalRunner:
    """Runner for executing evals and evaluating results."""
    
    def __init__(self):
        self.available_tools = {
            'query_metrics': query_metrics,
            'get_metric_details': get_metric_details,
            'get_dashboard_metrics': get_dashboard_metrics,
            'create_new_metric': create_new_metric,
            'edit_metric': edit_metric,
            'disable_metric': disable_metric,
            'enable_metric': enable_metric,
            'get_metrics_overview': get_metrics_overview,
            'find_metrics_by_endpoint': find_metrics_by_endpoint
        }
    
    def run_eval(self, eval_id: int, model_name: str = "test-model") -> Dict[str, Any]:
        """
        Run a specific eval and save the result.
        
        Args:
            eval_id: The eval ID to run
            model_name: Name of the model being tested
            
        Returns:
            dict: Result with status and evaluation details
        """
        start_time = time.time()
        
        try:
            # Get the eval details
            eval_result = get_eval_by_id(eval_id)
            if eval_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Failed to get eval: {eval_result['message']}"
                }
            
            eval_data = eval_result["eval"]
            prompt = eval_data["prompt"]
            
            # Execute the eval
            execution_result = self._execute_eval(eval_data, model_name)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save the result
            save_result = save_eval_result(
                eval_id=eval_id,
                model_name=model_name,
                prompt_used=prompt,
                response_received=execution_result.get("response"),
                tool_calls_made=execution_result.get("tool_calls"),
                success_score=execution_result.get("success_score"),
                success_details=execution_result.get("success_details"),
                execution_time_seconds=execution_time,
                error_message=execution_result.get("error"),
                status=execution_result.get("status", "completed"),
                conversation_history=execution_result.get("conversation_history")
            )
            
            if save_result["status"] != "success":
                logger.error(f"Failed to save eval result: {save_result['message']}")
            
            return {
                "status": "success",
                "eval_id": eval_id,
                "eval_name": eval_data["name"],
                "execution_result": execution_result,
                "execution_time": execution_time,
                "result_id": save_result.get("result_id")
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error running eval {eval_id}: {error_msg}")
            
            # Try to save the error result
            try:
                save_eval_result(
                    eval_id=eval_id,
                    model_name=model_name,
                    prompt_used="",  # We don't have the prompt in this error case
                    error_message=error_msg,
                    execution_time_seconds=execution_time,
                    status="failed"
                )
            except Exception as save_error:
                logger.error(f"Failed to save error result: {save_error}")
            
            return {
                "status": "error",
                "message": error_msg,
                "execution_time": execution_time
            }
    
    def _execute_eval(self, eval_data: Dict[str, Any], model_name: str = "gpt-4o") -> Dict[str, Any]:
        """
        Execute a single eval using the real AI agent and evaluate the results.
        
        Args:
            eval_data: The eval data from the database
            model_name: The model to use for execution
            
        Returns:
            dict: Execution result with success score and details
        """
        prompt = eval_data["prompt"]
        success_type = eval_data["success_type"]
        expected_tool_calls = eval_data.get("expected_tool_calls", [])
        
        try:
            if not AGENT_AVAILABLE:
                return {
                    "status": "failed",
                    "error": "Explainer agent not available",
                    "response": "",
                    "tool_calls": [],
                    "success_score": 0.0,
                    "success_details": {"error": "Agent not available"}
                }
            
            # Create a fresh LangChain agent instance for this eval with core, analysis, and metrics tools
            agent = create_explainer_agent(
                model_key=model_name,
                tool_groups=[ToolGroup.CORE, ToolGroup.METRICS]
            )
            
            # Execute the prompt with the real LangChain agent
            logger.info(f"Executing eval with prompt: {prompt}")
            result = agent.explain_change_sync(prompt, metric_details={})
            
            if not result.get('success', False):
                error_message = result.get('error', 'Unknown error')
                error_type = result.get('error_type', 'unknown')
                
                # Handle specific error types
                if error_type == 'overload':
                    return {
                        "status": "failed",
                        "error": error_message,
                        "response": result.get('explanation', ''),
                        "tool_calls": [],
                        "success_score": 0.0,
                        "success_details": {
                            "error": error_message,
                            "error_type": "overload",
                            "suggestion": result.get('suggestion', 'Try breaking your request into smaller parts.')
                        }
                    }
                elif error_type == 'rate_limit':
                    return {
                        "status": "failed",
                        "error": error_message,
                        "response": result.get('explanation', ''),
                        "tool_calls": [],
                        "success_score": 0.0,
                        "success_details": {
                            "error": error_message,
                            "error_type": "rate_limit"
                        }
                    }
                else:
                    return {
                        "status": "failed",
                        "error": error_message,
                        "response": result.get('explanation', ''),
                        "tool_calls": [],
                        "success_score": 0.0,
                        "success_details": {"error": error_message}
                    }
            
            # Extract the response content and ensure it's a string
            response = result.get('explanation', '')
            if isinstance(response, list):
                # If response is a list, join the elements
                response = ' '.join([str(item) for item in response])
            elif not isinstance(response, str):
                response = str(response)
            
            # Get conversation history to extract tool calls
            conversation_history = agent.messages
            
            # Extract execution trace from the agent result
            execution_trace = result.get("execution_trace", [])
            logger.info(f"Execution trace has {len(execution_trace)} steps")
            
            # Log all execution trace steps for debugging
            for i, step in enumerate(execution_trace):
                logger.info(f"Execution trace step {i}: type={step.get('type')}, tool={step.get('tool', 'N/A')}, successful={step.get('successful', 'N/A')}")
            
            # Extract tool calls from execution trace (including failed ones)
            # This is the authoritative source since it includes success/failure information
            all_tool_calls = self._extract_tool_calls_from_execution_trace(execution_trace)
            logger.info(f"Extracted {len(all_tool_calls)} tool calls from execution trace")
            
            # If no tool calls found in execution trace, fall back to conversation history
            if not all_tool_calls:
                logger.warning("No tool calls found in execution trace, falling back to conversation history")
                tool_calls_from_conversation = self._extract_tool_calls_from_conversation(conversation_history)
                all_tool_calls = tool_calls_from_conversation
                logger.info(f"Extracted {len(all_tool_calls)} tool calls from conversation history")
            
            # Convert LangChain messages to UI-friendly format
            ui_conversation_history = []
            for message in conversation_history:
                if hasattr(message, 'content'):
                    content = message.content
                    # Handle Anthropic's list format
                    if isinstance(content, list):
                        # Extract text content from the list
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    text_parts.append(item.get('text', ''))
                                elif item.get('type') == 'tool_use':
                                    text_parts.append(f"[Tool call: {item.get('name', 'unknown')}]")
                            else:
                                text_parts.append(str(item))
                        content_str = ' '.join(text_parts)
                    else:
                        content_str = str(content)
                    
                    ui_conversation_history.append({
                        "role": "user" if hasattr(message, 'type') and message.type == 'human' else "assistant",
                        "content": content_str,
                        "sender": "Explainer" if hasattr(message, 'type') and message.type == 'ai' else None
                    })
            
            # Convert execution trace to UI-friendly format
            ui_execution_trace = []
            for step in execution_trace:
                if step.get("type") == "tool_start":
                    # Determine if the tool call was successful
                    tool_output = step.get("output", "")
                    is_successful = True
                    if isinstance(tool_output, dict):
                        if "error" in tool_output:
                            is_successful = False
                        elif "status" in tool_output and tool_output["status"] != "success":
                            is_successful = False
                    elif isinstance(tool_output, str):
                        tool_output_lower = tool_output.lower()
                        error_indicators = [
                            "error", "exception", "failed", "failure", "invalid", 
                            "unexpected character", "syntax error", "parse error",
                            "no data returned", "empty result", "timeout",
                            "no-such-column", "query coordinator error", "soql error",
                            "column not found", "unknown column", "http error",
                            "400 client error", "bad request"
                        ]
                        if any(indicator in tool_output_lower for indicator in error_indicators):
                            is_successful = False
                        
                        # Also check if the string might be a JSON representation of an error dict
                        if tool_output.strip().startswith('{') and tool_output.strip().endswith('}'):
                            try:
                                import json
                                parsed_output = json.loads(tool_output)
                                if isinstance(parsed_output, dict) and "error" in parsed_output:
                                    is_successful = False
                            except json.JSONDecodeError:
                                pass  # Not valid JSON, continue with string analysis
                    
                    # Check for error patterns in the string representation
                    if "'error'" in tool_output or '"error"' in tool_output:
                        is_successful = False
                        logger.info(f"Tool call failed - string contains error key: {tool_output}")
                
                    # Additional check: if the tool output contains any error-like content, mark as failed
                    tool_output_str = str(tool_output)
                    if any(error_term in tool_output_str.lower() for error_term in [
                        "query coordinator error", "no-such-column", "400 client error", 
                        "bad request", "http error", "no such column"
                    ]):
                        is_successful = False
                        logger.info(f"Tool call failed - contains error content: {tool_output_str[:200]}...")
                
                    ui_execution_trace.append({
                        "type": "tool_call",
                        "tool": step.get("tool", "unknown"),
                        "input": step.get("input", ""),
                        "output": step.get("output", ""),
                        "duration": step.get("duration", 0),
                        "successful": is_successful
                    })
                elif step.get("type") == "chain_start":
                    ui_execution_trace.append({
                        "type": "chain_step",
                        "chain": step.get("chain", "unknown"),
                        "inputs": step.get("inputs", {}),
                        "outputs": step.get("outputs", {}),
                        "duration": step.get("duration", 0)
                    })
            
            # Evaluate success based on the actual response and tool calls
            success_evaluation = self.evaluate_success(eval_data, all_tool_calls, response)
            
            return {
                "status": "completed",
                "response": response,
                "tool_calls": all_tool_calls,
                "success_score": success_evaluation.get("success_score", 0.0),
                "success_details": success_evaluation.get("success_details", {}),
                "conversation_history": ui_conversation_history,
                "execution_trace": ui_execution_trace
            }
            
        except Exception as e:
            logger.error(f"Error executing eval: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "response": "",
                "tool_calls": [],
                "success_score": 0.0,
                "success_details": {"error": str(e)}
            }
    
    def _extract_tool_calls_from_conversation(self, conversation_history: List) -> List[Dict]:
        """
        Extract tool calls from the LangChain conversation history.
        
        Args:
            conversation_history: List of LangChain message objects
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        for message in conversation_history:
            # Check if this is an AIMessage with tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "function": tool_call.get('name', ''),
                        "arguments": tool_call.get('args', {}),
                        "id": tool_call.get('id', '')
                    })
            # Check for tool use in content (Anthropic format)
            elif hasattr(message, 'content') and message.content:
                content = message.content
                if isinstance(content, list):
                    # Handle Anthropic's list format
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            tool_calls.append({
                                "function": item.get('name', ''),
                                "arguments": item.get('input', {}),
                                "id": item.get('id', ''),
                                "source": "anthropic_tool_use"
                            })
                else:
                    # Fallback: check content string for function names
                    content_str = str(content)
                    for tool_name in self.available_tools.keys():
                        if tool_name in content_str:
                            tool_calls.append({
                                "function": tool_name,
                                "arguments": {},
                                "detected_in_content": True
                            })
        
        return tool_calls
    
    def _extract_tool_calls_from_execution_trace(self, execution_trace: List[Dict]) -> List[Dict]:
        """
        Extract tool calls from the execution trace (including failed ones).
        
        Args:
            execution_trace: List of execution trace steps
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        for step in execution_trace:
            if step.get("type") == "tool_start":
                # Extract tool call information from execution trace
                tool_name = step.get("tool", "unknown")
                tool_input = step.get("input", {})
                tool_output = step.get("output", "")
                
                # Determine if the tool call was successful
                # The ExecutionTraceCallback should have already determined this correctly
                # We'll trust the successful flag from the execution trace
                is_successful = step.get("successful", True)
                
                # Log for debugging
                if not is_successful:
                    logger.info(f"Tool call marked as failed by execution trace: {tool_name}")
                    logger.info(f"Tool output: {tool_output}")
                else:
                    logger.info(f"Tool call marked as successful by execution trace: {tool_name}")
                
                # Log tool call details for debugging
                logger.info(f"Tool call: {tool_name}, successful: {is_successful}, output type: {type(tool_output)}")
                if not is_successful:
                    logger.info(f"Failed tool output: {tool_output}")
                else:
                    # Also log successful outputs to see what we're getting
                    logger.info(f"Successful tool output (first 200 chars): {str(tool_output)[:200]}...")
                
                # Additional debugging for all tool calls
                logger.info(f"Raw tool output: {repr(tool_output)}")
                logger.info(f"Tool output type: {type(tool_output)}")
                logger.info(f"Tool output length: {len(str(tool_output))}")
                
                if isinstance(tool_output, dict):
                    logger.info(f"Tool output keys: {list(tool_output.keys())}")
                    if "error" in tool_output:
                        logger.info(f"ERROR DETECTED in dict: {tool_output['error']}")
                elif isinstance(tool_output, str):
                    logger.info(f"Tool output string length: {len(tool_output)}")
                    if "error" in tool_output.lower():
                        logger.info(f"ERROR DETECTED in string: {tool_output}")
                    
                    # Check if it looks like a string representation of a dict
                    if tool_output.strip().startswith('{') and tool_output.strip().endswith('}'):
                        logger.info("Tool output appears to be a string representation of a dict")
                        try:
                            import json
                            parsed = json.loads(tool_output)
                            logger.info(f"Parsed JSON keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
                        except json.JSONDecodeError:
                            logger.info("Not valid JSON")
                
                # Check for any error-like content in the output
                tool_output_str = str(tool_output)
                error_terms = ["error", "exception", "failed", "no-such-column", "query coordinator error"]
                found_errors = [term for term in error_terms if term in tool_output_str.lower()]
                if found_errors:
                    logger.info(f"Found error terms in output: {found_errors}")
                
                tool_calls.append({
                    "function": tool_name,
                    "arguments": tool_input,
                    "output": tool_output,
                    "successful": is_successful,
                    "source": "execution_trace"
                })
            elif step.get("type") == "agent_action":
                # Extract tool call information from agent action
                tool_name = step.get("tool", "unknown")
                tool_input = step.get("input", {})
                
                # For agent_action, we don't have the output, so we'll assume successful
                # The actual success/failure will be determined by the response content
                tool_calls.append({
                    "function": tool_name,
                    "arguments": tool_input,
                    "output": "",  # No output available in agent_action
                    "successful": True,  # Assume successful, will be refined later
                    "source": "agent_action"
                })
        
        return tool_calls
    
    def run_eval_group(self, group_id: int, model_name: str = "test-model") -> Dict[str, Any]:
        """
        Run all evals in a group.
        
        Args:
            group_id: The group ID to run
            model_name: Name of the model being tested
            
        Returns:
            dict: Results for all evals in the group
        """
        from .eval_manager import get_evals
        
        # Get all evals in the group
        evals_result = get_evals(group_id=group_id)
        if evals_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Failed to get evals for group {group_id}: {evals_result['message']}"
            }
        
        evals = evals_result["evals"]
        results = []
        
        for eval_data in evals:
            result = self.run_eval(eval_data["id"], model_name)
            results.append({
                "eval_id": eval_data["id"],
                "eval_name": eval_data["name"],
                "result": result
            })
        
        # Calculate group summary
        successful_runs = sum(1 for r in results if r["result"]["status"] == "success")
        total_runs = len(results)
        avg_success_score = sum(
            r["result"].get("execution_result", {}).get("success_score", 0)
            for r in results if r["result"]["status"] == "success"
        ) / max(successful_runs, 1)
        
        return {
            "status": "success",
            "group_id": group_id,
            "total_evals": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": total_runs - successful_runs,
            "avg_success_score": avg_success_score,
            "results": results
        }
    
    def evaluate_success(self, eval_data: Dict[str, Any], tool_calls: List[Dict], response: str) -> Dict[str, Any]:
        """
        Evaluate the success of an eval based on its criteria.
        
        Args:
            eval_data: The eval data
            tool_calls: Tool calls that were made
            response: Response from the model
            
        Returns:
            dict: Success evaluation with score and details
        """
        success_type = eval_data["success_type"]
        expected_tool_calls = eval_data.get("expected_tool_calls", [])
        
        # Debug logging
        logger.info(f"Evaluating success for {len(tool_calls)} tool calls")
        for i, tc in enumerate(tool_calls):
            output = tc.get('output', '')
            # Safely convert output to string for logging
            if isinstance(output, dict):
                output_str = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
            elif isinstance(output, str):
                output_str = output[:200] + "..." if len(output) > 200 else output
            else:
                output_str = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
            
            logger.info(f"Tool call {i}: function={tc.get('function')}, successful={tc.get('successful')}, output={output_str}")
        
        # Ensure response is a string for evaluation
        if not isinstance(response, str):
            response = str(response)
        
        # Check if any tool calls failed
        failed_tool_calls = [tc for tc in tool_calls if tc.get("successful") == False]
        successful_tool_calls = [tc for tc in tool_calls if tc.get("successful") == True]
        has_failed_tools = len(failed_tool_calls) > 0
        has_successful_tools = len(successful_tool_calls) > 0
        
        if success_type == "binary":
            # Simple pass/fail evaluation - now considers tool call success
            success = has_successful_tools and response.strip() != "" and not has_failed_tools
            return {
                "success_score": 1.0 if success else 0.0,
                "success_details": {
                    "binary_success": success,
                    "tool_calls_made": len(tool_calls),
                    "successful_tool_calls": len(successful_tool_calls),
                    "failed_tool_calls": len(failed_tool_calls),
                    "has_response": bool(response.strip()),
                    "failed_tool_details": [{"function": tc.get("function"), "error": str(tc.get("output", ""))} for tc in failed_tool_calls]
                }
            }
        
        elif success_type == "scored":
            # Scored evaluation based on multiple criteria
            score = 0.0
            details = {}
            
            # Check if expected tool calls were made successfully
            expected_functions = [tc.get("function") for tc in expected_tool_calls]
            made_successful_functions = [tc.get("function") for tc in successful_tool_calls]
            
            function_score = sum(1 for func in expected_functions if func in made_successful_functions) / max(len(expected_functions), 1)
            score += function_score * 0.6  # 60% weight for successful tool calls
            
            # Penalize for failed tool calls
            if has_failed_tools:
                score *= 0.5  # 50% penalty for any failed tool calls
            
            # Check response quality
            response_score = 0.5 if response.strip() else 0.0
            score += response_score * 0.4  # 40% weight for response
            
            return {
                "success_score": score,
                "success_details": {
                    "function_score": function_score,
                    "response_score": response_score,
                    "expected_functions": expected_functions,
                    "made_successful_functions": made_successful_functions,
                    "failed_functions": [tc.get("function") for tc in failed_tool_calls],
                    "tool_calls_made": len(tool_calls),
                    "successful_tool_calls": len(successful_tool_calls),
                    "failed_tool_calls": len(failed_tool_calls),
                    "failed_tool_details": [{"function": tc.get("function"), "error": str(tc.get("output", ""))} for tc in failed_tool_calls]
                }
            }
        
        elif success_type == "multi_step":
            # Multi-step evaluation
            steps = eval_data["success_criteria"].split('\n')
            completed_steps = 0
            
            for step in steps:
                step = step.strip()
                if not step or step.startswith('#'):
                    continue
                
                # Simple keyword-based step completion checking
                if "follow directions" in step.lower():
                    if response.strip():
                        completed_steps += 1
                elif "find endpoint" in step.lower():
                    if any("endpoint" in str(tc) for tc in successful_tool_calls):
                        completed_steps += 1
                elif "test queries" in step.lower():
                    if any("query" in str(tc) for tc in successful_tool_calls):
                        completed_steps += 1
                elif "set metric" in step.lower():
                    if any("create" in str(tc) or "edit" in str(tc) for tc in successful_tool_calls):
                        completed_steps += 1
                else:
                    # Generic step completion - only count if tools were successful
                    if has_successful_tools and response.strip():
                        completed_steps += 1
            
            total_steps = len([s for s in steps if s.strip() and not s.startswith('#')])
            step_score = completed_steps / max(total_steps, 1)
            
            # Penalize for failed tool calls
            if has_failed_tools:
                step_score *= 0.5  # 50% penalty for any failed tool calls
            
            return {
                "success_score": step_score,
                "success_details": {
                    "completed_steps": completed_steps,
                    "total_steps": total_steps,
                    "step_score": step_score,
                    "tool_calls_made": len(tool_calls),
                    "successful_tool_calls": len(successful_tool_calls),
                    "failed_tool_calls": len(failed_tool_calls),
                    "failed_tool_details": [{"function": tc.get("function"), "error": str(tc.get("output", ""))} for tc in failed_tool_calls]
                }
            }
        
        else:
            # Default evaluation - now considers tool call success
            base_score = 0.5
            if has_failed_tools:
                base_score *= 0.5  # 50% penalty for any failed tool calls
            
            return {
                "success_score": base_score,
                "success_details": {
                    "unknown_success_type": success_type,
                    "tool_calls_made": len(tool_calls),
                    "successful_tool_calls": len(successful_tool_calls),
                    "failed_tool_calls": len(failed_tool_calls),
                    "failed_tool_details": [{"function": tc.get("function"), "error": str(tc.get("output", ""))} for tc in failed_tool_calls]
                }
            } 
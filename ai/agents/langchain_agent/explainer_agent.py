"""
LangChain-based Explainer Agent for TransparentSF.
This version uses a modular system with selective tool inclusion.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import json
import logging
import time
import asyncio
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json


# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

# Import LangChain components
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler

# Import our model configuration
from ..config.models import create_langchain_llm, get_default_model, get_model_config, ModelProvider

# Import modular components
from .config.tool_config import ToolGroup, tool_config
from .tools.tool_factory import tool_factory
from .prompts.modular_prompts import prompt_builder

@dataclass_json
@dataclass
class ToolCall:
    """Represents a single tool call with detailed information."""
    tool_name: str
    arguments: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    success: Optional[bool] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    
    def finalize(self, result: Any, success: bool, error_message: Optional[str] = None):
        """Finalize the tool call with results."""
        self.end_time = time.time()
        self.result = result
        self.success = success
        self.error_message = error_message
        self.execution_time_ms = int((self.end_time - self.start_time) * 1000)

@dataclass_json
@dataclass
class ConversationMessage:
    """Represents a conversation message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    sender: Optional[str] = None

@dataclass_json
@dataclass
class AgentSession:
    """Represents a complete agent session with all details."""
    session_id: str
    timestamp: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    agent_type: str = "langchain_explainer"
    model: str = ""
    user_input: str = ""
    conversation: List[ConversationMessage] = None
    tool_calls: List[ToolCall] = None
    intermediate_responses: List[Dict[str, Any]] = None
    final_response: str = ""
    total_execution_time_ms: int = 0
    success: bool = True
    error_summary: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    tool_groups: Optional[List[str]] = None
    available_tools: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.conversation is None:
            self.conversation = []
        if self.tool_calls is None:
            self.tool_calls = []
        if self.intermediate_responses is None:
            self.intermediate_responses = []
        if self.model_config is None:
            self.model_config = {}
        if self.tool_groups is None:
            self.tool_groups = []
        if self.available_tools is None:
            self.available_tools = []

class SessionLogger:
    """Handles detailed session logging for the LangChain explainer agent."""
    
    def __init__(self, logs_dir: Optional[Path] = None):
        """Initialize the session logger."""
        if logs_dir is None:
            # Default to ai/logs/sessions
            current_dir = Path(__file__).parent
            ai_dir = current_dir.parent.parent
            logs_dir = ai_dir / 'logs' / 'sessions'
        
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"SessionLogger initialized with logs directory: {self.logs_dir}")
    
    def create_session(self, 
                      model: str, 
                      model_config: Dict[str, Any],
                      tool_groups: List[ToolGroup],
                      available_tools: List[str],
                      session_id: Optional[str] = None) -> AgentSession:
        """Create a new session for logging."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session = AgentSession(
            session_id=session_id,
            timestamp=timestamp,
            start_time=timestamp,  # Set start_time to the same as timestamp
            model=model,
            model_config=model_config,
            tool_groups=[g.value for g in tool_groups],
            available_tools=available_tools
        )
        
        self.logger.info(f"Created new session: {session_id}")
        return session
    
    def log_session(self, session: AgentSession):
        """Log the complete session to a JSON file."""
        try:
            # Use session ID directly as filename for simple lookup
            filename = f"{session.session_id}.json"
            filepath = self.logs_dir / filename
            
            # Convert session to dict and handle non-serializable objects
            session_dict = self._prepare_session_for_json(session)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Session logged to: {filepath}")
            
            # Also log a summary to the main log
            self._log_session_summary(session)
            
        except Exception as e:
            self.logger.error(f"Error logging session: {e}")
    
    def _prepare_session_for_json(self, session: AgentSession) -> Dict[str, Any]:
        """Prepare session data for JSON serialization."""
        session_dict = asdict(session)
        
        # Manually add intermediate_responses if it exists
        if hasattr(session, 'intermediate_responses') and session.intermediate_responses:
            session_dict['intermediate_responses'] = session.intermediate_responses
        
        # Handle non-serializable objects in tool calls
        for tool_call in session_dict['tool_calls']:
            if 'result' in tool_call and tool_call['result'] is not None:
                tool_call['result'] = self._make_json_serializable(tool_call['result'])
            if 'arguments' in tool_call and tool_call['arguments'] is not None:
                tool_call['arguments'] = self._make_json_serializable(tool_call['arguments'])
        
        return session_dict
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
    
    def _log_session_summary(self, session: AgentSession):
        """Log a summary of the session to the main log."""
        summary = {
            "session_id": session.session_id,
            "model": session.model,
            "user_input": session.user_input[:100] + "..." if len(session.user_input) > 100 else session.user_input,
            "tool_calls_count": len(session.tool_calls),
            "successful_tool_calls": len([tc for tc in session.tool_calls if tc.success]),
            "failed_tool_calls": len([tc for tc in session.tool_calls if not tc.success]),
            "total_execution_time_ms": session.total_execution_time_ms,
            "success": session.success,
            "error_summary": session.error_summary
        }
        
        self.logger.info(f"Session Summary: {json.dumps(summary, indent=2)}")

class EnhancedExecutionTraceCallback(BaseCallbackHandler):
    """Enhanced callback to capture detailed execution trace for session logging."""
    
    def __init__(self, session: AgentSession):
        self.session = session
        self.current_tool_call: Optional[ToolCall] = None
        self.logger = logging.getLogger(__name__)
        self.session_start_time = time.time()
        # Add execution_trace for compatibility with the streaming code
        self.execution_trace = []
        # Track intermediate responses
        self.intermediate_responses = []
        self.current_response_content = ""
        
    def on_llm_new_token(self, token: Union[str, List[str], List[Dict]], **kwargs) -> None:
        """Called when a new token is generated by the LLM."""
        try:
            # Handle different token types (some models pass token lists or dicts)
            if isinstance(token, list):
                # If it's a list, extract the content from each item
                if token and isinstance(token[0], dict):
                    # List of dictionaries - extract 'content' or 'text' field
                    token_parts = []
                    for item in token:
                        if isinstance(item, dict):
                            content = item.get('content') or item.get('text') or str(item)
                            token_parts.append(content)
                        else:
                            token_parts.append(str(item))
                    token = ''.join(token_parts)
                else:
                    # List of strings - join them
                    token = ''.join(str(t) for t in token)
            
            # Skip empty tokens
            if not token or not token.strip():
                return
            
            # Filter out object code and internal data patterns
            if (any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']) or
                token.strip().startswith('{') and token.strip().endswith('}') or
                'tool_use' in token.lower() or
                'type' in token.lower() and 'input' in token.lower() or
                'partial_json' in token.lower() or
                'index' in token.lower() and ('toolu_' in token or 'call_' in token) or
                any(pattern in token for pattern in ['toolu_', 'call_', 'input_json_delta', 'tool_use'])):
                # Skip this token entirely - don't add it to current_response_content
                return
                
            self.current_response_content += token
            
            # Be much less aggressive with segmentation
            # Only segment at major paragraph breaks (double newlines) or very long content
            should_segment = False
            
            # Segment at paragraph breaks
            if token.endswith('\n\n'):
                should_segment = True
            # Segment at very long content (500+ chars) ending with sentence punctuation
            elif (len(self.current_response_content) > 500 and 
                  (token.endswith('.') or token.endswith('?') or token.endswith('!'))):
                should_segment = True
            # Segment at markdown headers (new line starting with #)
            elif token.startswith('\n#') and len(self.current_response_content) > 50:
                should_segment = True
                
            if should_segment:
                # Only store if we have meaningful content (avoid tiny segments)
                if len(self.current_response_content.strip()) > 20:
                    self.intermediate_responses.append({
                        "content": self.current_response_content.strip(),
                        "timestamp": datetime.now().isoformat(),
                        "type": "intermediate_response"
                    })
                    self.current_response_content = ""
                    
        except Exception as e:
            self.logger.error(f"Error in on_llm_new_token: {e}")
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        try:
            # Store any remaining response content before tool call
            # Only store if we have meaningful content and it's not too short
            if self.current_response_content.strip() and len(self.current_response_content.strip()) > 10:
                self.intermediate_responses.append({
                    "content": self.current_response_content.strip(),
                    "timestamp": datetime.now().isoformat(),
                    "type": "intermediate_response"
                })
                self.current_response_content = ""
            
            tool_name = serialized.get("name", "unknown")
            
            # Parse arguments
            arguments = {}
            try:
                if input_str:
                    arguments = json.loads(input_str)
            except json.JSONDecodeError:
                arguments = {"raw_input": input_str}
            
            # Create tool call
            self.current_tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                start_time=time.time()
            )
            
            self.logger.info(f"Tool started: {tool_name} with args: {arguments}")
            
        except Exception as e:
            self.logger.error(f"Error in on_tool_start: {e}")
            self.current_tool_call = ToolCall(
                tool_name="unknown",
                arguments={"error": str(e)},
                start_time=time.time()
            )
        
    def on_tool_end(self, output, **kwargs) -> None:
        """Called when a tool finishes executing."""
        try:
            if self.current_tool_call:
                # Determine success and extract error message
                success, error_message, normalized_result = self._analyze_tool_output(output)
                
                # Finalize the tool call
                self.current_tool_call.finalize(
                    result=normalized_result,
                    success=success,
                    error_message=error_message
                )
                
                # Add to session
                self.session.tool_calls.append(self.current_tool_call)
                
                self.logger.info(f"Tool ended: {self.current_tool_call.tool_name}, "
                               f"success: {success}, duration: {self.current_tool_call.execution_time_ms}ms")
                
                self.current_tool_call = None
            else:
                self.logger.warning("on_tool_end called but no current_tool_call")
                
        except Exception as e:
            self.logger.error(f"Error in on_tool_end: {e}")
            if self.current_tool_call:
                self.current_tool_call.finalize(
                    result={"error": str(e)},
                    success=False,
                    error_message=str(e)
                )
                self.session.tool_calls.append(self.current_tool_call)
                self.current_tool_call = None
    
    def on_tool_error(self, error: str, **kwargs) -> None:
        """Called when a tool encounters an error."""
        try:
            if self.current_tool_call:
                self.current_tool_call.finalize(
                    result={"error": error},
                    success=False,
                    error_message=error
                )
                self.session.tool_calls.append(self.current_tool_call)
                self.logger.error(f"Tool error: {self.current_tool_call.tool_name} - {error}")
                self.current_tool_call = None
        except Exception as e:
            self.logger.error(f"Error in on_tool_error: {e}")
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when the agent finishes."""
        try:
            # Store any remaining response content
            if self.current_response_content.strip():
                self.intermediate_responses.append({
                    "content": self.current_response_content.strip(),
                    "timestamp": datetime.now().isoformat(),
                    "type": "intermediate_response"
                })
                self.current_response_content = ""
            
            # Calculate total execution time
            self.session.total_execution_time_ms = int((time.time() - self.session_start_time) * 1000)
            self.logger.info(f"Agent finished. Total execution time: {self.session.total_execution_time_ms}ms")
            
            # Store intermediate responses in session
            self.session.intermediate_responses = self.intermediate_responses
        except Exception as e:
            self.logger.error(f"Error in on_agent_finish: {e}")
    
    def _analyze_tool_output(self, output) -> tuple[bool, Optional[str], Any]:
        """Analyze tool output to determine success and extract error information."""
        success = True
        error_message = None
        normalized_result = output
        
        try:
            # Handle different output types
            if isinstance(output, dict):
                normalized_result = output
                if "error" in output:
                    success = False
                    error_message = str(output["error"])
                elif "status" in output and output["status"] != "success":
                    success = False
                    error_message = f"status={output['status']}"
                elif "error_type" in output:
                    success = False
                    error_message = str(output["error_type"])
            elif isinstance(output, str):
                # Check for error indicators in string
                output_lower = output.lower()
                error_indicators = [
                    "error", "exception", "failed", "failure", "invalid",
                    "no-such-column", "query coordinator error", "soql error",
                    "400 client error", "bad request", "could not parse",
                    "expected", "but got", "table identifier"
                ]
                if any(indicator in output_lower for indicator in error_indicators):
                    success = False
                    error_message = f"String contains error indicators: {output[:200]}"
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(output)
                    if isinstance(parsed, dict):
                        normalized_result = parsed
                        if "error" in parsed:
                            success = False
                            error_message = str(parsed["error"])
                except json.JSONDecodeError:
                    pass  # Not JSON, keep as string
                    
        except Exception as e:
            success = False
            error_message = f"Error analyzing output: {str(e)}"
            normalized_result = {"error": str(e), "raw_output": str(output)}
        
        return success, error_message, normalized_result

class ExecutionTraceCallback(BaseCallbackHandler):
    """Callback to capture detailed execution trace for UI display."""
    
    def __init__(self):
        self.execution_trace = []
        self.current_tool_start = None
        self.logger = logging.getLogger(__name__)
        self.tool_calls_in_progress = set()
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        try:
            self.logger.info(f"=== on_tool_start called ===")
            self.logger.info(f"Serialized: {serialized}")
            self.logger.info(f"Input: {input_str}")
            
            if not isinstance(serialized, dict):
                self.logger.warning(f"Serialized is not a dict: {serialized}")
                return
                
            tool_name = serialized.get("name", "unknown")
            self.current_tool_start = {
                "type": "tool_start",
                "tool": tool_name,
                "input": input_str,
                "start_time": time.time()
            }
            self.tool_calls_in_progress.add(tool_name)
            self.logger.info(f"Tool started: {tool_name}")
        except Exception as e:
            self.logger.error(f"Error in on_tool_start: {e}")
            # Create a minimal tool start entry
            self.current_tool_start = {
                "type": "tool_start",
                "tool": "unknown",
                "input": str(input_str) if input_str else "",
                "start_time": time.time()
            }
        
    def on_tool_end(self, output, **kwargs) -> None:
        """Called when a tool finishes executing."""
        try:
            self.logger.info(f"=== on_tool_end called ===")
            self.logger.info(f"Output type: {type(output)}")
            self.logger.info(f"Output: {output}")
            self.logger.info(f"Current tool start: {self.current_tool_start}")
            
            if self.current_tool_start:
                duration = time.time() - self.current_tool_start["start_time"]
                self.current_tool_start["duration"] = duration
                self.current_tool_start["output"] = output
                
                # Determine if the tool call was successful
                is_successful = True
                
                # Check if output is a dictionary with error
                if isinstance(output, dict):
                    self.logger.info(f"Processing dict output with keys: {list(output.keys())}")
                    if "error" in output:
                        is_successful = False
                        self.logger.info(f"Tool failed - dict contains error: {output['error']}")
                    elif "status" in output and output["status"] != "success":
                        is_successful = False
                        self.logger.info(f"Tool failed - status is not success: {output['status']}")
                    elif "error_type" in output:
                        is_successful = False
                        self.logger.info(f"Tool failed - dict contains error_type: {output['error_type']}")
                # Check if output is a string with error indicators
                elif isinstance(output, str):
                    self.logger.info(f"Processing string output: {output[:200]}...")
                    # Try to parse as JSON first (in case it's a stringified dict)
                    try:
                        import json
                        parsed_output = json.loads(output)
                        if isinstance(parsed_output, dict):
                            self.logger.info(f"Parsed JSON with keys: {list(parsed_output.keys())}")
                            if "error" in parsed_output:
                                is_successful = False
                                self.logger.info(f"Tool failed - string contains JSON with error: {parsed_output['error']}")
                            elif "status" in parsed_output and parsed_output["status"] != "success":
                                is_successful = False
                                self.logger.info(f"Tool failed - string contains JSON with non-success status: {parsed_output['status']}")
                            elif "error_type" in parsed_output:
                                is_successful = False
                                self.logger.info(f"Tool failed - string contains JSON with error_type: {parsed_output['error_type']}")
                            else:
                                # If it's valid JSON but no error indicators, assume success
                                self.logger.info("Valid JSON with no error indicators - assuming success")
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.info(f"Not valid JSON: {e}")
                        # Not valid JSON, check for error indicators in the string
                        output_lower = output.lower()
                        error_indicators = [
                            "error", "exception", "failed", "failure", "invalid",
                            "no-such-column", "query coordinator error", "soql error",
                            "400 client error", "bad request", "could not parse",
                            "expected", "but got", "table identifier"
                        ]
                        if any(indicator in output_lower for indicator in error_indicators):
                            is_successful = False
                            self.logger.info(f"Tool failed - string contains error: {output}")
                        else:
                            self.logger.info("No error indicators found in string")
                
                self.current_tool_start["successful"] = is_successful
                self.execution_trace.append(self.current_tool_start)
                self.logger.info(f"Tool ended: {self.current_tool_start['tool']}, successful: {is_successful}")
                
                # Remove from in-progress set
                if self.current_tool_start["tool"] in self.tool_calls_in_progress:
                    self.tool_calls_in_progress.remove(self.current_tool_start["tool"])
                
                self.current_tool_start = None
            else:
                self.logger.warning("on_tool_end called but no current_tool_start")
        except Exception as e:
            self.logger.error(f"Error in on_tool_end: {e}")
            # Try to clean up
            if self.current_tool_start:
                self.current_tool_start = None
            
    def on_tool_error(self, error: str, **kwargs) -> None:
        """Called when a tool encounters an error."""
        if self.current_tool_start:
            duration = time.time() - self.current_tool_start["start_time"]
            self.current_tool_start["duration"] = duration
            self.current_tool_start["output"] = f"Error: {error}"
            self.current_tool_start["successful"] = False
            self.execution_trace.append(self.current_tool_start)
            self.logger.info(f"Tool error: {self.current_tool_start['tool']}, error: {error}")
            self.current_tool_start = None
            
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts executing."""
        if serialized is None:
            serialized = {}
        if inputs is None:
            inputs = {}
            
        self.execution_trace.append({
            "type": "chain_start",
            "chain": serialized.get("name", "unknown"),
            "inputs": inputs,
            "start_time": time.time()
        })
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes executing."""
        if self.execution_trace and self.execution_trace[-1]["type"] == "chain_start":
            duration = time.time() - self.execution_trace[-1]["start_time"]
            self.execution_trace[-1]["duration"] = duration
            self.execution_trace[-1]["outputs"] = outputs
            self.execution_trace[-1]["type"] = "chain_step"
            
    def on_agent_action(self, action, **kwargs) -> None:
        """Called when the agent takes an action (tool call)."""
        self.logger.info(f"Agent action: {action}")
        if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
            self.execution_trace.append({
                "type": "agent_action",
                "tool": action.tool,
                "input": action.tool_input,
                "timestamp": time.time()
            })
            
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when the agent finishes."""
        self.logger.info(f"Agent finished: {finish}")

class LangChainExplainerAgent:
    """
    A LangChain-based explainer agent that uses a modular system with selective tool inclusion.
    """
    def __init__(self, 
                 model_key: Optional[str] = None,
                 tool_groups: Optional[List[ToolGroup]] = None,
                 include_all_sections: bool = False,
                 enable_session_logging: bool = True):
        """
        Initialize the LangChain explainer agent.
        
        Args:
            model_key: Model to use for the agent
            tool_groups: List of tool groups to include (default: CORE only)
            include_all_sections: Whether to include all prompt sections
            enable_session_logging: Whether to enable detailed session logging
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.model_key = model_key or get_default_model()
        self.logger.info(f"Model key: {self.model_key}")
        
        self.model_config = get_model_config(self.model_key)
        self.logger.info(f"Model config: {self.model_config}")
        
        self.logger.info(f"Creating LangChain LLM...")
        self.llm = create_langchain_llm(self.model_key)
        self.logger.info(f"Created LLM: {self.llm}")
        
        # Set up tool groups
        self.tool_groups = tool_groups or [ToolGroup.CORE]
        self.include_all_sections = include_all_sections
        
        # Create tools
        self.tools = self._create_tools()
        self.messages: List[BaseMessage] = []
        
        # Initialize session logging
        self.enable_session_logging = enable_session_logging
        if self.enable_session_logging:
            self.session_logger = SessionLogger()
            self.logger.info("Session logging enabled")
        else:
            self.session_logger = None
            self.logger.info("Session logging disabled")
        
        # Log configuration
        self.logger.info(f"LangChain Explainer agent initialized with:")
        self.logger.info(f"  Model: {self.model_key}")
        self.logger.info(f"  Tool groups: {[g.value for g in self.tool_groups]}")
        self.logger.info(f"  Tools: {[tool.name for tool in self.tools]}")
        self.logger.info(f"  Include all sections: {self.include_all_sections}")
        self.logger.info(f"  Session logging: {self.enable_session_logging}")

    def _make_json_serializable(self, obj):
        """Best-effort conversion to JSON-serializable objects."""
        try:
            import json
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def _parse_possible_dict_string(self, s: str):
        """Parse a string that may contain JSON or Python dict representation."""
        try:
            import json
            return json.loads(s)
        except Exception:
            try:
                import ast
                return ast.literal_eval(s)
            except Exception:
                return None

    from typing import Tuple
    def _determine_tool_success(self, output) -> Tuple[bool, Optional[str], object]:
        """
        Determine if a tool output indicates success.
        Returns (success, error_message, normalized_response)
        """
        error_message = None
        normalized = output
        success = True
        try:
            # If dict-like
            if isinstance(output, dict):
                normalized = output
                status_val = str(output.get('status', '')).lower()
                if 'error' in output:
                    success = False
                    error_message = str(output.get('error'))
                elif status_val and status_val != 'success':
                    success = False
                    error_message = f"status={output.get('status')}"
                elif 'error_type' in output:
                    success = False
                    error_message = str(output.get('error_type'))
            elif isinstance(output, str):
                # Try to parse JSON or python dict string
                parsed = self._parse_possible_dict_string(output)
                if isinstance(parsed, dict):
                    normalized = parsed
                    status_val = str(parsed.get('status', '')).lower()
                    if 'error' in parsed:
                        success = False
                        error_message = str(parsed.get('error'))
                    elif status_val and status_val != 'success':
                        success = False
                        error_message = f"status={parsed.get('status')}"
                    elif 'error_type' in parsed:
                        success = False
                        error_message = str(parsed.get('error_type'))
                else:
                    # Heuristic string checks
                    lower = output.lower()
                    error_indicators = [
                        'error', 'exception', 'failed', 'failure', 'invalid',
                        'no-such-column', 'query coordinator error', 'soql error',
                        '400 client error', 'bad request', 'could not parse',
                        'expected', 'table identifier'
                    ]
                    if any(ind in lower for ind in error_indicators):
                        success = False
                        error_message = output
            else:
                normalized = output
        except Exception as e:
            # Fallback to marking as success unless clear failure
            self.logger.warning(f"Error determining tool success: {e}")
        return success, error_message, self._make_json_serializable(normalized)

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools based on the configured tool groups."""
        tools = tool_factory.create_tools_for_groups(self.tool_groups)
        self.logger.info(f"Created {len(tools)} tools: {[tool.name for tool in tools]}")
        return tools

    def _create_agent(self, metric_details: Dict[str, Any]) -> AgentExecutor:
        """Create the LangChain agent executor with modular prompts."""
        try:
            self.logger.info(f"=== Creating agent ===")
            self.logger.info(f"Model: {self.model_key}")
            self.logger.info(f"Tools count: {len(self.tools)}")
            
            # Debug tool information
            for i, tool in enumerate(self.tools):
                self.logger.info(f"Tool {i}: {tool.name} - {type(tool)}")
                if hasattr(tool, 'args_schema'):
                    self.logger.info(f"  Args schema: {tool.args_schema}")
            
            # Get required prompt sections for the current tool groups
            required_sections = tool_factory.get_required_prompt_sections(self.tool_groups)
            self.logger.info(f"Required sections: {required_sections}")
            
            # Build the system prompt
            system_prompt = prompt_builder.build_system_prompt(
                required_sections=required_sections,
                metric_details=metric_details,
                include_all_sections=self.include_all_sections
            )
            self.logger.info(f"Built system prompt, length: {len(system_prompt)}")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            self.logger.info(f"Created prompt template")
            
            # Create the agent - use the working approach from our test
            self.logger.info(f"Creating agent with compatible method...")
            
            # Use tool calling agent as primary (it worked in our test)
            try:
                from langchain.agents import create_tool_calling_agent
                self.logger.info("Creating tool calling agent...")
                agent = create_tool_calling_agent(self.llm, self.tools, prompt)
                agent_type = "tool_calling"
                self.logger.info(f"Successfully created tool calling agent")
            except Exception as e:
                self.logger.warning(f"Tool calling agent failed: {e}")
                self.logger.warning(f"Error type: {type(e)}")
                self.logger.warning(f"Error details: {str(e)}")
                
                # Fallback to OpenAI functions agent
                try:
                    from langchain.agents import create_openai_functions_agent
                    self.logger.info("Creating OpenAI functions agent...")
                    agent = create_openai_functions_agent(self.llm, self.tools, prompt)
                    agent_type = "openai_functions"
                    self.logger.info(f"Successfully created OpenAI functions agent")
                except Exception as e2:
                    self.logger.error(f"All agent creation methods failed")
                    self.logger.error(f"OpenAI functions agent failed: {e2}")
                    self.logger.error(f"Error type: {type(e2)}")
                    self.logger.error(f"Error details: {str(e2)}")
                    raise Exception(f"Failed to create any compatible agent. Last error: {e2}")
            
            self.logger.info(f"Created agent: {agent}")
            self.logger.info(f"Agent class: {agent.__class__.__name__}")
            self.logger.info(f"Agent type: {agent_type}")
            self.logger.info(f"Agent has tools: {hasattr(agent, 'tools')}")
            if hasattr(agent, 'tools'):
                self.logger.info(f"Agent tools: {[tool.name for tool in agent.tools]}")
            
            self.logger.info(f"Creating AgentExecutor...")
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=50
            )
            self.logger.info(f"Created AgentExecutor: {executor}")
            
            # Ensure callbacks are properly set
            if hasattr(executor, 'callbacks'):
                self.logger.info(f"AgentExecutor has callbacks attribute")
            else:
                self.logger.warning(f"AgentExecutor does not have callbacks attribute")
                
            return executor
        except Exception as e:
            self.logger.error(f"Error creating agent: {e}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error details: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        if role == "user":
            self.messages.append(HumanMessage(content=content))
        elif role == "assistant":
            self.messages.append(AIMessage(content=content))

    def explain_change_sync(self, prompt: str, metric_details: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Synchronously explain a change using the LangChain agent."""
        session = None
        execution_callback = None
        
        try:
            # Create session for logging if enabled
            if self.enable_session_logging and self.session_logger:
                session = self.session_logger.create_session(
                    model=self.model_key,
                    model_config=self.model_config,
                    tool_groups=self.tool_groups,
                    available_tools=[tool.name for tool in self.tools],
                    session_id=session_id
                )
                session.user_input = prompt
                
                # Add user message to session conversation
                session.conversation.append(ConversationMessage(
                    role="user",
                    content=prompt,
                    timestamp=datetime.now().isoformat()
                ))
                
                # Create enhanced execution trace callback for session logging
                execution_callback = EnhancedExecutionTraceCallback(session)
                self.logger.info(f"Created enhanced execution callback for session logging")
                self.logger.info("Created EnhancedExecutionTraceCallback with session logging")
            else:
                # Fall back to original callback
                execution_callback = ExecutionTraceCallback()
                self.logger.info("Created ExecutionTraceCallback (no session logging)")
            
            # Create agent with callback
            self.agent_executor = self._create_agent(metric_details)
            
            # Set callbacks properly - try different approaches
            if hasattr(self.agent_executor, 'callbacks'):
                self.agent_executor.callbacks = [execution_callback]
                self.logger.info(f"Set callbacks on agent executor: {self.agent_executor.callbacks}")
            else:
                # Try setting callbacks on the agent itself
                if hasattr(self.agent_executor.agent, 'callbacks'):
                    self.agent_executor.agent.callbacks = [execution_callback]
                    self.logger.info("Set callbacks on agent.agent")
                else:
                    self.logger.warning("No callbacks attribute found on agent executor or agent")
            
            self.add_message("user", prompt)

            result = self.agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": self.messages[:-1] # Exclude current message
                },
                config={"callbacks": [execution_callback]}
            )

            response = result.get("output", "")
            cleaned_response = self._extract_clean_response(response)
            self.add_message("assistant", cleaned_response if cleaned_response else response)
            
            # Update session with final response
            if session:
                session.final_response = cleaned_response if cleaned_response else response
                session.success = True
                
                # Add assistant message to session conversation
                session.conversation.append(ConversationMessage(
                    role="assistant",
                    content=cleaned_response if cleaned_response else response,
                    timestamp=datetime.now().isoformat(),
                    sender="LangChain Explainer"
                ))
                
                # Log the complete session
                self.session_logger.log_session(session)

            return { 
                "success": True, 
                "explanation": cleaned_response if cleaned_response else response,
                "execution_trace": execution_callback.execution_trace if hasattr(execution_callback, 'execution_trace') else [],
                "session_id": session.session_id if session else None
            }
        except Exception as e:
            self.logger.error(f"Error in explain_change_sync: {e}")
            
            # Update session with error information
            if session:
                session.success = False
                session.error_summary = str(e)
                
                # Add error message to session conversation
                session.conversation.append(ConversationMessage(
                    role="assistant",
                    content=f"Error: {str(e)}",
                    timestamp=datetime.now().isoformat(),
                    sender="LangChain Explainer"
                ))
                
                # Log the session with error
                if self.session_logger:
                    self.session_logger.log_session(session)
            
            return { "success": False, "error": str(e), "session_id": session.session_id if session else None }
    
    async def explain_change_streaming(self, prompt: str, metric_details: Dict[str, Any] = None, session_id: Optional[str] = None):
        """Stream explanations using direct LLM streaming for real token-by-token output."""
        session = None
        try:
            self.logger.info(f"=== Starting explain_change_streaming ===")
            self.logger.info(f"Prompt: {prompt}")
            
            # Create session for logging if enabled
            self.logger.info(f"Session logging check: enable_session_logging={self.enable_session_logging}, session_logger={self.session_logger}")
            if self.enable_session_logging and self.session_logger:
                self.logger.info("Creating session for streaming logging")
                session = self.session_logger.create_session(
                    model=self.model_key,
                    model_config=self.model_config,
                    tool_groups=self.tool_groups,
                    available_tools=[tool.name for tool in self.tools],
                    session_id=session_id
                )
                session.user_input = prompt
                
                # Add user message to session conversation
                session.conversation.append(ConversationMessage(
                    role="user",
                    content=prompt,
                    timestamp=datetime.now().isoformat()
                ))
                
                # Log the session file immediately so it's available for conversation viewer
                self.session_logger.log_session(session)
                self.logger.info(f"Created and logged initial session {session.session_id} for streaming")
            
            # Check if this is an Anthropic model and handle it differently
            is_anthropic = hasattr(self.llm, 'model') and 'claude' in self.llm.model.lower()
            self.logger.info(f"Is Anthropic model: {is_anthropic}")
            
            # Create execution trace callback for session logging
            if self.enable_session_logging and session:
                execution_callback = EnhancedExecutionTraceCallback(session)
                self.logger.info(f"Created enhanced execution callback for session logging: {execution_callback}")
            else:
                execution_callback = ExecutionTraceCallback()
                self.logger.info(f"Created basic execution callback: {execution_callback}")
            
            # Create agent with callback
            self.agent_executor = self._create_agent(metric_details or {})
            self.logger.info(f"Created agent executor: {self.agent_executor}")
            
            # Set callbacks properly
            if hasattr(self.agent_executor, 'callbacks'):
                self.agent_executor.callbacks = [execution_callback]
                self.logger.info(f"Set callbacks on agent executor: {self.agent_executor.callbacks}")
            else:
                # Try setting callbacks on the agent itself
                if hasattr(self.agent_executor.agent, 'callbacks'):
                    self.agent_executor.agent.callbacks = [execution_callback]
                    self.logger.info("Set callbacks on agent.agent")
                else:
                    self.logger.warning("No callbacks attribute found on agent executor or agent")
            
            self.add_message("user", prompt)
            
            # Track tool calls and content
            tool_calls_in_progress = {}
            response_content = ""
            
            self.logger.info(f"Starting stream with input: {prompt}")
            self.logger.info(f"Available tools: {[tool.name for tool in self.tools]}")
            self.logger.info(f"Agent type: {type(self.agent_executor.agent)}")
            
            # For Anthropic models, use a modified agent approach that handles streaming better
            if is_anthropic:
                self.logger.info("Using modified agent streaming approach for Anthropic models")
                
                # Use astream_events but with better handling for Anthropic
                event_count = 0
                async for event in self.agent_executor.astream_events({
                    "input": prompt,
                    "chat_history": self.messages[:-1]
                }, version="v1", config={"callbacks": [execution_callback]}):
                    event_count += 1
                    self.logger.info(f"=== Anthropic Processing event #{event_count} ===")
                    self.logger.info(f"Event type: {event.get('event')}")
                    self.logger.info(f"Event: {event}")
                    
                    # Handle LLM streaming events for Anthropic
                    if event.get('event') == 'on_llm_new_token':
                        try:
                            token = event.get('data', {}).get('token', '')
                            if token:
                                # Check if this token contains internal agent data patterns
                                if any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']):
                                    self.logger.info(f"Skipping token with internal data: {token}")
                                    continue
                                
                                self.logger.info(f"Anthropic yielding token: {token}")
                                yield f"data: {json.dumps({'content': token})}\n\n"
                                response_content += token
                                
                                # Track streaming content for session logging (but don't add to conversation yet)
                                if self.enable_session_logging and session:
                                    # Store the streaming content for later use
                                    session._streaming_content = response_content
                        except Exception as e:
                            self.logger.error(f"Error processing Anthropic LLM token event: {e}")
                    
                    # Handle chat model streaming events for Anthropic
                    elif event.get('event') == 'on_chat_model_stream':
                        try:
                            chunk = event.get('data', {}).get('chunk', {})
                            token = ""
                            
                            # Handle Anthropic's content structure which can be a list of content blocks
                            if hasattr(chunk, 'content'):
                                content = chunk.content
                                if isinstance(content, list):
                                    # Extract text from content blocks
                                    for block in content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            token += block.get('text', '')
                                        elif isinstance(block, str):
                                            token += block
                                elif isinstance(content, str):
                                    token = content
                                else:
                                    token = str(content)
                            elif isinstance(chunk, dict):
                                content = chunk.get('content', '')
                                if isinstance(content, list):
                                    # Extract text from content blocks
                                    for block in content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            token += block.get('text', '')
                                        elif isinstance(block, str):
                                            token += block
                                elif isinstance(content, str):
                                    token = content
                                else:
                                    token = str(content)
                            else:
                                token = str(chunk)
                            
                            if token:
                                # Check if this token contains internal agent data patterns
                                if any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']):
                                    self.logger.info(f"Skipping token with internal data: {token}")
                                    continue
                                
                                self.logger.info(f"Anthropic yielding token from chat model: {token}")
                                yield f"data: {json.dumps({'content': token})}\n\n"
                                response_content += token
                                
                                # Track streaming content for session logging (but don't add to conversation yet)
                                if self.enable_session_logging and session:
                                    # Store the streaming content for later use
                                    session._streaming_content = response_content
                        except Exception as e:
                            self.logger.error(f"Error processing Anthropic chat model stream event: {e}")
                    
                    # Handle tool call events for Anthropic
                    elif event.get('event') == 'on_tool_start':
                        try:
                            tool_name = event.get('name', 'unknown')
                            tool_id = f"tool_{tool_name}_{int(time.time())}"
                            tool_calls_in_progress[tool_name] = tool_id
                            
                            # Send tool call start
                            tool_call_data = {'tool_call_start': tool_name, 'tool_id': tool_id}
                            self.logger.info(f"Anthropic yielding tool_call_start from event: {tool_call_data}")
                            yield f"data: {json.dumps(tool_call_data)}\n\n"
                            
                            # Try to get tool arguments from the event
                            try:
                                tool_input = event.get('data', {}).get('input', '')
                                if tool_input:
                                    # Try to parse as JSON if it's a string
                                    if isinstance(tool_input, str):
                                        try:
                                            arguments = json.loads(tool_input)
                                        except json.JSONDecodeError:
                                            arguments = tool_input
                                    else:
                                        arguments = tool_input
                                    
                                    tool_args_data = {
                                        'tool_call_args': tool_name, 
                                        'tool_id': tool_id, 
                                        'arguments': arguments
                                    }
                                    self.logger.info(f"Anthropic yielding tool_call_args from event: {tool_args_data}")
                                    yield f"data: {json.dumps(tool_args_data)}\n\n"
                            except Exception as e:
                                self.logger.error(f"Error processing Anthropic tool arguments from event: {e}")
                                
                        except Exception as e:
                            self.logger.error(f"Error processing Anthropic tool start event: {e}")
                    
                    elif event.get('event') == 'on_tool_end':
                        try:
                            tool_name = event.get('name', 'unknown')
                            output = event.get('data', {}).get('output', '')
                            tool_id = tool_calls_in_progress.get(tool_name, f"tool_{tool_name}")
                            # Determine success from output content
                            success, error_message, normalized = self._determine_tool_success(output)
                            tool_call_data = {
                                'tool_call_complete': tool_name,
                                'success': success,
                                'response': normalized,
                                'tool_id': tool_id
                            }
                            if not success and error_message:
                                tool_call_data['error'] = error_message
                            self.logger.info(f"Anthropic yielding tool_call_complete from event: {tool_call_data}")
                            yield f"data: {json.dumps(tool_call_data)}\n\n"
                            
                            # Remove from in-progress tracking
                            if tool_name in tool_calls_in_progress:
                                del tool_calls_in_progress[tool_name]
                        except Exception as e:
                            self.logger.error(f"Error processing Anthropic tool end event: {e}")
                    
                    # Handle final output events for Anthropic
                    elif event.get('event') == 'on_chain_end':
                        try:
                            output = event.get('data', {}).get('output', '')
                            if output:
                                # Check if this output contains internal agent state - if so, skip it entirely
                                output_str = str(output)
                                if any(pattern in output_str for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']):
                                    self.logger.info(f"Skipping output with internal agent state: {output_str[:100]}...")
                                    continue
                                
                                # Check if this output contains tool response data - if so, skip it entirely
                                if any(pattern in output_str for pattern in ['"status": "success"', '"results": {', 'ToolAgentAction', 'tool_call_id']):
                                    self.logger.info(f"Skipping output with tool response data: {output_str[:100]}...")
                                    continue
                                
                                # For agent responses, we need to extract only the final text response
                                if isinstance(output, dict):
                                    # Look for the actual response text in the output
                                    if 'output' in output:
                                        final_text = output['output']
                                    elif 'response' in output:
                                        final_text = output['response']
                                    elif 'text' in output:
                                        final_text = output['text']
                                    else:
                                        # If it's a dict but no clear text field, convert to string and clean
                                        final_text = str(output)
                                else:
                                    final_text = output
                                
                                # Use the new cleaning method to extract clean response text
                                text_content = self._extract_clean_response(final_text)
                                
                                # Additional check: if the content looks like technical data, don't stream it
                                if text_content and len(text_content) > 0:
                                    # Check if this looks like technical data that shouldn't be in chat
                                    technical_indicators = ['status', 'results', 'response', 'data', 'shape', 'columns', 'queryURL', 'tool_call_id', 'log=']
                                    is_technical = any(indicator in text_content.lower() for indicator in technical_indicators)
                                    
                                    if is_technical and len(text_content) < 100:
                                        self.logger.info(f"Skipping technical data: {text_content[:50]}...")
                                        continue
                                
                                # Only stream if we haven't already streamed this content and it's not empty
                                if text_content and text_content != response_content and len(text_content.strip()) > 0:
                                    # Check for duplication - if the new content is just a repeat of what we already have
                                    if response_content and text_content.startswith(response_content):
                                        # This is a duplicate, skip it
                                        self.logger.info(f"Skipping duplicate content: {text_content[:50]}...")
                                        continue
                                    
                                    remaining_content = text_content[len(response_content):]
                                    if remaining_content and len(remaining_content.strip()) > 0:
                                        self.logger.info(f"Anthropic yielding remaining content: {remaining_content[:100]}...")
                                        yield f"data: {json.dumps({'content': remaining_content})}\n\n"
                                        response_content = text_content
                                        
                                        # Call the callback's on_llm_new_token method to capture intermediate responses
                                        # For Anthropic models, capture the remaining content as an intermediate response
                                        if hasattr(execution_callback, 'on_llm_new_token'):
                                            execution_callback.on_llm_new_token(remaining_content)
                                            
                                        # Also capture the full response content for intermediate responses
                                        if hasattr(execution_callback, 'on_llm_new_token') and text_content:
                                            # For Anthropic models, capture the full response as a single intermediate response
                                            # This ensures we get at least one intermediate response even if streaming is minimal
                                            if not execution_callback.intermediate_responses:
                                                execution_callback.intermediate_responses.append({
                                                    "content": text_content.strip(),
                                                    "timestamp": datetime.now().isoformat(),
                                                    "type": "intermediate_response"
                                                })
                        except Exception as e:
                            self.logger.error(f"Error processing Anthropic chain end event: {e}")
                
                # Also check execution trace for any tool calls that might not be caught by events
                current_trace_count = len(execution_callback.execution_trace)
                if current_trace_count > 0:
                    self.logger.info(f"Processing {current_trace_count} tool traces from execution callback for Anthropic")
                    
                    for i in range(current_trace_count):
                        try:
                            trace = execution_callback.execution_trace[i]
                            self.logger.info(f"Processing trace {i}: {trace}")
                            
                            # Check if trace has the required fields
                            if not isinstance(trace, dict):
                                self.logger.warning(f"Trace {i} is not a dict: {trace}")
                                continue
                                
                            trace_type = trace.get("type")
                            if not trace_type:
                                self.logger.warning(f"Trace {i} has no type: {trace}")
                                continue
                                
                            if trace_type == "tool_start":
                                tool_name = trace.get('tool', 'unknown')
                                tool_id = f"tool_{i + 1}_{int(time.time())}"
                                tool_calls_in_progress[tool_name] = tool_id
                                
                                # Send tool call start
                                tool_call_data = {'tool_call_start': tool_name, 'tool_id': tool_id}
                                self.logger.info(f"Anthropic yielding tool_call_start from trace: {tool_call_data}")
                                yield f"data: {json.dumps(tool_call_data)}\n\n"
                                
                                # Send tool arguments if available
                                tool_input = trace.get('input', '')
                                if tool_input:
                                    try:
                                        # Try to parse as JSON if it's a string
                                        if isinstance(tool_input, str):
                                            try:
                                                arguments = json.loads(tool_input)
                                            except json.JSONDecodeError:
                                                arguments = tool_input
                                        else:
                                            arguments = tool_input
                                        
                                        tool_args_data = {
                                            'tool_call_args': tool_name, 
                                            'tool_id': tool_id, 
                                            'arguments': arguments
                                        }
                                        self.logger.info(f"Anthropic yielding tool_call_args from trace: {tool_args_data}")
                                        yield f"data: {json.dumps(tool_args_data)}\n\n"
                                    except Exception as e:
                                        self.logger.error(f"Error processing Anthropic tool arguments from trace: {e}")
                                
                            elif trace_type == "tool_end":
                                tool_name = trace.get('tool', 'unknown')
                                success = trace.get('successful', True)
                                output = trace.get('output', '')
                                tool_id = tool_calls_in_progress.get(tool_name, f"tool_{tool_name}")
                                # Normalize response and attach error if present
                                success2, error_message, normalized = self._determine_tool_success(output)
                                # Prefer callback success if available, but fall back to parsed result
                                final_success = success if success is not None else success2
                                tool_call_data = {
                                    'tool_call_complete': tool_name,
                                    'success': final_success,
                                    'response': self._make_json_serializable(normalized),
                                    'tool_id': tool_id
                                }
                                if not final_success and error_message:
                                    tool_call_data['error'] = error_message
                                self.logger.info(f"Anthropic yielding tool_call_complete from trace: {tool_call_data}")
                                yield f"data: {json.dumps(tool_call_data)}\n\n"
                                
                                # Remove from in-progress tracking
                                if tool_name in tool_calls_in_progress:
                                    del tool_calls_in_progress[tool_name]
                                    
                        except Exception as e:
                            self.logger.error(f"Error processing Anthropic trace {i}: {e}")
                            continue
                
                # Add assistant message to conversation history (use cleaned content)
                cleaned_response = self._extract_clean_response(response_content)
                self.add_message("assistant", cleaned_response if cleaned_response else response_content)
                
                # Complete and log session if enabled (for Anthropic path)
                if self.enable_session_logging and self.session_logger and session:
                    # Set final response first, then build conversation
                    session.final_response = cleaned_response if cleaned_response else response_content
                    
                    # Build conversation from intermediate responses and tool calls
                    self._build_conversation_from_streaming(session, execution_callback)
                    
                    # Mark session as successful
                    session.success = True
                    session.end_time = datetime.now().isoformat()
                    
                    # Log the complete session
                    self.session_logger.log_session(session)
                    self.logger.info(f"Logged streaming session {session.session_id}")
                    
                    # Send session_id in completion signal
                    completion_data = {'completed': True, 'session_id': session.session_id}
                else:
                    completion_data = {'completed': True}
                
                # Send completion signal
                self.logger.info("Sending completion signal")
                yield f"data: {json.dumps(completion_data)}\n\n"
                return
            
            # For non-Anthropic models, stream via LangChain agent events to ensure tools and identity are applied
            self.logger.info("Using LangChain agent astream_events for non-Anthropic models")
            
            # Use astream_events to get token and tool streaming with prompts and tools injected
            self.logger.info("Starting astream_events with agent executor")
            event_count = 0
            async for event in self.agent_executor.astream_events({
                "input": prompt,
                "chat_history": self.messages[:-1]
            }, version="v1", config={"callbacks": [execution_callback]}):
                event_count += 1
                self.logger.info(f"=== Processing event #{event_count} ===")
                self.logger.info(f"Event type: {event.get('event')}")
                self.logger.info(f"Event: {event}")
                
                # Handle LLM streaming events - this is where we get real token-by-token streaming
                if event.get('event') == 'on_llm_new_token':
                    try:
                        token = event.get('data', {}).get('token', '')
                        if token:
                            # Check if this token contains internal agent data patterns or raw JSON
                            if (any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']) or
                                token.strip().startswith('{') and token.strip().endswith('}') or
                                'tool_use' in token.lower() or
                                'type' in token.lower() and 'input' in token.lower() or
                                'partial_json' in token.lower() or
                                'index' in token.lower() and ('toolu_' in token or 'call_' in token) or
                                any(pattern in token for pattern in ['toolu_', 'call_', 'input_json_delta', 'tool_use'])):
                                self.logger.info(f"Skipping token with internal data or JSON: {token}")
                                continue
                            
                            self.logger.info(f"Yielding token: {token}")
                            yield f"data: {json.dumps({'content': token})}\n\n"
                            response_content += token
                            
                            # Note: LangChain's callback system automatically calls on_llm_new_token
                            # We don't need to call it manually here to avoid duplication
                            
                            # Track streaming content for session logging (but don't add to conversation yet)
                            if self.enable_session_logging and session:
                                # Store the streaming content for later use
                                session._streaming_content = response_content
                    except Exception as e:
                        self.logger.error(f"Error processing LLM token event: {e}")
                
                # Handle chat model streaming events
                elif event.get('event') == 'on_chat_model_stream':
                    try:
                        # Extract the token from the event
                        chunk = event.get('data', {}).get('chunk', {})
                        if hasattr(chunk, 'content'):
                            token = chunk.content
                        elif isinstance(chunk, dict):
                            token = chunk.get('content', '')
                        else:
                            token = str(chunk)
                        
                        if token:
                            # Check if this token contains internal agent data patterns or raw JSON
                            if (any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']) or
                                token.strip().startswith('{') and token.strip().endswith('}') or
                                'tool_use' in token.lower() or
                                'type' in token.lower() and 'input' in token.lower() or
                                'partial_json' in token.lower() or
                                'index' in token.lower() and ('toolu_' in token or 'call_' in token) or
                                any(pattern in token for pattern in ['toolu_', 'call_', 'input_json_delta', 'tool_use'])):
                                self.logger.info(f"Skipping token with internal data or JSON: {token}")
                                continue
                            
                            self.logger.info(f"Yielding token from chat model: {token}")
                            yield f"data: {json.dumps({'content': token})}\n\n"
                            response_content += token
                            
                            # Note: LangChain's callback system automatically calls on_llm_new_token
                            # We don't need to call it manually here to avoid duplication
                            
                            # Track streaming content for session logging (but don't add to conversation yet)
                            if self.enable_session_logging and session:
                                # Store the streaming content for later use
                                session._streaming_content = response_content
                    except Exception as e:
                        self.logger.error(f"Error processing chat model stream event: {e}")
                
                # Handle tool call events - these should stream in real-time
                elif event.get('event') == 'on_tool_start':
                    try:
                        tool_name = event.get('name', 'unknown')
                        tool_id = f"tool_{tool_name}_{int(time.time())}"
                        tool_calls_in_progress[tool_name] = tool_id
                        
                        # Send tool call start
                        tool_call_data = {'tool_call_start': tool_name, 'tool_id': tool_id}
                        self.logger.info(f"Yielding tool_call_start from event: {tool_call_data}")
                        yield f"data: {json.dumps(tool_call_data)}\n\n"
                        
                        # Try to get tool arguments from the event
                        try:
                            tool_input = event.get('data', {}).get('input', '')
                            if tool_input:
                                # Try to parse as JSON if it's a string
                                if isinstance(tool_input, str):
                                    try:
                                        arguments = json.loads(tool_input)
                                    except json.JSONDecodeError:
                                        arguments = tool_input
                                else:
                                    arguments = tool_input
                                
                                tool_args_data = {
                                    'tool_call_args': tool_name, 
                                    'tool_id': tool_id, 
                                    'arguments': arguments
                                }
                                self.logger.info(f"Yielding tool_call_args from event: {tool_args_data}")
                                yield f"data: {json.dumps(tool_args_data)}\n\n"
                        except Exception as e:
                            self.logger.error(f"Error processing tool arguments from event: {e}")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing tool start event: {e}")
                
                elif event.get('event') == 'on_tool_end':
                    try:
                        tool_name = event.get('name', 'unknown')
                        output = event.get('data', {}).get('output', '')
                        tool_id = tool_calls_in_progress.get(tool_name, f"tool_{tool_name}")
                        # Determine success from output content
                        success, error_message, normalized = self._determine_tool_success(output)
                        tool_call_data = {
                            'tool_call_complete': tool_name,
                            'success': success,
                            'response': normalized,
                            'tool_id': tool_id
                        }
                        if not success and error_message:
                            tool_call_data['error'] = error_message
                        self.logger.info(f"Yielding tool_call_complete from event: {tool_call_data}")
                        yield f"data: {json.dumps(tool_call_data)}\n\n"
                        
                        # Remove from in-progress tracking
                        if tool_name in tool_calls_in_progress:
                            del tool_calls_in_progress[tool_name]
                    except Exception as e:
                        self.logger.error(f"Error processing tool end event: {e}")
                
                # Handle final output events - but only if we haven't streamed the content yet
                elif event.get('event') == 'on_chain_end':
                    try:
                        output = event.get('data', {}).get('output', '')
                        if output:
                            # Check if this output contains internal agent state - if so, skip it entirely
                            output_str = str(output)
                            if any(pattern in output_str for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']):
                                self.logger.info(f"Skipping output with internal agent state: {output_str[:100]}...")
                                continue
                            
                            # Check if this output contains tool response data - if so, skip it entirely
                            if any(pattern in output_str for pattern in ['"status": "success"', '"results": {', 'ToolAgentAction', 'tool_call_id']):
                                self.logger.info(f"Skipping output with tool response data: {output_str[:100]}...")
                                continue
                            
                            # For agent responses, we need to extract only the final text response
                            # The agent might return a complex object with internal state
                            if isinstance(output, dict):
                                # Look for the actual response text in the output
                                if 'output' in output:
                                    final_text = output['output']
                                elif 'response' in output:
                                    final_text = output['response']
                                elif 'text' in output:
                                    final_text = output['text']
                                else:
                                    # If it's a dict but no clear text field, convert to string and clean
                                    final_text = str(output)
                            else:
                                final_text = output
                            
                            # Use the new cleaning method to extract clean response text
                            text_content = self._extract_clean_response(final_text)
                            
                            # Additional check: if the content looks like technical data, don't stream it
                            if text_content and len(text_content) > 0:
                                # Check if this looks like technical data that shouldn't be in chat
                                technical_indicators = ['status', 'results', 'response', 'data', 'shape', 'columns', 'queryURL', 'tool_call_id', 'log=']
                                is_technical = any(indicator in text_content.lower() for indicator in technical_indicators)
                                
                                if is_technical and len(text_content) < 100:
                                    self.logger.info(f"Skipping technical data: {text_content[:50]}...")
                                    continue
                            
                            # Only stream if we haven't already streamed this content and it's not empty
                            if text_content and text_content != response_content and len(text_content.strip()) > 0:
                                # Check for duplication - if the new content is just a repeat of what we already have
                                if response_content and text_content.startswith(response_content):
                                    # This is a duplicate, skip it
                                    self.logger.info(f"Skipping duplicate content: {text_content[:50]}...")
                                    continue
                                
                                remaining_content = text_content[len(response_content):]
                                if remaining_content and len(remaining_content.strip()) > 0:
                                    self.logger.info(f"Yielding remaining content: {remaining_content[:100]}...")
                                    yield f"data: {json.dumps({'content': remaining_content})}\n\n"
                                    response_content = text_content
                    except Exception as e:
                        self.logger.error(f"Error processing chain end event: {e}")
                
                # Handle error events that might indicate stop conditions
                elif event.get('event') == 'on_chain_error':
                    try:
                        error_data = event.get('data', {})
                        error = error_data.get('error') or error_data.get('exception')
                        if error:
                            error_str = str(error)
                            # Check if this is the specific stop condition warning
                            if 'stopping agent prematurely due to triggering stop condition' in error_str.lower():
                                self.logger.warning(f"Detected agent stop condition: {error_str}")
                                # Send special signal to frontend to show continue button
                                stop_message = {'agent_stopped': True, 'reason': 'stop_condition'}
                                self.logger.info(f"Sending stop condition message to frontend: {stop_message}")
                                yield f"data: {json.dumps(stop_message)}\n\n"
                    except Exception as e:
                        self.logger.error(f"Error processing chain error event: {e}")
                
                # Also check execution trace for any tool calls that might not be caught by events
                current_trace_count = len(execution_callback.execution_trace) if hasattr(execution_callback, 'execution_trace') else 0
                if current_trace_count > 0:
                    self.logger.info(f"Processing {current_trace_count} tool traces from execution callback")
                    
                    for i in range(current_trace_count):
                        try:
                            trace = execution_callback.execution_trace[i] if hasattr(execution_callback, 'execution_trace') else None
                            self.logger.info(f"Processing trace {i}: {trace}")
                            
                            # Check if trace has the required fields
                            if not isinstance(trace, dict):
                                self.logger.warning(f"Trace {i} is not a dict: {trace}")
                                continue
                                
                            trace_type = trace.get("type")
                            if not trace_type:
                                self.logger.warning(f"Trace {i} has no type: {trace}")
                                continue
                                
                            if trace_type == "tool_start":
                                tool_name = trace.get('tool', 'unknown')
                                tool_id = f"tool_{i + 1}_{int(time.time())}"
                                tool_calls_in_progress[tool_name] = tool_id
                                
                                # Send tool call start
                                tool_call_data = {'tool_call_start': tool_name, 'tool_id': tool_id}
                                self.logger.info(f"Yielding tool_call_start from trace: {tool_call_data}")
                                yield f"data: {json.dumps(tool_call_data)}\n\n"
                                
                                # Send tool arguments if available
                                tool_input = trace.get('input', '')
                                if tool_input:
                                    try:
                                        # Try to parse as JSON if it's a string
                                        if isinstance(tool_input, str):
                                            try:
                                                arguments = json.loads(tool_input)
                                            except json.JSONDecodeError:
                                                arguments = tool_input
                                        else:
                                            arguments = tool_input
                                        
                                        tool_args_data = {
                                            'tool_call_args': tool_name, 
                                            'tool_id': tool_id, 
                                            'arguments': arguments
                                        }
                                        self.logger.info(f"Yielding tool_call_args from trace: {tool_args_data}")
                                        yield f"data: {json.dumps(tool_args_data)}\n\n"
                                    except Exception as e:
                                        self.logger.error(f"Error processing tool arguments from trace: {e}")
                                
                            elif trace_type == "tool_end":
                                tool_name = trace.get('tool', 'unknown')
                                success = trace.get('successful', True)
                                output = trace.get('output', '')
                                tool_id = tool_calls_in_progress.get(tool_name, f"tool_{tool_name}")
                                # Normalize response and attach error if present
                                success2, error_message, normalized = self._determine_tool_success(output)
                                final_success = success if success is not None else success2
                                tool_call_data = {
                                    'tool_call_complete': tool_name,
                                    'success': final_success,
                                    'response': self._make_json_serializable(normalized),
                                    'tool_id': tool_id
                                }
                                if not final_success and error_message:
                                    tool_call_data['error'] = error_message
                                self.logger.info(f"Yielding tool_call_complete from trace: {tool_call_data}")
                                yield f"data: {json.dumps(tool_call_data)}\n\n"
                                
                                # Remove from in-progress tracking
                                if tool_name in tool_calls_in_progress:
                                    del tool_calls_in_progress[tool_name]
                                    
                        except Exception as e:
                            self.logger.error(f"Error processing trace {i}: {e}")
                            continue
                
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error in explain_change_streaming: {error_str}")
            
            # Check for various stop conditions
            stop_indicators = [
                'stopping agent prematurely due to triggering stop condition',
                'max_iterations',
                'maximum iterations',
                'iteration limit'
            ]
            
            if any(indicator in error_str.lower() for indicator in stop_indicators):
                self.logger.warning(f"Detected agent stop condition in main exception: {error_str}")
                # Send special signal to frontend to show continue button
                stop_message = {'agent_stopped': True, 'reason': 'stop_condition'}
                self.logger.info(f"Sending stop condition message to frontend from main exception: {stop_message}")
                yield f"data: {json.dumps(stop_message)}\n\n"
            else:
                yield f"data: {json.dumps({'error': error_str})}\n\n"
            
            # Log session with error if enabled
            if self.enable_session_logging and self.session_logger and session:
                session.success = False
                session.error = error_str
                session.end_time = datetime.now().isoformat()
                self.session_logger.log_session(session)
                self.logger.info(f"Logged streaming session {session.session_id} with error")
        
        # Add assistant message to conversation history (use cleaned content)
        cleaned_response = self._extract_clean_response(response_content)
        self.add_message("assistant", cleaned_response if cleaned_response else response_content)
        
        # Complete and log session if enabled
        if self.enable_session_logging and self.session_logger and session:
            # Set final response first, then build conversation
            session.final_response = cleaned_response if cleaned_response else response_content
            
            # Build conversation from intermediate responses and tool calls
            self._build_conversation_from_streaming(session, execution_callback)
            
            # Mark session as successful
            session.success = True
            session.end_time = datetime.now().isoformat()
            
            # Log the complete session
            self.session_logger.log_session(session)
            self.logger.info(f"Logged streaming session {session.session_id}")
            
            # Send session_id in completion signal
            completion_data = {'completed': True, 'session_id': session.session_id}
        else:
            completion_data = {'completed': True}
        
        # Send completion signal
        self.logger.info("Sending completion signal")
        yield f"data: {json.dumps(completion_data)}\n\n"
    
    def _build_conversation_from_streaming(self, session: AgentSession, execution_callback):
        """Build conversation from intermediate responses and tool calls captured during streaming."""
        try:
            # Clear existing conversation (except user message)
            user_message = None
            for msg in session.conversation:
                if msg.role == "user":
                    user_message = msg
                    break
            
            session.conversation = []
            if user_message:
                session.conversation.append(user_message)
            
            # Get intermediate responses from the callback
            intermediate_responses = getattr(execution_callback, 'intermediate_responses', [])
            
            # Build conversation by interleaving intermediate responses and tool calls
            # (but NOT the final response - that will be added separately at the end)
            
            # Create a list of all events with their timestamps for proper sorting
            events = []
            
            # Add intermediate responses
            for response in intermediate_responses:
                events.append({
                    'type': 'response',
                    'content': response["content"],
                    'timestamp': response["timestamp"]
                })
            
            # Add tool calls
            for tool_call in session.tool_calls:
                # Convert Unix timestamp to ISO format for proper sorting
                if isinstance(tool_call.start_time, (int, float)):
                    # Convert Unix timestamp to ISO format
                    from datetime import datetime
                    tool_timestamp = datetime.fromtimestamp(tool_call.start_time).isoformat()
                elif hasattr(tool_call.start_time, 'isoformat'):
                    tool_timestamp = tool_call.start_time.isoformat()
                else:
                    tool_timestamp = str(tool_call.start_time)
                
                events.append({
                    'type': 'tool_call',
                    'tool_name': tool_call.tool_name,
                    'timestamp': tool_timestamp
                })
            
            # Sort events by timestamp
            events.sort(key=lambda x: x['timestamp'])
            
            # Add events to conversation in chronological order
            for event in events:
                if event['type'] == 'response':
                    session.conversation.append(ConversationMessage(
                        role="assistant",
                        content=event["content"],
                        timestamp=event["timestamp"]
                    ))
                elif event['type'] == 'tool_call':
                    session.conversation.append(ConversationMessage(
                        role="tool_call",
                        content=f"Tool: {event['tool_name']}",
                        timestamp=event["timestamp"]
                    ))
            
            # Add the final response at the end (this will be set in the calling method)
            # The final response should be the last thing in the conversation
            if session.final_response:
                # Check if the final response is already in the intermediate responses
                # If so, don't add it again to avoid duplication
                final_response_in_intermediates = any(
                    resp["content"] == session.final_response 
                    for resp in intermediate_responses
                )
                
                # Check if the final response is just an accumulation of intermediate responses
                final_response_is_accumulation = False
                if intermediate_responses:
                    # Concatenate all intermediate responses without spaces
                    accumulated_content = "".join([resp["content"] for resp in intermediate_responses])
                    # Normalize whitespace for comparison
                    normalized_final = " ".join(session.final_response.strip().split())
                    normalized_accumulated = " ".join(accumulated_content.strip().split())
                    final_response_is_accumulation = normalized_final == normalized_accumulated
                
                # Add the final response only if:
                # 1. It's not already in intermediate responses AND
                # 2. It's not just an accumulation of intermediate responses AND
                # 3. We have meaningful intermediate responses (to avoid adding empty final response)
                if (not final_response_in_intermediates and 
                    not final_response_is_accumulation and 
                    intermediate_responses):
                    session.conversation.append(ConversationMessage(
                        role="assistant",
                        content=session.final_response,
                        timestamp=datetime.now().isoformat()
                    ))
            
            # Store intermediate responses in session for JSON serialization
            session.intermediate_responses = intermediate_responses
            
            self.logger.info(f"Built conversation with {len(session.conversation)} messages from streaming data")
            
        except Exception as e:
            self.logger.error(f"Error building conversation from streaming: {e}")
            # Fall back to simple conversation with just user and final assistant message
            session.conversation = [msg for msg in session.conversation if msg.role in ["user", "assistant"]]
    
    def _extract_clean_response(self, output):
        """Extract clean response text from agent output, filtering out internal state."""
        if isinstance(output, dict):
            # Handle Anthropic format with return_values and log keys
            if 'return_values' in output and 'log' in output:
                # Extract the actual response from the log field
                log_content = output['log']
                if isinstance(log_content, str):
                    # Try to parse as JSON if it's a string representation of a list
                    try:
                        import ast
                        parsed_log = ast.literal_eval(log_content)
                        if isinstance(parsed_log, list) and len(parsed_log) > 0:
                            # Extract text from the first item
                            first_item = parsed_log[0]
                            if isinstance(first_item, dict) and 'text' in first_item:
                                return first_item['text']
                    except (ValueError, SyntaxError):
                        pass
                    # If parsing fails, try to extract text directly
                    return log_content
                elif isinstance(log_content, list) and len(log_content) > 0:
                    # Direct list format
                    first_item = log_content[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        return first_item['text']
            
            # Handle other dict formats
            if 'output' in output:
                return self._extract_clean_response(output['output'])
            elif 'response' in output:
                return self._extract_clean_response(output['response'])
            elif 'text' in output:
                return output['text']
            else:
                # Convert dict to string and clean
                return self._extract_clean_response(str(output))
        elif isinstance(output, str):
            # If it's already a string, try to clean it
            import re
            
            # Handle the specific case where the output is a string representation of a dict
            # like "return_values={'output': [...]} log='[...]'"
            if 'return_values=' in output and 'log=' in output:
                # Try to extract the log content
                log_match = re.search(r"log='([^']*)'", output)
                if log_match:
                    log_content = log_match.group(1)
                    try:
                        import ast
                        parsed_log = ast.literal_eval(log_content)
                        if isinstance(parsed_log, list) and len(parsed_log) > 0:
                            first_item = parsed_log[0]
                            if isinstance(first_item, dict) and 'text' in first_item:
                                return first_item['text']
                    except (ValueError, SyntaxError):
                        pass
            
            # Simplified cleaning - only remove obvious internal artifacts
            cleaned = output
            
            # Remove obvious internal agent artifacts
            cleaned = re.sub(r'\{[^}]*agent_scratchpad[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*intermediate_steps[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*tool_call_id[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*tool_calls[^}]*\}', '', cleaned)
            
            # Remove log= artifacts
            cleaned = re.sub(r'log\s*=\s*"[^"]*"', '', cleaned)
            cleaned = re.sub(r'log\s*=\s*[^\s]*', '', cleaned)
            
            # Remove return_values patterns
            cleaned = re.sub(r'return_values\s*=\s*\{[^}]*\}', '', cleaned)
            cleaned = re.sub(r'return_values\s*=\s*[^\s]*', '', cleaned)
            
            # Remove ToolAgentAction patterns
            cleaned = re.sub(r'ToolAgentAction\([^)]*\)', '', cleaned)
            cleaned = re.sub(r'\[ToolAgentAction\([^\]]*\)\]', '', cleaned)
            
            # Clean up whitespace but preserve structure
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove empty lines at the beginning and end
            lines = cleaned.split('\n')
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            
            cleaned = '\n'.join(lines)
            
            # Only return empty if it's clearly just technical/logging data
            if len(cleaned) < 20 and ('log=' in cleaned.lower() or 'return_values=' in cleaned.lower()):
                return ""
            
            return cleaned if cleaned else output
        elif isinstance(output, list):
            # Handle Anthropic format: [{"text": "...", "type": "text", "index": 0}]
            if len(output) > 0 and isinstance(output[0], dict):
                text_content = output[0].get('text', '')
                return self._extract_clean_response(text_content)
            else:
                return str(output)
        else:
            return str(output)
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get information about the current agent configuration."""
        return {
            "model": self.model_key,
            "tool_groups": [g.value for g in self.tool_groups],
            "tools": [tool.name for tool in self.tools],
            "include_all_sections": self.include_all_sections,
            "required_prompt_sections": tool_factory.get_required_prompt_sections(self.tool_groups)
        }
    
    async def continue_analysis_streaming(self, continuation_prompt: str = None, metric_details: Dict[str, Any] = None):
        """Continue analysis from where the agent stopped due to stop condition."""
        try:
            self.logger.info("=== Starting continue_analysis_streaming ===")
            
            if continuation_prompt:
                # User provided additional context for continuation
                prompt = f"Please continue your analysis. {continuation_prompt}"
            else:
                # Default continuation prompt
                prompt = "Please continue your analysis from where you left off. Provide any additional insights or complete your previous thought."
            
            # Use the existing streaming method but with the continuation prompt
            async for chunk in self.explain_change_streaming(prompt, metric_details):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error in continue_analysis_streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def update_tool_groups(self, new_tool_groups: List[ToolGroup]):
        """Update the tool groups and recreate the agent."""
        self.tool_groups = new_tool_groups
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent(metric_details={})
        self.logger.info(f"Updated tool groups to: {[g.value for g in self.tool_groups]}")

def create_explainer_agent(
    model_key: Optional[str] = None,
    tool_groups: Optional[List[ToolGroup]] = None,
    include_all_sections: bool = False,
    enable_session_logging: bool = True
) -> LangChainExplainerAgent:
    """
    Factory function to create a new LangChain explainer agent instance.
    
    Args:
        model_key: Model to use for the agent
        tool_groups: List of tool groups to include (default: CORE only)
        include_all_sections: Whether to include all prompt sections
        enable_session_logging: Whether to enable detailed session logging
        
    Returns:
        Configured LangChainExplainerAgent instance
    """
    return LangChainExplainerAgent(
        model_key=model_key,
        tool_groups=tool_groups,
        include_all_sections=include_all_sections,
        enable_session_logging=enable_session_logging
    )

def get_available_tool_groups() -> Dict[str, List[str]]:
    """Get all available tool groups and their tools."""
    return tool_factory.list_available_tools()

def validate_tool_configuration(tool_groups: List[ToolGroup]) -> Dict[str, Any]:
    """
    Validate a tool group configuration.
    
    Args:
        tool_groups: List of tool groups to validate
        
    Returns:
        Validation results including available and missing tools
    """
    return tool_factory.validate_tool_availability(tool_groups)

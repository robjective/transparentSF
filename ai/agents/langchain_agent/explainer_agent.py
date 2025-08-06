"""
LangChain-based Explainer Agent for TransparentSF.
This version uses a modular system with selective tool inclusion.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import logging
import time
import asyncio

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

# Import our model configuration
from ..config.models import create_langchain_llm, get_default_model, get_model_config, ModelProvider

# Import modular components
from .config.tool_config import ToolGroup, tool_config
from .tools.tool_factory import tool_factory
from .prompts.modular_prompts import prompt_builder

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
                 include_all_sections: bool = False):
        """
        Initialize the LangChain explainer agent.
        
        Args:
            model_key: Model to use for the agent
            tool_groups: List of tool groups to include (default: CORE only)
            include_all_sections: Whether to include all prompt sections
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
        
        # Log configuration
        self.logger.info(f"LangChain Explainer agent initialized with:")
        self.logger.info(f"  Model: {self.model_key}")
        self.logger.info(f"  Tool groups: {[g.value for g in self.tool_groups]}")
        self.logger.info(f"  Tools: {[tool.name for tool in self.tools]}")
        self.logger.info(f"  Include all sections: {self.include_all_sections}")

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
                max_iterations=10
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

    def explain_change_sync(self, prompt: str, metric_details: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously explain a change using the LangChain agent."""
        try:
            # Create execution trace callback
            execution_callback = ExecutionTraceCallback()
            self.logger.info("Created ExecutionTraceCallback")
            
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

            return { 
                "success": True, 
                "explanation": cleaned_response if cleaned_response else response,
                "execution_trace": execution_callback.execution_trace
            }
        except Exception as e:
            self.logger.error(f"Error in explain_change_sync: {e}")
            return { "success": False, "error": str(e) }
    
    async def explain_change_streaming(self, prompt: str, metric_details: Dict[str, Any] = None):
        """Stream explanations using direct LLM streaming for real token-by-token output."""
        try:
            self.logger.info(f"=== Starting explain_change_streaming ===")
            self.logger.info(f"Prompt: {prompt}")
            
            # Create execution trace callback
            execution_callback = ExecutionTraceCallback()
            self.logger.info(f"Created execution callback: {execution_callback}")
            
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
            
            # Use direct LLM streaming for real token-by-token output
            try:
                # Get the LLM from the agent
                llm = self.llm
                self.logger.info(f"Using LLM: {type(llm)}")
                
                # Create a proper streaming callback that inherits from BaseCallbackHandler
                class StreamingCallback(BaseCallbackHandler):
                    def __init__(self):
                        super().__init__()
                        self.tokens = []
                        self.logger = logging.getLogger(__name__)
                    
                    def on_llm_new_token(self, token: str, **kwargs):
                        self.tokens.append(token)
                        self.logger.info(f"Received token: {token}")
                
                streaming_callback = StreamingCallback()
                
                # Set up the agent with streaming callback
                if hasattr(self.agent_executor, 'callbacks'):
                    self.agent_executor.callbacks = [execution_callback, streaming_callback]
                
                # Use the agent's invoke method but with streaming enabled
                # We'll use astream_events to get the actual streaming with version parameter
                async for event in self.agent_executor.astream_events({
                    "input": prompt,
                    "chat_history": self.messages[:-1]
                }, version="v1"):
                    self.logger.info(f"=== Processing event ===")
                    self.logger.info(f"Event type: {event.get('event')}")
                    self.logger.info(f"Event: {event}")
                    
                    # Handle LLM streaming events - this is where we get real token-by-token streaming
                    if event.get('event') == 'on_llm_new_token':
                        try:
                            token = event.get('data', {}).get('token', '')
                            if token:
                                # Check if this token contains internal agent data patterns
                                if any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']):
                                    self.logger.info(f"Skipping token with internal data: {token}")
                                    continue
                                
                                self.logger.info(f"Yielding token: {token}")
                                yield f"data: {json.dumps({'content': token})}\n\n"
                                response_content += token
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
                                # Check if this token contains internal agent data patterns
                                if any(pattern in token.lower() for pattern in ['agent_scratchpad', 'intermediate_steps', 'chat_history', 'log=']):
                                    self.logger.info(f"Skipping token with internal data: {token}")
                                    continue
                                
                                self.logger.info(f"Yielding token from chat model: {token}")
                                yield f"data: {json.dumps({'content': token})}\n\n"
                                response_content += token
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
                            
                            tool_call_data = {
                                'tool_call_complete': tool_name, 
                                'success': True, 
                                'response': str(output),
                                'tool_id': tool_id
                            }
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
                    
                    # Also check execution trace for any tool calls that might not be caught by events
                    current_trace_count = len(execution_callback.execution_trace)
                    if current_trace_count > 0:
                        self.logger.info(f"Processing {current_trace_count} tool traces from execution callback")
                        
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
                                    
                                    tool_call_data = {
                                        'tool_call_complete': tool_name, 
                                        'success': success, 
                                        'response': output,
                                        'tool_id': tool_id
                                    }
                                    self.logger.info(f"Yielding tool_call_complete from trace: {tool_call_data}")
                                    yield f"data: {json.dumps(tool_call_data)}\n\n"
                                    
                                    # Remove from in-progress tracking
                                    if tool_name in tool_calls_in_progress:
                                        del tool_calls_in_progress[tool_name]
                                        
                            except Exception as e:
                                self.logger.error(f"Error processing trace {i}: {e}")
                                continue
                            
            except Exception as e:
                self.logger.error(f"Error in streaming loop: {e}")
                self.logger.error(f"Error type: {type(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Fallback: try non-streaming approach
                try:
                    self.logger.info("Trying fallback non-streaming approach...")
                    result = self.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": self.messages[:-1]
                    })
                    
                    response_content = result.get("output", "")
                    if response_content:
                        # Use the cleaning method to extract clean response
                        cleaned_response = self._extract_clean_response(response_content)
                        if cleaned_response:
                            yield f"data: {json.dumps({'content': cleaned_response})}\n\n"
                        
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
                    yield f"data: {json.dumps({'error': f'Streaming failed: {str(e)}. Fallback also failed: {str(fallback_error)}'})}\n\n"
            
            # Add assistant message to conversation history (use cleaned content)
            cleaned_response = self._extract_clean_response(response_content)
            self.add_message("assistant", cleaned_response if cleaned_response else response_content)
            
            # Send completion signal
            self.logger.info("Sending completion signal")
            yield f"data: {json.dumps({'completed': True})}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error in explain_change_streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
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
            
            # Continue with normal string cleaning
            # Remove any JSON-like structures that contain internal agent data
            cleaned = re.sub(r'\{[^}]*agent_scratchpad[^}]*\}', '', output)
            cleaned = re.sub(r'\{[^}]*intermediate_steps[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*steps[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*message_log[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*tool_call_id[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*tool_calls[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*tool_call_chunks[^}]*\}', '', cleaned)
            
            # Remove tool response data that might be mixed in
            cleaned = re.sub(r'\{[^}]*status[^}]*success[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*results[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*response[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*data[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*shape[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*columns[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*queryURL[^}]*\}', '', cleaned)
            
            # Remove any remaining JSON-like structures that might contain tool data
            cleaned = re.sub(r'\{[^}]*"status"[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*"results"[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*"response"[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*"data"[^}]*\}', '', cleaned)
            
            # Remove any remaining tool call artifacts
            cleaned = re.sub(r'Tool Call Details', '', cleaned)
            cleaned = re.sub(r'Status: [^\\n]*', '', cleaned)
            cleaned = re.sub(r'Arguments:', '', cleaned)
            cleaned = re.sub(r'Response:', '', cleaned)
            
            # More aggressive cleaning for specific patterns
            # Remove patterns like {'agent_scratchpad': []}u?', 'chat_history': [...], 'intermediate_steps': []
            cleaned = re.sub(r"\{'agent_scratchpad': \[\]\}[^}]*", '', cleaned)
            cleaned = re.sub(r"'chat_history': \[[^\]]*\]", '', cleaned)
            cleaned = re.sub(r"'intermediate_steps': \[\]", '', cleaned)
            cleaned = re.sub(r"'agent_scratchpad': \[\]", '', cleaned)
            
            # Remove log= artifacts and response duplication
            cleaned = re.sub(r'log="[^"]*"', '', cleaned)
            cleaned = re.sub(r'log=\s*"[^"]*"', '', cleaned)
            cleaned = re.sub(r'log=\s*[^\s]*', '', cleaned)
            # More comprehensive log pattern removal
            cleaned = re.sub(r'log=\s*"[^"]*"', '', cleaned)
            cleaned = re.sub(r'log=\s*[^\s]*', '', cleaned)
            # Remove any remaining log= patterns with different spacing
            cleaned = re.sub(r'log\s*=\s*"[^"]*"', '', cleaned)
            cleaned = re.sub(r'log\s*=\s*[^\s]*', '', cleaned)
            # Additional comprehensive log pattern removal
            cleaned = re.sub(r'log\s*=\s*[^,\s]*', '', cleaned)  # Match until comma or whitespace
            cleaned = re.sub(r'log\s*=\s*[^}\s]*', '', cleaned)  # Match until closing brace or whitespace
            cleaned = re.sub(r'log\s*=\s*[^)\s]*', '', cleaned)  # Match until closing parenthesis or whitespace
            # Remove trailing quotes and braces that might be artifacts
            cleaned = re.sub(r'"[^"]*"\s*$', '', cleaned)
            cleaned = re.sub(r'\}\s*$', '', cleaned)
            # Remove duplicated lines (common pattern where the same line appears twice)
            lines = cleaned.split('\n')
            unique_lines = []
            for line in lines:
                if line.strip() and line not in unique_lines:  # Don't strip for comparison
                    unique_lines.append(line)  # Keep original line with whitespace
            cleaned = '\n'.join(unique_lines)
            
            # Remove ToolAgentAction patterns
            cleaned = re.sub(r'ToolAgentAction\([^)]*\)', '', cleaned)
            cleaned = re.sub(r'\[ToolAgentAction\([^\]]*\)\]', '', cleaned)
            
            # Remove tool response data patterns (the long JSON responses from tools)
            cleaned = re.sub(r'\{[^}]*"status": "success"[^}]*"results"[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*"status": "success"[^}]*"response"[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\{[^}]*"results": \{[^}]*"response"[^}]*\}', '', cleaned)
            
            # Remove any remaining quotes and brackets that might be artifacts
            cleaned = re.sub(r"^\s*['\"]\s*", '', cleaned)  # Remove leading quotes
            cleaned = re.sub(r"\s*['\"]\s*$", '', cleaned)  # Remove trailing quotes
            cleaned = re.sub(r"^\s*\{\s*", '', cleaned)     # Remove leading braces
            cleaned = re.sub(r"\s*\}\s*$", '', cleaned)     # Remove trailing braces
            
            # Clean up whitespace and normalize, but preserve newlines for Markdown
            # First, normalize line endings
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
            # Then clean up multiple spaces within lines, but preserve newlines and leading spaces for Markdown
            lines = cleaned.split('\n')
            cleaned_lines = []
            for line in lines:
                # Preserve leading spaces for Markdown (important for lists, code blocks, etc.)
                leading_spaces = len(line) - len(line.lstrip())
                # Clean up multiple spaces within each line, but preserve the leading spaces
                cleaned_line = line[:leading_spaces] + re.sub(r'[ \t]+', ' ', line[leading_spaces:].strip())
                if cleaned_line.strip():  # Only add non-empty lines
                    cleaned_lines.append(cleaned_line)
            cleaned = '\n'.join(cleaned_lines)
            
            # Remove duplicated phrases at the end (common pattern where the same phrase appears twice)
            # This handles cases like "feel free to ask!indings, feel free to ask!"
            # But be more careful to preserve Markdown structure
            lines = cleaned.split('\n')
            if len(lines) > 3:  # Only process if there are enough lines
                # Look for repeated lines at the end
                for i in range(len(lines) - 2, 0, -1):
                    end_lines = lines[i:]
                    if end_lines == lines[:len(end_lines)]:
                        # Found duplicate lines at the end, remove them
                        cleaned = '\n'.join(lines[:i])
                        break
            
            # Additional cleaning for any remaining log artifacts
            # Remove any text that looks like logging output
            cleaned = re.sub(r'log\s*=\s*"[^"]*"', '', cleaned)
            cleaned = re.sub(r'log\s*=\s*[^\s]*', '', cleaned)
            # Final aggressive log pattern removal
            cleaned = re.sub(r'log\s*=\s*[^,\s}]*', '', cleaned)  # Match until comma, whitespace, or closing brace
            cleaned = re.sub(r'log\s*=\s*[^)\s}]*', '', cleaned)  # Match until parenthesis, whitespace, or closing brace
            # Remove return_values patterns
            cleaned = re.sub(r'return_values\s*=\s*\{[^}]*\}', '', cleaned)
            cleaned = re.sub(r'return_values\s*=\s*[^\\s]*', '', cleaned)
            # Remove any remaining quotes at the end that might be artifacts
            cleaned = re.sub(r'"[^"]*"\s*$', '', cleaned)
            # Remove any remaining braces at the end
            cleaned = re.sub(r'\}\s*$', '', cleaned)
            # Remove any remaining backslashes or escape characters
            cleaned = re.sub(r'\\[^\\]*$', '', cleaned)
            
            # Handle the specific pattern mentioned by the user
            # Remove duplicated text at the end like "feel free to ask!indings, feel free to ask!"
            # This pattern suggests the response got duplicated with some corruption
            if '!' in cleaned and cleaned.count('!') > 1:
                # Split by exclamation marks and look for duplication
                parts = cleaned.split('!')
                if len(parts) > 2:
                    # Check if the last part is a duplicate of an earlier part
                    last_part = parts[-1].strip()
                    for i in range(len(parts) - 2, 0, -1):
                        if parts[i].strip() == last_part:
                            # Found a duplicate, remove everything after the first occurrence
                            cleaned = '!'.join(parts[:i+1])
                            break
            
            # If the cleaned result is mostly JSON or technical data, return empty
            if len(cleaned) < 50 and ('{' in cleaned or '}' in cleaned or 'status' in cleaned.lower()):
                return ""
            
            # Final check: remove any log= data that might be at the very end
            if 'log=' in cleaned:
                # Split by 'log=' and take only the part before it
                parts = cleaned.split('log=')
                if len(parts) > 1:
                    cleaned = parts[0].rstrip()
            
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
    
    def update_tool_groups(self, new_tool_groups: List[ToolGroup]):
        """Update the tool groups and recreate the agent."""
        self.tool_groups = new_tool_groups
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent(metric_details={})
        self.logger.info(f"Updated tool groups to: {[g.value for g in self.tool_groups]}")

def create_explainer_agent(
    model_key: Optional[str] = None,
    tool_groups: Optional[List[ToolGroup]] = None,
    include_all_sections: bool = False
) -> LangChainExplainerAgent:
    """
    Factory function to create a new LangChain explainer agent instance.
    
    Args:
        model_key: Model to use for the agent
        tool_groups: List of tool groups to include (default: CORE only)
        include_all_sections: Whether to include all prompt sections
        
    Returns:
        Configured LangChainExplainerAgent instance
    """
    return LangChainExplainerAgent(
        model_key=model_key,
        tool_groups=tool_groups,
        include_all_sections=include_all_sections
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

#!/usr/bin/env python3
"""
Test script for improved session logging functionality.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the ai directory to the Python path so we can import modules
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_session_logging():
    """Test the improved session logging functionality."""
    try:
        from agents.langchain_agent.explainer_agent import LangChainExplainerAgent
        from agents.langchain_agent.config.tool_config import ToolGroup
        
        logger.info("=== Testing Improved Session Logging ===")
        
        # Create agent with session logging enabled
        agent = LangChainExplainerAgent(
            model_key="gpt-4o",
            tool_groups=[ToolGroup.CORE],
            enable_session_logging=True
        )
        
        logger.info("Created agent with session logging enabled")
        
        # Test a simple query that should trigger tool calls
        test_query = "What are the current crime trends in San Francisco?"
        
        logger.info(f"Running test query: {test_query}")
        
        # Run the query
        result = agent.explain_change_sync(
            test_query,
            metric_details={}
        )
        
        # Check if we got a session ID
        session_id = result.get('session_id')
        if not session_id:
            logger.error("No session ID returned in result")
            return False
        
        logger.info(f"Session ID: {session_id}")
        
        # Check if session log file was created
        logs_dir = Path(__file__).parent.parent.parent / 'logs' / 'sessions'
        session_file = logs_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.error(f"Session log file not found: {session_file}")
            return False
        
        logger.info(f"Session log file created: {session_file}")
        
        # Load and analyze the session log
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Check for intermediate responses
        intermediate_responses = session_data.get('intermediate_responses', [])
        logger.info(f"Found {len(intermediate_responses)} intermediate responses")
        
        for i, response in enumerate(intermediate_responses):
            logger.info(f"  Response {i+1}: {response['content'][:100]}...")
            logger.info(f"    Timestamp: {response['timestamp']}")
        
        # Check for tool calls
        tool_calls = session_data.get('tool_calls', [])
        logger.info(f"Found {len(tool_calls)} tool calls")
        
        for i, tool_call in enumerate(tool_calls):
            logger.info(f"  Tool {i+1}: {tool_call['tool_name']}")
            logger.info(f"    Success: {tool_call['success']}")
            logger.info(f"    Duration: {tool_call['execution_time_ms']}ms")
        
        # Check conversation structure
        conversation = session_data.get('conversation', [])
        logger.info(f"Found {len(conversation)} conversation messages")
        
        for i, msg in enumerate(conversation):
            logger.info(f"  Message {i+1}: {msg['role']} - {msg['content'][:50]}...")
        
        # Verify no duplicate final responses
        assistant_messages = [msg for msg in conversation if msg['role'] == 'assistant']
        if len(assistant_messages) > len(intermediate_responses):
            logger.warning("Possible duplicate assistant messages detected")
        else:
            logger.info("No duplicate assistant messages detected")
        
        logger.info("=== Session Logging Test Completed Successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Error testing session logging: {e}", exc_info=True)
        return False

async def test_streaming_session_logging():
    """Test session logging with streaming responses."""
    try:
        from agents.langchain_agent.explainer_agent import LangChainExplainerAgent
        from agents.langchain_agent.config.tool_config import ToolGroup
        
        logger.info("=== Testing Streaming Session Logging ===")
        
        # Create agent with session logging enabled
        agent = LangChainExplainerAgent(
            model_key="gpt-4o",
            tool_groups=[ToolGroup.CORE],
            enable_session_logging=True
        )
        
        logger.info("Created agent with session logging enabled")
        
        # Test a simple query with streaming
        test_query = "What are the housing permit trends?"
        
        logger.info(f"Running streaming test query: {test_query}")
        
        # Run the streaming query
        session_id = None
        response_chunks = []
        
        async for chunk in agent.explain_change_streaming(
            test_query,
            metric_details={},
            session_id="test-streaming-session"
        ):
            if chunk:
                # Parse the chunk
                try:
                    if chunk.startswith("data: "):
                        data_str = chunk[6:]  # Remove "data: " prefix
                        if data_str.strip():
                            data = json.loads(data_str)
                            
                            # Capture session ID
                            if 'session_id' in data:
                                session_id = data['session_id']
                                logger.info(f"Session ID from streaming: {session_id}")
                            
                            # Capture content chunks
                            if 'content' in data:
                                response_chunks.append(data['content'])
                            
                            # Check for completion
                            if data.get('completed'):
                                logger.info("Streaming completed")
                                break
                                
                except json.JSONDecodeError:
                    continue
        
        if not session_id:
            logger.error("No session ID received from streaming")
            return False
        
        logger.info(f"Received {len(response_chunks)} response chunks")
        
        # Check if session log file was created
        logs_dir = Path(__file__).parent.parent.parent / 'logs' / 'sessions'
        session_file = logs_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.error(f"Session log file not found: {session_file}")
            return False
        
        logger.info(f"Session log file created: {session_file}")
        
        # Load and analyze the session log
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Check for intermediate responses
        intermediate_responses = session_data.get('intermediate_responses', [])
        logger.info(f"Found {len(intermediate_responses)} intermediate responses in streaming session")
        
        # Check conversation structure
        conversation = session_data.get('conversation', [])
        logger.info(f"Found {len(conversation)} conversation messages in streaming session")
        
        # Verify tool call integration
        tool_calls = session_data.get('tool_calls', [])
        logger.info(f"Found {len(tool_calls)} tool calls in streaming session")
        
        logger.info("=== Streaming Session Logging Test Completed Successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Error testing streaming session logging: {e}", exc_info=True)
        return False

def main():
    """Run all session logging tests."""
    logger.info("Starting session logging tests...")
    
    # Test sync session logging
    sync_success = test_session_logging()
    
    # Test streaming session logging
    streaming_success = asyncio.run(test_streaming_session_logging())
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Sync session logging: {'PASS' if sync_success else 'FAIL'}")
    logger.info(f"Streaming session logging: {'PASS' if streaming_success else 'FAIL'}")
    
    if sync_success and streaming_success:
        logger.info("All tests passed! Session logging is working correctly.")
        return 0
    else:
        logger.error("Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())

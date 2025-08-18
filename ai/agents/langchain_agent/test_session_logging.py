#!/usr/bin/env python3
"""
Test script for enhanced session logging in LangChain explainer agent.
"""
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

from .explainer_agent import LangChainExplainerAgent
from .config.tool_config import ToolGroup

def test_session_logging():
    """Test the session logging functionality."""
    print("Testing LangChain Explainer Agent with session logging...")
    
    try:
        # Create agent with session logging enabled
        agent = LangChainExplainerAgent(
            model_key="gpt-4o",  # Use a valid model for testing
            tool_groups=[ToolGroup.CORE],
            include_all_sections=False,
            enable_session_logging=True
        )
        
        print("Agent created successfully")
        
        # Test with a simple prompt
        test_prompt = "What is the weather like in San Francisco?"
        metric_details = {}
        
        print(f"Testing with prompt: {test_prompt}")
        
        # Run the agent
        result = agent.explain_change_sync(test_prompt, metric_details)
        
        print("Agent execution completed")
        print(f"Success: {result.get('success')}")
        print(f"Session ID: {result.get('session_id')}")
        
        if result.get('success'):
            print(f"Response length: {len(result.get('explanation', ''))}")
        else:
            print(f"Error: {result.get('error')}")
        
        # Check if session log was created
        if result.get('session_id'):
            logs_dir = ai_dir / 'logs' / 'sessions'
            session_files = list(logs_dir.glob(f"*{result['session_id'][:8]}*.json"))
            
            if session_files:
                print(f"Session log created: {session_files[0]}")
                
                # Read and display a summary of the log
                import json
                with open(session_files[0], 'r') as f:
                    session_data = json.load(f)
                
                print("\nSession Summary:")
                print(f"  Session ID: {session_data.get('session_id')}")
                print(f"  Model: {session_data.get('model')}")
                print(f"  User Input: {session_data.get('user_input')[:100]}...")
                print(f"  Tool Calls: {len(session_data.get('tool_calls', []))}")
                print(f"  Success: {session_data.get('success')}")
                print(f"  Total Execution Time: {session_data.get('total_execution_time_ms')}ms")
                
                # Show tool call details
                tool_calls = session_data.get('tool_calls', [])
                if tool_calls:
                    print("\nTool Calls:")
                    for i, tc in enumerate(tool_calls):
                        print(f"  {i+1}. {tc.get('tool_name')} - Success: {tc.get('success')} - Duration: {tc.get('execution_time_ms')}ms")
                        if tc.get('error_message'):
                            print(f"     Error: {tc.get('error_message')}")
            else:
                print("No session log file found")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_session_logging()
    if success:
        print("\n✅ Session logging test completed successfully!")
    else:
        print("\n❌ Session logging test failed!")
        sys.exit(1)

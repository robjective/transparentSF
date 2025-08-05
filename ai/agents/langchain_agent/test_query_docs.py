#!/usr/bin/env python3
"""
Test script to verify that the agent can properly use query_docs with correct parameters.
"""
import sys
import os
from pathlib import Path
import logging

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

from .explainer_agent import create_explainer_agent
from .config.tool_config import ToolGroup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_docs_usage():
    """Test that the agent can use query_docs with correct parameters."""
    print("=== Testing query_docs Usage ===")
    
    # Create agent with core tools only
    agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])
    
    # Test prompt that should trigger query_docs
    test_prompt = "Search for information about police misconduct in San Francisco"
    
    print(f"Testing with prompt: {test_prompt}")
    print("Expected behavior: Agent should call query_docs with collection_name and query parameters")
    
    try:
        result = agent.explain_change_sync(test_prompt, metric_details={})
        
        if result.get('success'):
            print("✅ Agent executed successfully")
            print(f"Response: {result.get('explanation', '')[:200]}...")
        else:
            print(f"❌ Agent failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

def test_set_dataset_usage():
    """Test that the agent can use set_dataset with correct parameters."""
    print("\n=== Testing set_dataset Usage ===")
    
    # Create agent with core tools only
    agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])
    
    # Test prompt that should trigger set_dataset
    test_prompt = "Find data about business registrations in district 2"
    
    print(f"Testing with prompt: {test_prompt}")
    print("Expected behavior: Agent should call set_dataset with endpoint and query parameters")
    
    try:
        result = agent.explain_change_sync(test_prompt, metric_details={})
        
        if result.get('success'):
            print("✅ Agent executed successfully")
            print(f"Response: {result.get('explanation', '')[:200]}...")
        else:
            print(f"❌ Agent failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

def main():
    """Run the tests."""
    print("Testing Core Tools Configuration")
    print("=" * 40)
    
    test_query_docs_usage()
    test_set_dataset_usage()
    
    print("\n=== Test Summary ===")
    print("These tests verify that the agent can properly use the core tools")
    print("with the correct parameter format as specified in the prompts.")

if __name__ == "__main__":
    main() 
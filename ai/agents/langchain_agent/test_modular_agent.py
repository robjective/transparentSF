#!/usr/bin/env python3
"""
Test script for the modular LangChain Explainer Agent.
Demonstrates different tool group configurations and their effects.
"""
import sys
import os
from pathlib import Path
import logging

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

from .explainer_agent import (
    create_explainer_agent, 
    get_available_tool_groups, 
    validate_tool_configuration
)
from .config.tool_config import ToolGroup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tool_group_configurations():
    """Test different tool group configurations."""
    print("=== Testing Modular Explainer Agent ===\n")
    
    # Show available tool groups
    print("Available Tool Groups:")
    available_groups = get_available_tool_groups()
    for group_name, tools in available_groups.items():
        print(f"  {group_name}: {len(tools)} tools")
        for tool in tools[:3]:  # Show first 3 tools
            print(f"    - {tool}")
        if len(tools) > 3:
            print(f"    ... and {len(tools) - 3} more")
    print()
    
    # Test 1: Core tools only
    print("=== Test 1: Core Tools Only ===")
    agent1 = create_explainer_agent(tool_groups=[ToolGroup.CORE])
    config1 = agent1.get_configuration_info()
    print(f"Model: {config1['model']}")
    print(f"Tool groups: {config1['tool_groups']}")
    print(f"Tools: {config1['tools']}")
    print(f"Prompt sections: {config1['required_prompt_sections']}")
    print()
    
    # Test 2: Core + Metrics tools
    print("=== Test 2: Core + Metrics Tools ===")
    agent2 = create_explainer_agent(tool_groups=[ToolGroup.CORE, ToolGroup.METRICS])
    config2 = agent2.get_configuration_info()
    print(f"Tool groups: {config2['tool_groups']}")
    print(f"Tools: {config2['tools']}")
    print(f"Prompt sections: {config2['required_prompt_sections']}")
    print()
    
    # Test 3: Core + Data Analysis + Visualization
    print("=== Test 3: Core + Data Analysis + Visualization ===")
    agent3 = create_explainer_agent(tool_groups=[
        ToolGroup.CORE, 
        ToolGroup.DATA_ANALYSIS, 
        ToolGroup.VISUALIZATION
    ])
    config3 = agent3.get_configuration_info()
    print(f"Tool groups: {config3['tool_groups']}")
    print(f"Tools: {config3['tools']}")
    print(f"Prompt sections: {config3['required_prompt_sections']}")
    print()
    
    # Test 4: All tools
    print("=== Test 4: All Tools ===")
    agent4 = create_explainer_agent(tool_groups=[
        ToolGroup.CORE, 
        ToolGroup.METRICS, 
        ToolGroup.DATA_ANALYSIS, 
        ToolGroup.VISUALIZATION
    ])
    config4 = agent4.get_configuration_info()
    print(f"Tool groups: {config4['tool_groups']}")
    print(f"Tools: {config4['tools']}")
    print(f"Prompt sections: {config4['required_prompt_sections']}")
    print()
    
    # Test 5: Core tools with all prompt sections
    print("=== Test 5: Core Tools + All Prompt Sections ===")
    agent5 = create_explainer_agent(
        tool_groups=[ToolGroup.CORE],
        include_all_sections=True
    )
    config5 = agent5.get_configuration_info()
    print(f"Tool groups: {config5['tool_groups']}")
    print(f"Tools: {config5['tools']}")
    print(f"Include all sections: {config5['include_all_sections']}")
    print()

def test_tool_validation():
    """Test tool validation functionality."""
    print("=== Testing Tool Validation ===\n")
    
    # Test validation for different configurations
    test_configs = [
        [ToolGroup.CORE],
        [ToolGroup.CORE, ToolGroup.METRICS],
        [ToolGroup.CORE, ToolGroup.DATA_ANALYSIS, ToolGroup.VISUALIZATION],
        [ToolGroup.CORE, ToolGroup.METRICS, ToolGroup.DATA_ANALYSIS, ToolGroup.VISUALIZATION]
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"Configuration {i}: {[g.value for g in config]}")
        validation = validate_tool_configuration(config)
        print(f"  Available tools: {len(validation['available'])}")
        print(f"  Missing tools: {len(validation['missing'])}")
        if validation['missing']:
            print(f"  Missing: {validation['missing']}")
        print()

def test_agent_reconfiguration():
    """Test dynamic reconfiguration of the agent."""
    print("=== Testing Agent Reconfiguration ===\n")
    
    # Create agent with core tools only
    agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])
    print("Initial configuration:")
    config = agent.get_configuration_info()
    print(f"  Tools: {config['tools']}")
    
    # Update to include metrics tools
    print("\nUpdating to include metrics tools...")
    agent.update_tool_groups([ToolGroup.CORE, ToolGroup.METRICS])
    config = agent.get_configuration_info()
    print(f"  Tools: {config['tools']}")
    
    # Update to include visualization tools
    print("\nUpdating to include visualization tools...")
    agent.update_tool_groups([ToolGroup.CORE, ToolGroup.VISUALIZATION])
    config = agent.get_configuration_info()
    print(f"  Tools: {config['tools']}")
    print()

def main():
    """Run all tests."""
    try:
        test_tool_group_configurations()
        test_tool_validation()
        test_agent_reconfiguration()
        print("=== All tests completed successfully! ===")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 
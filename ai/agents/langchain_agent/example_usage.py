#!/usr/bin/env python3
"""
Example usage of the modular LangChain Explainer Agent.
This script demonstrates how to create and use agents with different tool configurations.
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

def example_basic_usage():
    """Example of basic agent usage with core tools only."""
    print("=== Example 1: Basic Usage (Core Tools Only) ===")
    
    # Create agent with just core tools (simplified set)
    agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])
    
    # Get configuration info
    config = agent.get_configuration_info()
    print(f"Agent created with {len(config['tools'])} tools:")
    for tool in config['tools']:
        print(f"  - {tool}")
    
    print(f"Prompt sections: {config['required_prompt_sections']}")
    print()

def example_analysis_agent():
    """Example of agent with analysis capabilities."""
    print("=== Example 2: Analysis Agent ===")
    
    # Create agent with core and analysis tools
    agent = create_explainer_agent(tool_groups=[ToolGroup.CORE, ToolGroup.ANALYSIS])
    
    config = agent.get_configuration_info()
    print(f"Agent created with {len(config['tools'])} tools")
    print(f"Tool groups: {config['tool_groups']}")
    print(f"Prompt sections: {config['required_prompt_sections']}")
    print()

def example_data_analysis_agent():
    """Example of agent with data analysis capabilities."""
    print("=== Example 3: Data Analysis Agent ===")
    
    # Create agent with core, data analysis, and visualization tools
    agent = create_explainer_agent(tool_groups=[
        ToolGroup.CORE, 
        ToolGroup.DATA_ANALYSIS, 
        ToolGroup.VISUALIZATION
    ])
    
    config = agent.get_configuration_info()
    print(f"Agent created with {len(config['tools'])} tools")
    print(f"Tool groups: {config['tool_groups']}")
    print(f"Prompt sections: {config['required_prompt_sections']}")
    print()

def example_dynamic_reconfiguration():
    """Example of dynamically reconfiguring an agent."""
    print("=== Example 4: Dynamic Reconfiguration ===")
    
    # Start with core tools only
    agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])
    config = agent.get_configuration_info()
    print(f"Initial: {len(config['tools'])} tools")
    
    # Add metrics tools
    agent.update_tool_groups([ToolGroup.CORE, ToolGroup.METRICS])
    config = agent.get_configuration_info()
    print(f"After adding metrics: {len(config['tools'])} tools")
    
    # Add visualization tools
    agent.update_tool_groups([ToolGroup.CORE, ToolGroup.METRICS, ToolGroup.VISUALIZATION])
    config = agent.get_configuration_info()
    print(f"After adding visualization: {len(config['tools'])} tools")
    print()

def example_custom_model():
    """Example of using a custom model."""
    print("=== Example 5: Custom Model ===")
    
    # Create agent with custom model (if available)
    try:
        agent = create_explainer_agent(
            model_key="gpt-4",
            tool_groups=[ToolGroup.CORE, ToolGroup.METRICS]
        )
        config = agent.get_configuration_info()
        print(f"Agent created with model: {config['model']}")
        print(f"Tools: {len(config['tools'])}")
    except Exception as e:
        print(f"Could not create agent with custom model: {e}")
    print()

def example_all_sections():
    """Example of including all prompt sections."""
    print("=== Example 6: All Prompt Sections ===")
    
    # Create agent with core tools but all prompt sections
    agent = create_explainer_agent(
        tool_groups=[ToolGroup.CORE],
        include_all_sections=True
    )
    
    config = agent.get_configuration_info()
    print(f"Agent with all sections: {config['include_all_sections']}")
    print(f"Tools: {len(config['tools'])}")
    print()

def main():
    """Run all examples."""
    print("Modular LangChain Explainer Agent - Usage Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_usage()
        example_analysis_agent()
        example_data_analysis_agent()
        example_dynamic_reconfiguration()
        example_custom_model()
        example_all_sections()
        
        print("=== All examples completed successfully! ===")
        print("\nKey benefits of the modular system:")
        print("1. Selective tool inclusion for focused testing")
        print("2. Dynamic reconfiguration at runtime")
        print("3. Automatic prompt section selection")
        print("4. Better organization and debugging")
        print("5. Flexible deployment options")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    main() 
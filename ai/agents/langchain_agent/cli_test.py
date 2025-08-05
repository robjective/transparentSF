#!/usr/bin/env python3
"""
CLI interface for testing the modular LangChain Explainer Agent.
"""
import sys
import os
from pathlib import Path
import argparse
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

def list_available_configurations():
    """List all available tool groups and their tools."""
    print("Available Tool Groups:")
    print("=====================")
    
    available_groups = get_available_tool_groups()
    for group_name, tools in available_groups.items():
        print(f"\n{group_name.upper()}:")
        for tool in tools:
            print(f"  - {tool}")
    
    print(f"\nTotal tool groups: {len(available_groups)}")
    total_tools = sum(len(tools) for tools in available_groups.values())
    print(f"Total tools: {total_tools}")

def test_configuration(tool_groups, model_key=None, include_all_sections=False):
    """Test a specific tool group configuration."""
    print(f"\nTesting Configuration:")
    print(f"Tool groups: {[g.value for g in tool_groups]}")
    print(f"Model: {model_key or 'default'}")
    print(f"Include all sections: {include_all_sections}")
    print("-" * 50)
    
    try:
        # Validate configuration
        validation = validate_tool_configuration(tool_groups)
        print(f"Validation:")
        print(f"  Available tools: {len(validation['available'])}")
        print(f"  Missing tools: {len(validation['missing'])}")
        
        if validation['missing']:
            print(f"  WARNING: Missing tools: {validation['missing']}")
        
        # Create agent
        agent = create_explainer_agent(
            model_key=model_key,
            tool_groups=tool_groups,
            include_all_sections=include_all_sections
        )
        
        # Get configuration info
        config = agent.get_configuration_info()
        print(f"\nAgent Configuration:")
        print(f"  Model: {config['model']}")
        print(f"  Tool groups: {config['tool_groups']}")
        print(f"  Tools: {config['tools']}")
        print(f"  Prompt sections: {config['required_prompt_sections']}")
        print(f"  Include all sections: {config['include_all_sections']}")
        
        print("\n✅ Configuration test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        return False

def interactive_test():
    """Run an interactive test session."""
    print("\nInteractive Test Session")
    print("=======================")
    
    # Show available groups
    available_groups = get_available_tool_groups()
    print("\nAvailable tool groups:")
    for i, group_name in enumerate(available_groups.keys(), 1):
        print(f"  {i}. {group_name}")
    
    # Get user selection
    try:
        selection = input("\nEnter tool group numbers (comma-separated, e.g., 1,2,3): ").strip()
        group_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        
        group_names = list(available_groups.keys())
        selected_groups = []
        
        for idx in group_indices:
            if 0 <= idx < len(group_names):
                group_name = group_names[idx]
                # Convert to ToolGroup enum
                for group in ToolGroup:
                    if group.value == group_name:
                        selected_groups.append(group)
                        break
        
        if not selected_groups:
            print("No valid tool groups selected.")
            return
        
        # Test the configuration
        test_configuration(selected_groups)
        
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test the modular LangChain Explainer Agent")
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available tool groups and tools"
    )
    parser.add_argument(
        "--test", 
        nargs="+", 
        choices=[g.value for g in ToolGroup],
        help="Test specific tool groups"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model key to use for testing"
    )
    parser.add_argument(
        "--all-sections", 
        action="store_true", 
        help="Include all prompt sections"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run interactive test session"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_configurations()
    
    elif args.test:
        # Convert group names to ToolGroup enums
        tool_groups = []
        for group_name in args.test:
            for group in ToolGroup:
                if group.value == group_name:
                    tool_groups.append(group)
                    break
        
        if tool_groups:
            test_configuration(tool_groups, args.model, args.all_sections)
        else:
            print("No valid tool groups specified.")
    
    elif args.interactive:
        interactive_test()
    
    else:
        # Default: show available configurations
        list_available_configurations()
        print("\nUse --help for usage options.")

if __name__ == "__main__":
    main() 
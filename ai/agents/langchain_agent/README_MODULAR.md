# Modular LangChain Explainer Agent

This directory contains a modular implementation of the LangChain Explainer Agent that allows selective inclusion of tools and prompt sections for easier testing and configuration.

## Overview

The modular system consists of several key components:

1. **Tool Configuration** (`config/tool_config.py`) - Defines tool groups and their metadata
2. **Tool Factory** (`tools/tool_factory.py`) - Dynamically creates LangChain tools
3. **Modular Prompts** (`prompts/modular_prompts.py`) - Builds system prompts with selective sections
4. **Explainer Agent** (`explainer_agent.py`) - Main agent class using the modular system

## Tool Groups

The system organizes tools into logical groups:

### Core Tools (`ToolGroup.CORE`)
Essential tools for basic data operations:
- `set_dataset` - Set dataset for analysis by querying DataSF
- `query_docs` - Search for additional context in documentation

### Analysis Tools (`ToolGroup.ANALYSIS`)
Tools for anomaly analysis and investigation:
- `get_notes` - Get summary of available analysis and documentation
- `get_dashboard_metric` - Retrieve dashboard metric data containing anomalies
- `query_anomalies_db` - Query anomalies from database
- `get_anomaly_details` - Get detailed anomaly information
- `get_charts_for_review` - Get available charts
- `get_dataset_columns` - Get column information

### Metrics Tools (`ToolGroup.METRICS`)
Tools for managing and querying metrics:
- `query_metrics` - Search and filter metrics
- `get_metric_details` - Get metric details

- `get_dashboard_metrics` - Get dashboard metrics
- `create_new_metric` - Create new metrics
- `edit_metric` - Edit existing metrics
- `disable_metric` / `enable_metric` - Manage metric status
- `get_metrics_overview` - Get metrics summary
- `find_metrics_by_endpoint` - Find metrics by endpoint

### Data Analysis Tools (`ToolGroup.DATA_ANALYSIS`)
Tools for data querying and analysis:
- `set_dataset` - Query DataSF datasets
- `get_map_by_id` - Retrieve maps by ID
- `get_recent_maps` - Get recent maps

### Visualization Tools (`ToolGroup.VISUALIZATION`)
Tools for creating visualizations:
- `generate_map` - Create map visualizations

## Usage

### Basic Usage

```python
from ai.agents.langchain_agent.explainer_agent import create_explainer_agent
from ai.agents.langchain_agent.config.tool_config import ToolGroup

# Create agent with core tools only
agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])

# Create agent with core and analysis tools
agent = create_explainer_agent(tool_groups=[ToolGroup.CORE, ToolGroup.ANALYSIS])

# Create agent with all tools
agent = create_explainer_agent(tool_groups=[
    ToolGroup.CORE, 
    ToolGroup.ANALYSIS,
    ToolGroup.METRICS, 
    ToolGroup.DATA_ANALYSIS, 
    ToolGroup.VISUALIZATION
])
```

### Advanced Configuration

```python
# Create agent with custom model and all prompt sections
agent = create_explainer_agent(
    model_key="gpt-4",
    tool_groups=[ToolGroup.CORE, ToolGroup.METRICS],
    include_all_sections=True
)

# Get configuration information
config = agent.get_configuration_info()
print(f"Tools: {config['tools']}")
print(f"Prompt sections: {config['required_prompt_sections']}")

# Update tool groups dynamically
agent.update_tool_groups([ToolGroup.CORE, ToolGroup.VISUALIZATION])
```

### Testing

Use the CLI interface for testing different configurations:

```bash
# List all available tool groups
python -m ai.agents.langchain_agent.cli_test --list

# Test specific tool groups
python -m ai.agents.langchain_agent.cli_test --test core metrics

# Test with custom model
python -m ai.agents.langchain_agent.cli_test --test core visualization --model gpt-4

# Run interactive test session
python -m ai.agents.langchain_agent.cli_test --interactive
```

### Programmatic Testing

```python
from ai.agents.langchain_agent.explainer_agent import (
    get_available_tool_groups, 
    validate_tool_configuration
)

# Get available tool groups
groups = get_available_tool_groups()
print(groups)

# Validate configuration
validation = validate_tool_configuration([ToolGroup.CORE, ToolGroup.METRICS])
print(f"Available: {validation['available']}")
print(f"Missing: {validation['missing']}")
```

## Prompt Sections

The system automatically includes relevant prompt sections based on the selected tools:

- **persona** - Core personality and tone (always included)
- **task** - Main objectives and output format (always included)
- **workflow** - Step-by-step analysis workflow
- **categories** - Category-specific best practices
- **charts** - Chart generation rules
- **core_tools** - Primary tool instructions
- **set_dataset** - DataSF querying instructions
- **generate_map** - Map creation instructions
- **datasf_maps** - Map examples
- **metrics_tools** - Metrics management instructions

## Benefits

1. **Selective Testing** - Test specific tool groups without loading everything
2. **Reduced Complexity** - Smaller prompts for focused testing
3. **Dynamic Configuration** - Change tools at runtime
4. **Better Organization** - Tools grouped by functionality
5. **Easier Debugging** - Isolate issues to specific tool groups
6. **Flexible Deployment** - Deploy agents with different capabilities

## Migration from Old System

The old system used a monolithic approach with all tools and prompts included. The new system:

1. **Backward Compatible** - Can still include all tools and sections
2. **Gradual Migration** - Start with core tools, add others as needed
3. **Better Testing** - Test individual components in isolation
4. **Cleaner Architecture** - Separation of concerns between tools, prompts, and agent

## Example Migration

```python
# Old way (still works)
from ai.agents.langchain_agent.explainer_agent import create_explainer_agent
agent = create_explainer_agent()  # Uses all tools and sections

# New way (modular)
from ai.agents.langchain_agent.explainer_agent import create_explainer_agent
from ai.agents.langchain_agent.config.tool_config import ToolGroup

# Start with core tools only
agent = create_explainer_agent(tool_groups=[ToolGroup.CORE])

# Add more tools as needed
agent.update_tool_groups([ToolGroup.CORE, ToolGroup.METRICS])
```

## Troubleshooting

### Missing Tools
If tools are missing, check:
1. Tool function imports in `tool_factory.py`
2. Tool definitions in `tool_config.py`
3. Function availability in the main tools directory

### Import Errors
Ensure proper path setup:
```python
import sys
from pathlib import Path
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))
```

### Validation Errors
Use the validation functions to check configurations:
```python
validation = validate_tool_configuration(tool_groups)
if validation['missing']:
    print(f"Missing tools: {validation['missing']}")
``` 
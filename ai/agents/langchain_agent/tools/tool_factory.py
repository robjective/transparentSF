"""
Tool factory for creating LangChain tools dynamically based on configuration.
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the parent directory to sys.path for absolute imports
current_dir = Path(__file__).parent
ai_dir = current_dir.parent.parent
sys.path.insert(0, str(ai_dir))

from langchain.tools import Tool, StructuredTool
from ..config.tool_config import ToolGroup, ToolDefinition, tool_config

# Import metrics tools from the local tools directory
from .explainer_metrics_tools import (
    query_metrics, get_metric_details, get_dashboard_metrics,
    create_new_metric, edit_metric, disable_metric, enable_metric,
    get_metrics_overview, find_metrics_by_endpoint
)

# Import visualization tools
from .map_generation_tools import generate_map_tool, get_map_by_id_tool, get_recent_maps_tool

# Import new clean tools
from .explainer_tools import (
    set_dataset_tool, query_docs_tool,
    get_notes_tool, get_dashboard_metric_tool, query_anomalies_db_tool,
    get_anomaly_details_tool, get_dataset_columns_tool, get_charts_for_review_tool
)

logger = logging.getLogger(__name__)

class ToolFactory:
    """Factory for creating LangChain tools based on configuration."""
    
    def __init__(self):
        self.available_functions = self._get_available_functions()
        self.logger = logging.getLogger(__name__)
    
    def _get_available_functions(self) -> Dict[str, Any]:
        """Get all available tool functions."""
        return {
            # Core tools (simplified set) - using new clean tools
            "set_dataset": set_dataset_tool,
            "query_docs": query_docs_tool,
            
            # Analysis tools
            "get_notes": get_notes_tool,
            "get_dashboard_metric": get_dashboard_metric_tool,
            "query_anomalies_db": query_anomalies_db_tool,
            "get_anomaly_details": get_anomaly_details_tool,
            "get_charts_for_review": get_charts_for_review_tool,
            "get_dataset_columns": get_dataset_columns_tool,
            
            # Metrics tools
            "query_metrics": query_metrics,
            "get_metric_details": get_metric_details,
            "get_dashboard_metrics": get_dashboard_metrics,
            "create_new_metric": create_new_metric,
            "edit_metric": edit_metric,
            "disable_metric": disable_metric,
            "enable_metric": enable_metric,
            "get_metrics_overview": get_metrics_overview,
            "find_metrics_by_endpoint": find_metrics_by_endpoint,
            
            # Data analysis tools
            "get_map_by_id": get_map_by_id_tool,
            "get_recent_maps": get_recent_maps_tool,
            
            # Visualization tools
            "generate_map": generate_map_tool,
        }
    
    def create_tools_for_groups(self, groups: List[ToolGroup]) -> List[Tool]:
        """
        Create LangChain tools for the specified tool groups.
        
        Args:
            groups: List of tool groups to include
            
        Returns:
            List of LangChain Tool objects
        """
        if not tool_config.validate_tool_groups(groups):
            raise ValueError(f"Invalid tool groups: {groups}")
        
        tool_definitions = tool_config.get_tools_for_groups(groups)
        tools = []
        
        for tool_def in tool_definitions:
            if tool_def.name in self.available_functions:
                # Create the LangChain tool - all new tools use StructuredTool
                tool = StructuredTool.from_function(
                    func=self.available_functions[tool_def.name],
                    name=tool_def.name,
                    description=tool_def.description
                )
                tools.append(tool)
                self.logger.info(f"Created tool: {tool_def.name}")
            else:
                self.logger.warning(f"Function not found for tool: {tool_def.name}")
        
        self.logger.info(f"Created {len(tools)} tools for groups: {[g.value for g in groups]}")
        return tools
    
    def get_tool_names_for_groups(self, groups: List[ToolGroup]) -> List[str]:
        """Get the names of tools that would be created for the specified groups."""
        return tool_config.get_tool_names_for_groups(groups)
    
    def get_required_prompt_sections(self, groups: List[ToolGroup]) -> List[str]:
        """Get the required prompt sections for the specified tool groups."""
        return tool_config.get_required_prompt_sections(groups)
    
    def list_available_tools(self) -> Dict[str, List[str]]:
        """List all available tools grouped by their tool groups."""
        result = {}
        for group in ToolGroup:
            if group in tool_config.tool_groups:
                result[group.value] = [tool.name for tool in tool_config.tool_groups[group]]
        return result
    
    def validate_tool_availability(self, groups: List[ToolGroup]) -> Dict[str, List[str]]:
        """
        Validate that all tools in the specified groups are available.
        
        Returns:
            Dict with 'available' and 'missing' tool lists
        """
        tool_definitions = tool_config.get_tools_for_groups(groups)
        available = []
        missing = []
        
        for tool_def in tool_definitions:
            if tool_def.name in self.available_functions:
                available.append(tool_def.name)
            else:
                missing.append(tool_def.name)
        
        return {
            "available": available,
            "missing": missing
        }

# Global instance
tool_factory = ToolFactory() 
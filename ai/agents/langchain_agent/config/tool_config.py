"""
Tool configuration for the LangChain Explainer Agent.
Defines tool groups and their associated prompt sections for modular agent creation.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ToolGroup(Enum):
    """Available tool groups for the explainer agent."""
    CORE = "core"
    ANALYSIS = "analysis"
    METRICS = "metrics"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    DOCUMENTATION = "documentation"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class ToolDefinition:
    """Definition of a single tool with its metadata."""
    name: str
    function: Any
    description: str
    group: ToolGroup
    required_prompt_sections: List[str]
    examples: Optional[List[str]] = None

class ToolConfig:
    """Configuration manager for tool groups and their associated prompts."""
    
    def __init__(self):
        self.tool_groups: Dict[ToolGroup, List[ToolDefinition]] = {}
        self.prompt_sections: Dict[str, str] = {}
        self._initialize_tool_groups()
        self._initialize_prompt_sections()
    
    def _initialize_tool_groups(self):
        """Initialize the available tool groups."""
        # Core tools - always available (simplified set)
        self.tool_groups[ToolGroup.CORE] = [
            ToolDefinition(
                name="set_dataset",
                function=None,
                description="Set dataset for analysis by querying DataSF",
                group=ToolGroup.CORE,
                required_prompt_sections=["core_tools", "categories"],
                examples=[
                    "set_dataset(context_variables, 'wg3w-h783', 'SELECT incident_category, COUNT(*) as count WHERE date_trunc_ym(report_datetime) = date_trunc_ym(CURRENT_DATE) GROUP BY incident_category')",
                    "Use this to query DataSF datasets for analysis"
                ]
            ),
            ToolDefinition(
                name="query_docs",
                function=None,
                description="Search for additional context in documentation",
                group=ToolGroup.CORE,
                required_prompt_sections=["core_tools", "categories"],
                examples=[
                    "query_docs(context_variables, collection_name='SFPublicData', query='information related to crime trends')",
                    "Use this to find domain-specific information that might explain anomalies"
                ]
            ),
            ToolDefinition(
                name="get_dataset_columns",
                function=None,
                description="Get column information for a dataset endpoint",
                group=ToolGroup.CORE,
                required_prompt_sections=["core_tools"],
                examples=[
                    "get_dataset_columns(endpoint='wg3w-h783')",
                    "Use this to explore what columns are available in a specific dataset"
                ]
            )
        ]
        
        # Analysis tools (previously part of core)
        self.tool_groups[ToolGroup.ANALYSIS] = [
            ToolDefinition(
                name="get_notes",
                function=None,
                description="Get summary of available analysis and documentation",
                group=ToolGroup.ANALYSIS,
                required_prompt_sections=["core_tools", "workflow"],
                examples=[
                    "Always start with get_notes() to see what analysis is available",
                    "Use get_notes() to understand what data and documentation you have access to"
                ]
            ),
            ToolDefinition(
                name="get_dashboard_metric",
                function=None,
                description="Retrieve dashboard metric data with analysis files (monthly, weekly, annual, YTD)",
                group=ToolGroup.ANALYSIS,
                required_prompt_sections=["core_tools", "analysis_tools", "workflow"],
                examples=[
                    "get_dashboard_metric(context_variables, district_number=0, metric_id=123)",
                    "Use this to get the dashboard metric data and all available analysis files for a specific metric"
                ]
            ),
            ToolDefinition(
                name="query_anomalies_db",
                function=None,
                description="Query anomalies directly from the PostgreSQL database",
                group=ToolGroup.ANALYSIS,
                required_prompt_sections=["core_tools", "analysis_tools", "workflow"],
                examples=[
                    "query_anomalies_db(context_variables, query_type='by_metric_id', metric_id=123, district_filter=2, only_anomalies=True)",
                    "Use this to find anomalies for a specific metric and district"
                ]
            ),
            ToolDefinition(
                name="get_anomaly_details",
                function=None,
                description="Get detailed information about a specific anomaly by ID",
                group=ToolGroup.ANALYSIS,
                required_prompt_sections=["core_tools", "workflow"],
                examples=[
                    "get_anomaly_details(context_variables, anomaly_id=123)",
                    "Use this to get complete information about a specific anomaly"
                ]
            ),
            ToolDefinition(
                name="get_charts_for_review",
                function=None,
                description="Get available charts for newsletter inclusion review",
                group=ToolGroup.ANALYSIS,
                required_prompt_sections=["core_tools", "analysis_tools", "charts", "workflow"],
                examples=[
                    "get_charts_for_review(context_variables, limit=20, days_back=30, district_filter='2', metric_id='3')",
                    "Use this to find relevant charts and maps that support your explanations"
                ]
            )
        ]
        
        # Metrics management tools
        self.tool_groups[ToolGroup.METRICS] = [
            ToolDefinition(
                name="query_metrics",
                function=None,
                description="Search and filter metrics in the database",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "query_metrics(context_variables, category='crime', search_term='police', active_only=True)",
                    "Use this to find metrics by category, search terms, or other filters"
                ]
            ),
            ToolDefinition(
                name="get_metric_details",
                function=None,
                description="Get detailed information about a specific metric",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "get_metric_details(context_variables, metric_identifier=1)",
                    "Use this to get complete information about a metric by ID or key"
                ]
            ),
            ToolDefinition(
                name="get_dashboard_metrics",
                function=None,
                description="Get all metrics that are shown on the dashboard",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "get_dashboard_metrics(context_variables)",
                    "Use this to get all metrics currently displayed on the dashboard"
                ]
            ),
            ToolDefinition(
                name="create_new_metric",
                function=None,
                description="Add a new metric to the database",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "create_new_metric(context_variables, name='Vehicle Thefts', key='vehicle_thefts', category='crime', endpoint='wg3w-h783')",
                    "Use this to add new metrics to the system"
                ]
            ),
            ToolDefinition(
                name="edit_metric",
                function=None,
                description="Update an existing metric",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "edit_metric(context_variables, metric_identifier=1, updates={'summary': 'Updated summary'})",
                    "Use this to modify existing metrics"
                ]
            ),
            ToolDefinition(
                name="disable_metric",
                function=None,
                description="Deactivate a metric (soft delete)",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "disable_metric(context_variables, metric_identifier='metric_key')",
                    "Use this to disable a metric without deleting it"
                ]
            ),
            ToolDefinition(
                name="enable_metric",
                function=None,
                description="Reactivate a previously disabled metric",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "enable_metric(context_variables, metric_identifier='metric_key')",
                    "Use this to reactivate a disabled metric"
                ]
            ),
            ToolDefinition(
                name="get_metrics_overview",
                function=None,
                description="Get summary statistics about the metrics system",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "get_metrics_overview(context_variables)",
                    "Use this to get high-level information about total metrics, active metrics, etc."
                ]
            ),
            ToolDefinition(
                name="find_metrics_by_endpoint",
                function=None,
                description="Find all metrics using a specific DataSF endpoint",
                group=ToolGroup.METRICS,
                required_prompt_sections=["metrics_tools"],
                examples=[
                    "find_metrics_by_endpoint(context_variables, endpoint='wg3w-h783')",
                    "Use this to see what metrics are built on a particular dataset"
                ]
            )
        ]
        
        # Data analysis tools
        self.tool_groups[ToolGroup.DATA_ANALYSIS] = [
            ToolDefinition(
                name="set_dataset",
                function=None,
                description="Set dataset for analysis by querying DataSF",
                group=ToolGroup.DATA_ANALYSIS,
                required_prompt_sections=["set_dataset"],
                examples=[
                    "set_dataset(context_variables, endpoint='wg3w-h783', query='SELECT incident_category, COUNT(*) as count WHERE date_trunc_ym(report_datetime) = date_trunc_ym(CURRENT_DATE) GROUP BY incident_category')",
                    "Use this to query DataSF datasets for analysis"
                ]
            ),
            ToolDefinition(
                name="get_map_by_id",
                function=None,
                description="Retrieve a previously created map by ID",
                group=ToolGroup.DATA_ANALYSIS,
                required_prompt_sections=["map_generation"],
                examples=[
                    "get_map_by_id(context_variables, map_id=123)",
                    "Use this to retrieve the details of a map that was previously created"
                ]
            ),
            ToolDefinition(
                name="get_recent_maps",
                function=None,
                description="Get a list of recently created maps",
                group=ToolGroup.DATA_ANALYSIS,
                required_prompt_sections=["map_generation"],
                examples=[
                    "get_recent_maps(context_variables, limit=10, map_type='supervisor_district')",
                    "Use this to see what maps have been created recently"
                ]
            )
        ]
        
        # Visualization tools
        self.tool_groups[ToolGroup.VISUALIZATION] = [
            ToolDefinition(
                name="generate_map",
                function=None,
                description="Create a map visualization for geographic data using the TransparentSF map generation system",
                group=ToolGroup.VISUALIZATION,
                required_prompt_sections=["map_generation"],
                examples=[
                    "generate_map(context_variables, map_title='Crime by District', map_type='supervisor_district', map_metadata={'description': 'Crime incidents by district'})",
                    "Use this to create maps showing geographic patterns in the data"
                ]
            ),
            ToolDefinition(
                name="generate_map_with_query",
                function=None,
                description="Generate a map by querying DataSF and creating a map visualization in one step",
                group=ToolGroup.VISUALIZATION,
                required_prompt_sections=["map_generation"],
                examples=[
                    "generate_map_with_query(endpoint='wg3w-h783', query='SELECT supervisor_district, COUNT(*) as value WHERE date_trunc_ym(report_datetime) = date_trunc_ym(CURRENT_DATE) GROUP BY supervisor_district', map_title='Crime by District', map_type='supervisor_district')",
                    "Use this to query data and create maps in one step, preferred over separate set_dataset + generate_map calls"
                ]
            )
        ]
        
        # Documentation tools (empty for now, but initialized to prevent validation errors)
        self.tool_groups[ToolGroup.DOCUMENTATION] = []
        
        # Anomaly detection tools (empty for now, but initialized to prevent validation errors)
        self.tool_groups[ToolGroup.ANOMALY_DETECTION] = []
    
    def _initialize_prompt_sections(self):
        """Initialize the prompt sections that correspond to tool groups."""
        # These will be populated from the explainer_prompts.py file
        self.prompt_sections = {
            "persona": "Core personality and tone instructions for Seymour",
            "task": "Main objectives and output format",
            "workflow": "Step-by-step analysis workflow",
            "categories": "Specific guidance for different data categories",
            "charts": "Rules for selecting and referencing charts",
            "core_tools": "Primary tools for data analysis and anomaly investigation",
            "set_dataset": "Instructions for querying DataSF datasets",
            "map_generation": "Comprehensive instructions for creating maps using the TransparentSF system",
            "metrics_tools": "Tools for managing and querying metrics database"
        }
    
    def get_tools_for_groups(self, groups: List[ToolGroup]) -> List[ToolDefinition]:
        """Get all tools for the specified groups."""
        tools = []
        for group in groups:
            if group in self.tool_groups:
                tools.extend(self.tool_groups[group])
        return tools
    
    def get_required_prompt_sections(self, groups: List[ToolGroup]) -> List[str]:
        """Get all required prompt sections for the specified tool groups."""
        sections = set()
        for group in groups:
            if group in self.tool_groups:
                for tool in self.tool_groups[group]:
                    sections.update(tool.required_prompt_sections)
        return list(sections)
    
    def get_tool_names_for_groups(self, groups: List[ToolGroup]) -> List[str]:
        """Get just the tool names for the specified groups."""
        return [tool.name for tool in self.get_tools_for_groups(groups)]
    
    def validate_tool_groups(self, groups: List[ToolGroup]) -> bool:
        """Validate that the specified tool groups exist."""
        # Check if all groups are valid ToolGroup enum values and have been initialized
        return all(group in self.tool_groups for group in groups)

# Global instance
tool_config = ToolConfig() 
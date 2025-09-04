"""
Modular prompt builder for the LangChain Explainer Agent.
Allows selective inclusion of prompt sections based on tool groups.
"""
from typing import Dict, List, Any, Optional
import logging

from .explainer_prompts import PROMPT_SECTIONS

logger = logging.getLogger(__name__)

class ModularPromptBuilder:
    """Builder for creating modular system prompts based on tool groups."""
    
    def __init__(self):
        self.prompt_sections = PROMPT_SECTIONS
        self.logger = logging.getLogger(__name__)
    
    def build_system_prompt(
        self, 
        required_sections: List[str],
        metric_details: Optional[Dict[str, Any]] = None,
        include_all_sections: bool = False
    ) -> str:
        """
        Build a system prompt with only the required sections.
        
        Args:
            required_sections: List of section keys to include
            metric_details: Optional metric details to include in context
            include_all_sections: If True, include all sections regardless of requirements
            
        Returns:
            Complete system prompt string
        """
        # Always include core sections
        core_sections = ["persona"]
        sections_to_include = set(core_sections)
        
        if include_all_sections:
            # Include all available sections
            sections_to_include.update(self.prompt_sections.keys())
        else:
            # Include only required sections plus core sections
            sections_to_include.update(required_sections)
        
        # Build the prompt parts
        prompt_parts = []
        
        # Add metric context if provided
        if metric_details and isinstance(metric_details, dict):
            metric_context = self._build_metric_context(metric_details)
            if metric_context:  # Only add if we got a valid context
                prompt_parts.append(metric_context)
        
        # Add each required section
        for section_key in sections_to_include:
            if section_key in self.prompt_sections:
                section_content = self.prompt_sections[section_key]['content']
                prompt_parts.append(section_content)
                self.logger.debug(f"Included prompt section: {section_key}")
            else:
                self.logger.warning(f"Prompt section not found: {section_key}")
        
        # Join all parts with double newlines
        complete_prompt = "\n\n".join(prompt_parts)
        
        self.logger.info(f"Built system prompt with {len(sections_to_include)} sections: {list(sections_to_include)}")
        return complete_prompt
    
    def _build_metric_context(self, metric_details: Dict[str, Any]) -> str:
        """Build the metric context section for the prompt."""
        # Ensure metric_details is a dictionary
        if not isinstance(metric_details, dict):
            self.logger.warning(f"metric_details is not a dictionary: {type(metric_details)}")
            return ""
        
        # Handle different field naming conventions
        name = metric_details.get("name", metric_details.get("metric_name", ""))
        description = metric_details.get("description", "")
        summary = metric_details.get("summary", "")
        map_type = metric_details.get("map_type", "")
        
        # Get district information for proper context
        district = metric_details.get("district", "0")
        district_display = "citywide" if district == "0" else f"district {district}"
        
        # Determine if this is an anomaly or metric based on available fields
        has_anomaly_fields = any(field in metric_details for field in ["recent_mean", "comparison_mean", "difference", "percent_change"])
        has_metric_fields = any(field in metric_details for field in ["metric_id", "metric_name"])
        
        if has_anomaly_fields:
            # This is an anomaly - use anomaly-specific language
            recent_mean = metric_details.get("recent_mean", "")
            comparison_mean = metric_details.get("comparison_mean", "")
            difference = metric_details.get("difference", "")
            percent_change = metric_details.get("percent_change", "")
            direction = metric_details.get("direction", "")
            period_type = metric_details.get("period_type", "month")
            
            return f"""You are being asked to explain an ANOMALY in the following metric for {district_display}:
- Metric Name: {name}
- Metric ID: {metric_details.get("metric_id", "N/A")}
- Period Type: {period_type}
- Change: {direction} from {comparison_mean} to {recent_mean} (difference: {difference}, {percent_change}%)
- District: {district_display}

This is an ANOMALY analysis - you are explaining why this specific change occurred in this time period."""
        
        elif has_metric_fields:
            # This is a metric - use metric-specific language
            return f"""You are being asked to explain a change in the following METRIC for {district_display}:
- Name: {name}
- Description: {description}  
- Summary: {summary}
- Map Type: {map_type}
- District: {district_display}

This is a METRIC analysis - you are explaining changes in this metric over time."""
        
        else:
            # Generic case - try to provide useful context
            return f"""You are being asked to explain a change for {district_display}:
- Name: {name}
- District: {district_display}

Please analyze this change using the available tools and provide a comprehensive explanation."""
    
    def get_available_sections(self) -> Dict[str, Dict[str, str]]:
        """Get all available prompt sections with their metadata."""
        return self.prompt_sections
    
    def validate_sections(self, sections: List[str]) -> Dict[str, List[str]]:
        """
        Validate that the specified sections exist.
        
        Returns:
            Dict with 'valid' and 'invalid' section lists
        """
        valid = []
        invalid = []
        
        for section in sections:
            if section in self.prompt_sections:
                valid.append(section)
            else:
                invalid.append(section)
        
        return {
            "valid": valid,
            "invalid": invalid
        }
    
    def get_section_summary(self) -> Dict[str, str]:
        """Get a summary of all available sections."""
        return {
            key: data['description'] 
            for key, data in self.prompt_sections.items()
        }

# Global instance
prompt_builder = ModularPromptBuilder() 
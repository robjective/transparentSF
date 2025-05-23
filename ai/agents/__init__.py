"""
AI Agents package for TransparentSF.

This package contains clean, reusable AI agents that can be instantiated
from anywhere in the application.
"""

from .explainer_agent import (
    ExplainerAgent,
    create_explainer_agent,
    explain_metric_change,
    EXPLAINER_INSTRUCTIONS
)

__all__ = [
    'ExplainerAgent',
    'create_explainer_agent', 
    'explain_metric_change',
    'EXPLAINER_INSTRUCTIONS'
] 
"""
Model factory for creating agents with different models.
Works with existing Swarm framework and tool structure.
"""

import logging
from typing import Dict, Any, List, Optional
from swarm import Agent
from .config.models import ModelConfig, get_model_config, get_default_model
import os

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating agents with different model configurations."""
    
    def __init__(self):
        self._clients = {}
        self.logger = logger
    
    def get_client(self, provider: str):
        """Get or create client for the specified provider."""
        if provider not in self._clients:
            try:
                if provider == "openai":
                    from openai import OpenAI
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY not found in environment")
                    self._clients[provider] = OpenAI(api_key=api_key)
                    
                elif provider == "anthropic":
                    import anthropic
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        raise ValueError("ANTHROPIC_API_KEY not found in environment")
                    self._clients[provider] = anthropic.Anthropic(api_key=api_key)
                    
                elif provider == "google":
                    import google.generativeai as genai
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY not found in environment")
                    genai.configure(api_key=api_key)
                    self._clients[provider] = genai
                    
                elif provider == "mistral":
                    from mistralai.client import MistralClient
                    api_key = os.getenv("MISTRAL_API_KEY")
                    if not api_key:
                        raise ValueError("MISTRAL_API_KEY not found in environment")
                    self._clients[provider] = MistralClient(api_key=api_key)
                    
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                    
            except ImportError as e:
                raise ImportError(f"Required package for {provider} not installed: {e}")
        
        return self._clients[provider]
    
    def create_agent(self, 
                    model_key: Optional[str] = None,
                    name: str = "Agent",
                    instructions: str = "",
                    functions: Optional[List] = None,
                    context_variables: Optional[Dict[str, Any]] = None,
                    **kwargs) -> Agent:
        """
        Create an agent with the specified model configuration.
        
        Args:
            model_key: Model identifier (e.g., "gpt-4o", "claude-3-sonnet")
            name: Agent name
            instructions: Agent instructions
            functions: List of functions available to the agent
            context_variables: Context variables for the agent
            **kwargs: Additional arguments passed to Agent constructor
        
        Returns:
            Agent instance
        """
        # Use default model if none specified
        if model_key is None:
            model_key = get_default_model()
        
        # Get model configuration
        model_config = get_model_config(model_key)
        
        # Check if model is available
        if not model_config.is_available():
            raise ValueError(f"Model {model_key} is not available. Check API key for {model_config.provider.value}")
        
        # Prepare agent configuration
        agent_config = {
            "name": name,
            "instructions": instructions,
            "functions": functions or [],
            "context_variables": context_variables or {},
            **kwargs
        }
        
        # Add model-specific configuration
        if model_config.provider.value == "openai":
            agent_config["model"] = model_config.model_name
        elif model_config.provider.value == "anthropic":
            agent_config["model"] = model_config.model_name
            # Note: Swarm may need updates to support Anthropic directly
        elif model_config.provider.value == "google":
            agent_config["model"] = model_config.model_name
            # Note: Swarm may need updates to support Google directly
        elif model_config.provider.value == "mistral":
            agent_config["model"] = model_config.model_name
            # Note: Swarm may need updates to support Mistral directly
        
        self.logger.info(f"Creating agent '{name}' with model {model_key} ({model_config.full_name})")
        
        try:
            return Agent(**agent_config)
        except Exception as e:
            self.logger.error(f"Failed to create agent with model {model_key}: {e}")
            raise

# Global factory instance
model_factory = ModelFactory()

def create_agent_with_model(model_key: str, **kwargs) -> Agent:
    """Convenience function to create an agent with a specific model."""
    return model_factory.create_agent(model_key=model_key, **kwargs) 
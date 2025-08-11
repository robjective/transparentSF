"""
LangChain-based model configuration for TransparentSF.
Supports multiple LLM providers with unified interface.
"""
import os
from enum import Enum
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class ModelConfig:
    """Configuration for a specific model."""
    def __init__(self, provider: ModelProvider, model_name: str, config: Dict[str, Any] = None, api_key_env_var: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.config = config or {}
        self.api_key_env_var = api_key_env_var or f"{provider.value.upper()}_API_KEY"

    @property
    def full_name(self) -> str:
        return f"{self.provider.value}/{self.model_name}"

    def get_api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env_var)

    def is_available(self) -> bool:
        return bool(self.get_api_key())

    def create_langchain_llm(self):
        try:
            api_key = self.get_api_key()
            if not api_key:
                raise ValueError(f"API key not found for {self.full_name}")
            
            print(f"Creating LLM for {self.full_name}")
            print(f"Provider: {self.provider}")
            print(f"Model name: {self.model_name}")
            print(f"Config: {self.config}")
            
            if self.provider == ModelProvider.OPENAI:
                return ChatOpenAI(model=self.model_name, api_key=api_key, **self.config)
            elif self.provider == ModelProvider.ANTHROPIC:
                return ChatAnthropic(model=self.model_name, api_key=api_key, **self.config)
            elif self.provider == ModelProvider.GOOGLE:
                print(f"Creating Google LLM with model: {self.model_name}")
                return ChatGoogleGenerativeAI(model=self.model_name, google_api_key=api_key, **self.config)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"Error creating LLM for {self.full_name}: {e}")
            raise

# Predefined model configurations
MODEL_CONFIGS = {
    # OpenAI Models
    "gpt-5": ModelConfig(ModelProvider.OPENAI, "gpt-5"),
    "gpt-4o": ModelConfig(ModelProvider.OPENAI, "gpt-4o"),
    "gpt-4.1": ModelConfig(ModelProvider.OPENAI, "gpt-4.1"),
    "gpt-4": ModelConfig(ModelProvider.OPENAI, "gpt-4"),
    "gpt-4-turbo": ModelConfig(ModelProvider.OPENAI, "gpt-4-turbo-preview"),
    "gpt-3.5-turbo": ModelConfig(ModelProvider.OPENAI, "gpt-3.5-turbo"),
    "gpt-3.5-turbo-16k": ModelConfig(ModelProvider.OPENAI, "gpt-3.5-turbo-16k"),
    # Anthropic Models (Latest versions - updated 2025)
    "claude-opus-4": ModelConfig(ModelProvider.ANTHROPIC, "claude-opus-4-20250514"),
    "claude-sonnet-4": ModelConfig(ModelProvider.ANTHROPIC, "claude-sonnet-4-20250514"),
    "claude-3-7-sonnet": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-7-sonnet-20250219"),
    "claude-3-5-haiku": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-5-haiku-20241022"),
    "claude-3-5-sonnet": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
    "claude-3-haiku": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-haiku-20240307"),
    # Google Models (Latest versions - with specific config for tool calling)
    "gemini-2.5-pro": ModelConfig(ModelProvider.GOOGLE, "gemini-2.5-pro", {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192
    }),
    "gemini-1.5-pro": ModelConfig(ModelProvider.GOOGLE, "gemini-1.5-pro", {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192
    }),
    "gemini-1.5-flash": ModelConfig(ModelProvider.GOOGLE, "gemini-1.5-flash", {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192
    }),
}

def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]

def get_available_models() -> Dict[str, ModelConfig]:
    """Get all available models (those with API keys)."""
    return {key: config for key, config in MODEL_CONFIGS.items() if config.is_available()}

def get_default_model() -> str:
    """Get the default model from environment or fallback."""
    return os.getenv("AGENT_MODEL", "gpt-5")

def create_langchain_llm(model_key: Optional[str] = None):
    """Create a LangChain LLM instance for the specified model."""
    if model_key is None:
        model_key = get_default_model()
    model_config = get_model_config(model_key)
    return model_config.create_langchain_llm()

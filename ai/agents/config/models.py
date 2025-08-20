"""
LangChain-based model configuration for TransparentSF.
Supports multiple LLM providers with unified interface.

Token Limits Configuration:
- OpenAI models: max_tokens=8192 (except GPT-3.5-turbo: 4096)
- Anthropic models: max_tokens=8192
- Google models: max_output_tokens=8192
- All models set to temperature=0.1 for consistency
"""
import os
from enum import Enum
from typing import Dict, Any, Optional, List
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
    # OpenAI Models - with increased token limits for longer responses
    # Note: GPT-5 only supports default temperature (1.0)
    "gpt-5": ModelConfig(ModelProvider.OPENAI, "gpt-5", {
        "max_tokens": 8192
        # temperature parameter removed - GPT-5 only supports default (1.0)
    }),
    "gpt-4o": ModelConfig(ModelProvider.OPENAI, "gpt-4o", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "gpt-4.1": ModelConfig(ModelProvider.OPENAI, "gpt-4.1", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "gpt-4": ModelConfig(ModelProvider.OPENAI, "gpt-4", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "gpt-4-turbo": ModelConfig(ModelProvider.OPENAI, "gpt-4-turbo-preview", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "gpt-3.5-turbo": ModelConfig(ModelProvider.OPENAI, "gpt-3.5-turbo", {
        "max_tokens": 4096,
        "temperature": 0.1
    }),
    "gpt-3.5-turbo-16k": ModelConfig(ModelProvider.OPENAI, "gpt-3.5-turbo-16k", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    # Anthropic Models (Latest versions - updated 2025) - with increased token limits
    "claude-opus-4": ModelConfig(ModelProvider.ANTHROPIC, "claude-opus-4-20250514", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "claude-sonnet-4": ModelConfig(ModelProvider.ANTHROPIC, "claude-sonnet-4-20250514", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "claude-3-7-sonnet": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-7-sonnet-20250219", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "claude-3-5-haiku": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-5-haiku-20241022", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "claude-3-5-sonnet": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
    "claude-3-haiku": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-haiku-20240307", {
        "max_tokens": 8192,
        "temperature": 0.1
    }),
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
    return os.getenv("AGENT_MODEL", "claude-3-7-sonnet")

def create_langchain_llm(model_key: Optional[str] = None, max_tokens: Optional[int] = None):
    """Create a LangChain LLM instance for the specified model.
    
    Args:
        model_key: Model identifier (defaults to AGENT_MODEL env var or claude-3-7-sonnet)
        max_tokens: Override the default max_tokens for this model instance
    """
    if model_key is None:
        model_key = get_default_model()
    model_config = get_model_config(model_key)
    
    # Override max_tokens if specified
    if max_tokens is not None:
        config_copy = model_config.config.copy()
        if model_config.provider == ModelProvider.GOOGLE:
            config_copy['max_output_tokens'] = max_tokens
        else:
            config_copy['max_tokens'] = max_tokens
        # Create a temporary model config with the overridden settings
        temp_model_config = ModelConfig(model_config.provider, model_config.model_name, config_copy, model_config.api_key_env_var)
        return temp_model_config.create_langchain_llm()
    
    return model_config.create_langchain_llm()

def get_default_token_limit(model_key: str) -> int:
    """Get the default token limit for a specific model."""
    model_config = get_model_config(model_key)
    if model_config.provider == ModelProvider.GOOGLE:
        return model_config.config.get('max_output_tokens', 8192)
    else:
        return model_config.config.get('max_tokens', 8192)

def get_fallback_models(preferred_model: str) -> List[str]:
    """Get a list of fallback models when the preferred model fails.
    
    Returns models in order of preference, excluding the failed model.
    """
    # Define fallback order by provider
    fallback_order = [
        "gpt-4o",           # OpenAI GPT-4o (reliable)
        "gpt-4",            # OpenAI GPT-4 (reliable)
        "claude-3-5-sonnet", # Anthropic (if not rate limited)
        "gpt-4-turbo",      # OpenAI GPT-4 Turbo
        "gpt-3.5-turbo",    # OpenAI GPT-3.5 (fastest fallback)
    ]
    
    # Remove the preferred model from fallbacks and filter by availability
    available_models = get_available_models()
    fallbacks = [model for model in fallback_order 
                if model != preferred_model and model in available_models]
    
    return fallbacks

def create_langchain_llm_with_fallback(model_key: Optional[str] = None, max_tokens: Optional[int] = None):
    """Create a LangChain LLM with automatic fallback on errors.
    
    Args:
        model_key: Model identifier (defaults to AGENT_MODEL env var or claude-3-7-sonnet)
        max_tokens: Override the default max_tokens for this model instance
        
    Returns:
        Tuple of (llm_instance, actual_model_used)
    """
    if model_key is None:
        model_key = get_default_model()
    
    # Try the preferred model first
    models_to_try = [model_key] + get_fallback_models(model_key)
    
    last_error = None
    for model in models_to_try:
        try:
            print(f"Attempting to create LLM for {model}...")
            llm = create_langchain_llm(model, max_tokens)
            print(f"Successfully created LLM for {model}")
            return llm, model
        except Exception as e:
            error_str = str(e).lower()
            print(f"Failed to create LLM for {model}: {e}")
            last_error = e
            
            # Check for rate limit errors - skip other models from same provider
            if "rate_limit" in error_str or "rate limit" in error_str:
                print(f"Rate limit detected for {model}, trying different provider...")
                # Skip other models from the same provider
                failed_provider = get_model_config(model).provider
                models_to_try = [m for m in models_to_try if get_model_config(m).provider != failed_provider]
                continue
            
            # For other errors, continue to next model
            continue
    
    # If all models failed, raise the last error
    raise Exception(f"All fallback models failed. Last error: {last_error}")

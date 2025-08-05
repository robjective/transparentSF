#!/usr/bin/env python3
"""
Test script to check available models
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Check environment variables
print("Environment variables:")
print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
print(f"GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}")

# Test model config
try:
    from agents.config.models import get_available_models, MODEL_CONFIGS
    print("\nAll configured models:")
    for key, config in MODEL_CONFIGS.items():
        print(f"  {key}: {config.full_name} ({'Available' if config.is_available() else 'Not available'})")
    
    print("\nAvailable models:")
    available = get_available_models()
    for key, config in available.items():
        print(f"  {key}: {config.full_name}")
        
except Exception as e:
    print(f"Error testing models: {e}") 
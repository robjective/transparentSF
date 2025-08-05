#!/usr/bin/env python3
"""
Test script for the enhanced evaluation system.
Tests the new dashboard functionality and model abstraction.
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_config():
    """Test the model configuration system."""
    print("Testing model configuration...")
    
    try:
        from agents.config.models import get_available_models, get_default_model, get_model_config
        
        # Test default model
        default_model = get_default_model()
        print(f"✓ Default model: {default_model}")
        
        # Test available models
        available_models = get_available_models()
        print(f"✓ Available models: {list(available_models.keys())}")
        
        # Test model config
        if available_models:
            first_model = list(available_models.keys())[0]
            config = get_model_config(first_model)
            print(f"✓ Model config for {first_model}: {config.provider.value}")
        
        return True
    except Exception as e:
        print(f"✗ Model config test failed: {e}")
        return False

def test_model_factory():
    """Test the model factory."""
    print("\nTesting model factory...")
    
    try:
        from agents.model_factory import ModelFactory
        
        factory = ModelFactory()
        print("✓ Model factory created")
        
        # Test getting a client
        client = factory.get_client("openai")
        print("✓ OpenAI client created")
        
        return True
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        return False

def test_explainer_agent():
    """Test the explainer agent with model abstraction."""
    print("\nTesting explainer agent...")
    
    try:
        from agents.explainer_agent import create_explainer_agent
        
        # Test with default model
        agent = create_explainer_agent()
        print("✓ Explainer agent created with default model")
        
        # Test with specific model
        agent = create_explainer_agent(model_key="gpt-4o")
        print("✓ Explainer agent created with gpt-4o")
        
        return True
    except Exception as e:
        print(f"✗ Explainer agent test failed: {e}")
        return False

def test_eval_functions():
    """Test the evaluation functions."""
    print("\nTesting evaluation functions...")
    
    try:
        from evals import run_all_evals, run_model_comparison, TEST_CASES
        
        print(f"✓ Found {len(TEST_CASES)} test cases")
        
        # Test model comparison function exists
        if hasattr(run_model_comparison, '__call__'):
            print("✓ Model comparison function available")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation functions test failed: {e}")
        return False

def test_dashboard_endpoints():
    """Test the dashboard endpoints."""
    print("\nTesting dashboard endpoints...")
    
    try:
        from routes.evals import get_latest_eval_summary, get_eval_history
        
        print("✓ Dashboard endpoints available")
        
        # Test that functions are callable
        if hasattr(get_latest_eval_summary, '__call__'):
            print("✓ Latest eval summary endpoint available")
        
        if hasattr(get_eval_history, '__call__'):
            print("✓ Eval history endpoint available")
        
        return True
    except Exception as e:
        print(f"✗ Dashboard endpoints test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Enhanced Evaluation System")
    print("=" * 50)
    
    tests = [
        test_model_config,
        test_model_factory,
        test_explainer_agent,
        test_eval_functions,
        test_dashboard_endpoints
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced evaluation system is ready.")
        print("\nNext steps:")
        print("1. Start your server: python main.py")
        print("2. Navigate to: http://localhost:8000/backend/evals-interface")
        print("3. Try the new dashboard features:")
        print("   - View overall performance metrics")
        print("   - Run evaluations with different models")
        print("   - Compare model performance")
        print("   - Drill down into evaluation history")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
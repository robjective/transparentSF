#!/usr/bin/env python3
"""
Test script for the enhanced regenerate_explanations_for_report function
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all required imports work correctly"""
    try:
        from ai.monthly_report import regenerate_explanations_for_report
        print("✅ Successfully imported regenerate_explanations_for_report")
        
        from ai.monthly_report import get_perplexity_context
        print("✅ Successfully imported get_perplexity_context")
        
        from ai.tools.generate_report_text import generate_report_text
        print("✅ Successfully imported generate_report_text")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_function_signature():
    """Test that the function has the expected signature"""
    try:
        from ai.monthly_report import regenerate_explanations_for_report
        import inspect
        
        sig = inspect.signature(regenerate_explanations_for_report)
        params = list(sig.parameters.keys())
        
        expected_params = ['filename']
        if params == expected_params:
            print("✅ Function signature is correct")
            return True
        else:
            print(f"❌ Function signature mismatch. Expected {expected_params}, got {params}")
            return False
    except Exception as e:
        print(f"❌ Error checking function signature: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing enhanced regenerate_explanations_for_report function...")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Function Signature Test", test_function_signature),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced function should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
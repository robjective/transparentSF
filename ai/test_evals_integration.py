#!/usr/bin/env python3
"""Test script to verify evals integration is working correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evals import run_all_evals

def test_evals_integration():
    """Test that the evals system is working correctly."""
    print("Testing evals integration...")
    
    try:
        # Run all evals
        results = run_all_evals()
        
        print(f"\nEval Results:")
        print(f"Total tests: {results['total']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        # Print individual test results
        print("\nDetailed Results:")
        for test in results['test_results']:
            status_symbol = "✓" if test['status'] == 'success' else "✗"
            print(f"{status_symbol} {test['test_name']}: {test['query'][:50]}...")
            if test['error_message']:
                print(f"  Error: {test['error_message']}")
        
        print(f"\nSummary log saved to: {results['summary_log_filename']}")
        
        # Return success if all tests passed
        return results['failed'] == 0
        
    except Exception as e:
        print(f"Error running evals: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_evals_integration()
    sys.exit(0 if success else 1) 
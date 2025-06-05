#!/usr/bin/env python3
"""
Test script for the revised monthly report workflow.
Tests each step individually to make debugging easier.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the functions we want to test
from monthly_report import (
    initialize_monthly_reporting_table,
    collect_all_significant_changes,
    store_all_metric_changes,
    generate_explanations,
    get_perplexity_context,
    prioritize_and_activate_items,
    generate_monthly_report
)

def test_step_1_collect_changes(district="0", period_type="month"):
    """Test Step 1: Collect all significant changes"""
    print("\n" + "="*60)
    print("STEP 1: Testing collect_all_significant_changes")
    print("="*60)
    
    try:
        result = collect_all_significant_changes(
            period_type=period_type,
            district=district,
            top_n=20  # Increased to 20 for testing since we're now getting both increases and decreases
        )
        
        print(f"Result status: {result.get('status')}")
        print(f"Total changes: {result.get('total_changes', 0)}")
        print(f"Report ID: {result.get('report_id')}")
        print(f"Inserted IDs: {len(result.get('inserted_ids', []))}")
        
        if result.get('status') == 'success':
            print("✅ Step 1 PASSED")
            return result
        else:
            print(f"❌ Step 1 FAILED: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ Step 1 ERROR: {str(e)}")
        return None

def test_step_2_generate_explanations(inserted_ids):
    """Test Step 2: Generate explanations"""
    print("\n" + "="*60)
    print("STEP 2: Testing generate_explanations")
    print("="*60)
    
    if not inserted_ids:
        print("❌ No inserted IDs to test with")
        return False
    
    # Test with just the first few IDs
    test_ids = inserted_ids[:3]
    print(f"Testing with {len(test_ids)} items: {test_ids}")
    
    try:
        result = generate_explanations(test_ids)
        
        print(f"Result status: {result.get('status')}")
        print(f"Message: {result.get('message', 'No message')}")
        
        if result.get('status') == 'success':
            print("✅ Step 2 PASSED")
            return True
        else:
            print(f"❌ Step 2 FAILED: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Step 2 ERROR: {str(e)}")
        return False

def test_step_3_perplexity_context(report_id):
    """Test Step 3: Get Perplexity context"""
    print("\n" + "="*60)
    print("STEP 3: Testing get_perplexity_context")
    print("="*60)
    
    try:
        # First, get the report items from the database
        from monthly_report import execute_with_connection, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
        import psycopg2.extras
        
        def get_items_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT mr.*, r.id as report_id 
                FROM monthly_reporting mr
                JOIN reports r ON mr.report_id = r.id
                WHERE mr.report_id = %s
                LIMIT 2
            """, (report_id,))
            
            items = cursor.fetchall()
            
            report_data = []
            for item in items:
                report_data.append({
                    "id": item["id"],
                    "metric": item["metric_name"],
                    "metric_id": item["metric_id"],
                    "group": item["group_value"],
                    "recent_mean": item["recent_mean"],
                    "comparison_mean": item["comparison_mean"],
                    "difference": item["difference"],
                    "percent_change": item["percent_change"],
                    "explanation": item["explanation"],
                    "metadata": item["metadata"] if item["metadata"] else {}
                })
            
            cursor.close()
            return report_data
        
        result = execute_with_connection(
            operation=get_items_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] != "success":
            print(f"❌ Failed to get items: {result['message']}")
            return False
        
        report_items = result["result"]
        print(f"Got {len(report_items)} items to test Perplexity with")
        
        if not report_items:
            print("❌ No items found to test with")
            return False
        
        # Test Perplexity context
        context_result = get_perplexity_context(report_items)
        
        print(f"Result status: {context_result.get('status')}")
        if context_result.get('status') == 'success':
            print("✅ Step 3 PASSED")
            return True
        else:
            print(f"❌ Step 3 FAILED: {context_result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Step 3 ERROR: {str(e)}")
        return False

def test_step_4_prioritize_and_activate(report_id, max_items=3):
    """Test Step 4: Prioritize and activate items"""
    print("\n" + "="*60)
    print("STEP 4: Testing prioritize_and_activate_items")
    print("="*60)
    
    try:
        result = prioritize_and_activate_items(report_id, max_items=max_items)
        
        print(f"Result status: {result.get('status')}")
        print(f"Activated count: {result.get('activated_count', 0)}")
        
        if result.get('status') == 'success':
            print("✅ Step 4 PASSED")
            return True
        else:
            print(f"❌ Step 4 FAILED: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Step 4 ERROR: {str(e)}")
        return False

def test_step_5_generate_report(district, report_id):
    """Test Step 5: Generate monthly report"""
    print("\n" + "="*60)
    print("STEP 5: Testing generate_monthly_report")
    print("="*60)
    
    try:
        result = generate_monthly_report(district=district, report_id=report_id)
        
        print(f"Result status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"Report path: {result.get('report_path')}")
            print("✅ Step 5 PASSED")
            return True
        else:
            print(f"❌ Step 5 FAILED: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Step 5 ERROR: {str(e)}")
        return False

def check_database_state(report_id):
    """Check the current state of the database"""
    print("\n" + "="*60)
    print("DATABASE STATE CHECK")
    print("="*60)
    
    try:
        from monthly_report import execute_with_connection, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
        import psycopg2.extras
        
        def check_state_operation(connection):
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Check report
            cursor.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
            report = cursor.fetchone()
            
            # Check monthly_reporting items
            cursor.execute("""
                SELECT id, metric_name, group_value, is_active, 
                       LENGTH(explanation) as expl_len,
                       LENGTH(report_text) as report_len,
                       metadata IS NOT NULL as has_metadata
                FROM monthly_reporting 
                WHERE report_id = %s 
                ORDER BY id
            """, (report_id,))
            items = cursor.fetchall()
            
            cursor.close()
            return {"report": dict(report) if report else None, "items": [dict(item) for item in items]}
        
        result = execute_with_connection(
            operation=check_state_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] == "success":
            data = result["result"]
            report = data["report"]
            items = data["items"]
            
            print(f"Report ID {report_id}:")
            if report:
                print(f"  District: {report['district']}")
                print(f"  Filename: {report['original_filename']}")
                print(f"  Created: {report['created_at']}")
            else:
                print("  Report not found!")
                return
            
            print(f"\nItems ({len(items)} total):")
            active_count = 0
            for item in items:
                status = "ACTIVE" if item['is_active'] else "inactive"
                if item['is_active']:
                    active_count += 1
                print(f"  ID {item['id']}: {item['metric_name']} - {status}")
                print(f"    Explanation: {item['expl_len']} chars")
                print(f"    Report text: {item['report_len']} chars") 
                print(f"    Has metadata: {item['has_metadata']}")
                
            print(f"\nSummary: {active_count}/{len(items)} items are active")
            
    except Exception as e:
        print(f"❌ Database check ERROR: {str(e)}")

def main():
    """Run all tests"""
    print("🧪 MONTHLY REPORT WORKFLOW TEST SCRIPT")
    print("="*60)
    
    # Configuration
    district = "0"  # Citywide
    period_type = "month"
    
    print(f"Testing with district: {district}, period: {period_type}")
    
    # Initialize database
    print("\n📋 Initializing database...")
    if not initialize_monthly_reporting_table():
        print("❌ Failed to initialize database")
        return
    
    # Step 1: Collect changes
    step1_result = test_step_1_collect_changes(district, period_type)
    if not step1_result:
        return
    
    report_id = step1_result.get('report_id')
    inserted_ids = step1_result.get('inserted_ids', [])
    
    # Check database state after step 1
    check_database_state(report_id)
    
    # Step 2: Generate explanations
    if not test_step_2_generate_explanations(inserted_ids):
        print("⚠️  Continuing with Step 3 anyway...")
    
    # Check database state after step 2
    check_database_state(report_id)
    
    # Step 3: Perplexity context
    if not test_step_3_perplexity_context(report_id):
        print("⚠️  Continuing with Step 4 anyway...")
    
    # Check database state after step 3
    check_database_state(report_id)
    
    # Step 4: Prioritize and activate
    if not test_step_4_prioritize_and_activate(report_id, max_items=3):
        return
    
    # Check database state after step 4
    check_database_state(report_id)
    
    # Step 5: Generate report
    if not test_step_5_generate_report(district, report_id):
        return
    
    print("\n🎉 ALL TESTS COMPLETED!")
    print(f"Report ID for further testing: {report_id}")

if __name__ == "__main__":
    main() 
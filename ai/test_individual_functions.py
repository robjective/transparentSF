#!/usr/bin/env python3
"""
Interactive test script for individual monthly report functions.
Run this to test specific functions without running the entire workflow.
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
    generate_monthly_report,
    execute_with_connection,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)

def test_database_connection():
    """Test basic database connection"""
    print("🔌 Testing database connection...")
    
    def test_operation(connection):
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        return result[0] == 1
    
    result = execute_with_connection(
        operation=test_operation,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD
    )
    
    if result["status"] == "success" and result["result"]:
        print("✅ Database connection successful!")
        return True
    else:
        print(f"❌ Database connection failed: {result.get('message', 'Unknown error')}")
        return False

def test_init_table():
    """Test table initialization"""
    print("📋 Testing table initialization...")
    
    try:
        result = initialize_monthly_reporting_table()
        if result:
            print("✅ Table initialization successful!")
            return True
        else:
            print("❌ Table initialization failed!")
            return False
    except Exception as e:
        print(f"❌ Table initialization error: {str(e)}")
        return False

def test_collect_changes():
    """Test collecting significant changes"""
    print("📊 Testing collect_all_significant_changes...")
    
    district = "0"
    period_type = "month"
    
    try:
        result = collect_all_significant_changes(
            period_type=period_type,
            district=district,
            top_n=50  # Increased to 50 for testing since we're now getting both increases and decreases
        )
        
        print(f"Status: {result.get('status')}")
        print(f"Total changes: {result.get('total_changes', 0)}")
        print(f"Report ID: {result.get('report_id')}")
        print(f"Inserted IDs count: {len(result.get('inserted_ids', []))}")
        
        if result.get('status') == 'success':
            print("✅ Collect changes successful!")
            return result
        else:
            print(f"❌ Collect changes failed: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ Collect changes error: {str(e)}")
        return None

def test_explanations(report_id):
    """Test explanation generation for a specific report"""
    print(f"💭 Testing generate_explanations for report {report_id}...")
    
    # Get some IDs from the report
    def get_ids_operation(connection):
        cursor = connection.cursor()
        cursor.execute("""
            SELECT id FROM monthly_reporting 
            WHERE report_id = %s 
            LIMIT 3
        """, (report_id,))
        ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return ids
    
    try:
        result = execute_with_connection(
            operation=get_ids_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] != "success":
            print(f"❌ Failed to get IDs: {result['message']}")
            return False
        
        test_ids = result["result"]
        if not test_ids:
            print("❌ No IDs found to test with")
            return False
        
        print(f"Testing with {len(test_ids)} items: {test_ids}")
        
        explanation_result = generate_explanations(test_ids)
        
        print(f"Status: {explanation_result.get('status')}")
        print(f"Message: {explanation_result.get('message', 'No message')}")
        
        if explanation_result.get('status') == 'success':
            print("✅ Generate explanations successful!")
            return True
        else:
            print(f"❌ Generate explanations failed: {explanation_result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Generate explanations error: {str(e)}")
        return False

def test_prioritization(report_id):
    """Test prioritization and activation"""
    print(f"🎯 Testing prioritize_and_activate_items for report {report_id}...")
    
    try:
        result = prioritize_and_activate_items(report_id, max_items=3)
        
        print(f"Status: {result.get('status')}")
        print(f"Activated count: {result.get('activated_count', 0)}")
        
        if result.get('status') == 'success':
            print("✅ Prioritization successful!")
            return True
        else:
            print(f"❌ Prioritization failed: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Prioritization error: {str(e)}")
        return False

def check_report_status(report_id):
    """Check the status of a specific report"""
    print(f"📋 Checking status of report {report_id}...")
    
    def check_operation(connection):
        import psycopg2.extras
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get report info
        cursor.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
        report = cursor.fetchone()
        
        # Get items summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total_items,
                COUNT(CASE WHEN is_active THEN 1 END) as active_items,
                COUNT(CASE WHEN explanation IS NOT NULL AND explanation != '' THEN 1 END) as with_explanation,
                COUNT(CASE WHEN report_text IS NOT NULL AND report_text != '' THEN 1 END) as with_report_text,
                COUNT(CASE WHEN metadata IS NOT NULL THEN 1 END) as with_metadata
            FROM monthly_reporting 
            WHERE report_id = %s
        """, (report_id,))
        summary = cursor.fetchone()
        
        # Get sample items
        cursor.execute("""
            SELECT id, metric_name, group_value, is_active, 
                   LENGTH(explanation) as expl_len,
                   LENGTH(report_text) as report_len
            FROM monthly_reporting 
            WHERE report_id = %s 
            ORDER BY id 
            LIMIT 10
        """, (report_id,))
        items = cursor.fetchall()
        
        cursor.close()
        return {
            "report": dict(report) if report else None,
            "summary": dict(summary) if summary else None,
            "items": [dict(item) for item in items]
        }
    
    try:
        result = execute_with_connection(
            operation=check_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] == "success":
            data = result["result"]
            report = data["report"]
            summary = data["summary"]
            items = data["items"]
            
            if not report:
                print("❌ Report not found!")
                return False
            
            print(f"Report {report_id}:")
            print(f"  District: {report['district']}")
            print(f"  Created: {report['created_at']}")
            print(f"  Filename: {report['original_filename']}")
            
            if summary:
                print(f"\nSummary:")
                print(f"  Total items: {summary['total_items']}")
                print(f"  Active items: {summary['active_items']}")
                print(f"  With explanation: {summary['with_explanation']}")
                print(f"  With report text: {summary['with_report_text']}")
                print(f"  With metadata: {summary['with_metadata']}")
            
            print(f"\nSample items:")
            for item in items[:5]:  # Show first 5
                status = "ACTIVE" if item['is_active'] else "inactive"
                print(f"  ID {item['id']}: {item['metric_name']} ({status})")
                print(f"    Explanation: {item['expl_len'] or 0} chars")
                print(f"    Report text: {item['report_len'] or 0} chars")
            
            return True
        else:
            print(f"❌ Database error: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Check report error: {str(e)}")
        return False

def list_recent_reports():
    """List recent reports"""
    print("📋 Listing recent reports...")
    
    def list_operation(connection):
        import psycopg2.extras
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cursor.execute("""
            SELECT r.id, r.district, r.created_at, r.original_filename,
                   COUNT(mr.id) as total_items,
                   COUNT(CASE WHEN mr.is_active THEN 1 END) as active_items
            FROM reports r
            LEFT JOIN monthly_reporting mr ON r.id = mr.report_id
            GROUP BY r.id, r.district, r.created_at, r.original_filename
            ORDER BY r.created_at DESC
            LIMIT 10
        """)
        
        reports = cursor.fetchall()
        cursor.close()
        return [dict(report) for report in reports]
    
    try:
        result = execute_with_connection(
            operation=list_operation,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        
        if result["status"] == "success":
            reports = result["result"]
            
            if not reports:
                print("No reports found.")
                return []
            
            print("Recent reports:")
            for report in reports:
                print(f"  ID {report['id']}: District {report['district']} - {report['created_at']}")
                print(f"    Items: {report['total_items']} total, {report['active_items']} active")
                print(f"    File: {report['original_filename']}")
                print()
            
            return reports
        else:
            print(f"❌ Database error: {result['message']}")
            return []
            
    except Exception as e:
        print(f"❌ List reports error: {str(e)}")
        return []

def main():
    """Interactive main function"""
    print("🧪 MONTHLY REPORT FUNCTION TESTER")
    print("="*50)
    
    while True:
        print("\nAvailable tests:")
        print("1. Test database connection")
        print("2. Initialize tables")
        print("3. Test collect changes")
        print("4. Test explanations (need report ID)")
        print("5. Test prioritization (need report ID)")
        print("6. Check report status (need report ID)")
        print("7. List recent reports")
        print("8. Full step-by-step test")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            test_database_connection()
        elif choice == "2":
            test_init_table()
        elif choice == "3":
            result = test_collect_changes()
            if result:
                print(f"\n💡 Tip: Use report ID {result.get('report_id')} for other tests")
        elif choice == "4":
            report_id = input("Enter report ID: ").strip()
            if report_id.isdigit():
                test_explanations(int(report_id))
            else:
                print("❌ Invalid report ID")
        elif choice == "5":
            report_id = input("Enter report ID: ").strip()
            if report_id.isdigit():
                test_prioritization(int(report_id))
            else:
                print("❌ Invalid report ID")
        elif choice == "6":
            report_id = input("Enter report ID: ").strip()
            if report_id.isdigit():
                check_report_status(int(report_id))
            else:
                print("❌ Invalid report ID")
        elif choice == "7":
            list_recent_reports()
        elif choice == "8":
            # Run a complete test
            print("🚀 Running full step-by-step test...")
            
            if not test_database_connection():
                continue
            
            if not test_init_table():
                continue
            
            result = test_collect_changes()
            if not result:
                continue
            
            report_id = result.get('report_id')
            print(f"\n📝 Generated report ID: {report_id}")
            
            check_report_status(report_id)
            
            test_explanations(report_id)
            check_report_status(report_id)
            
            test_prioritization(report_id)
            check_report_status(report_id)
            
            print(f"\n✅ Full test completed! Report ID: {report_id}")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 
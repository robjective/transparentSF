import sys
sys.path.append('ai')
from ai.tools.db_utils import get_postgres_connection
import psycopg2.extras

try:
    conn = get_postgres_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    # Check total active datasets
    cursor.execute('SELECT COUNT(*) as total FROM datasets WHERE is_active = true')
    result = cursor.fetchone()
    print(f'Total active datasets: {result["total"]}')
    
    # Check for jail-related datasets
    cursor.execute("""
        SELECT id, title, endpoint, description 
        FROM datasets 
        WHERE is_active = true 
        AND (LOWER(title) LIKE '%jail%' OR LOWER(title) LIKE '%booking%' OR LOWER(description) LIKE '%jail%' OR LOWER(description) LIKE '%booking%')
        ORDER BY title
    """)
    jail_datasets = cursor.fetchall()
    print(f'\nJail/booking related datasets found: {len(jail_datasets)}')
    for ds in jail_datasets:
        print(f'- ID: {ds["id"]}, Title: {ds["title"]}, Endpoint: {ds["endpoint"]}')
        if ds["description"]:
            print(f'  Description: {ds["description"][:100]}...')
    
    # Show all datasets to understand what's available
    cursor.execute('SELECT id, title, endpoint FROM datasets WHERE is_active = true ORDER BY title')
    all_datasets = cursor.fetchall()
    print(f'\nAll active datasets ({len(all_datasets)} total):')
    for ds in all_datasets:
        print(f'- ID: {ds["id"]}, Title: {ds["title"]}, Endpoint: {ds["endpoint"]}')
        
    cursor.close()
    conn.close()
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc() 
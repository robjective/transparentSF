import sys
sys.path.append('ai')
from ai.tools.db_utils import get_postgres_connection
import psycopg2.extras

try:
    conn = get_postgres_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    # Check the format of rows_updated_at for a few datasets
    cursor.execute("""
        SELECT id, title, rows_updated_at, pg_typeof(rows_updated_at) as data_type
        FROM datasets 
        WHERE is_active = true 
        AND rows_updated_at IS NOT NULL
        ORDER BY id
        LIMIT 10
    """)
    datasets = cursor.fetchall()
    
    print("Sample rows_updated_at values:")
    for ds in datasets:
        print(f"ID: {ds['id']}, Title: {ds['title'][:50]}...")
        print(f"  rows_updated_at: {ds['rows_updated_at']} (type: {ds['data_type']})")
        print(f"  Python type: {type(ds['rows_updated_at'])}")
        print()
        
    cursor.close()
    conn.close()
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc() 
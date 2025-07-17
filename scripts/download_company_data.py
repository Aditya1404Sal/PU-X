import psycopg2
import pandas as pd

# I need to tweak this to include other columns as well.

DB_PARAMS = {
    'dbname': 'Stock',
    'user': 'adity',
    'password': 'qwertypoi',
    'host': 'localhost'
}

# Define your deduplicated SQL query
query = """
SELECT * from companies;
"""

# Connect and execute
try:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query(query, conn)
    df.to_csv("company_data_basic.csv", index=False)
    print("✅ CSV exported successfully: company_data_basic.csv")
except Exception as e:
    print("❌ Error:", e)
finally:
    if conn:
        conn.close()

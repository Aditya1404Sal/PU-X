import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Database connection parameters
DB_PARAMS = {
    'dbname': 'Stock',
    'user': 'adity',
    'password': 'qwertypoi',
    'host': 'localhost'
}

def categorize_market_cap(market_cap):
    """
    Categorize the market cap according to Indian stock market standards
    Large Cap: > 20,000 Crore rupees (> 200 billion)
    Mid Cap: 5,000 - 20,000 Crore rupees (50-200 billion)
    Small Cap: 500 - 5,000 Crore rupees (5-50 billion)
    Micro Cap: < 500 Crore rupees (< 5 billion)
    """
    # Market cap thresholds in rupees
    large_cap_threshold = 200_000_000_000  # 20,000 Crore
    mid_cap_threshold = 50_000_000_000     # 5,000 Crore
    small_cap_threshold = 5_000_000_000    # 500 Crore
    
    if market_cap >= large_cap_threshold:
        return 'Large'
    elif market_cap >= mid_cap_threshold:
        return 'Mid'
    elif market_cap >= small_cap_threshold:
        return 'Small'
    else:
        return 'Micro'  # Includes what would be "Nano" in the original schema

def get_intrinsic_label(symbol, nifty_data):
    """
    Determine intrinsic label based on Nifty index membership
    """
    if symbol in nifty_data['nifty100']:
        return 'IN-L'
    elif symbol in nifty_data['niftymidcap150']:
        return 'IN-M'
    elif symbol in nifty_data['niftysmallcap100']:
        return 'IN-S'
    elif symbol in nifty_data['niftymicrocap250']:
        return 'IN-Mi'
    else:
        # For stocks not in any Nifty index, use market cap category
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Get market cap data
        query = """
        SELECT sym.avg_last_price * com.issued_size AS market_cap
        FROM stock_yearly_metrics sym
        JOIN companies com ON sym.symbol = com.symbol
        WHERE sym.symbol = %s
        """
        
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result and result[0]:
            market_cap = result[0]
            category = categorize_market_cap(market_cap)
            
            if category == 'Large':
                return 'IN-L'
            elif category == 'Mid':
                return 'IN-M'
            elif category == 'Small':
                return 'IN-S'
            else:  # Micro
                return 'IN-Mi'
        
        # Default if no market cap data available
        return 'IN-Mi'

def load_nifty_data():
    """
    Load Nifty index data from CSV files
    """
    nifty_data = {
        'nifty100': set(),
        'niftymidcap150': set(),
        'niftysmallcap100': set(),
        'niftymicrocap250': set()
    }
    
    # Load Nifty 100
    try:
        df = pd.read_csv('nifty100list.csv')
        nifty_data['nifty100'] = set(df['Symbol'].tolist())
    except Exception as e:
        print(f"Error loading nifty100list.csv: {e}")
    
    # Load Nifty Midcap 150
    try:
        df = pd.read_csv('niftymidcap150list.csv')
        nifty_data['niftymidcap150'] = set(df['Symbol'].tolist())
    except Exception as e:
        print(f"Error loading niftymidcap150list.csv: {e}")
    
    # Load Nifty Smallcap 100
    try:
        df = pd.read_csv('niftysmallcap100list.csv')
        nifty_data['niftysmallcap100'] = set(df['Symbol'].tolist())
    except Exception as e:
        print(f"Error loading niftysmallcap100list.csv: {e}")
    
    # Load Nifty Microcap 250
    try:
        df = pd.read_csv('niftymicrocap250_list.csv')
        nifty_data['niftymicrocap250'] = set(df['Symbol'].tolist())
    except Exception as e:
        print(f"Error loading niftymicrocap250_list.csv: {e}")
    
    return nifty_data

def populate_company_categorization():
    """
    Populate the company_categorization table
    """
    # Load Nifty data
    nifty_data = load_nifty_data()
    
    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    try:
        # Get data from companies and stock_yearly_metrics tables
        query = """
        SELECT 
            c.symbol, 
            c.industry, 
            c.face_value, 
            sym.avg_last_price, 
            c.issued_size
        FROM 
            companies c
        JOIN 
            stock_yearly_metrics sym ON c.symbol = sym.symbol
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Prepare data for insertion
        data = []
        for row in rows:
            symbol = row[0]
            industry = row[1] if row[1] else 'Unknown'
            face_value = row[2] if row[2] else 0.0
            avg_last_price = row[3] if row[3] else 0.0
            issued_size = row[4] if row[4] else 0
            
            # Calculate market cap
            market_cap = avg_last_price * issued_size if avg_last_price and issued_size else 0
            
            # Categorize market cap
            market_cap_category = categorize_market_cap(market_cap)
            
            # Get intrinsic label
            intrinsic_label = get_intrinsic_label(symbol, nifty_data)
            
            data.append((
                symbol,
                market_cap_category,
                industry,
                face_value,
                intrinsic_label
            ))
        
        # Insert data into company_categorization table
        insert_query = """
        INSERT INTO company_categorization (
            symbol, 
            market_cap_category, 
            industry, 
            face_value, 
            intrinsic_label
        ) VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
            market_cap_category = EXCLUDED.market_cap_category,
            industry = EXCLUDED.industry,
            face_value = EXCLUDED.face_value,
            intrinsic_label = EXCLUDED.intrinsic_label
        """
        
        execute_values(cursor, insert_query, data)
        conn.commit()
        print(f"Successfully inserted/updated {len(data)} records")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    populate_company_categorization()
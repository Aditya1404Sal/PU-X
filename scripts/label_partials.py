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

def update_intrinsic_labels():
    """
    Update intrinsic_label column based on membership in Nifty index CSV files only
    """
    # Load Nifty index data from CSV files
    nifty_symbols = {}
    
    # Load Nifty 100 (Large Cap)
    try:
        df = pd.read_csv('nifty100list.csv')
        nifty_symbols['IN-L'] = set(df['Symbol'].tolist())
        print(f"Loaded {len(nifty_symbols['IN-L'])} symbols from nifty100list.csv")
    except Exception as e:
        print(f"Error loading nifty100list.csv: {e}")
        nifty_symbols['IN-L'] = set()
    
    # Load Nifty Midcap 150
    try:
        df = pd.read_csv('niftymidcap150list.csv')
        nifty_symbols['IN-M'] = set(df['Symbol'].tolist())
        print(f"Loaded {len(nifty_symbols['IN-M'])} symbols from niftymidcap150list.csv")
    except Exception as e:
        print(f"Error loading niftymidcap150list.csv: {e}")
        nifty_symbols['IN-M'] = set()
    
    # Load Nifty Smallcap 100
    try:
        df = pd.read_csv('niftysmallcap100list.csv')
        nifty_symbols['IN-S'] = set(df['Symbol'].tolist())
        print(f"Loaded {len(nifty_symbols['IN-S'])} symbols from niftysmallcap100list.csv")
    except Exception as e:
        print(f"Error loading niftysmallcap100list.csv: {e}")
        nifty_symbols['IN-S'] = set()
    
    # Load Nifty Microcap 250
    try:
        df = pd.read_csv('niftymicrocap250_list.csv')
        nifty_symbols['IN-Mi'] = set(df['Symbol'].tolist())
        print(f"Loaded {len(nifty_symbols['IN-Mi'])} symbols from niftymicrocap250_list.csv")
    except Exception as e:
        print(f"Error loading niftymicrocap250_list.csv: {e}")
        nifty_symbols['IN-Mi'] = set()
    
    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    try:
        # First, set all intrinsic_label values to 'Unknown'
        cursor.execute("UPDATE company_categorization SET intrinsic_label = 'Unknown'")
        print("Reset all intrinsic_label values to 'Unknown'")
        
        # Now update each category
        for label, symbols in nifty_symbols.items():
            if symbols:
                # Convert symbols to a list to use with execute_values
                symbols_list = [(symbol,) for symbol in symbols]
                
                # Update the intrinsic_label for these symbols
                query = """
                UPDATE company_categorization 
                SET intrinsic_label = %s 
                WHERE symbol IN %s
                """
                
                # Execute for each symbol in the category
                for symbol in symbols:
                    cursor.execute(
                        "UPDATE company_categorization SET intrinsic_label = %s WHERE symbol = %s",
                        (label, symbol)
                    )
                
                print(f"Updated {len(symbols)} symbols with label {label}")
        
        conn.commit()
        print("All updates committed successfully")
        
        # Verify the updates
        cursor.execute("SELECT COUNT(*) FROM company_categorization WHERE intrinsic_label != 'Unknown'")
        count = cursor.fetchone()[0]
        print(f"Total records with intrinsic labels: {count}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    update_intrinsic_labels()
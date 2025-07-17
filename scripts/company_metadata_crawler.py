import requests
import json
import psycopg2
import os
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='company_metadata_crawler.log'
)

# Database connection parameters
DB_PARAMS = {
    'dbname': 'Stock',
    'user': 'adity',
    'password': 'qwertypoi',
    'host': 'localhost'
}

def get_db_connection():
    """Establish a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return None
# Change this to actually use the Csv file ka column
def fetch_initial_symbols():
    """Fetch list of symbols from local CSV file"""
    all_symbols = []
    csv_file_path = 'StockData.csv'  # Path to your CSV file
    
    try:
        import csv
        
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Extract the stock symbol from the 'TckrSymb' column
                symbol = row.get('TckrSymb')
                if symbol and symbol not in all_symbols:
                    all_symbols.append(symbol)
        
        logging.info(f"Successfully loaded {len(all_symbols)} symbols from {csv_file_path}")
        
    except Exception as e:
        logging.error(f"Error reading symbols from CSV file: {e}")
    
    return all_symbols

def fetch_company_details_batch(symbols, batch_size=10):
    """Fetch company details for multiple symbols using a single session"""
    session = requests.Session()
    
    # Headers that mimic Chrome browser
    headers = {
        'authority': 'www.nseindia.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
    }
    
    results = {}
    
    try:
        # First visit the NSE homepage to get cookies
        home_url = 'https://www.nseindia.com/'
        home_response = session.get(home_url, headers=headers, timeout=30)
        
        if home_response.status_code != 200:
            logging.warning(f"Failed to connect to NSE homepage: Status {home_response.status_code}")
            return results
        
        session.cookies.set(
        'nseappid', 
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTc0NDQwMjI4NCwiZXhwIjoxNzQ0NDA5NDg0fQ.JxckRsnFRKqJWRGBnIP2VQIk_Bky81Hw6_W_UDsHZj4',
        domain='www.nseindia.com'
        )
        session.cookies.set(
        'nsit',
        'SEqNE1JVN_00yfUTVjQ9PbSd',
        domain='www.nseindia.com'
        )
        # Wait a bit to allow cookies to be properly set
        time.sleep(1)
        
        # Process symbols in batches to avoid rate limiting
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
            
            for symbol in batch:
                try:
                    # Fetch company details
                    api_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
                    api_response = session.get(api_url, headers=headers, timeout=30)
                    
                    if api_response.status_code == 200:
                        results[symbol] = api_response.json()
                        logging.info(f"Successfully fetched details for {symbol}")
                    else:
                        logging.warning(f"Failed to fetch details for {symbol}: Status {api_response.status_code}")
                        results[symbol] = None
                        
                    # Small delay between requests to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error fetching details for {symbol}: {e}")
                    results[symbol] = None
        
    except Exception as e:
        logging.error(f"Batch processing error: {e}")
    finally:
        # Close the session
        session.close()
    
    return results

def parse_date(date_str):
    """Parse date string in the format DD-MMM-YYYY"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%d-%b-%Y').date()
    except:
        return None

def store_company_data(company_data, conn):
    """Store company metadata in the database using a shared connection"""
    if not company_data or not conn:
        return False

    try:
        cursor = conn.cursor()

        info = company_data.get('info', {})
        metadata = company_data.get('metadata', {})
        security_info = company_data.get('securityInfo', {})
        industry_info = company_data.get('industryInfo', {})

        symbol = info.get('symbol')
        company_name = info.get('companyName')
        isin = info.get('isin')

        if not (symbol and company_name and isin):
            logging.warning("Missing essential company information")
            cursor.close()
            return False

        industry = metadata.get('industry')
        listing_date = parse_date(metadata.get('listingDate'))
        face_value = security_info.get('faceValue')
        issued_size = security_info.get('issuedSize')
        sector = industry_info.get('sector')
        macro = industry_info.get('macro')
        basic_industry = industry_info.get('basicIndustry')

        cursor.execute("""
            INSERT INTO companies 
            (symbol, company_name, isin, industry, sector, macro_category, basic_industry, listing_date, face_value, issued_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) 
            DO UPDATE SET 
                company_name = EXCLUDED.company_name,
                isin = EXCLUDED.isin,
                industry = EXCLUDED.industry,
                sector = EXCLUDED.sector,
                macro_category = EXCLUDED.macro_category,
                basic_industry = EXCLUDED.basic_industry,
                listing_date = EXCLUDED.listing_date,
                face_value = EXCLUDED.face_value,
                issued_size = EXCLUDED.issued_size
        """, (symbol, company_name, isin, industry, sector, macro, basic_industry, listing_date, face_value, issued_size))

        conn.commit()
        cursor.close()
        return True

    except Exception as e:
        logging.error(f"Error storing company data: {e}")
        return False


def main():
    """Main crawler function"""
    logging.info("Starting company metadata crawler")

    # Check DB connection first
    conn = get_db_connection()
    if not conn:
        logging.error("Cannot proceed without a working database connection. Exiting.")
        return

    # Fetch initial symbols from CSV
    symbols = fetch_initial_symbols()
    logging.info(f"Found {len(symbols)} symbols to process")

    symbols = symbols[2450:]
    logging.info(f"Processing only the first {len(symbols)} symbols")

    # Process all symbols in batches
    results = fetch_company_details_batch(symbols)

    # Store results in database
    success_count = 0
    for symbol, company_data in results.items():
        if company_data:
            if store_company_data(company_data, conn):
                success_count += 1

    conn.close()
    logging.info(f"Successfully stored data for {success_count} out of {len(symbols)} companies")
    logging.info("Company metadata crawler completed")

if __name__ == "__main__":
    main()
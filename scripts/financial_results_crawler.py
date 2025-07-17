import requests
import os
import time
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='quarterly_results_downloader.log'
)

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

# Create directories for storing the results
for quarter in ['first', 'second', 'third']:
    os.makedirs(f'./results/{quarter}', exist_ok=True)

def fetch_stock_symbols():
    """Retrieve all stock symbols from the database"""
    try:
        import psycopg2
        # Database connection parameters
        DB_PARAMS = {
            'dbname': 'Stock',
            'user': 'adity',
            'password': 'qwertypoi',
            'host': 'localhost'
        }
        
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM companies")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return symbols
    except Exception as e:
        logging.error(f"Error fetching symbols from database: {e}")
        logging.info("Using a sample list of symbols instead")
        # Return a sample list of symbols if database connection fails
        return ["KINGFA", "RELIANCE", "TCS", "INFY", "HDFCBANK"]

def initialize_session():
    """Initialize a session with the necessary cookies"""
    session = requests.Session()
    
    try:
        # First visit the NSE homepage to get cookies
        home_url = 'https://www.nseindia.com/'
        home_response = session.get(home_url, headers=headers, timeout=30)
        
        if home_response.status_code != 200:
            logging.warning(f"Failed to connect to NSE homepage: Status {home_response.status_code}")
            return None
        
        # Set cookies manually that might be required
        session.cookies.set(
            'nseappid', 
            'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTc0NDY0OTE2OSwiZXhwIjoxNzQ0NjU2MzY5fQ.ntHcIGijF4rJdFiFfoIEmXnYz4ymv3AyNBYZlfQ0X3U',
            domain='www.nseindia.com'
        )
        session.cookies.set(
            'nsit',
            'utHjZrwsKtgRt5qSan5yI9BH',
            domain='www.nseindia.com'
        )
        
        # Wait a bit to allow cookies to be properly set
        time.sleep(1)
        
        # Debug: Print cookies
        logging.info(f"Session cookies: {dict(session.cookies)}")
        
        return session
    except Exception as e:
        logging.error(f"Error initializing session: {e}")
        return None

def fetch_financial_results(session, symbol):
    """Fetch financial results for a given stock symbol"""
    if not session:
        logging.error("No active session provided")
        return None
    
    # Calculate date range for last 2 years
    today = datetime.now()
    from_date = (today - timedelta(days=730)).strftime('%d-%m-%Y')
    to_date = today.strftime('%d-%m-%Y')
    
    # Update headers for API request
    api_headers = headers.copy()
    api_headers.update({
        'accept': 'application/json, text/plain, */*',
        'referer': f'https://www.nseindia.com/get-quotes/equity?symbol={symbol}',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
    })
    
    try:
        # First visit the symbol page to set additional cookies
        symbol_url = f'https://www.nseindia.com/get-quotes/equity?symbol={symbol}'
        symbol_response = session.get(symbol_url, headers=headers, timeout=30)
        
        if symbol_response.status_code != 200:
            logging.warning(f"Failed to access symbol page for {symbol}: Status {symbol_response.status_code}")
            # Continue anyway, but note the issue
        
        # Small delay
        time.sleep(1)
        
        # Now fetch the financial results
        api_url = f"https://www.nseindia.com/api/corporates-financial-results?index=equities&from_date={from_date}&to_date={to_date}&symbol={symbol}&period=Quarterly"
        api_response = session.get(api_url, headers=api_headers, timeout=30)
        
        if api_response.status_code == 200:
            try:
                data = api_response.json()
                logging.info(f"Successfully fetched financial results for {symbol}")
                return data
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON for {symbol}")
                logging.debug(f"Response text: {api_response.text[:200]}...")  # Log first 200 chars
                return None
        else:
            logging.warning(f"Failed to fetch financial results for {symbol}: Status {api_response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error fetching financial results for {symbol}: {e}")
        return None

def download_xbrl(session, url, symbol, quarter_directory):
    """Download XBRL file from URL and save it to the specified directory"""
    if not session:
        logging.error("No active session provided")
        return False
    
    try:
        response = session.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Create filename from the last part of the URL
            filename = url.split('/')[-1]
            filepath = os.path.join(quarter_directory, f"{symbol}_{filename}")
            
            # Save the file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"Downloaded {symbol} financial result to {filepath}")
            return True
        else:
            logging.warning(f"Failed to download XBRL for {symbol}: Status {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error downloading XBRL for {symbol}: {e}")
        return False

def get_quarter_info(result):
    """Determine which quarter the result belongs to"""
    relating_to = result.get('relatingTo', '').lower()
    
    if 'first' in relating_to or 'q1' in relating_to:
        return 'first'
    elif 'second' in relating_to or 'q2' in relating_to:
        return 'second'
    elif 'third' in relating_to or 'q3' in relating_to:
        return 'third'
    else:
        return None

def process_symbols_in_batches(symbols, batch_size=5):
    """Process symbols in batches to avoid overwhelming the server"""
    logging.info("Starting quarterly financial results downloader")
    session = initialize_session()
    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
        
        # Initialize new session for each batch
        
        for symbol in batch:
            try:
                logging.info(f"Processing {symbol}")
                financial_results_data = fetch_financial_results(session, symbol)
                
                if not financial_results_data:
                    logging.warning(f"No financial results found for {symbol}")
                    continue
                    
                # Keep track of which quarters we've found for this symbol
                quarters_found = set()
                
                for result in financial_results_data:
                    quarter = get_quarter_info(result)
                    
                    # Skip if not one of the quarters we're interested in or if we already found this quarter
                    if not quarter or quarter in quarters_found or quarter == 'fourth':
                        continue
                    
                    xbrl_url = result.get('xbrl')
                    if xbrl_url:
                        quarter_directory = f"./results/{quarter}"
                        download_success = download_xbrl(session, xbrl_url, symbol, quarter_directory)
                        if download_success:
                            quarters_found.add(quarter)
                    
                    # If we've found all three quarters, move on to the next symbol
                    if len(quarters_found) == 3:
                        break
                
                # Report quarters not found
                missing_quarters = set(['first', 'second', 'third']) - quarters_found
                if missing_quarters:
                    logging.warning(f"Could not find {', '.join(missing_quarters)} quarter results for {symbol}")
                
                
            except Exception as e:
                logging.error(f"Error processing symbol {symbol}: {e}")
        
        # Sleep between batches
        logging.info(f"Completed batch {i//batch_size + 1}. Sleeping before next batch...")
        time.sleep(1)  # Sleep between batches to avoid getting blocked
    
    logging.info("Quarterly financial results downloader completed")

def main():
    """Main function to download quarterly results"""
    # Get all stock symbols
    symbols = fetch_stock_symbols()
    logging.info(f"Found {len(symbols)} symbols to process")
    
    # Process symbols in batches
    process_symbols_in_batches(symbols, batch_size=5)

if __name__ == "__main__":
    main()
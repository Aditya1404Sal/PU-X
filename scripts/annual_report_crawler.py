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
    filename='annual_reports_crawler.log'
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

def fetch_stock_symbols():
    """Retrieve all stock symbols from the database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM companies")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return symbols
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        if conn:
            conn.close()
        return []

def fetch_annual_reports(symbol):
    """Fetch annual reports for a given stock symbol"""
    url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logging.warning(f"Failed to fetch annual reports for {symbol}: Status {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error fetching annual reports for {symbol}: {e}")
        return None

def download_pdf(url, symbol, from_year, to_year):
    """Download PDF from URL and store in buffer table"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            # Store in buffer table
            cursor.execute(
                "INSERT INTO document_buffer (symbol, document_type, document_data) VALUES (%s, %s, %s) RETURNING id",
                (symbol, f"annual_report_{from_year}_{to_year}", psycopg2.Binary(response.content))
            )
            buffer_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            # Return buffer ID for processing
            return buffer_id
        else:
            logging.warning(f"Failed to download PDF for {symbol} ({from_year}-{to_year}): Status {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error downloading PDF for {symbol} ({from_year}-{to_year}): {e}")
        return False

def store_annual_report_info(symbol, report_data):
    """Store annual report metadata in the database"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        for report in report_data.get('data', []):
            from_year = report.get('fromYr')
            to_year = report.get('toYr')
            file_url = report.get('fileName')
            
            # Skip if missing important data
            if not (from_year and to_year and file_url):
                continue
                
            # Parse dates if available
            submission_date = None
            dissemination_date = None
            
            if report.get('broadcast_dttm') and report.get('broadcast_dttm') != '-':
                try:
                    submission_date = datetime.strptime(report.get('broadcast_dttm'), '%d-%b-%Y %H:%M:%S')
                except:
                    pass
                    
            if report.get('disseminationDateTime') and report.get('disseminationDateTime') != '-':
                try:
                    dissemination_date = datetime.strptime(report.get('disseminationDateTime'), '%d-%b-%Y %H:%M:%S')
                except:
                    pass
            
            # Insert or update annual report information
            cursor.execute("""
                INSERT INTO annual_reports (symbol, from_year, to_year, submission_date, dissemination_date, file_url)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, from_year, to_year) 
                DO UPDATE SET file_url = EXCLUDED.file_url, 
                             submission_date = EXCLUDED.submission_date,
                             dissemination_date = EXCLUDED.dissemination_date
                RETURNING id
            """, (symbol, from_year, to_year, submission_date, dissemination_date, file_url))
            
            report_id = cursor.fetchone()[0]
            conn.commit()
            
            # Only download the latest annual report (to avoid downloading all historical reports)
            if report == report_data.get('data', [])[0]:
                buffer_id = download_pdf(file_url, symbol, from_year, to_year)
                if buffer_id:
                    logging.info(f"Downloaded {symbol} annual report ({from_year}-{to_year}) to buffer ID: {buffer_id}")
                
        cursor.close()
        conn.close()
        
    except Exception as e:
        logging.error(f"Error storing annual report info for {symbol}: {e}")
        if conn:
            conn.close()

def main():
    """Main crawler function"""
    logging.info("Starting annual reports crawler")
    
    # Get all stock symbols from database
    symbols = fetch_stock_symbols()
    logging.info(f"Found {len(symbols)} symbols to process")
    
    for i, symbol in enumerate(symbols):
        try:
            logging.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            annual_reports_data = fetch_annual_reports(symbol)
            
            if annual_reports_data:
                store_annual_report_info(symbol, annual_reports_data)
            
            # Sleep to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error processing symbol {symbol}: {e}")
    
    logging.info("Annual reports crawler completed")

if __name__ == "__main__":
    main()
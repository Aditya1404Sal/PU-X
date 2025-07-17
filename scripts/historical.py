import os
import pandas as pd
import numpy as np
import psycopg2
import logging
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_aggregation.log"),
        logging.StreamHandler()
    ]
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
        logging.error(traceback.format_exc())
        return None

def create_yearly_metrics_table():
    """Create table for storing stock-level metrics (one row per symbol)"""
    conn = get_db_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Drop old tables if needed (optional and dangerous — use with care)
        # cursor.execute("DROP TABLE IF EXISTS stock_yearly_engineered_features CASCADE;")
        # cursor.execute("DROP TABLE IF EXISTS stock_yearly_metrics CASCADE;")

        # Create stock-level metrics table (no 'year' field)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_yearly_metrics (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) UNIQUE REFERENCES companies(symbol),
            avg_open DECIMAL(20, 2),
            avg_high DECIMAL(20, 2),
            avg_low DECIMAL(20, 2),
            avg_close DECIMAL(20, 2),
            avg_last_price DECIMAL(20, 2),
            avg_vwap DECIMAL(20, 2),
            avg_volume BIGINT,
            max_high DECIMAL(20, 2),
            min_low DECIMAL(20, 2),
            volatility DECIMAL(10, 4),
            total_volume BIGINT,
            total_value DECIMAL(25, 2),
            total_trades INTEGER,
            trading_days INTEGER,
            price_range_percent DECIMAL(10, 2),
            year_start_price DECIMAL(20, 2),
            year_end_price DECIMAL(20, 2),
            year_change_percent DECIMAL(10, 2),
            average_daily_return DECIMAL(10, 4),
            sharpe_ratio DECIMAL(10, 4),
            beta DECIMAL(10, 4),
            fifty_two_week_high DECIMAL(20, 2),
            fifty_two_week_low DECIMAL(20, 2),
            avg_intraday_volatility DECIMAL(10, 4),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create engineered features table (linked to metrics by symbol instead of year)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_yearly_engineered_features (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) UNIQUE REFERENCES companies(symbol),
            yearly_metrics_id INTEGER UNIQUE REFERENCES stock_yearly_metrics(id),
            avg_daily_return DECIMAL(10, 6),
            avg_intraday_volatility DECIMAL(10, 6),
            avg_vwap_distance DECIMAL(10, 6),
            avg_volume_spike DECIMAL(10, 6),
            avg_fifty_two_week_proximity DECIMAL(10, 6),
            avg_atr DECIMAL(10, 6),
            avg_ltp_vs_close_delta DECIMAL(10, 6),
            yearly_momentum_score DECIMAL(10, 4),
            liquidity_score DECIMAL(10, 4),
            stability_score DECIMAL(10, 4),
            trend_strength DECIMAL(10, 4)
        )
        """)


        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Stock-level metrics tables created or confirmed")
        return True

    except Exception as e:
        logging.error(f"Error creating stock metrics tables: {e}")
        logging.error(traceback.format_exc())
        if conn:
            conn.close()
        return False

    
def convert_numpy_types(value):
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    else:
        return value

def clean_numeric_columns(df):
    """Clean and convert numeric columns in the dataframe"""
    logging.info(f"DataFrame columns before cleaning: {df.columns.tolist()}")
    
    df.columns = df.columns.str.strip()
    logging.info(f"DataFrame columns after stripping spaces: {df.columns.tolist()}")
    # Remove commas and convert to appropriate types
    numeric_cols = ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap', '52W H', '52W L']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').astype(float)
            logging.info(f"Cleaned column {col}, sample value: {df[col].iloc[0] if len(df) > 0 else 'no data'}")
        else:
            logging.warning(f"Column {col} not found in DataFrame")
    
    volume_cols = ['VOLUME', 'No of trades']
    for col in volume_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').astype(int)
            logging.info(f"Cleaned column {col}, sample value: {df[col].iloc[0] if len(df) > 0 else 'no data'}")
        else:
            logging.warning(f"Column {col} not found in DataFrame")
    
    if 'VALUE' in df.columns:
        df['VALUE'] = df['VALUE'].astype(str).str.replace(',', '').str.replace('"', '').astype(float)
        logging.info(f"Cleaned VALUE column, sample value: {df['VALUE'].iloc[0] if len(df) > 0 else 'no data'}")
    else:
        logging.warning("VALUE column not found in DataFrame")
    
    # Convert date column
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
            df['year'] = df['Date'].dt.year
            logging.info(f"Converted Date column, sample value: {df['Date'].iloc[0] if len(df) > 0 else 'no data'}")
        except Exception as e:
            logging.error(f"Error converting Date column: {e}")
            logging.error(traceback.format_exc())
            logging.info(f"First few Date values: {df['Date'].head().tolist()}")
            # Fall back to a more flexible date parsing
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['year'] = df['Date'].dt.year
    else:
        logging.warning("Date column not found in DataFrame")
    
    return df

def calculate_stock_metrics(df, symbol):
    metrics = {}

    df_sorted = df.sort_values('Date')
    start_price = df_sorted['close'].iloc[0]
    end_price = df_sorted['close'].iloc[-1]
    df_sorted['daily_return'] = df_sorted['close'].pct_change()

    metrics = {
        'symbol': symbol,
        'avg_open': df['OPEN'].mean(),
        'avg_high': df['HIGH'].mean(),
        'avg_low': df['LOW'].mean(),
        'avg_close': df['close'].mean(),
        'avg_last_price': df['ltp'].mean() if 'ltp' in df.columns else None,
        'avg_vwap': df['vwap'].mean() if 'vwap' in df.columns else None,
        'avg_volume': int(df['VOLUME'].mean()),
        'max_high': df['HIGH'].max(),
        'min_low': df['LOW'].min(),
        'volatility': df['close'].std(),
        'total_volume': df['VOLUME'].sum(),
        'total_value': df['VALUE'].sum() if 'VALUE' in df.columns else None,
        'total_trades': df['No of trades'].sum() if 'No of trades' in df.columns else None,
        'trading_days': len(df),
        'price_range_percent': ((df['HIGH'].max() - df['LOW'].min()) / df['LOW'].min()) * 100,
        'year_start_price': start_price,
        'year_end_price': end_price,
        'year_change_percent': ((end_price - start_price) / start_price) * 100,
        'average_daily_return': df_sorted['daily_return'].mean() * 100,
        'sharpe_ratio': (df_sorted['daily_return'].mean() / df_sorted['daily_return'].std() * np.sqrt(252)) if df_sorted['daily_return'].std() > 0 else None,
        'beta': None,  # still placeholder
        'fifty_two_week_high': df['HIGH'].max(),
        'fifty_two_week_low': df['LOW'].min(),
        'avg_intraday_volatility': ((df['HIGH'] - df['LOW']) / df['LOW']).mean() * 100
    }

    return metrics


def calculate_stock_engineered_features(df, symbol):
    """Calculate engineered features over entire dataset for a stock symbol"""
    df_sorted = df.sort_values('Date')
    df_sorted['daily_return'] = df_sorted['close'].pct_change()

    # Intraday volatility
    df['intraday_volatility'] = (df['HIGH'] - df['LOW']) / df['LOW']

    # VWAP distance
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] if 'vwap' in df.columns else None

    # Volume spike
    df['volume_ma10'] = df['VOLUME'].rolling(window=10).mean()
    df['volume_spike'] = df['VOLUME'] / df['volume_ma10']

    # 52-week proximity
    df['52w_range'] = df['52W H'] - df['52W L']
    df['fifty_two_week_proximity'] = (df['close'] - df['52W L']) / df['52w_range']

    # LTP vs Close
    df['ltp_vs_close_delta'] = (df['ltp'] - df['close']) / df['close'] if 'ltp' in df.columns else None

    # ATR (Average True Range)
    df['tr1'] = df['HIGH'] - df['LOW']
    df['tr2'] = abs(df['HIGH'] - df['close'].shift(1))
    df['tr3'] = abs(df['LOW'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()

    # Trend metrics
    price_changes = df_sorted['close'].pct_change().dropna()
    momentum = price_changes.mean() * len(price_changes)
    liquidity = min(df['VOLUME'].mean() / 10000, 10) * (len(df) / 252)
    stability = 1 / (df['intraday_volatility'].std() * 10) if df['intraday_volatility'].std() > 0 else 10
    trend_strength = abs((price_changes > 0).sum() - (price_changes < 0).sum()) / len(price_changes)

    features = {
        'symbol': symbol,
        'avg_daily_return': df_sorted['daily_return'].mean(),
        'avg_intraday_volatility': df['intraday_volatility'].mean(),
        'avg_vwap_distance': df['vwap_distance'].mean() if 'vwap_distance' in df.columns else None,
        'avg_volume_spike': df['volume_spike'].dropna().mean(),
        'avg_fifty_two_week_proximity': df['fifty_two_week_proximity'].mean(),
        'avg_atr': df['atr'].dropna().mean(),
        'avg_ltp_vs_close_delta': df['ltp_vs_close_delta'].mean(),
        'yearly_momentum_score': momentum,
        'liquidity_score': liquidity,
        'stability_score': stability,
        'trend_strength': trend_strength
    }

    return features


def save_metrics_to_db(metrics, features):
    """Save metrics and features for a single stock symbol"""
    conn = get_db_connection()
    if not conn:
        return 0

    try:
        cursor = conn.cursor()

        # --- Save metrics ---
        metrics = {k: convert_numpy_types(v) for k, v in metrics.items() if v is not None}
        columns = ', '.join(metrics.keys())
        placeholders = ', '.join(['%s'] * len(metrics))
        update_stmt = ', '.join([f"{k}=EXCLUDED.{k}" for k in metrics if k != 'symbol'])

        sql = f"""
        INSERT INTO stock_yearly_metrics ({columns})
        VALUES ({placeholders})
        ON CONFLICT (symbol) DO UPDATE SET {update_stmt}, last_updated = CURRENT_TIMESTAMP
        RETURNING id
        """
        cursor.execute(sql, list(metrics.values()))
        metric_id = cursor.fetchone()[0]

        # --- Save features ---
        features['yearly_metrics_id'] = metric_id
        features = {k: convert_numpy_types(v) for k, v in features.items() if v is not None}

        feat_cols = ', '.join(features.keys())
        feat_placeholders = ', '.join(['%s'] * len(features))
        feat_update_stmt = ', '.join([f"{k}=EXCLUDED.{k}" for k in features if k != 'yearly_metrics_id'])

        feat_sql = f"""
        INSERT INTO stock_yearly_engineered_features ({feat_cols})
        VALUES ({feat_placeholders})
        ON CONFLICT (yearly_metrics_id) DO UPDATE SET {feat_update_stmt}
        """
        cursor.execute(feat_sql, list(features.values()))

        conn.commit()
        cursor.close()
        conn.close()
        return 1

    except Exception as e:
        logging.error(f"Error saving to database: {e}")
        logging.error(traceback.format_exc())
        conn.rollback()
        conn.close()
        return 0


def process_stock_file(file_path):
    """Process a single stock CSV file and calculate yearly metrics"""
    try:
        # Extract symbol from filename
        symbol = os.path.basename(file_path).replace('.csv', '')
        
        logging.info(f"Processing {symbol} data from {file_path}")
        
        # Read data
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read {file_path}, shape: {df.shape}")
        
        # Print the first few rows and column names for debugging
        logging.info(f"First few rows of {symbol} data:\n{df.head().to_string()}")
        logging.info(f"Columns in {symbol} data: {df.columns.tolist()}")
        
        # Clean and prepare data
        df = clean_numeric_columns(df)
        
        # Calculate daily returns for engineered features
        if 'close' in df.columns:
            df['daily_return'] = df['close'].pct_change()
        else:
            logging.error(f"Missing 'close' column in {file_path}")
            return symbol, False
        
        # Calculate yearly metrics
        metrics = calculate_stock_metrics(df, symbol)
        
        if not metrics:
            logging.warning(f"No yearly metrics calculated for {symbol}")
            return symbol, False
        
        # Calculate yearly engineered features
        features = calculate_stock_engineered_features(df, symbol)
        
        # Save to database
        saved_count = save_metrics_to_db(metrics, features)
        
        if saved_count > 0:
            logging.info(f"Successfully processed {symbol}: saved metrics for {saved_count} years")
            return symbol, True
        else:
            logging.warning(f"No metrics saved for {symbol}")
            return symbol, False
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        logging.error(traceback.format_exc())
        return os.path.basename(file_path).replace('.csv', ''), False

def aggregate_all_stocks(data_directory="historical_data", parallel=True, max_workers=5):
    """Aggregate yearly metrics for all stock CSV files in the directory"""
    # Ensure the yearly metrics table exists
    if not create_yearly_metrics_table():
        logging.error("Failed to create or confirm yearly metrics tables, aborting")
        return [], []
    
    # Get all CSV files
    if not os.path.exists(data_directory):
        logging.error(f"Data directory {data_directory} does not exist")
        return [], []
    
    csv_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) 
                if f.endswith('.csv') and os.path.isfile(os.path.join(data_directory, f))]
    
    if not csv_files:
        logging.error(f"No CSV files found in {data_directory}")
        return [], []
    
    logging.info(f"Found {len(csv_files)} stock files to process")
    
    successful = []
    failed = []
    
    if parallel:
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_stock_file, csv_files))
        
        # Parse results
        successful = [symbol for symbol, success in results if success]
        failed = [symbol for symbol, success in results if not success]
    else:
        # Sequential processing
        for file_path in csv_files:
            result = process_stock_file(file_path)
            if result[1]:  # If successful
                successful.append(result[0])
            else:
                failed.append(result[0])
    
    # Report results
    logging.info(f"Aggregation complete. Successful: {len(successful)}, Failed: {len(failed)}")
    if failed:
        logging.warning(f"Failed symbols: {failed}")
    
    return successful, failed

def generate_aggregation_report():
    """Generate a report of the aggregated yearly metrics"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Count metrics by year
        cursor.execute("""
        SELECT year, COUNT(*) as stock_count
        FROM stock_yearly_metrics
        GROUP BY year
        ORDER BY year DESC
        """)
        
        year_counts = cursor.fetchall()
        
        # Get stocks with highest yearly gains
        cursor.execute("""
        SELECT symbol, year, year_change_percent
        FROM stock_yearly_metrics
        ORDER BY year_change_percent DESC
        LIMIT 10
        """)
        
        top_gainers = cursor.fetchall()
        
        # Get stocks with highest volatility
        cursor.execute("""
        SELECT symbol, year, volatility
        FROM stock_yearly_metrics
        ORDER BY volatility DESC
        LIMIT 10
        """)
        
        most_volatile = cursor.fetchall()
        
        # Get stocks with best engineered features scores
        try:
            cursor.execute("""
            SELECT m.symbol, m.year, f.yearly_momentum_score, f.liquidity_score, f.stability_score
            FROM stock_yearly_metrics m
            JOIN stock_yearly_engineered_features f ON m.id = f.yearly_metrics_id
            ORDER BY (
                COALESCE(f.yearly_momentum_score, 0) + 
                COALESCE(f.liquidity_score, 0) + 
                COALESCE(f.stability_score, 0)
            ) DESC
            LIMIT 10
            """)
            
            best_features = cursor.fetchall()
        except Exception as e:
            logging.error(f"Error querying best features: {e}")
            best_features = []
        
        cursor.close()
        conn.close()
        
        # Print report
        print("\n=== AGGREGATION REPORT ===")
        
        print("\nMetrics by Year:")
        for year, count in year_counts:
            print(f"  {year}: {count} stocks")
        
        print("\nTop 10 Gainers:")
        for symbol, year, gain in top_gainers:
            print(f"  {symbol} ({year}): {gain:.2f}%")
        
        print("\nTop 10 Most Volatile Stocks:")
        for symbol, year, vol in most_volatile:
            print(f"  {symbol} ({year}): {vol:.2f}")
            
        print("\nTop 10 Stocks by Combined Feature Scores:")
        for symbol, year, momentum, liquidity, stability in best_features:
            print(f"  {symbol} ({year}): Momentum={momentum:.2f}, Liquidity={liquidity:.2f}, Stability={stability:.2f}")
        
        print("\n=========================")
        
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        logging.error(traceback.format_exc())
        if conn:
            conn.close()

if __name__ == "__main__":
    data_dir = "historical_data"  # Directory containing stock CSV files
    
    logging.info("Starting stock data aggregation")
    
    # Test processing a single file first
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))]
    
    if csv_files:
        test_file = csv_files[0]  # Take the first CSV file
        logging.info(f"Testing with single file: {test_file}")
        test_result = process_stock_file(test_file)
        logging.info(f"Test processing result: {test_result}")
    
    # Aggregate all stocks with parallel processing disabled for better error tracing
    successful, failed = aggregate_all_stocks(
        data_directory=data_dir,
        parallel=False,  # Set to False for easier debugging
        max_workers=5
    )
    
    # Generate report of the aggregated data
    generate_aggregation_report()
    
    logging.info("Aggregation script execution completed")
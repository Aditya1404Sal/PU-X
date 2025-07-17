-- Companies table - Core information about each company
CREATE TABLE companies (
    symbol VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    isin VARCHAR(20) UNIQUE NOT NULL,
    industry VARCHAR(100),
    sector VARCHAR(100),
    macro_category VARCHAR(100),
    basic_industry VARCHAR(100),
    listing_date DATE,
    face_value DECIMAL(10, 2),
    issued_size BIGINT
);

-- Annual Reports table - Links to PDFs and metadata
CREATE TABLE annual_reports (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol),
    from_year VARCHAR(10) NOT NULL,
    to_year VARCHAR(10) NOT NULL,
    submission_date TIMESTAMP,
    dissemination_date TIMESTAMP,
    file_url TEXT NOT NULL,
    is_processed BOOLEAN DEFAULT FALSE,
    processed_date TIMESTAMP,
    UNIQUE (symbol, from_year, to_year)
);

-- Annual Report Sentiment - Extracted from report text
CREATE TABLE annual_report_sentiment (
    id SERIAL PRIMARY KEY,
    annual_report_id INT REFERENCES annual_reports(id),
    uncertainty_score DECIMAL(5, 2),
    risk_keywords_percentage DECIMAL(5, 2),
    opportunity_keywords_percentage DECIMAL(5, 2),
    forward_looking_score DECIMAL(5, 2),
    historical_score DECIMAL(5, 2),
    overall_sentiment_score DECIMAL(5, 2),
    confidence_score DECIMAL(5, 2),
    supply_chain_companies TEXT[]
);

-- Financial Metrics - From quarterly/annual reports
CREATE TABLE financial_metrics_averages (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol),
    quarters_count INTEGER,
    avg_revenue_from_operations DECIMAL(30, 10),
    avg_other_income DECIMAL(30, 10),
    avg_income DECIMAL(30, 10),
    avg_employee_benefit_expense DECIMAL(30, 10),
    avg_finance_costs DECIMAL(30, 10),
    avg_depreciation_depletion_amortisation_expense DECIMAL(30, 10),
    avg_other_expenses DECIMAL(30, 10),
    avg_expenses DECIMAL(30, 10),
    avg_profit_before_tax DECIMAL(30, 10),
    avg_tax_expense DECIMAL(30, 10),
    avg_profit_loss_for_period DECIMAL(30, 10),
    avg_basic_earnings_per_share DECIMAL(30, 10),
    avg_diluted_earnings_per_share DECIMAL(30, 10),
    avg_comprehensive_income DECIMAL(30, 10),
    avg_paid_up_equity_share_capital DECIMAL(30, 10),
    avg_face_value_equity_share_capital DECIMAL(30, 10),
    UNIQUE (symbol)
);

-- Daily Stock Data
CREATE TABLE stock_historical_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol),
    date DATE NOT NULL,
    open DECIMAL(20, 2),
    high DECIMAL(20, 2),
    low DECIMAL(20, 2),
    close DECIMAL(20, 2),
    last_price DECIMAL(20, 2),
    vwap DECIMAL(20, 2),
    volume BIGINT,
    UNIQUE (symbol, date)
);

-- Stores the categorization helpers
CREATE TABLE company_categorization (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    market_cap_category VARCHAR(10) NOT NULL, -- 'Micro', 'Nano', 'Small', 'Mid', 'Large'
    industry VARCHAR(50) NOT NULL,
    face_value DECIMAL(10, 2) NOT NULL,
    intrinsic_label VARCHAR(10) NOT NULL, -- 'IN-L', 'IN-M', 'IN-S', 'IN-Mi', 'IN-N'
    CONSTRAINT valid_market_cap CHECK (market_cap_category IN ('Micro', 'Small', 'Mid', 'Large')),
    CONSTRAINT valid_intrinsic_label CHECK (intrinsic_label IN ('IN-L', 'IN-M', 'IN-S', 'IN-Mi'))
);


-- Engineered Features from Historical Data
CREATE TABLE stock_engineered_features (
    id SERIAL PRIMARY KEY,
    stock_data_id INT REFERENCES stock_historical_data(id),
    daily_return DECIMAL(10, 6),
    intraday_volatility DECIMAL(10, 6),
    vwap_distance DECIMAL(10, 6),
    volume_spike DECIMAL(10, 6),
    fifty_two_week_proximity DECIMAL(10, 6),
    atr DECIMAL(10, 6),
    ltp_vs_close_delta DECIMAL(10, 6)
);

-- Social Media Sentiment
CREATE TABLE social_media_sentiment (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol) NULL, -- Can be NULL for general market sentiment
    source VARCHAR(50) NOT NULL, -- Reddit, Twitter, etc.
    title TEXT,
    url TEXT,
    post_date TIMESTAMP,
    collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    score INT,
    sentiment_polarity DECIMAL(5, 2),
    sentiment_subjectivity DECIMAL(5, 2)
);

-- News Sentiment
CREATE TABLE news_sentiment (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol) NULL, -- Can be NULL for general market news
    source VARCHAR(100),
    headline TEXT,
    url TEXT,
    published_date TIMESTAMP,
    collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentiment_polarity DECIMAL(5, 2),
    sentiment_subjectivity DECIMAL(5, 2),
    news_type VARCHAR(50) -- Supply chain, Geopolitical, General, etc.
);

-- Supply Chain Data
CREATE TABLE supply_chain_relationships (
    id SERIAL PRIMARY KEY,
    company_symbol VARCHAR(20) REFERENCES companies(symbol),
    related_company VARCHAR(255),
    relationship_type VARCHAR(50), -- Supplier, Customer, etc.
    strength_indicator DECIMAL(5, 2) -- How important is this relationship
);

-- Macro Economic Indicators
CREATE TABLE macro_indicators (
    id SERIAL PRIMARY KEY,
    indicator_name VARCHAR(100),
    date DATE,
    value DECIMAL(20, 6),
    previous_value DECIMAL(20, 6),
    percentage_change DECIMAL(10, 6),
    UNIQUE (indicator_name, date)
);

-- Commodity Prices
CREATE TABLE commodity_prices (
    id SERIAL PRIMARY KEY,
    commodity_name VARCHAR(100),
    date DATE,
    price DECIMAL(20, 6),
    currency VARCHAR(10),
    UNIQUE (commodity_name, date)
);

-- Foreign Exchange Rates
CREATE TABLE forex_rates (
    id SERIAL PRIMARY KEY,
    currency_pair VARCHAR(20),
    date DATE,
    rate DECIMAL(20, 6),
    UNIQUE (currency_pair, date)
);

-- Industry to Commodity/Forex Correlation
CREATE TABLE industry_correlations (
    id SERIAL PRIMARY KEY,
    industry VARCHAR(100),
    correlated_item VARCHAR(100), -- commodity name or forex pair
    item_type VARCHAR(20), -- 'commodity' or 'forex'
    correlation_coefficient DECIMAL(5, 4),
    significance_level DECIMAL(5, 4),
    UNIQUE (industry, correlated_item)
);

-- Model Outputs - Final Classifications and Scores
CREATE TABLE stock_classifications (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol),
    date DATE,
    intrinsic_value_score DECIMAL(5, 2),
    confidence_score DECIMAL(5, 2),
    cluster_id INT,
    nifty_match_proximity DECIMAL(5, 2),
    summary TEXT,
    financial_health_score DECIMAL(5, 2),
    sentiment_score DECIMAL(5, 2),
    supply_chain_outlook_score DECIMAL(5, 2),
    macro_outlook_score DECIMAL(5, 2),
    UNIQUE (symbol, date)
);

CREATE TABLE nifty_index_membership (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol),
    index_name VARCHAR(100) NOT NULL,
    entry_date DATE,
    exit_date DATE,
    is_current BOOLEAN DEFAULT TRUE,
    UNIQUE (symbol, index_name, entry_date)
);

-- Modify stock_classifications table to include index-based information
ALTER TABLE stock_classifications 
ADD COLUMN is_nifty_member BOOLEAN DEFAULT FALSE,
ADD COLUMN nifty_index_category VARCHAR(50);

-- Create an index for faster lookups
CREATE INDEX idx_nifty_membership_symbol ON nifty_index_membership(symbol);
CREATE INDEX idx_nifty_membership_index ON nifty_index_membership(index_name);
CREATE INDEX idx_nifty_membership_current ON nifty_index_membership(is_current);

-- Buffer table for temporary storage of large PDFs
CREATE TABLE document_buffer (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) REFERENCES companies(symbol),
    document_type VARCHAR(50),
    document_data BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);

-- Create indexes for performance
CREATE INDEX idx_companies_industry ON companies(industry);
CREATE INDEX idx_financial_metrics_symbol ON financial_metrics(symbol);
CREATE INDEX idx_financial_metrics_date ON financial_metrics(to_date);
CREATE INDEX idx_stock_data_symbol ON stock_historical_data(symbol);
CREATE INDEX idx_stock_data_date ON stock_historical_data(date);
CREATE INDEX idx_social_media_symbol ON social_media_sentiment(symbol);
CREATE INDEX idx_news_sentiment_symbol ON news_sentiment(symbol);
CREATE INDEX idx_news_sentiment_date ON news_sentiment(published_date);
CREATE INDEX idx_stock_class_score ON stock_classifications(intrinsic_value_score);

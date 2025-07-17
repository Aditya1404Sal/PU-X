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
SELECT 
  cc.symbol,
  c.company_name,
  c.isin,
  c.industry AS company_industry,
  c.sector,
  c.macro_category,
  c.basic_industry,
  c.listing_date,
  c.face_value AS company_face_value,
  c.issued_size,
  
  cc.market_cap_category,
  cc.intrinsic_label,
  
  fma.avg_revenue_from_operations,
  fma.avg_other_income,
  fma.avg_income,
  fma.avg_employee_benefit_expense,
  fma.avg_finance_costs,
  fma.avg_depreciation_depletion_amortisation_expense,
  fma.avg_other_expenses,
  fma.avg_profit_before_tax,
  fma.avg_tax_expense,
  fma.avg_profit_loss_for_period,
  fma.avg_basic_earnings_per_share,
  fma.avg_diluted_earnings_per_share,
  fma.avg_comprehensive_income,
  fma.avg_paid_up_equity_share_capital,
  fma.avg_face_value_equity_share_capital,
  
  sym.avg_open,
  sym.avg_close,
  sym.avg_volume,
  sym.volatility,
  sym.year_change_percent,
  sym.sharpe_ratio,
  sym.avg_high,
  sym.avg_low,
  sym.year_start_price,
  sym.year_end_price,

  syef.avg_daily_return,
  syef.avg_vwap_distance,
  syef.avg_volume_spike,
  syef.yearly_momentum_score,
  syef.liquidity_score,
  syef.stability_score,
  syef.trend_strength

FROM company_categorization cc
INNER JOIN companies c ON cc.symbol = c.symbol
INNER JOIN financial_metrics_averages fma ON c.symbol = fma.symbol
INNER JOIN stock_yearly_metrics sym ON c.symbol = sym.symbol
INNER JOIN stock_yearly_engineered_features syef ON c.symbol = syef.symbol;
"""

# Connect and execute
try:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query(query, conn)
    df.to_csv("company_financials_clean.csv", index=False)
    print("✅ CSV exported successfully: company_financials_clean.csv")
except Exception as e:
    print("❌ Error:", e)
finally:
    if conn:
        conn.close()

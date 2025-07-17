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

def load_data_to_db():
    # Read CSV file
    df = pd.read_csv('quarterly_averages.csv')
    
    # Extract quarters count from QuartersAvailable column
    df['quarters_count'] = pd.to_numeric(df['QuartersAvailable'], errors='coerce')
    
    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    try:
        # Prepare data for insertion
        data = []
        for _, row in df.iterrows():
            data.append((
                row['Symbol'],
                row['quarters_count'],
                row['Avg_RevenueFromOperations'],
                row['Avg_OtherIncome'],
                row['Avg_Income'],
                row['Avg_EmployeeBenefitExpense'],
                row['Avg_FinanceCosts'],
                row['Avg_DepreciationDepletionAndAmortisationExpense'],
                row['Avg_OtherExpenses'],
                row['Avg_Expenses'],
                row['Avg_ProfitBeforeTax'],
                row['Avg_TaxExpense'],
                row['Avg_ProfitLossForPeriod'],
                row['Avg_BasicEarningsLossPerShareFromContinuingOperations'],
                row['Avg_DilutedEarningsLossPerShareFromContinuingOperations'],
                row['Avg_ComprehensiveIncomeForThePeriod'],
                row['Avg_PaidUpValueOfEquityShareCapital'],
                row['Avg_FaceValueOfEquityShareCapital']
            ))
        
        # Execute batch insert
        query = """
        INSERT INTO financial_metrics_averages (
            symbol, quarters_count, avg_revenue_from_operations, avg_other_income, 
            avg_income, avg_employee_benefit_expense, avg_finance_costs, 
            avg_depreciation_depletion_amortisation_expense, avg_other_expenses, 
            avg_expenses, avg_profit_before_tax, avg_tax_expense, 
            avg_profit_loss_for_period, avg_basic_earnings_per_share, 
            avg_diluted_earnings_per_share, avg_comprehensive_income, 
            avg_paid_up_equity_share_capital, avg_face_value_equity_share_capital
        ) VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET 
            quarters_count = EXCLUDED.quarters_count,
            avg_revenue_from_operations = EXCLUDED.avg_revenue_from_operations,
            avg_other_income = EXCLUDED.avg_other_income,
            avg_income = EXCLUDED.avg_income,
            avg_employee_benefit_expense = EXCLUDED.avg_employee_benefit_expense,
            avg_finance_costs = EXCLUDED.avg_finance_costs,
            avg_depreciation_depletion_amortisation_expense = EXCLUDED.avg_depreciation_depletion_amortisation_expense,
            avg_other_expenses = EXCLUDED.avg_other_expenses,
            avg_expenses = EXCLUDED.avg_expenses,
            avg_profit_before_tax = EXCLUDED.avg_profit_before_tax,
            avg_tax_expense = EXCLUDED.avg_tax_expense,
            avg_profit_loss_for_period = EXCLUDED.avg_profit_loss_for_period,
            avg_basic_earnings_per_share = EXCLUDED.avg_basic_earnings_per_share,
            avg_diluted_earnings_per_share = EXCLUDED.avg_diluted_earnings_per_share,
            avg_comprehensive_income = EXCLUDED.avg_comprehensive_income,
            avg_paid_up_equity_share_capital = EXCLUDED.avg_paid_up_equity_share_capital,
            avg_face_value_equity_share_capital = EXCLUDED.avg_face_value_equity_share_capital
        """
        
        execute_values(cursor, query, data)
        conn.commit()
        print(f"Successfully inserted {len(data)} records")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_data_to_db()
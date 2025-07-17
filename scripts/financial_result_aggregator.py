import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='quarterly_results_parser.log'
)

# Define the financial metrics to extract
METRICS = [
    'RevenueFromOperations',
    'OtherIncome',
    'Income',
    'EmployeeBenefitExpense',
    'FinanceCosts',
    'DepreciationDepletionAndAmortisationExpense',
    'OtherExpenses',
    'Expenses',
    'ProfitBeforeTax',
    'TaxExpense',
    'ProfitLossForPeriod',
    'BasicEarningsLossPerShareFromContinuingOperations',
    'DilutedEarningsLossPerShareFromContinuingOperations',
    'ComprehensiveIncomeForThePeriod',
    'PaidUpValueOfEquityShareCapital',
    'FaceValueOfEquityShareCapital',
    'DilutedEarningsLossPerShare'
]

# Define possible XBRL namespaces (might vary between files)
NAMESPACES = [
    'in-bse-fin:',
    'in-gaap-fin:',
    'in-nse-fin:',
    'ind-as:',
    ''  # Also try without namespace
]

def extract_stock_symbol(filename):
    """Extract stock symbol from filename."""
    return filename.split('_')[0]

def parse_xbrl_file(file_path):
    """Parse an XBRL file and extract financial metrics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, 'lxml-xml')
        
        # Find the quarterly context
        contexts = soup.find_all('context')
        quarterly_context = None
        
        # Look for quarterly context (often has duration of ~3 months)
        for ctx in contexts:
            period = ctx.find('period')
            if not period:
                continue
                
            start_date = period.find('startDate')
            end_date = period.find('endDate')
            
            if start_date and end_date:
                try:
                    # Calculate duration in days
                    start = pd.to_datetime(start_date.text)
                    end = pd.to_datetime(end_date.text)
                    duration = (end - start).days
                    
                    # Look for a duration close to a quarter (60-100 days)
                    if 60 <= duration <= 100:
                        quarterly_context = ctx.get('id')
                        break
                except:
                    continue
        
        if not quarterly_context:
            # If we couldn't find a quarterly context, try some common context IDs
            common_contexts = ['OneD', 'CurrentQuarter', 'CurrentYTDQuarter', 'Q1', 'Q2', 'Q3']
            for ctx_id in common_contexts:
                if soup.find('context', id=ctx_id):
                    quarterly_context = ctx_id
                    break
        
        # If still no context found, use the first one
        if not quarterly_context and contexts:
            quarterly_context = contexts[0].get('id')
        
        if not quarterly_context:
            logging.warning(f"Could not identify quarterly context in {file_path}")
            return None
            
        # Extract data using the identified context
        data = {}
        
        for metric in METRICS:
            value = None
            # Try different namespaces
            for ns in NAMESPACES:
                tag = soup.find(f'{ns}{metric}', attrs={'contextRef': quarterly_context})
                if tag:
                    try:
                        value = float(tag.text.strip())
                        break
                    except ValueError:
                        continue
            
            data[metric] = value
        
        return data
        
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return None

def process_all_files():
    """Process all XBRL files in the results directories."""
    # Get all filenames in the three quarterly directories
    quarters = ['first', 'second', 'third']
    results_dir = './results'
    
    # Dictionary to store data by stock symbol
    stock_data = defaultdict(list)
    
    # Process each quarterly directory
    for quarter in quarters:
        quarter_dir = os.path.join(results_dir, quarter)
        
        if not os.path.exists(quarter_dir):
            logging.warning(f"Directory {quarter_dir} does not exist")
            continue
            
        for filename in os.listdir(quarter_dir):
            if filename.endswith('.xml'):
                file_path = os.path.join(quarter_dir, filename)
                symbol = extract_stock_symbol(filename)
                
                logging.info(f"Processing {symbol} - {quarter} quarter")
                
                data = parse_xbrl_file(file_path)
                if data:
                    # Add quarter information and symbol
                    data['Quarter'] = quarter
                    data['Symbol'] = symbol
                    stock_data[symbol].append(data)
    
    # Calculate averages and create final dataframe
    final_data = []
    
    for symbol, quarters_data in stock_data.items():
        logging.info(f"Calculating averages for {symbol} with {len(quarters_data)} quarters of data")
        
        # Initialize a dict to store the averages
        avg_data = {'Symbol': symbol, 'QuartersAvailable': len(quarters_data)}
        
        # Calculate the average for each metric
        for metric in METRICS:
            values = [q[metric] for q in quarters_data if q.get(metric) is not None]
            if values:
                avg_data[f'Avg_{metric}'] = sum(values) / len(values)
            else:
                avg_data[f'Avg_{metric}'] = None
                
        # Add the quarters that were available
        available_quarters = [q['Quarter'] for q in quarters_data]
        avg_data['AvailableQuarters'] = ','.join(available_quarters)
        
        final_data.append(avg_data)
    
    # Create the final dataframe
    if final_data:
        df = pd.DataFrame(final_data)
        output_file = 'quarterly_averages.csv'
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved quarterly averages to {output_file}")
        print(f"Successfully processed {len(stock_data)} stocks and saved results to {output_file}")
    else:
        logging.warning("No data was extracted from the XBRL files")
        print("No data was extracted from the XML files")

if __name__ == "__main__":
    process_all_files()
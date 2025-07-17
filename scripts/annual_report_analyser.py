import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import PyPDF2
from transformers import pipeline
import logging
from collections import defaultdict
import textract  # For better text extraction from PDFs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

# Financial domain specific dictionaries
FINANCIAL_POSITIVE = [
    'growth', 'profit', 'increase', 'improved', 'strong', 'favorable', 'gain', 'success',
    'opportunity', 'efficient', 'leading', 'innovative', 'exceed', 'outperform', 'strengthen',
    'upside', 'advantage', 'positive', 'robust', 'enhancement', 'breakthrough', 'milestone'
]

FINANCIAL_NEGATIVE = [
    'loss', 'decline', 'decrease', 'weak', 'risk', 'challenge', 'difficult', 'adverse',
    'uncertain', 'liability', 'litigation', 'penalty', 'volatile', 'downturn', 'downward',
    'underperform', 'concern', 'threat', 'recession', 'deficit', 'deterioration', 'slowdown'
]

# Supply chain specific patterns
SUPPLY_CHAIN_PREFIXES = [
    'supplier', 'vendor', 'distributor', 'retailer', 'wholesaler', 'manufacturer',
    'logistics', 'provider', 'partner', 'sourcing', 'procurement', 'supply', 'contractor'
]

SUPPLY_CHAIN_PATTERNS = [
    r'(?:our|key|major|primary|strategic|preferred|authorized)\s+(?:supplier|vendor|distributor|partner)s?(?:\s+(?:include|are|is|includes|such as))?[\s:]+([A-Z][A-Za-z0-9\s\.\,&]+?)(?:\.|\,|and|which|\(|\)|\n)',
    r'(?:supplied|provided|manufactured|distributed|sourced)\s+by\s+([A-Z][A-Za-z0-9\s\.\,&]+?)(?:\.|\,|and|\(|\)|\n)',
    r'partnership\s+with\s+([A-Z][A-Za-z0-9\s\.\,&]+?)(?:\.|\,|and|\(|\)|\n)',
    r'([A-Z][A-Za-z0-9\s\.\,&]+?)(?:\s+is|\s+are|\s+has been|\s+have been)\s+our\s+(?:supplier|vendor|distributor|partner)'
]

BUSINESS_RELATIONSHIP_TERMS = [
    'supplier', 'vendor', 'distributor', 'partner', 'manufacturer', 'contractor',
    'alliance', 'collaboration', 'joint venture', 'outsource', 'procurement'
]

# Section-specific identifiers for financial reports
SECTION_IDENTIFIERS = {
    'executive_summary': ['management discussion', 'executive summary', 'company overview', 'business overview'],
    'risk_factors': ['risk factors', 'risks and uncertainties', 'principal risks', 'key risks'],
    'outlook': ['outlook', 'future outlook', 'forward looking', 'guidance', 'forecast'],
    'financial_results': ['financial results', 'financial performance', 'financial highlights', 'consolidated results'],
    'supply_chain': ['supply chain', 'procurement', 'sourcing', 'suppliers', 'vendors']
}

def extract_text_from_pdf(pdf_path):
    """Extract text from a local PDF file using multiple methods for better extraction."""
    try:
        # Try using textract first which has better extraction capabilities
        try:
            text = textract.process(pdf_path).decode('utf-8')
            return text
        except:
            logger.warning("Textract extraction failed, falling back to PyPDF2")
            
        # Fallback to PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def split_into_sections(text):
    """Split the annual report into relevant sections."""
    sections = defaultdict(str)
    current_section = "unknown"
    lines = text.split('\n')
    
    for line in lines:
        # Check if this line contains a section header
        line_lower = line.lower()
        matched = False
        
        for section_name, identifiers in SECTION_IDENTIFIERS.items():
            if any(identifier in line_lower for identifier in identifiers):
                current_section = section_name
                matched = True
                break
                
        # Add this line to the current section
        sections[current_section] += line + "\n"
        
    return sections

def extract_all_companies(text):
    """Extract potential company names using regex patterns only."""
    companies = set()
    
    # Pattern-based extraction for company names
    patterns = [
        r'([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)+)(?:\s+(?:Inc\.|Corp\.|Ltd\.|LLC|Limited|Company|Co\.|Group))',
        r'([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:Technologies|Solutions|Industries|Systems|Partners))',
        r'([A-Z][A-Za-z0-9&\s]{3,40}?)\s+(?:Inc\.|Corp\.|Ltd\.|LLC|Limited|Company|Co\.|Group|Technologies|Systems)'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            company = match.group(1).strip()
            if len(company) > 3:
                companies.add(company)

    return list(companies)


def identify_supply_chain_companies(text, all_companies):
    """Identify companies that are likely part of the supply chain."""
    supply_chain_companies = []
    relationship_evidence = {}
    sentences = sent_tokenize(text)
    
    # Check for supply chain relationship patterns
    for company in all_companies:
        evidence = []
        for sentence in sentences:
            if company in sentence:
                # Check if any supply chain prefix appears in the same sentence
                if any(prefix in sentence.lower() for prefix in SUPPLY_CHAIN_PREFIXES):
                    evidence.append(sentence)
                
                # Check if any business relationship term appears in the same sentence
                if any(term in sentence.lower() for term in BUSINESS_RELATIONSHIP_TERMS):
                    evidence.append(sentence)
        
        # If we have evidence, add this company to our supply chain list
        if evidence:
            supply_chain_companies.append(company)
            relationship_evidence[company] = evidence
            
    # Also use explicit pattern matching to find additional supply chain companies
    for pattern in SUPPLY_CHAIN_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            company = match.group(1).strip()
            if len(company) > 3:  # Filter out very short names
                supply_chain_companies.append(company)
                relationship_evidence[company] = [match.group(0)]
    
    # Return a dictionary with companies and their relationship evidence
    return {company: relationship_evidence.get(company, []) for company in supply_chain_companies}

def calculate_sentiment(text, section_type="general"):
    """Calculate financial sentiment with domain-specific adjustments."""
    # Base scores
    positive_score = 0
    negative_score = 0
    
    # Count financial sentiment words
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    positive_count = sum(1 for word in words if word in FINANCIAL_POSITIVE)
    negative_count = sum(1 for word in words if word in FINANCIAL_NEGATIVE)
    
    # Calculate base sentiment ratio
    total_words = len(words) if words else 1
    base_positive = positive_count / total_words * 100
    base_negative = negative_count / total_words * 100
    
    # Adjust based on section (risk sections are expected to be more negative)
    if section_type == "risk_factors":
        base_negative *= 0.8  # Reduce negative weight in risk sections
    elif section_type == "outlook":
        base_positive *= 1.2  # Increase positive weight in outlook sections
    
    # Calculate overall sentiment (-100 to 100 scale)
    sentiment_score = base_positive - base_negative
    confidence = (positive_count + negative_count) / total_words * 100
    
    # Advanced sentiment if transformers is available
    try:
        # Get sentences for more detailed analysis
        sentences = sent_tokenize(text)
        # Take a sample to avoid processing too much text
        sample_sentences = sentences[:min(100, len(sentences))]
        sample_text = " ".join(sample_sentences)
        
        # Use transformers sentiment analysis
        sentiment_analyzer = pipeline("sentiment-analysis")
        result = sentiment_analyzer(sample_text)
        
        if result[0]['label'] == 'POSITIVE':
            transformer_score = result[0]['score'] * 100
        else:
            transformer_score = -result[0]['score'] * 100
            
        # Combined score (70% transformers, 30% lexicon-based)
        sentiment_score = (transformer_score * 0.7) + (sentiment_score * 0.3)
    except:
        logger.warning("Transformers model not available, using lexicon-based sentiment only")
    
    return sentiment_score, confidence

def analyze_annual_report(report_text):
    """Perform detailed analysis on annual report text with section-specific processing."""
    if not report_text:
        logger.warning("Empty report text provided")
        return None
    
    # Split the report into sections
    sections = split_into_sections(report_text)
    
    # Extract all potential companies
    all_companies = extract_all_companies(report_text)
    
    # Identify supply chain companies with evidence
    supply_chain_info = identify_supply_chain_companies(report_text, all_companies)
    
    # Calculate section-specific sentiment
    section_sentiments = {}
    for section_name, section_text in sections.items():
        if section_text:
            sentiment_score, confidence = calculate_sentiment(section_text, section_name)
            section_sentiments[section_name] = {
                'sentiment_score': round(sentiment_score, 2),
                'confidence': round(confidence, 2)
            }
    
    # Calculate overall sentiment
    overall_sentiment, overall_confidence = calculate_sentiment(report_text)
    
    # Count risk and opportunity keywords
    words = word_tokenize(report_text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    risk_count = sum(1 for word in filtered_words if word in FINANCIAL_NEGATIVE)
    opportunity_count = sum(1 for word in filtered_words if word in FINANCIAL_POSITIVE)
    
    risk_percentage = (risk_count / len(filtered_words) * 100) if filtered_words else 0
    opportunity_percentage = (opportunity_count / len(filtered_words) * 100) if filtered_words else 0
    
    # Format supply chain companies with evidence
    formatted_supply_chain = []
    for company, evidence in supply_chain_info.items():
        formatted_supply_chain.append({
            'company': company,
            'evidence': evidence[:3]  # Limit to 3 pieces of evidence for brevity
        })
    
    return {
        'overall_sentiment_score': round(overall_sentiment, 2),
        'overall_confidence_score': round(overall_confidence, 2),
        'risk_keywords_percentage': round(risk_percentage, 2),
        'opportunity_keywords_percentage': round(opportunity_percentage, 2),
        'section_sentiments': section_sentiments,
        'supply_chain_companies': formatted_supply_chain
    }

def process_single_report(pdf_path, company_symbol=None):
    """Process a single annual report and save detailed results to CSV."""
    try:
        logger.info(f"Processing report: {pdf_path}")
        
        # Extract filename if company_symbol not provided
        if not company_symbol:
            company_symbol = os.path.basename(pdf_path).split('_')[0]
        
        # Extract text from report
        report_text = extract_text_from_pdf(pdf_path)
        
        if not report_text:
            logger.error(f"Failed to extract text from {pdf_path}")
            return False
        
        # Analyze report content
        analysis_results = analyze_annual_report(report_text)
        
        if not analysis_results:
            logger.error(f"Failed to analyze report content for {pdf_path}")
            return False
        
        # Create main results DataFrame
        main_df = pd.DataFrame([{
            'file_name': os.path.basename(pdf_path),
            'company_symbol': company_symbol,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_sentiment_score': analysis_results['overall_sentiment_score'],
            'overall_confidence_score': analysis_results['overall_confidence_score'],
            'risk_keywords_percentage': analysis_results['risk_keywords_percentage'],
            'opportunity_keywords_percentage': analysis_results['opportunity_keywords_percentage']
        }])
        
        # Create section sentiments DataFrame
        section_rows = []
        for section_name, sentiment_data in analysis_results['section_sentiments'].items():
            section_rows.append({
                'company_symbol': company_symbol,
                'section_name': section_name,
                'sentiment_score': sentiment_data['sentiment_score'],
                'confidence': sentiment_data['confidence']
            })
        
        section_df = pd.DataFrame(section_rows) if section_rows else pd.DataFrame()
        
        # Create supply chain companies DataFrame
        supply_chain_rows = []
        for sc_info in analysis_results['supply_chain_companies']:
            supply_chain_rows.append({
                'company_symbol': company_symbol,
                'supply_chain_company': sc_info['company'],
                'evidence': '|||'.join(sc_info['evidence'][:3]) if sc_info['evidence'] else ''
            })
        
        supply_chain_df = pd.DataFrame(supply_chain_rows) if supply_chain_rows else pd.DataFrame()
        
        # Save results to CSV files
        main_output_file = f"{company_symbol}_sentiment_analysis.csv"
        section_output_file = f"{company_symbol}_section_sentiments.csv"
        supply_chain_output_file = f"{company_symbol}_supply_chain.csv"
        
        main_df.to_csv(main_output_file, index=False)
        
        if not section_df.empty:
            section_df.to_csv(section_output_file, index=False)
            
        if not supply_chain_df.empty:
            supply_chain_df.to_csv(supply_chain_output_file, index=False)
        
        logger.info(f"Main results saved to {main_output_file}")
        logger.info(f"Section sentiments saved to {section_output_file}")
        logger.info(f"Supply chain information saved to {supply_chain_output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing report: {e}")
        return False

def main():
    """Main function to run the enhanced analysis on a single file."""
    pdf_file = "KINGFA.pdf"
    
    if not os.path.exists(pdf_file):
        logger.error(f"File not found: {pdf_file}")
        return
    
    logger.info("Starting enhanced annual report analysis")
    success = process_single_report(pdf_file, "KINGFA")
    
    if success:
        logger.info("Completed annual report analysis successfully")
    else:
        logger.error("Failed to complete analysis")

if __name__ == "__main__":
    main()
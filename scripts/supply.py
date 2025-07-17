import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import PyPDF2
import logging
import textract  # For better text extraction from PDFs
import json
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

# Section-specific identifiers for financial reports (simplified for supply chain focus)
SECTION_IDENTIFIERS = {
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
    """Split the annual report into relevant sections focused on supply chain."""
    sections = {}
    for section_name, identifiers in SECTION_IDENTIFIERS.items():
        section_text = ""
        for identifier in identifiers:
            # Simplified: Find any occurrence of the keywords to mark the section
            if identifier in text.lower():
                section_text = text  # Just grab the whole text if supply chain related keywords are found
                sections[section_name] = section_text
                break  # Stop searching once a keyword is found
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

def analyze_annual_report_for_supply_chain(report_text):
    """Analyze annual report text to extract supply chain companies."""
    if not report_text:
        logger.warning("Empty report text provided")
        return None
    
    # Split the report into sections (only supply chain relevant sections)
    sections = split_into_sections(report_text)
    
    # If no supply chain section is found, return empty results
    if not sections:
        logger.info("No supply chain related content found.")
        return {'supply_chain_companies': []}
    
    supply_chain_text = sections.get('supply_chain', report_text) # Use entire report if no specific supply chain section is found
    
    # Extract all potential companies
    all_companies = extract_all_companies(supply_chain_text)
    
    # Identify supply chain companies with evidence
    supply_chain_info = identify_supply_chain_companies(supply_chain_text, all_companies)
    
    # Format supply chain companies with evidence
    formatted_supply_chain = []
    for company, evidence in supply_chain_info.items():
        formatted_supply_chain.append({
            'company': company,
            'evidence': evidence[:3]  # Limit to 3 pieces of evidence for brevity
        })
    
    return {
        'supply_chain_companies': formatted_supply_chain
    }

def process_single_report(pdf_path, company_symbol=None):
    """Process a single annual report to extract supply chain companies and save results to JSON."""
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
        
        # Analyze report content for supply chain companies
        analysis_results = analyze_annual_report_for_supply_chain(report_text)
        
        if not analysis_results:
            logger.error(f"Failed to analyze report content for {pdf_path}")
            return False
        
        # Prepare data for JSON output
        output_data = {
            'company_symbol': company_symbol,
            'supply_chain_companies': []
        }
        
        for sc_info in analysis_results['supply_chain_companies']:
            output_data['supply_chain_companies'].append({
                'supply_chain_company': sc_info['company'],
                'evidence': sc_info['evidence'][:3] if sc_info['evidence'] else []
            })
        
        # Save results to JSON file
        output_file = f"{company_symbol}_supply_chain.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f"Supply chain information saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing report: {e}")
        return False

def main():
    """Main function to run the supply chain analysis on a single file."""
    pdf_file = "KINGFA.pdf"
    
    if not os.path.exists(pdf_file):
        logger.error(f"File not found: {pdf_file}")
        return
    
    logger.info("Starting supply chain analysis")
    success = process_single_report(pdf_file, "KINGFA")
    
    if success:
        logger.info("Completed supply chain analysis successfully")
    else:
        logger.error("Failed to complete analysis")

if __name__ == "__main__":
    main()

"""
Script to prepare and validate data for vector store ingestion.
Loads raw data, cleans it, and prepares documents for embedding.
"""

import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_clean_data():
    """Load raw CSV files and clean data."""
    logger.info("Loading raw data...")
    
    # Create directories if needed
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Load financial data
    financial_file = "data/raw/financial_data.csv"
    if Path(financial_file).exists():
        df_financial = pd.read_csv(financial_file)
        logger.info(f"Loaded {len(df_financial)} financial records")
        
        # Basic cleaning
        df_financial = df_financial.dropna(subset=['company_name'])
        df_financial.fillna(0, inplace=True)
        
        # Save cleaned
        df_financial.to_csv("data/processed/financial_data_clean.csv", index=False)
        logger.info(f"Cleaned financial data: {len(df_financial)} records")
    else:
        logger.warning(f"{financial_file} not found. Creating mock data...")
        create_mock_financial_data()
    
    # Load news/documents
    news_file = "data/raw/financial_news.csv"
    if Path(news_file).exists():
        df_news = pd.read_csv(news_file)
        logger.info(f"Loaded {len(df_news)} news documents")
        
        # Clean
        df_news = df_news.dropna(subset=['title', 'content'])
        
        # Save cleaned
        df_news.to_csv("data/processed/news_clean.csv", index=False)
        logger.info(f"Cleaned news data: {len(df_news)} documents")
    else:
        logger.warning(f"{news_file} not found. Creating mock data...")
        create_mock_news_data()


def create_mock_financial_data():
    """Create mock financial data for testing."""
    logger.info("Creating mock financial data...")
    
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'WMT']
    
    records = []
    for company in companies:
        records.append({
            'company_name': company,
            'pe_ratio': np.random.uniform(10, 40),
            'debt_equity': np.random.uniform(0.2, 2.0),
            'current_ratio': np.random.uniform(0.5, 3.0),
            'roe': np.random.uniform(-0.5, 1.5),
            'beta': np.random.uniform(0.5, 2.5),
            'revenue_growth': np.random.uniform(-0.2, 0.5),
            'sector_risk': np.random.uniform(0.1, 0.9),
            'date': datetime.now().date()
        })
    
    df = pd.DataFrame(records)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/financial_data.csv", index=False)
    logger.info(f"Created mock financial data: {len(df)} companies")


def create_mock_news_data():
    """Create mock news/document data for testing."""
    logger.info("Creating mock news data...")
    
    news_templates = [
        {
            'title': '{company} Q3 2024 Earnings Report',
            'content': '{company} reported strong Q3 results with revenue growth of 15.3% YoY. Gross margin improved to 46.2%, up from 44.8% in Q3 2023. EPS grew 18% YoY. The company guided for 12-15% growth in Q4.',
            'source': 'Investor Relations',
        },
        {
            'title': 'Technology Sector Analysis - Q4 2024',
            'content': 'The technology sector continues to benefit from AI adoption and cloud computing growth. Major players are investing heavily in R&D. Analyst consensus remains positive with 12-month price targets suggesting 20-25% upside.',
            'source': 'Morgan Stanley Research',
        },
        {
            'title': '{company} Stock Rating: BUY',
            'content': 'We initiate coverage of {company} with a BUY rating and $250 price target. The company has a strong competitive moat, excellent cash generation, and attractive growth prospects. Valuation appears reasonable given growth rates.',
            'source': 'Goldman Sachs Equity Research',
        },
        {
            'title': 'Interest Rate Impact on Technology Stocks',
            'content': 'With the Fed pausing rate hikes, technology stocks are rebounding. Lower rates increase the present value of future earnings. We expect outperformance of growth stocks in this environment.',
            'source': 'Bloomberg Intelligence',
        },
        {
            'title': '{company} Competitive Position Strengthens',
            'content': 'Recent market share gains and product launches position {company} well for 2025. The company is expanding into high-margin services, which should support margin expansion.',
            'source': 'Barclays Equity Research',
        },
    ]
    
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    records = []
    
    for template in news_templates:
        for company in companies:
            records.append({
                'title': template['title'].format(company=company),
                'content': template['content'].format(company=company),
                'source': template['source'],
                'company': company,
                'date': datetime.now().date(),
                'relevance_score': np.random.uniform(0.7, 1.0)
            })
    
    df = pd.DataFrame(records)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/financial_news.csv", index=False)
    logger.info(f"Created mock news data: {len(df)} documents")


def prepare_documents_for_embedding():
    """Prepare documents for vector embedding."""
    logger.info("Preparing documents for embedding...")
    
    # Load cleaned news
    news_file = "data/processed/news_clean.csv"
    if not Path(news_file).exists():
        logger.warning("Cleaned news file not found. Running data cleaning...")
        load_and_clean_data()
    
    df_news = pd.read_csv(news_file)
    
    # Create document objects
    documents = []
    for idx, row in df_news.iterrows():
        doc = {
            'id': f"doc_{idx}",
            'title': row.get('title', 'Unknown'),
            'content': row.get('content', ''),
            'source': row.get('source', 'Unknown'),
            'company': row.get('company', 'Unknown').upper(),
            'date': str(row.get('date', '')),
        }
        documents.append(doc)
    
    # Save as JSON
    output_file = "data/processed/documents.json"
    with open(output_file, 'w') as f:
        json.dump(documents, f, indent=2)
    
    logger.info(f"Prepared {len(documents)} documents for embedding")
    return documents


def validate_data():
    """Validate prepared data."""
    logger.info("Validating prepared data...")
    
    # Check financial data
    if Path("data/processed/financial_data_clean.csv").exists():
        df_fin = pd.read_csv("data/processed/financial_data_clean.csv")
        logger.info(f"✓ Financial data: {len(df_fin)} records with {len(df_fin.columns)} columns")
        logger.info(f"  Columns: {list(df_fin.columns)}")
    
    # Check documents
    if Path("data/processed/documents.json").exists():
        with open("data/processed/documents.json") as f:
            docs = json.load(f)
        logger.info(f"✓ Documents: {len(docs)} documents")
        logger.info(f"  Sample document keys: {list(docs[0].keys()) if docs else 'N/A'}")
    
    logger.info("✓ Data validation complete")


if __name__ == "__main__":
    logger.info("Starting data preparation...")
    
    # Execute pipeline
    load_and_clean_data()
    documents = prepare_documents_for_embedding()
    validate_data()
    
    logger.info("✅ Data preparation complete!")
    logger.info(f"Output location: data/processed/")

"""
RAG Pipeline: Orchestrates Retrieval-Augmented Generation workflow.
Combines vector search, LLM generation, and ML predictions.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from utils.logger import get_logger
from utils.llm_wrapper import get_llm_client
from vector_store.embeddings import get_embedding_manager
from vector_store.faiss_store import get_vector_store
from ml_models.risk_classifier import get_risk_classifier
from config.settings import get_settings

logger = get_logger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for financial analysis.
    
    Workflow:
    1. Embed user query
    2. Retrieve relevant documents from vector store
    3. Run ML model for risk prediction
    4. Generate LLM response with context
    5. Combine and format output
    """
    
    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing RAG Pipeline...")
        self.settings = get_settings()
        self.embedding_manager = get_embedding_manager()
        self.vector_store = get_vector_store()
        self.llm_client = get_llm_client()
        self.risk_classifier = get_risk_classifier()
        logger.info("RAG Pipeline initialized successfully")
    
    def retrieve_context(
        self,
        query: str,
        company_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query/question
            company_name: Company to filter by (optional)
            top_k: Number of top documents to retrieve
            
        Returns:
            List[Dict]: Retrieved documents with scores
        """
        logger.info(f"Retrieving top-{top_k} documents for query: {query[:50]}...")
        
        # Embed query
        query_embedding = self.embedding_manager.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filters={"company": company_name.upper()} if company_name else None
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            str: Formatted context
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[{i}] {doc.get('title', 'Document')}\n"
                f"Source: {doc.get('source', 'Unknown')}\n"
                f"Date: {doc.get('date', 'Unknown')}\n"
                f"Content: {doc.get('content', '')[:500]}...\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def predict_risk(
        self,
        company_name: str,
        company_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Predict investment risk using ML model.
        
        Args:
            company_name: Company name
            company_metrics: Financial metrics (P/E, Debt/Equity, etc.)
            
        Returns:
            Dict: Risk prediction with level, confidence, importances
        """
        logger.info(f"Predicting risk for {company_name}...")
        
        try:
            risk_level, confidence, importances = self.risk_classifier.predict(
                company_metrics
            )
            
            result = {
                "risk_level": risk_level,
                "confidence": float(confidence),
                "importances": importances,
                "metrics": company_metrics
            }
            
            logger.info(f"Risk prediction: {risk_level} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return {
                "risk_level": "UNKNOWN",
                "confidence": 0.0,
                "importances": {},
                "error": str(e)
            }
    
    def generate_report(
        self,
        company_name: str,
        investment_amount: float,
        time_horizon_months: int,
        company_metrics: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Generate complete investment report.
        
        Args:
            company_name: Company to analyze
            investment_amount: Investment amount in USD
            time_horizon_months: Investment time horizon
            company_metrics: Optional financial metrics for ML
            
        Returns:
            Dict: Complete report with analysis and recommendations
        """
        logger.info(
            f"Generating report for {company_name} (${investment_amount}, "
            f"{time_horizon_months}mo)"
        )
        
        start_time = datetime.now()
        
        # ====== Step 1: Build query ======
        query = (
            f"Provide investment analysis for {company_name}. "
            f"Investment: ${investment_amount:,.0f}, "
            f"Time horizon: {time_horizon_months} months. "
            f"Consider financial strength, growth prospects, risks, and valuation."
        )
        
        # ====== Step 2: Retrieve context ======
        documents = self.retrieve_context(query, company_name)
        context = self.format_context(documents)
        
        # ====== Step 3: ML risk prediction ======
        if company_metrics:
            risk_prediction = self.predict_risk(company_name, company_metrics)
        else:
            risk_prediction = {
                "risk_level": "UNKNOWN",
                "confidence": 0.0,
                "importances": {}
            }
        
        # ====== Step 4: Generate LLM analysis ======
        system_prompt = (
            "You are an expert financial analyst with 25 years of experience. "
            "Provide accurate, data-driven investment analysis. "
            "Base all statements on provided documents only. "
            "Cite specific sources using [1], [2], etc. format. "
            "Include risk assessment, growth potential, and clear recommendation. "
            "Tone: Professional, analytical, risk-aware. "
            "Format: Structured sections with clear headings."
        )
        
        try:
            llm_analysis = self.llm_client.generate(
                system_prompt=system_prompt,
                user_message=query,
                context=context
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            llm_analysis = f"Error generating analysis: {str(e)}"
        
        # ====== Step 5: Compile final report ======
        report = {
            "metadata": {
                "company_name": company_name.upper(),
                "investment_amount": investment_amount,
                "time_horizon_months": time_horizon_months,
                "analysis_date": datetime.now().isoformat(),
                "model_version": "1.0"
            },
            "risk_assessment": risk_prediction,
            "analysis": llm_analysis,
            "sources": [
                {
                    "id": i + 1,
                    "title": doc.get("title", "Unknown"),
                    "source": doc.get("source", "Unknown"),
                    "date": doc.get("date", "Unknown")
                }
                for i, doc in enumerate(documents)
            ],
            "disclaimer": (
                "This analysis is AI-generated for educational purposes only. "
                "Not financial advice. Consult licensed financial advisors."
            ),
            "processing_time_seconds": (
                datetime.now() - start_time
            ).total_seconds()
        }
        
        logger.info(
            f"Report generated in {report['processing_time_seconds']:.2f}s"
        )
        
        return report
    
    def format_report_text(self, report: Dict) -> str:
        """
        Format report dictionary as readable text.
        
        Args:
            report: Report dictionary
            
        Returns:
            str: Formatted report text
        """
        meta = report["metadata"]
        risk = report["risk_assessment"]
        
        text = f"""
╔════════════════════════════════════════════════════════════════╗
║           FINANCIAL RESEARCH & INVESTMENT REPORT               ║
╚════════════════════════════════════════════════════════════════╝

Company: {meta['company_name']}
Investment Amount: ${meta['investment_amount']:,.2f}
Time Horizon: {meta['time_horizon_months']} months
Analysis Date: {meta['analysis_date'][:10]}

┌─ RISK ASSESSMENT ──────────────────────────────────────────────┐
│ Risk Level: {risk.get('risk_level', 'UNKNOWN')}
│ Confidence: {risk.get('confidence', 0):.1%}
│ 
│ Top Risk Factors:
"""
        
        for factor, importance in list(
            risk.get('importances', {}).items()
        )[:3]:
            text += f"│   • {factor}: {importance:.2%}\n"
        
        text += f"""│
└───────────────────────────────────────────────────────────────┘

┌─ AI ANALYSIS ──────────────────────────────────────────────────┐
{report['analysis']}
└───────────────────────────────────────────────────────────────┘

┌─ SOURCES ──────────────────────────────────────────────────────┐
"""
        
        for source in report['sources']:
            text += (
                f"│ [{source['id']}] {source['title']}\n"
                f"│     Source: {source['source']}\n"
            )
        
        text += f"""│
└───────────────────────────────────────────────────────────────┘

⚠️  DISCLAIMER: {report['disclaimer']}

Processing Time: {report['processing_time_seconds']:.2f}s
════════════════════════════════════════════════════════════════
"""
        
        return text


# Global instance
_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline (singleton)."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


if __name__ == "__main__":
    # Test RAG pipeline
    pipeline = get_rag_pipeline()
    
    # Mock financial metrics for testing
    metrics = {
        "pe_ratio": 28.4,
        "debt_equity": 0.89,
        "current_ratio": 1.5,
        "roe": 0.89,
        "beta": 1.15,
        "revenue_growth": 0.153,
        "sector_risk": 0.3
    }
    
    report = pipeline.generate_report(
        company_name="Apple",
        investment_amount=10000,
        time_horizon_months=24,
        company_metrics=metrics
    )
    
    print(pipeline.format_report_text(report))

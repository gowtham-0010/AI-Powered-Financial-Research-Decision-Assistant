"""
FastAPI Backend API for AI Financial Research Assistant.
Provides REST endpoints for report generation and health checks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies
from config.settings import get_settings
from backend.rag_pipeline import get_rag_pipeline
from utils.validators import ReportRequest, ReportResponse, HealthCheckResponse

# Initialize app
app = FastAPI(
    title="AI Financial Research Assistant",
    description="RAG + ML-powered investment analysis API",
    version="1.0.0"
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ HEALTH CHECK ============
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of all system components.
    """
    logger.info("Health check requested")
    
    return HealthCheckResponse(
        status="healthy",
        llm_available=True,  # In production, actually test connectivity
        vector_db_available=True,
        ml_model_available=True,
        timestamp=datetime.now().isoformat()
    )

# ============ REPORT GENERATION ============
@app.post("/api/generate-report", response_model=Dict)
async def generate_report(request: ReportRequest):
    """
    Generate investment report for a company.
    
    Args:
        request: ReportRequest with company_name, investment_amount, time_horizon_months
        
    Returns:
        Dict: Comprehensive investment report
        
    Example:
        {
            "company_name": "APPLE",
            "investment_amount": 10000,
            "time_horizon_months": 24,
            "risk_assessment": {
                "risk_level": "LOW",
                "confidence": 0.87
            },
            "analysis": "Detailed AI analysis...",
            "sources": [...]
        }
    """
    try:
        logger.info(
            f"Report generation requested: {request.company_name}, "
            f"${request.investment_amount:,.0f}, {request.time_horizon_months}mo"
        )
        
        # Mock financial metrics (in production, fetch from API)
        metrics = {
            "pe_ratio": 28.4,
            "debt_equity": 0.89,
            "current_ratio": 1.5,
            "roe": 0.89,
            "beta": 1.15,
            "revenue_growth": 0.153,
            "sector_risk": 0.3
        }
        
        # Generate report
        pipeline = get_rag_pipeline()
        report = pipeline.generate_report(
            company_name=request.company_name,
            investment_amount=request.investment_amount,
            time_horizon_months=request.time_horizon_months,
            company_metrics=metrics
        )
        
        logger.info(f"Report generated successfully for {request.company_name}")
        
        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ BATCH ANALYSIS ============
@app.post("/api/batch-analysis")
async def batch_analysis(companies: List[str]):
    """
    Generate reports for multiple companies.
    
    Args:
        companies: List of company names
        
    Returns:
        Dict: Reports for all companies
    """
    try:
        logger.info(f"Batch analysis requested for {len(companies)} companies")
        
        results = []
        pipeline = get_rag_pipeline()
        
        for company in companies:
            try:
                report = pipeline.generate_report(
                    company_name=company,
                    investment_amount=10000,
                    time_horizon_months=24,
                    company_metrics={}
                )
                results.append({
                    "company_name": company,
                    "success": True,
                    "report": report
                })
            except Exception as e:
                logger.error(f"Failed to generate report for {company}: {e}")
                results.append({
                    "company_name": company,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total_requested": len(companies),
            "total_successful": sum(1 for r in results if r["success"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ SYSTEM INFO ============
@app.get("/api/system-info")
async def system_info():
    """Get system configuration and statistics."""
    try:
        from vector_store.faiss_store import get_vector_store
        
        vector_store = get_vector_store()
        vs_info = vector_store.info()
        
        return {
            "system": {
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "vector_db_type": settings.vector_db_type,
            },
            "vector_store": vs_info,
            "ml_model": {
                "type": "Random Forest Classifier",
                "features": 7,
                "classes": ["LOW", "MEDIUM", "HIGH"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ERROR HANDLERS ============
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return {
        "error": "Validation Error",
        "detail": str(exc)
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )

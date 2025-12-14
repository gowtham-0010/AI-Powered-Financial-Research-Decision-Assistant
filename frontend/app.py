"""
Main Streamlit Application for AI Financial Research Assistant.
Provides interactive web interface for investment analysis.
"""

import sys
from pathlib import Path

# Fix Python path for Streamlit execution from subdirectory
# Add project root to PYTHONPATH so imports work correctly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import logging

# Configure page
st.set_page_config(
    page_title="AI Financial Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
""", unsafe_allow_html=True)

# Initialize session state
if 'report' not in st.session_state:
    st.session_state.report = None

if 'loading' not in st.session_state:
    st.session_state.loading = False

# ============ SETUP LOGGING ============
from utils.logger import get_logger

logger = get_logger(__name__)

# ============ MAIN UI ============
st.title("📊 AI-Powered Financial Research Assistant")

st.markdown(
    "Generate AI-driven investment analysis reports combining RAG, ML, and LLM"
)

# ============ SIDEBAR - INPUT FORM ============
with st.sidebar:
    st.header("Investment Parameters")
    
    company_name = st.text_input(
        "Company Name",
        value="Apple",
        help="Enter the company to analyze"
    )
    
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=100,
        max_value=1_000_000_000,
        value=10000,
        step=1000,
        help="Amount you plan to invest"
    )
    
    time_horizon = st.slider(
        "Time Horizon (months)",
        min_value=1,
        max_value=120,
        value=24,
        step=1,
        help="Investment duration"
    )
    
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        options=["Low", "Medium", "High"],
        index=1,
        help="Your comfort with volatility"
    )
    
    st.divider()
    
    # Generate Report Button
    if st.button("🔍 Generate Report", use_container_width=True, type="primary"):
        st.session_state.loading = True
        
        try:
            # Import pipeline
            from backend.rag_pipeline import get_rag_pipeline
            
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
            with st.spinner("Generating report..."):
                pipeline = get_rag_pipeline()
                report = pipeline.generate_report(
                    company_name=company_name,
                    investment_amount=investment_amount,
                    time_horizon_months=time_horizon,
                    company_metrics=metrics
                )
            
            st.session_state.report = report
            st.success("✅ Report generated successfully!")
            logger.info(f"Report generated for {company_name}")
            
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            logger.error(f"Report generation failed: {e}")
        
        finally:
            st.session_state.loading = False

# ============ MAIN CONTENT ============
if st.session_state.report:
    report = st.session_state.report
    meta = report["metadata"]
    risk = report["risk_assessment"]
    
    # Executive Summary
    st.header("📈 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Company",
            meta["company_name"],
            delta="Listed on NYSE/NASDAQ"
        )
    
    with col2:
        st.metric(
            "Investment",
            f"${meta['investment_amount']:,.0f}",
            delta=f"{meta['time_horizon_months']}mo horizon"
        )
    
    with col3:
        risk_color = "🟢" if risk["risk_level"] == "LOW" else (
            "🟡" if risk["risk_level"] == "MEDIUM" else "🔴"
        )
        st.metric(
            "Risk Level",
            f"{risk_color} {risk['risk_level']}",
            delta=f"{risk['confidence']:.1%} confidence"
        )
    
    with col4:
        st.metric(
            "Analysis Date",
            meta['analysis_date'][:10],
            delta=f"{report['processing_time_seconds']:.2f}s generation time"
        )
    
    st.divider()
    
    # Risk Assessment
    st.header("⚠️ Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Risk Level**: {risk['risk_level']}")
        st.write(f"**Confidence**: {risk['confidence']:.1%}")
        st.write("**Financial Metrics**:")
        
        for metric, value in risk.get("metrics", {}).items():
            if isinstance(value, float):
                st.write(f" • {metric}: {value:.2f}")
    
    with col2:
        st.write("**Top Risk Factors**:")
        
        importances = sorted(
            risk.get('importances', {}).items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for factor, importance in importances[:5]:
            st.write(f" • {factor.replace('_', ' ').title()}: {importance:.1%}")
    
    st.divider()
    
    # AI Analysis
    st.header("🤖 AI-Generated Analysis")
    
    with st.expander("View Full Analysis", expanded=True):
        st.write(report["analysis"])
    
    st.divider()
    
    # Sources
    st.header("📚 Sources & References")
    
    for source in report["sources"]:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**[{source['id']}] {source['title']}**")
                st.caption(f"Source: {source['source']} | Date: {source['date']}")
            
            with col2:
                st.badge(f"Source {source['id']}", key=f"source_{source['id']}")
    
    st.divider()
    
    # Disclaimer
    st.warning(f"⚠️ **Disclaimer**: {report['disclaimer']}")
    
    # Download Options
    st.subheader("📥 Download Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON Download
        json_str = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📄 Download as JSON",
            data=json_str,
            file_name=f"{meta['company_name']}_analysis_{meta['analysis_date'][:10]}.json",
            mime="application/json"
        )
    
    with col2:
        # Text Download
        from backend.rag_pipeline import RAGPipeline
        
        text_report = RAGPipeline().format_report_text(report)
        st.download_button(
            label="📋 Download as TXT",
            data=text_report,
            file_name=f"{meta['company_name']}_analysis_{meta['analysis_date'][:10]}.txt",
            mime="text/plain"
        )

else:
    # Default view
    st.info(
        "👈 **Enter investment parameters in the sidebar and click "
        "'Generate Report' to analyze a company**"
    )
    
    # Show example companies
    st.subheader("📊 Example Companies")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tech", "AAPL, MSFT, GOOGL")
        st.caption("Technology sector leaders")
    
    with col2:
        st.metric("Finance", "JPM, BAC, WFC")
        st.caption("Banking and financial services")
    
    with col3:
        st.metric("Healthcare", "JNJ, UNH, PFE")
        st.caption("Healthcare and pharmaceuticals")
    
    st.divider()
    
    # Feature Overview
    st.subheader("✨ How It Works")
    
    st.markdown("""
1. **Input Parameters**: Enter company name, investment amount, and time horizon
2. **Vector Search**: RAG retrieves relevant financial documents
3. **ML Analysis**: Risk classifier predicts investment risk
4. **LLM Generation**: Generates comprehensive analysis with sources
5. **Report**: Combines risk prediction + AI analysis + citations
    """)
    
    st.divider()
    
    # Technology Stack
    st.subheader("🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
**Frontend**
- Streamlit
- Plotly
- Pandas
        """)
    
    with col2:
        st.markdown("""
**Backend**
- FastAPI
- LangChain
- Pydantic
        """)
    
    with col3:
        st.markdown("""
**AI/ML**
- OpenAI API
- FAISS
- Scikit-Learn
        """)

# Footer
st.divider()

st.caption(
    "AI Financial Research Assistant © 2024 | "
    "For educational purposes only | Not financial advice"
)
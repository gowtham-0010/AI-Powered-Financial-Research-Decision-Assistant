"""
Input validation utilities using Pydantic.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ReportRequest(BaseModel):
    """Validated request for investment report generation."""
    
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Company name to analyze"
    )
    investment_amount: float = Field(
        ...,
        gt=0,
        le=1_000_000_000,
        description="Investment amount in USD"
    )
    time_horizon_months: int = Field(
        ...,
        ge=1,
        le=120,
        description="Investment time horizon in months (1-120)"
    )
    risk_tolerance: Optional[str] = Field(
        default="medium",
        description="Investor risk tolerance"
    )
    
    @field_validator('company_name')
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        """Ensure company name is alphabetic."""
        if not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError('Company name must contain only letters and spaces')
        return v.strip().upper()
    
    @field_validator('investment_amount')
    @classmethod
    def validate_amount(cls, v: float) -> float:
        """Ensure minimum investment."""
        if v < 100:
            raise ValueError('Minimum investment is $100')
        return v

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    llm_available: bool
    vector_db_available: bool
    ml_model_available: bool
    timestamp: str

class ReportResponse(BaseModel):
    """Response model for report generation."""
    company_name: str
    investment_amount: float
    time_horizon_months: int
    risk_level: str
    confidence: float
    analysis: str
    sources: list
    timestamp: str

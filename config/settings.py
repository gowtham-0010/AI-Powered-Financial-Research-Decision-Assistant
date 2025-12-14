"""
Application settings management using Pydantic.
Loads configuration from environment variables (.env file).
Supports multiple LLM providers: mock, huggingface, openai.
"""

from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    Uses Pydantic for type validation and .env file support.
    """

    # ============ LLM CONFIGURATION ============
    # LLM Provider selection (mock, huggingface, openai)
    llm_provider: str = Field(
        default="mock",
        description="LLM provider: mock (offline), huggingface (free), openai (production)"
    )

    # OpenAI Configuration
    llm_api_key: str = Field(
        default="sk-",
        description="OpenAI API key (required if llm_provider=openai)"
    )
    llm_model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model name (OpenAI)"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation (0-2)"
    )
    llm_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=4000,
        description="Maximum tokens to generate per request"
    )

    # Hugging Face Configuration
    huggingface_api_token: str = Field(
        default="hf_",
        description="Hugging Face API token (required if llm_provider=huggingface)"
    )

    # ============ VECTOR DATABASE CONFIGURATION ============
    vector_db_type: str = Field(
        default="faiss",
        description="Vector database type: faiss or chroma"
    )
    vector_db_path: str = Field(
        default="./vector_store/database",
        description="Path to vector database files"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence Transformers model for embeddings"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors"
    )

    # ============ DATA CONFIGURATION ============
    data_path: str = Field(
        default="./data/raw",
        description="Path to raw data files"
    )
    processed_data_path: str = Field(
        default="./data/processed",
        description="Path to processed data files"
    )
    financial_data_csv: str = Field(
        default="financial_data.csv",
        description="Financial data CSV filename"
    )
    news_data_csv: str = Field(
        default="financial_news.csv",
        description="Financial news CSV filename"
    )

    # ============ ML MODEL CONFIGURATION ============
    ml_model_path: str = Field(
        default="./ml_models/trained_models",
        description="Path to trained ML models"
    )
    risk_model_name: str = Field(
        default="risk_classifier_model.pkl",
        description="Risk classifier model filename"
    )
    feature_scaler_name: str = Field(
        default="feature_scaler.pkl",
        description="Feature scaler filename"
    )

    # ============ API CONFIGURATION ============
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host address"
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="FastAPI port"
    )
    api_debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        description="CORS allowed origins"
    )

    # ============ LOGGING CONFIGURATION ============
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string for logging.Formatter"
    )
    log_file: str = Field(
        default="./logs/app.log",
        description="Log file path"
    )

    # ============ STREAMLIT CONFIGURATION ============
    streamlit_theme: str = Field(
        default="dark",
        description="Streamlit theme (light, dark)"
    )
    streamlit_server_headless: bool = Field(
        default=True,
        description="Run Streamlit in headless mode"
    )
    streamlit_server_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Streamlit server port"
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False
        validate_default = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get application settings (cached singleton).
    
    Returns:
        Settings: Singleton instance of application settings
    """
    return Settings()
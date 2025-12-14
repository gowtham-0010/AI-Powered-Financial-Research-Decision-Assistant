"""
Logging configuration and utilities.
Sets up structured logging with file and console handlers.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from config.settings import get_settings

def setup_logging():
    """
    Configure application logging with both file and console handlers.
    Creates logs directory if it doesn't exist.
    """
    settings = get_settings()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter
    formatter = logging.Formatter(settings.log_format)
    
    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Initialize logger
logger = setup_logging()

def get_logger(name: str) -> logging.Logger:
    """
    Get logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logging configured successfully")
    log.debug("Debug level message")
    log.warning("Warning level message")
    log.error("Error level message")
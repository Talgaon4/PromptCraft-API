# prompt_optimizer/utils/logging.py

import logging
import os
import sys

def setup_logging(log_dir='./logs', log_level=logging.INFO):
    """Configure logging for the application.
    
    Args:
        log_dir: Directory where log files will be stored
        log_level: Logging level to use
        
    Returns:
        Configured root logger
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "prompt_optimizer.log"))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress verbose logs from libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger
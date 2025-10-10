# utils/logger.py

"""
Production-ready logging with rotation and structured formatting.
FIXED: Added RotatingFileHandler to prevent disk space issues.
"""

import logging
import os
from typing import Optional
from logging.handlers import RotatingFileHandler


def get_logger(
    name: str,
    log_level: str = 'INFO',
    console: bool = True,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB per file
    backup_count: int = 5  # Keep 5 backup files
) -> logging.Logger:
    """
    Get or create a logger with console and/or file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Enable console output
        log_file: Log filename (relative to logs/ directory)
        max_bytes: Max log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    
    Returns:
        Configured logger instance
    
    Features:
        - Automatic log rotation (prevents disk fill)
        - Separate console and file formatters
        - Thread-safe
        - Prevents duplicate handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger
    
    # Console handler (stdout)
    if console:
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            "{asctime} | {name} | {levelname} | {message}",
            style='{',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if log_file:
        base_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        base_log_dir = os.path.abspath(base_log_dir)
        log_path = os.path.join(base_log_dir, log_file)
        
        # Create logs directory if not exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # ✅ FIX: Use RotatingFileHandler instead of FileHandler
        file_handler = RotatingFileHandler(
            log_path,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Detailed format for file logs
        file_format = logging.Formatter(
            "{asctime} | {levelname:8s} | {name}:{funcName}:L{lineno} | {message}",
            style='{',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"File logging enabled: {log_path} (max={max_bytes/1024/1024:.1f}MB, backups={backup_count})")
    
    return logger


def get_structured_logger(
    name: str,
    log_level: str = 'INFO',
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Get logger with JSON-like structured format (useful for log aggregation).
    
    Example output:
        {"timestamp": "2024-10-11 10:30:00", "level": "INFO", "name": "api", "message": "Request received"}
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    if logger.hasHandlers():
        return logger
    
    # Structured format for machine parsing
    structured_format = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", '
        '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(structured_format)
    logger.addHandler(console_handler)
    
    if log_file:
        base_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        log_path = os.path.join(os.path.abspath(base_log_dir), log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(structured_format)
        logger.addHandler(file_handler)
    
    return logger
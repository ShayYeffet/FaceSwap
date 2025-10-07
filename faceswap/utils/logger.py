"""
Logging configuration and utilities for FaceSwap application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class FaceSwapLogger:
    """Centralized logging configuration for the FaceSwap application."""
    
    def __init__(self):
        self._logger = None
        self._log_file = None
    
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_file: Optional[str] = None,
                     console_output: bool = True) -> logging.Logger:
        """
        Set up logging configuration for the application.
        
        Args:
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file (str, optional): Path to log file. If None, creates default log file
            console_output (bool): Whether to output logs to console
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger('faceswap')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file is None:
            # Create default log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"faceswap_{timestamp}.log"
        
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            self._log_file = log_file
            logger.info(f"Logging initialized. Log file: {log_file}")
            
        except Exception as e:
            logger.warning(f"Could not create log file {log_file}: {str(e)}")
        
        self._logger = logger
        return logger
    
    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.
        
        Returns:
            logging.Logger: Logger instance
        """
        if self._logger is None:
            return self.setup_logging()
        return self._logger
    
    def get_log_file_path(self) -> Optional[str]:
        """
        Get the path to the current log file.
        
        Returns:
            str: Path to log file or None if not set
        """
        return self._log_file
    
    def log_system_info(self):
        """Log system information for debugging purposes."""
        if self._logger is None:
            return
        
        import platform
        import torch
        
        self._logger.info("=== System Information ===")
        self._logger.info(f"Platform: {platform.platform()}")
        self._logger.info(f"Python version: {platform.python_version()}")
        self._logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            self._logger.info(f"CUDA available: True")
            self._logger.info(f"CUDA version: {torch.version.cuda}")
            self._logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self._logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self._logger.info("CUDA available: False")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._logger.info("MPS (Apple Silicon) available: True")
        else:
            self._logger.info("MPS (Apple Silicon) available: False")
        
        self._logger.info("=== End System Information ===")
    
    def log_configuration(self, config: dict):
        """
        Log configuration parameters.
        
        Args:
            config (dict): Configuration dictionary to log
        """
        if self._logger is None:
            return
        
        self._logger.info("=== Configuration ===")
        for key, value in config.items():
            self._logger.info(f"{key}: {value}")
        self._logger.info("=== End Configuration ===")


# Global logger instance
faceswap_logger = FaceSwapLogger()


def get_logger(name: str = 'faceswap') -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
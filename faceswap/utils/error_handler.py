"""
Comprehensive error handling and recovery utilities for FaceSwap application.
"""

import logging
import traceback
import sys
import os
import psutil
import torch
from typing import Optional, Dict, Any, Callable, Union
from pathlib import Path
from functools import wraps
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorizing different types of errors."""
    LOW = "low"           # Non-critical errors that don't stop execution
    MEDIUM = "medium"     # Errors that affect quality but allow continuation
    HIGH = "high"         # Critical errors that require intervention
    FATAL = "fatal"       # Errors that must stop execution


class ErrorCategory(Enum):
    """Categories of errors for better handling and user feedback."""
    INPUT_VALIDATION = "input_validation"
    FILE_IO = "file_io"
    GPU_MEMORY = "gpu_memory"
    MODEL_LOADING = "model_loading"
    FACE_DETECTION = "face_detection"
    VIDEO_PROCESSING = "video_processing"
    TRAINING = "training"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK = "network"
    UNKNOWN = "unknown"


class FaceSwapError(Exception):
    """Base exception class for FaceSwap application errors."""
    
    def __init__(self, 
                 message: str, 
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 suggestions: Optional[list] = None,
                 technical_details: Optional[str] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.technical_details = technical_details
        self.timestamp = None


class InputValidationError(FaceSwapError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, suggestions: Optional[list] = None):
        super().__init__(
            message, 
            ErrorCategory.INPUT_VALIDATION, 
            ErrorSeverity.HIGH,
            suggestions
        )


class GPUMemoryError(FaceSwapError):
    """Raised when GPU memory issues occur."""
    
    def __init__(self, message: str, suggestions: Optional[list] = None):
        super().__init__(
            message,
            ErrorCategory.GPU_MEMORY,
            ErrorSeverity.HIGH,
            suggestions or [
                "Try reducing batch size",
                "Use CPU processing instead",
                "Close other GPU-intensive applications",
                "Use a smaller model size"
            ]
        )


class FileIOError(FaceSwapError):
    """Raised when file I/O operations fail."""
    
    def __init__(self, message: str, suggestions: Optional[list] = None):
        super().__init__(
            message,
            ErrorCategory.FILE_IO,
            ErrorSeverity.HIGH,
            suggestions
        )


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.fallback_strategies = {}
        self._setup_fallback_strategies()
    
    def _setup_fallback_strategies(self):
        """Setup fallback strategies for different error categories."""
        self.fallback_strategies = {
            ErrorCategory.GPU_MEMORY: self._gpu_memory_fallback,
            ErrorCategory.FILE_IO: self._file_io_fallback,
            ErrorCategory.FACE_DETECTION: self._face_detection_fallback,
            ErrorCategory.MODEL_LOADING: self._model_loading_fallback,
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    allow_recovery: bool = True) -> bool:
        """
        Handle an error with appropriate logging, user feedback, and recovery attempts.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            allow_recovery: Whether to attempt automatic recovery
            
        Returns:
            bool: True if error was handled and execution can continue, False otherwise
        """
        # Convert to FaceSwapError if needed
        if not isinstance(error, FaceSwapError):
            fs_error = self._convert_to_faceswap_error(error)
        else:
            fs_error = error
        
        # Log the error
        self._log_error(fs_error, context)
        
        # Track error frequency
        self._track_error(fs_error)
        
        # Display user-friendly error message
        self._display_error_message(fs_error)
        
        # Attempt recovery if allowed and appropriate
        if allow_recovery and fs_error.severity != ErrorSeverity.FATAL:
            return self._attempt_recovery(fs_error, context)
        
        return False
    
    def _convert_to_faceswap_error(self, error: Exception) -> FaceSwapError:
        """Convert generic exceptions to FaceSwapError with appropriate categorization."""
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Categorize based on error type and message
        if isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            category = ErrorCategory.FILE_IO
            severity = ErrorSeverity.HIGH
            suggestions = self._get_file_io_suggestions(error)
        
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            category = ErrorCategory.GPU_MEMORY
            severity = ErrorSeverity.HIGH
            suggestions = [
                "Reduce batch size (try --batch-size 8 or lower)",
                "Use CPU processing (--device cpu)",
                "Close other applications using GPU memory",
                "Try a smaller model (--model-size small)"
            ]
        
        elif "CUDA" in error_msg or "GPU" in error_msg:
            category = ErrorCategory.GPU_MEMORY
            severity = ErrorSeverity.MEDIUM
            suggestions = ["Check GPU drivers", "Try CPU fallback"]
        
        elif "face" in error_msg.lower() or "detection" in error_msg.lower():
            category = ErrorCategory.FACE_DETECTION
            severity = ErrorSeverity.MEDIUM
            suggestions = [
                "Check image quality and lighting",
                "Ensure faces are clearly visible",
                "Try different images in dataset"
            ]
        
        elif isinstance(error, (MemoryError, RuntimeError)):
            category = ErrorCategory.SYSTEM_RESOURCE
            severity = ErrorSeverity.HIGH
            suggestions = [
                "Close other applications to free memory",
                "Reduce processing parameters",
                "Use smaller input files"
            ]
        
        else:
            category = ErrorCategory.UNKNOWN
            severity = ErrorSeverity.MEDIUM
            suggestions = ["Check logs for more details"]
        
        return FaceSwapError(
            message=error_msg,
            category=category,
            severity=severity,
            suggestions=suggestions,
            technical_details=f"{error_type}: {traceback.format_exc()}"
        )
    
    def _get_file_io_suggestions(self, error: Exception) -> list:
        """Get specific suggestions for file I/O errors."""
        suggestions = []
        error_msg = str(error).lower()
        
        if "permission" in error_msg or "access" in error_msg:
            suggestions.extend([
                "Check file/folder permissions",
                "Run as administrator if necessary",
                "Ensure files are not open in other applications"
            ])
        
        elif "not found" in error_msg or "no such file" in error_msg:
            suggestions.extend([
                "Verify file/folder paths are correct",
                "Check if files exist at specified locations",
                "Use absolute paths if relative paths fail"
            ])
        
        elif "disk" in error_msg or "space" in error_msg:
            suggestions.extend([
                "Free up disk space",
                "Choose different output location",
                "Clean temporary files"
            ])
        
        else:
            suggestions.extend([
                "Check file/folder paths",
                "Verify file permissions",
                "Ensure sufficient disk space"
            ])
        
        return suggestions
    
    def _log_error(self, error: FaceSwapError, context: Optional[Dict[str, Any]]):
        """Log error with appropriate level and context."""
        log_msg = f"[{error.category.value.upper()}] {str(error)}"
        
        if context:
            log_msg += f" | Context: {context}"
        
        if error.severity == ErrorSeverity.FATAL:
            logger.critical(log_msg)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Log technical details at debug level
        if error.technical_details:
            logger.debug(f"Technical details: {error.technical_details}")
    
    def _track_error(self, error: FaceSwapError):
        """Track error frequency for pattern analysis."""
        error_key = f"{error.category.value}:{str(error)[:50]}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log if error is becoming frequent
        if self.error_counts[error_key] > 3:
            logger.warning(f"Frequent error detected: {error_key} (count: {self.error_counts[error_key]})")
    
    def _display_error_message(self, error: FaceSwapError):
        """Display user-friendly error message with suggestions."""
        severity_icons = {
            ErrorSeverity.LOW: "ℹ️",
            ErrorSeverity.MEDIUM: "⚠️",
            ErrorSeverity.HIGH: "❌",
            ErrorSeverity.FATAL: "💥"
        }
        
        icon = severity_icons.get(error.severity, "❌")
        print(f"\n{icon} {error.severity.value.upper()}: {str(error)}")
        
        if error.suggestions:
            print("\n💡 Suggestions:")
            for i, suggestion in enumerate(error.suggestions, 1):
                print(f"   {i}. {suggestion}")
        
        print()  # Empty line for readability
    
    def _attempt_recovery(self, 
                         error: FaceSwapError, 
                         context: Optional[Dict[str, Any]]) -> bool:
        """Attempt automatic recovery based on error category."""
        recovery_key = f"{error.category.value}:{str(error)[:30]}"
        
        # Check if we've already tried recovering from this error too many times
        attempts = self.recovery_attempts.get(recovery_key, 0)
        if attempts >= self.max_recovery_attempts:
            logger.warning(f"Max recovery attempts reached for: {recovery_key}")
            return False
        
        # Increment attempt counter
        self.recovery_attempts[recovery_key] = attempts + 1
        
        # Try category-specific recovery strategy
        recovery_func = self.fallback_strategies.get(error.category)
        if recovery_func:
            logger.info(f"Attempting recovery for {error.category.value} error (attempt {attempts + 1})")
            try:
                return recovery_func(error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {recovery_error}")
        
        return False
    
    def _gpu_memory_fallback(self, 
                           error: FaceSwapError, 
                           context: Optional[Dict[str, Any]]) -> bool:
        """Fallback strategy for GPU memory errors."""
        logger.info("Attempting GPU memory recovery...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        
        # Suggest batch size reduction
        if context and 'batch_size' in context:
            new_batch_size = max(1, context['batch_size'] // 2)
            logger.info(f"Suggesting batch size reduction: {context['batch_size']} -> {new_batch_size}")
            context['suggested_batch_size'] = new_batch_size
            return True
        
        return False
    
    def _file_io_fallback(self, 
                         error: FaceSwapError, 
                         context: Optional[Dict[str, Any]]) -> bool:
        """Fallback strategy for file I/O errors."""
        logger.info("Attempting file I/O recovery...")
        
        # Try creating directories if they don't exist
        if context and 'output_path' in context:
            try:
                output_path = Path(context['output_path'])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_path.parent}")
                return True
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
        
        return False
    
    def _face_detection_fallback(self, 
                               error: FaceSwapError, 
                               context: Optional[Dict[str, Any]]) -> bool:
        """Fallback strategy for face detection errors."""
        logger.info("Attempting face detection recovery...")
        
        # Suggest using different detection parameters
        if context:
            context['suggested_min_face_size'] = 20  # Lower threshold
            context['suggested_confidence_threshold'] = 0.8  # Lower confidence
            logger.info("Suggesting relaxed face detection parameters")
            return True
        
        return False
    
    def _model_loading_fallback(self, 
                              error: FaceSwapError, 
                              context: Optional[Dict[str, Any]]) -> bool:
        """Fallback strategy for model loading errors."""
        logger.info("Attempting model loading recovery...")
        
        # Suggest CPU fallback
        if context:
            context['suggested_device'] = 'cpu'
            logger.info("Suggesting CPU fallback for model loading")
            return True
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered during execution."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': len(self.error_counts),
            'most_common_errors': sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'recovery_attempts': dict(self.recovery_attempts)
        }


def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          allow_recovery: bool = True):
    """
    Decorator for automatic error handling in functions.
    
    Args:
        category: Error category for this function
        severity: Default severity level
        allow_recovery: Whether to allow automatic recovery
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get global error handler instance
                error_handler = getattr(wrapper, '_error_handler', ErrorHandler())
                
                # Create context from function arguments
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs': list(kwargs.keys())
                }
                
                # Handle the error
                if error_handler.handle_error(e, context, allow_recovery):
                    # If recovery was successful, try again
                    return func(*args, **kwargs)
                else:
                    # Re-raise if no recovery possible
                    raise
        
        # Attach error handler to wrapper for reuse
        wrapper._error_handler = ErrorHandler()
        return wrapper
    return decorator


def validate_system_resources() -> Dict[str, Any]:
    """
    Validate system resources and return status information.
    
    Returns:
        Dictionary with system resource status
    """
    status = {
        'memory': {},
        'disk': {},
        'gpu': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        # Memory information
        memory = psutil.virtual_memory()
        status['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent,
            'sufficient': memory.available > 2 * (1024**3)  # At least 2GB free
        }
        
        if not status['memory']['sufficient']:
            status['warnings'].append(
                f"Low system memory: {status['memory']['available_gb']:.1f}GB available. "
                "Consider closing other applications."
            )
    
    except Exception as e:
        status['errors'].append(f"Could not check memory status: {e}")
    
    try:
        # Disk space information
        disk = psutil.disk_usage('.')
        status['disk'] = {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent_used': (disk.used / disk.total) * 100,
            'sufficient': disk.free > 5 * (1024**3)  # At least 5GB free
        }
        
        if not status['disk']['sufficient']:
            status['warnings'].append(
                f"Low disk space: {status['disk']['free_gb']:.1f}GB available. "
                "Consider freeing up space."
            )
    
    except Exception as e:
        status['errors'].append(f"Could not check disk status: {e}")
    
    try:
        # GPU information
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            status['gpu'] = {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_gb': gpu_memory / (1024**3),
                'sufficient': gpu_memory > 4 * (1024**3)  # At least 4GB VRAM
            }
            
            if not status['gpu']['sufficient']:
                status['warnings'].append(
                    f"Limited GPU memory: {status['gpu']['memory_gb']:.1f}GB. "
                    "Consider using smaller batch sizes or CPU processing."
                )
        else:
            status['gpu'] = {
                'available': False,
                'name': 'None',
                'memory_gb': 0,
                'sufficient': False
            }
            status['warnings'].append(
                "No GPU detected. Processing will be significantly slower on CPU."
            )
    
    except Exception as e:
        status['errors'].append(f"Could not check GPU status: {e}")
    
    return status


# Global error handler instance
global_error_handler = ErrorHandler()
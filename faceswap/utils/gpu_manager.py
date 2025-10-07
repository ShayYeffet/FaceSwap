"""
GPU detection and device management utilities for FaceSwap application.
"""

import torch
import logging
import psutil
import gc
from typing import Optional, Tuple, Dict, Any
import subprocess
import sys

from .error_handler import GPUMemoryError, ErrorSeverity

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU detection and device selection for optimal performance."""
    
    def __init__(self):
        self._device = None
        self._device_name = None
        self._memory_info = None
    
    def detect_gpu(self) -> str:
        """
        Detect available GPU resources and return the best device.
        
        Returns:
            str: Device string ('cuda', 'mps', or 'cpu')
        """
        if self._device is not None:
            return self._device
            
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            self._device = "cuda"
            self._device_name = torch.cuda.get_device_name(0)
            self._memory_info = self._get_cuda_memory_info()
            logger.info(f"CUDA GPU detected: {self._device_name}")
            logger.info(f"GPU Memory: {self._memory_info['total']:.1f}GB total, "
                       f"{self._memory_info['available']:.1f}GB available")
        
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = "mps"
            self._device_name = "Apple Silicon GPU"
            logger.info("Apple Silicon GPU (MPS) detected")
        
        # Fallback to CPU
        else:
            self._device = "cpu"
            self._device_name = "CPU"
            logger.warning("No GPU detected. Falling back to CPU processing. "
                          "Training and inference will be significantly slower.")
        
        return self._device
    
    def get_device(self) -> torch.device:
        """
        Get the torch device object for the detected hardware.
        
        Returns:
            torch.device: PyTorch device object
        """
        if self._device is None:
            self.detect_gpu()
        return torch.device(self._device)
    
    def get_device_name(self) -> str:
        """
        Get the name of the detected device.
        
        Returns:
            str: Device name
        """
        if self._device is None:
            self.detect_gpu()
        return self._device_name
    
    def get_optimal_batch_size(self, base_batch_size: int = 16, safety_factor: float = 0.8) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            base_batch_size (int): Base batch size to scale from
            safety_factor (float): Safety factor to prevent OOM (0.0-1.0)
            
        Returns:
            int: Recommended batch size
        """
        if self._device is None:
            self.detect_gpu()
        
        if self._device == "cpu":
            # Conservative batch size for CPU based on system RAM
            try:
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)
                
                if available_gb >= 16:
                    return min(base_batch_size, 8)
                elif available_gb >= 8:
                    return min(base_batch_size, 4)
                else:
                    return min(base_batch_size, 2)
            except:
                return min(base_batch_size, 4)
        
        elif self._device == "cuda" and self._memory_info:
            # Scale batch size based on available GPU memory with safety factor
            available_gb = self._memory_info['available'] * safety_factor
            
            # Estimate memory usage per batch item (rough approximation)
            # For 256x256 images: ~2MB per image for model processing
            estimated_memory_per_item = 0.002  # GB
            max_batch_from_memory = int(available_gb / estimated_memory_per_item)
            
            if available_gb >= 8:
                optimal = min(base_batch_size, max_batch_from_memory)
            elif available_gb >= 6:
                optimal = min(max(base_batch_size // 2, 8), max_batch_from_memory)
            elif available_gb >= 4:
                optimal = min(max(base_batch_size // 4, 4), max_batch_from_memory)
            elif available_gb >= 2:
                optimal = min(2, max_batch_from_memory)
            else:
                optimal = 1
            
            return max(1, optimal)
        
        else:
            # Conservative default for MPS or unknown devices
            return max(base_batch_size // 2, 4)
    
    def _get_cuda_memory_info(self) -> dict:
        """Get CUDA memory information."""
        if not torch.cuda.is_available():
            return None
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        available_memory = total_memory - allocated_memory
        
        return {
            'total': total_memory / (1024**3),  # Convert to GB
            'allocated': allocated_memory / (1024**3),
            'available': available_memory / (1024**3)
        }
    
    def clear_cache(self):
        """Clear GPU memory cache if using CUDA."""
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def get_memory_usage(self) -> Optional[dict]:
        """
        Get current GPU memory usage.
        
        Returns:
            dict: Memory usage information or None if not available
        """
        if self._device == "cuda" and torch.cuda.is_available():
            return self._get_cuda_memory_info()
        return None
    
    def handle_gpu_memory_error(self, error: Exception, current_batch_size: int) -> Dict[str, Any]:
        """
        Handle GPU memory errors with automatic recovery strategies.
        
        Args:
            error: The GPU memory error that occurred
            current_batch_size: Current batch size being used
            
        Returns:
            Dict with recovery suggestions and new parameters
        """
        recovery_info = {
            'error_handled': False,
            'suggested_batch_size': current_batch_size,
            'suggested_device': self._device,
            'recovery_actions': [],
            'warnings': []
        }
        
        try:
            # Clear GPU cache first
            if self._device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                recovery_info['recovery_actions'].append("Cleared GPU cache")
            
            # Get updated memory info
            if self._device == "cuda":
                self._memory_info = self._get_cuda_memory_info()
            
            # Suggest batch size reduction
            if current_batch_size > 1:
                new_batch_size = max(1, current_batch_size // 2)
                recovery_info['suggested_batch_size'] = new_batch_size
                recovery_info['recovery_actions'].append(f"Reduced batch size: {current_batch_size} -> {new_batch_size}")
                recovery_info['error_handled'] = True
            
            # If batch size is already 1, suggest CPU fallback
            elif current_batch_size == 1:
                recovery_info['suggested_device'] = 'cpu'
                recovery_info['suggested_batch_size'] = 4  # CPU can handle slightly larger batches
                recovery_info['recovery_actions'].append("Suggested CPU fallback")
                recovery_info['warnings'].append("GPU memory insufficient even for batch size 1")
                recovery_info['error_handled'] = True
            
            # Additional suggestions
            if self._memory_info and self._memory_info['available'] < 2:
                recovery_info['warnings'].append(f"Very low GPU memory: {self._memory_info['available']:.1f}GB available")
                recovery_info['recovery_actions'].append("Consider closing other GPU applications")
            
        except Exception as recovery_error:
            logger.error(f"Error during GPU memory recovery: {recovery_error}")
            recovery_info['warnings'].append("Automatic recovery failed")
        
        return recovery_info
    
    def validate_gpu_setup(self) -> Dict[str, Any]:
        """
        Validate GPU setup and return comprehensive status.
        
        Returns:
            Dict with GPU validation results
        """
        validation = {
            'gpu_available': False,
            'device_name': 'None',
            'memory_info': None,
            'cuda_version': None,
            'driver_version': None,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                validation['gpu_available'] = True
                validation['device_name'] = torch.cuda.get_device_name(0)
                validation['memory_info'] = self._get_cuda_memory_info()
                validation['cuda_version'] = torch.version.cuda
                
                # Try to get driver version
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        validation['driver_version'] = result.stdout.strip()
                except:
                    validation['warnings'].append("Could not determine NVIDIA driver version")
                
                # Check memory sufficiency
                if validation['memory_info']:
                    available_gb = validation['memory_info']['available']
                    if available_gb < 2:
                        validation['warnings'].append(f"Low GPU memory: {available_gb:.1f}GB available")
                        validation['recommendations'].append("Consider using CPU processing or smaller batch sizes")
                    elif available_gb < 4:
                        validation['warnings'].append(f"Limited GPU memory: {available_gb:.1f}GB available")
                        validation['recommendations'].append("Use smaller batch sizes for optimal performance")
                
                # Test basic GPU operations
                try:
                    test_tensor = torch.randn(10, 10).cuda()
                    test_result = test_tensor @ test_tensor.T
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                except Exception as e:
                    validation['errors'].append(f"GPU operation test failed: {e}")
                    validation['recommendations'].append("GPU may not be functioning properly")
            
            # Check MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                validation['gpu_available'] = True
                validation['device_name'] = "Apple Silicon GPU (MPS)"
                validation['recommendations'].append("Using Apple Silicon GPU acceleration")
                
                # Test MPS operations
                try:
                    test_tensor = torch.randn(10, 10).to('mps')
                    test_result = test_tensor @ test_tensor.T
                    del test_tensor, test_result
                except Exception as e:
                    validation['errors'].append(f"MPS operation test failed: {e}")
            
            else:
                validation['warnings'].append("No GPU acceleration available")
                validation['recommendations'].extend([
                    "Processing will use CPU (significantly slower)",
                    "Consider using a system with NVIDIA GPU for better performance"
                ])
        
        except Exception as e:
            validation['errors'].append(f"GPU validation failed: {e}")
        
        return validation
    
    def monitor_memory_usage(self, threshold_percent: float = 90.0) -> Dict[str, Any]:
        """
        Monitor GPU memory usage and warn if approaching limits.
        
        Args:
            threshold_percent: Warning threshold as percentage of total memory
            
        Returns:
            Dict with memory monitoring results
        """
        monitoring = {
            'memory_info': None,
            'usage_percent': 0,
            'warning_triggered': False,
            'recommendations': []
        }
        
        try:
            if self._device == "cuda" and torch.cuda.is_available():
                memory_info = self._get_cuda_memory_info()
                monitoring['memory_info'] = memory_info
                
                if memory_info:
                    usage_percent = (memory_info['allocated'] / memory_info['total']) * 100
                    monitoring['usage_percent'] = usage_percent
                    
                    if usage_percent > threshold_percent:
                        monitoring['warning_triggered'] = True
                        monitoring['recommendations'].extend([
                            f"High GPU memory usage: {usage_percent:.1f}%",
                            "Consider reducing batch size",
                            "Clear GPU cache if possible"
                        ])
                        
                        # Automatically clear cache if usage is very high
                        if usage_percent > 95:
                            torch.cuda.empty_cache()
                            monitoring['recommendations'].append("Automatically cleared GPU cache")
        
        except Exception as e:
            logger.warning(f"GPU memory monitoring failed: {e}")
        
        return monitoring


# Global GPU manager instance
gpu_manager = GPUManager()
"""
File validation and path handling utilities for FaceSwap application.
"""

import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import mimetypes
import cv2
import subprocess

from .error_handler import FileIOError, ErrorSeverity

logger = logging.getLogger(__name__)


class FileManager:
    """Handles file validation, path operations, and file system utilities."""
    
    # Supported video formats
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    @staticmethod
    def validate_video_file(video_path: str, detailed: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Validate if the provided path is a valid video file.
        
        Args:
            video_path (str): Path to the video file
            detailed (bool): Return detailed validation results
            
        Returns:
            bool or Dict: True/False if detailed=False, detailed results dict if detailed=True
        """
        result = {
            'valid': False,
            'path': video_path,
            'exists': False,
            'is_file': False,
            'format_supported': False,
            'readable': False,
            'size_mb': 0,
            'duration_seconds': 0,
            'fps': 0,
            'resolution': (0, 0),
            'errors': [],
            'warnings': []
        }
        
        try:
            path = Path(video_path)
            
            # Check if file exists
            if not path.exists():
                error_msg = f"Video file does not exist: {video_path}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False
                return result
            
            result['exists'] = True
            
            # Check if it's a file (not directory)
            if not path.is_file():
                error_msg = f"Path is not a file: {video_path}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False
                return result
            
            result['is_file'] = True
            
            # Check file extension
            if path.suffix.lower() not in FileManager.SUPPORTED_VIDEO_FORMATS:
                error_msg = f"Unsupported video format: {path.suffix}. Supported formats: {', '.join(FileManager.SUPPORTED_VIDEO_FORMATS)}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False
                return result
            
            result['format_supported'] = True
            
            # Check file size
            file_size = path.stat().st_size
            result['size_mb'] = file_size / (1024 * 1024)
            
            if file_size == 0:
                error_msg = f"Video file is empty: {video_path}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False
                return result
            
            # Try to read video properties with OpenCV
            try:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    error_msg = f"Cannot open video file with OpenCV: {video_path}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
                    if not detailed:
                        return False
                    return result
                
                result['readable'] = True
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                result['fps'] = fps
                result['resolution'] = (width, height)
                
                if fps > 0 and frame_count > 0:
                    result['duration_seconds'] = frame_count / fps
                
                cap.release()
                
                # Validate video properties
                if fps <= 0:
                    result['warnings'].append("Invalid or unknown frame rate")
                elif fps < 10:
                    result['warnings'].append(f"Low frame rate: {fps:.1f} FPS")
                elif fps > 120:
                    result['warnings'].append(f"Very high frame rate: {fps:.1f} FPS")
                
                if width <= 0 or height <= 0:
                    result['warnings'].append("Invalid or unknown resolution")
                elif width < 480 or height < 360:
                    result['warnings'].append(f"Low resolution: {width}x{height}")
                
                if result['duration_seconds'] <= 0:
                    result['warnings'].append("Could not determine video duration")
                elif result['duration_seconds'] < 1:
                    result['warnings'].append(f"Very short video: {result['duration_seconds']:.1f} seconds")
                
            except Exception as video_error:
                error_msg = f"Error reading video properties: {video_error}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False
                return result
            
            result['valid'] = True
            logger.info(f"Video file validation successful: {video_path}")
            
            if not detailed:
                return True
            return result
            
        except Exception as e:
            error_msg = f"Error validating video file {video_path}: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            if not detailed:
                return False
            return result
    
    @staticmethod
    def validate_dataset_folder(dataset_path: str, detailed: bool = False) -> Union[Tuple[bool, List[str]], Dict[str, Any]]:
        """
        Validate dataset folder and return list of valid image files.
        
        Args:
            dataset_path (str): Path to the dataset folder
            detailed (bool): Return detailed validation results
            
        Returns:
            Tuple or Dict: (is_valid, list_of_image_paths) if detailed=False, detailed results dict if detailed=True
        """
        result = {
            'valid': False,
            'path': dataset_path,
            'exists': False,
            'is_directory': False,
            'readable': False,
            'total_files': 0,
            'valid_images': [],
            'invalid_images': [],
            'empty_images': [],
            'corrupted_images': [],
            'unsupported_formats': [],
            'image_stats': {
                'total_size_mb': 0,
                'avg_resolution': (0, 0),
                'min_resolution': (float('inf'), float('inf')),
                'max_resolution': (0, 0)
            },
            'errors': [],
            'warnings': []
        }
        
        try:
            path = Path(dataset_path)
            
            # Check if folder exists
            if not path.exists():
                error_msg = f"Dataset folder does not exist: {dataset_path}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False, []
                return result
            
            result['exists'] = True
            
            # Check if it's a directory
            if not path.is_dir():
                error_msg = f"Path is not a directory: {dataset_path}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False, []
                return result
            
            result['is_directory'] = True
            
            # Check if directory is readable
            try:
                list(path.iterdir())
                result['readable'] = True
            except PermissionError:
                error_msg = f"No permission to read directory: {dataset_path}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False, []
                return result
            
            # Find and validate all files
            all_files = list(path.iterdir())
            result['total_files'] = len([f for f in all_files if f.is_file()])
            
            total_size = 0
            resolutions = []
            
            for file_path in all_files:
                if not file_path.is_file():
                    continue
                
                file_ext = file_path.suffix.lower()
                
                # Check if it's a supported image format
                if file_ext not in FileManager.SUPPORTED_IMAGE_FORMATS:
                    if file_ext:  # Only log if there's an extension
                        result['unsupported_formats'].append(str(file_path))
                    continue
                
                # Check if file is empty
                file_size = file_path.stat().st_size
                if file_size == 0:
                    result['empty_images'].append(str(file_path))
                    logger.warning(f"Skipping empty image file: {file_path}")
                    continue
                
                # Try to validate image by loading it
                try:
                    image = cv2.imread(str(file_path))
                    if image is None:
                        result['corrupted_images'].append(str(file_path))
                        logger.warning(f"Cannot read image file: {file_path}")
                        continue
                    
                    # Check image properties
                    height, width = image.shape[:2]
                    if height < 64 or width < 64:
                        result['warnings'].append(f"Small image ({width}x{height}): {file_path.name}")
                    
                    # Update statistics
                    total_size += file_size
                    resolutions.append((width, height))
                    
                    # Update min/max resolution
                    current_min = result['image_stats']['min_resolution']
                    current_max = result['image_stats']['max_resolution']
                    result['image_stats']['min_resolution'] = (
                        min(current_min[0], width),
                        min(current_min[1], height)
                    )
                    result['image_stats']['max_resolution'] = (
                        max(current_max[0], width),
                        max(current_max[1], height)
                    )
                    
                    result['valid_images'].append(str(file_path))
                    
                except Exception as img_error:
                    result['corrupted_images'].append(str(file_path))
                    logger.warning(f"Error validating image {file_path}: {img_error}")
            
            # Calculate statistics
            if resolutions:
                result['image_stats']['total_size_mb'] = total_size / (1024 * 1024)
                avg_width = sum(r[0] for r in resolutions) / len(resolutions)
                avg_height = sum(r[1] for r in resolutions) / len(resolutions)
                result['image_stats']['avg_resolution'] = (int(avg_width), int(avg_height))
            
            # Check minimum number of images
            min_images = 10
            valid_count = len(result['valid_images'])
            
            if valid_count < min_images:
                error_msg = f"Dataset contains only {valid_count} valid images. Minimum {min_images} images required for training."
                result['errors'].append(error_msg)
                logger.error(error_msg)
                if not detailed:
                    return False, result['valid_images']
                return result
            
            # Add warnings for dataset quality
            if len(result['corrupted_images']) > 0:
                result['warnings'].append(f"{len(result['corrupted_images'])} corrupted images found")
            
            if len(result['empty_images']) > 0:
                result['warnings'].append(f"{len(result['empty_images'])} empty images found")
            
            if len(result['unsupported_formats']) > 0:
                result['warnings'].append(f"{len(result['unsupported_formats'])} unsupported file formats found")
            
            if valid_count < 50:
                result['warnings'].append(f"Small dataset ({valid_count} images). Consider adding more images for better results.")
            
            result['valid'] = True
            logger.info(f"Dataset validation successful: {valid_count} valid images found in {dataset_path}")
            
            if not detailed:
                return True, result['valid_images']
            return result
            
        except Exception as e:
            error_msg = f"Error validating dataset folder {dataset_path}: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            if not detailed:
                return False, []
            return result
    
    @staticmethod
    def create_output_filename(input_path: str, suffix: str = "_faceswap") -> str:
        """
        Create output filename based on input path.
        
        Args:
            input_path (str): Original input file path
            suffix (str): Suffix to add to filename
            
        Returns:
            str: Output file path
        """
        try:
            path = Path(input_path)
            output_name = f"{path.stem}{suffix}{path.suffix}"
            output_path = path.parent / output_name
            
            # Handle file conflicts by adding numbers
            counter = 1
            while output_path.exists():
                output_name = f"{path.stem}{suffix}_{counter}{path.suffix}"
                output_path = path.parent / output_name
                counter += 1
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating output filename for {input_path}: {str(e)}")
            return f"{input_path}_output"
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """
        Ensure that a directory exists, create if it doesn't.
        
        Args:
            directory_path (str): Path to directory
            
        Returns:
            bool: True if directory exists or was created successfully
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {str(e)}")
            return False
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Get file size in megabytes.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            float: File size in MB
        """
        try:
            size_bytes = Path(file_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting file size for {file_path}: {str(e)}")
            return 0.0
    
    @staticmethod
    def clean_temp_files(temp_dir: str) -> bool:
        """
        Clean up temporary files in the specified directory.
        
        Args:
            temp_dir (str): Path to temporary directory
            
        Returns:
            bool: True if cleanup successful
        """
        try:
            temp_path = Path(temp_dir)
            if temp_path.exists() and temp_path.is_dir():
                for file_path in temp_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Deleted temporary file: {file_path}")
                temp_path.rmdir()
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning temporary files in {temp_dir}: {str(e)}")
            return False
    
    @staticmethod
    def validate_write_permissions(directory_path: str) -> bool:
        """
        Check if we have write permissions in the specified directory.
        
        Args:
            directory_path (str): Path to directory
            
        Returns:
            bool: True if write permissions exist
        """
        try:
            test_file = Path(directory_path) / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"No write permissions in directory {directory_path}: {str(e)}")
            return False
    
    @staticmethod
    def validate_input_paths(video_path: str, dataset_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of all input paths with detailed error reporting.
        
        Args:
            video_path (str): Path to input video
            dataset_path (str): Path to dataset folder
            output_path (str, optional): Path for output file
            
        Returns:
            Dict with validation results and suggestions
        """
        validation_result = {
            'valid': False,
            'video': {},
            'dataset': {},
            'output': {},
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Validate video file
        try:
            video_result = FileManager.validate_video_file(video_path, detailed=True)
            validation_result['video'] = video_result
            
            if not video_result['valid']:
                validation_result['errors'].extend([f"Video: {err}" for err in video_result['errors']])
                validation_result['suggestions'].extend([
                    "Check if the video file path is correct",
                    "Ensure the video file is not corrupted",
                    f"Use supported formats: {', '.join(FileManager.SUPPORTED_VIDEO_FORMATS)}"
                ])
            else:
                validation_result['warnings'].extend([f"Video: {warn}" for warn in video_result['warnings']])
                
        except Exception as e:
            validation_result['errors'].append(f"Video validation failed: {e}")
        
        # Validate dataset folder
        try:
            dataset_result = FileManager.validate_dataset_folder(dataset_path, detailed=True)
            validation_result['dataset'] = dataset_result
            
            if not dataset_result['valid']:
                validation_result['errors'].extend([f"Dataset: {err}" for err in dataset_result['errors']])
                validation_result['suggestions'].extend([
                    "Check if the dataset folder path is correct",
                    "Ensure the folder contains at least 10 valid images",
                    f"Use supported image formats: {', '.join(FileManager.SUPPORTED_IMAGE_FORMATS)}",
                    "Remove corrupted or empty image files"
                ])
            else:
                validation_result['warnings'].extend([f"Dataset: {warn}" for warn in dataset_result['warnings']])
                
        except Exception as e:
            validation_result['errors'].append(f"Dataset validation failed: {e}")
        
        # Validate output path if provided
        if output_path:
            try:
                output_dir = Path(output_path).parent
                
                # Check if output directory exists or can be created
                if not output_dir.exists():
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        validation_result['output']['directory_created'] = True
                    except Exception as e:
                        validation_result['errors'].append(f"Cannot create output directory: {e}")
                        validation_result['suggestions'].append("Choose a different output location")
                
                # Check write permissions
                if output_dir.exists():
                    if FileManager.validate_write_permissions(str(output_dir)):
                        validation_result['output']['writable'] = True
                    else:
                        validation_result['errors'].append("No write permissions in output directory")
                        validation_result['suggestions'].extend([
                            "Choose a different output location",
                            "Run with administrator privileges",
                            "Check folder permissions"
                        ])
                
                # Check available disk space
                try:
                    disk_usage = shutil.disk_usage(output_dir)
                    free_gb = disk_usage.free / (1024**3)
                    validation_result['output']['free_space_gb'] = free_gb
                    
                    if free_gb < 1:
                        validation_result['errors'].append(f"Insufficient disk space: {free_gb:.1f}GB available")
                        validation_result['suggestions'].append("Free up disk space or choose different output location")
                    elif free_gb < 5:
                        validation_result['warnings'].append(f"Low disk space: {free_gb:.1f}GB available")
                        
                except Exception as e:
                    validation_result['warnings'].append(f"Could not check disk space: {e}")
                    
            except Exception as e:
                validation_result['errors'].append(f"Output path validation failed: {e}")
        
        # Overall validation status
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    @staticmethod
    def create_safe_output_path(input_path: str, suffix: str = "_faceswap", max_attempts: int = 100) -> str:
        """
        Create a safe output path that doesn't conflict with existing files.
        
        Args:
            input_path (str): Original input file path
            suffix (str): Suffix to add to filename
            max_attempts (int): Maximum attempts to find unique filename
            
        Returns:
            str: Safe output file path
            
        Raises:
            FileIOError: If unable to create safe output path
        """
        try:
            path = Path(input_path)
            base_name = path.stem
            extension = path.suffix
            directory = path.parent
            
            # Try the basic suffix first
            output_name = f"{base_name}{suffix}{extension}"
            output_path = directory / output_name
            
            if not output_path.exists():
                return str(output_path)
            
            # If file exists, try with numbers
            for i in range(1, max_attempts + 1):
                output_name = f"{base_name}{suffix}_{i}{extension}"
                output_path = directory / output_name
                
                if not output_path.exists():
                    return str(output_path)
            
            # If all attempts failed, raise error
            raise FileIOError(
                f"Unable to create unique output filename after {max_attempts} attempts",
                suggestions=[
                    "Manually specify output filename",
                    "Clean up existing output files",
                    "Choose different output directory"
                ]
            )
            
        except Exception as e:
            if isinstance(e, FileIOError):
                raise
            raise FileIOError(f"Error creating output filename: {e}")
    
    @staticmethod
    def check_file_locks(file_paths: List[str]) -> Dict[str, bool]:
        """
        Check if files are locked by other processes.
        
        Args:
            file_paths (List[str]): List of file paths to check
            
        Returns:
            Dict mapping file paths to lock status (True if locked)
        """
        lock_status = {}
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    lock_status[file_path] = False
                    continue
                
                # Try to open file in write mode to check if it's locked
                with open(path, 'r+b') as f:
                    lock_status[file_path] = False
                    
            except (PermissionError, OSError):
                lock_status[file_path] = True
                logger.warning(f"File appears to be locked: {file_path}")
            except Exception as e:
                lock_status[file_path] = True
                logger.warning(f"Cannot access file {file_path}: {e}")
        
        return lock_status
    
    @staticmethod
    def create_backup(file_path: str, backup_suffix: str = ".backup") -> Optional[str]:
        """
        Create a backup copy of a file.
        
        Args:
            file_path (str): Path to file to backup
            backup_suffix (str): Suffix for backup file
            
        Returns:
            str: Path to backup file, or None if backup failed
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.warning(f"Cannot backup non-existent file: {file_path}")
                return None
            
            backup_path = source_path.with_suffix(source_path.suffix + backup_suffix)
            
            # If backup already exists, add number
            counter = 1
            while backup_path.exists():
                backup_path = source_path.with_suffix(f"{source_path.suffix}{backup_suffix}.{counter}")
                counter += 1
            
            shutil.copy2(source_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup of {file_path}: {e}")
            return None
    
    @staticmethod
    def cleanup_temp_files(patterns: List[str], max_age_hours: int = 24) -> int:
        """
        Clean up temporary files matching given patterns.
        
        Args:
            patterns (List[str]): List of file patterns to clean up
            max_age_hours (int): Maximum age of files to keep (in hours)
            
        Returns:
            int: Number of files cleaned up
        """
        import time
        import glob
        
        cleaned_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for pattern in patterns:
            try:
                for file_path in glob.glob(pattern):
                    try:
                        path = Path(file_path)
                        if not path.is_file():
                            continue
                        
                        # Check file age
                        file_age = current_time - path.stat().st_mtime
                        if file_age > max_age_seconds:
                            path.unlink()
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old temp file: {file_path}")
                            
                    except Exception as e:
                        logger.warning(f"Could not clean up file {file_path}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error processing pattern {pattern}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
        
        return cleaned_count
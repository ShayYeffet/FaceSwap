"""
Data preprocessing pipeline for face dataset loading, validation, and augmentation.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A

from .face_detector import FaceDetector, FaceData
from ..utils.logger import get_logger
from ..utils.file_manager import FileManager

logger = get_logger(__name__)


class DataProcessor:
    """
    Data preprocessing pipeline for face datasets.
    
    Handles dataset loading, validation, face extraction, normalization,
    and data augmentation for training robustness.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 face_detector: Optional[FaceDetector] = None,
                 augmentation_enabled: bool = True):
        """
        Initialize the data processor.
        
        Args:
            target_size: Target size for processed face images
            face_detector: FaceDetector instance (creates new one if None)
            augmentation_enabled: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.augmentation_enabled = augmentation_enabled
        self.file_manager = FileManager()
        
        # Initialize face detector
        if face_detector is None:
            self.face_detector = FaceDetector()
        else:
            self.face_detector = face_detector
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Initialize augmentation pipeline
        self._setup_augmentation()
        
        logger.info(f"DataProcessor initialized with target size: {target_size}")
    
    def _setup_augmentation(self):
        """Setup data augmentation pipeline using Albumentations."""
        if not self.augmentation_enabled:
            self.augmentation_pipeline = None
            return
        
        self.augmentation_pipeline = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3
            ),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.3
            ),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),
            
            # Quality degradation
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
        ])
        
        logger.info("Data augmentation pipeline initialized")
    
    def validate_dataset_folder(self, dataset_path: str) -> Dict[str, Union[bool, int, List[str]]]:
        """
        Validate dataset folder structure and contents.
        
        Args:
            dataset_path: Path to dataset folder
            
        Returns:
            Dictionary with validation results:
            - 'valid': bool indicating if dataset is valid
            - 'total_images': int number of valid image files
            - 'valid_images': list of valid image file paths
            - 'invalid_images': list of invalid/problematic image files
            - 'errors': list of error messages
        """
        result = {
            'valid': False,
            'total_images': 0,
            'valid_images': [],
            'invalid_images': [],
            'errors': []
        }
        
        try:
            dataset_path = Path(dataset_path)
            
            # Check if directory exists
            if not dataset_path.exists():
                result['errors'].append(f"Dataset directory does not exist: {dataset_path}")
                return result
            
            if not dataset_path.is_dir():
                result['errors'].append(f"Path is not a directory: {dataset_path}")
                return result
            
            # Find all image files
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(dataset_path.glob(f"*{ext}"))
                image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                result['errors'].append("No supported image files found in dataset directory")
                return result
            
            logger.info(f"Found {len(image_files)} potential image files")
            
            # Validate each image file
            for img_path in image_files:
                try:
                    # Try to load and validate image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        result['invalid_images'].append(str(img_path))
                        logger.warning(f"Could not load image: {img_path}")
                        continue
                    
                    # Check minimum size requirements
                    h, w = image.shape[:2]
                    if h < 64 or w < 64:
                        result['invalid_images'].append(str(img_path))
                        logger.warning(f"Image too small ({w}x{h}): {img_path}")
                        continue
                    
                    result['valid_images'].append(str(img_path))
                    
                except Exception as e:
                    result['invalid_images'].append(str(img_path))
                    logger.warning(f"Error validating image {img_path}: {e}")
            
            result['total_images'] = len(result['valid_images'])
            
            # Check minimum dataset size
            if result['total_images'] < 5:
                result['errors'].append(f"Dataset too small: {result['total_images']} images (minimum 5 recommended)")
                return result
            
            result['valid'] = True
            logger.info(f"Dataset validation successful: {result['total_images']} valid images")
            
        except Exception as e:
            result['errors'].append(f"Dataset validation failed: {e}")
            logger.error(f"Dataset validation error: {e}")
        
        return result
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and validate a single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array or None if loading fails
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            # Ensure image is in correct format
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.warning(f"Invalid image format: {image_path}")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def extract_faces_from_dataset(self, dataset_path: str) -> List[FaceData]:
        """
        Extract faces from all images in a dataset folder.
        
        Args:
            dataset_path: Path to dataset folder
            
        Returns:
            List of FaceData objects for all detected faces
        """
        # Validate dataset first
        validation_result = self.validate_dataset_folder(dataset_path)
        
        if not validation_result['valid']:
            logger.error(f"Dataset validation failed: {validation_result['errors']}")
            return []
        
        valid_images = validation_result['valid_images']
        faces = []
        
        logger.info(f"Extracting faces from {len(valid_images)} images")
        
        for i, img_path in enumerate(valid_images):
            try:
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Extract face
                face_data = self.face_detector.process_image(image)
                
                if face_data is not None:
                    faces.append(face_data)
                    logger.debug(f"Face extracted from {img_path} (confidence: {face_data.confidence:.3f})")
                else:
                    logger.warning(f"No face detected in {img_path}")
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue
        
        logger.info(f"Face extraction complete: {len(faces)} faces from {len(valid_images)} images")
        return faces
    
    def normalize_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Normalize face image for training.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Normalized face image (values in range [0, 1])
        """
        try:
            # Ensure image is in correct format
            if face_image.dtype != np.uint8:
                face_image = face_image.astype(np.uint8)
            
            # Resize to target size if needed
            if face_image.shape[:2] != self.target_size:
                face_image = cv2.resize(face_image, self.target_size, interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [0, 1] range
            normalized = face_image.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Face normalization failed: {e}")
            return face_image
    
    def apply_augmentation(self, face_image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to a face image.
        
        Args:
            face_image: Input face image as numpy array
            
        Returns:
            Augmented face image
        """
        if not self.augmentation_enabled or self.augmentation_pipeline is None:
            return face_image
        
        try:
            # Ensure image is uint8 for augmentation
            if face_image.dtype == np.float32:
                aug_image = (face_image * 255).astype(np.uint8)
            else:
                aug_image = face_image.astype(np.uint8)
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(image=aug_image)['image']
            
            # Convert back to original dtype
            if face_image.dtype == np.float32:
                augmented = augmented.astype(np.float32) / 255.0
            
            return augmented
            
        except Exception as e:
            logger.warning(f"Augmentation failed, returning original image: {e}")
            return face_image
    
    def create_training_pairs(self, 
                            source_faces: List[FaceData], 
                            target_faces: List[FaceData],
                            augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training pairs from source and target face datasets.
        
        Args:
            source_faces: List of source face data
            target_faces: List of target face data
            augment_factor: Number of augmented versions per original image
            
        Returns:
            Tuple of (source_array, target_array) for training
        """
        if not source_faces or not target_faces:
            logger.error("Cannot create training pairs: empty face datasets")
            return np.array([]), np.array([])
        
        logger.info(f"Creating training pairs from {len(source_faces)} source and {len(target_faces)} target faces")
        
        # Prepare source faces
        source_images = []
        for face_data in source_faces:
            # Normalize original image
            normalized = self.normalize_face(face_data.image)
            source_images.append(normalized)
            
            # Create augmented versions
            if self.augmentation_enabled:
                for _ in range(augment_factor):
                    augmented = self.apply_augmentation(normalized)
                    source_images.append(augmented)
        
        # Prepare target faces
        target_images = []
        for face_data in target_faces:
            # Normalize original image
            normalized = self.normalize_face(face_data.image)
            target_images.append(normalized)
            
            # Create augmented versions
            if self.augmentation_enabled:
                for _ in range(augment_factor):
                    augmented = self.apply_augmentation(normalized)
                    target_images.append(augmented)
        
        # Balance datasets by repeating smaller dataset
        source_count = len(source_images)
        target_count = len(target_images)
        
        if source_count > target_count:
            # Repeat target images to match source count
            repeat_factor = (source_count + target_count - 1) // target_count
            target_images = target_images * repeat_factor
            target_images = target_images[:source_count]
        elif target_count > source_count:
            # Repeat source images to match target count
            repeat_factor = (target_count + source_count - 1) // source_count
            source_images = source_images * repeat_factor
            source_images = source_images[:target_count]
        
        # Convert to numpy arrays
        source_array = np.array(source_images, dtype=np.float32)
        target_array = np.array(target_images, dtype=np.float32)
        
        logger.info(f"Training pairs created: {source_array.shape[0]} pairs")
        logger.info(f"Source shape: {source_array.shape}, Target shape: {target_array.shape}")
        
        return source_array, target_array
    
    def process_dataset(self, dataset_path: str, progress_callback=None) -> Dict:
        """
        Complete dataset processing pipeline.
        
        Args:
            dataset_path: Path to dataset folder
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing dataset: {dataset_path}")
        
        if progress_callback:
            progress_callback(0, 100, "Starting dataset processing...")
        
        # Extract faces from dataset
        if progress_callback:
            progress_callback(20, 100, "Extracting faces from images...")
        
        faces = self.extract_faces_from_dataset(dataset_path)
        
        if not faces:
            logger.error("No faces extracted from dataset")
            return {
                'processed_count': 0,
                'detection_rate': 0.0,
                'avg_face_size': (0, 0),
                'face_data': []
            }
        
        if progress_callback:
            progress_callback(60, 100, "Filtering and validating faces...")
        
        # Filter faces by confidence threshold
        confidence_threshold = 0.9
        high_confidence_faces = [
            face for face in faces 
            if face.confidence >= confidence_threshold
        ]
        
        if len(high_confidence_faces) < len(faces) * 0.5:
            logger.warning(f"Low confidence faces detected. Using threshold {confidence_threshold}")
            logger.warning(f"High confidence faces: {len(high_confidence_faces)}/{len(faces)}")
        
        # Use high confidence faces if we have enough, otherwise use all
        final_faces = high_confidence_faces if len(high_confidence_faces) >= 5 else faces
        
        if progress_callback:
            progress_callback(90, 100, "Finalizing dataset processing...")
        
        # Calculate statistics
        detection_rate = len(faces) / max(1, len(self._get_image_files(dataset_path)))
        avg_face_size = self._calculate_average_face_size(final_faces)
        
        if progress_callback:
            progress_callback(100, 100, "Dataset processing complete")
        
        logger.info(f"Dataset processing complete: {len(final_faces)} faces ready for training")
        
        return {
            'processed_count': len(final_faces),
            'detection_rate': detection_rate,
            'avg_face_size': avg_face_size,
            'face_data': final_faces
        }
    
    def _get_image_files(self, dataset_path: str) -> List[Path]:
        """Get list of image files in dataset directory."""
        dataset_dir = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return [f for f in dataset_dir.iterdir() 
                if f.suffix.lower() in image_extensions and f.is_file()]
    
    def _calculate_average_face_size(self, faces: List[FaceData]) -> Tuple[int, int]:
        """Calculate average face size from face data."""
        if not faces:
            return (0, 0)
        
        total_width = sum(face.image.shape[1] for face in faces)
        total_height = sum(face.image.shape[0] for face in faces)
        count = len(faces)
        
        return (total_width // count, total_height // count)

    def save_processed_faces(self, faces: List[FaceData], output_dir: str) -> bool:
        """
        Save processed faces to disk for inspection or caching.
        
        Args:
            faces: List of FaceData objects to save
            output_dir: Directory to save processed faces
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, face_data in enumerate(faces):
                # Convert normalized image back to uint8 for saving
                if face_data.image.dtype == np.float32:
                    save_image = (face_data.image * 255).astype(np.uint8)
                else:
                    save_image = face_data.image
                
                # Save image
                filename = f"face_{i:04d}_conf_{face_data.confidence:.3f}.jpg"
                filepath = output_path / filename
                
                cv2.imwrite(str(filepath), save_image)
            
            logger.info(f"Saved {len(faces)} processed faces to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed faces: {e}")
            return False
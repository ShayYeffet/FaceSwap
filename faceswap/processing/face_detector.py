"""
Face detection module using MTCNN for robust face detection and alignment.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from mtcnn import MTCNN
from PIL import Image

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """Represents a face bounding box with confidence score."""
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class FaceData:
    """Complete face data structure with image, landmarks, and metadata."""
    image: np.ndarray          # Face image (256x256x3)
    landmarks: np.ndarray      # 5 facial landmarks from MTCNN
    bbox: BoundingBox         # Face bounding box
    confidence: float         # Detection confidence
    alignment_matrix: Optional[np.ndarray] = None  # Transformation matrix


class FaceDetector:
    """
    Face detection and alignment using MTCNN.
    
    Handles face detection, extraction, and alignment for both single images
    and batch processing scenarios.
    """
    
    def __init__(self, 
                 min_face_size: int = 40,
                 thresholds: List[float] = [0.6, 0.7, 0.7],
                 factor: float = 0.709,
                 device: str = 'cpu'):
        """
        Initialize the face detector.
        
        Args:
            min_face_size: Minimum face size to detect
            thresholds: MTCNN detection thresholds for P-Net, R-Net, O-Net
            factor: Scale factor for image pyramid
            device: Device to run detection on ('cpu' or 'cuda')
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.device = device
        self.target_size = (256, 256)
        
        # Initialize MTCNN detector
        try:
            # Try to initialize MTCNN first
            tf_device = 'CPU:0' if device == 'cpu' else 'GPU:0'
            self.detector = MTCNN(
                stages='face_and_landmarks_detection',
                device=tf_device
            )
            self.detector_type = 'mtcnn'
            logger.info(f"MTCNN face detector initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize MTCNN: {e}")
            logger.info("Falling back to OpenCV Haar Cascade face detector")
            
            # Fallback to OpenCV Haar Cascade
            try:
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.detector_type = 'opencv'
                logger.info(f"OpenCV Haar Cascade face detector initialized")
            except Exception as e2:
                logger.error(f"Failed to initialize any face detector: {e2}")
                raise
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image using MTCNN.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            List of detection dictionaries with 'box', 'confidence', 'keypoints'
        """
        try:
            # Ensure image is in the correct format
            if image is None:
                logger.error("Input image is None")
                return []
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # Assume image is in [0, 1] range, convert to [0, 255]
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            if self.detector_type == 'mtcnn':
                return self._detect_faces_mtcnn(image)
            elif self.detector_type == 'opencv':
                return self._detect_faces_opencv(image)
            else:
                logger.error("Unknown detector type")
                return []
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[dict]:
        """Detect faces using MTCNN."""
        try:
            # Convert BGR to RGB for MTCNN
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Ensure RGB image is uint8
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # MTCNN expects numpy array directly
            detections = self.detector.detect_faces(rgb_image)
            
            if not detections:
                logger.debug("No faces detected with MTCNN")
                return []
            
            logger.debug(f"MTCNN detected {len(detections)} face(s)")
            return detections
            
        except Exception as e:
            logger.error(f"MTCNN face detection failed: {e}")
            return []
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV Haar Cascade."""
        try:
            # Convert to grayscale for OpenCV
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                logger.debug("No faces detected with OpenCV")
                return []
            
            # Convert OpenCV format to MTCNN-like format
            detections = []
            for (x, y, w, h) in faces:
                detection = {
                    'box': [x, y, w, h],
                    'confidence': 0.95,  # OpenCV doesn't provide confidence, use default
                    'keypoints': {
                        'left_eye': (x + w*0.3, y + h*0.35),
                        'right_eye': (x + w*0.7, y + h*0.35),
                        'nose': (x + w*0.5, y + h*0.5),
                        'mouth_left': (x + w*0.35, y + h*0.75),
                        'mouth_right': (x + w*0.65, y + h*0.75)
                    }
                }
                detections.append(detection)
            
            logger.debug(f"OpenCV detected {len(detections)} face(s)")
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def extract_face(self, 
                    image: np.ndarray, 
                    detection: dict,
                    margin: float = 0.2) -> Optional[FaceData]:
        """
        Extract and align a face from an image based on detection results.
        
        Args:
            image: Source image as numpy array
            detection: MTCNN detection dictionary
            margin: Additional margin around face (as fraction of face size)
            
        Returns:
            FaceData object with extracted and aligned face, or None if extraction fails
        """
        try:
            box = detection['box']
            keypoints = detection['keypoints']
            confidence = detection['confidence']
            
            # Create bounding box with margin
            x, y, w, h = box
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            # Expand bounding box with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logger.warning("Empty face region extracted")
                return None
            
            # Align face using landmarks
            aligned_face, alignment_matrix = self.align_face(face_region, keypoints, (x1, y1))
            
            # Create landmarks array (5 points from MTCNN)
            landmarks_array = np.array([
                [keypoints['left_eye'][0] - x1, keypoints['left_eye'][1] - y1],
                [keypoints['right_eye'][0] - x1, keypoints['right_eye'][1] - y1],
                [keypoints['nose'][0] - x1, keypoints['nose'][1] - y1],
                [keypoints['mouth_left'][0] - x1, keypoints['mouth_left'][1] - y1],
                [keypoints['mouth_right'][0] - x1, keypoints['mouth_right'][1] - y1]
            ])
            
            # Create bounding box object
            bbox = BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1, confidence=confidence)
            
            return FaceData(
                image=aligned_face,
                landmarks=landmarks_array,
                bbox=bbox,
                confidence=confidence,
                alignment_matrix=alignment_matrix
            )
            
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return None
    
    def align_face(self, 
                  face_image: np.ndarray, 
                  keypoints: dict,
                  offset: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face using eye positions and resize to target size.
        
        Args:
            face_image: Face region image
            keypoints: MTCNN keypoints dictionary
            offset: Offset to adjust keypoint coordinates
            
        Returns:
            Tuple of (aligned_face_image, transformation_matrix)
        """
        try:
            # Get eye positions (adjust for offset)
            left_eye = np.array(keypoints['left_eye']) - np.array(offset)
            right_eye = np.array(keypoints['right_eye']) - np.array(offset)
            
            # Calculate angle between eyes
            eye_vector = right_eye - left_eye
            angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            
            # Calculate center point between eyes
            eye_center = (left_eye + right_eye) / 2
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(face_image, rotation_matrix, 
                                   (face_image.shape[1], face_image.shape[0]))
            
            # Resize to target size
            aligned_face = cv2.resize(rotated, self.target_size, interpolation=cv2.INTER_CUBIC)
            
            return aligned_face, rotation_matrix
            
        except Exception as e:
            logger.warning(f"Face alignment failed, using simple resize: {e}")
            # Fallback to simple resize if alignment fails
            aligned_face = cv2.resize(face_image, self.target_size, interpolation=cv2.INTER_CUBIC)
            identity_matrix = np.eye(2, 3, dtype=np.float32)
            return aligned_face, identity_matrix
    
    def select_best_face(self, detections: List[dict]) -> Optional[dict]:
        """
        Select the best face from multiple detections.
        
        Prioritizes faces with higher confidence and larger size.
        
        Args:
            detections: List of MTCNN detection dictionaries
            
        Returns:
            Best detection dictionary or None if no valid faces
        """
        if not detections:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Score faces based on confidence and size
        best_detection = None
        best_score = 0
        
        for detection in detections:
            confidence = detection['confidence']
            box = detection['box']
            face_area = box[2] * box[3]  # width * height
            
            # Combined score: confidence weighted by face size
            score = confidence * (1 + face_area / 10000)  # Normalize area
            
            if score > best_score:
                best_score = score
                best_detection = detection
        
        logger.debug(f"Selected best face with score: {best_score:.3f}")
        return best_detection
    
    def process_image(self, image: np.ndarray) -> Optional[FaceData]:
        """
        Complete face processing pipeline for a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            FaceData object with the best detected face, or None if no face found
        """
        # Detect all faces
        detections = self.detect_faces(image)
        
        if not detections:
            return None
        
        # Select best face
        best_detection = self.select_best_face(detections)
        
        if best_detection is None:
            return None
        
        # Extract and align the best face
        return self.extract_face(image, best_detection)
    
    def process_batch(self, images: List[np.ndarray]) -> List[Optional[FaceData]]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of FaceData objects (None for images with no detected faces)
        """
        results = []
        
        logger.info(f"Processing batch of {len(images)} images")
        
        for i, image in enumerate(images):
            try:
                face_data = self.process_image(image)
                results.append(face_data)
                
                if face_data is not None:
                    logger.debug(f"Image {i+1}/{len(images)}: Face detected (confidence: {face_data.confidence:.3f})")
                else:
                    logger.debug(f"Image {i+1}/{len(images)}: No face detected")
                    
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                results.append(None)
        
        successful_detections = sum(1 for r in results if r is not None)
        logger.info(f"Batch processing complete: {successful_detections}/{len(images)} faces detected")
        
        return results
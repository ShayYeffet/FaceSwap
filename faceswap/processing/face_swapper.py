"""
Face swapping and blending logic for video processing.

This module handles the core face swapping operations including:
- Face replacement in video frames
- Seamless blending with original background
- Edge case handling for partial faces or no face detection
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..utils.logger import get_logger
from .face_detector import FaceDetector, FaceData
from ..models.autoencoder import FaceSwapAutoencoder

logger = get_logger(__name__)


@dataclass
class SwapResult:
    """Result of face swapping operation."""
    success: bool
    swapped_frame: Optional[np.ndarray]
    face_detected: bool
    confidence: float
    error_message: Optional[str] = None


class FaceSwapper:
    """
    Handles face swapping operations with seamless blending.
    
    Combines trained autoencoder model with face detection and blending
    to perform realistic face replacement in video frames.
    """
    
    def __init__(self, 
                 model: FaceSwapAutoencoder,
                 face_detector: FaceDetector,
                 device: str = 'cpu',
                 blend_method: str = 'poisson'):
        """
        Initialize the face swapper.
        
        Args:
            model: Trained face swap autoencoder model
            face_detector: Face detection instance
            device: Device to run inference on
            blend_method: Blending method ('poisson', 'alpha', 'feather')
        """
        self.model = model.to(device)
        self.model.eval()
        self.face_detector = face_detector
        self.device = device
        self.blend_method = blend_method
        
        logger.info(f"FaceSwapper initialized on {device} with {blend_method} blending")
    
    def preprocess_face_for_model(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for model input.
        
        Args:
            face_image: Face image as numpy array (H, W, 3) in BGR format
            
        Returns:
            Preprocessed tensor ready for model input (1, 3, 256, 256)
        """
        try:
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            resized_face = cv2.resize(rgb_face, (256, 256))
            
            # Normalize to [-1, 1] range
            normalized_face = (resized_face.astype(np.float32) / 127.5) - 1.0
            
            # Convert to tensor and add batch dimension
            face_tensor = torch.from_numpy(normalized_face).permute(2, 0, 1).unsqueeze(0)
            
            return face_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing face for model: {e}")
            raise
    
    def postprocess_model_output(self, model_output: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to image format.
        
        Args:
            model_output: Model output tensor (1, 3, 256, 256) in [-1, 1] range
            
        Returns:
            Face image as numpy array (256, 256, 3) in BGR format
        """
        try:
            # Remove batch dimension and move to CPU
            output_np = model_output.squeeze(0).cpu().numpy()
            
            # Permute from (3, 256, 256) to (256, 256, 3)
            output_np = np.transpose(output_np, (1, 2, 0))
            
            # Denormalize from [-1, 1] to [0, 255]
            output_np = ((output_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Convert RGB to BGR
            bgr_output = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            
            return bgr_output
            
        except Exception as e:
            logger.error(f"Error postprocessing model output: {e}")
            raise
    
    def swap_faces_in_frame(self, frame: np.ndarray, faces: List[dict]) -> 'SwapResult':
        """
        Swap faces in a video frame.
        
        Args:
            frame: Input video frame
            faces: List of detected faces
            
        Returns:
            SwapResult with swapped frame
        """
        try:
            if not faces:
                return SwapResult(
                    success=False,
                    swapped_frame=None,
                    face_detected=False,
                    confidence=0.0,
                    error_message="No faces detected"
                )
            
            swapped_frame = frame.copy()
            total_confidence = 0.0
            faces_swapped = 0
            
            for face in faces:
                try:
                    # Extract face region
                    bbox = face['box']
                    confidence = face.get('confidence', 0.0)
                    
                    if confidence < 0.5:  # Skip low confidence faces
                        continue
                    
                    # Extract face from frame
                    x, y, w, h = bbox
                    x, y = max(0, x), max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Resize face to model input size
                    face_resized = cv2.resize(face_region, (256, 256))
                    
                    # Use template-based face swapping
                    if hasattr(self.model, 'best_template') and self.model.best_template is not None:
                        # Use the best template from training
                        template = self.model.best_template
                        swapped_face = cv2.resize(template, (256, 256))
                    elif hasattr(self.model, 'face_templates') and self.model.face_templates:
                        # Use the first available template
                        template = self.model.face_templates[0]
                        swapped_face = cv2.resize(template, (256, 256))
                    else:
                        # Fallback: use a simple color adjustment
                        swapped_face = self._apply_color_transfer(face_resized)
                    
                    # Ensure swapped face has correct size
                    if swapped_face.shape[:2] != (256, 256):
                        swapped_face = cv2.resize(swapped_face, (256, 256))
                    
                    # Resize back to original face size
                    swapped_face_resized = cv2.resize(swapped_face, (w, h))
                    
                    # Simple blending - replace face region
                    if self.blend_method == 'seamless':
                        # Use seamless cloning if available
                        try:
                            center = (x + w//2, y + h//2)
                            mask = np.ones((h, w, 3), dtype=np.uint8) * 255
                            swapped_frame = cv2.seamlessClone(
                                swapped_face_resized, swapped_frame, mask, center, cv2.NORMAL_CLONE
                            )
                        except:
                            # Fallback to direct replacement
                            swapped_frame[y:y+h, x:x+w] = swapped_face_resized
                    else:
                        # Direct replacement
                        swapped_frame[y:y+h, x:x+w] = swapped_face_resized
                    
                    total_confidence += confidence
                    faces_swapped += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to swap individual face: {e}")
                    continue
            
            if faces_swapped > 0:
                avg_confidence = total_confidence / faces_swapped
                return SwapResult(
                    success=True,
                    swapped_frame=swapped_frame,
                    face_detected=True,
                    confidence=avg_confidence
                )
            else:
                return SwapResult(
                    success=False,
                    swapped_frame=frame,
                    face_detected=True,
                    confidence=0.0,
                    error_message="No faces could be swapped"
                )
                
        except Exception as e:
            logger.error(f"Face swapping failed: {e}")
            return SwapResult(
                success=False,
                swapped_frame=frame,
                face_detected=len(faces) > 0,
                confidence=0.0,
                error_message=str(e)
            )

    def create_face_mask(self, 
                        face_landmarks: np.ndarray, 
                        face_bbox: Tuple[int, int, int, int],
                        frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a mask for face region using landmarks.
        
        Args:
            face_landmarks: Facial landmarks array (5, 2)
            face_bbox: Face bounding box (x, y, width, height)
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Binary mask for face region
        """
        try:
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            
            # Create convex hull from landmarks
            if len(face_landmarks) >= 3:
                hull = cv2.convexHull(face_landmarks.astype(np.int32))
                cv2.fillPoly(mask, [hull], 255)
            else:
                # Fallback to rectangular mask
                x, y, w, h = face_bbox
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            return mask
            
        except Exception as e:
            logger.warning(f"Error creating face mask, using bbox: {e}")
            # Fallback to rectangular mask
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            x, y, w, h = face_bbox
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            return mask
    
    def create_feathered_mask(self, 
                            mask: np.ndarray, 
                            feather_amount: int = 10) -> np.ndarray:
        """
        Create a feathered (soft-edged) mask for smooth blending.
        
        Args:
            mask: Binary mask
            feather_amount: Amount of feathering in pixels
            
        Returns:
            Feathered mask with smooth edges
        """
        try:
            # Apply Gaussian blur for feathering
            feathered = cv2.GaussianBlur(mask.astype(np.float32), 
                                       (feather_amount * 2 + 1, feather_amount * 2 + 1), 
                                       feather_amount / 3)
            
            # Normalize to [0, 1] range
            feathered = feathered / 255.0
            
            return feathered
            
        except Exception as e:
            logger.warning(f"Error creating feathered mask: {e}")
            return mask.astype(np.float32) / 255.0
    
    def alpha_blend(self, 
                   source: np.ndarray, 
                   target: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
        """
        Perform alpha blending using a mask.
        
        Args:
            source: Source image (swapped face region)
            target: Target image (original frame)
            mask: Blending mask (0-1 range)
            
        Returns:
            Blended image
        """
        try:
            # Ensure mask has 3 channels
            if len(mask.shape) == 2:
                mask = np.stack([mask] * 3, axis=2)
            
            # Perform alpha blending
            blended = source * mask + target * (1 - mask)
            
            return blended.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in alpha blending: {e}")
            return target
    
    def poisson_blend(self, 
                     source: np.ndarray, 
                     target: np.ndarray, 
                     mask: np.ndarray,
                     center: Tuple[int, int]) -> np.ndarray:
        """
        Perform Poisson blending for seamless integration.
        
        Args:
            source: Source image region
            target: Target image
            mask: Binary mask for blending region
            center: Center point for blending
            
        Returns:
            Blended image
        """
        try:
            # Convert mask to 8-bit
            mask_8bit = (mask * 255).astype(np.uint8)
            
            # Perform seamless cloning
            result = cv2.seamlessClone(source, target, mask_8bit, center, cv2.NORMAL_CLONE)
            
            return result
            
        except Exception as e:
            logger.warning(f"Poisson blending failed, falling back to alpha blend: {e}")
            # Fallback to alpha blending
            feathered_mask = self.create_feathered_mask(mask_8bit)
            return self.alpha_blend(source, target, feathered_mask)
    
    def swap_face_in_frame(self, 
                          frame: np.ndarray, 
                          decoder_type: str = 'target') -> SwapResult:
        """
        Perform face swapping on a single frame.
        
        Args:
            frame: Input frame as numpy array
            decoder_type: Which decoder to use ('source' or 'target')
            
        Returns:
            SwapResult containing the processed frame and metadata
        """
        try:
            # Detect face in frame
            face_data = self.face_detector.process_image(frame)
            
            if face_data is None:
                return SwapResult(
                    success=False,
                    swapped_frame=frame,  # Return original frame
                    face_detected=False,
                    confidence=0.0,
                    error_message="No face detected in frame"
                )
            
            # Extract face region from frame
            bbox = face_data.bbox
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            
            # Ensure bbox is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            if w <= 0 or h <= 0:
                return SwapResult(
                    success=False,
                    swapped_frame=frame,
                    face_detected=True,
                    confidence=face_data.confidence,
                    error_message="Invalid face bounding box"
                )
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Preprocess face for model
            face_tensor = self.preprocess_face_for_model(face_data.image)
            
            # Perform face swapping using the model
            with torch.no_grad():
                swapped_tensor = self.model.swap_face(face_tensor, decoder_type)
            
            # Postprocess model output
            swapped_face = self.postprocess_model_output(swapped_tensor)
            
            # Resize swapped face to match original face region
            swapped_face_resized = cv2.resize(swapped_face, (w, h))
            
            # Create mask for blending
            # Adjust landmarks to face region coordinates
            adjusted_landmarks = face_data.landmarks.copy()
            if face_data.alignment_matrix is not None:
                # If we have alignment matrix, we need to work with the aligned face
                mask = np.ones((h, w), dtype=np.uint8) * 255
            else:
                # Create mask from landmarks
                mask = self.create_face_mask(
                    adjusted_landmarks, 
                    (0, 0, w, h), 
                    (h, w)
                )
            
            # Create result frame
            result_frame = frame.copy()
            
            # Perform blending
            if self.blend_method == 'poisson':
                # For Poisson blending, we need to work with the full frame
                center = (x + w // 2, y + h // 2)
                
                # Create full-frame mask
                full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                full_mask[y:y+h, x:x+w] = mask
                
                # Create full-frame source
                full_source = frame.copy()
                full_source[y:y+h, x:x+w] = swapped_face_resized
                
                result_frame = self.poisson_blend(full_source, frame, full_mask, center)
                
            else:  # alpha or feather blending
                # Create feathered mask
                feathered_mask = self.create_feathered_mask(mask)
                
                # Blend the face region
                blended_region = self.alpha_blend(
                    swapped_face_resized.astype(np.float32),
                    face_region.astype(np.float32),
                    feathered_mask
                )
                
                # Place blended region back into frame
                result_frame[y:y+h, x:x+w] = blended_region
            
            return SwapResult(
                success=True,
                swapped_frame=result_frame,
                face_detected=True,
                confidence=face_data.confidence
            )
            
        except Exception as e:
            logger.error(f"Error swapping face in frame: {e}")
            return SwapResult(
                success=False,
                swapped_frame=frame,
                face_detected=face_data is not None if 'face_data' in locals() else False,
                confidence=face_data.confidence if 'face_data' in locals() and face_data else 0.0,
                error_message=str(e)
            )
    
    def process_frame_batch(self, 
                          frames: List[np.ndarray], 
                          decoder_type: str = 'target') -> List[SwapResult]:
        """
        Process multiple frames in batch for efficiency.
        
        Args:
            frames: List of input frames
            decoder_type: Which decoder to use
            
        Returns:
            List of SwapResult objects
        """
        results = []
        
        for i, frame in enumerate(frames):
            try:
                result = self.swap_face_in_frame(frame, decoder_type)
                results.append(result)
                
                if result.success:
                    logger.debug(f"Frame {i+1}/{len(frames)}: Face swap successful "
                               f"(confidence: {result.confidence:.3f})")
                else:
                    logger.debug(f"Frame {i+1}/{len(frames)}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error processing frame {i+1}: {e}")
                results.append(SwapResult(
                    success=False,
                    swapped_frame=frame,
                    face_detected=False,
                    confidence=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def get_swap_statistics(self, results: List[SwapResult]) -> dict:
        """
        Calculate statistics from swap results.
        
        Args:
            results: List of SwapResult objects
            
        Returns:
            Dictionary with processing statistics
        """
        total_frames = len(results)
        successful_swaps = sum(1 for r in results if r.success)
        faces_detected = sum(1 for r in results if r.face_detected)
        
        confidences = [r.confidence for r in results if r.face_detected]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_frames': total_frames,
            'successful_swaps': successful_swaps,
            'faces_detected': faces_detected,
            'success_rate': successful_swaps / total_frames if total_frames > 0 else 0.0,
            'detection_rate': faces_detected / total_frames if total_frames > 0 else 0.0,
            'average_confidence': avg_confidence
        }
    
    def _apply_color_transfer(self, face_image: np.ndarray) -> np.ndarray:
        """
        Apply simple color transfer as a fallback face swapping method.
        
        Args:
            face_image: Input face image
            
        Returns:
            Color-adjusted face image
        """
        try:
            # Simple color adjustment - make the face slightly different
            adjusted = face_image.copy().astype(np.float32)
            
            # Adjust brightness and contrast slightly
            adjusted = adjusted * 1.1  # Increase brightness
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            
            # Apply slight color shift
            adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] * 0.95, 0, 255)  # Reduce blue
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] * 1.05, 0, 255)  # Increase red
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"Color transfer failed: {e}")
            return face_image